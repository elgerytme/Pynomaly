/**
 * Authentication Service
 * 
 * Comprehensive authentication and authorization service with JWT tokens,
 * role-based access control, and session management
 */

export class AuthService {
    constructor(options = {}) {
        this.options = {
            apiBaseUrl: options.apiBaseUrl || '/api/auth',
            tokenKey: options.tokenKey || 'pynomaly_token',
            refreshTokenKey: options.refreshTokenKey || 'pynomaly_refresh_token',
            userKey: options.userKey || 'pynomaly_user',
            autoRefresh: options.autoRefresh !== false,
            refreshThreshold: options.refreshThreshold || 300000, // 5 minutes
            sessionTimeout: options.sessionTimeout || 3600000, // 1 hour
            enableLogging: options.enableLogging || false,
            ...options
        };
        
        this.currentUser = null;
        this.token = null;
        this.refreshToken = null;
        this.refreshTimer = null;
        this.sessionTimer = null;
        this.listeners = new Map();
        
        this.init();
    }
    
    init() {
        this.loadStoredAuth();
        this.startTokenRefreshTimer();
        this.bindEvents();
    }
    
    bindEvents() {
        // Listen for storage changes in other tabs
        window.addEventListener('storage', (e) => {
            if (e.key === this.options.tokenKey || e.key === this.options.userKey) {
                this.loadStoredAuth();
            }
        });
        
        // Listen for visibility changes to refresh tokens when tab becomes active
        document.addEventListener('visibilitychange', () => {
            if (!document.hidden && this.isAuthenticated() && this.shouldRefreshToken()) {
                this.refreshAccessToken();
            }
        });
    }
    
    // Authentication methods
    async login(credentials) {
        try {
            this.log('Attempting login for user:', credentials.username);
            
            const response = await this.makeRequest('/login', {
                method: 'POST',
                body: JSON.stringify(credentials)
            });
            
            if (response.success) {
                await this.handleAuthResponse(response);
                this.emit('login', { user: this.currentUser });
                return { success: true, user: this.currentUser };
            } else {
                throw new Error(response.message || 'Login failed');
            }
        } catch (error) {
            this.log('Login error:', error);
            this.emit('login_error', { error: error.message });
            throw error;
        }
    }
    
    async register(userData) {
        try {
            this.log('Attempting registration for user:', userData.username);
            
            const response = await this.makeRequest('/register', {
                method: 'POST',
                body: JSON.stringify(userData)
            });
            
            if (response.success) {
                // Don't auto-login after registration, require email verification
                this.emit('registration_success', { message: response.message });
                return { success: true, message: response.message };
            } else {
                throw new Error(response.message || 'Registration failed');
            }
        } catch (error) {
            this.log('Registration error:', error);
            this.emit('registration_error', { error: error.message });
            throw error;
        }
    }
    
    async logout() {
        try {
            if (this.token) {
                // Notify server about logout
                await this.makeRequest('/logout', {
                    method: 'POST',
                    headers: {
                        'Authorization': `Bearer ${this.token}`
                    }
                });
            }
        } catch (error) {
            this.log('Logout API error (continuing with local logout):', error);
        } finally {
            this.clearAuth();
            this.emit('logout');
        }
    }
    
    async refreshAccessToken() {
        if (!this.refreshToken) {
            throw new Error('No refresh token available');
        }
        
        try {
            this.log('Refreshing access token');
            
            const response = await this.makeRequest('/refresh', {
                method: 'POST',
                body: JSON.stringify({ refreshToken: this.refreshToken })
            });
            
            if (response.success) {
                this.setToken(response.accessToken);
                if (response.refreshToken) {
                    this.setRefreshToken(response.refreshToken);
                }
                this.emit('token_refreshed');
                return response.accessToken;
            } else {
                throw new Error(response.message || 'Token refresh failed');
            }
        } catch (error) {
            this.log('Token refresh error:', error);
            this.clearAuth();
            this.emit('token_refresh_error', { error: error.message });
            throw error;
        }
    }
    
    async verifyEmail(token) {
        try {
            const response = await this.makeRequest('/verify-email', {
                method: 'POST',
                body: JSON.stringify({ token })
            });
            
            if (response.success) {
                this.emit('email_verified', { message: response.message });
                return { success: true, message: response.message };
            } else {
                throw new Error(response.message || 'Email verification failed');
            }
        } catch (error) {
            this.log('Email verification error:', error);
            this.emit('email_verification_error', { error: error.message });
            throw error;
        }
    }
    
    async requestPasswordReset(email) {
        try {
            const response = await this.makeRequest('/request-password-reset', {
                method: 'POST',
                body: JSON.stringify({ email })
            });
            
            if (response.success) {
                this.emit('password_reset_requested', { message: response.message });
                return { success: true, message: response.message };
            } else {
                throw new Error(response.message || 'Password reset request failed');
            }
        } catch (error) {
            this.log('Password reset request error:', error);
            this.emit('password_reset_error', { error: error.message });
            throw error;
        }
    }
    
    async resetPassword(token, newPassword) {
        try {
            const response = await this.makeRequest('/reset-password', {
                method: 'POST',
                body: JSON.stringify({ token, newPassword })
            });
            
            if (response.success) {
                this.emit('password_reset_success', { message: response.message });
                return { success: true, message: response.message };
            } else {
                throw new Error(response.message || 'Password reset failed');
            }
        } catch (error) {
            this.log('Password reset error:', error);
            this.emit('password_reset_error', { error: error.message });
            throw error;
        }
    }
    
    async changePassword(currentPassword, newPassword) {
        try {
            const response = await this.makeAuthenticatedRequest('/change-password', {
                method: 'POST',
                body: JSON.stringify({ currentPassword, newPassword })
            });
            
            if (response.success) {
                this.emit('password_changed', { message: response.message });
                return { success: true, message: response.message };
            } else {
                throw new Error(response.message || 'Password change failed');
            }
        } catch (error) {
            this.log('Password change error:', error);
            this.emit('password_change_error', { error: error.message });
            throw error;
        }
    }
    
    async updateProfile(profileData) {
        try {
            const response = await this.makeAuthenticatedRequest('/profile', {
                method: 'PUT',
                body: JSON.stringify(profileData)
            });
            
            if (response.success) {
                this.currentUser = { ...this.currentUser, ...response.user };
                this.storeUser(this.currentUser);
                this.emit('profile_updated', { user: this.currentUser });
                return { success: true, user: this.currentUser };
            } else {
                throw new Error(response.message || 'Profile update failed');
            }
        } catch (error) {
            this.log('Profile update error:', error);
            this.emit('profile_update_error', { error: error.message });
            throw error;
        }
    }
    
    // Authorization methods
    hasRole(role) {
        return this.currentUser && this.currentUser.roles && this.currentUser.roles.includes(role);
    }
    
    hasPermission(permission) {
        return this.currentUser && this.currentUser.permissions && this.currentUser.permissions.includes(permission);
    }
    
    hasAnyRole(roles) {
        return roles.some(role => this.hasRole(role));
    }
    
    hasAllRoles(roles) {
        return roles.every(role => this.hasRole(role));
    }
    
    hasAnyPermission(permissions) {
        return permissions.some(permission => this.hasPermission(permission));
    }
    
    hasAllPermissions(permissions) {
        return permissions.every(permission => this.hasPermission(permission));
    }
    
    canAccess(resource, action = 'read') {
        // Basic RBAC implementation
        const resourcePermissions = this.getResourcePermissions(resource);
        return resourcePermissions.includes(`${resource}:${action}`) || 
               resourcePermissions.includes(`${resource}:*`) ||
               this.hasRole('admin');
    }
    
    getResourcePermissions(resource) {
        if (!this.currentUser || !this.currentUser.permissions) return [];
        
        return this.currentUser.permissions.filter(permission => 
            permission.startsWith(resource + ':')
        );
    }
    
    // Token management
    setToken(token) {
        this.token = token;
        this.storeToken(token);
        this.startTokenRefreshTimer();
        this.startSessionTimer();
    }
    
    setRefreshToken(refreshToken) {
        this.refreshToken = refreshToken;
        this.storeRefreshToken(refreshToken);
    }
    
    getToken() {
        return this.token;
    }
    
    isAuthenticated() {
        return !!this.token && !!this.currentUser && !this.isTokenExpired();
    }
    
    isTokenExpired() {
        if (!this.token) return true;
        
        try {
            const payload = this.parseJWT(this.token);
            return payload.exp * 1000 < Date.now();
        } catch (error) {
            this.log('Error parsing token:', error);
            return true;
        }
    }
    
    shouldRefreshToken() {
        if (!this.token) return false;
        
        try {
            const payload = this.parseJWT(this.token);
            const timeUntilExpiry = payload.exp * 1000 - Date.now();
            return timeUntilExpiry < this.options.refreshThreshold;
        } catch (error) {
            return false;
        }
    }
    
    parseJWT(token) {
        const base64Url = token.split('.')[1];
        const base64 = base64Url.replace(/-/g, '+').replace(/_/g, '/');
        const jsonPayload = decodeURIComponent(atob(base64).split('').map(function(c) {
            return '%' + ('00' + c.charCodeAt(0).toString(16)).slice(-2);
        }).join(''));
        
        return JSON.parse(jsonPayload);
    }
    
    // Session management
    startSessionTimer() {
        if (this.sessionTimer) {
            clearTimeout(this.sessionTimer);
        }
        
        this.sessionTimer = setTimeout(() => {
            this.emit('session_expired');
            this.clearAuth();
        }, this.options.sessionTimeout);
    }
    
    extendSession() {
        this.startSessionTimer();
        this.emit('session_extended');
    }
    
    startTokenRefreshTimer() {
        if (!this.options.autoRefresh || this.refreshTimer) {
            return;
        }
        
        if (this.shouldRefreshToken()) {
            this.refreshAccessToken().catch(error => {
                this.log('Auto-refresh failed:', error);
            });
        }
        
        this.refreshTimer = setInterval(() => {
            if (this.isAuthenticated() && this.shouldRefreshToken()) {
                this.refreshAccessToken().catch(error => {
                    this.log('Auto-refresh failed:', error);
                });
            }
        }, 60000); // Check every minute
    }
    
    stopTokenRefreshTimer() {
        if (this.refreshTimer) {
            clearInterval(this.refreshTimer);
            this.refreshTimer = null;
        }
    }
    
    // Storage methods
    storeToken(token) {
        try {
            localStorage.setItem(this.options.tokenKey, token);
        } catch (error) {
            this.log('Error storing token:', error);
        }
    }
    
    storeRefreshToken(refreshToken) {
        try {
            localStorage.setItem(this.options.refreshTokenKey, refreshToken);
        } catch (error) {
            this.log('Error storing refresh token:', error);
        }
    }
    
    storeUser(user) {
        try {
            localStorage.setItem(this.options.userKey, JSON.stringify(user));
        } catch (error) {
            this.log('Error storing user:', error);
        }
    }
    
    loadStoredAuth() {
        try {
            const token = localStorage.getItem(this.options.tokenKey);
            const refreshToken = localStorage.getItem(this.options.refreshTokenKey);
            const userJson = localStorage.getItem(this.options.userKey);
            
            if (token && userJson) {
                this.token = token;
                this.refreshToken = refreshToken;
                this.currentUser = JSON.parse(userJson);
                
                if (this.isTokenExpired()) {
                    if (this.refreshToken) {
                        this.refreshAccessToken().catch(() => {
                            this.clearAuth();
                        });
                    } else {
                        this.clearAuth();
                    }
                } else {
                    this.startTokenRefreshTimer();
                    this.startSessionTimer();
                    this.emit('auth_restored', { user: this.currentUser });
                }
            }
        } catch (error) {
            this.log('Error loading stored auth:', error);
            this.clearAuth();
        }
    }
    
    clearAuth() {
        this.currentUser = null;
        this.token = null;
        this.refreshToken = null;
        
        this.stopTokenRefreshTimer();
        
        if (this.sessionTimer) {
            clearTimeout(this.sessionTimer);
            this.sessionTimer = null;
        }
        
        try {
            localStorage.removeItem(this.options.tokenKey);
            localStorage.removeItem(this.options.refreshTokenKey);
            localStorage.removeItem(this.options.userKey);
        } catch (error) {
            this.log('Error clearing auth storage:', error);
        }
    }
    
    // HTTP request methods
    async makeRequest(endpoint, options = {}) {
        const url = `${this.options.apiBaseUrl}${endpoint}`;
        
        const defaultOptions = {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            }
        };
        
        const requestOptions = { ...defaultOptions, ...options };
        
        const response = await fetch(url, requestOptions);
        
        if (!response.ok) {
            if (response.status === 401) {
                this.clearAuth();
                this.emit('unauthorized');
            }
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        return await response.json();
    }
    
    async makeAuthenticatedRequest(endpoint, options = {}) {
        if (!this.isAuthenticated()) {
            throw new Error('Not authenticated');
        }
        
        const authOptions = {
            ...options,
            headers: {
                'Authorization': `Bearer ${this.token}`,
                ...options.headers
            }
        };
        
        try {
            return await this.makeRequest(endpoint, authOptions);
        } catch (error) {
            if (error.message.includes('401') && this.refreshToken) {
                // Try to refresh token and retry request
                await this.refreshAccessToken();
                authOptions.headers['Authorization'] = `Bearer ${this.token}`;
                return await this.makeRequest(endpoint, authOptions);
            }
            throw error;
        }
    }
    
    // User management methods
    async getUsers(filters = {}) {
        const queryParams = new URLSearchParams(filters).toString();
        const endpoint = `/users${queryParams ? `?${queryParams}` : ''}`;
        
        const response = await this.makeAuthenticatedRequest(endpoint);
        return response.users || [];
    }
    
    async createUser(userData) {
        const response = await this.makeAuthenticatedRequest('/users', {
            method: 'POST',
            body: JSON.stringify(userData)
        });
        
        this.emit('user_created', { user: response.user });
        return response.user;
    }
    
    async updateUser(userId, userData) {
        const response = await this.makeAuthenticatedRequest(`/users/${userId}`, {
            method: 'PUT',
            body: JSON.stringify(userData)
        });
        
        this.emit('user_updated', { user: response.user });
        return response.user;
    }
    
    async deleteUser(userId) {
        await this.makeAuthenticatedRequest(`/users/${userId}`, {
            method: 'DELETE'
        });
        
        this.emit('user_deleted', { userId });
    }
    
    async assignRole(userId, role) {
        const response = await this.makeAuthenticatedRequest(`/users/${userId}/roles`, {
            method: 'POST',
            body: JSON.stringify({ role })
        });
        
        this.emit('role_assigned', { userId, role });
        return response;
    }
    
    async removeRole(userId, role) {
        const response = await this.makeAuthenticatedRequest(`/users/${userId}/roles/${role}`, {
            method: 'DELETE'
        });
        
        this.emit('role_removed', { userId, role });
        return response;
    }
    
    // Utility methods
    async handleAuthResponse(response) {
        if (response.accessToken) {
            this.setToken(response.accessToken);
        }
        
        if (response.refreshToken) {
            this.setRefreshToken(response.refreshToken);
        }
        
        if (response.user) {
            this.currentUser = response.user;
            this.storeUser(this.currentUser);
        }
    }
    
    getCurrentUser() {
        return this.currentUser;
    }
    
    getUserRoles() {
        return this.currentUser ? this.currentUser.roles || [] : [];
    }
    
    getUserPermissions() {
        return this.currentUser ? this.currentUser.permissions || [] : [];
    }
    
    // Event management
    on(event, listener) {
        if (!this.listeners.has(event)) {
            this.listeners.set(event, new Set());
        }
        this.listeners.get(event).add(listener);
        
        return () => this.off(event, listener);
    }
    
    off(event, listener) {
        const eventListeners = this.listeners.get(event);
        if (eventListeners) {
            eventListeners.delete(listener);
            if (eventListeners.size === 0) {
                this.listeners.delete(event);
            }
        }
    }
    
    emit(event, data) {
        const eventListeners = this.listeners.get(event);
        if (eventListeners) {
            eventListeners.forEach(listener => {
                try {
                    listener(data);
                } catch (error) {
                    this.log('Error in event listener:', error);
                }
            });
        }
    }
    
    log(...args) {
        if (this.options.enableLogging) {
            console.log('[AuthService]', ...args);
        }
    }
    
    // Cleanup
    destroy() {
        this.clearAuth();
        this.listeners.clear();
        
        window.removeEventListener('storage', this.loadStoredAuth);
        document.removeEventListener('visibilitychange', this.handleVisibilityChange);
    }
}

/**
 * Role-Based Access Control Helper
 */
export class RBACHelper {
    constructor(authService) {
        this.authService = authService;
    }
    
    // UI element access control
    showIfAuthorized(element, requirements) {
        const isAuthorized = this.checkAuthorization(requirements);
        if (element) {
            element.style.display = isAuthorized ? '' : 'none';
        }
        return isAuthorized;
    }
    
    enableIfAuthorized(element, requirements) {
        const isAuthorized = this.checkAuthorization(requirements);
        if (element) {
            element.disabled = !isAuthorized;
        }
        return isAuthorized;
    }
    
    checkAuthorization(requirements) {
        if (!this.authService.isAuthenticated()) {
            return false;
        }
        
        if (requirements.roles) {
            if (requirements.requireAll) {
                if (!this.authService.hasAllRoles(requirements.roles)) {
                    return false;
                }
            } else {
                if (!this.authService.hasAnyRole(requirements.roles)) {
                    return false;
                }
            }
        }
        
        if (requirements.permissions) {
            if (requirements.requireAll) {
                if (!this.authService.hasAllPermissions(requirements.permissions)) {
                    return false;
                }
            } else {
                if (!this.authService.hasAnyPermission(requirements.permissions)) {
                    return false;
                }
            }
        }
        
        if (requirements.custom) {
            return requirements.custom(this.authService.getCurrentUser());
        }
        
        return true;
    }
    
    // Route protection
    protectRoute(routeHandler, requirements) {
        return (...args) => {
            if (this.checkAuthorization(requirements)) {
                return routeHandler(...args);
            } else {
                throw new Error('Access denied');
            }
        };
    }
    
    // API request protection
    protectApiCall(apiCall, requirements) {
        return async (...args) => {
            if (this.checkAuthorization(requirements)) {
                return await apiCall(...args);
            } else {
                throw new Error('Access denied');
            }
        };
    }
}

// Default export
export default AuthService;
