/**
 * Enhanced Session Management Module
 * Handles session timeout, renewal notifications, and concurrent session management
 */

class SessionManager {
    constructor(options = {}) {
        this.config = {
            sessionTimeout: options.sessionTimeout || 30 * 60 * 1000, // 30 minutes
            warningTime: options.warningTime || 5 * 60 * 1000, // 5 minutes before expiry
            checkInterval: options.checkInterval || 60 * 1000, // Check every minute
            renewalEndpoint: options.renewalEndpoint || '/api/v1/auth/renew',
            heartbeatEndpoint: options.heartbeatEndpoint || '/api/v1/auth/heartbeat',
            logoutEndpoint: options.logoutEndpoint || '/logout',
            loginUrl: options.loginUrl || '/login',
            ...options
        };

        this.sessionData = {
            lastActivity: Date.now(),
            sessionId: this.getSessionId(),
            isActive: true,
            hasWarned: false,
            renewalAttempts: 0,
            maxRenewalAttempts: 3
        };

        this.timers = {
            checkTimer: null,
            warningTimer: null,
            expireTimer: null
        };

        this.init();
    }

    init() {
        this.setupActivityTracking();
        this.setupSessionChecking();
        this.setupRenewalHandling();
        this.setupConcurrentSessionHandling();
        this.setupStorageListeners();
        this.startSessionMonitoring();
    }

    /**
     * Get session ID from cookie or localStorage
     */
    getSessionId() {
        // Try to get from cookie first
        const cookies = document.cookie.split(';');
        for (let cookie of cookies) {
            const [name, value] = cookie.trim().split('=');
            if (name === 'session_id' || name === 'sessionid') {
                return value;
            }
        }

        // Fallback to localStorage
        return localStorage.getItem('sessionId') || null;
    }

    /**
     * Setup user activity tracking
     */
    setupActivityTracking() {
        const activityEvents = ['mousedown', 'mousemove', 'keypress', 'scroll', 'touchstart'];

        const updateActivity = () => {
            this.sessionData.lastActivity = Date.now();
            this.sessionData.hasWarned = false;
            this.resetTimers();
            this.hideSessionWarning();
        };

        // Throttle activity updates to avoid excessive calls
        const throttledUpdate = this.throttle(updateActivity, 10000); // Max once per 10 seconds

        activityEvents.forEach(event => {
            document.addEventListener(event, throttledUpdate, true);
        });
    }

    /**
     * Setup session expiry checking
     */
    setupSessionChecking() {
        this.timers.checkTimer = setInterval(() => {
            this.checkSessionStatus();
        }, this.config.checkInterval);
    }

    /**
     * Check current session status and handle expiry
     */
    checkSessionStatus() {
        const now = Date.now();
        const timeSinceActivity = now - this.sessionData.lastActivity;
        const timeUntilExpiry = this.config.sessionTimeout - timeSinceActivity;

        // Session has expired
        if (timeUntilExpiry <= 0) {
            this.handleSessionExpiry();
            return;
        }

        // Show warning if close to expiry
        if (timeUntilExpiry <= this.config.warningTime && !this.sessionData.hasWarned) {
            this.showSessionWarning(timeUntilExpiry);
            this.sessionData.hasWarned = true;
        }

        // Update UI if warning is showing
        if (this.sessionData.hasWarned) {
            this.updateWarningCountdown(timeUntilExpiry);
        }
    }

    /**
     * Show session expiry warning
     */
    showSessionWarning(timeRemaining) {
        const modal = this.createWarningModal(timeRemaining);
        document.body.appendChild(modal);

        // Add backdrop click handling
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                this.hideSessionWarning();
            }
        });
    }

    /**
     * Create session warning modal
     */
    createWarningModal(timeRemaining) {
        const modal = document.createElement('div');
        modal.id = 'session-warning-modal';
        modal.className = 'session-modal-overlay';

        const minutes = Math.ceil(timeRemaining / 60000);

        modal.innerHTML = `
            <div class="session-modal">
                <div class="session-modal-header">
                    <h3>Session Expiring Soon</h3>
                    <button class="modal-close" onclick="sessionManager.hideSessionWarning()">&times;</button>
                </div>
                <div class="session-modal-body">
                    <div class="session-warning-icon">
                        <svg class="w-12 h-12 text-yellow-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                                  d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L4.082 16.5c-.77.833.192 2.5 1.732 2.5z">
                            </path>
                        </svg>
                    </div>
                    <div class="session-warning-text">
                        <p>Your session will expire in <strong id="countdown-display">${minutes} minute${minutes !== 1 ? 's' : ''}</strong>.</p>
                        <p>Would you like to extend your session?</p>
                    </div>
                </div>
                <div class="session-modal-footer">
                    <button class="btn btn-primary" onclick="sessionManager.renewSession()">
                        Extend Session
                    </button>
                    <button class="btn btn-secondary" onclick="sessionManager.logout()">
                        Logout Now
                    </button>
                </div>
            </div>
        `;

        this.addModalStyles();
        return modal;
    }

    /**
     * Add CSS styles for session modal
     */
    addModalStyles() {
        if (document.querySelector('#session-modal-styles')) return;

        const styles = `
            .session-modal-overlay {
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: rgba(0, 0, 0, 0.5);
                display: flex;
                align-items: center;
                justify-content: center;
                z-index: 10000;
                animation: fadeIn 0.2s ease;
            }

            .session-modal {
                background: white;
                border-radius: 8px;
                box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2);
                max-width: 450px;
                width: 90%;
                animation: slideUp 0.3s ease;
            }

            .session-modal-header {
                padding: 20px 24px 0;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }

            .session-modal-header h3 {
                margin: 0;
                font-size: 1.25rem;
                font-weight: 600;
                color: #374151;
            }

            .modal-close {
                background: none;
                border: none;
                font-size: 24px;
                cursor: pointer;
                color: #9ca3af;
                padding: 0;
                width: 24px;
                height: 24px;
                display: flex;
                align-items: center;
                justify-content: center;
            }

            .session-modal-body {
                padding: 20px 24px;
                text-align: center;
            }

            .session-warning-icon {
                margin-bottom: 16px;
            }

            .session-warning-text p {
                margin: 8px 0;
                color: #374151;
            }

            .session-modal-footer {
                padding: 0 24px 24px;
                display: flex;
                gap: 12px;
                justify-content: center;
            }

            .btn {
                padding: 8px 16px;
                border-radius: 6px;
                border: none;
                cursor: pointer;
                font-weight: 500;
                transition: all 0.2s;
            }

            .btn-primary {
                background: #3b82f6;
                color: white;
            }

            .btn-primary:hover {
                background: #2563eb;
            }

            .btn-secondary {
                background: #e5e7eb;
                color: #374151;
            }

            .btn-secondary:hover {
                background: #d1d5db;
            }

            @keyframes fadeIn {
                from { opacity: 0; }
                to { opacity: 1; }
            }

            @keyframes slideUp {
                from { transform: translateY(20px); opacity: 0; }
                to { transform: translateY(0); opacity: 1; }
            }
        `;

        const styleSheet = document.createElement('style');
        styleSheet.id = 'session-modal-styles';
        styleSheet.textContent = styles;
        document.head.appendChild(styleSheet);
    }

    /**
     * Update warning countdown display
     */
    updateWarningCountdown(timeRemaining) {
        const countdownDisplay = document.getElementById('countdown-display');
        if (countdownDisplay) {
            const minutes = Math.ceil(timeRemaining / 60000);
            countdownDisplay.textContent = `${minutes} minute${minutes !== 1 ? 's' : ''}`;
        }
    }

    /**
     * Hide session warning modal
     */
    hideSessionWarning() {
        const modal = document.getElementById('session-warning-modal');
        if (modal) {
            modal.remove();
        }
    }

    /**
     * Handle session expiry
     */
    handleSessionExpiry() {
        this.sessionData.isActive = false;
        this.clearTimers();

        // Show expiry notification
        this.showExpiryNotification();

        // Redirect to login after short delay
        setTimeout(() => {
            this.redirectToLogin();
        }, 3000);
    }

    /**
     * Show session expiry notification
     */
    showExpiryNotification() {
        const notification = document.createElement('div');
        notification.className = 'session-expired-notification';
        notification.innerHTML = `
            <div class="notification-content">
                <h3>Session Expired</h3>
                <p>Your session has expired for security reasons. You will be redirected to login.</p>
            </div>
        `;

        document.body.appendChild(notification);

        // Add styles
        const styles = `
            .session-expired-notification {
                position: fixed;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                background: white;
                border: 2px solid #dc2626;
                border-radius: 8px;
                padding: 24px;
                box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2);
                z-index: 10001;
                text-align: center;
                animation: fadeIn 0.3s ease;
            }

            .notification-content h3 {
                color: #dc2626;
                margin: 0 0 12px;
            }

            .notification-content p {
                margin: 0;
                color: #374151;
            }
        `;

        if (!document.querySelector('#session-expired-styles')) {
            const styleSheet = document.createElement('style');
            styleSheet.id = 'session-expired-styles';
            styleSheet.textContent = styles;
            document.head.appendChild(styleSheet);
        }
    }

    /**
     * Renew the current session
     */
    async renewSession() {
        try {
            this.sessionData.renewalAttempts++;

            const response = await fetch(this.config.renewalEndpoint, {
                method: 'POST',
                credentials: 'include',
                headers: {
                    'Content-Type': 'application/json',
                    'X-Requested-With': 'XMLHttpRequest'
                }
            });

            if (response.ok) {
                // Session renewed successfully
                this.sessionData.lastActivity = Date.now();
                this.sessionData.hasWarned = false;
                this.sessionData.renewalAttempts = 0;
                this.hideSessionWarning();
                this.resetTimers();

                this.showSuccessNotification('Session extended successfully');
            } else {
                throw new Error(`Renewal failed: ${response.status}`);
            }
        } catch (error) {
            console.error('Session renewal failed:', error);

            if (this.sessionData.renewalAttempts >= this.sessionData.maxRenewalAttempts) {
                this.handleSessionExpiry();
            } else {
                this.showErrorNotification('Failed to extend session. Please try again.');
            }
        }
    }

    /**
     * Setup session renewal handling
     */
    setupRenewalHandling() {
        // Add keyboard shortcut for renewal (Ctrl+R or Cmd+R when warning is shown)
        document.addEventListener('keydown', (e) => {
            if (this.sessionData.hasWarned && (e.ctrlKey || e.metaKey) && e.key === 'r') {
                e.preventDefault();
                this.renewSession();
            }
        });
    }

    /**
     * Setup concurrent session handling
     */
    setupConcurrentSessionHandling() {
        // Listen for storage events to detect concurrent sessions
        window.addEventListener('storage', (e) => {
            if (e.key === 'sessionActivity' && e.newValue) {
                const otherSessionActivity = JSON.parse(e.newValue);

                // If another session is more recent, show warning
                if (otherSessionActivity.timestamp > this.sessionData.lastActivity) {
                    this.handleConcurrentSession(otherSessionActivity);
                }
            }
        });

        // Broadcast session activity
        setInterval(() => {
            if (this.sessionData.isActive) {
                localStorage.setItem('sessionActivity', JSON.stringify({
                    sessionId: this.sessionData.sessionId,
                    timestamp: this.sessionData.lastActivity,
                    tabId: this.getTabId()
                }));
            }
        }, 5000);
    }

    /**
     * Handle concurrent session detection
     */
    handleConcurrentSession(otherSession) {
        if (otherSession.sessionId === this.sessionData.sessionId &&
            otherSession.tabId !== this.getTabId()) {

            this.showConcurrentSessionWarning();
        }
    }

    /**
     * Show concurrent session warning
     */
    showConcurrentSessionWarning() {
        const notification = document.createElement('div');
        notification.className = 'concurrent-session-warning';
        notification.innerHTML = `
            <div class="warning-content">
                <h4>Multiple Sessions Detected</h4>
                <p>This account is being used in another tab or window.</p>
                <button onclick="this.parentElement.parentElement.remove()" class="dismiss-btn">Dismiss</button>
            </div>
        `;

        document.body.appendChild(notification);

        // Auto-remove after 10 seconds
        setTimeout(() => {
            if (notification.parentElement) {
                notification.remove();
            }
        }, 10000);
    }

    /**
     * Get unique tab identifier
     */
    getTabId() {
        if (!sessionStorage.getItem('tabId')) {
            sessionStorage.setItem('tabId', 'tab_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9));
        }
        return sessionStorage.getItem('tabId');
    }

    /**
     * Setup localStorage listeners for cross-tab communication
     */
    setupStorageListeners() {
        window.addEventListener('beforeunload', () => {
            // Clean up this tab's activity
            const currentActivity = localStorage.getItem('sessionActivity');
            if (currentActivity) {
                const activity = JSON.parse(currentActivity);
                if (activity.tabId === this.getTabId()) {
                    localStorage.removeItem('sessionActivity');
                }
            }
        });
    }

    /**
     * Start session monitoring
     */
    startSessionMonitoring() {
        // Send periodic heartbeat to server
        setInterval(() => {
            if (this.sessionData.isActive) {
                this.sendHeartbeat();
            }
        }, 5 * 60 * 1000); // Every 5 minutes
    }

    /**
     * Send heartbeat to server
     */
    async sendHeartbeat() {
        try {
            await fetch(this.config.heartbeatEndpoint, {
                method: 'POST',
                credentials: 'include',
                headers: {
                    'X-Requested-With': 'XMLHttpRequest'
                }
            });
        } catch (error) {
            console.warn('Heartbeat failed:', error);
        }
    }

    /**
     * Logout user
     */
    async logout() {
        try {
            await fetch(this.config.logoutEndpoint, {
                method: 'POST',
                credentials: 'include'
            });
        } catch (error) {
            console.error('Logout request failed:', error);
        } finally {
            this.redirectToLogin();
        }
    }

    /**
     * Redirect to login page
     */
    redirectToLogin() {
        window.location.href = this.config.loginUrl;
    }

    /**
     * Reset all timers
     */
    resetTimers() {
        Object.values(this.timers).forEach(timer => {
            if (timer) clearTimeout(timer);
        });
    }

    /**
     * Clear all timers
     */
    clearTimers() {
        this.resetTimers();
        this.timers = { checkTimer: null, warningTimer: null, expireTimer: null };
    }

    /**
     * Show success notification
     */
    showSuccessNotification(message) {
        this.showNotification(message, 'success');
    }

    /**
     * Show error notification
     */
    showErrorNotification(message) {
        this.showNotification(message, 'error');
    }

    /**
     * Show notification
     */
    showNotification(message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `session-notification notification-${type}`;
        notification.textContent = message;

        document.body.appendChild(notification);

        setTimeout(() => {
            if (notification.parentElement) {
                notification.remove();
            }
        }, 3000);
    }

    /**
     * Throttle function to limit execution frequency
     */
    throttle(func, limit) {
        let inThrottle;
        return function() {
            const args = arguments;
            const context = this;
            if (!inThrottle) {
                func.apply(context, args);
                inThrottle = true;
                setTimeout(() => inThrottle = false, limit);
            }
        };
    }

    /**
     * Get session status
     */
    getSessionStatus() {
        return {
            ...this.sessionData,
            timeUntilExpiry: this.config.sessionTimeout - (Date.now() - this.sessionData.lastActivity)
        };
    }

    /**
     * Manually extend session
     */
    extendSession(additionalTime = 30 * 60 * 1000) { // 30 minutes default
        this.sessionData.lastActivity = Date.now();
        this.sessionData.hasWarned = false;
        this.resetTimers();
        this.hideSessionWarning();
    }

    /**
     * Destroy session manager
     */
    destroy() {
        this.clearTimers();
        this.hideSessionWarning();
        this.sessionData.isActive = false;
    }
}

// Initialize session manager when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    // Get configuration from meta tags or data attributes
    const config = {
        sessionTimeout: parseInt(document.querySelector('meta[name="session-timeout"]')?.content) || 30 * 60 * 1000,
        warningTime: parseInt(document.querySelector('meta[name="session-warning"]')?.content) || 5 * 60 * 1000
    };

    window.sessionManager = new SessionManager(config);
});

// Export for use in modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = SessionManager;
}
