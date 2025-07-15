/**
 * Pynomaly Frontend Integration
 * Main entry point for all frontend utilities and monitoring
 */

class PynominalyFrontend {
    constructor() {
        this.config = null;
        this.performanceMonitor = null;
        this.securityMonitor = null;
        this.apiClient = null;
        this.initialized = false;
        this.initPromise = null;
    }

    /**
     * Initialize the frontend system
     */
    async init() {
        if (this.initialized) {
            return;
        }

        if (this.initPromise) {
            return this.initPromise;
        }

        this.initPromise = this._performInit();
        return this.initPromise;
    }

    async _performInit() {
        try {
            console.log('Initializing Pynomaly Frontend...');
            
            // Load configuration
            await this.loadConfig();
            
            // Initialize API client
            this.initializeAPIClient();
            
            // Initialize monitoring systems
            this.initializeMonitoring();
            
            // Setup UI enhancements
            this.setupUIEnhancements();
            
            // Initialize features based on config
            this.initializeFeatures();
            
            this.initialized = true;
            console.log('Pynomaly Frontend initialized successfully');
            
            // Dispatch initialization event
            document.dispatchEvent(new CustomEvent('pynomaly:initialized', {
                detail: { frontend: this }
            }));
            
        } catch (error) {
            console.error('Failed to initialize Pynomaly Frontend:', error);
            throw error;
        }
    }

    /**
     * Load configuration from backend
     */
    async loadConfig() {
        try {
            const response = await fetch('/api/ui/config');
            if (!response.ok) {
                throw new Error(`Config load failed: ${response.status}`);
            }
            this.config = await response.json();
            console.log('Configuration loaded:', this.config);
        } catch (error) {
            console.error('Failed to load configuration:', error);
            // Use fallback config
            this.config = this.getFallbackConfig();
        }
    }

    /**
     * Get fallback configuration
     */
    getFallbackConfig() {
        return {
            performance_monitoring: {
                enabled: true,
                critical_thresholds: {
                    LCP: 2500,
                    FID: 100,
                    CLS: 0.1,
                    memory_used: 50 * 1024 * 1024
                }
            },
            security: {
                csrf_protection: true,
                xss_protection: true,
                sql_injection_protection: true,
                session_timeout: 30 * 60 * 1000
            },
            features: {
                dark_mode: true,
                lazy_loading: true,
                caching: true,
                offline_support: true
            }
        };
    }

    /**
     * Initialize API client
     */
    initializeAPIClient() {
        this.apiClient = new PynominalyAPIClient();
    }

    /**
     * Initialize monitoring systems
     */
    initializeMonitoring() {
        if (this.config.performance_monitoring.enabled) {
            this.performanceMonitor = new FrontendPerformanceMonitor();
            this.performanceMonitor.startMonitoring();
            
            // Setup performance dashboard integration
            this.setupPerformanceDashboard();
        }

        if (this.config.security.csrf_protection) {
            this.securityMonitor = new FrontendSecurityMonitor();
            this.securityMonitor.startMonitoring();
        }
    }
    
    /**
     * Setup performance dashboard
     */
    setupPerformanceDashboard() {
        // Add keyboard shortcut (Ctrl+Shift+P) to toggle performance dashboard
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey && e.shiftKey && e.key === 'P') {
                e.preventDefault();
                if (window.performanceDashboard) {
                    window.performanceDashboard.toggle();
                }
            }
        });
        
        // Add performance dashboard button to UI
        this.addPerformanceDashboardButton();
    }
    
    /**
     * Add performance dashboard button to UI
     */
    addPerformanceDashboardButton() {
        const button = document.createElement('button');
        button.className = 'performance-dashboard-btn';
        button.innerHTML = '📊';
        button.title = 'Performance Dashboard (Ctrl+Shift+P)';
        button.onclick = () => {
            if (window.performanceDashboard) {
                window.performanceDashboard.toggle();
            }
        };
        
        button.style.cssText = `
            position: fixed;
            bottom: 80px;
            right: 20px;
            width: 50px;
            height: 50px;
            border: none;
            border-radius: 50%;
            background: #28a745;
            color: white;
            font-size: 20px;
            cursor: pointer;
            z-index: 1000;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
            transition: all 0.3s ease;
        `;
        
        button.addEventListener('mouseenter', () => {
            button.style.transform = 'scale(1.1)';
            button.style.boxShadow = '0 4px 20px rgba(0,0,0,0.3)';
        });
        
        button.addEventListener('mouseleave', () => {
            button.style.transform = 'scale(1)';
            button.style.boxShadow = '0 2px 10px rgba(0,0,0,0.2)';
        });

        document.body.appendChild(button);
    }

    /**
     * Setup UI enhancements
     */
    setupUIEnhancements() {
        // Loading indicators
        this.setupLoadingIndicators();
        
        // Form enhancements
        this.setupFormEnhancements();
        
        // Navigation enhancements
        this.setupNavigationEnhancements();
        
        // Error handling
        this.setupErrorHandling();
    }

    /**
     * Setup loading indicators
     */
    setupLoadingIndicators() {
        // Add loading indicators to all forms
        document.querySelectorAll('form').forEach(form => {
            form.addEventListener('submit', (e) => {
                this.showLoadingIndicator(form);
            });
        });

        // Add loading indicators to HTMX requests
        document.addEventListener('htmx:beforeRequest', (e) => {
            this.showLoadingIndicator(e.target);
        });

        document.addEventListener('htmx:afterRequest', (e) => {
            this.hideLoadingIndicator(e.target);
        });
    }

    /**
     * Show loading indicator
     */
    showLoadingIndicator(element) {
        const indicator = document.createElement('div');
        indicator.className = 'loading-indicator';
        indicator.innerHTML = `
            <div class="spinner"></div>
            <span>Loading...</span>
        `;
        
        element.style.position = 'relative';
        element.appendChild(indicator);
    }

    /**
     * Hide loading indicator
     */
    hideLoadingIndicator(element) {
        const indicator = element.querySelector('.loading-indicator');
        if (indicator) {
            indicator.remove();
        }
    }

    /**
     * Setup form enhancements
     */
    setupFormEnhancements() {
        // Auto-save forms
        document.querySelectorAll('form[data-autosave]').forEach(form => {
            this.setupAutoSave(form);
        });

        // Form validation
        document.querySelectorAll('form').forEach(form => {
            this.setupFormValidation(form);
        });

        // CSRF token management
        this.setupCSRFTokenManagement();
    }

    /**
     * Setup auto-save functionality
     */
    setupAutoSave(form) {
        const autosaveInterval = 30000; // 30 seconds
        let timeout;

        const saveForm = () => {
            const formData = new FormData(form);
            const data = Object.fromEntries(formData.entries());
            
            localStorage.setItem(`autosave_${form.id}`, JSON.stringify({
                data: data,
                timestamp: Date.now()
            }));
        };

        form.addEventListener('input', () => {
            clearTimeout(timeout);
            timeout = setTimeout(saveForm, autosaveInterval);
        });

        // Restore saved data on page load
        const savedData = localStorage.getItem(`autosave_${form.id}`);
        if (savedData) {
            const { data, timestamp } = JSON.parse(savedData);
            const age = Date.now() - timestamp;
            
            // Only restore if less than 24 hours old
            if (age < 24 * 60 * 60 * 1000) {
                Object.entries(data).forEach(([key, value]) => {
                    const field = form.querySelector(`[name="${key}"]`);
                    if (field) {
                        field.value = value;
                    }
                });
            }
        }
    }

    /**
     * Setup form validation
     */
    setupFormValidation(form) {
        form.addEventListener('submit', (e) => {
            if (!this.validateForm(form)) {
                e.preventDefault();
                return false;
            }
        });
    }

    /**
     * Validate form
     */
    validateForm(form) {
        const errors = [];
        
        // Required field validation
        form.querySelectorAll('[required]').forEach(field => {
            if (!field.value.trim()) {
                errors.push(`${field.name || field.id} is required`);
                field.classList.add('error');
            } else {
                field.classList.remove('error');
            }
        });

        // Email validation
        form.querySelectorAll('[type="email"]').forEach(field => {
            if (field.value && !this.isValidEmail(field.value)) {
                errors.push(`${field.name || field.id} must be a valid email`);
                field.classList.add('error');
            } else {
                field.classList.remove('error');
            }
        });

        if (errors.length > 0) {
            this.showValidationErrors(errors);
            return false;
        }

        return true;
    }

    /**
     * Validate email format
     */
    isValidEmail(email) {
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        return emailRegex.test(email);
    }

    /**
     * Show validation errors
     */
    showValidationErrors(errors) {
        // Remove existing error messages
        document.querySelectorAll('.validation-error').forEach(el => el.remove());

        // Create error container
        const errorContainer = document.createElement('div');
        errorContainer.className = 'validation-error alert alert-danger';
        errorContainer.innerHTML = `
            <h4>Please fix the following errors:</h4>
            <ul>
                ${errors.map(error => `<li>${error}</li>`).join('')}
            </ul>
        `;

        // Insert at top of form
        const firstForm = document.querySelector('form');
        if (firstForm) {
            firstForm.insertBefore(errorContainer, firstForm.firstChild);
        }
    }

    /**
     * Setup CSRF token management
     */
    setupCSRFTokenManagement() {
        // Refresh CSRF token periodically
        setInterval(() => {
            this.refreshCSRFToken();
        }, 10 * 60 * 1000); // Every 10 minutes

        // Add CSRF token to all forms
        document.querySelectorAll('form').forEach(form => {
            this.addCSRFTokenToForm(form);
        });
    }

    /**
     * Refresh CSRF token
     */
    async refreshCSRFToken() {
        try {
            const response = await fetch('/api/session/status');
            const sessionData = await response.json();
            
            if (sessionData.csrf_token) {
                // Update meta tag
                const metaTag = document.querySelector('meta[name="csrf-token"]');
                if (metaTag) {
                    metaTag.setAttribute('content', sessionData.csrf_token);
                }
                
                // Update all forms
                document.querySelectorAll('form').forEach(form => {
                    this.updateCSRFTokenInForm(form, sessionData.csrf_token);
                });
            }
        } catch (error) {
            console.error('Failed to refresh CSRF token:', error);
        }
    }

    /**
     * Add CSRF token to form
     */
    addCSRFTokenToForm(form) {
        const csrfToken = this.getCSRFToken();
        if (csrfToken) {
            let csrfInput = form.querySelector('input[name="csrf_token"]');
            if (!csrfInput) {
                csrfInput = document.createElement('input');
                csrfInput.type = 'hidden';
                csrfInput.name = 'csrf_token';
                form.appendChild(csrfInput);
            }
            csrfInput.value = csrfToken;
        }
    }

    /**
     * Update CSRF token in form
     */
    updateCSRFTokenInForm(form, token) {
        const csrfInput = form.querySelector('input[name="csrf_token"]');
        if (csrfInput) {
            csrfInput.value = token;
        }
    }

    /**
     * Get CSRF token from meta tag
     */
    getCSRFToken() {
        const meta = document.querySelector('meta[name="csrf-token"]');
        return meta ? meta.getAttribute('content') : null;
    }

    /**
     * Setup navigation enhancements
     */
    setupNavigationEnhancements() {
        // Smooth scrolling for anchor links
        document.querySelectorAll('a[href^="#"]').forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const target = document.querySelector(link.getAttribute('href'));
                if (target) {
                    target.scrollIntoView({ behavior: 'smooth' });
                }
            });
        });

        // Back to top button
        this.setupBackToTopButton();
    }

    /**
     * Setup back to top button
     */
    setupBackToTopButton() {
        const backToTop = document.createElement('button');
        backToTop.className = 'back-to-top';
        backToTop.innerHTML = '↑';
        backToTop.style.cssText = `
            position: fixed;
            bottom: 20px;
            right: 20px;
            width: 50px;
            height: 50px;
            border: none;
            border-radius: 50%;
            background: #007bff;
            color: white;
            font-size: 20px;
            cursor: pointer;
            display: none;
            z-index: 1000;
        `;

        document.body.appendChild(backToTop);

        // Show/hide based on scroll position
        window.addEventListener('scroll', () => {
            if (window.scrollY > 300) {
                backToTop.style.display = 'block';
            } else {
                backToTop.style.display = 'none';
            }
        });

        backToTop.addEventListener('click', () => {
            window.scrollTo({ top: 0, behavior: 'smooth' });
        });
    }

    /**
     * Setup error handling
     */
    setupErrorHandling() {
        // Global error handler
        window.addEventListener('error', (event) => {
            this.handleError(event.error, 'JavaScript Error');
        });

        // Promise rejection handler
        window.addEventListener('unhandledrejection', (event) => {
            this.handleError(event.reason, 'Unhandled Promise Rejection');
        });

        // Network error handler
        document.addEventListener('htmx:responseError', (event) => {
            this.handleError(event.detail, 'Network Error');
        });
    }

    /**
     * Handle errors
     */
    handleError(error, type = 'Error') {
        console.error(`${type}:`, error);
        
        // Report to monitoring if available
        if (this.performanceMonitor) {
            this.performanceMonitor.recordError(error, type);
        }

        // Show user-friendly error message
        this.showErrorMessage(`An error occurred. Please try again or contact support if the problem persists.`);
    }

    /**
     * Show error message
     */
    showErrorMessage(message) {
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-message alert alert-danger';
        errorDiv.textContent = message;
        
        // Add close button
        const closeBtn = document.createElement('button');
        closeBtn.className = 'close';
        closeBtn.innerHTML = '×';
        closeBtn.onclick = () => errorDiv.remove();
        errorDiv.appendChild(closeBtn);

        // Insert at top of page
        document.body.insertBefore(errorDiv, document.body.firstChild);

        // Auto-remove after 5 seconds
        setTimeout(() => {
            errorDiv.remove();
        }, 5000);
    }

    /**
     * Initialize features based on configuration
     */
    initializeFeatures() {
        if (this.config.features.dark_mode) {
            this.initializeDarkMode();
        }

        if (this.config.features.lazy_loading) {
            this.initializeLazyLoading();
        }

        if (this.config.features.caching) {
            this.initializeCaching();
        }

        if (this.config.features.offline_support) {
            this.initializeOfflineSupport();
        }
    }

    /**
     * Initialize dark mode
     */
    initializeDarkMode() {
        // Check for saved preference
        const savedTheme = localStorage.getItem('theme');
        if (savedTheme) {
            document.documentElement.setAttribute('data-theme', savedTheme);
        }

        // Add theme toggle button
        const themeToggle = document.createElement('button');
        themeToggle.className = 'theme-toggle';
        themeToggle.innerHTML = '🌙';
        themeToggle.onclick = () => this.toggleTheme();
        
        // Add to navigation or header
        const nav = document.querySelector('nav') || document.querySelector('header');
        if (nav) {
            nav.appendChild(themeToggle);
        }
    }

    /**
     * Toggle theme
     */
    toggleTheme() {
        const currentTheme = document.documentElement.getAttribute('data-theme');
        const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
        
        document.documentElement.setAttribute('data-theme', newTheme);
        localStorage.setItem('theme', newTheme);
    }

    /**
     * Initialize lazy loading
     */
    initializeLazyLoading() {
        if ('IntersectionObserver' in window) {
            const lazyObserver = new IntersectionObserver((entries) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        const element = entry.target;
                        
                        if (element.dataset.src) {
                            element.src = element.dataset.src;
                            element.removeAttribute('data-src');
                        }
                        
                        if (element.dataset.srcset) {
                            element.srcset = element.dataset.srcset;
                            element.removeAttribute('data-srcset');
                        }
                        
                        lazyObserver.unobserve(element);
                    }
                });
            });

            // Observe all lazy-loadable elements
            document.querySelectorAll('[data-src], [data-srcset]').forEach(element => {
                lazyObserver.observe(element);
            });
        }
    }

    /**
     * Initialize caching
     */
    initializeCaching() {
        // Simple cache implementation
        window.pynomaly = window.pynomaly || {};
        window.pynomaly.cache = new Map();

        // Cache API responses
        const originalFetch = window.fetch;
        window.fetch = async (url, options = {}) => {
            const method = options.method || 'GET';
            
            if (method === 'GET' && typeof url === 'string') {
                const cached = window.pynomaly.cache.get(url);
                if (cached && (Date.now() - cached.timestamp) < 300000) { // 5 minutes
                    return new Response(cached.data, {
                        status: 200,
                        headers: { 'Content-Type': 'application/json' }
                    });
                }
            }

            const response = await originalFetch(url, options);
            
            if (method === 'GET' && response.ok && typeof url === 'string') {
                const data = await response.clone().text();
                window.pynomaly.cache.set(url, {
                    data: data,
                    timestamp: Date.now()
                });
            }

            return response;
        };
    }

    /**
     * Initialize offline support
     */
    initializeOfflineSupport() {
        // Register service worker if available
        if ('serviceWorker' in navigator) {
            navigator.serviceWorker.register('/sw.js')
                .then(registration => {
                    console.log('Service Worker registered:', registration);
                })
                .catch(error => {
                    console.log('Service Worker registration failed:', error);
                });
        }

        // Handle online/offline events
        window.addEventListener('online', () => {
            this.showMessage('Connection restored', 'success');
        });

        window.addEventListener('offline', () => {
            this.showMessage('No internet connection', 'warning');
        });
    }

    /**
     * Show message to user
     */
    showMessage(message, type = 'info') {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message alert alert-${type}`;
        messageDiv.textContent = message;
        
        document.body.insertBefore(messageDiv, document.body.firstChild);

        setTimeout(() => {
            messageDiv.remove();
        }, 3000);
    }
}

/**
 * API Client for consistent API interactions
 */
class PynominalyAPIClient {
    constructor() {
        this.baseURL = '/api';
        this.defaultHeaders = {
            'Content-Type': 'application/json'
        };
    }

    /**
     * Make API request
     */
    async request(endpoint, options = {}) {
        const url = `${this.baseURL}${endpoint}`;
        const config = {
            ...options,
            headers: {
                ...this.defaultHeaders,
                ...options.headers,
                'X-CSRF-Token': this.getCSRFToken()
            }
        };

        try {
            const response = await fetch(url, config);
            
            if (!response.ok) {
                throw new Error(`API request failed: ${response.status} ${response.statusText}`);
            }

            return await response.json();
        } catch (error) {
            console.error('API request failed:', error);
            throw error;
        }
    }

    /**
     * GET request
     */
    async get(endpoint, params = {}) {
        const url = new URL(`${this.baseURL}${endpoint}`, window.location.origin);
        Object.entries(params).forEach(([key, value]) => {
            url.searchParams.append(key, value);
        });

        return this.request(endpoint + '?' + url.searchParams.toString(), {
            method: 'GET'
        });
    }

    /**
     * POST request
     */
    async post(endpoint, data = {}) {
        return this.request(endpoint, {
            method: 'POST',
            body: JSON.stringify(data)
        });
    }

    /**
     * PUT request
     */
    async put(endpoint, data = {}) {
        return this.request(endpoint, {
            method: 'PUT',
            body: JSON.stringify(data)
        });
    }

    /**
     * DELETE request
     */
    async delete(endpoint) {
        return this.request(endpoint, {
            method: 'DELETE'
        });
    }

    /**
     * Get CSRF token
     */
    getCSRFToken() {
        const meta = document.querySelector('meta[name="csrf-token"]');
        return meta ? meta.getAttribute('content') : null;
    }
    
    /**
     * Report performance metric
     */
    async reportPerformanceMetric(metric, value, url = window.location.pathname) {
        return this.post('/metrics/critical', {
            metric: metric,
            value: value,
            timestamp: Date.now(),
            url: url
        });
    }
    
    /**
     * Report security event
     */
    async reportSecurityEvent(eventType, details) {
        return this.post('/security/events', {
            type: eventType,
            timestamp: Date.now(),
            url: window.location.pathname,
            userAgent: navigator.userAgent,
            data: details
        });
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.pynomaly = new PynominalyFrontend();
    window.pynomaly.init().catch(error => {
        console.error('Failed to initialize Pynomaly frontend:', error);
    });
});

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        PynominalyFrontend,
        PynominalyAPIClient
    };
}