/**
 * Frontend Performance and Security Monitoring Utilities
 * Integrates with backend API endpoints for comprehensive monitoring
 */

class FrontendPerformanceMonitor {
    constructor() {
        this.apiEndpoint = '/api/monitoring/performance';
        this.metricsBuffer = [];
        this.flushInterval = 5000; // 5 seconds
        this.observer = null;
        this.vitalsObserver = null;
        this.init();
    }

    init() {
        this.setupPerformanceObserver();
        this.setupCoreWebVitals();
        this.setupPageLoadMonitoring();
        this.setupAPIMonitoring();
        this.startMetricsFlush();
    }

    setupPerformanceObserver() {
        if ('PerformanceObserver' in window) {
            this.observer = new PerformanceObserver((list) => {
                for (const entry of list.getEntries()) {
                    if (entry.entryType === 'navigation') {
                        this.recordPageLoadTime(entry.name, entry.loadEventEnd - entry.loadEventStart);
                    }
                    if (entry.entryType === 'resource') {
                        this.recordResourceLoadTime(entry.name, entry.duration);
                    }
                }
            });

            this.observer.observe({ entryTypes: ['navigation', 'resource'] });
        }
    }

    setupCoreWebVitals() {
        // Load web-vitals library if available
        if (typeof webVitals !== 'undefined') {
            webVitals.getCLS(this.onCLS.bind(this));
            webVitals.getFID(this.onFID.bind(this));
            webVitals.getFCP(this.onFCP.bind(this));
            webVitals.getLCP(this.onLCP.bind(this));
            webVitals.getTTFB(this.onTTFB.bind(this));
        } else {
            // Fallback implementations
            this.setupManualVitals();
        }
    }

    setupManualVitals() {
        // Manual LCP calculation
        const observer = new PerformanceObserver((list) => {
            for (const entry of list.getEntries()) {
                if (entry.entryType === 'largest-contentful-paint') {
                    this.onLCP({ value: entry.startTime, name: 'LCP' });
                }
            }
        });
        observer.observe({ entryTypes: ['largest-contentful-paint'] });

        // Manual FID calculation
        ['mousedown', 'keydown', 'touchstart', 'pointerdown'].forEach(type => {
            document.addEventListener(type, this.onFirstInput.bind(this), { once: true, passive: true });
        });
    }

    onFirstInput(event) {
        const fidValue = performance.now() - event.timeStamp;
        this.onFID({ value: fidValue, name: 'FID' });
    }

    onCLS(metric) {
        this.recordCoreWebVital('CLS', metric.value);
    }

    onFID(metric) {
        this.recordCoreWebVital('FID', metric.value);
    }

    onFCP(metric) {
        this.recordCoreWebVital('FCP', metric.value);
    }

    onLCP(metric) {
        this.recordCoreWebVital('LCP', metric.value);
    }

    onTTFB(metric) {
        this.recordCoreWebVital('TTFB', metric.value);
    }

    recordCoreWebVital(metric, value) {
        console.log(`Core Web Vital: ${metric} = ${value}`);
        this.bufferMetric({
            type: 'core_web_vital',
            core_web_vital: {
                metric: metric,
                value: value
            },
            timestamp: Date.now(),
            url: window.location.pathname
        });

        // Report critical metrics immediately
        const thresholds = {
            'LCP': 2500,
            'FID': 100,
            'CLS': 0.1
        };

        if (value > thresholds[metric]) {
            this.sendCriticalMetric(metric, value);
        }
    }

    recordPageLoadTime(page, loadTime) {
        console.log(`Page Load Time: ${page} = ${loadTime}ms`);
        this.bufferMetric({
            type: 'page_load_time',
            page_load_time: loadTime,
            page: page || window.location.pathname,
            timestamp: Date.now()
        });
    }

    recordResourceLoadTime(resource, duration) {
        if (duration > 1000) { // Only log slow resources
            console.log(`Slow Resource: ${resource} = ${duration}ms`);
            this.bufferMetric({
                type: 'resource_load_time',
                resource_load_time: duration,
                resource: resource,
                timestamp: Date.now()
            });
        }
    }

    recordAPIResponseTime(endpoint, responseTime) {
        console.log(`API Response Time: ${endpoint} = ${responseTime}ms`);
        this.bufferMetric({
            type: 'api_response_time',
            api_response_time: responseTime,
            endpoint: endpoint,
            timestamp: Date.now()
        });
    }

    setupPageLoadMonitoring() {
        window.addEventListener('load', () => {
            const navigationEntry = performance.getEntriesByType('navigation')[0];
            if (navigationEntry) {
                const loadTime = navigationEntry.loadEventEnd - navigationEntry.loadEventStart;
                this.recordPageLoadTime(window.location.pathname, loadTime);
            }
        });
    }

    setupAPIMonitoring() {
        // Intercept fetch calls to monitor API response times
        const originalFetch = window.fetch;
        const self = this;

        window.fetch = function(...args) {
            const start = performance.now();
            const url = args[0];

            return originalFetch.apply(this, args).then(response => {
                const duration = performance.now() - start;

                if (typeof url === 'string' && url.startsWith('/api/')) {
                    self.recordAPIResponseTime(url, duration);
                }

                return response;
            });
        };

        // Intercept XMLHttpRequest
        const originalXHROpen = XMLHttpRequest.prototype.open;
        const originalXHRSend = XMLHttpRequest.prototype.send;

        XMLHttpRequest.prototype.open = function(method, url) {
            this._url = url;
            return originalXHROpen.apply(this, arguments);
        };

        XMLHttpRequest.prototype.send = function() {
            const start = performance.now();
            const url = this._url;

            this.addEventListener('loadend', () => {
                const duration = performance.now() - start;

                if (typeof url === 'string' && url.startsWith('/api/')) {
                    self.recordAPIResponseTime(url, duration);
                }
            });

            return originalXHRSend.apply(this, arguments);
        };
    }

    bufferMetric(metric) {
        this.metricsBuffer.push(metric);

        // Flush if buffer is getting full
        if (this.metricsBuffer.length >= 50) {
            this.flushMetrics();
        }
    }

    async sendCriticalMetric(metric, value) {
        const data = {
            metric: metric,
            value: value,
            timestamp: Date.now(),
            url: window.location.pathname
        };

        try {
            await fetch('/api/metrics/critical', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRF-Token': this.getCSRFToken()
                },
                body: JSON.stringify(data)
            });
        } catch (error) {
            console.error('Failed to send critical metric:', error);
        }
    }

    async flushMetrics() {
        if (this.metricsBuffer.length === 0) return;

        const metrics = [...this.metricsBuffer];
        this.metricsBuffer = [];

        try {
            for (const metric of metrics) {
                await fetch(this.apiEndpoint, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'X-CSRF-Token': this.getCSRFToken()
                    },
                    body: JSON.stringify(metric)
                });
            }
        } catch (error) {
            console.error('Failed to flush metrics:', error);
            // Re-add failed metrics to buffer
            this.metricsBuffer.unshift(...metrics);
        }
    }

    startMetricsFlush() {
        setInterval(() => {
            this.flushMetrics();
        }, this.flushInterval);

        // Flush on page unload
        window.addEventListener('beforeunload', () => {
            this.flushMetrics();
        });
    }

    getCSRFToken() {
        const meta = document.querySelector('meta[name="csrf-token"]');
        return meta ? meta.getAttribute('content') : '';
    }

    // Memory usage monitoring
    recordMemoryUsage() {
        if (performance.memory) {
            const memory = performance.memory;
            this.bufferMetric({
                type: 'memory_usage',
                memory_usage: {
                    used: memory.usedJSHeapSize,
                    total: memory.totalJSHeapSize,
                    limit: memory.jsHeapSizeLimit
                },
                timestamp: Date.now()
            });
        }
    }

    // Error monitoring
    setupErrorMonitoring() {
        window.addEventListener('error', (event) => {
            this.bufferMetric({
                type: 'javascript_error',
                error: {
                    message: event.message,
                    filename: event.filename,
                    lineno: event.lineno,
                    colno: event.colno,
                    stack: event.error ? event.error.stack : null
                },
                timestamp: Date.now(),
                url: window.location.pathname
            });
        });

        window.addEventListener('unhandledrejection', (event) => {
            this.bufferMetric({
                type: 'promise_rejection',
                error: {
                    reason: event.reason,
                    stack: event.reason && event.reason.stack
                },
                timestamp: Date.now(),
                url: window.location.pathname
            });
        });
    }

    // User interaction monitoring
    setupUserInteractionMonitoring() {
        const interactionTypes = ['click', 'keypress', 'scroll', 'touch'];

        interactionTypes.forEach(type => {
            document.addEventListener(type, (event) => {
                this.bufferMetric({
                    type: 'user_interaction',
                    interaction: {
                        type: type,
                        target: event.target.tagName,
                        timestamp: Date.now()
                    }
                });
            }, { passive: true });
        });
    }

    // Start monitoring
    startMonitoring() {
        this.setupErrorMonitoring();
        this.setupUserInteractionMonitoring();

        // Record memory usage every 30 seconds
        setInterval(() => {
            this.recordMemoryUsage();
        }, 30000);

        // Integrate with performance dashboard
        this.integrateWithDashboard();
    }

    // Integrate with performance dashboard
    integrateWithDashboard() {
        if (window.performanceDashboard) {
            // Add monitoring data to dashboard
            const originalBufferMetric = this.bufferMetric.bind(this);
            this.bufferMetric = (metric) => {
                originalBufferMetric(metric);

                // Forward to dashboard
                if (metric.type === 'core_web_vital') {
                    window.performanceDashboard.addMetric('core-web-vital', {
                        metric: metric.core_web_vital.metric,
                        value: metric.core_web_vital.value,
                        timestamp: metric.timestamp
                    });
                } else if (metric.type === 'page_load_time') {
                    window.performanceDashboard.addMetric('page-load', {
                        loadTime: metric.page_load_time,
                        page: metric.page,
                        timestamp: metric.timestamp
                    });
                } else if (metric.type === 'api_response_time') {
                    window.performanceDashboard.addMetric('api-call', {
                        endpoint: metric.endpoint,
                        responseTime: metric.api_response_time,
                        timestamp: metric.timestamp
                    });
                } else if (metric.type === 'javascript_error' || metric.type === 'promise_rejection') {
                    window.performanceDashboard.addMetric('error', {
                        message: metric.error.message || metric.error.reason,
                        stack: metric.error.stack,
                        timestamp: metric.timestamp
                    });
                }
            };
        }
    }

    // Stop monitoring
    stopMonitoring() {
        if (this.observer) {
            this.observer.disconnect();
        }
        if (this.vitalsObserver) {
            this.vitalsObserver.disconnect();
        }
    }
}

class FrontendSecurityMonitor {
    constructor() {
        this.apiEndpoint = '/api/monitoring/security';
        this.securityEvents = [];
        this.threatPatterns = {
            xss: [/<script/i, /javascript:/i, /onerror=/i, /onload=/i, /alert\(/i],
            sql: [/union select/i, /drop table/i, /insert into/i, /delete from/i],
            csrf: [/authenticity_token/i, /csrf_token/i]
        };
        this.init();
    }

    init() {
        this.setupCSPViolationReporting();
        this.setupInputSanitization();
        this.setupFormValidation();
        this.setupSecurityHeaders();
        this.monitorConsoleUsage();
    }

    setupCSPViolationReporting() {
        document.addEventListener('securitypolicyviolation', (event) => {
            this.reportSecurityEvent('csp_violation', {
                directive: event.violatedDirective,
                blockedURI: event.blockedURI,
                documentURI: event.documentURI,
                originalPolicy: event.originalPolicy,
                sample: event.sample
            });
        });
    }

    setupInputSanitization() {
        // Monitor all input fields for potential threats
        document.addEventListener('input', (event) => {
            if (event.target.tagName === 'INPUT' || event.target.tagName === 'TEXTAREA') {
                this.validateInput(event.target);
            }
        });
    }

    validateInput(input) {
        const value = input.value;

        for (const [threatType, patterns] of Object.entries(this.threatPatterns)) {
            for (const pattern of patterns) {
                if (pattern.test(value)) {
                    this.reportSecurityEvent(`input_threat_${threatType}`, {
                        field: input.name || input.id,
                        value: value.substring(0, 100), // Limit logged value
                        pattern: pattern.toString(),
                        threat_type: threatType
                    });

                    // Optionally sanitize the input
                    if (threatType === 'xss') {
                        input.value = this.sanitizeXSS(value);
                    }
                    break;
                }
            }
        }
    }

    sanitizeXSS(input) {
        const div = document.createElement('div');
        div.textContent = input;
        return div.innerHTML;
    }

    setupFormValidation() {
        document.addEventListener('submit', (event) => {
            const form = event.target;
            if (form.tagName === 'FORM') {
                this.validateForm(form);
            }
        });
    }

    validateForm(form) {
        const formData = new FormData(form);
        const csrfToken = formData.get('csrf_token') ||
                         form.querySelector('input[name="csrf_token"]')?.value ||
                         document.querySelector('meta[name="csrf-token"]')?.getAttribute('content');

        if (!csrfToken) {
            this.reportSecurityEvent('csrf_token_missing', {
                form: form.action || form.id,
                method: form.method
            });
        }

        // Validate all form fields
        for (const [key, value] of formData.entries()) {
            if (typeof value === 'string' && value.length > 0) {
                this.validateInputValue(key, value);
            }
        }
    }

    validateInputValue(fieldName, value) {
        for (const [threatType, patterns] of Object.entries(this.threatPatterns)) {
            for (const pattern of patterns) {
                if (pattern.test(value)) {
                    this.reportSecurityEvent(`form_threat_${threatType}`, {
                        field: fieldName,
                        value: value.substring(0, 100),
                        pattern: pattern.toString(),
                        threat_type: threatType
                    });
                }
            }
        }
    }

    setupSecurityHeaders() {
        // Check for required security headers
        fetch(window.location.href, { method: 'HEAD' })
            .then(response => {
                const requiredHeaders = [
                    'X-Frame-Options',
                    'X-Content-Type-Options',
                    'Content-Security-Policy',
                    'X-XSS-Protection'
                ];

                const missingHeaders = requiredHeaders.filter(header =>
                    !response.headers.has(header)
                );

                if (missingHeaders.length > 0) {
                    this.reportSecurityEvent('missing_security_headers', {
                        missing_headers: missingHeaders,
                        url: window.location.href
                    });
                }
            })
            .catch(error => {
                console.error('Failed to check security headers:', error);
            });
    }

    monitorConsoleUsage() {
        const originalConsoleLog = console.log;
        const originalConsoleError = console.error;
        const originalConsoleWarn = console.warn;

        console.log = (...args) => {
            this.checkConsoleUsage('log', args);
            return originalConsoleLog.apply(console, args);
        };

        console.error = (...args) => {
            this.checkConsoleUsage('error', args);
            return originalConsoleError.apply(console, args);
        };

        console.warn = (...args) => {
            this.checkConsoleUsage('warn', args);
            return originalConsoleWarn.apply(console, args);
        };
    }

    checkConsoleUsage(level, args) {
        const message = args.join(' ');

        // Check for potential security issues in console output
        if (message.includes('password') || message.includes('token') || message.includes('secret')) {
            this.reportSecurityEvent('sensitive_data_in_console', {
                level: level,
                message: message.substring(0, 200),
                stack: new Error().stack
            });
        }
    }

    async reportSecurityEvent(eventType, details) {
        const event = {
            event_type: eventType,
            details: details,
            timestamp: Date.now(),
            url: window.location.pathname,
            user_agent: navigator.userAgent,
            referrer: document.referrer
        };

        this.securityEvents.push(event);

        try {
            await fetch(this.apiEndpoint, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRF-Token': this.getCSRFToken()
                },
                body: JSON.stringify(event)
            });
        } catch (error) {
            console.error('Failed to report security event:', error);
        }
    }

    getCSRFToken() {
        const meta = document.querySelector('meta[name="csrf-token"]');
        return meta ? meta.getAttribute('content') : '';
    }

    // Session management
    setupSessionMonitoring() {
        let sessionWarned = false;

        const checkSession = async () => {
            try {
                const response = await fetch('/api/session/status');
                const sessionData = await response.json();

                if (sessionData.expires_at) {
                    const now = Date.now() / 1000;
                    const timeUntilExpiry = sessionData.expires_at - now;

                    // Warn when 5 minutes left
                    if (timeUntilExpiry < 300 && !sessionWarned) {
                        sessionWarned = true;
                        this.reportSecurityEvent('session_expiry_warning', {
                            expires_at: sessionData.expires_at,
                            time_remaining: timeUntilExpiry
                        });

                        // Show user notification
                        this.showSessionWarning(timeUntilExpiry);
                    }
                }
            } catch (error) {
                console.error('Failed to check session status:', error);
            }
        };

        // Check session every minute
        setInterval(checkSession, 60000);
    }

    showSessionWarning(timeRemaining) {
        const minutes = Math.floor(timeRemaining / 60);
        const message = `Your session will expire in ${minutes} minute(s). Do you want to extend it?`;

        if (confirm(message)) {
            this.extendSession();
        }
    }

    async extendSession() {
        try {
            const response = await fetch('/api/session/extend', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRF-Token': this.getCSRFToken()
                }
            });

            if (response.ok) {
                this.reportSecurityEvent('session_extended', {
                    extended_at: Date.now()
                });
            }
        } catch (error) {
            console.error('Failed to extend session:', error);
        }
    }

    // Start monitoring
    startMonitoring() {
        this.setupSessionMonitoring();
        console.log('Frontend security monitoring started');
    }

    // Stop monitoring
    stopMonitoring() {
        console.log('Frontend security monitoring stopped');
    }
}

// Initialize monitoring when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.frontendMonitor = new FrontendPerformanceMonitor();
    window.securityMonitor = new FrontendSecurityMonitor();

    // Start monitoring
    window.frontendMonitor.startMonitoring();
    window.securityMonitor.startMonitoring();
});

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        FrontendPerformanceMonitor,
        FrontendSecurityMonitor
    };
}
