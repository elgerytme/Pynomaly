/**
 * Content Security Policy Manager
 * Handles CSP nonce generation, violation reporting, and dynamic policy updates
 */

class CSPManager {
    constructor(options = {}) {
        this.config = {
            reportEndpoint: options.reportEndpoint || '/api/v1/security/csp-violations',
            nonceLength: options.nonceLength || 16,
            enableViolationReporting: options.enableViolationReporting !== false,
            autoRefreshNonce: options.autoRefreshNonce || false,
            refreshInterval: options.refreshInterval || 300000, // 5 minutes
            ...options
        };

        this.currentNonce = null;
        this.violationCount = 0;
        this.reportedViolations = new Set();

        this.init();
    }

    init() {
        this.generateNonce();
        this.setupViolationReporting();
        this.setupNonceRefresh();
        this.setupSecurityHeaders();
        this.monitorCSPCompliance();
    }

    /**
     * Generate a new CSP nonce
     */
    generateNonce() {
        const array = new Uint8Array(this.config.nonceLength);
        crypto.getRandomValues(array);
        this.currentNonce = Array.from(array, byte => byte.toString(16).padStart(2, '0')).join('');

        // Update meta tag if it exists
        this.updateNonceMeta();

        return this.currentNonce;
    }

    /**
     * Update nonce meta tag
     */
    updateNonceMeta() {
        let meta = document.querySelector('meta[name="csp-nonce"]');
        if (!meta) {
            meta = document.createElement('meta');
            meta.name = 'csp-nonce';
            document.head.appendChild(meta);
        }
        meta.content = this.currentNonce;
    }

    /**
     * Get current CSP nonce
     */
    getNonce() {
        return this.currentNonce;
    }

    /**
     * Setup CSP violation reporting
     */
    setupViolationReporting() {
        if (!this.config.enableViolationReporting) return;

        document.addEventListener('securitypolicyviolation', (event) => {
            this.handleCSPViolation(event);
        });

        // Also listen for report-to API
        if ('ReportingObserver' in window) {
            const observer = new ReportingObserver((reports) => {
                for (const report of reports) {
                    if (report.type === 'csp-violation') {
                        this.handleCSPViolationReport(report);
                    }
                }
            }, { buffered: true });
            observer.observe();
        }
    }

    /**
     * Handle CSP violation events
     */
    handleCSPViolation(event) {
        this.violationCount++;

        const violation = {
            type: 'csp-violation',
            blockedURI: event.blockedURI,
            violatedDirective: event.violatedDirective,
            effectiveDirective: event.effectiveDirective,
            originalPolicy: event.originalPolicy,
            sourceFile: event.sourceFile,
            lineNumber: event.lineNumber,
            columnNumber: event.columnNumber,
            timestamp: new Date().toISOString(),
            userAgent: navigator.userAgent,
            url: window.location.href,
            referrer: document.referrer,
            violationId: this.generateViolationId(event)
        };

        // Avoid duplicate reports
        if (this.reportedViolations.has(violation.violationId)) {
            return;
        }

        this.reportedViolations.add(violation.violationId);

        // Log locally
        console.warn('CSP Violation:', violation);

        // Report to server
        this.reportViolation(violation);

        // Show UI notification for development
        if (this.isDevelopmentMode()) {
            this.showViolationNotification(violation);
        }

        // Trigger automatic policy adjustment if needed
        this.analyzeViolationForPolicyUpdate(violation);
    }

    /**
     * Handle CSP violation from ReportingObserver
     */
    handleCSPViolationReport(report) {
        const violation = {
            type: 'csp-violation-report',
            ...report.body,
            timestamp: new Date().toISOString(),
            url: window.location.href,
            violationId: this.generateViolationId(report.body)
        };

        if (!this.reportedViolations.has(violation.violationId)) {
            this.reportedViolations.add(violation.violationId);
            this.reportViolation(violation);
        }
    }

    /**
     * Generate unique violation ID
     */
    generateViolationId(violation) {
        const key = `${violation.blockedURI || violation['blocked-uri']}_${violation.violatedDirective || violation['violated-directive']}_${violation.sourceFile || violation['source-file']}`;
        return btoa(key).replace(/[^a-zA-Z0-9]/g, '').substring(0, 16);
    }

    /**
     * Report violation to server
     */
    async reportViolation(violation) {
        try {
            await fetch(this.config.reportEndpoint, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-Requested-With': 'XMLHttpRequest'
                },
                body: JSON.stringify(violation)
            });
        } catch (error) {
            console.error('Failed to report CSP violation:', error);
        }
    }

    /**
     * Show violation notification in development mode
     */
    showViolationNotification(violation) {
        const notification = document.createElement('div');
        notification.className = 'csp-violation-notification';
        notification.innerHTML = `
            <div class="violation-header">
                <strong>CSP Violation</strong>
                <button class="close-btn" onclick="this.parentElement.parentElement.remove()">×</button>
            </div>
            <div class="violation-details">
                <p><strong>Blocked:</strong> ${violation.blockedURI}</p>
                <p><strong>Directive:</strong> ${violation.violatedDirective}</p>
                <p><strong>Source:</strong> ${violation.sourceFile}:${violation.lineNumber}</p>
            </div>
        `;

        document.body.appendChild(notification);

        // Add styles if not already present
        this.addViolationNotificationStyles();

        // Auto-remove after 10 seconds
        setTimeout(() => {
            if (notification.parentElement) {
                notification.remove();
            }
        }, 10000);
    }

    /**
     * Add styles for violation notifications
     */
    addViolationNotificationStyles() {
        if (document.querySelector('#csp-violation-styles')) return;

        const styles = `
            .csp-violation-notification {
                position: fixed;
                top: 20px;
                right: 20px;
                background: #fef2f2;
                border: 2px solid #fca5a5;
                border-radius: 8px;
                padding: 16px;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
                z-index: 10000;
                max-width: 400px;
                font-family: monospace;
                font-size: 12px;
            }

            .violation-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 8px;
                color: #dc2626;
                font-weight: bold;
            }

            .close-btn {
                background: none;
                border: none;
                font-size: 18px;
                cursor: pointer;
                color: #dc2626;
            }

            .violation-details p {
                margin: 4px 0;
                color: #374151;
                word-break: break-all;
            }
        `;

        const styleSheet = document.createElement('style');
        styleSheet.id = 'csp-violation-styles';
        styleSheet.textContent = styles;
        document.head.appendChild(styleSheet);
    }

    /**
     * Analyze violation for potential policy updates
     */
    analyzeViolationForPolicyUpdate(violation) {
        // Common safe adjustments
        const safeAdjustments = {
            'script-src': {
                patterns: [
                    { match: /googleapis\.com/, allow: 'https://apis.googleapis.com' },
                    { match: /cdnjs\.cloudflare\.com/, allow: 'https://cdnjs.cloudflare.com' },
                    { match: /unpkg\.com/, allow: 'https://unpkg.com' }
                ]
            },
            'style-src': {
                patterns: [
                    { match: /fonts\.googleapis\.com/, allow: 'https://fonts.googleapis.com' },
                    { match: /cdnjs\.cloudflare\.com/, allow: 'https://cdnjs.cloudflare.com' }
                ]
            },
            'font-src': {
                patterns: [
                    { match: /fonts\.gstatic\.com/, allow: 'https://fonts.gstatic.com' }
                ]
            }
        };

        const directive = violation.violatedDirective;
        const blockedURI = violation.blockedURI;

        if (safeAdjustments[directive]) {
            for (const pattern of safeAdjustments[directive].patterns) {
                if (pattern.match.test(blockedURI)) {
                    this.suggestPolicyUpdate(directive, pattern.allow);
                    break;
                }
            }
        }
    }

    /**
     * Suggest CSP policy update
     */
    suggestPolicyUpdate(directive, allowedSource) {
        console.log(`CSP Suggestion: Consider adding '${allowedSource}' to ${directive}`);

        // In development, could show UI suggestion
        if (this.isDevelopmentMode()) {
            this.showPolicyUpdateSuggestion(directive, allowedSource);
        }
    }

    /**
     * Show policy update suggestion
     */
    showPolicyUpdateSuggestion(directive, allowedSource) {
        const suggestion = document.createElement('div');
        suggestion.className = 'csp-suggestion';
        suggestion.innerHTML = `
            <div class="suggestion-content">
                <h4>CSP Policy Suggestion</h4>
                <p>Consider updating <code>${directive}</code> to include:</p>
                <code>${allowedSource}</code>
                <button onclick="this.parentElement.parentElement.remove()">Dismiss</button>
            </div>
        `;

        document.body.appendChild(suggestion);

        setTimeout(() => {
            if (suggestion.parentElement) {
                suggestion.remove();
            }
        }, 15000);
    }

    /**
     * Setup automatic nonce refresh
     */
    setupNonceRefresh() {
        if (!this.config.autoRefreshNonce) return;

        setInterval(() => {
            this.refreshNonce();
        }, this.config.refreshInterval);
    }

    /**
     * Refresh CSP nonce
     */
    async refreshNonce() {
        try {
            const response = await fetch('/api/v1/security/refresh-nonce', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-Requested-With': 'XMLHttpRequest'
                }
            });

            if (response.ok) {
                const data = await response.json();
                if (data.nonce) {
                    this.currentNonce = data.nonce;
                    this.updateNonceMeta();

                    // Update any existing script tags with nonce
                    this.updateScriptNonces();
                }
            }
        } catch (error) {
            console.error('Failed to refresh CSP nonce:', error);
        }
    }

    /**
     * Update script tags with new nonce
     */
    updateScriptNonces() {
        const scripts = document.querySelectorAll('script[nonce]');
        for (const script of scripts) {
            script.setAttribute('nonce', this.currentNonce);
        }

        const styles = document.querySelectorAll('style[nonce]');
        for (const style of styles) {
            style.setAttribute('nonce', this.currentNonce);
        }
    }

    /**
     * Setup security headers monitoring
     */
    setupSecurityHeaders() {
        // Monitor for missing security headers
        this.checkSecurityHeaders();

        // Check for mixed content
        this.checkMixedContent();

        // Monitor for insecure protocols
        this.monitorInsecureProtocols();
    }

    /**
     * Check for required security headers
     */
    checkSecurityHeaders() {
        const requiredHeaders = [
            'X-Frame-Options',
            'X-Content-Type-Options',
            'X-XSS-Protection',
            'Strict-Transport-Security',
            'Content-Security-Policy'
        ];

        // Note: We can't directly check response headers from JavaScript
        // This would need to be implemented server-side
        console.log('Security headers check should be implemented server-side');
    }

    /**
     * Check for mixed content issues
     */
    checkMixedContent() {
        if (window.location.protocol === 'https:') {
            // Check for HTTP resources
            const httpResources = [];

            // Check images
            document.querySelectorAll('img[src^="http:"]').forEach(img => {
                httpResources.push({ type: 'image', src: img.src });
            });

            // Check scripts
            document.querySelectorAll('script[src^="http:"]').forEach(script => {
                httpResources.push({ type: 'script', src: script.src });
            });

            // Check stylesheets
            document.querySelectorAll('link[href^="http:"]').forEach(link => {
                httpResources.push({ type: 'stylesheet', href: link.href });
            });

            if (httpResources.length > 0) {
                console.warn('Mixed content detected:', httpResources);
                this.reportMixedContent(httpResources);
            }
        }
    }

    /**
     * Report mixed content to server
     */
    async reportMixedContent(resources) {
        try {
            await fetch('/api/v1/security/mixed-content', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-Requested-With': 'XMLHttpRequest'
                },
                body: JSON.stringify({
                    url: window.location.href,
                    resources: resources,
                    timestamp: new Date().toISOString()
                })
            });
        } catch (error) {
            console.error('Failed to report mixed content:', error);
        }
    }

    /**
     * Monitor for insecure protocols
     */
    monitorInsecureProtocols() {
        // Check for insecure WebSocket connections
        const originalWebSocket = window.WebSocket;
        window.WebSocket = function(url, protocols) {
            if (url.startsWith('ws:') && window.location.protocol === 'https:') {
                console.warn('Insecure WebSocket connection attempt:', url);
            }
            return new originalWebSocket(url, protocols);
        };

        // Monitor fetch requests for insecure URLs
        const originalFetch = window.fetch;
        window.fetch = function(url, options) {
            if (typeof url === 'string' && url.startsWith('http:') && window.location.protocol === 'https:') {
                console.warn('Insecure fetch request:', url);
            }
            return originalFetch.apply(this, arguments);
        };
    }

    /**
     * Monitor CSP compliance
     */
    monitorCSPCompliance() {
        // Check for inline scripts without nonce
        const inlineScripts = document.querySelectorAll('script:not([src]):not([nonce])');
        if (inlineScripts.length > 0) {
            console.warn('Inline scripts without nonce detected:', inlineScripts.length);
        }

        // Check for inline styles without nonce
        const inlineStyles = document.querySelectorAll('style:not([nonce])');
        if (inlineStyles.length > 0) {
            console.warn('Inline styles without nonce detected:', inlineStyles.length);
        }

        // Monitor dynamically added content
        const observer = new MutationObserver((mutations) => {
            for (const mutation of mutations) {
                if (mutation.type === 'childList') {
                    for (const node of mutation.addedNodes) {
                        if (node.nodeType === Node.ELEMENT_NODE) {
                            this.checkNewElement(node);
                        }
                    }
                }
            }
        });

        observer.observe(document.body, {
            childList: true,
            subtree: true
        });
    }

    /**
     * Check newly added elements for CSP compliance
     */
    checkNewElement(element) {
        // Check for scripts without nonce
        if (element.tagName === 'SCRIPT' && !element.src && !element.hasAttribute('nonce')) {
            console.warn('Inline script added without nonce:', element);

            // Automatically add nonce if possible
            if (this.currentNonce) {
                element.setAttribute('nonce', this.currentNonce);
            }
        }

        // Check for styles without nonce
        if (element.tagName === 'STYLE' && !element.hasAttribute('nonce')) {
            console.warn('Inline style added without nonce:', element);

            // Automatically add nonce if possible
            if (this.currentNonce) {
                element.setAttribute('nonce', this.currentNonce);
            }
        }

        // Check child elements
        const scripts = element.querySelectorAll && element.querySelectorAll('script:not([src]):not([nonce])');
        const styles = element.querySelectorAll && element.querySelectorAll('style:not([nonce])');

        if (scripts && scripts.length > 0) {
            console.warn('Container has inline scripts without nonce:', scripts.length);
        }

        if (styles && styles.length > 0) {
            console.warn('Container has inline styles without nonce:', styles.length);
        }
    }

    /**
     * Check if in development mode
     */
    isDevelopmentMode() {
        return window.location.hostname === 'localhost' ||
               window.location.hostname === '127.0.0.1' ||
               window.location.hostname.includes('dev') ||
               window.location.search.includes('debug=true');
    }

    /**
     * Get CSP violation statistics
     */
    getViolationStats() {
        return {
            totalViolations: this.violationCount,
            uniqueViolations: this.reportedViolations.size,
            reportingEnabled: this.config.enableViolationReporting
        };
    }

    /**
     * Add trusted source to CSP (for development)
     */
    addTrustedSource(directive, source) {
        console.log(`CSP: Adding trusted source '${source}' to ${directive} (development only)`);
        // This would typically be handled server-side
    }

    /**
     * Create script element with nonce
     */
    createScriptWithNonce(src, content) {
        const script = document.createElement('script');
        if (this.currentNonce) {
            script.setAttribute('nonce', this.currentNonce);
        }

        if (src) {
            script.src = src;
        } else if (content) {
            script.textContent = content;
        }

        return script;
    }

    /**
     * Create style element with nonce
     */
    createStyleWithNonce(content) {
        const style = document.createElement('style');
        if (this.currentNonce) {
            style.setAttribute('nonce', this.currentNonce);
        }

        if (content) {
            style.textContent = content;
        }

        return style;
    }
}

// Initialize CSP manager when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.cspManager = new CSPManager();
});

// Export for use in modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = CSPManager;
}
