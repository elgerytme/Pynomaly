/**
 * Rate Limiting UI Enhancement Module
 * Provides visual feedback and progressive delays for rate-limited requests
 */

class RateLimitUI {
    constructor() {
        this.rateLimitStatus = {
            remaining: null,
            resetTime: null,
            retryAfter: null,
            isLimited: false
        };

        this.requestQueue = [];
        this.isProcessing = false;
        this.progressiveDelay = 1000; // Start with 1 second
        this.maxDelay = 30000; // Max 30 seconds

        this.init();
    }

    init() {
        this.createRateLimitIndicator();
        this.interceptAjaxRequests();
        this.setupProgressiveDelayHandling();
        this.setupErrorRecovery();
    }

    /**
     * Create rate limit status indicator in the UI
     */
    createRateLimitIndicator() {
        const indicator = document.createElement('div');
        indicator.id = 'rate-limit-indicator';
        indicator.className = 'rate-limit-indicator hidden';
        indicator.innerHTML = `
            <div class="rate-limit-content">
                <div class="rate-limit-icon">
                    <svg class="animate-spin h-5 w-5 text-yellow-500" fill="none" viewBox="0 0 24 24">
                        <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                        <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                </div>
                <div class="rate-limit-text">
                    <span class="rate-limit-message">Rate limit active</span>
                    <div class="rate-limit-details">
                        <span class="remaining-requests">Remaining: <span id="remaining-count">--</span></span>
                        <span class="reset-timer">Reset in: <span id="reset-countdown">--</span></span>
                    </div>
                </div>
                <div class="rate-limit-progress">
                    <div class="progress-bar" id="rate-limit-progress"></div>
                </div>
            </div>
        `;

        // Add to page
        document.body.appendChild(indicator);

        // Add CSS styles
        this.addRateLimitStyles();
    }

    addRateLimitStyles() {
        const styles = `
            .rate-limit-indicator {
                position: fixed;
                top: 20px;
                right: 20px;
                background: rgba(255, 255, 255, 0.95);
                border: 2px solid #f59e0b;
                border-radius: 8px;
                padding: 16px;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
                z-index: 9999;
                min-width: 280px;
                transition: all 0.3s ease;
            }

            .rate-limit-indicator.hidden {
                opacity: 0;
                transform: translateX(100%);
                pointer-events: none;
            }

            .rate-limit-content {
                display: flex;
                align-items: center;
                gap: 12px;
            }

            .rate-limit-icon {
                flex-shrink: 0;
            }

            .rate-limit-text {
                flex: 1;
            }

            .rate-limit-message {
                font-weight: 600;
                color: #92400e;
                display: block;
                margin-bottom: 4px;
            }

            .rate-limit-details {
                font-size: 0.875rem;
                color: #78716c;
                display: flex;
                gap: 16px;
            }

            .rate-limit-progress {
                width: 100%;
                height: 4px;
                background: #fef3c7;
                border-radius: 2px;
                margin-top: 8px;
                overflow: hidden;
            }

            .progress-bar {
                height: 100%;
                background: linear-gradient(90deg, #f59e0b, #d97706);
                transition: width 0.3s ease;
                width: 0%;
            }

            .rate-limit-indicator.severe {
                border-color: #dc2626;
                background: rgba(254, 242, 242, 0.95);
            }

            .rate-limit-indicator.severe .rate-limit-message {
                color: #dc2626;
            }

            .rate-limit-indicator.severe .progress-bar {
                background: linear-gradient(90deg, #dc2626, #b91c1c);
            }

            @keyframes pulse-warning {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.7; }
            }

            .rate-limit-indicator.pulsing {
                animation: pulse-warning 1.5s ease-in-out infinite;
            }
        `;

        const styleSheet = document.createElement('style');
        styleSheet.textContent = styles;
        document.head.appendChild(styleSheet);
    }

    /**
     * Intercept AJAX requests to monitor rate limits
     */
    interceptAjaxRequests() {
        const originalFetch = window.fetch;
        const rateLimitUI = this;

        window.fetch = async function(...args) {
            try {
                const response = await originalFetch.apply(this, args);
                rateLimitUI.handleResponse(response);
                return response;
            } catch (error) {
                rateLimitUI.handleError(error);
                throw error;
            }
        };

        // Intercept XMLHttpRequest
        const originalOpen = XMLHttpRequest.prototype.open;
        const originalSend = XMLHttpRequest.prototype.send;

        XMLHttpRequest.prototype.open = function(...args) {
            this._rateLimitUI = rateLimitUI;
            return originalOpen.apply(this, args);
        };

        XMLHttpRequest.prototype.send = function(...args) {
            this.addEventListener('load', function() {
                if (this._rateLimitUI) {
                    this._rateLimitUI.handleXHRResponse(this);
                }
            });
            return originalSend.apply(this, args);
        };
    }

    /**
     * Handle fetch response for rate limit headers
     */
    handleResponse(response) {
        this.updateRateLimitStatus(response.headers);

        if (response.status === 429) {
            this.handleRateLimitExceeded(response);
        }
    }

    /**
     * Handle XMLHttpRequest response
     */
    handleXHRResponse(xhr) {
        const headers = {
            get: (name) => xhr.getResponseHeader(name)
        };

        this.updateRateLimitStatus(headers);

        if (xhr.status === 429) {
            this.handleRateLimitExceeded({ headers });
        }
    }

    /**
     * Update rate limit status from response headers
     */
    updateRateLimitStatus(headers) {
        const remaining = headers.get('X-RateLimit-Remaining');
        const reset = headers.get('X-RateLimit-Reset');
        const retryAfter = headers.get('Retry-After');

        if (remaining !== null) {
            this.rateLimitStatus.remaining = parseInt(remaining);
        }

        if (reset !== null) {
            this.rateLimitStatus.resetTime = new Date(parseInt(reset) * 1000);
        }

        if (retryAfter !== null) {
            this.rateLimitStatus.retryAfter = parseInt(retryAfter);
        }

        this.updateUI();
    }

    /**
     * Handle rate limit exceeded response
     */
    handleRateLimitExceeded(response) {
        this.rateLimitStatus.isLimited = true;
        this.showRateLimitIndicator(true);

        // Apply progressive delay
        this.applyProgressiveDelay();

        // Show user-friendly message
        this.showRateLimitMessage();

        // Log for monitoring
        console.warn('Rate limit exceeded:', this.rateLimitStatus);
    }

    /**
     * Apply progressive delay for subsequent requests
     */
    applyProgressiveDelay() {
        this.progressiveDelay = Math.min(this.progressiveDelay * 1.5, this.maxDelay);

        // Visual feedback for delay
        this.showDelayCountdown(this.progressiveDelay);
    }

    /**
     * Show delay countdown to user
     */
    showDelayCountdown(delayMs) {
        const indicator = document.getElementById('rate-limit-indicator');
        const countdown = document.getElementById('reset-countdown');

        if (!countdown) return;

        let remaining = Math.ceil(delayMs / 1000);

        const updateCountdown = () => {
            countdown.textContent = `${remaining}s`;

            if (remaining > 0) {
                remaining--;
                setTimeout(updateCountdown, 1000);
            } else {
                this.hideRateLimitIndicator();
                this.resetProgressiveDelay();
            }
        };

        updateCountdown();
    }

    /**
     * Reset progressive delay on successful request
     */
    resetProgressiveDelay() {
        this.progressiveDelay = 1000;
        this.rateLimitStatus.isLimited = false;
    }

    /**
     * Show rate limit indicator
     */
    showRateLimitIndicator(severe = false) {
        const indicator = document.getElementById('rate-limit-indicator');
        if (indicator) {
            indicator.classList.remove('hidden');

            if (severe) {
                indicator.classList.add('severe', 'pulsing');
            }
        }
    }

    /**
     * Hide rate limit indicator
     */
    hideRateLimitIndicator() {
        const indicator = document.getElementById('rate-limit-indicator');
        if (indicator) {
            indicator.classList.add('hidden');
            indicator.classList.remove('severe', 'pulsing');
        }
    }

    /**
     * Update UI elements with current rate limit status
     */
    updateUI() {
        const remainingElement = document.getElementById('remaining-count');
        const progressBar = document.getElementById('rate-limit-progress');

        if (remainingElement && this.rateLimitStatus.remaining !== null) {
            remainingElement.textContent = this.rateLimitStatus.remaining;
        }

        if (progressBar && this.rateLimitStatus.remaining !== null) {
            // Assume limit of 60 requests per minute
            const limit = 60;
            const percentage = (this.rateLimitStatus.remaining / limit) * 100;
            progressBar.style.width = `${Math.max(0, percentage)}%`;
        }

        // Show indicator if requests are getting low
        if (this.rateLimitStatus.remaining !== null && this.rateLimitStatus.remaining < 10) {
            this.showRateLimitIndicator();
        } else if (this.rateLimitStatus.remaining >= 30) {
            this.hideRateLimitIndicator();
        }
    }

    /**
     * Show user-friendly rate limit message
     */
    showRateLimitMessage() {
        // Create or update toast notification
        this.showToast('Rate limit reached',
            'Please wait a moment before making more requests.',
            'warning');
    }

    /**
     * Show toast notification
     */
    showToast(title, message, type = 'info') {
        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        toast.innerHTML = `
            <div class="toast-content">
                <strong>${title}</strong>
                <p>${message}</p>
            </div>
            <button class="toast-close" onclick="this.parentElement.remove()">×</button>
        `;

        // Add toast styles if not already present
        if (!document.querySelector('#toast-styles')) {
            const toastStyles = `
                .toast {
                    position: fixed;
                    top: 80px;
                    right: 20px;
                    background: white;
                    border-radius: 8px;
                    padding: 16px;
                    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
                    z-index: 10000;
                    max-width: 350px;
                    border-left: 4px solid #3b82f6;
                    animation: slideIn 0.3s ease;
                }

                .toast-warning {
                    border-left-color: #f59e0b;
                }

                .toast-error {
                    border-left-color: #dc2626;
                }

                .toast-content strong {
                    display: block;
                    margin-bottom: 4px;
                    color: #374151;
                }

                .toast-content p {
                    margin: 0;
                    color: #6b7280;
                    font-size: 0.875rem;
                }

                .toast-close {
                    position: absolute;
                    top: 8px;
                    right: 8px;
                    background: none;
                    border: none;
                    font-size: 18px;
                    cursor: pointer;
                    color: #9ca3af;
                }

                @keyframes slideIn {
                    from { transform: translateX(100%); opacity: 0; }
                    to { transform: translateX(0); opacity: 1; }
                }
            `;

            const styleSheet = document.createElement('style');
            styleSheet.id = 'toast-styles';
            styleSheet.textContent = toastStyles;
            document.head.appendChild(styleSheet);
        }

        document.body.appendChild(toast);

        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (toast.parentElement) {
                toast.remove();
            }
        }, 5000);
    }

    /**
     * Setup progressive delay handling for form submissions
     */
    setupProgressiveDelayHandling() {
        // Intercept form submissions
        document.addEventListener('submit', (event) => {
            if (this.rateLimitStatus.isLimited) {
                event.preventDefault();
                this.showToast('Please wait',
                    `Please wait ${Math.ceil(this.progressiveDelay / 1000)} seconds before submitting.`,
                    'warning');

                setTimeout(() => {
                    event.target.submit();
                }, this.progressiveDelay);
            }
        });
    }

    /**
     * Setup error recovery mechanisms
     */
    setupErrorRecovery() {
        // Reset rate limit status on successful requests
        document.addEventListener('htmx:afterRequest', (event) => {
            if (event.detail.xhr.status < 400) {
                this.resetProgressiveDelay();
            }
        });

        // Handle network errors gracefully
        window.addEventListener('online', () => {
            this.resetProgressiveDelay();
            this.hideRateLimitIndicator();
        });
    }

    /**
     * Handle request errors
     */
    handleError(error) {
        console.error('Request error:', error);

        if (error.name === 'TypeError' && error.message.includes('Failed to fetch')) {
            this.showToast('Connection Error',
                'Unable to connect to server. Please check your connection.',
                'error');
        }
    }

    /**
     * Public method to manually trigger rate limit UI
     */
    triggerRateLimit(remaining = 0, resetTime = null) {
        this.rateLimitStatus.remaining = remaining;
        this.rateLimitStatus.resetTime = resetTime;
        this.rateLimitStatus.isLimited = true;

        this.showRateLimitIndicator(remaining === 0);
        this.updateUI();
    }

    /**
     * Get current rate limit status
     */
    getStatus() {
        return { ...this.rateLimitStatus };
    }
}

// Initialize rate limit UI when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.rateLimitUI = new RateLimitUI();
});

// Export for use in modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = RateLimitUI;
}
