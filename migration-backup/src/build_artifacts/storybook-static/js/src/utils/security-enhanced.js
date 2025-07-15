/**
 * Enhanced Security Manager for Frontend Protection
 * Implements comprehensive client-side security monitoring and protection
 */

class EnhancedSecurityManager {
  constructor(options = {}) {
    this.options = {
      enableXSSProtection: true,
      enableCSRFProtection: true,
      enableClickjackingProtection: true,
      enableContentValidation: true,
      enableSecurityHeaders: true,
      enableActivityMonitoring: true,
      enableThreatDetection: true,
      reportingEndpoint: '/api/security/report',
      ...options
    };

    this.securityMetrics = {
      xssAttempts: 0,
      csrfViolations: 0,
      suspiciousActivity: 0,
      blockedRequests: 0,
      securityEvents: []
    };

    this.cspNonce = this.generateNonce();
    this.trustedDomains = [
      window.location.origin,
      'https://cdn.jsdelivr.net',
      'https://unpkg.com'
    ];

    this.init();
  }

  init() {
    // Initialize security protections
    this.initXSSProtection();
    this.initCSRFProtection();
    this.initClickjackingProtection();
    this.initContentValidation();
    this.initActivityMonitoring();
    this.initThreatDetection();
    this.initSecurityHeaders();

    // Setup security event listeners
    this.setupSecurityEventListeners();

    // Start security monitoring
    this.startSecurityMonitoring();

    console.log('🔒 Enhanced Security Manager initialized');
  }

  generateNonce() {
    const array = new Uint8Array(16);
    crypto.getRandomValues(array);
    return btoa(String.fromCharCode.apply(null, array));
  }

  // XSS Protection
  initXSSProtection() {
    if (!this.options.enableXSSProtection) return;

    // Monitor dangerous DOM modifications
    this.setupDOMMonitoring();

    // Sanitize user inputs
    this.setupInputSanitization();

    // Monitor eval usage
    this.monitorEvalUsage();
  }

  setupDOMMonitoring() {
    const observer = new MutationObserver((mutations) => {
      mutations.forEach((mutation) => {
        if (mutation.type === 'childList') {
          mutation.addedNodes.forEach((node) => {
            if (node.nodeType === Node.ELEMENT_NODE) {
              this.validateElement(node);
            }
          });
        }
      });
    });

    observer.observe(document.body, {
      childList: true,
      subtree: true
    });
  }

  validateElement(element) {
    // Check for dangerous scripts
    if (element.tagName === 'SCRIPT') {
      const src = element.src;
      const content = element.textContent;

      if (src && !this.isTrustedDomain(src)) {
        this.blockElement(element, 'Untrusted script source');
        return;
      }

      if (content && this.containsSuspiciousCode(content)) {
        this.blockElement(element, 'Suspicious script content');
        return;
      }
    }

    // Check for dangerous attributes
    const dangerousAttrs = ['onclick', 'onload', 'onerror', 'onmouseover'];
    dangerousAttrs.forEach(attr => {
      if (element.hasAttribute(attr)) {
        element.removeAttribute(attr);
        this.reportSecurityEvent('xss_attempt', `Removed dangerous attribute: ${attr}`);
      }
    });
  }

  containsSuspiciousCode(code) {
    const suspiciousPatterns = [
      /eval\s*\(/,
      /document\.write/,
      /innerHTML\s*=/,
      /javascript:/,
      /vbscript:/,
      /onload\s*=/,
      /onerror\s*=/
    ];

    return suspiciousPatterns.some(pattern => pattern.test(code));
  }

  blockElement(element, reason) {
    element.remove();
    this.securityMetrics.blockedRequests++;
    this.reportSecurityEvent('element_blocked', reason);
  }

  isTrustedDomain(url) {
    try {
      const urlObj = new URL(url);
      return this.trustedDomains.some(domain =>
        urlObj.origin === domain || urlObj.hostname.endsWith(domain.replace('https://', ''))
      );
    } catch {
      return false;
    }
  }

  setupInputSanitization() {
    // Monitor form inputs
    document.addEventListener('input', (event) => {
      if (event.target.matches('input, textarea')) {
        this.sanitizeInput(event.target);
      }
    });

    // Monitor paste events
    document.addEventListener('paste', (event) => {
      setTimeout(() => {
        if (event.target.matches('input, textarea')) {
          this.sanitizeInput(event.target);
        }
      }, 0);
    });
  }

  sanitizeInput(input) {
    const originalValue = input.value;
    const sanitizedValue = this.sanitizeString(originalValue);

    if (originalValue !== sanitizedValue) {
      input.value = sanitizedValue;
      this.securityMetrics.xssAttempts++;
      this.reportSecurityEvent('input_sanitized', 'XSS attempt detected and blocked');
    }
  }

  sanitizeString(str) {
    // Remove script tags
    str = str.replace(/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, '');

    // Remove dangerous protocols
    str = str.replace(/(javascript|vbscript|data):/gi, '');

    // Remove event handlers
    str = str.replace(/on\w+\s*=/gi, '');

    // Encode HTML entities
    str = str.replace(/[<>"'&]/g, (match) => {
      const entities = {
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#x27;',
        '&': '&amp;'
      };
      return entities[match];
    });

    return str;
  }

  monitorEvalUsage() {
    const originalEval = window.eval;
    window.eval = (...args) => {
      this.reportSecurityEvent('eval_usage', 'eval() function called');
      this.securityMetrics.suspiciousActivity++;

      // In strict security mode, block eval
      if (this.options.blockEval) {
        throw new Error('eval() is blocked for security reasons');
      }

      return originalEval.apply(this, args);
    };
  }

  // CSRF Protection
  initCSRFProtection() {
    if (!this.options.enableCSRFProtection) return;

    this.csrfToken = this.getCSRFToken();
    this.setupCSRFHeaders();
    this.validateCSRFTokens();
  }

  getCSRFToken() {
    const metaTag = document.querySelector('meta[name="csrf-token"]');
    return metaTag ? metaTag.getAttribute('content') : null;
  }

  setupCSRFHeaders() {
    // Intercept all AJAX requests to add CSRF token
    const originalFetch = window.fetch;
    window.fetch = (url, options = {}) => {
      if (this.shouldAddCSRFToken(url, options.method)) {
        options.headers = options.headers || {};
        options.headers['X-CSRFToken'] = this.csrfToken;
      }

      return originalFetch(url, options);
    };

    // Intercept XMLHttpRequest
    const originalOpen = XMLHttpRequest.prototype.open;
    XMLHttpRequest.prototype.open = function(method, url, ...args) {
      this._url = url;
      this._method = method;
      return originalOpen.apply(this, [method, url, ...args]);
    };

    const originalSend = XMLHttpRequest.prototype.send;
    XMLHttpRequest.prototype.send = function(...args) {
      if (this._url && this._method &&
          this.shouldAddCSRFToken && this.shouldAddCSRFToken(this._url, this._method)) {
        this.setRequestHeader('X-CSRFToken', this.csrfToken);
      }
      return originalSend.apply(this, args);
    };
  }

  shouldAddCSRFToken(url, method) {
    // Only add CSRF token for state-changing requests to same origin
    const safeMethods = ['GET', 'HEAD', 'OPTIONS', 'TRACE'];
    const isSafeMethod = !method || safeMethods.includes(method.toUpperCase());
    const isSameOrigin = url.startsWith('/') || url.startsWith(window.location.origin);

    return !isSafeMethod && isSameOrigin;
  }

  validateCSRFTokens() {
    // Monitor form submissions
    document.addEventListener('submit', (event) => {
      if (event.target.matches('form')) {
        this.validateFormCSRF(event.target);
      }
    });
  }

  validateFormCSRF(form) {
    const method = (form.method || 'GET').toUpperCase();
    const safeMethods = ['GET', 'HEAD'];

    if (!safeMethods.includes(method)) {
      const csrfInput = form.querySelector('input[name="csrf_token"]');

      if (!csrfInput || !csrfInput.value) {
        this.securityMetrics.csrfViolations++;
        this.reportSecurityEvent('csrf_violation', 'Form submitted without CSRF token');

        // Add CSRF token if missing
        if (!csrfInput && this.csrfToken) {
          const hiddenInput = document.createElement('input');
          hiddenInput.type = 'hidden';
          hiddenInput.name = 'csrf_token';
          hiddenInput.value = this.csrfToken;
          form.appendChild(hiddenInput);
        }
      }
    }
  }

  // Clickjacking Protection
  initClickjackingProtection() {
    if (!this.options.enableClickjackingProtection) return;

    // Check if page is in iframe
    if (window !== window.top) {
      this.reportSecurityEvent('clickjacking_attempt', 'Page loaded in iframe');

      // Optional: Break out of iframe
      if (this.options.breakoutOfFrames) {
        window.top.location = window.location;
      }
    }

    // Monitor for overlay attacks
    this.monitorOverlayAttacks();
  }

  monitorOverlayAttacks() {
    document.addEventListener('click', (event) => {
      const element = event.target;
      const rect = element.getBoundingClientRect();

      // Check for elements with suspicious positioning
      const style = getComputedStyle(element);
      if (style.position === 'fixed' || style.position === 'absolute') {
        if (style.zIndex > 999999) {
          this.reportSecurityEvent('overlay_attack', 'Suspicious high z-index element clicked');
        }
      }
    });
  }

  // Content Validation
  initContentValidation() {
    if (!this.options.enableContentValidation) return;

    this.validateImages();
    this.validateLinks();
    this.validateForms();
  }

  validateImages() {
    document.addEventListener('error', (event) => {
      if (event.target.tagName === 'IMG') {
        const src = event.target.src;
        if (!this.isTrustedDomain(src)) {
          this.reportSecurityEvent('untrusted_image', `Blocked image from: ${src}`);
          event.target.remove();
        }
      }
    }, true);
  }

  validateLinks() {
    document.addEventListener('click', (event) => {
      if (event.target.matches('a[href]')) {
        const href = event.target.href;

        // Check for dangerous protocols
        if (href.match(/^(javascript|vbscript|data):/i)) {
          event.preventDefault();
          this.reportSecurityEvent('dangerous_link', `Blocked dangerous link: ${href}`);
        }

        // Check for external links
        if (!href.startsWith(window.location.origin) && !href.startsWith('/')) {
          // Add rel="noopener noreferrer" for external links
          event.target.rel = 'noopener noreferrer';
        }
      }
    });
  }

  validateForms() {
    document.addEventListener('submit', (event) => {
      const form = event.target;
      if (form.matches('form')) {
        // Validate form action
        const action = form.action;
        if (action && !this.isTrustedDomain(action)) {
          event.preventDefault();
          this.reportSecurityEvent('untrusted_form_action', `Blocked form submission to: ${action}`);
        }
      }
    });
  }

  // Activity Monitoring
  initActivityMonitoring() {
    if (!this.options.enableActivityMonitoring) return;

    this.activityData = {
      pageViews: 0,
      clicks: 0,
      keystrokes: 0,
      mouseMoves: 0,
      suspiciousActivity: 0
    };

    this.monitorUserActivity();
  }

  monitorUserActivity() {
    document.addEventListener('click', () => {
      this.activityData.clicks++;
    });

    document.addEventListener('keydown', () => {
      this.activityData.keystrokes++;
    });

    document.addEventListener('mousemove', () => {
      this.activityData.mouseMoves++;
    });

    // Report activity periodically
    setInterval(() => {
      this.reportActivityMetrics();
    }, 60000); // Every minute
  }

  reportActivityMetrics() {
    if (this.activityData.clicks > 1000 || this.activityData.keystrokes > 10000) {
      this.reportSecurityEvent('suspicious_activity', 'Unusually high user activity detected');
    }
  }

  // Threat Detection
  initThreatDetection() {
    if (!this.options.enableThreatDetection) return;

    this.setupThreatPatterns();
    this.monitorNetworkRequests();
    this.detectAutomatedBehavior();
  }

  setupThreatPatterns() {
    this.threatPatterns = {
      xss: [
        /<script/i,
        /javascript:/i,
        /vbscript:/i,
        /onload=/i,
        /onerror=/i
      ],
      sqlInjection: [
        /union\s+select/i,
        /drop\s+table/i,
        /insert\s+into/i,
        /delete\s+from/i,
        /update\s+set/i
      ],
      commandInjection: [
        /;\s*(rm|del|format)/i,
        /\|\s*(curl|wget)/i,
        /&&\s*(cat|type)/i
      ]
    };
  }

  monitorNetworkRequests() {
    // Monitor fetch requests
    const originalFetch = window.fetch;
    window.fetch = async (...args) => {
      const [url, options] = args;
      this.analyzeRequest(url, options);
      return originalFetch.apply(this, args);
    };
  }

  analyzeRequest(url, options = {}) {
    const body = options.body;

    if (body) {
      Object.values(this.threatPatterns).forEach(patterns => {
        patterns.forEach(pattern => {
          if (pattern.test(body)) {
            this.reportSecurityEvent('threat_detected', `Malicious pattern detected in request to: ${url}`);
            this.securityMetrics.suspiciousActivity++;
          }
        });
      });
    }
  }

  detectAutomatedBehavior() {
    let rapidClicks = 0;
    let lastClickTime = 0;

    document.addEventListener('click', () => {
      const now = Date.now();
      if (now - lastClickTime < 100) { // Less than 100ms between clicks
        rapidClicks++;
        if (rapidClicks > 10) {
          this.reportSecurityEvent('bot_detected', 'Automated clicking behavior detected');
        }
      } else {
        rapidClicks = 0;
      }
      lastClickTime = now;
    });
  }

  // Security Headers
  initSecurityHeaders() {
    if (!this.options.enableSecurityHeaders) return;

    // Check for security headers
    this.validateSecurityHeaders();
  }

  validateSecurityHeaders() {
    // This would typically be done server-side, but we can check for CSP
    const metaCsp = document.querySelector('meta[http-equiv="Content-Security-Policy"]');
    if (!metaCsp) {
      this.reportSecurityEvent('missing_csp', 'Content Security Policy not found');
    }
  }

  // Security Event Management
  setupSecurityEventListeners() {
    // Listen for CSP violations
    document.addEventListener('securitypolicyviolation', (event) => {
      this.reportSecurityEvent('csp_violation', {
        blockedURI: event.blockedURI,
        violatedDirective: event.violatedDirective,
        originalPolicy: event.originalPolicy
      });
    });

    // Listen for unhandled errors that might indicate attacks
    window.addEventListener('error', (event) => {
      if (event.message.includes('Script error')) {
        this.reportSecurityEvent('script_error', 'Potential XSS attempt blocked');
      }
    });
  }

  startSecurityMonitoring() {
    // Periodic security checks
    setInterval(() => {
      this.performSecurityCheck();
    }, 30000); // Every 30 seconds

    // Report metrics periodically
    setInterval(() => {
      this.reportSecurityMetrics();
    }, 300000); // Every 5 minutes
  }

  performSecurityCheck() {
    // Check for suspicious DOM modifications
    const scripts = document.querySelectorAll('script');
    scripts.forEach(script => {
      if (script.src && !this.isTrustedDomain(script.src)) {
        this.blockElement(script, 'Untrusted script detected during security check');
      }
    });

    // Check for suspicious iframes
    const iframes = document.querySelectorAll('iframe');
    iframes.forEach(iframe => {
      if (iframe.src && !this.isTrustedDomain(iframe.src)) {
        this.blockElement(iframe, 'Untrusted iframe detected');
      }
    });
  }

  reportSecurityEvent(type, details) {
    const event = {
      type,
      details,
      timestamp: new Date().toISOString(),
      url: window.location.href,
      userAgent: navigator.userAgent
    };

    this.securityMetrics.securityEvents.push(event);

    // Limit stored events
    if (this.securityMetrics.securityEvents.length > 100) {
      this.securityMetrics.securityEvents = this.securityMetrics.securityEvents.slice(-50);
    }

    // Report to server
    this.sendSecurityReport(event);

    console.warn('🚨 Security Event:', event);
  }

  reportSecurityMetrics() {
    const report = {
      ...this.securityMetrics,
      timestamp: new Date().toISOString(),
      url: window.location.href
    };

    this.sendSecurityReport(report, 'metrics');
  }

  async sendSecurityReport(data, type = 'event') {
    try {
      const response = await fetch(this.options.reportingEndpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-CSRFToken': this.csrfToken
        },
        body: JSON.stringify({
          type,
          data,
          client_timestamp: Date.now()
        })
      });

      if (!response.ok) {
        console.warn('Failed to send security report:', response.status);
      }
    } catch (error) {
      console.warn('Error sending security report:', error);
    }
  }

  // Public API
  getSecurityMetrics() {
    return { ...this.securityMetrics };
  }

  addTrustedDomain(domain) {
    if (!this.trustedDomains.includes(domain)) {
      this.trustedDomains.push(domain);
    }
  }

  removeTrustedDomain(domain) {
    const index = this.trustedDomains.indexOf(domain);
    if (index > -1) {
      this.trustedDomains.splice(index, 1);
    }
  }

  updateCSRFToken(token) {
    this.csrfToken = token;

    // Update meta tag
    const metaTag = document.querySelector('meta[name="csrf-token"]');
    if (metaTag) {
      metaTag.setAttribute('content', token);
    }
  }

  // Cleanup
  destroy() {
    // Remove event listeners and restore original functions
    // Implementation would restore original fetch, eval, etc.
    console.log('🔒 Security Manager destroyed');
  }
}

// Initialize enhanced security manager
const enhancedSecurityManager = new EnhancedSecurityManager({
  enableXSSProtection: true,
  enableCSRFProtection: true,
  enableClickjackingProtection: true,
  enableContentValidation: true,
  enableActivityMonitoring: true,
  enableThreatDetection: true,
  blockEval: true,
  breakoutOfFrames: true
});

// Export for global access
window.EnhancedSecurityManager = EnhancedSecurityManager;
window.enhancedSecurityManager = enhancedSecurityManager;

export { EnhancedSecurityManager, enhancedSecurityManager };
export default enhancedSecurityManager;
