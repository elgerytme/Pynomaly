/**
 * Security Manager for Web UI
 * Handles input validation, sanitization, and security monitoring
 */

class SecurityManager {
  constructor() {
    this.rateLimiters = new Map();
    this.sessionManager = new SessionManager();
    this.csrfToken = null;
    this.nonce = null;
    this.initSecurity();
  }

  initSecurity() {
    this.setupCSRFProtection();
    this.setupContentSecurityPolicy();
    this.setupInputValidation();
    this.setupSecurityMonitoring();
  }

  setupCSRFProtection() {
    // Get CSRF token from meta tag
    const csrfMeta = document.querySelector('meta[name="csrf-token"]');
    if (csrfMeta) {
      this.csrfToken = csrfMeta.getAttribute('content');
    }

    // Add CSRF token to all forms
    this.addCSRFTokenToForms();

    // Add CSRF token to AJAX requests
    this.setupAjaxCSRFProtection();
  }

  addCSRFTokenToForms() {
    const forms = document.querySelectorAll('form');
    forms.forEach(form => {
      if (!form.querySelector('input[name="csrf_token"]')) {
        const csrfInput = document.createElement('input');
        csrfInput.type = 'hidden';
        csrfInput.name = 'csrf_token';
        csrfInput.value = this.csrfToken;
        form.appendChild(csrfInput);
      }
    });
  }

  setupAjaxCSRFProtection() {
    // Override fetch to add CSRF token
    const originalFetch = window.fetch;
    window.fetch = (url, options = {}) => {
      if (this.requiresCSRFToken(url, options.method)) {
        options.headers = {
          ...options.headers,
          'X-CSRFToken': this.csrfToken
        };
      }
      return originalFetch(url, options);
    };

    // Override XMLHttpRequest
    const originalOpen = XMLHttpRequest.prototype.open;
    XMLHttpRequest.prototype.open = function(method, url, ...args) {
      this._url = url;
      this._method = method;
      return originalOpen.apply(this, [method, url, ...args]);
    };

    const originalSend = XMLHttpRequest.prototype.send;
    XMLHttpRequest.prototype.send = function(data) {
      if (securityManager.requiresCSRFToken(this._url, this._method)) {
        this.setRequestHeader('X-CSRFToken', securityManager.csrfToken);
      }
      return originalSend.apply(this, [data]);
    };
  }

  requiresCSRFToken(url, method) {
    // Add CSRF token to state-changing requests
    const stateMethods = ['POST', 'PUT', 'DELETE', 'PATCH'];
    return stateMethods.includes(method?.toUpperCase()) &&
           !url.startsWith('http') && // Same-origin requests only
           !url.includes('/api/public/'); // Exclude public APIs
  }

  setupContentSecurityPolicy() {
    // Monitor CSP violations
    document.addEventListener('securitypolicyviolation', (event) => {
      this.handleCSPViolation(event);
    });

    // Generate nonce for inline scripts
    this.nonce = this.generateNonce();
  }

  handleCSPViolation(event) {
    const violation = {
      directive: event.violatedDirective,
      blockedURI: event.blockedURI,
      documentURI: event.documentURI,
      originalPolicy: event.originalPolicy,
      timestamp: Date.now()
    };

    // Log security violation
    console.warn('CSP Violation:', violation);

    // Send to security monitoring
    this.reportSecurityEvent('csp_violation', violation);
  }

  generateNonce() {
    const array = new Uint8Array(16);
    crypto.getRandomValues(array);
    return btoa(String.fromCharCode(...array));
  }

  setupInputValidation() {
    // Global input sanitization
    document.addEventListener('input', (event) => {
      this.validateInput(event.target);
    });

    document.addEventListener('paste', (event) => {
      this.handlePaste(event);
    });
  }

  validateInput(input) {
    const value = input.value;
    const type = input.type || 'text';

    // Basic XSS prevention
    if (this.containsXSS(value)) {
      this.handleXSSAttempt(input, value);
      return false;
    }

    // SQL injection prevention
    if (this.containsSQLInjection(value)) {
      this.handleSQLInjectionAttempt(input, value);
      return false;
    }

    // Type-specific validation
    switch (type) {
      case 'email':
        return this.validateEmail(value);
      case 'url':
        return this.validateURL(value);
      case 'file':
        return this.validateFile(input);
      default:
        return this.validateText(value);
    }
  }

  containsXSS(value) {
    const xssPatterns = [
      /<script[^>]*>.*?<\/script>/gi,
      /javascript:/gi,
      /on\w+\s*=/gi,
      /<iframe[^>]*>.*?<\/iframe>/gi,
      /<object[^>]*>.*?<\/object>/gi,
      /<embed[^>]*>/gi,
      /<link[^>]*>/gi,
      /<meta[^>]*>/gi
    ];

    return xssPatterns.some(pattern => pattern.test(value));
  }

  containsSQLInjection(value) {
    const sqlPatterns = [
      /(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b)/gi,
      /('|(\\')|(')|(\\\\'))/gi,
      /(;|--|\/\*|\*\/)/gi,
      /(\bOR\b|\bAND\b).*(\b=\b|\b<\b|\b>\b)/gi
    ];

    return sqlPatterns.some(pattern => pattern.test(value));
  }

  handleXSSAttempt(input, value) {
    input.value = this.sanitizeXSS(value);
    this.showSecurityWarning(input, 'Potential XSS attempt blocked');
    this.reportSecurityEvent('xss_attempt', {
      input: input.name,
      value: value,
      sanitized: input.value
    });
  }

  handleSQLInjectionAttempt(input, value) {
    input.value = this.sanitizeSQL(value);
    this.showSecurityWarning(input, 'Potential SQL injection blocked');
    this.reportSecurityEvent('sql_injection_attempt', {
      input: input.name,
      value: value,
      sanitized: input.value
    });
  }

  sanitizeXSS(value) {
    return value
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#x27;')
      .replace(/\//g, '&#x2F;');
  }

  sanitizeSQL(value) {
    return value
      .replace(/['";]/g, '')
      .replace(/--/g, '')
      .replace(/\/\*.*?\*\//g, '')
      .replace(/\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b/gi, '');
  }

  showSecurityWarning(input, message) {
    // Create warning element
    const warning = document.createElement('div');
    warning.className = 'security-warning text-red-600 text-sm mt-1';
    warning.textContent = message;

    // Remove existing warnings
    const existingWarning = input.parentNode.querySelector('.security-warning');
    if (existingWarning) {
      existingWarning.remove();
    }

    // Add new warning
    input.parentNode.appendChild(warning);

    // Remove warning after 5 seconds
    setTimeout(() => {
      warning.remove();
    }, 5000);
  }

  validateEmail(email) {
    const emailPattern = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailPattern.test(email);
  }

  validateURL(url) {
    try {
      new URL(url);
      return true;
    } catch {
      return false;
    }
  }

  validateFile(fileInput) {
    const file = fileInput.files[0];
    if (!file) return true;

    // Check file size (10MB limit)
    if (file.size > 10 * 1024 * 1024) {
      this.showSecurityWarning(fileInput, 'File size exceeds 10MB limit');
      return false;
    }

    // Check file type
    const allowedTypes = [
      'text/csv',
      'application/json',
      'text/plain',
      'application/vnd.ms-excel',
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    ];

    if (!allowedTypes.includes(file.type)) {
      this.showSecurityWarning(fileInput, 'File type not allowed');
      return false;
    }

    return true;
  }

  validateText(value) {
    // Basic length validation
    if (value.length > 10000) {
      return false;
    }

    // Check for suspicious patterns
    const suspiciousPatterns = [
      /\.\./g, // Directory traversal
      /\x00/g, // Null bytes
      /[\x01-\x08\x0B\x0C\x0E-\x1F\x7F]/g // Control characters
    ];

    return !suspiciousPatterns.some(pattern => pattern.test(value));
  }

  handlePaste(event) {
    const pastedText = event.clipboardData.getData('text');

    // Check pasted content for security issues
    if (this.containsXSS(pastedText) || this.containsSQLInjection(pastedText)) {
      event.preventDefault();
      this.showSecurityWarning(event.target, 'Pasted content contains potentially malicious code');
      this.reportSecurityEvent('malicious_paste', {
        content: pastedText,
        target: event.target.name
      });
    }
  }

  setupSecurityMonitoring() {
    // Monitor for unusual activity
    this.monitorClickPatterns();
    this.monitorFormSubmissions();
    this.monitorDevToolsUsage();
  }

  monitorClickPatterns() {
    let clickCount = 0;
    let lastClickTime = 0;

    document.addEventListener('click', (event) => {
      const now = Date.now();
      const timeDiff = now - lastClickTime;

      if (timeDiff < 100) { // Suspiciously fast clicks
        clickCount++;
        if (clickCount > 10) {
          this.reportSecurityEvent('suspicious_clicking', {
            clickCount,
            timeDiff,
            target: event.target.tagName
          });
          clickCount = 0;
        }
      } else {
        clickCount = 0;
      }

      lastClickTime = now;
    });
  }

  monitorFormSubmissions() {
    document.addEventListener('submit', (event) => {
      const form = event.target;
      const formData = new FormData(form);

      // Check for automated submissions
      if (this.isAutomatedSubmission(form)) {
        this.reportSecurityEvent('automated_submission', {
          form: form.action,
          timestamp: Date.now()
        });
      }
    });
  }

  isAutomatedSubmission(form) {
    // Check for honeypot field
    const honeypot = form.querySelector('input[name="honeypot"]');
    if (honeypot && honeypot.value) {
      return true;
    }

    // Check submission speed
    const submitTime = form.dataset.submitTime;
    if (submitTime) {
      const timeToSubmit = Date.now() - parseInt(submitTime);
      if (timeToSubmit < 2000) { // Less than 2 seconds
        return true;
      }
    }

    return false;
  }

  monitorDevToolsUsage() {
    // Detect dev tools opening
    let devtools = {
      open: false,
      orientation: null
    };

    const threshold = 160;

    setInterval(() => {
      if (window.outerHeight - window.innerHeight > threshold ||
          window.outerWidth - window.innerWidth > threshold) {
        if (!devtools.open) {
          devtools.open = true;
          this.reportSecurityEvent('devtools_opened', {
            timestamp: Date.now(),
            userAgent: navigator.userAgent
          });
        }
      } else {
        devtools.open = false;
      }
    }, 500);
  }

  reportSecurityEvent(eventType, data) {
    const event = {
      type: eventType,
      timestamp: Date.now(),
      url: window.location.href,
      userAgent: navigator.userAgent,
      data
    };

    // Log locally
    console.warn('Security Event:', event);

    // Send to backend
    fetch('/api/security/events', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-CSRFToken': this.csrfToken
      },
      body: JSON.stringify(event)
    }).catch(error => {
      console.warn('Failed to report security event:', error);
    });
  }

  // Rate limiting
  checkRateLimit(action, limit = 10, window = 60000) {
    const now = Date.now();
    const key = `${action}_${Math.floor(now / window)}`;

    if (!this.rateLimiters.has(key)) {
      this.rateLimiters.set(key, 0);
    }

    const count = this.rateLimiters.get(key) + 1;
    this.rateLimiters.set(key, count);

    if (count > limit) {
      this.reportSecurityEvent('rate_limit_exceeded', {
        action,
        count,
        limit,
        window
      });
      return false;
    }

    return true;
  }

  // Session management integration
  getSessionManager() {
    return this.sessionManager;
  }
}

// Session Manager
class SessionManager {
  constructor() {
    this.sessionTimeout = 30 * 60 * 1000; // 30 minutes
    this.warningTimeout = 5 * 60 * 1000; // 5 minutes before expiry
    this.setupSessionMonitoring();
  }

  setupSessionMonitoring() {
    // Monitor user activity
    this.lastActivity = Date.now();

    ['mousedown', 'mousemove', 'keypress', 'scroll', 'touchstart'].forEach(event => {
      document.addEventListener(event, () => {
        this.lastActivity = Date.now();
      });
    });

    // Check session status every minute
    setInterval(() => {
      this.checkSessionStatus();
    }, 60000);
  }

  checkSessionStatus() {
    const now = Date.now();
    const inactiveTime = now - this.lastActivity;

    if (inactiveTime > this.sessionTimeout) {
      this.handleSessionExpiry();
    } else if (inactiveTime > this.sessionTimeout - this.warningTimeout) {
      this.showSessionWarning();
    }
  }

  handleSessionExpiry() {
    // Clear sensitive data
    sessionStorage.clear();

    // Redirect to login
    window.location.href = '/login?reason=session_expired';
  }

  showSessionWarning() {
    const warning = document.createElement('div');
    warning.className = 'session-warning fixed top-0 left-0 w-full bg-yellow-500 text-white p-4 z-50';
    warning.innerHTML = `
      <div class="container mx-auto flex justify-between items-center">
        <span>Your session will expire in 5 minutes due to inactivity.</span>
        <button onclick="this.parentElement.parentElement.remove()" class="bg-yellow-600 px-4 py-2 rounded">
          Dismiss
        </button>
      </div>
    `;

    document.body.appendChild(warning);

    // Auto-remove after 10 seconds
    setTimeout(() => {
      warning.remove();
    }, 10000);
  }

  extendSession() {
    this.lastActivity = Date.now();

    // Notify backend
    fetch('/api/session/extend', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-CSRFToken': securityManager.csrfToken
      }
    }).catch(error => {
      console.warn('Failed to extend session:', error);
    });
  }
}

// Global security manager instance
const securityManager = new SecurityManager();

// Export for use in other modules
export default securityManager;
