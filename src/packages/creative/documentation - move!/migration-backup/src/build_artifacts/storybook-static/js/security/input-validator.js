/**
 * Enhanced Input Validation and Sanitization Module
 * Provides real-time validation, sanitization, and security checks for user inputs
 */

class InputValidator {
    constructor(options = {}) {
        this.config = {
            enableRealTimeValidation: options.enableRealTimeValidation !== false,
            enableSanitization: options.enableSanitization !== false,
            strictMode: options.strictMode || false,
            maxLength: options.maxLength || 10000,
            allowedTags: options.allowedTags || [],
            blockedPatterns: options.blockedPatterns || [],
            validationDelay: options.validationDelay || 300,
            ...options
        };

        this.validators = {
            email: /^[^\s@]+@[^\s@]+\.[^\s@]+$/,
            url: /^https?:\/\/.+/,
            phone: /^[\+]?[1-9][\d]{0,15}$/,
            alphanumeric: /^[a-zA-Z0-9]+$/,
            numeric: /^[0-9]+$/,
            decimal: /^[0-9]+\.?[0-9]*$/,
            creditCard: /^[0-9]{13,19}$/,
            ssn: /^[0-9]{3}-?[0-9]{2}-?[0-9]{4}$/
        };

        this.securityPatterns = {
            xss: [
                /<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi,
                /javascript:/gi,
                /vbscript:/gi,
                /onload\s*=/gi,
                /onerror\s*=/gi,
                /onclick\s*=/gi,
                /onmouseover\s*=/gi,
                /<iframe/gi,
                /<object/gi,
                /<embed/gi,
                /<form/gi
            ],
            sqlInjection: [
                /(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b)/gi,
                /(;|\s)(DROP|DELETE)\s+(TABLE|DATABASE)/gi,
                /1\s*=\s*1/gi,
                /OR\s+1\s*=\s*1/gi,
                /\'\s*OR\s*\'/gi,
                /--/g,
                /\/\*/g
            ],
            commandInjection: [
                /(\||&|;|`|\$\(|\))/g,
                /(rm|del|format|fdisk)/gi,
                /(wget|curl|nc|telnet)/gi,
                /(eval|exec|system|shell_exec)/gi
            ],
            pathTraversal: [
                /\.\.\//g,
                /\.\.\\\\g,
                /%2e%2e%2f/gi,
                /%2e%2e%5c/gi,
                /\.\.%2f/gi,
                /\.\.%5c/gi
            ]
        };

        this.validationRules = new Map();
        this.sanitizedValues = new Map();

        this.init();
    }

    init() {
        this.setupFormValidation();
        this.setupRealTimeValidation();
        this.setupFileUploadSecurity();
        this.setupGlobalValidation();
        this.addValidationStyles();
    }

    /**
     * Setup form validation on submit
     */
    setupFormValidation() {
        document.addEventListener('submit', async (event) => {
            const form = event.target;
            if (!form.hasAttribute('data-validate')) return;

            event.preventDefault();

            const isValid = await this.validateForm(form);
            if (isValid) {
                // Sanitize form data before submission
                this.sanitizeFormData(form);
                form.submit();
            }
        });
    }

    /**
     * Setup real-time validation for inputs
     */
    setupRealTimeValidation() {
        if (!this.config.enableRealTimeValidation) return;

        // Debounced validation function
        const debouncedValidate = this.debounce((input) => {
            this.validateInput(input);
        }, this.config.validationDelay);

        // Setup input event listeners
        document.addEventListener('input', (event) => {
            const input = event.target;
            if (this.shouldValidateInput(input)) {
                debouncedValidate(input);
            }
        });

        // Setup blur validation for more thorough checks
        document.addEventListener('blur', (event) => {
            const input = event.target;
            if (this.shouldValidateInput(input)) {
                this.validateInput(input, true);
            }
        }, true);
    }

    /**
     * Check if input should be validated
     */
    shouldValidateInput(input) {
        return input.type !== 'hidden' &&
               input.type !== 'submit' &&
               input.type !== 'button' &&
               !input.disabled &&
               (input.hasAttribute('data-validate') ||
                input.closest('form[data-validate]'));
    }

    /**
     * Validate entire form
     */
    async validateForm(form) {
        const inputs = form.querySelectorAll('input, textarea, select');
        let isValid = true;
        const errors = [];

        for (const input of inputs) {
            if (this.shouldValidateInput(input)) {
                const inputValid = await this.validateInput(input, true);
                if (!inputValid) {
                    isValid = false;
                    errors.push({
                        field: input.name || input.id,
                        message: this.getValidationMessage(input)
                    });
                }
            }
        }

        // Show form-level errors if any
        if (!isValid) {
            this.showFormErrors(form, errors);
        }

        return isValid;
    }

    /**
     * Validate individual input
     */
    async validateInput(input, showErrors = false) {
        const value = input.value;
        const validationResult = {
            isValid: true,
            errors: [],
            warnings: []
        };

        // Skip validation for empty non-required fields
        if (!value && !input.required) {
            this.clearValidationUI(input);
            return true;
        }

        // Required field validation
        if (input.required && !value.trim()) {
            validationResult.isValid = false;
            validationResult.errors.push('This field is required');
        }

        // Length validation
        if (value.length > this.config.maxLength) {
            validationResult.isValid = false;
            validationResult.errors.push(`Maximum length is ${this.config.maxLength} characters`);
        }

        // Type-specific validation
        await this.validateByType(input, value, validationResult);

        // Security pattern validation
        this.validateSecurity(value, validationResult);

        // Custom validation rules
        this.validateCustomRules(input, value, validationResult);

        // Update UI
        if (showErrors || validationResult.isValid) {
            this.updateValidationUI(input, validationResult);
        }

        return validationResult.isValid;
    }

    /**
     * Validate input by type
     */
    async validateByType(input, value, validationResult) {
        const type = input.type || input.getAttribute('data-type');

        switch (type) {
            case 'email':
                if (!this.validators.email.test(value)) {
                    validationResult.isValid = false;
                    validationResult.errors.push('Please enter a valid email address');
                }
                break;

            case 'url':
                if (!this.validators.url.test(value)) {
                    validationResult.isValid = false;
                    validationResult.errors.push('Please enter a valid URL');
                }
                break;

            case 'tel':
                if (!this.validators.phone.test(value)) {
                    validationResult.isValid = false;
                    validationResult.errors.push('Please enter a valid phone number');
                }
                break;

            case 'number':
                if (!this.validators.numeric.test(value)) {
                    validationResult.isValid = false;
                    validationResult.errors.push('Please enter a valid number');
                }
                break;

            case 'password':
                this.validatePassword(value, validationResult);
                break;

            case 'file':
                await this.validateFileInput(input, validationResult);
                break;
        }

        // HTML5 pattern validation
        if (input.pattern && !new RegExp(input.pattern).test(value)) {
            validationResult.isValid = false;
            validationResult.errors.push(input.title || 'Please match the required format');
        }

        // Min/max length validation
        if (input.minLength && value.length < input.minLength) {
            validationResult.isValid = false;
            validationResult.errors.push(`Minimum length is ${input.minLength} characters`);
        }

        if (input.maxLength && value.length > input.maxLength) {
            validationResult.isValid = false;
            validationResult.errors.push(`Maximum length is ${input.maxLength} characters`);
        }
    }

    /**
     * Validate password strength
     */
    validatePassword(password, validationResult) {
        const requirements = {
            minLength: 8,
            requireUppercase: true,
            requireLowercase: true,
            requireNumbers: true,
            requireSymbols: false
        };

        if (password.length < requirements.minLength) {
            validationResult.isValid = false;
            validationResult.errors.push(`Password must be at least ${requirements.minLength} characters`);
        }

        if (requirements.requireUppercase && !/[A-Z]/.test(password)) {
            validationResult.isValid = false;
            validationResult.errors.push('Password must contain at least one uppercase letter');
        }

        if (requirements.requireLowercase && !/[a-z]/.test(password)) {
            validationResult.isValid = false;
            validationResult.errors.push('Password must contain at least one lowercase letter');
        }

        if (requirements.requireNumbers && !/\d/.test(password)) {
            validationResult.isValid = false;
            validationResult.errors.push('Password must contain at least one number');
        }

        if (requirements.requireSymbols && !/[!@#$%^&*]/.test(password)) {
            validationResult.warnings.push('Consider adding special characters for stronger security');
        }

        // Check for common weak passwords
        const weakPasswords = ['password', '123456', 'qwerty', 'admin', 'letmein'];
        if (weakPasswords.includes(password.toLowerCase())) {
            validationResult.isValid = false;
            validationResult.errors.push('This password is too common and weak');
        }
    }

    /**
     * Validate file input
     */
    async validateFileInput(input, validationResult) {
        const files = input.files;
        if (!files || files.length === 0) return;

        const allowedTypes = input.getAttribute('data-allowed-types');
        const maxSize = parseInt(input.getAttribute('data-max-size')) || 10 * 1024 * 1024; // 10MB default
        const maxFiles = parseInt(input.getAttribute('data-max-files')) || 10;

        if (files.length > maxFiles) {
            validationResult.isValid = false;
            validationResult.errors.push(`Maximum ${maxFiles} files allowed`);
            return;
        }

        for (const file of files) {
            // Size validation
            if (file.size > maxSize) {
                validationResult.isValid = false;
                validationResult.errors.push(`File "${file.name}" is too large (max ${this.formatFileSize(maxSize)})`);
                continue;
            }

            // Type validation
            if (allowedTypes) {
                const allowedTypesArray = allowedTypes.split(',').map(t => t.trim());
                if (!allowedTypesArray.includes(file.type)) {
                    validationResult.isValid = false;
                    validationResult.errors.push(`File type "${file.type}" not allowed for "${file.name}"`);
                    continue;
                }
            }

            // Security scan
            await this.scanFileForSecurity(file, validationResult);
        }
    }

    /**
     * Scan file for security issues
     */
    async scanFileForSecurity(file, validationResult) {
        // Check file extension against MIME type
        const extension = file.name.split('.').pop().toLowerCase();
        const expectedMimeTypes = this.getExpectedMimeTypes(extension);

        if (expectedMimeTypes.length > 0 && !expectedMimeTypes.includes(file.type)) {
            validationResult.warnings.push(`File extension and type mismatch for "${file.name}"`);
        }

        // Check for dangerous extensions
        const dangerousExtensions = ['exe', 'bat', 'cmd', 'com', 'pif', 'scr', 'vbs', 'js', 'jar'];
        if (dangerousExtensions.includes(extension)) {
            validationResult.isValid = false;
            validationResult.errors.push(`File type "${extension}" is not allowed for security reasons`);
        }

        // Check file size against type expectations
        if (file.type.startsWith('image/') && file.size > 50 * 1024 * 1024) { // 50MB for images
            validationResult.warnings.push(`Image file "${file.name}" is unusually large`);
        }
    }

    /**
     * Get expected MIME types for file extension
     */
    getExpectedMimeTypes(extension) {
        const mimeMap = {
            'jpg': ['image/jpeg'],
            'jpeg': ['image/jpeg'],
            'png': ['image/png'],
            'gif': ['image/gif'],
            'pdf': ['application/pdf'],
            'doc': ['application/msword'],
            'docx': ['application/vnd.openxmlformats-officedocument.wordprocessingml.document'],
            'xls': ['application/vnd.ms-excel'],
            'xlsx': ['application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'],
            'txt': ['text/plain'],
            'csv': ['text/csv', 'application/csv'],
            'json': ['application/json'],
            'xml': ['application/xml', 'text/xml']
        };

        return mimeMap[extension] || [];
    }

    /**
     * Validate against security patterns
     */
    validateSecurity(value, validationResult) {
        // XSS pattern detection
        for (const pattern of this.securityPatterns.xss) {
            if (pattern.test(value)) {
                if (this.config.strictMode) {
                    validationResult.isValid = false;
                    validationResult.errors.push('Input contains potentially dangerous content');
                } else {
                    validationResult.warnings.push('Input will be sanitized for security');
                }
                break;
            }
        }

        // SQL injection pattern detection
        for (const pattern of this.securityPatterns.sqlInjection) {
            if (pattern.test(value)) {
                if (this.config.strictMode) {
                    validationResult.isValid = false;
                    validationResult.errors.push('Input contains SQL-like patterns');
                } else {
                    validationResult.warnings.push('Input will be sanitized for security');
                }
                break;
            }
        }

        // Command injection detection
        for (const pattern of this.securityPatterns.commandInjection) {
            if (pattern.test(value)) {
                validationResult.isValid = false;
                validationResult.errors.push('Input contains potentially dangerous characters');
                break;
            }
        }

        // Path traversal detection
        for (const pattern of this.securityPatterns.pathTraversal) {
            if (pattern.test(value)) {
                validationResult.isValid = false;
                validationResult.errors.push('Input contains invalid path characters');
                break;
            }
        }
    }

    /**
     * Validate custom rules
     */
    validateCustomRules(input, value, validationResult) {
        const customRules = this.validationRules.get(input.name || input.id);
        if (!customRules) return;

        for (const rule of customRules) {
            const result = rule.validator(value, input);
            if (!result.isValid) {
                validationResult.isValid = false;
                validationResult.errors.push(result.message);
            }
        }
    }

    /**
     * Sanitize form data before submission
     */
    sanitizeFormData(form) {
        if (!this.config.enableSanitization) return;

        const inputs = form.querySelectorAll('input, textarea');
        for (const input of inputs) {
            if (input.type !== 'file' && input.type !== 'password') {
                const sanitized = this.sanitizeValue(input.value);
                this.sanitizedValues.set(input.name || input.id, {
                    original: input.value,
                    sanitized: sanitized
                });
                input.value = sanitized;
            }
        }
    }

    /**
     * Sanitize individual value
     */
    sanitizeValue(value) {
        if (typeof value !== 'string') return value;

        // HTML entity encoding
        let sanitized = value
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#x27;')
            .replace(/\//g, '&#x2F;');

        // Remove dangerous patterns
        for (const patterns of Object.values(this.securityPatterns)) {
            for (const pattern of patterns) {
                sanitized = sanitized.replace(pattern, '');
            }
        }

        // Trim whitespace
        sanitized = sanitized.trim();

        return sanitized;
    }

    /**
     * Update validation UI
     */
    updateValidationUI(input, validationResult) {
        this.clearValidationUI(input);

        const container = this.getOrCreateValidationContainer(input);

        if (!validationResult.isValid) {
            input.classList.add('validation-error');
            container.classList.add('has-errors');

            for (const error of validationResult.errors) {
                const errorElement = document.createElement('div');
                errorElement.className = 'validation-message error';
                errorElement.textContent = error;
                container.appendChild(errorElement);
            }
        } else {
            input.classList.add('validation-success');
        }

        // Show warnings
        for (const warning of validationResult.warnings) {
            const warningElement = document.createElement('div');
            warningElement.className = 'validation-message warning';
            warningElement.textContent = warning;
            container.appendChild(warningElement);
        }
    }

    /**
     * Clear validation UI
     */
    clearValidationUI(input) {
        input.classList.remove('validation-error', 'validation-success');

        const container = this.getValidationContainer(input);
        if (container) {
            container.classList.remove('has-errors');
            container.querySelectorAll('.validation-message').forEach(el => el.remove());
        }
    }

    /**
     * Get or create validation container
     */
    getOrCreateValidationContainer(input) {
        let container = this.getValidationContainer(input);

        if (!container) {
            container = document.createElement('div');
            container.className = 'validation-container';
            input.parentNode.insertBefore(container, input.nextSibling);
        }

        return container;
    }

    /**
     * Get validation container
     */
    getValidationContainer(input) {
        return input.parentNode.querySelector('.validation-container');
    }

    /**
     * Show form-level errors
     */
    showFormErrors(form, errors) {
        let errorContainer = form.querySelector('.form-errors');

        if (!errorContainer) {
            errorContainer = document.createElement('div');
            errorContainer.className = 'form-errors';
            form.insertBefore(errorContainer, form.firstChild);
        }

        errorContainer.innerHTML = `
            <h4>Please fix the following errors:</h4>
            <ul>
                ${errors.map(error => `<li>${error.message}</li>`).join('')}
            </ul>
        `;
    }

    /**
     * Add validation styles
     */
    addValidationStyles() {
        if (document.querySelector('#input-validation-styles')) return;

        const styles = `
            .validation-error {
                border-color: #dc2626 !important;
                box-shadow: 0 0 0 3px rgba(220, 38, 38, 0.1) !important;
            }

            .validation-success {
                border-color: #16a34a !important;
                box-shadow: 0 0 0 3px rgba(22, 163, 74, 0.1) !important;
            }

            .validation-container {
                margin-top: 4px;
            }

            .validation-message {
                font-size: 0.875rem;
                margin-top: 2px;
                padding: 4px 8px;
                border-radius: 4px;
            }

            .validation-message.error {
                color: #dc2626;
                background: #fef2f2;
                border-left: 3px solid #dc2626;
            }

            .validation-message.warning {
                color: #d97706;
                background: #fffbeb;
                border-left: 3px solid #f59e0b;
            }

            .form-errors {
                background: #fef2f2;
                border: 1px solid #fecaca;
                border-radius: 6px;
                padding: 16px;
                margin-bottom: 16px;
            }

            .form-errors h4 {
                color: #dc2626;
                margin: 0 0 8px;
                font-size: 1rem;
            }

            .form-errors ul {
                margin: 0;
                padding-left: 20px;
            }

            .form-errors li {
                color: #dc2626;
                margin-bottom: 4px;
            }
        `;

        const styleSheet = document.createElement('style');
        styleSheet.id = 'input-validation-styles';
        styleSheet.textContent = styles;
        document.head.appendChild(styleSheet);
    }

    /**
     * Setup file upload security
     */
    setupFileUploadSecurity() {
        document.addEventListener('change', (event) => {
            if (event.target.type === 'file') {
                this.validateInput(event.target, true);
            }
        });
    }

    /**
     * Setup global validation
     */
    setupGlobalValidation() {
        // Prevent form submission with Enter on invalid inputs
        document.addEventListener('keydown', (event) => {
            if (event.key === 'Enter' && event.target.matches('input')) {
                const form = event.target.closest('form');
                if (form && form.hasAttribute('data-validate')) {
                    if (event.target.classList.contains('validation-error')) {
                        event.preventDefault();
                    }
                }
            }
        });
    }

    /**
     * Add custom validation rule
     */
    addValidationRule(fieldName, validator) {
        if (!this.validationRules.has(fieldName)) {
            this.validationRules.set(fieldName, []);
        }
        this.validationRules.get(fieldName).push(validator);
    }

    /**
     * Remove validation rule
     */
    removeValidationRule(fieldName) {
        this.validationRules.delete(fieldName);
    }

    /**
     * Get validation message for input
     */
    getValidationMessage(input) {
        const container = this.getValidationContainer(input);
        if (container) {
            const errorMessage = container.querySelector('.validation-message.error');
            return errorMessage ? errorMessage.textContent : 'Invalid input';
        }
        return 'Invalid input';
    }

    /**
     * Format file size
     */
    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    /**
     * Debounce function
     */
    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }

    /**
     * Validate specific field
     */
    async validateField(fieldName) {
        const input = document.querySelector(`[name="${fieldName}"], #${fieldName}`);
        if (input) {
            return await this.validateInput(input, true);
        }
        return false;
    }

    /**
     * Get sanitized value
     */
    getSanitizedValue(fieldName) {
        const data = this.sanitizedValues.get(fieldName);
        return data ? data.sanitized : null;
    }

    /**
     * Get original value before sanitization
     */
    getOriginalValue(fieldName) {
        const data = this.sanitizedValues.get(fieldName);
        return data ? data.original : null;
    }
}

// Initialize input validator when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.inputValidator = new InputValidator();
});

// Export for use in modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = InputValidator;
}
