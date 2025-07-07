/**
 * Advanced Multi-Step Form Component
 * 
 * Comprehensive form system with validation, file upload, progress tracking,
 * and dynamic field generation for anomaly detection workflows
 */

export class MultiStepForm {
    constructor(container, options = {}) {
        this.container = typeof container === 'string' ? document.querySelector(container) : container;
        this.options = {
            showProgress: true,
            showStepNumbers: true,
            allowStepNavigation: true,
            validateOnStepChange: true,
            saveProgress: true,
            progressKey: 'multi-step-form-progress',
            submitUrl: null,
            submitMethod: 'POST',
            fileUploadUrl: '/api/upload',
            maxFileSize: 10 * 1024 * 1024, // 10MB
            allowedFileTypes: ['.csv', '.json', '.xlsx', '.parquet'],
            ...options
        };
        
        this.steps = [];
        this.currentStep = 0;
        this.formData = {};
        this.validationErrors = {};
        this.uploadedFiles = new Map();
        this.isSubmitting = false;
        
        this.validators = new Map();
        this.fieldComponents = new Map();
        this.conditionalFields = new Map();
        
        this.init();
    }
    
    init() {
        this.setupContainer();
        this.registerDefaultValidators();
        this.registerDefaultFieldTypes();
        this.loadSavedProgress();
        this.bindEvents();
    }
    
    setupContainer() {
        this.container.classList.add('multi-step-form');
        this.container.innerHTML = '';
        
        // Create form structure
        this.form = document.createElement('form');
        this.form.className = 'form-container';
        this.form.setAttribute('novalidate', '');
        
        // Progress indicator
        if (this.options.showProgress) {
            this.progressContainer = document.createElement('div');
            this.progressContainer.className = 'form-progress';
            this.container.appendChild(this.progressContainer);
        }
        
        // Steps container
        this.stepsContainer = document.createElement('div');
        this.stepsContainer.className = 'form-steps';
        this.form.appendChild(this.stepsContainer);
        
        // Navigation
        this.navigationContainer = document.createElement('div');
        this.navigationContainer.className = 'form-navigation';
        this.form.appendChild(this.navigationContainer);
        
        this.container.appendChild(this.form);
    }
    
    addStep(stepConfig) {
        const step = {
            id: stepConfig.id || `step_${this.steps.length}`,
            title: stepConfig.title || `Step ${this.steps.length + 1}`,
            description: stepConfig.description || '',
            fields: stepConfig.fields || [],
            validation: stepConfig.validation || null,
            onEnter: stepConfig.onEnter || null,
            onExit: stepConfig.onExit || null,
            conditional: stepConfig.conditional || null,
            ...stepConfig
        };
        
        this.steps.push(step);
        this.renderProgress();
        
        return step;
    }
    
    renderProgress() {
        if (!this.options.showProgress || !this.progressContainer) return;
        
        this.progressContainer.innerHTML = '';
        
        const progressBar = document.createElement('div');
        progressBar.className = 'progress-bar';
        
        const progressFill = document.createElement('div');
        progressFill.className = 'progress-fill';
        progressFill.style.width = `${((this.currentStep + 1) / this.steps.length) * 100}%`;
        
        progressBar.appendChild(progressFill);
        this.progressContainer.appendChild(progressBar);
        
        // Step indicators
        if (this.options.showStepNumbers) {
            const stepsIndicator = document.createElement('div');
            stepsIndicator.className = 'steps-indicator';
            
            this.steps.forEach((step, index) => {
                const stepIndicator = document.createElement('div');
                stepIndicator.className = 'step-indicator';
                stepIndicator.setAttribute('data-step', index);
                
                if (index < this.currentStep) {
                    stepIndicator.classList.add('completed');
                } else if (index === this.currentStep) {
                    stepIndicator.classList.add('active');
                }
                
                if (this.options.allowStepNavigation && index <= this.currentStep) {
                    stepIndicator.classList.add('clickable');
                    stepIndicator.addEventListener('click', () => this.goToStep(index));
                }
                
                const stepNumber = document.createElement('span');
                stepNumber.className = 'step-number';
                stepNumber.textContent = index + 1;
                
                const stepTitle = document.createElement('span');
                stepTitle.className = 'step-title';
                stepTitle.textContent = step.title;
                
                stepIndicator.appendChild(stepNumber);
                stepIndicator.appendChild(stepTitle);
                stepsIndicator.appendChild(stepIndicator);
            });
            
            this.progressContainer.appendChild(stepsIndicator);
        }
    }
    
    renderCurrentStep() {
        const step = this.steps[this.currentStep];
        if (!step) return;
        
        // Check conditional display
        if (step.conditional && !this.evaluateCondition(step.conditional)) {
            this.nextStep();
            return;
        }
        
        // Clear previous step
        this.stepsContainer.innerHTML = '';
        
        // Create step container
        const stepElement = document.createElement('div');
        stepElement.className = 'form-step active';
        stepElement.setAttribute('data-step-id', step.id);
        
        // Step header
        const stepHeader = document.createElement('div');
        stepHeader.className = 'step-header';
        
        const stepTitle = document.createElement('h2');
        stepTitle.className = 'step-title';
        stepTitle.textContent = step.title;
        stepHeader.appendChild(stepTitle);
        
        if (step.description) {
            const stepDescription = document.createElement('p');
            stepDescription.className = 'step-description';
            stepDescription.textContent = step.description;
            stepHeader.appendChild(stepDescription);
        }
        
        stepElement.appendChild(stepHeader);
        
        // Step content
        const stepContent = document.createElement('div');
        stepContent.className = 'step-content';
        
        // Render fields
        step.fields.forEach(fieldConfig => {
            const fieldElement = this.renderField(fieldConfig);
            stepContent.appendChild(fieldElement);
        });
        
        stepElement.appendChild(stepContent);
        this.stepsContainer.appendChild(stepElement);
        
        // Update navigation
        this.renderNavigation();
        
        // Call step enter callback
        if (step.onEnter) {
            step.onEnter(this.formData, this);
        }
        
        // Update progress
        this.renderProgress();
    }
    
    renderField(fieldConfig) {
        const field = {
            type: 'text',
            name: fieldConfig.name || '',
            label: fieldConfig.label || '',
            placeholder: fieldConfig.placeholder || '',
            required: fieldConfig.required || false,
            validation: fieldConfig.validation || [],
            options: fieldConfig.options || [],
            defaultValue: fieldConfig.defaultValue,
            conditional: fieldConfig.conditional || null,
            ...fieldConfig
        };
        
        // Check conditional display
        if (field.conditional && !this.evaluateCondition(field.conditional)) {
            const hiddenDiv = document.createElement('div');
            hiddenDiv.style.display = 'none';
            hiddenDiv.setAttribute('data-field-name', field.name);
            return hiddenDiv;
        }
        
        // Get field component
        const fieldComponent = this.fieldComponents.get(field.type) || this.fieldComponents.get('text');
        return fieldComponent(field, this);
    }
    
    renderNavigation() {
        this.navigationContainer.innerHTML = '';
        
        const navContainer = document.createElement('div');
        navContainer.className = 'nav-buttons';
        
        // Previous button
        if (this.currentStep > 0) {
            const prevButton = document.createElement('button');
            prevButton.type = 'button';
            prevButton.className = 'btn btn-secondary prev-btn';
            prevButton.textContent = 'Previous';
            prevButton.addEventListener('click', () => this.previousStep());
            navContainer.appendChild(prevButton);
        }
        
        // Next/Submit button
        const nextButton = document.createElement('button');
        nextButton.type = 'button';
        nextButton.className = 'btn btn-primary next-btn';
        
        if (this.currentStep === this.steps.length - 1) {
            nextButton.textContent = this.isSubmitting ? 'Submitting...' : 'Submit';
            nextButton.disabled = this.isSubmitting;
            nextButton.addEventListener('click', () => this.submitForm());
        } else {
            nextButton.textContent = 'Next';
            nextButton.addEventListener('click', () => this.nextStep());
        }
        
        navContainer.appendChild(nextButton);
        
        // Cancel button
        const cancelButton = document.createElement('button');
        cancelButton.type = 'button';
        cancelButton.className = 'btn btn-ghost cancel-btn';
        cancelButton.textContent = 'Cancel';
        cancelButton.addEventListener('click', () => this.cancel());
        navContainer.appendChild(cancelButton);
        
        this.navigationContainer.appendChild(navContainer);
    }
    
    registerDefaultFieldTypes() {
        // Text input
        this.fieldComponents.set('text', (field, form) => {
            const container = document.createElement('div');
            container.className = 'field-group';
            container.setAttribute('data-field-name', field.name);
            
            const label = document.createElement('label');
            label.className = 'field-label';
            label.textContent = field.label;
            if (field.required) {
                label.innerHTML += ' <span class="required">*</span>';
            }
            
            const input = document.createElement('input');
            input.type = field.subtype || 'text';
            input.name = field.name;
            input.className = 'field-input';
            input.placeholder = field.placeholder;
            input.required = field.required;
            input.value = form.formData[field.name] || field.defaultValue || '';
            
            if (field.pattern) {
                input.pattern = field.pattern;
            }
            
            input.addEventListener('input', (e) => {
                form.updateFieldValue(field.name, e.target.value);
            });
            
            input.addEventListener('blur', () => {
                form.validateField(field.name);
            });
            
            container.appendChild(label);
            container.appendChild(input);
            
            if (field.help) {
                const help = document.createElement('div');
                help.className = 'field-help';
                help.textContent = field.help;
                container.appendChild(help);
            }
            
            return container;
        });
        
        // Select dropdown
        this.fieldComponents.set('select', (field, form) => {
            const container = document.createElement('div');
            container.className = 'field-group';
            container.setAttribute('data-field-name', field.name);
            
            const label = document.createElement('label');
            label.className = 'field-label';
            label.textContent = field.label;
            if (field.required) {
                label.innerHTML += ' <span class="required">*</span>';
            }
            
            const select = document.createElement('select');
            select.name = field.name;
            select.className = 'field-select';
            select.required = field.required;
            
            // Add default option
            if (field.placeholder) {
                const defaultOption = document.createElement('option');
                defaultOption.value = '';
                defaultOption.textContent = field.placeholder;
                defaultOption.disabled = true;
                defaultOption.selected = !form.formData[field.name];
                select.appendChild(defaultOption);
            }
            
            // Add options
            field.options.forEach(option => {
                const optionElement = document.createElement('option');
                
                if (typeof option === 'string') {
                    optionElement.value = option;
                    optionElement.textContent = option;
                } else {
                    optionElement.value = option.value;
                    optionElement.textContent = option.label;
                }
                
                if (form.formData[field.name] === optionElement.value) {
                    optionElement.selected = true;
                }
                
                select.appendChild(optionElement);
            });
            
            select.addEventListener('change', (e) => {
                form.updateFieldValue(field.name, e.target.value);
            });
            
            container.appendChild(label);
            container.appendChild(select);
            
            return container;
        });
        
        // Textarea
        this.fieldComponents.set('textarea', (field, form) => {
            const container = document.createElement('div');
            container.className = 'field-group';
            container.setAttribute('data-field-name', field.name);
            
            const label = document.createElement('label');
            label.className = 'field-label';
            label.textContent = field.label;
            if (field.required) {
                label.innerHTML += ' <span class="required">*</span>';
            }
            
            const textarea = document.createElement('textarea');
            textarea.name = field.name;
            textarea.className = 'field-textarea';
            textarea.placeholder = field.placeholder;
            textarea.required = field.required;
            textarea.rows = field.rows || 4;
            textarea.value = form.formData[field.name] || field.defaultValue || '';
            
            textarea.addEventListener('input', (e) => {
                form.updateFieldValue(field.name, e.target.value);
            });
            
            container.appendChild(label);
            container.appendChild(textarea);
            
            return container;
        });
        
        // Checkbox
        this.fieldComponents.set('checkbox', (field, form) => {
            const container = document.createElement('div');
            container.className = 'field-group field-checkbox';
            container.setAttribute('data-field-name', field.name);
            
            const checkbox = document.createElement('input');
            checkbox.type = 'checkbox';
            checkbox.name = field.name;
            checkbox.className = 'field-input-checkbox';
            checkbox.required = field.required;
            checkbox.checked = form.formData[field.name] || field.defaultValue || false;
            
            const label = document.createElement('label');
            label.className = 'field-label-checkbox';
            label.appendChild(checkbox);
            
            const labelText = document.createElement('span');
            labelText.textContent = field.label;
            if (field.required) {
                labelText.innerHTML += ' <span class="required">*</span>';
            }
            label.appendChild(labelText);
            
            checkbox.addEventListener('change', (e) => {
                form.updateFieldValue(field.name, e.target.checked);
            });
            
            container.appendChild(label);
            
            return container;
        });
        
        // File upload
        this.fieldComponents.set('file', (field, form) => {
            const container = document.createElement('div');
            container.className = 'field-group field-file';
            container.setAttribute('data-field-name', field.name);
            
            const label = document.createElement('label');
            label.className = 'field-label';
            label.textContent = field.label;
            if (field.required) {
                label.innerHTML += ' <span class="required">*</span>';
            }
            
            // File input
            const fileInput = document.createElement('input');
            fileInput.type = 'file';
            fileInput.name = field.name;
            fileInput.className = 'field-file-input';
            fileInput.required = field.required;
            fileInput.style.display = 'none';
            
            if (field.multiple) {
                fileInput.multiple = true;
            }
            
            if (field.accept) {
                fileInput.accept = field.accept;
            } else {
                fileInput.accept = form.options.allowedFileTypes.join(',');
            }
            
            // Custom file upload button
            const uploadButton = document.createElement('button');
            uploadButton.type = 'button';
            uploadButton.className = 'btn btn-outline file-upload-btn';
            uploadButton.innerHTML = `
                <svg class="upload-icon" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                    <polyline points="7,10 12,15 17,10"></polyline>
                    <line x1="12" y1="15" x2="12" y2="3"></line>
                </svg>
                Choose File${field.multiple ? 's' : ''}
            `;
            
            uploadButton.addEventListener('click', () => fileInput.click());
            
            // File preview area
            const previewArea = document.createElement('div');
            previewArea.className = 'file-preview-area';
            
            fileInput.addEventListener('change', (e) => {
                form.handleFileUpload(field.name, e.target.files);
            });
            
            container.appendChild(label);
            container.appendChild(fileInput);
            container.appendChild(uploadButton);
            container.appendChild(previewArea);
            
            if (field.help) {
                const help = document.createElement('div');
                help.className = 'field-help';
                help.textContent = field.help;
                container.appendChild(help);
            }
            
            return container;
        });
        
        // Range slider
        this.fieldComponents.set('range', (field, form) => {
            const container = document.createElement('div');
            container.className = 'field-group field-range';
            container.setAttribute('data-field-name', field.name);
            
            const label = document.createElement('label');
            label.className = 'field-label';
            label.textContent = field.label;
            if (field.required) {
                label.innerHTML += ' <span class="required">*</span>';
            }
            
            const rangeContainer = document.createElement('div');
            rangeContainer.className = 'range-container';
            
            const range = document.createElement('input');
            range.type = 'range';
            range.name = field.name;
            range.className = 'field-range-input';
            range.min = field.min || 0;
            range.max = field.max || 100;
            range.step = field.step || 1;
            range.value = form.formData[field.name] || field.defaultValue || field.min || 0;
            
            const valueDisplay = document.createElement('span');
            valueDisplay.className = 'range-value';
            valueDisplay.textContent = range.value;
            
            range.addEventListener('input', (e) => {
                valueDisplay.textContent = e.target.value;
                form.updateFieldValue(field.name, parseFloat(e.target.value));
            });
            
            rangeContainer.appendChild(range);
            rangeContainer.appendChild(valueDisplay);
            
            container.appendChild(label);
            container.appendChild(rangeContainer);
            
            return container;
        });
    }
    
    registerDefaultValidators() {
        this.validators.set('required', (value, field) => {
            if (field.required) {
                if (value === null || value === undefined || value === '') {
                    return `${field.label} is required`;
                }
            }
            return null;
        });
        
        this.validators.set('email', (value, field) => {
            if (value && field.type === 'email') {
                const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
                if (!emailRegex.test(value)) {
                    return 'Please enter a valid email address';
                }
            }
            return null;
        });
        
        this.validators.set('minLength', (value, field) => {
            if (value && field.minLength && value.length < field.minLength) {
                return `${field.label} must be at least ${field.minLength} characters`;
            }
            return null;
        });
        
        this.validators.set('maxLength', (value, field) => {
            if (value && field.maxLength && value.length > field.maxLength) {
                return `${field.label} must be no more than ${field.maxLength} characters`;
            }
            return null;
        });
        
        this.validators.set('pattern', (value, field) => {
            if (value && field.pattern) {
                const regex = new RegExp(field.pattern);
                if (!regex.test(value)) {
                    return field.patternMessage || `${field.label} format is invalid`;
                }
            }
            return null;
        });
        
        this.validators.set('fileSize', (value, field) => {
            if (field.type === 'file' && value) {
                const files = Array.isArray(value) ? value : [value];
                for (const file of files) {
                    if (file.size > this.options.maxFileSize) {
                        return `File size must be less than ${this.formatFileSize(this.options.maxFileSize)}`;
                    }
                }
            }
            return null;
        });
    }
    
    updateFieldValue(fieldName, value) {
        this.formData[fieldName] = value;
        this.clearFieldError(fieldName);
        this.saveProgress();
        
        // Check for conditional fields that depend on this field
        this.updateConditionalFields();
    }
    
    validateField(fieldName) {
        const step = this.steps[this.currentStep];
        const field = step.fields.find(f => f.name === fieldName);
        
        if (!field) return true;
        
        const value = this.formData[fieldName];
        const errors = [];
        
        // Run built-in validators
        for (const [validatorName, validator] of this.validators) {
            const error = validator(value, field);
            if (error) {
                errors.push(error);
            }
        }
        
        // Run custom field validation
        if (field.validation) {
            if (typeof field.validation === 'function') {
                const result = field.validation(value, this.formData);
                if (result && typeof result === 'string') {
                    errors.push(result);
                }
            } else if (Array.isArray(field.validation)) {
                field.validation.forEach(validator => {
                    if (typeof validator === 'function') {
                        const result = validator(value, this.formData);
                        if (result && typeof result === 'string') {
                            errors.push(result);
                        }
                    }
                });
            }
        }
        
        if (errors.length > 0) {
            this.setFieldError(fieldName, errors[0]);
            return false;
        }
        
        this.clearFieldError(fieldName);
        return true;
    }
    
    validateStep(stepIndex = this.currentStep) {
        const step = this.steps[stepIndex];
        if (!step) return true;
        
        let isValid = true;
        
        // Validate individual fields
        step.fields.forEach(field => {
            if (!this.validateField(field.name)) {
                isValid = false;
            }
        });
        
        // Run step-level validation
        if (step.validation && typeof step.validation === 'function') {
            const result = step.validation(this.formData);
            if (result !== true) {
                isValid = false;
                if (typeof result === 'string') {
                    this.showStepError(result);
                }
            }
        }
        
        return isValid;
    }
    
    setFieldError(fieldName, message) {
        this.validationErrors[fieldName] = message;
        
        const fieldContainer = this.container.querySelector(`[data-field-name="${fieldName}"]`);
        if (fieldContainer) {
            fieldContainer.classList.add('field-error');
            
            // Remove existing error message
            const existingError = fieldContainer.querySelector('.field-error-message');
            if (existingError) {
                existingError.remove();
            }
            
            // Add error message
            const errorElement = document.createElement('div');
            errorElement.className = 'field-error-message';
            errorElement.textContent = message;
            fieldContainer.appendChild(errorElement);
        }
    }
    
    clearFieldError(fieldName) {
        delete this.validationErrors[fieldName];
        
        const fieldContainer = this.container.querySelector(`[data-field-name="${fieldName}"]`);
        if (fieldContainer) {
            fieldContainer.classList.remove('field-error');
            
            const errorElement = fieldContainer.querySelector('.field-error-message');
            if (errorElement) {
                errorElement.remove();
            }
        }
    }
    
    showStepError(message) {
        // Implementation for step-level errors
        const existingError = this.stepsContainer.querySelector('.step-error');
        if (existingError) {
            existingError.remove();
        }
        
        const errorElement = document.createElement('div');
        errorElement.className = 'step-error alert alert-danger';
        errorElement.textContent = message;
        
        this.stepsContainer.insertBefore(errorElement, this.stepsContainer.firstChild);
    }
    
    updateConditionalFields() {
        // Re-render current step to show/hide conditional fields
        this.renderCurrentStep();
    }
    
    evaluateCondition(condition) {
        if (typeof condition === 'function') {
            return condition(this.formData);
        }
        
        if (typeof condition === 'object') {
            const { field, operator, value } = condition;
            const fieldValue = this.formData[field];
            
            switch (operator) {
                case 'equals':
                    return fieldValue === value;
                case 'not_equals':
                    return fieldValue !== value;
                case 'greater_than':
                    return fieldValue > value;
                case 'less_than':
                    return fieldValue < value;
                case 'contains':
                    return fieldValue && fieldValue.includes(value);
                case 'in':
                    return Array.isArray(value) && value.includes(fieldValue);
                default:
                    return true;
            }
        }
        
        return true;
    }
    
    async handleFileUpload(fieldName, files) {
        const filesArray = Array.from(files);
        const uploadedFiles = [];
        
        for (const file of filesArray) {
            // Validate file size
            if (file.size > this.options.maxFileSize) {
                this.setFieldError(fieldName, `File "${file.name}" is too large. Maximum size is ${this.formatFileSize(this.options.maxFileSize)}`);
                continue;
            }
            
            // Validate file type
            const fileExtension = '.' + file.name.split('.').pop().toLowerCase();
            if (!this.options.allowedFileTypes.includes(fileExtension)) {
                this.setFieldError(fieldName, `File type "${fileExtension}" is not allowed`);
                continue;
            }
            
            try {
                const uploadedFile = await this.uploadFile(file);
                uploadedFiles.push(uploadedFile);
                
                // Show file preview
                this.showFilePreview(fieldName, uploadedFile);
                
            } catch (error) {
                this.setFieldError(fieldName, `Failed to upload "${file.name}": ${error.message}`);
            }
        }
        
        if (uploadedFiles.length > 0) {
            this.updateFieldValue(fieldName, uploadedFiles);
        }
    }
    
    async uploadFile(file) {
        const formData = new FormData();
        formData.append('file', file);
        
        const response = await fetch(this.options.fileUploadUrl, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`Upload failed: ${response.statusText}`);
        }
        
        const result = await response.json();
        return {
            id: result.id || Date.now(),
            name: file.name,
            size: file.size,
            type: file.type,
            url: result.url,
            uploadedAt: new Date().toISOString()
        };
    }
    
    showFilePreview(fieldName, file) {
        const fieldContainer = this.container.querySelector(`[data-field-name="${fieldName}"]`);
        const previewArea = fieldContainer.querySelector('.file-preview-area');
        
        if (!previewArea) return;
        
        const preview = document.createElement('div');
        preview.className = 'file-preview';
        preview.innerHTML = `
            <div class="file-info">
                <span class="file-name">${file.name}</span>
                <span class="file-size">${this.formatFileSize(file.size)}</span>
            </div>
            <button type="button" class="btn btn-ghost btn-sm remove-file" data-file-id="${file.id}">
                Remove
            </button>
        `;
        
        preview.querySelector('.remove-file').addEventListener('click', () => {
            this.removeFile(fieldName, file.id);
            preview.remove();
        });
        
        previewArea.appendChild(preview);
    }
    
    removeFile(fieldName, fileId) {
        const currentFiles = this.formData[fieldName] || [];
        const updatedFiles = currentFiles.filter(file => file.id !== fileId);
        this.updateFieldValue(fieldName, updatedFiles);
    }
    
    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
    
    nextStep() {
        if (this.options.validateOnStepChange && !this.validateStep()) {
            return false;
        }
        
        // Call step exit callback
        const currentStep = this.steps[this.currentStep];
        if (currentStep && currentStep.onExit) {
            const result = currentStep.onExit(this.formData, this);
            if (result === false) {
                return false;
            }
        }
        
        if (this.currentStep < this.steps.length - 1) {
            this.currentStep++;
            this.renderCurrentStep();
            this.saveProgress();
            
            this.emitEvent('stepChanged', {
                currentStep: this.currentStep,
                totalSteps: this.steps.length,
                formData: this.formData
            });
            
            return true;
        }
        
        return false;
    }
    
    previousStep() {
        if (this.currentStep > 0) {
            this.currentStep--;
            this.renderCurrentStep();
            this.saveProgress();
            
            this.emitEvent('stepChanged', {
                currentStep: this.currentStep,
                totalSteps: this.steps.length,
                formData: this.formData
            });
            
            return true;
        }
        
        return false;
    }
    
    goToStep(stepIndex) {
        if (stepIndex >= 0 && stepIndex < this.steps.length && stepIndex <= this.currentStep) {
            this.currentStep = stepIndex;
            this.renderCurrentStep();
            this.saveProgress();
            
            this.emitEvent('stepChanged', {
                currentStep: this.currentStep,
                totalSteps: this.steps.length,
                formData: this.formData
            });
            
            return true;
        }
        
        return false;
    }
    
    async submitForm() {
        // Validate all steps
        for (let i = 0; i < this.steps.length; i++) {
            if (!this.validateStep(i)) {
                this.goToStep(i);
                return false;
            }
        }
        
        this.isSubmitting = true;
        this.renderNavigation();
        
        try {
            this.emitEvent('submitStart', { formData: this.formData });
            
            let result;
            if (this.options.submitUrl) {
                const response = await fetch(this.options.submitUrl, {
                    method: this.options.submitMethod,
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(this.formData)
                });
                
                if (!response.ok) {
                    throw new Error(`Submit failed: ${response.statusText}`);
                }
                
                result = await response.json();
            } else {
                result = { success: true, data: this.formData };
            }
            
            this.emitEvent('submitSuccess', { result, formData: this.formData });
            this.clearSavedProgress();
            
            return result;
            
        } catch (error) {
            this.emitEvent('submitError', { error, formData: this.formData });
            throw error;
            
        } finally {
            this.isSubmitting = false;
            this.renderNavigation();
        }
    }
    
    cancel() {
        this.emitEvent('cancel', { formData: this.formData });
        this.clearSavedProgress();
    }
    
    reset() {
        this.currentStep = 0;
        this.formData = {};
        this.validationErrors = {};
        this.uploadedFiles.clear();
        this.clearSavedProgress();
        this.renderCurrentStep();
        
        this.emitEvent('reset', {});
    }
    
    saveProgress() {
        if (!this.options.saveProgress) return;
        
        try {
            const progress = {
                currentStep: this.currentStep,
                formData: this.formData,
                timestamp: Date.now()
            };
            
            localStorage.setItem(this.options.progressKey, JSON.stringify(progress));
        } catch (error) {
            console.warn('Failed to save form progress:', error);
        }
    }
    
    loadSavedProgress() {
        if (!this.options.saveProgress) return;
        
        try {
            const saved = localStorage.getItem(this.options.progressKey);
            if (saved) {
                const progress = JSON.parse(saved);
                this.currentStep = progress.currentStep || 0;
                this.formData = progress.formData || {};
            }
        } catch (error) {
            console.warn('Failed to load saved progress:', error);
        }
    }
    
    clearSavedProgress() {
        if (!this.options.saveProgress) return;
        
        try {
            localStorage.removeItem(this.options.progressKey);
        } catch (error) {
            console.warn('Failed to clear saved progress:', error);
        }
    }
    
    bindEvents() {
        // Form submission
        this.form.addEventListener('submit', (e) => {
            e.preventDefault();
            this.submitForm();
        });
        
        // Keyboard navigation
        this.container.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && e.ctrlKey) {
                e.preventDefault();
                if (this.currentStep === this.steps.length - 1) {
                    this.submitForm();
                } else {
                    this.nextStep();
                }
            }
        });
    }
    
    emitEvent(eventName, detail) {
        const event = new CustomEvent(`multiStepForm:${eventName}`, {
            detail,
            bubbles: true,
            cancelable: true
        });
        this.container.dispatchEvent(event);
    }
    
    // Public API
    getData() {
        return { ...this.formData };
    }
    
    setData(data) {
        this.formData = { ...this.formData, ...data };
        this.renderCurrentStep();
    }
    
    getCurrentStep() {
        return this.currentStep;
    }
    
    getTotalSteps() {
        return this.steps.length;
    }
    
    isLastStep() {
        return this.currentStep === this.steps.length - 1;
    }
    
    isFirstStep() {
        return this.currentStep === 0;
    }
    
    getProgress() {
        return {
            current: this.currentStep + 1,
            total: this.steps.length,
            percentage: ((this.currentStep + 1) / this.steps.length) * 100
        };
    }
    
    addCustomValidator(name, validator) {
        this.validators.set(name, validator);
    }
    
    addCustomFieldType(type, renderer) {
        this.fieldComponents.set(type, renderer);
    }
    
    start() {
        if (this.steps.length === 0) {
            throw new Error('No steps defined. Add steps before starting the form.');
        }
        
        this.renderCurrentStep();
        this.emitEvent('started', { totalSteps: this.steps.length });
    }
    
    destroy() {
        this.clearSavedProgress();
        this.container.innerHTML = '';
        this.steps = [];
        this.formData = {};
        this.validationErrors = {};
        this.uploadedFiles.clear();
    }
}

// Form configuration examples
export const anomalyDetectionFormSteps = [
    {
        id: 'dataset',
        title: 'Dataset Configuration',
        description: 'Upload and configure your dataset for anomaly detection',
        fields: [
            {
                type: 'text',
                name: 'dataset_name',
                label: 'Dataset Name',
                placeholder: 'Enter a name for your dataset',
                required: true,
                help: 'Choose a descriptive name that will help you identify this dataset later'
            },
            {
                type: 'file',
                name: 'dataset_file',
                label: 'Dataset File',
                required: true,
                accept: '.csv,.json,.xlsx,.parquet',
                help: 'Upload your dataset in CSV, JSON, Excel, or Parquet format'
            },
            {
                type: 'checkbox',
                name: 'has_header',
                label: 'Dataset has header row',
                defaultValue: true
            }
        ]
    },
    {
        id: 'algorithm',
        title: 'Algorithm Selection',
        description: 'Choose the anomaly detection algorithm and configure parameters',
        fields: [
            {
                type: 'select',
                name: 'algorithm',
                label: 'Detection Algorithm',
                required: true,
                options: [
                    { value: 'isolation_forest', label: 'Isolation Forest' },
                    { value: 'one_class_svm', label: 'One-Class SVM' },
                    { value: 'local_outlier_factor', label: 'Local Outlier Factor' },
                    { value: 'elliptic_envelope', label: 'Elliptic Envelope' }
                ]
            },
            {
                type: 'range',
                name: 'contamination',
                label: 'Contamination Rate',
                min: 0.01,
                max: 0.5,
                step: 0.01,
                defaultValue: 0.1,
                help: 'Expected proportion of outliers in the dataset'
            },
            {
                type: 'select',
                name: 'features',
                label: 'Feature Selection',
                options: [
                    { value: 'all', label: 'Use all features' },
                    { value: 'select', label: 'Select specific features' },
                    { value: 'auto', label: 'Auto-select features' }
                ],
                defaultValue: 'all'
            }
        ]
    },
    {
        id: 'execution',
        title: 'Execution Settings',
        description: 'Configure how the anomaly detection will be executed',
        fields: [
            {
                type: 'select',
                name: 'execution_mode',
                label: 'Execution Mode',
                required: true,
                options: [
                    { value: 'immediate', label: 'Run immediately' },
                    { value: 'scheduled', label: 'Schedule for later' },
                    { value: 'batch', label: 'Batch processing' }
                ]
            },
            {
                type: 'checkbox',
                name: 'save_model',
                label: 'Save trained model',
                defaultValue: true,
                help: 'Save the model for future use and comparison'
            },
            {
                type: 'checkbox',
                name: 'generate_report',
                label: 'Generate detailed report',
                defaultValue: true
            }
        ]
    }
];

export default MultiStepForm;
