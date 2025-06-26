/**
 * Interactive Form Components
 * Advanced form handling with validation, multi-step workflows, and dynamic interactions
 */

export class MultiStepForm {
  constructor(container, options = {}) {
    this.container = container;
    this.options = {
      showProgress: true,
      allowBack: true,
      autoSave: true,
      validationMode: 'onSubmit', // 'onSubmit', 'onChange', 'onBlur'
      ...options
    };
    
    this.currentStep = 0;
    this.steps = [];
    this.formData = new Map();
    this.validators = new Map();
    this.isValid = false;
    
    this.init();
  }
  
  init() {
    this.parseSteps();
    this.createProgressIndicator();
    this.setupValidation();
    this.setupEventHandlers();
    this.showStep(0);
  }
  
  parseSteps() {
    this.steps = Array.from(this.container.querySelectorAll('[data-step]'))
      .map((stepElement, index) => ({
        element: stepElement,
        index,
        title: stepElement.dataset.stepTitle || `Step ${index + 1}`,
        isValid: false,
        isRequired: stepElement.hasAttribute('data-required'),
        fields: Array.from(stepElement.querySelectorAll('input, select, textarea'))
      }));
  }
  
  createProgressIndicator() {
    if (!this.options.showProgress) return;
    
    const progressContainer = document.createElement('div');
    progressContainer.className = 'form-progress';
    progressContainer.innerHTML = `
      <div class="progress-steps">
        ${this.steps.map((step, index) => `
          <div class="progress-step" data-step="${index}">
            <div class="step-indicator">
              <span class="step-number">${index + 1}</span>
              <div class="step-checkmark" style="display: none;">✓</div>
            </div>
            <div class="step-title">${step.title}</div>
          </div>
        `).join('')}
      </div>
      <div class="progress-bar">
        <div class="progress-fill" style="width: 0%"></div>
      </div>
    `;
    
    this.container.insertBefore(progressContainer, this.container.firstChild);
    this.progressContainer = progressContainer;
  }
  
  setupValidation() {
    this.steps.forEach(step => {
      step.fields.forEach(field => {
        const validators = this.parseValidationRules(field);
        if (validators.length > 0) {
          this.validators.set(field, validators);
        }
        
        // Add real-time validation based on mode
        if (this.options.validationMode === 'onChange') {
          field.addEventListener('input', () => this.validateField(field));
        } else if (this.options.validationMode === 'onBlur') {
          field.addEventListener('blur', () => this.validateField(field));
        }
      });
    });
  }
  
  parseValidationRules(field) {
    const rules = [];
    
    // Required validation
    if (field.required || field.hasAttribute('data-required')) {
      rules.push({
        type: 'required',
        message: field.dataset.requiredMessage || 'This field is required'
      });
    }
    
    // Email validation
    if (field.type === 'email' || field.hasAttribute('data-email')) {
      rules.push({
        type: 'email',
        message: field.dataset.emailMessage || 'Please enter a valid email address'
      });
    }
    
    // Min/Max length
    if (field.minLength) {
      rules.push({
        type: 'minLength',
        value: field.minLength,
        message: field.dataset.minlengthMessage || `Minimum ${field.minLength} characters required`
      });
    }
    
    if (field.maxLength) {
      rules.push({
        type: 'maxLength',
        value: field.maxLength,
        message: field.dataset.maxlengthMessage || `Maximum ${field.maxLength} characters allowed`
      });
    }
    
    // Number validation
    if (field.type === 'number') {
      if (field.min !== '') {
        rules.push({
          type: 'min',
          value: parseFloat(field.min),
          message: field.dataset.minMessage || `Value must be at least ${field.min}`
        });
      }
      
      if (field.max !== '') {
        rules.push({
          type: 'max',
          value: parseFloat(field.max),
          message: field.dataset.maxMessage || `Value must be at most ${field.max}`
        });
      }
    }
    
    // Custom pattern validation
    if (field.pattern) {
      rules.push({
        type: 'pattern',
        value: new RegExp(field.pattern),
        message: field.dataset.patternMessage || 'Please match the requested format'
      });
    }
    
    // Custom validation function
    if (field.dataset.customValidator) {
      const validatorName = field.dataset.customValidator;
      const customValidator = this.options.customValidators?.[validatorName];
      if (customValidator) {
        rules.push({
          type: 'custom',
          validator: customValidator,
          message: field.dataset.customMessage || 'Please enter a valid value'
        });
      }
    }
    
    return rules;
  }
  
  validateField(field) {
    const validators = this.validators.get(field);
    if (!validators) return true;
    
    const value = field.value.trim();
    const errors = [];
    
    for (const rule of validators) {
      switch (rule.type) {
        case 'required':
          if (!value) errors.push(rule.message);
          break;
          
        case 'email':
          const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
          if (value && !emailRegex.test(value)) errors.push(rule.message);
          break;
          
        case 'minLength':
          if (value && value.length < rule.value) errors.push(rule.message);
          break;
          
        case 'maxLength':
          if (value && value.length > rule.value) errors.push(rule.message);
          break;
          
        case 'min':
          if (value && parseFloat(value) < rule.value) errors.push(rule.message);
          break;
          
        case 'max':
          if (value && parseFloat(value) > rule.value) errors.push(rule.message);
          break;
          
        case 'pattern':
          if (value && !rule.value.test(value)) errors.push(rule.message);
          break;
          
        case 'custom':
          if (value && !rule.validator(value, field)) errors.push(rule.message);
          break;
      }
    }
    
    this.showFieldErrors(field, errors);
    return errors.length === 0;
  }
  
  showFieldErrors(field, errors) {
    // Remove existing error messages
    const existingError = field.parentElement.querySelector('.field-error');
    if (existingError) {
      existingError.remove();
    }
    
    // Update field styling
    field.classList.remove('field-valid', 'field-invalid');
    
    if (errors.length > 0) {
      field.classList.add('field-invalid');
      
      // Create error message element
      const errorElement = document.createElement('div');
      errorElement.className = 'field-error';
      errorElement.textContent = errors[0]; // Show first error
      
      field.parentElement.appendChild(errorElement);
    } else if (field.value.trim()) {
      field.classList.add('field-valid');
    }
  }
  
  setupEventHandlers() {
    // Navigation buttons
    this.container.addEventListener('click', (e) => {
      if (e.target.matches('[data-action="next"]')) {
        this.nextStep();
      } else if (e.target.matches('[data-action="prev"]')) {
        this.prevStep();
      } else if (e.target.matches('[data-action="submit"]')) {
        this.submitForm();
      }
    });
    
    // Step indicator clicks
    if (this.progressContainer) {
      this.progressContainer.addEventListener('click', (e) => {
        const stepElement = e.target.closest('.progress-step');
        if (stepElement) {
          const stepIndex = parseInt(stepElement.dataset.step);
          if (stepIndex <= this.currentStep || this.isStepAccessible(stepIndex)) {
            this.showStep(stepIndex);
          }
        }
      });
    }
    
    // Auto-save functionality
    if (this.options.autoSave) {
      this.container.addEventListener('input', (e) => {
        if (e.target.matches('input, select, textarea')) {
          this.saveFieldData(e.target);
        }
      });
    }
    
    // Form submission
    this.container.addEventListener('submit', (e) => {
      e.preventDefault();
      this.submitForm();
    });
  }
  
  showStep(stepIndex) {
    if (stepIndex < 0 || stepIndex >= this.steps.length) return;
    
    // Hide all steps
    this.steps.forEach(step => {
      step.element.style.display = 'none';
    });
    
    // Show current step
    this.steps[stepIndex].element.style.display = 'block';
    this.currentStep = stepIndex;
    
    // Update progress indicator
    this.updateProgress();
    
    // Update navigation buttons
    this.updateNavigation();
    
    // Focus first input in step
    const firstInput = this.steps[stepIndex].element.querySelector('input, select, textarea');
    if (firstInput) {
      setTimeout(() => firstInput.focus(), 100);
    }
    
    // Emit step change event
    this.container.dispatchEvent(new CustomEvent('stepChanged', {
      detail: { stepIndex, step: this.steps[stepIndex] }
    }));
  }
  
  updateProgress() {
    if (!this.progressContainer) return;
    
    const progressSteps = this.progressContainer.querySelectorAll('.progress-step');
    const progressFill = this.progressContainer.querySelector('.progress-fill');
    
    progressSteps.forEach((stepElement, index) => {
      const indicator = stepElement.querySelector('.step-indicator');
      const number = stepElement.querySelector('.step-number');
      const checkmark = stepElement.querySelector('.step-checkmark');
      
      stepElement.classList.remove('active', 'completed');
      
      if (index < this.currentStep) {
        stepElement.classList.add('completed');
        number.style.display = 'none';
        checkmark.style.display = 'block';
      } else if (index === this.currentStep) {
        stepElement.classList.add('active');
        number.style.display = 'block';
        checkmark.style.display = 'none';
      } else {
        number.style.display = 'block';
        checkmark.style.display = 'none';
      }
    });
    
    // Update progress bar
    const progress = ((this.currentStep + 1) / this.steps.length) * 100;
    progressFill.style.width = `${progress}%`;
  }
  
  updateNavigation() {
    const prevButton = this.container.querySelector('[data-action="prev"]');
    const nextButton = this.container.querySelector('[data-action="next"]');
    const submitButton = this.container.querySelector('[data-action="submit"]');
    
    // Previous button
    if (prevButton) {
      prevButton.style.display = this.currentStep > 0 && this.options.allowBack ? 'inline-block' : 'none';
    }
    
    // Next/Submit button
    const isLastStep = this.currentStep === this.steps.length - 1;
    
    if (nextButton) {
      nextButton.style.display = !isLastStep ? 'inline-block' : 'none';
    }
    
    if (submitButton) {
      submitButton.style.display = isLastStep ? 'inline-block' : 'none';
    }
  }
  
  validateStep(stepIndex) {
    const step = this.steps[stepIndex];
    if (!step) return false;
    
    let isValid = true;
    
    step.fields.forEach(field => {
      if (!this.validateField(field)) {
        isValid = false;
      }
    });
    
    step.isValid = isValid;
    return isValid;
  }
  
  nextStep() {
    if (!this.validateStep(this.currentStep)) {
      // Show validation errors and don't proceed
      this.container.dispatchEvent(new CustomEvent('validationFailed', {
        detail: { stepIndex: this.currentStep }
      }));
      return;
    }
    
    if (this.currentStep < this.steps.length - 1) {
      this.showStep(this.currentStep + 1);
    }
  }
  
  prevStep() {
    if (this.currentStep > 0) {
      this.showStep(this.currentStep - 1);
    }
  }
  
  isStepAccessible(stepIndex) {
    // Check if user can navigate to this step
    // Usually, all previous steps must be valid
    for (let i = 0; i < stepIndex; i++) {
      if (!this.steps[i].isValid) {
        return false;
      }
    }
    return true;
  }
  
  saveFieldData(field) {
    const stepIndex = this.getFieldStepIndex(field);
    const fieldName = field.name || field.id;
    
    if (fieldName) {
      this.formData.set(`${stepIndex}-${fieldName}`, field.value);
      
      // Save to localStorage for persistence
      if (this.options.autoSave) {
        localStorage.setItem(`form-${this.getFormId()}`, JSON.stringify([...this.formData]));
      }
    }
  }
  
  getFieldStepIndex(field) {
    return this.steps.findIndex(step => step.element.contains(field));
  }
  
  getFormId() {
    return this.container.id || 'multi-step-form';
  }
  
  loadSavedData() {
    if (!this.options.autoSave) return;
    
    const savedData = localStorage.getItem(`form-${this.getFormId()}`);
    if (savedData) {
      try {
        const data = JSON.parse(savedData);
        this.formData = new Map(data);
        
        // Restore field values
        this.formData.forEach((value, key) => {
          const [stepIndex, fieldName] = key.split('-', 2);
          const field = this.container.querySelector(`[name="${fieldName}"], #${fieldName}`);
          if (field) {
            field.value = value;
          }
        });
      } catch (error) {
        console.warn('Failed to load saved form data:', error);
      }
    }
  }
  
  clearSavedData() {
    localStorage.removeItem(`form-${this.getFormId()}`);
    this.formData.clear();
  }
  
  getAllFormData() {
    const data = {};
    
    this.steps.forEach(step => {
      step.fields.forEach(field => {
        if (field.name) {
          if (field.type === 'checkbox') {
            data[field.name] = field.checked;
          } else if (field.type === 'radio') {
            if (field.checked) {
              data[field.name] = field.value;
            }
          } else {
            data[field.name] = field.value;
          }
        }
      });
    });
    
    return data;
  }
  
  validateAllSteps() {
    let allValid = true;
    
    this.steps.forEach((step, index) => {
      if (!this.validateStep(index)) {
        allValid = false;
      }
    });
    
    return allValid;
  }
  
  submitForm() {
    if (!this.validateAllSteps()) {
      // Find first invalid step and navigate to it
      const firstInvalidStep = this.steps.findIndex(step => !step.isValid);
      if (firstInvalidStep !== -1) {
        this.showStep(firstInvalidStep);
      }
      return;
    }
    
    const formData = this.getAllFormData();
    
    // Emit submit event
    const submitEvent = new CustomEvent('formSubmit', {
      detail: { formData },
      cancelable: true
    });
    
    this.container.dispatchEvent(submitEvent);
    
    if (!submitEvent.defaultPrevented) {
      // Default submission behavior
      this.performSubmission(formData);
    }
  }
  
  async performSubmission(formData) {
    const submitButton = this.container.querySelector('[data-action="submit"]');
    const originalText = submitButton?.textContent;
    
    try {
      // Show loading state
      if (submitButton) {
        submitButton.disabled = true;
        submitButton.innerHTML = '<span class="spinner"></span> Submitting...';
      }
      
      // Simulate API call or actual submission
      const response = await this.submitToServer(formData);
      
      // Handle success
      this.handleSubmissionSuccess(response);
      
      // Clear saved data
      this.clearSavedData();
      
    } catch (error) {
      this.handleSubmissionError(error);
    } finally {
      // Restore button state
      if (submitButton) {
        submitButton.disabled = false;
        submitButton.textContent = originalText;
      }
    }
  }
  
  async submitToServer(formData) {
    const submitUrl = this.container.dataset.submitUrl || '/api/form-submit';
    
    const response = await fetch(submitUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(formData)
    });
    
    if (!response.ok) {
      throw new Error(`Submission failed: ${response.statusText}`);
    }
    
    return response.json();
  }
  
  handleSubmissionSuccess(response) {
    // Show success message
    this.showSuccessMessage();
    
    // Emit success event
    this.container.dispatchEvent(new CustomEvent('formSubmitSuccess', {
      detail: { response }
    }));
  }
  
  handleSubmissionError(error) {
    // Show error message
    this.showErrorMessage(error.message);
    
    // Emit error event
    this.container.dispatchEvent(new CustomEvent('formSubmitError', {
      detail: { error }
    }));
  }
  
  showSuccessMessage() {
    const message = document.createElement('div');
    message.className = 'form-success-message';
    message.innerHTML = `
      <div class="success-icon">✓</div>
      <div class="success-text">
        <h3>Form Submitted Successfully!</h3>
        <p>Thank you for your submission. You will receive a confirmation shortly.</p>
      </div>
    `;
    
    this.container.innerHTML = '';
    this.container.appendChild(message);
  }
  
  showErrorMessage(errorText) {
    // Remove existing error messages
    const existingError = this.container.querySelector('.form-error-message');
    if (existingError) {
      existingError.remove();
    }
    
    const message = document.createElement('div');
    message.className = 'form-error-message';
    message.innerHTML = `
      <div class="error-icon">✕</div>
      <div class="error-text">
        <strong>Submission Failed:</strong> ${errorText}
      </div>
      <button class="error-dismiss" onclick="this.parentElement.remove()">Dismiss</button>
    `;
    
    this.container.insertBefore(message, this.container.firstChild);
    
    // Auto-remove after 10 seconds
    setTimeout(() => {
      if (message.parentElement) {
        message.remove();
      }
    }, 10000);
  }
  
  reset() {
    // Reset form data
    this.formData.clear();
    this.clearSavedData();
    
    // Reset all fields
    this.steps.forEach(step => {
      step.fields.forEach(field => {
        if (field.type === 'checkbox' || field.type === 'radio') {
          field.checked = false;
        } else {
          field.value = '';
        }
        
        // Clear validation styling
        field.classList.remove('field-valid', 'field-invalid');
        
        // Remove error messages
        const errorElement = field.parentElement.querySelector('.field-error');
        if (errorElement) {
          errorElement.remove();
        }
      });
      
      step.isValid = false;
    });
    
    // Return to first step
    this.showStep(0);
  }
  
  destroy() {
    // Clean up event listeners and DOM modifications
    this.clearSavedData();
    
    if (this.progressContainer) {
      this.progressContainer.remove();
    }
  }
}

// Factory function for easy instantiation
export function createMultiStepForm(container, options = {}) {
  return new MultiStepForm(container, options);
}

// File Upload Component with Progress
export class FileUploadComponent {
  constructor(container, options = {}) {
    this.container = container;
    this.options = {
      maxFileSize: 10 * 1024 * 1024, // 10MB
      allowedTypes: ['csv', 'json', 'xlsx', 'parquet'],
      multiple: false,
      uploadUrl: '/api/upload',
      ...options
    };
    
    this.files = [];
    this.uploadQueue = [];
    
    this.init();
  }
  
  init() {
    this.createUploadInterface();
    this.setupEventHandlers();
  }
  
  createUploadInterface() {
    this.container.innerHTML = `
      <div class="file-upload-area">
        <div class="upload-zone" id="upload-zone">
          <div class="upload-icon">📁</div>
          <div class="upload-text">
            <strong>Drag and drop files here</strong>
            <span>or <button type="button" class="upload-browse">browse files</button></span>
          </div>
          <div class="upload-hint">
            Supported formats: ${this.options.allowedTypes.join(', ')} 
            (max ${this.formatFileSize(this.options.maxFileSize)})
          </div>
        </div>
        <input type="file" id="file-input" style="display: none;" 
               ${this.options.multiple ? 'multiple' : ''}
               accept=".${this.options.allowedTypes.join(',.')}">
      </div>
      <div class="file-list" id="file-list" style="display: none;"></div>
    `;
  }
  
  setupEventHandlers() {
    const uploadZone = this.container.querySelector('#upload-zone');
    const fileInput = this.container.querySelector('#file-input');
    const browseButton = this.container.querySelector('.upload-browse');
    
    // Browse button
    browseButton.addEventListener('click', () => {
      fileInput.click();
    });
    
    // File input change
    fileInput.addEventListener('change', (e) => {
      this.handleFiles(Array.from(e.target.files));
    });
    
    // Drag and drop
    uploadZone.addEventListener('dragover', (e) => {
      e.preventDefault();
      uploadZone.classList.add('drag-over');
    });
    
    uploadZone.addEventListener('dragleave', () => {
      uploadZone.classList.remove('drag-over');
    });
    
    uploadZone.addEventListener('drop', (e) => {
      e.preventDefault();
      uploadZone.classList.remove('drag-over');
      this.handleFiles(Array.from(e.dataTransfer.files));
    });
  }
  
  handleFiles(fileList) {
    const validFiles = fileList.filter(file => this.validateFile(file));
    
    if (!this.options.multiple) {
      this.files = validFiles.slice(0, 1);
    } else {
      this.files.push(...validFiles);
    }
    
    this.updateFileList();
    this.startUploads();
  }
  
  validateFile(file) {
    // Check file size
    if (file.size > this.options.maxFileSize) {
      this.showError(`File "${file.name}" is too large. Maximum size is ${this.formatFileSize(this.options.maxFileSize)}.`);
      return false;
    }
    
    // Check file type
    const extension = file.name.split('.').pop().toLowerCase();
    if (!this.options.allowedTypes.includes(extension)) {
      this.showError(`File type "${extension}" is not allowed. Supported formats: ${this.options.allowedTypes.join(', ')}.`);
      return false;
    }
    
    return true;
  }
  
  updateFileList() {
    const fileList = this.container.querySelector('#file-list');
    
    if (this.files.length === 0) {
      fileList.style.display = 'none';
      return;
    }
    
    fileList.style.display = 'block';
    fileList.innerHTML = this.files.map((file, index) => `
      <div class="file-item" data-index="${index}">
        <div class="file-info">
          <div class="file-icon">${this.getFileIcon(file)}</div>
          <div class="file-details">
            <div class="file-name">${file.name}</div>
            <div class="file-meta">
              ${this.formatFileSize(file.size)} • ${file.type || 'Unknown type'}
            </div>
          </div>
        </div>
        <div class="file-progress">
          <div class="progress-bar">
            <div class="progress-fill" style="width: 0%"></div>
          </div>
          <div class="progress-text">Pending</div>
        </div>
        <div class="file-actions">
          <button class="btn-ghost btn-xs remove-file" data-index="${index}">Remove</button>
        </div>
      </div>
    `).join('');
    
    // Add remove file event handlers
    fileList.querySelectorAll('.remove-file').forEach(button => {
      button.addEventListener('click', (e) => {
        const index = parseInt(e.target.dataset.index);
        this.removeFile(index);
      });
    });
  }
  
  getFileIcon(file) {
    const extension = file.name.split('.').pop().toLowerCase();
    const icons = {
      'csv': '📊',
      'json': '📄',
      'xlsx': '📈',
      'xls': '📈',
      'parquet': '🗃️'
    };
    return icons[extension] || '📄';
  }
  
  formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }
  
  removeFile(index) {
    this.files.splice(index, 1);
    this.updateFileList();
  }
  
  async startUploads() {
    for (let i = 0; i < this.files.length; i++) {
      const file = this.files[i];
      if (!file.uploaded && !file.uploading) {
        await this.uploadFile(file, i);
      }
    }
  }
  
  async uploadFile(file, index) {
    file.uploading = true;
    
    const fileItem = this.container.querySelector(`[data-index="${index}"]`);
    const progressFill = fileItem.querySelector('.progress-fill');
    const progressText = fileItem.querySelector('.progress-text');
    
    try {
      progressText.textContent = 'Uploading...';
      
      const formData = new FormData();
      formData.append('file', file);
      
      const xhr = new XMLHttpRequest();
      
      // Track upload progress
      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable) {
          const percentComplete = (e.loaded / e.total) * 100;
          progressFill.style.width = `${percentComplete}%`;
          progressText.textContent = `${Math.round(percentComplete)}%`;
        }
      };
      
      // Handle completion
      xhr.onload = () => {
        if (xhr.status === 200) {
          const response = JSON.parse(xhr.responseText);
          file.uploaded = true;
          file.response = response;
          progressText.textContent = 'Complete';
          progressFill.style.backgroundColor = '#22c55e';
          
          // Emit upload success event
          this.container.dispatchEvent(new CustomEvent('fileUploaded', {
            detail: { file, response, index }
          }));
        } else {
          throw new Error(`Upload failed: ${xhr.statusText}`);
        }
      };
      
      xhr.onerror = () => {
        throw new Error('Upload failed due to network error');
      };
      
      xhr.open('POST', this.options.uploadUrl);
      xhr.send(formData);
      
    } catch (error) {
      file.uploading = false;
      file.error = error.message;
      progressText.textContent = 'Failed';
      progressFill.style.backgroundColor = '#ef4444';
      
      this.showError(`Upload failed for "${file.name}": ${error.message}`);
      
      // Emit upload error event
      this.container.dispatchEvent(new CustomEvent('fileUploadError', {
        detail: { file, error, index }
      }));
    }
  }
  
  showError(message) {
    // Create or update error message
    let errorContainer = this.container.querySelector('.upload-error');
    if (!errorContainer) {
      errorContainer = document.createElement('div');
      errorContainer.className = 'upload-error';
      this.container.appendChild(errorContainer);
    }
    
    const errorItem = document.createElement('div');
    errorItem.className = 'error-item';
    errorItem.innerHTML = `
      <span class="error-text">${message}</span>
      <button class="error-dismiss" onclick="this.parentElement.remove()">×</button>
    `;
    
    errorContainer.appendChild(errorItem);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
      if (errorItem.parentElement) {
        errorItem.remove();
      }
    }, 5000);
  }
  
  getUploadedFiles() {
    return this.files.filter(file => file.uploaded);
  }
  
  clearFiles() {
    this.files = [];
    this.updateFileList();
  }
}

// Auto-initialize components
export function initializeInteractiveForms() {
  // Initialize multi-step forms
  document.querySelectorAll('[data-component="multi-step-form"]').forEach(container => {
    new MultiStepForm(container);
  });
  
  // Initialize file upload components
  document.querySelectorAll('[data-component="file-upload"]').forEach(container => {
    new FileUploadComponent(container);
  });
}

// Initialize on DOM ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initializeInteractiveForms);
} else {
  initializeInteractiveForms();
}