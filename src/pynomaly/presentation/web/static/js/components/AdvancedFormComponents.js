/**
 * Advanced Form Components for Pynomaly
 * 
 * Features:
 * - Multi-step form wizard with validation
 * - Dynamic field generation and validation
 * - Real-time validation with debouncing
 * - Accessibility-first design with ARIA support
 * - File upload with progress and drag-and-drop
 * - Advanced input types (date ranges, multi-select, etc.)
 * - Form state management integration
 * - Conditional field rendering
 */

class AdvancedFormComponents {
  constructor(options = {}) {
    this.options = {
      validateOnChange: true,
      validateOnBlur: true,
      debounceMs: 300,
      accessibility: {
        announceErrors: true,
        liveValidation: true
      },
      theme: 'light',
      ...options
    };

    this.validators = new Map();
    this.forms = new Map();
    this.components = new Map();
    
    this.init();
  }

  init() {
    this.setupValidators();
    this.setupGlobalStyles();
  }

  setupValidators() {
    // Basic validators
    this.addValidator('required', (value, field) => {
      if (!value || (Array.isArray(value) && value.length === 0)) {
        return `${field.label || 'This field'} is required`;
      }
      return null;
    });

    this.addValidator('email', (value) => {
      if (value && !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(value)) {
        return 'Please enter a valid email address';
      }
      return null;
    });

    this.addValidator('minLength', (value, field) => {
      if (value && value.length < field.minLength) {
        return `Must be at least ${field.minLength} characters long`;
      }
      return null;
    });

    this.addValidator('maxLength', (value, field) => {
      if (value && value.length > field.maxLength) {
        return `Must be no more than ${field.maxLength} characters long`;
      }
      return null;
    });

    this.addValidator('number', (value, field) => {
      if (value && isNaN(value)) {
        return 'Please enter a valid number';
      }
      if (value && field.min !== undefined && parseFloat(value) < field.min) {
        return `Must be at least ${field.min}`;
      }
      if (value && field.max !== undefined && parseFloat(value) > field.max) {
        return `Must be no more than ${field.max}`;
      }
      return null;
    });

    this.addValidator('date', (value) => {
      if (value && isNaN(new Date(value).getTime())) {
        return 'Please enter a valid date';
      }
      return null;
    });

    this.addValidator('url', (value) => {
      if (value && !/^https?:\/\/.+/.test(value)) {
        return 'Please enter a valid URL starting with http:// or https://';
      }
      return null;
    });

    this.addValidator('custom', (value, field) => {
      if (field.customValidator && typeof field.customValidator === 'function') {
        return field.customValidator(value, field);
      }
      return null;
    });
  }

  setupGlobalStyles() {
    if (!document.querySelector('#advanced-form-styles')) {
      const styleSheet = document.createElement('style');
      styleSheet.id = 'advanced-form-styles';
      styleSheet.textContent = this.getFormStyles();
      document.head.appendChild(styleSheet);
    }
  }

  getFormStyles() {
    return `
      /* Advanced Form Components Styles */
      .advanced-form {
        font-family: var(--font-family-sans, -apple-system, BlinkMacSystemFont, sans-serif);
        max-width: 100%;
        margin: 0 auto;
      }

      .form-step {
        display: none;
        opacity: 0;
        transform: translateX(20px);
        transition: all 0.3s ease-out;
      }

      .form-step.active {
        display: block;
        opacity: 1;
        transform: translateX(0);
      }

      .form-step.previous {
        transform: translateX(-20px);
      }

      .form-progress {
        display: flex;
        align-items: center;
        margin-bottom: var(--spacing-6, 24px);
        padding: var(--spacing-4, 16px);
        background: var(--color-bg-secondary, #f9fafb);
        border-radius: var(--border-radius-lg, 12px);
      }

      .progress-step {
        display: flex;
        align-items: center;
        flex: 1;
        position: relative;
      }

      .progress-step:not(:last-child)::after {
        content: '';
        position: absolute;
        top: 50%;
        right: -50%;
        width: 100%;
        height: 2px;
        background: var(--color-border-light, #e5e7eb);
        transform: translateY(-50%);
        z-index: 1;
      }

      .progress-step.completed::after {
        background: var(--color-primary-500, #3b82f6);
      }

      .progress-indicator {
        width: 32px;
        height: 32px;
        border-radius: 50%;
        background: var(--color-border-light, #e5e7eb);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: var(--font-size-sm, 14px);
        font-weight: var(--font-weight-medium, 500);
        color: var(--color-text-muted, #6b7280);
        z-index: 2;
        position: relative;
      }

      .progress-step.active .progress-indicator {
        background: var(--color-primary-500, #3b82f6);
        color: white;
      }

      .progress-step.completed .progress-indicator {
        background: var(--color-success-500, #10b981);
        color: white;
      }

      .progress-label {
        margin-left: var(--spacing-2, 8px);
        font-size: var(--font-size-sm, 14px);
        font-weight: var(--font-weight-medium, 500);
        color: var(--color-text-secondary, #6b7280);
      }

      .progress-step.active .progress-label {
        color: var(--color-text-primary, #1f2937);
      }

      .form-field {
        margin-bottom: var(--spacing-4, 16px);
      }

      .form-field.field-group {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: var(--spacing-3, 12px);
      }

      .form-label {
        display: block;
        font-size: var(--font-size-sm, 14px);
        font-weight: var(--font-weight-medium, 500);
        color: var(--color-text-primary, #1f2937);
        margin-bottom: var(--spacing-1, 4px);
      }

      .form-label.required::after {
        content: ' *';
        color: var(--color-danger-500, #ef4444);
      }

      .form-input {
        width: 100%;
        padding: var(--spacing-3, 12px);
        border: 1px solid var(--color-border-medium, #d1d5db);
        border-radius: var(--border-radius-md, 6px);
        font-size: var(--font-size-base, 16px);
        transition: all 0.2s ease;
        background: var(--color-bg-primary, #ffffff);
        color: var(--color-text-primary, #1f2937);
      }

      .form-input:focus {
        outline: none;
        border-color: var(--color-primary-500, #3b82f6);
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
      }

      .form-input.error {
        border-color: var(--color-danger-500, #ef4444);
        box-shadow: 0 0 0 3px rgba(239, 68, 68, 0.1);
      }

      .form-input.success {
        border-color: var(--color-success-500, #10b981);
        box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.1);
      }

      .form-textarea {
        min-height: 100px;
        resize: vertical;
        font-family: inherit;
      }

      .form-select {
        background-image: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 20 20'%3e%3cpath stroke='%236b7280' stroke-linecap='round' stroke-linejoin='round' stroke-width='1.5' d='M6 8l4 4 4-4'/%3e%3c/svg%3e");
        background-position: right 8px center;
        background-repeat: no-repeat;
        background-size: 16px;
        padding-right: 40px;
        appearance: none;
      }

      .form-checkbox,
      .form-radio {
        margin-right: var(--spacing-2, 8px);
        accent-color: var(--color-primary-500, #3b82f6);
      }

      .checkbox-group,
      .radio-group {
        display: flex;
        flex-direction: column;
        gap: var(--spacing-2, 8px);
      }

      .checkbox-item,
      .radio-item {
        display: flex;
        align-items: center;
        padding: var(--spacing-2, 8px);
        border-radius: var(--border-radius-md, 6px);
        transition: background-color 0.2s ease;
      }

      .checkbox-item:hover,
      .radio-item:hover {
        background: var(--color-bg-secondary, #f9fafb);
      }

      .form-error {
        color: var(--color-danger-500, #ef4444);
        font-size: var(--font-size-sm, 14px);
        margin-top: var(--spacing-1, 4px);
        display: flex;
        align-items: center;
        gap: var(--spacing-1, 4px);
      }

      .form-success {
        color: var(--color-success-500, #10b981);
        font-size: var(--font-size-sm, 14px);
        margin-top: var(--spacing-1, 4px);
        display: flex;
        align-items: center;
        gap: var(--spacing-1, 4px);
      }

      .form-help {
        color: var(--color-text-muted, #6b7280);
        font-size: var(--font-size-sm, 14px);
        margin-top: var(--spacing-1, 4px);
      }

      .form-actions {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-top: var(--spacing-6, 24px);
        padding-top: var(--spacing-4, 16px);
        border-top: 1px solid var(--color-border-light, #e5e7eb);
      }

      .btn-group {
        display: flex;
        gap: var(--spacing-3, 12px);
      }

      .file-upload {
        border: 2px dashed var(--color-border-medium, #d1d5db);
        border-radius: var(--border-radius-lg, 12px);
        padding: var(--spacing-6, 24px);
        text-align: center;
        transition: all 0.2s ease;
        background: var(--color-bg-primary, #ffffff);
        cursor: pointer;
      }

      .file-upload:hover,
      .file-upload.dragover {
        border-color: var(--color-primary-500, #3b82f6);
        background: var(--color-primary-50, #eff6ff);
      }

      .file-upload.error {
        border-color: var(--color-danger-500, #ef4444);
        background: var(--color-danger-50, #fef2f2);
      }

      .file-upload-icon {
        font-size: var(--font-size-3xl, 30px);
        color: var(--color-text-muted, #6b7280);
        margin-bottom: var(--spacing-2, 8px);
      }

      .file-list {
        margin-top: var(--spacing-4, 16px);
      }

      .file-item {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: var(--spacing-3, 12px);
        background: var(--color-bg-secondary, #f9fafb);
        border-radius: var(--border-radius-md, 6px);
        margin-bottom: var(--spacing-2, 8px);
      }

      .file-progress {
        width: 100%;
        height: 4px;
        background: var(--color-border-light, #e5e7eb);
        border-radius: 2px;
        overflow: hidden;
        margin: var(--spacing-1, 4px) 0;
      }

      .file-progress-bar {
        height: 100%;
        background: var(--color-primary-500, #3b82f6);
        transition: width 0.3s ease;
      }

      .date-range-picker {
        display: grid;
        grid-template-columns: 1fr auto 1fr;
        gap: var(--spacing-2, 8px);
        align-items: center;
      }

      .date-range-separator {
        color: var(--color-text-muted, #6b7280);
        font-weight: var(--font-weight-medium, 500);
      }

      .multi-select {
        position: relative;
      }

      .multi-select-input {
        cursor: pointer;
        background: var(--color-bg-primary, #ffffff);
      }

      .multi-select-dropdown {
        position: absolute;
        top: 100%;
        left: 0;
        right: 0;
        background: var(--color-bg-primary, #ffffff);
        border: 1px solid var(--color-border-medium, #d1d5db);
        border-radius: var(--border-radius-md, 6px);
        box-shadow: var(--shadow-lg, 0 10px 15px -3px rgba(0, 0, 0, 0.1));
        z-index: 1000;
        max-height: 200px;
        overflow-y: auto;
      }

      .multi-select-option {
        padding: var(--spacing-2, 8px) var(--spacing-3, 12px);
        cursor: pointer;
        transition: background-color 0.2s ease;
        display: flex;
        align-items: center;
        gap: var(--spacing-2, 8px);
      }

      .multi-select-option:hover {
        background: var(--color-bg-secondary, #f9fafb);
      }

      .multi-select-option.selected {
        background: var(--color-primary-50, #eff6ff);
        color: var(--color-primary-700, #1d4ed8);
      }

      .selected-tags {
        display: flex;
        flex-wrap: wrap;
        gap: var(--spacing-1, 4px);
        margin-top: var(--spacing-2, 8px);
      }

      .tag {
        display: inline-flex;
        align-items: center;
        gap: var(--spacing-1, 4px);
        background: var(--color-primary-100, #dbeafe);
        color: var(--color-primary-700, #1d4ed8);
        padding: var(--spacing-1, 4px) var(--spacing-2, 8px);
        border-radius: var(--border-radius-md, 6px);
        font-size: var(--font-size-sm, 14px);
      }

      .tag-remove {
        cursor: pointer;
        border: none;
        background: none;
        color: var(--color-primary-500, #3b82f6);
        padding: 0;
        width: 16px;
        height: 16px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 50%;
        transition: background-color 0.2s ease;
      }

      .tag-remove:hover {
        background: var(--color-primary-200, #bfdbfe);
      }

      @media (prefers-reduced-motion: reduce) {
        .form-step,
        .form-input,
        .file-upload,
        .multi-select-option,
        .checkbox-item,
        .radio-item {
          transition: none;
        }
      }

      @media (max-width: 768px) {
        .form-field.field-group {
          grid-template-columns: 1fr;
        }

        .form-actions {
          flex-direction: column;
          gap: var(--spacing-3, 12px);
        }

        .btn-group {
          width: 100%;
          justify-content: space-between;
        }

        .date-range-picker {
          grid-template-columns: 1fr;
          gap: var(--spacing-3, 12px);
        }

        .date-range-separator {
          text-align: center;
        }
      }
    `;
  }

  // Multi-step Form Wizard
  createMultiStepForm(container, config) {
    const formId = 'form_' + Math.random().toString(36).substr(2, 9);
    const formInstance = new MultiStepForm(container, { ...config, formId }, this);
    this.forms.set(formId, formInstance);
    return formInstance;
  }

  // Advanced Input Components
  createFileUpload(container, options = {}) {
    return new FileUploadComponent(container, options, this);
  }

  createDateRangePicker(container, options = {}) {
    return new DateRangePickerComponent(container, options, this);
  }

  createMultiSelect(container, options = {}) {
    return new MultiSelectComponent(container, options, this);
  }

  createDynamicFieldset(container, options = {}) {
    return new DynamicFieldsetComponent(container, options, this);
  }

  // Validator management
  addValidator(name, validatorFn) {
    this.validators.set(name, validatorFn);
  }

  removeValidator(name) {
    this.validators.delete(name);
  }

  validateField(value, field) {
    const errors = [];
    
    if (field.validators) {
      for (const validatorConfig of field.validators) {
        const validatorName = typeof validatorConfig === 'string' 
          ? validatorConfig 
          : validatorConfig.name;
        
        const validator = this.validators.get(validatorName);
        if (validator) {
          const error = validator(value, { ...field, ...validatorConfig });
          if (error) {
            errors.push(error);
          }
        }
      }
    }
    
    return errors;
  }

  // Utility methods
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

  announceToUser(message) {
    if (!this.options.accessibility.announceErrors) return;

    let liveRegion = document.getElementById('form-announcements');
    if (!liveRegion) {
      liveRegion = document.createElement('div');
      liveRegion.id = 'form-announcements';
      liveRegion.className = 'sr-only';
      liveRegion.setAttribute('aria-live', 'polite');
      liveRegion.setAttribute('aria-atomic', 'true');
      document.body.appendChild(liveRegion);
    }

    liveRegion.textContent = message;
  }

  destroy() {
    this.forms.forEach(form => form.destroy());
    this.components.forEach(component => component.destroy());
    this.forms.clear();
    this.components.clear();
    this.validators.clear();
  }
}

// Multi-Step Form Implementation
class MultiStepForm {
  constructor(container, config, parent) {
    this.container = typeof container === 'string' 
      ? document.querySelector(container) 
      : container;
    this.config = config;
    this.parent = parent;
    this.currentStep = 0;
    this.formData = {};
    this.errors = {};
    this.touched = {};
    
    this.init();
  }

  init() {
    this.render();
    this.bindEvents();
    this.updateProgress();
  }

  render() {
    const { steps, title } = this.config;
    
    this.container.innerHTML = `
      <div class="advanced-form" role="form" aria-labelledby="form-title">
        ${title ? `<h2 id="form-title" class="form-title">${title}</h2>` : ''}
        
        <div class="form-progress" role="progressbar" aria-valuenow="${this.currentStep + 1}" aria-valuemin="1" aria-valuemax="${steps.length}">
          ${steps.map((step, index) => `
            <div class="progress-step ${index === this.currentStep ? 'active' : ''} ${index < this.currentStep ? 'completed' : ''}" 
                 data-step="${index}">
              <div class="progress-indicator" aria-label="Step ${index + 1}">
                ${index < this.currentStep ? '✓' : index + 1}
              </div>
              <div class="progress-label">${step.title}</div>
            </div>
          `).join('')}
        </div>

        <form id="${this.config.formId}" novalidate>
          ${steps.map((step, index) => `
            <div class="form-step ${index === this.currentStep ? 'active' : ''}" 
                 data-step="${index}"
                 role="tabpanel"
                 aria-labelledby="step-${index}-title">
              <h3 id="step-${index}-title" class="step-title">${step.title}</h3>
              ${step.description ? `<p class="step-description">${step.description}</p>` : ''}
              <div class="step-content">
                ${this.renderFields(step.fields)}
              </div>
            </div>
          `).join('')}

          <div class="form-actions">
            <button type="button" class="btn btn--secondary" id="prev-btn" ${this.currentStep === 0 ? 'disabled' : ''}>
              Previous
            </button>
            <div class="btn-group">
              <button type="button" class="btn btn--outline" id="save-draft-btn">
                Save Draft
              </button>
              <button type="button" class="btn btn--primary" id="next-btn">
                ${this.currentStep === steps.length - 1 ? 'Submit' : 'Next'}
              </button>
            </div>
          </div>
        </form>
      </div>
    `;

    this.form = this.container.querySelector(`#${this.config.formId}`);
  }

  renderFields(fields) {
    return fields.map(field => this.renderField(field)).join('');
  }

  renderField(field) {
    const value = this.formData[field.name] || field.defaultValue || '';
    const error = this.errors[field.name];
    const hasError = error && this.touched[field.name];

    const baseClasses = 'form-input';
    const errorClass = hasError ? 'error' : '';
    const successClass = this.touched[field.name] && !error ? 'success' : '';
    const inputClasses = `${baseClasses} ${errorClass} ${successClass}`.trim();

    const commonAttributes = `
      id="${field.name}"
      name="${field.name}"
      class="${inputClasses}"
      ${field.required ? 'required aria-required="true"' : ''}
      ${field.disabled ? 'disabled' : ''}
      ${hasError ? `aria-invalid="true" aria-describedby="${field.name}-error"` : ''}
      ${field.helpText ? `aria-describedby="${field.name}-help"` : ''}
    `;

    let inputHTML = '';

    switch (field.type) {
      case 'text':
      case 'email':
      case 'password':
      case 'url':
      case 'tel':
        inputHTML = `
          <input type="${field.type}" 
                 ${commonAttributes}
                 value="${value}"
                 placeholder="${field.placeholder || ''}"
                 ${field.minLength ? `minlength="${field.minLength}"` : ''}
                 ${field.maxLength ? `maxlength="${field.maxLength}"` : ''}
                 ${field.pattern ? `pattern="${field.pattern}"` : ''}>
        `;
        break;

      case 'number':
        inputHTML = `
          <input type="number" 
                 ${commonAttributes}
                 value="${value}"
                 placeholder="${field.placeholder || ''}"
                 ${field.min !== undefined ? `min="${field.min}"` : ''}
                 ${field.max !== undefined ? `max="${field.max}"` : ''}
                 ${field.step ? `step="${field.step}"` : ''}>
        `;
        break;

      case 'textarea':
        inputHTML = `
          <textarea ${commonAttributes.replace('form-input', 'form-input form-textarea')}
                    placeholder="${field.placeholder || ''}"
                    ${field.rows ? `rows="${field.rows}"` : ''}>${value}</textarea>
        `;
        break;

      case 'select':
        inputHTML = `
          <select ${commonAttributes.replace('form-input', 'form-input form-select')}>
            ${field.placeholder ? `<option value="">${field.placeholder}</option>` : ''}
            ${field.options.map(option => `
              <option value="${option.value}" ${value === option.value ? 'selected' : ''}>
                ${option.label}
              </option>
            `).join('')}
          </select>
        `;
        break;

      case 'checkbox-group':
        inputHTML = `
          <div class="checkbox-group" role="group" aria-labelledby="${field.name}-label">
            ${field.options.map((option, index) => `
              <div class="checkbox-item">
                <input type="checkbox" 
                       id="${field.name}-${index}"
                       name="${field.name}"
                       value="${option.value}"
                       class="form-checkbox"
                       ${Array.isArray(value) && value.includes(option.value) ? 'checked' : ''}>
                <label for="${field.name}-${index}">${option.label}</label>
              </div>
            `).join('')}
          </div>
        `;
        break;

      case 'radio-group':
        inputHTML = `
          <div class="radio-group" role="radiogroup" aria-labelledby="${field.name}-label">
            ${field.options.map((option, index) => `
              <div class="radio-item">
                <input type="radio" 
                       id="${field.name}-${index}"
                       name="${field.name}"
                       value="${option.value}"
                       class="form-radio"
                       ${value === option.value ? 'checked' : ''}>
                <label for="${field.name}-${index}">${option.label}</label>
              </div>
            `).join('')}
          </div>
        `;
        break;

      case 'date':
        inputHTML = `
          <input type="date" 
                 ${commonAttributes}
                 value="${value}">
        `;
        break;

      case 'file':
        inputHTML = `<div class="file-upload-placeholder" data-field="${field.name}"></div>`;
        break;

      case 'date-range':
        inputHTML = `<div class="date-range-placeholder" data-field="${field.name}"></div>`;
        break;

      case 'multi-select':
        inputHTML = `<div class="multi-select-placeholder" data-field="${field.name}"></div>`;
        break;

      default:
        inputHTML = `
          <input type="text" 
                 ${commonAttributes}
                 value="${value}"
                 placeholder="${field.placeholder || ''}">
        `;
    }

    return `
      <div class="form-field ${field.width ? `field-${field.width}` : ''} ${field.group ? 'field-group' : ''}">
        <label for="${field.name}" id="${field.name}-label" class="form-label ${field.required ? 'required' : ''}">
          ${field.label}
        </label>
        ${inputHTML}
        ${hasError ? `
          <div class="form-error" id="${field.name}-error" role="alert" aria-live="polite">
            <span aria-hidden="true">⚠</span>
            ${error}
          </div>
        ` : ''}
        ${!hasError && this.touched[field.name] && !error ? `
          <div class="form-success">
            <span aria-hidden="true">✓</span>
            Valid
          </div>
        ` : ''}
        ${field.helpText ? `
          <div class="form-help" id="${field.name}-help">${field.helpText}</div>
        ` : ''}
      </div>
    `;
  }

  bindEvents() {
    // Navigation buttons
    const prevBtn = this.container.querySelector('#prev-btn');
    const nextBtn = this.container.querySelector('#next-btn');
    const saveDraftBtn = this.container.querySelector('#save-draft-btn');

    prevBtn?.addEventListener('click', () => this.previousStep());
    nextBtn?.addEventListener('click', () => this.nextStep());
    saveDraftBtn?.addEventListener('click', () => this.saveDraft());

    // Form field events
    this.form.addEventListener('input', this.parent.debounce((event) => {
      this.handleFieldChange(event);
    }, this.parent.options.debounceMs));

    this.form.addEventListener('blur', (event) => {
      this.handleFieldBlur(event);
    }, true);

    // Initialize special components
    this.initializeSpecialComponents();
  }

  initializeSpecialComponents() {
    // Initialize file uploads
    this.container.querySelectorAll('.file-upload-placeholder').forEach(placeholder => {
      const fieldName = placeholder.dataset.field;
      const field = this.findField(fieldName);
      if (field) {
        this.parent.createFileUpload(placeholder, {
          ...field,
          onFileChange: (files) => {
            this.formData[fieldName] = files;
            this.validateField(fieldName);
          }
        });
      }
    });

    // Initialize date range pickers
    this.container.querySelectorAll('.date-range-placeholder').forEach(placeholder => {
      const fieldName = placeholder.dataset.field;
      const field = this.findField(fieldName);
      if (field) {
        this.parent.createDateRangePicker(placeholder, {
          ...field,
          onDateChange: (range) => {
            this.formData[fieldName] = range;
            this.validateField(fieldName);
          }
        });
      }
    });

    // Initialize multi-selects
    this.container.querySelectorAll('.multi-select-placeholder').forEach(placeholder => {
      const fieldName = placeholder.dataset.field;
      const field = this.findField(fieldName);
      if (field) {
        this.parent.createMultiSelect(placeholder, {
          ...field,
          onSelectionChange: (selected) => {
            this.formData[fieldName] = selected;
            this.validateField(fieldName);
          }
        });
      }
    });
  }

  findField(fieldName) {
    for (const step of this.config.steps) {
      const field = step.fields.find(f => f.name === fieldName);
      if (field) return field;
    }
    return null;
  }

  handleFieldChange(event) {
    const { name, value, type, checked } = event.target;
    
    if (type === 'checkbox') {
      if (!this.formData[name]) this.formData[name] = [];
      
      if (checked) {
        this.formData[name].push(value);
      } else {
        this.formData[name] = this.formData[name].filter(v => v !== value);
      }
    } else {
      this.formData[name] = value;
    }

    if (this.parent.options.validateOnChange) {
      this.validateField(name);
    }
  }

  handleFieldBlur(event) {
    const { name } = event.target;
    this.touched[name] = true;

    if (this.parent.options.validateOnBlur) {
      this.validateField(name);
    }
  }

  validateField(fieldName) {
    const field = this.findField(fieldName);
    if (!field) return;

    const value = this.formData[fieldName];
    const errors = this.parent.validateField(value, field);
    
    if (errors.length > 0) {
      this.errors[fieldName] = errors[0]; // Show first error
    } else {
      delete this.errors[fieldName];
    }

    this.updateFieldDisplay(fieldName);
  }

  updateFieldDisplay(fieldName) {
    const fieldElement = this.form.querySelector(`[name="${fieldName}"]`);
    if (!fieldElement) return;

    const hasError = this.errors[fieldName] && this.touched[fieldName];
    const isValid = this.touched[fieldName] && !this.errors[fieldName];

    fieldElement.classList.toggle('error', hasError);
    fieldElement.classList.toggle('success', isValid);
    
    fieldElement.setAttribute('aria-invalid', hasError ? 'true' : 'false');

    // Update error display
    const existingError = this.form.querySelector(`#${fieldName}-error`);
    const existingSuccess = this.form.querySelector(`#${fieldName}-success`);

    if (existingError) existingError.remove();
    if (existingSuccess) existingSuccess.remove();

    if (hasError) {
      const errorDiv = document.createElement('div');
      errorDiv.className = 'form-error';
      errorDiv.id = `${fieldName}-error`;
      errorDiv.setAttribute('role', 'alert');
      errorDiv.setAttribute('aria-live', 'polite');
      errorDiv.innerHTML = `<span aria-hidden="true">⚠</span> ${this.errors[fieldName]}`;
      fieldElement.parentNode.appendChild(errorDiv);

      if (this.parent.options.accessibility.announceErrors) {
        this.parent.announceToUser(`Error in ${fieldName}: ${this.errors[fieldName]}`);
      }
    } else if (isValid) {
      const successDiv = document.createElement('div');
      successDiv.className = 'form-success';
      successDiv.id = `${fieldName}-success`;
      successDiv.innerHTML = `<span aria-hidden="true">✓</span> Valid`;
      fieldElement.parentNode.appendChild(successDiv);
    }
  }

  validateCurrentStep() {
    const currentStepConfig = this.config.steps[this.currentStep];
    let isValid = true;

    for (const field of currentStepConfig.fields) {
      this.touched[field.name] = true;
      this.validateField(field.name);
      
      if (this.errors[field.name]) {
        isValid = false;
      }
    }

    return isValid;
  }

  nextStep() {
    if (this.currentStep === this.config.steps.length - 1) {
      this.submitForm();
      return;
    }

    if (this.validateCurrentStep()) {
      this.currentStep++;
      this.updateDisplay();
      this.updateProgress();
      this.parent.announceToUser(`Moved to step ${this.currentStep + 1}: ${this.config.steps[this.currentStep].title}`);
    } else {
      this.parent.announceToUser('Please fix the errors before continuing');
    }
  }

  previousStep() {
    if (this.currentStep > 0) {
      this.currentStep--;
      this.updateDisplay();
      this.updateProgress();
      this.parent.announceToUser(`Moved to step ${this.currentStep + 1}: ${this.config.steps[this.currentStep].title}`);
    }
  }

  updateDisplay() {
    // Update step visibility
    this.container.querySelectorAll('.form-step').forEach((step, index) => {
      step.classList.toggle('active', index === this.currentStep);
      if (index < this.currentStep) {
        step.classList.add('previous');
      } else {
        step.classList.remove('previous');
      }
    });

    // Update buttons
    const prevBtn = this.container.querySelector('#prev-btn');
    const nextBtn = this.container.querySelector('#next-btn');

    prevBtn.disabled = this.currentStep === 0;
    nextBtn.textContent = this.currentStep === this.config.steps.length - 1 ? 'Submit' : 'Next';

    // Focus management
    const currentStepElement = this.container.querySelector('.form-step.active');
    const firstInput = currentStepElement.querySelector('input, select, textarea');
    if (firstInput) {
      firstInput.focus();
    }
  }

  updateProgress() {
    // Update progress indicators
    this.container.querySelectorAll('.progress-step').forEach((step, index) => {
      step.classList.toggle('active', index === this.currentStep);
      step.classList.toggle('completed', index < this.currentStep);
    });

    // Update progress bar ARIA
    const progressBar = this.container.querySelector('.form-progress');
    progressBar.setAttribute('aria-valuenow', this.currentStep + 1);
  }

  submitForm() {
    if (this.validateCurrentStep()) {
      // Call submit callback if provided
      if (this.config.onSubmit) {
        this.config.onSubmit(this.formData, this);
      }

      this.parent.announceToUser('Form submitted successfully');
    }
  }

  saveDraft() {
    if (this.config.onSaveDraft) {
      this.config.onSaveDraft(this.formData, this);
    }

    this.parent.announceToUser('Draft saved');
  }

  destroy() {
    if (this.container) {
      this.container.innerHTML = '';
    }
  }
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
  module.exports = AdvancedFormComponents;
}

// Global access
window.AdvancedFormComponents = AdvancedFormComponents;
