/**
 * Advanced Form Components for Pynomaly
 * Multi-step forms, dynamic validation, file upload with progress, and form persistence
 * with accessibility features and real-time validation
 */

/**
 * Base Form Component
 * Provides common functionality for all form types
 */
class BaseForm {
  constructor(container, options = {}) {
    this.container =
      typeof container === "string"
        ? document.querySelector(container)
        : container;
    this.options = {
      accessibility: true,
      realTimeValidation: true,
      persistence: true,
      submitOnEnter: true,
      resetOnSubmit: false,
      autoSave: false,
      autoSaveInterval: 30000,
      ...options,
    };

    this.formId = this.options.id || `form-${Date.now()}`;
    this.validators = new Map();
    this.data = {};
    this.errors = {};
    this.touched = {};
    this.isValid = true;
    this.isSubmitting = false;
    this.isDirty = false;

    this.autoSaveTimer = null;
    this.validationDebounceTimer = null;

    this.init();
  }

  init() {
    this.setupForm();
    this.setupValidation();
    this.setupEventListeners();
    this.setupAccessibility();

    if (this.options.persistence) {
      this.loadSavedData();
    }

    if (this.options.autoSave) {
      this.startAutoSave();
    }

    // Register with form store
    if (window.formStore) {
      window.formStore.getState().createForm(this.formId, this.data);
    }
  }

  setupForm() {
    this.form =
      this.container.querySelector("form") || this.createFormElement();
    this.form.setAttribute("novalidate", ""); // Use custom validation
    this.form.id = this.formId;
  }

  createFormElement() {
    const form = document.createElement("form");
    this.container.appendChild(form);
    return form;
  }

  setupValidation() {
    // Add default validators
    this.addValidator("required", (value, field) => {
      if (!value || (typeof value === "string" && value.trim() === "")) {
        return `${field.label || field.name} is required`;
      }
      return null;
    });

    this.addValidator("email", (value) => {
      const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
      if (value && !emailRegex.test(value)) {
        return "Please enter a valid email address";
      }
      return null;
    });

    this.addValidator("minLength", (value, field) => {
      if (value && value.length < field.minLength) {
        return `Must be at least ${field.minLength} characters long`;
      }
      return null;
    });

    this.addValidator("maxLength", (value, field) => {
      if (value && value.length > field.maxLength) {
        return `Must be no more than ${field.maxLength} characters long`;
      }
      return null;
    });

    this.addValidator("pattern", (value, field) => {
      if (value && field.pattern && !new RegExp(field.pattern).test(value)) {
        return field.patternMessage || "Invalid format";
      }
      return null;
    });

    this.addValidator("number", (value, field) => {
      if (value && isNaN(value)) {
        return "Must be a valid number";
      }
      if (value && field.min !== undefined && parseFloat(value) < field.min) {
        return `Must be at least ${field.min}`;
      }
      if (value && field.max !== undefined && parseFloat(value) > field.max) {
        return `Must be no more than ${field.max}`;
      }
      return null;
    });
  }

  setupEventListeners() {
    // Form submission
    this.form.addEventListener("submit", (e) => {
      e.preventDefault();
      this.handleSubmit();
    });

    // Field changes
    this.form.addEventListener("input", (e) => {
      this.handleFieldChange(e.target);
    });

    this.form.addEventListener(
      "blur",
      (e) => {
        this.handleFieldBlur(e.target);
      },
      true,
    );

    // Keyboard navigation
    this.form.addEventListener("keydown", (e) => {
      this.handleKeyDown(e);
    });

    // Prevent data loss
    window.addEventListener("beforeunload", (e) => {
      if (this.isDirty && !this.isSubmitting) {
        e.preventDefault();
        e.returnValue =
          "You have unsaved changes. Are you sure you want to leave?";
        return e.returnValue;
      }
    });
  }

  setupAccessibility() {
    if (!this.options.accessibility) return;

    // Add form role and aria-label
    this.form.setAttribute("role", "form");
    if (this.options.ariaLabel) {
      this.form.setAttribute("aria-label", this.options.ariaLabel);
    }

    // Add live region for announcements
    if (!document.getElementById("form-announcer")) {
      const announcer = document.createElement("div");
      announcer.id = "form-announcer";
      announcer.className = "sr-only";
      announcer.setAttribute("aria-live", "polite");
      announcer.setAttribute("aria-atomic", "true");
      document.body.appendChild(announcer);
    }
  }

  addValidator(name, validator) {
    this.validators.set(name, validator);
  }

  validateField(fieldName, value) {
    const field = this.getFieldConfig(fieldName);
    if (!field) return null;

    const errors = [];

    // Run all applicable validators
    if (field.required) {
      const error = this.validators.get("required")(value, field);
      if (error) errors.push(error);
    }

    if (field.type === "email") {
      const error = this.validators.get("email")(value, field);
      if (error) errors.push(error);
    }

    if (field.type === "number") {
      const error = this.validators.get("number")(value, field);
      if (error) errors.push(error);
    }

    if (field.minLength) {
      const error = this.validators.get("minLength")(value, field);
      if (error) errors.push(error);
    }

    if (field.maxLength) {
      const error = this.validators.get("maxLength")(value, field);
      if (error) errors.push(error);
    }

    if (field.pattern) {
      const error = this.validators.get("pattern")(value, field);
      if (error) errors.push(error);
    }

    // Custom validator
    if (field.validator && typeof field.validator === "function") {
      const error = field.validator(value, field, this.data);
      if (error) errors.push(error);
    }

    return errors.length > 0 ? errors[0] : null;
  }

  validateForm() {
    const errors = {};
    let isValid = true;

    Object.keys(this.data).forEach((fieldName) => {
      const error = this.validateField(fieldName, this.data[fieldName]);
      if (error) {
        errors[fieldName] = error;
        isValid = false;
      }
    });

    this.errors = errors;
    this.isValid = isValid;

    return isValid;
  }

  handleFieldChange(field) {
    const fieldName = field.name;
    const value = this.getFieldValue(field);

    this.data[fieldName] = value;
    this.isDirty = true;

    // Real-time validation with debouncing
    if (this.options.realTimeValidation) {
      clearTimeout(this.validationDebounceTimer);
      this.validationDebounceTimer = setTimeout(() => {
        this.validateAndUpdateField(fieldName);
      }, 300);
    }

    // Update form store
    if (window.formStore) {
      window.formStore
        .getState()
        .updateFormField(this.formId, fieldName, value);
    }

    // Auto-save
    if (this.options.autoSave) {
      this.saveData();
    }

    // Emit change event
    this.emit("fieldChange", { field: fieldName, value, data: this.data });
  }

  handleFieldBlur(field) {
    const fieldName = field.name;
    this.touched[fieldName] = true;

    // Validate on blur if not real-time validation
    if (!this.options.realTimeValidation) {
      this.validateAndUpdateField(fieldName);
    }

    this.emit("fieldBlur", { field: fieldName, data: this.data });
  }

  validateAndUpdateField(fieldName) {
    const error = this.validateField(fieldName, this.data[fieldName]);

    if (error) {
      this.errors[fieldName] = error;
    } else {
      delete this.errors[fieldName];
    }

    this.updateFieldErrorDisplay(fieldName, error);
    this.isValid = Object.keys(this.errors).length === 0;
  }

  updateFieldErrorDisplay(fieldName, error) {
    const field = this.form.querySelector(`[name="${fieldName}"]`);
    if (!field) return;

    const errorElement = document.getElementById(`${fieldName}-error`);

    if (error) {
      field.setAttribute("aria-invalid", "true");
      field.classList.add("error");

      if (errorElement) {
        errorElement.textContent = error;
        errorElement.style.display = "block";
      } else {
        this.createErrorElement(field, fieldName, error);
      }
    } else {
      field.setAttribute("aria-invalid", "false");
      field.classList.remove("error");

      if (errorElement) {
        errorElement.style.display = "none";
      }
    }
  }

  createErrorElement(field, fieldName, error) {
    const errorElement = document.createElement("div");
    errorElement.id = `${fieldName}-error`;
    errorElement.className = "field-error";
    errorElement.textContent = error;
    errorElement.setAttribute("role", "alert");
    errorElement.setAttribute("aria-live", "polite");

    field.setAttribute("aria-describedby", errorElement.id);
    field.parentNode.insertBefore(errorElement, field.nextSibling);
  }

  getFieldValue(field) {
    switch (field.type) {
      case "checkbox":
        return field.checked;
      case "radio":
        return field.checked ? field.value : null;
      case "file":
        return field.files;
      case "number":
        return field.value ? parseFloat(field.value) : null;
      default:
        return field.value;
    }
  }

  getFieldConfig(fieldName) {
    // Override in subclasses to provide field configuration
    return null;
  }

  handleKeyDown(e) {
    if (e.key === "Enter" && this.options.submitOnEnter) {
      if (e.target.tagName !== "TEXTAREA") {
        e.preventDefault();
        this.handleSubmit();
      }
    }
  }

  async handleSubmit() {
    if (this.isSubmitting) return;

    this.isSubmitting = true;
    this.emit("beforeSubmit", { data: this.data });

    // Validate entire form
    if (!this.validateForm()) {
      this.isSubmitting = false;
      this.focusFirstError();
      this.announceErrors();
      this.emit("validationFailed", { errors: this.errors });
      return;
    }

    try {
      const result = await this.submit(this.data);

      if (this.options.resetOnSubmit) {
        this.reset();
      } else {
        this.isDirty = false;
      }

      if (this.options.persistence) {
        this.clearSavedData();
      }

      this.announceSuccess();
      this.emit("submitSuccess", { data: this.data, result });
    } catch (error) {
      console.error("Form submission error:", error);
      this.emit("submitError", { error, data: this.data });
      this.announceError("Form submission failed. Please try again.");
    } finally {
      this.isSubmitting = false;
    }
  }

  async submit(data) {
    // Override in subclasses
    throw new Error("submit() method must be implemented by subclass");
  }

  reset() {
    this.form.reset();
    this.data = {};
    this.errors = {};
    this.touched = {};
    this.isDirty = false;
    this.isValid = true;

    // Clear error displays
    this.form.querySelectorAll(".field-error").forEach((error) => {
      error.style.display = "none";
    });

    this.form.querySelectorAll('[aria-invalid="true"]').forEach((field) => {
      field.setAttribute("aria-invalid", "false");
      field.classList.remove("error");
    });

    this.emit("reset");
  }

  focusFirstError() {
    const firstErrorField = Object.keys(this.errors)[0];
    if (firstErrorField) {
      const field = this.form.querySelector(`[name="${firstErrorField}"]`);
      if (field) {
        field.focus();
      }
    }
  }

  announceErrors() {
    const errorCount = Object.keys(this.errors).length;
    const message = `Form has ${errorCount} error${errorCount !== 1 ? "s" : ""}. Please correct the highlighted fields.`;
    this.announce(message);
  }

  announceSuccess() {
    this.announce("Form submitted successfully");
  }

  announceError(message) {
    this.announce(message);
  }

  announce(message) {
    const announcer = document.getElementById("form-announcer");
    if (announcer) {
      announcer.textContent = message;
    }
  }

  // Persistence methods
  saveData() {
    if (!this.options.persistence) return;

    try {
      const saveData = {
        data: this.data,
        timestamp: Date.now(),
      };
      localStorage.setItem(`form-${this.formId}`, JSON.stringify(saveData));
    } catch (error) {
      console.warn("Failed to save form data:", error);
    }
  }

  loadSavedData() {
    if (!this.options.persistence) return;

    try {
      const saved = localStorage.getItem(`form-${this.formId}`);
      if (saved) {
        const { data, timestamp } = JSON.parse(saved);

        // Only load if saved within last 24 hours
        if (Date.now() - timestamp < 24 * 60 * 60 * 1000) {
          this.data = data;
          this.populateForm();
        }
      }
    } catch (error) {
      console.warn("Failed to load saved form data:", error);
    }
  }

  clearSavedData() {
    if (!this.options.persistence) return;

    try {
      localStorage.removeItem(`form-${this.formId}`);
    } catch (error) {
      console.warn("Failed to clear saved form data:", error);
    }
  }

  populateForm() {
    Object.entries(this.data).forEach(([fieldName, value]) => {
      const field = this.form.querySelector(`[name="${fieldName}"]`);
      if (field) {
        if (field.type === "checkbox") {
          field.checked = Boolean(value);
        } else if (field.type === "radio") {
          if (field.value === value) {
            field.checked = true;
          }
        } else {
          field.value = value || "";
        }
      }
    });
  }

  startAutoSave() {
    this.autoSaveTimer = setInterval(() => {
      if (this.isDirty) {
        this.saveData();
      }
    }, this.options.autoSaveInterval);
  }

  stopAutoSave() {
    if (this.autoSaveTimer) {
      clearInterval(this.autoSaveTimer);
      this.autoSaveTimer = null;
    }
  }

  // Event system
  emit(eventName, data) {
    const event = new CustomEvent(`form:${eventName}`, {
      detail: { form: this, ...data },
    });
    this.container.dispatchEvent(event);
  }

  destroy() {
    this.stopAutoSave();

    if (this.validationDebounceTimer) {
      clearTimeout(this.validationDebounceTimer);
    }

    // Remove from form store
    if (window.formStore) {
      window.formStore.getState().removeForm(this.formId);
    }
  }
}

/**
 * Multi-Step Form Component
 * Wizard-style forms with step navigation and validation
 */
class MultiStepForm extends BaseForm {
  constructor(container, options = {}) {
    super(container, {
      showProgress: true,
      allowStepSkipping: false,
      validateOnStepChange: true,
      persistCurrentStep: true,
      ...options,
    });

    this.steps = this.options.steps || [];
    this.currentStep = 0;
    this.stepData = {};
    this.stepValidations = new Map();

    this.initMultiStep();
  }

  initMultiStep() {
    this.createStepContainer();
    this.createProgressIndicator();
    this.createNavigationButtons();
    this.loadCurrentStep();
    this.showStep(this.currentStep);
  }

  createStepContainer() {
    this.stepContainer = document.createElement("div");
    this.stepContainer.className = "step-container";
    this.form.appendChild(this.stepContainer);
  }

  createProgressIndicator() {
    if (!this.options.showProgress) return;

    this.progressContainer = document.createElement("div");
    this.progressContainer.className = "step-progress";
    this.progressContainer.setAttribute("role", "progressbar");
    this.progressContainer.setAttribute("aria-label", "Form progress");
    this.progressContainer.setAttribute("aria-valuemin", "0");
    this.progressContainer.setAttribute(
      "aria-valuemax",
      this.steps.length.toString(),
    );

    this.form.insertBefore(this.progressContainer, this.stepContainer);

    this.createProgressSteps();
  }

  createProgressSteps() {
    const progressList = document.createElement("ol");
    progressList.className = "progress-steps";

    this.steps.forEach((step, index) => {
      const stepItem = document.createElement("li");
      stepItem.className = "progress-step";
      stepItem.setAttribute("data-step", index.toString());

      const stepButton = document.createElement("button");
      stepButton.type = "button";
      stepButton.className = "progress-step-button";
      stepButton.setAttribute("aria-label", `Step ${index + 1}: ${step.title}`);
      stepButton.textContent = (index + 1).toString();

      if (this.options.allowStepSkipping) {
        stepButton.addEventListener("click", () => this.goToStep(index));
      } else {
        stepButton.disabled = index > this.currentStep;
      }

      const stepTitle = document.createElement("span");
      stepTitle.className = "progress-step-title";
      stepTitle.textContent = step.title;

      stepItem.appendChild(stepButton);
      stepItem.appendChild(stepTitle);
      progressList.appendChild(stepItem);
    });

    this.progressContainer.appendChild(progressList);
  }

  createNavigationButtons() {
    this.navigationContainer = document.createElement("div");
    this.navigationContainer.className = "step-navigation";

    this.prevButton = document.createElement("button");
    this.prevButton.type = "button";
    this.prevButton.className = "btn btn-secondary step-prev";
    this.prevButton.textContent = "Previous";
    this.prevButton.addEventListener("click", () => this.previousStep());

    this.nextButton = document.createElement("button");
    this.nextButton.type = "button";
    this.nextButton.className = "btn btn-primary step-next";
    this.nextButton.textContent = "Next";
    this.nextButton.addEventListener("click", () => this.nextStep());

    this.submitButton = document.createElement("button");
    this.submitButton.type = "submit";
    this.submitButton.className = "btn btn-primary step-submit";
    this.submitButton.textContent = "Submit";

    this.navigationContainer.appendChild(this.prevButton);
    this.navigationContainer.appendChild(this.nextButton);
    this.navigationContainer.appendChild(this.submitButton);

    this.form.appendChild(this.navigationContainer);
  }

  showStep(stepIndex) {
    if (stepIndex < 0 || stepIndex >= this.steps.length) return;

    // Hide all steps
    this.stepContainer.querySelectorAll(".form-step").forEach((step) => {
      step.style.display = "none";
      step.setAttribute("aria-hidden", "true");
    });

    // Show current step
    const currentStepElement = this.stepContainer.querySelector(
      `[data-step="${stepIndex}"]`,
    );
    if (currentStepElement) {
      currentStepElement.style.display = "block";
      currentStepElement.setAttribute("aria-hidden", "false");

      // Focus first focusable element
      const firstInput = currentStepElement.querySelector(
        "input, select, textarea",
      );
      if (firstInput) {
        firstInput.focus();
      }
    } else {
      this.createStepElement(stepIndex);
    }

    this.updateProgress();
    this.updateNavigation();
    this.announceStepChange();
  }

  createStepElement(stepIndex) {
    const step = this.steps[stepIndex];
    const stepElement = document.createElement("div");
    stepElement.className = "form-step";
    stepElement.setAttribute("data-step", stepIndex.toString());
    stepElement.setAttribute("role", "tabpanel");
    stepElement.setAttribute("aria-labelledby", `step-${stepIndex}-title`);

    // Step title
    const title = document.createElement("h2");
    title.id = `step-${stepIndex}-title`;
    title.className = "step-title";
    title.textContent = step.title;
    stepElement.appendChild(title);

    // Step description
    if (step.description) {
      const description = document.createElement("p");
      description.className = "step-description";
      description.textContent = step.description;
      stepElement.appendChild(description);
    }

    // Step fields
    if (step.fields) {
      const fieldsContainer = document.createElement("div");
      fieldsContainer.className = "step-fields";

      step.fields.forEach((field) => {
        const fieldElement = this.createFieldElement(field);
        fieldsContainer.appendChild(fieldElement);
      });

      stepElement.appendChild(fieldsContainer);
    }

    this.stepContainer.appendChild(stepElement);
  }

  createFieldElement(field) {
    const fieldContainer = document.createElement("div");
    fieldContainer.className = "form-group";

    // Label
    const label = document.createElement("label");
    label.setAttribute("for", field.name);
    label.className = "form-label";
    label.textContent = field.label;
    if (field.required) {
      label.innerHTML +=
        ' <span class="required" aria-label="required">*</span>';
    }
    fieldContainer.appendChild(label);

    // Input
    let input;
    switch (field.type) {
      case "textarea":
        input = document.createElement("textarea");
        break;
      case "select":
        input = document.createElement("select");
        field.options?.forEach((option) => {
          const optionElement = document.createElement("option");
          optionElement.value = option.value;
          optionElement.textContent = option.label;
          input.appendChild(optionElement);
        });
        break;
      default:
        input = document.createElement("input");
        input.type = field.type || "text";
    }

    input.id = field.name;
    input.name = field.name;
    input.className = "form-input";

    if (field.placeholder) {
      input.placeholder = field.placeholder;
    }

    if (field.required) {
      input.setAttribute("aria-required", "true");
    }

    fieldContainer.appendChild(input);

    // Help text
    if (field.help) {
      const helpText = document.createElement("div");
      helpText.className = "form-help";
      helpText.id = `${field.name}-help`;
      helpText.textContent = field.help;
      input.setAttribute("aria-describedby", helpText.id);
      fieldContainer.appendChild(helpText);
    }

    return fieldContainer;
  }

  updateProgress() {
    if (!this.options.showProgress) return;

    const progressValue = (
      ((this.currentStep + 1) / this.steps.length) *
      100
    ).toFixed(0);
    this.progressContainer.setAttribute(
      "aria-valuenow",
      (this.currentStep + 1).toString(),
    );
    this.progressContainer.setAttribute(
      "aria-valuetext",
      `Step ${this.currentStep + 1} of ${this.steps.length}`,
    );

    // Update visual progress
    this.progressContainer
      .querySelectorAll(".progress-step")
      .forEach((step, index) => {
        step.classList.remove("current", "completed");

        if (index < this.currentStep) {
          step.classList.add("completed");
        } else if (index === this.currentStep) {
          step.classList.add("current");
        }
      });
  }

  updateNavigation() {
    this.prevButton.style.display =
      this.currentStep > 0 ? "inline-block" : "none";
    this.nextButton.style.display =
      this.currentStep < this.steps.length - 1 ? "inline-block" : "none";
    this.submitButton.style.display =
      this.currentStep === this.steps.length - 1 ? "inline-block" : "none";

    // Update button states
    this.prevButton.disabled = this.isSubmitting;
    this.nextButton.disabled = this.isSubmitting;
    this.submitButton.disabled = this.isSubmitting;
  }

  announceStepChange() {
    const step = this.steps[this.currentStep];
    const message = `Now on step ${this.currentStep + 1} of ${this.steps.length}: ${step.title}`;
    this.announce(message);
  }

  validateCurrentStep() {
    const step = this.steps[this.currentStep];
    if (!step.fields) return true;

    let isValid = true;
    const errors = {};

    step.fields.forEach((field) => {
      const value = this.data[field.name];
      const error = this.validateField(field.name, value);

      if (error) {
        errors[field.name] = error;
        isValid = false;
      }
    });

    if (!isValid) {
      this.errors = { ...this.errors, ...errors };
      this.focusFirstError();
    }

    return isValid;
  }

  nextStep() {
    if (this.options.validateOnStepChange && !this.validateCurrentStep()) {
      return;
    }

    if (this.currentStep < this.steps.length - 1) {
      this.currentStep++;
      this.showStep(this.currentStep);

      if (this.options.persistCurrentStep) {
        this.saveCurrentStep();
      }
    }
  }

  previousStep() {
    if (this.currentStep > 0) {
      this.currentStep--;
      this.showStep(this.currentStep);

      if (this.options.persistCurrentStep) {
        this.saveCurrentStep();
      }
    }
  }

  goToStep(stepIndex) {
    if (!this.options.allowStepSkipping) return;

    if (stepIndex >= 0 && stepIndex < this.steps.length) {
      this.currentStep = stepIndex;
      this.showStep(this.currentStep);
    }
  }

  saveCurrentStep() {
    try {
      localStorage.setItem(
        `form-step-${this.formId}`,
        this.currentStep.toString(),
      );
    } catch (error) {
      console.warn("Failed to save current step:", error);
    }
  }

  loadCurrentStep() {
    try {
      const saved = localStorage.getItem(`form-step-${this.formId}`);
      if (saved) {
        this.currentStep = parseInt(saved, 10);
        if (this.currentStep >= this.steps.length) {
          this.currentStep = 0;
        }
      }
    } catch (error) {
      console.warn("Failed to load current step:", error);
    }
  }

  getFieldConfig(fieldName) {
    for (const step of this.steps) {
      if (step.fields) {
        const field = step.fields.find((f) => f.name === fieldName);
        if (field) return field;
      }
    }
    return null;
  }

  async submit(data) {
    // Validate all steps
    let isValid = true;
    for (let i = 0; i < this.steps.length; i++) {
      const step = this.steps[i];
      if (step.fields) {
        step.fields.forEach((field) => {
          const error = this.validateField(field.name, data[field.name]);
          if (error) {
            this.errors[field.name] = error;
            isValid = false;
          }
        });
      }
    }

    if (!isValid) {
      throw new Error("Form validation failed");
    }

    // Call the provided submit handler
    if (this.options.onSubmit) {
      return await this.options.onSubmit(data);
    }

    // Default submission
    return { success: true, data };
  }

  reset() {
    super.reset();
    this.currentStep = 0;
    this.stepData = {};
    this.showStep(0);

    if (this.options.persistCurrentStep) {
      localStorage.removeItem(`form-step-${this.formId}`);
    }
  }
}

/**
 * File Upload Form Component
 * Advanced file upload with progress, validation, and drag-drop
 */
class FileUploadForm extends BaseForm {
  constructor(container, options = {}) {
    super(container, {
      multiple: false,
      maxFiles: 10,
      maxFileSize: 10 * 1024 * 1024, // 10MB
      acceptedTypes: ["*/*"],
      dragDrop: true,
      showProgress: true,
      showPreview: true,
      chunkSize: 1024 * 1024, // 1MB chunks
      ...options,
    });

    this.files = [];
    this.uploads = new Map();
    this.initFileUpload();
  }

  initFileUpload() {
    this.createUploadArea();
    this.createFileList();
    this.createProgressArea();
    this.setupFileValidation();
  }

  createUploadArea() {
    this.uploadArea = document.createElement("div");
    this.uploadArea.className = "file-upload-area";
    this.uploadArea.setAttribute("role", "button");
    this.uploadArea.setAttribute("tabindex", "0");
    this.uploadArea.setAttribute("aria-label", "Click or drag files to upload");

    const uploadContent = document.createElement("div");
    uploadContent.className = "upload-content";
    uploadContent.innerHTML = `
      <div class="upload-icon">📁</div>
      <div class="upload-text">
        <p>Click to select files or drag and drop</p>
        <p class="upload-help">
          ${this.options.acceptedTypes.join(", ")} • 
          Max ${this.formatFileSize(this.options.maxFileSize)}
        </p>
      </div>
    `;

    this.fileInput = document.createElement("input");
    this.fileInput.type = "file";
    this.fileInput.name = "files";
    this.fileInput.className = "file-input";
    this.fileInput.style.display = "none";

    if (this.options.multiple) {
      this.fileInput.multiple = true;
    }

    if (
      this.options.acceptedTypes.length > 0 &&
      !this.options.acceptedTypes.includes("*/*")
    ) {
      this.fileInput.accept = this.options.acceptedTypes.join(",");
    }

    this.uploadArea.appendChild(uploadContent);
    this.uploadArea.appendChild(this.fileInput);
    this.form.appendChild(this.uploadArea);

    this.setupUploadAreaEvents();
  }

  setupUploadAreaEvents() {
    // Click to select files
    this.uploadArea.addEventListener("click", () => {
      this.fileInput.click();
    });

    // Keyboard accessibility
    this.uploadArea.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        this.fileInput.click();
      }
    });

    // File selection
    this.fileInput.addEventListener("change", (e) => {
      this.handleFileSelection(e.target.files);
    });

    // Drag and drop
    if (this.options.dragDrop) {
      this.setupDragDrop();
    }
  }

  setupDragDrop() {
    ["dragenter", "dragover", "dragleave", "drop"].forEach((eventName) => {
      this.uploadArea.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
      });
    });

    ["dragenter", "dragover"].forEach((eventName) => {
      this.uploadArea.addEventListener(eventName, () => {
        this.uploadArea.classList.add("drag-over");
      });
    });

    ["dragleave", "drop"].forEach((eventName) => {
      this.uploadArea.addEventListener(eventName, () => {
        this.uploadArea.classList.remove("drag-over");
      });
    });

    this.uploadArea.addEventListener("drop", (e) => {
      const files = e.dataTransfer.files;
      this.handleFileSelection(files);
    });
  }

  createFileList() {
    this.fileListContainer = document.createElement("div");
    this.fileListContainer.className = "file-list";
    this.fileListContainer.setAttribute("role", "list");
    this.fileListContainer.setAttribute("aria-label", "Selected files");
    this.form.appendChild(this.fileListContainer);
  }

  createProgressArea() {
    if (!this.options.showProgress) return;

    this.progressArea = document.createElement("div");
    this.progressArea.className = "upload-progress-area";
    this.progressArea.style.display = "none";
    this.form.appendChild(this.progressArea);
  }

  setupFileValidation() {
    this.addValidator("fileType", (file) => {
      if (this.options.acceptedTypes.includes("*/*")) return null;

      const isValid = this.options.acceptedTypes.some((type) => {
        if (type.endsWith("/*")) {
          return file.type.startsWith(type.slice(0, -1));
        }
        return (
          file.type === type ||
          file.name.toLowerCase().endsWith(type.toLowerCase())
        );
      });

      if (!isValid) {
        return `File type not supported. Accepted types: ${this.options.acceptedTypes.join(", ")}`;
      }
      return null;
    });

    this.addValidator("fileSize", (file) => {
      if (file.size > this.options.maxFileSize) {
        return `File size exceeds limit of ${this.formatFileSize(this.options.maxFileSize)}`;
      }
      return null;
    });
  }

  handleFileSelection(fileList) {
    const newFiles = Array.from(fileList);

    // Validate file count
    if (!this.options.multiple && newFiles.length > 1) {
      this.announce("Only one file can be selected");
      return;
    }

    if (this.files.length + newFiles.length > this.options.maxFiles) {
      this.announce(`Maximum ${this.options.maxFiles} files allowed`);
      return;
    }

    // Validate each file
    const validFiles = [];
    const errors = [];

    newFiles.forEach((file) => {
      const typeError = this.validators.get("fileType")(file);
      const sizeError = this.validators.get("fileSize")(file);

      if (typeError || sizeError) {
        errors.push(`${file.name}: ${typeError || sizeError}`);
      } else {
        validFiles.push(file);
      }
    });

    if (errors.length > 0) {
      this.announce(`File validation errors: ${errors.join("; ")}`);
      return;
    }

    // Add valid files
    validFiles.forEach((file) => {
      const fileData = {
        id: Date.now() + Math.random(),
        file: file,
        name: file.name,
        size: file.size,
        type: file.type,
        status: "selected",
        progress: 0,
        error: null,
      };

      this.files.push(fileData);
      this.createFileItem(fileData);
    });

    this.updateFileList();
    this.emit("filesSelected", { files: validFiles });
  }

  createFileItem(fileData) {
    const fileItem = document.createElement("div");
    fileItem.className = "file-item";
    fileItem.setAttribute("data-file-id", fileData.id);
    fileItem.setAttribute("role", "listitem");

    fileItem.innerHTML = `
      <div class="file-icon">${this.getFileIcon(fileData.type)}</div>
      <div class="file-info">
        <div class="file-name">${fileData.name}</div>
        <div class="file-size">${this.formatFileSize(fileData.size)}</div>
        <div class="file-status">${fileData.status}</div>
      </div>
      <div class="file-progress">
        <div class="progress-bar">
          <div class="progress-fill" style="width: ${fileData.progress}%"></div>
        </div>
        <div class="progress-text">${fileData.progress}%</div>
      </div>
      <div class="file-actions">
        <button type="button" class="btn-remove" aria-label="Remove ${fileData.name}">✕</button>
      </div>
    `;

    // Remove button
    const removeButton = fileItem.querySelector(".btn-remove");
    removeButton.addEventListener("click", () => {
      this.removeFile(fileData.id);
    });

    this.fileListContainer.appendChild(fileItem);
  }

  updateFileItem(fileData) {
    const fileItem = this.fileListContainer.querySelector(
      `[data-file-id="${fileData.id}"]`,
    );
    if (!fileItem) return;

    const statusElement = fileItem.querySelector(".file-status");
    const progressFill = fileItem.querySelector(".progress-fill");
    const progressText = fileItem.querySelector(".progress-text");

    if (statusElement) statusElement.textContent = fileData.status;
    if (progressFill) progressFill.style.width = `${fileData.progress}%`;
    if (progressText) progressText.textContent = `${fileData.progress}%`;

    // Update accessibility
    fileItem.setAttribute(
      "aria-label",
      `${fileData.name}, ${fileData.status}, ${fileData.progress}% complete`,
    );
  }

  removeFile(fileId) {
    const index = this.files.findIndex((f) => f.id === fileId);
    if (index === -1) return;

    const fileData = this.files[index];

    // Cancel upload if in progress
    if (this.uploads.has(fileId)) {
      const upload = this.uploads.get(fileId);
      if (upload.xhr) {
        upload.xhr.abort();
      }
      this.uploads.delete(fileId);
    }

    // Remove from array
    this.files.splice(index, 1);

    // Remove from DOM
    const fileItem = this.fileListContainer.querySelector(
      `[data-file-id="${fileId}"]`,
    );
    if (fileItem) {
      fileItem.remove();
    }

    this.updateFileList();
    this.emit("fileRemoved", { file: fileData });
  }

  updateFileList() {
    // Show/hide file list
    this.fileListContainer.style.display =
      this.files.length > 0 ? "block" : "none";

    // Update upload area state
    if (this.files.length >= this.options.maxFiles) {
      this.uploadArea.classList.add("disabled");
      this.uploadArea.setAttribute("aria-disabled", "true");
    } else {
      this.uploadArea.classList.remove("disabled");
      this.uploadArea.setAttribute("aria-disabled", "false");
    }
  }

  async startUpload() {
    if (this.files.length === 0) {
      this.announce("No files selected for upload");
      return;
    }

    const filesToUpload = this.files.filter((f) => f.status === "selected");
    if (filesToUpload.length === 0) {
      this.announce("No files ready for upload");
      return;
    }

    this.isSubmitting = true;
    this.emit("uploadStart", { files: filesToUpload });

    try {
      const uploadPromises = filesToUpload.map((fileData) =>
        this.uploadFile(fileData),
      );
      const results = await Promise.allSettled(uploadPromises);

      const successful = results.filter((r) => r.status === "fulfilled").length;
      const failed = results.filter((r) => r.status === "rejected").length;

      this.announce(
        `Upload complete. ${successful} successful, ${failed} failed.`,
      );
      this.emit("uploadComplete", { successful, failed, results });
    } catch (error) {
      console.error("Upload error:", error);
      this.announce("Upload failed");
      this.emit("uploadError", { error });
    } finally {
      this.isSubmitting = false;
    }
  }

  async uploadFile(fileData) {
    fileData.status = "uploading";
    fileData.progress = 0;
    this.updateFileItem(fileData);

    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      const formData = new FormData();
      formData.append("file", fileData.file);

      // Store upload reference
      this.uploads.set(fileData.id, { xhr, fileData });

      // Progress tracking
      xhr.upload.addEventListener("progress", (e) => {
        if (e.lengthComputable) {
          fileData.progress = Math.round((e.loaded / e.total) * 100);
          this.updateFileItem(fileData);
        }
      });

      // Success
      xhr.addEventListener("load", () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          fileData.status = "completed";
          fileData.progress = 100;
          this.updateFileItem(fileData);
          this.uploads.delete(fileData.id);
          resolve(JSON.parse(xhr.responseText));
        } else {
          fileData.status = "error";
          fileData.error = `Upload failed: ${xhr.statusText}`;
          this.updateFileItem(fileData);
          this.uploads.delete(fileData.id);
          reject(new Error(fileData.error));
        }
      });

      // Error
      xhr.addEventListener("error", () => {
        fileData.status = "error";
        fileData.error = "Upload failed";
        this.updateFileItem(fileData);
        this.uploads.delete(fileData.id);
        reject(new Error("Upload failed"));
      });

      // Abort
      xhr.addEventListener("abort", () => {
        fileData.status = "cancelled";
        this.updateFileItem(fileData);
        this.uploads.delete(fileData.id);
        reject(new Error("Upload cancelled"));
      });

      // Send request
      xhr.open("POST", this.options.uploadUrl || "/api/upload");
      xhr.send(formData);
    });
  }

  getFileIcon(fileType) {
    if (fileType.startsWith("image/")) return "🖼️";
    if (fileType.startsWith("video/")) return "🎥";
    if (fileType.startsWith("audio/")) return "🎵";
    if (fileType.includes("pdf")) return "📄";
    if (fileType.includes("word")) return "📝";
    if (fileType.includes("excel") || fileType.includes("spreadsheet"))
      return "📊";
    if (fileType.includes("zip") || fileType.includes("compressed"))
      return "📦";
    return "📁";
  }

  formatFileSize(bytes) {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
  }

  async submit(data) {
    await this.startUpload();
    return { files: this.files };
  }

  reset() {
    super.reset();

    // Clear uploads
    this.uploads.forEach((upload) => {
      if (upload.xhr) {
        upload.xhr.abort();
      }
    });
    this.uploads.clear();

    // Clear files
    this.files = [];
    this.fileListContainer.innerHTML = "";
    this.updateFileList();
  }
}

// Export classes for use in other modules
if (typeof module !== "undefined" && module.exports) {
  module.exports = {
    BaseForm,
    MultiStepForm,
    FileUploadForm,
  };
} else {
  // Browser environment
  window.BaseForm = BaseForm;
  window.MultiStepForm = MultiStepForm;
  window.FileUploadForm = FileUploadForm;
}
