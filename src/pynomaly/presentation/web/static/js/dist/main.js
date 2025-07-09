// main.js\n// Pynomaly Web UI - Main JavaScript Entry Point
import "./components/anomaly-detector.js";
import "./components/data-uploader.js";
import "./components/chart-components.js";
import "./utils/htmx-extensions.js";
import "./utils/pwa-manager.js";
import "./utils/accessibility.js";
import "./utils/performance-monitor.js";

// Core Application Class
class PynomalyApp {
  constructor() {
    this.version = "1.0.0";
    this.isOnline = navigator.onLine;
    this.theme = localStorage.getItem("theme") || "light";
    this.components = new Map();

    this.init();
  }

  // Initialize the application
  async init() {
    console.log("🚀 Pynomaly App initializing...");

    try {
      // Initialize core features
      await this.initServiceWorker();
      this.initTheme();
      this.initHTMX();
      this.initAccessibility();
      this.initPerformanceMonitoring();
      this.bindEventListeners();

      // Initialize components
      this.initComponents();

      // Setup offline handling
      this.setupOfflineHandling();

      console.log("✅ Pynomaly App initialized successfully");
    } catch (error) {
      console.error("❌ Failed to initialize Pynomaly App:", error);
      this.showErrorMessage("Failed to initialize application");
    }
  }

  // Initialize Service Worker for PWA functionality
  async initServiceWorker() {
    if ("serviceWorker" in navigator) {
      try {
        const registration = await navigator.serviceWorker.register("/sw.js");
        console.log("Service Worker registered:", registration);

        // Handle service worker updates
        registration.addEventListener("updatefound", () => {
          const newWorker = registration.installing;
          newWorker.addEventListener("statechange", () => {
            if (
              newWorker.state === "installed" &&
              navigator.serviceWorker.controller
            ) {
              this.showUpdateNotification();
            }
          });
        });
      } catch (error) {
        console.warn("Service Worker registration failed:", error);
      }
    }
  }

  // Initialize theme system
  initTheme() {
    // Apply saved theme
    document.documentElement.classList.toggle("dark", this.theme === "dark");

    // Update theme toggle button if it exists
    const themeToggle = document.querySelector("[data-theme-toggle]");
    if (themeToggle) {
      themeToggle.setAttribute("aria-pressed", this.theme === "dark");
    }

    // Listen for system theme changes
    if (window.matchMedia) {
      const mediaQuery = window.matchMedia("(prefers-color-scheme: dark)");
      mediaQuery.addEventListener("change", (e) => {
        if (!localStorage.getItem("theme")) {
          this.setTheme(e.matches ? "dark" : "light");
        }
      });
    }
  }

  // Toggle theme
  toggleTheme() {
    const newTheme = this.theme === "light" ? "dark" : "light";
    this.setTheme(newTheme);
  }

  // Set theme
  setTheme(theme) {
    this.theme = theme;
    localStorage.setItem("theme", theme);
    document.documentElement.classList.toggle("dark", theme === "dark");

    // Update theme toggle button
    const themeToggle = document.querySelector("[data-theme-toggle]");
    if (themeToggle) {
      themeToggle.setAttribute("aria-pressed", theme === "dark");
    }

    // Dispatch theme change event
    document.dispatchEvent(
      new CustomEvent("themechange", {
        detail: { theme },
      }),
    );
  }

  // Initialize HTMX with custom extensions
  initHTMX() {
    if (typeof htmx !== "undefined") {
      // Configure HTMX
      htmx.config.responseHandling = [
        { code: "204", swap: false },
        { code: "[23].+", swap: true },
        { code: "[45].+", swap: false, error: true },
      ];

      // Add loading indicators
      htmx.on("htmx:beforeRequest", (event) => {
        this.showLoadingIndicator(event.target);
      });

      htmx.on("htmx:afterRequest", (event) => {
        this.hideLoadingIndicator(event.target);
      });

      // Handle HTMX errors
      htmx.on("htmx:responseError", (event) => {
        console.error("HTMX error:", event.detail);
        this.showErrorMessage("Request failed. Please try again.");
      });

      // Handle network errors
      htmx.on("htmx:sendError", (event) => {
        console.error("HTMX network error:", event.detail);
        this.showErrorMessage("Network error. Please check your connection.");
      });
    }
  }

  // Initialize accessibility features
  initAccessibility() {
    // Skip link focus management
    this.setupSkipLinks();

    // Focus management for dynamic content
    this.setupFocusManagement();

    // Keyboard navigation enhancement
    this.setupKeyboardNavigation();

    // Announce dynamic content changes
    this.setupLiveRegions();
  }

  // Initialize performance monitoring
  initPerformanceMonitoring() {
    if ("PerformanceObserver" in window) {
      // Monitor Core Web Vitals
      this.monitorCoreWebVitals();

      // Monitor resource loading
      this.monitorResourceLoading();

      // Monitor long tasks
      this.monitorLongTasks();
    }
  }

  // Monitor Core Web Vitals
  monitorCoreWebVitals() {
    // LCP (Largest Contentful Paint)
    new PerformanceObserver((entryList) => {
      for (const entry of entryList.getEntries()) {
        console.log("LCP:", entry.startTime);
        this.reportMetric("lcp", entry.startTime);
      }
    }).observe({ type: "largest-contentful-paint", buffered: true });

    // FID (First Input Delay)
    new PerformanceObserver((entryList) => {
      for (const entry of entryList.getEntries()) {
        const fid = entry.processingStart - entry.startTime;
        console.log("FID:", fid);
        this.reportMetric("fid", fid);
      }
    }).observe({ type: "first-input", buffered: true });

    // CLS (Cumulative Layout Shift)
    let clsValue = 0;
    new PerformanceObserver((entryList) => {
      for (const entry of entryList.getEntries()) {
        if (!entry.hadRecentInput) {
          clsValue += entry.value;
        }
      }
      console.log("CLS:", clsValue);
      this.reportMetric("cls", clsValue);
    }).observe({ type: "layout-shift", buffered: true });
  }

  // Monitor resource loading performance
  monitorResourceLoading() {
    new PerformanceObserver((entryList) => {
      for (const entry of entryList.getEntries()) {
        if (entry.duration > 1000) {
          // Log slow resources
          console.warn("Slow resource:", entry.name, entry.duration);
        }
      }
    }).observe({ type: "resource", buffered: true });
  }

  // Monitor long tasks that block the main thread
  monitorLongTasks() {
    new PerformanceObserver((entryList) => {
      for (const entry of entryList.getEntries()) {
        console.warn("Long task detected:", entry.duration);
        this.reportMetric("long-task", entry.duration);
      }
    }).observe({ type: "longtask", buffered: true });
  }

  // Report performance metrics
  reportMetric(name, value) {
    // Send to analytics or monitoring service
    if (navigator.sendBeacon) {
      const data = JSON.stringify({
        metric: name,
        value,
        timestamp: Date.now(),
      });
      navigator.sendBeacon("/api/metrics", data);
    }
  }

  // Initialize components
  initComponents() {
    // Auto-initialize components with data attributes
    document.querySelectorAll("[data-component]").forEach((element) => {
      const componentName = element.dataset.component;
      this.initializeComponent(componentName, element);
    });
  }

  // Initialize a specific component
  initializeComponent(name, element) {
    try {
      switch (name) {
        case "anomaly-detector":
          import("./components/anomaly-detector.js").then((module) => {
            this.components.set(element, new module.AnomalyDetector(element));
          });
          break;
        case "data-uploader":
          import("./components/data-uploader.js").then((module) => {
            this.components.set(element, new module.DataUploader(element));
          });
          break;
        case "chart":
          import("./components/chart-components.js").then((module) => {
            this.components.set(element, new module.ChartComponent(element));
          });
          break;
        default:
          console.warn(`Unknown component: ${name}`);
      }
    } catch (error) {
      console.error(`Failed to initialize component ${name}:`, error);
    }
  }

  // Setup offline handling
  setupOfflineHandling() {
    window.addEventListener("online", () => {
      this.isOnline = true;
      this.hideOfflineNotification();
      console.log("✅ Back online");
    });

    window.addEventListener("offline", () => {
      this.isOnline = false;
      this.showOfflineNotification();
      console.log("📡 Gone offline");
    });
  }

  // Bind global event listeners
  bindEventListeners() {
    // Theme toggle
    document.addEventListener("click", (e) => {
      if (e.target.matches("[data-theme-toggle]")) {
        e.preventDefault();
        this.toggleTheme();
      }
    });

    // Skip links
    document.addEventListener("click", (e) => {
      if (e.target.matches(".skip-link")) {
        e.preventDefault();
        const target = document.querySelector(e.target.getAttribute("href"));
        if (target) {
          target.focus();
          target.scrollIntoView();
        }
      }
    });

    // Global keyboard shortcuts
    document.addEventListener("keydown", (e) => {
      // Alt + T: Toggle theme
      if (e.altKey && e.key === "t") {
        e.preventDefault();
        this.toggleTheme();
      }

      // Alt + D: Go to dashboard
      if (e.altKey && e.key === "d") {
        e.preventDefault();
        window.location.href = "/dashboard";
      }
    });

    // Handle component events
    document.addEventListener("component:ready", (e) => {
      console.log("Component ready:", e.detail);
    });

    document.addEventListener("component:error", (e) => {
      console.error("Component error:", e.detail);
      this.showErrorMessage(e.detail.message);
    });
  }

  // Setup skip links for accessibility
  setupSkipLinks() {
    const skipLinks = document.querySelectorAll(".skip-link");
    skipLinks.forEach((link) => {
      link.addEventListener("focus", () => {
        link.style.position = "absolute";
        link.style.left = "6px";
        link.style.top = "7px";
        link.style.zIndex = "999999";
      });

      link.addEventListener("blur", () => {
        link.style.position = "absolute";
        link.style.left = "-10000px";
        link.style.top = "auto";
      });
    });
  }

  // Setup focus management for dynamic content
  setupFocusManagement() {
    // Restore focus after HTMX requests
    htmx.on("htmx:afterSwap", (event) => {
      const newContent = event.target;
      const focusTarget =
        newContent.querySelector("[autofocus]") ||
        newContent.querySelector("h1, h2, h3") ||
        newContent;

      if (focusTarget && focusTarget.focus) {
        focusTarget.focus();
      }
    });
  }

  // Setup keyboard navigation
  setupKeyboardNavigation() {
    // Enhanced tab navigation
    document.addEventListener("keydown", (e) => {
      if (e.key === "Tab") {
        document.body.classList.add("keyboard-navigation");
      }
    });

    document.addEventListener("mousedown", () => {
      document.body.classList.remove("keyboard-navigation");
    });
  }

  // Setup live regions for screen readers
  setupLiveRegions() {
    // Create live region if it doesn't exist
    if (!document.getElementById("live-region")) {
      const liveRegion = document.createElement("div");
      liveRegion.id = "live-region";
      liveRegion.setAttribute("aria-live", "polite");
      liveRegion.setAttribute("aria-atomic", "true");
      liveRegion.className = "sr-only";
      document.body.appendChild(liveRegion);
    }
  }

  // Announce message to screen readers
  announceToScreenReader(message) {
    const liveRegion = document.getElementById("live-region");
    if (liveRegion) {
      liveRegion.textContent = message;
      setTimeout(() => {
        liveRegion.textContent = "";
      }, 1000);
    }
  }

  // Show loading indicator
  showLoadingIndicator(element) {
    const indicator = document.createElement("div");
    indicator.className = "loading-indicator";
    indicator.innerHTML = '<div class="loading-spinner w-4 h-4"></div>';
    indicator.setAttribute("aria-label", "Loading");

    element.style.position = "relative";
    element.appendChild(indicator);
  }

  // Hide loading indicator
  hideLoadingIndicator(element) {
    const indicator = element.querySelector(".loading-indicator");
    if (indicator) {
      indicator.remove();
    }
  }

  // Show error message
  showErrorMessage(message) {
    this.showNotification(message, "error");
    this.announceToScreenReader(`Error: ${message}`);
  }

  // Show success message
  showSuccessMessage(message) {
    this.showNotification(message, "success");
    this.announceToScreenReader(message);
  }

  // Show notification
  showNotification(message, type = "info") {
    const notification = document.createElement("div");
    notification.className = `notification notification-${type}`;
    notification.innerHTML = `
      <div class="notification-content">
        <span class="notification-message">${message}</span>
        <button class="notification-close" aria-label="Close notification">&times;</button>
      </div>
    `;

    // Auto-remove after 5 seconds
    setTimeout(() => {
      if (notification.parentNode) {
        notification.remove();
      }
    }, 5000);

    // Close button
    notification
      .querySelector(".notification-close")
      .addEventListener("click", () => {
        notification.remove();
      });

    // Add to page
    const container = document.getElementById("notifications") || document.body;
    container.appendChild(notification);
  }

  // Show offline notification
  showOfflineNotification() {
    this.showNotification(
      "You are currently offline. Some features may not be available.",
      "warning",
    );
  }

  // Hide offline notification
  hideOfflineNotification() {
    this.showNotification("You are back online!", "success");
  }

  // Show update notification
  showUpdateNotification() {
    const notification = document.createElement("div");
    notification.className = "notification notification-info";
    notification.innerHTML = `
      <div class="notification-content">
        <span class="notification-message">A new version is available!</span>
        <button class="btn btn-sm btn-primary" onclick="window.location.reload()">Update</button>
        <button class="notification-close" aria-label="Close notification">&times;</button>
      </div>
    `;

    notification
      .querySelector(".notification-close")
      .addEventListener("click", () => {
        notification.remove();
      });

    const container = document.getElementById("notifications") || document.body;
    container.appendChild(notification);
  }

  // Get component instance
  getComponent(element) {
    return this.components.get(element);
  }

  // Destroy component
  destroyComponent(element) {
    const component = this.components.get(element);
    if (component && typeof component.destroy === "function") {
      component.destroy();
    }
    this.components.delete(element);
  }

  // Cleanup
  destroy() {
    this.components.forEach((component, element) => {
      this.destroyComponent(element);
    });
    this.components.clear();
  }
}

// Initialize the application when DOM is ready
document.addEventListener("DOMContentLoaded", () => {
  window.PynomalyApp = new PynomalyApp();
});

// Export for module usage
export default PynomalyApp;
\n\n// advanced-forms.js\n/**
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
\n\n// analytics-charts.js\n/**
 * Advanced Analytics Charts Component
 *
 * Interactive data exploration and analysis charts using D3.js and ECharts
 * for comprehensive anomaly detection analytics and insights
 */

import * as d3 from "d3";

export class AnalyticsCharts {
  constructor(container, options = {}) {
    this.container =
      typeof container === "string"
        ? document.querySelector(container)
        : container;
    this.options = {
      width: 800,
      height: 400,
      theme: "light",
      enableInteraction: true,
      enableZoom: true,
      enableBrush: true,
      enableLegend: true,
      enableTooltip: true,
      animationDuration: 750,
      margin: { top: 20, right: 80, bottom: 40, left: 60 },
      ...options,
    };

    this.data = [];
    this.charts = new Map();
    this.scales = {};
    this.brushes = new Map();
    this.zoom = null;

    this.init();
  }

  init() {
    this.setupContainer();
    this.setupScales();
    this.bindEvents();
  }

  setupContainer() {
    this.container.classList.add("analytics-charts");
    this.container.innerHTML = "";

    // Create SVG
    this.svg = d3
      .select(this.container)
      .append("svg")
      .attr("width", this.options.width)
      .attr("height", this.options.height)
      .style(
        "background",
        this.options.theme === "dark" ? "#1a1a1a" : "#ffffff",
      );

    // Create main group
    this.mainGroup = this.svg
      .append("g")
      .attr(
        "transform",
        `translate(${this.options.margin.left}, ${this.options.margin.top})`,
      );

    // Calculate dimensions
    this.innerWidth =
      this.options.width - this.options.margin.left - this.options.margin.right;
    this.innerHeight =
      this.options.height -
      this.options.margin.top -
      this.options.margin.bottom;

    // Create chart groups
    this.chartArea = this.mainGroup.append("g").attr("class", "chart-area");
    this.axesGroup = this.mainGroup.append("g").attr("class", "axes");
    this.legendGroup = this.mainGroup.append("g").attr("class", "legend");
    this.tooltipGroup = this.mainGroup
      .append("g")
      .attr("class", "tooltip-group");

    // Create tooltip
    this.tooltip = d3
      .select("body")
      .append("div")
      .attr("class", "chart-tooltip")
      .style("opacity", 0)
      .style("position", "absolute")
      .style("background", "rgba(0, 0, 0, 0.8)")
      .style("color", "white")
      .style("padding", "8px 12px")
      .style("border-radius", "4px")
      .style("font-size", "12px")
      .style("pointer-events", "none")
      .style("z-index", 1000);
  }

  setupScales() {
    this.scales.x = d3.scaleTime().range([0, this.innerWidth]);
    this.scales.y = d3.scaleLinear().range([this.innerHeight, 0]);
    this.scales.color = d3.scaleOrdinal(d3.schemeCategory10);
    this.scales.size = d3.scaleSqrt().range([2, 10]);
  }

  bindEvents() {
    // Setup zoom
    if (this.options.enableZoom) {
      this.zoom = d3
        .zoom()
        .scaleExtent([0.1, 10])
        .on("zoom", (event) => {
          this.onZoom(event);
        });

      this.svg.call(this.zoom);
    }

    // Setup brush
    if (this.options.enableBrush) {
      this.brush = d3
        .brushX()
        .extent([
          [0, 0],
          [this.innerWidth, this.innerHeight],
        ])
        .on("brush end", (event) => {
          this.onBrush(event);
        });
    }
  }

  // Chart creation methods
  createScatterPlot(data, config = {}) {
    const chartConfig = {
      xField: "x",
      yField: "y",
      colorField: "category",
      sizeField: "value",
      showTrendline: false,
      enableClustering: false,
      ...config,
    };

    this.clearChart();
    this.data = data;

    // Update scales
    this.scales.x.domain(d3.extent(data, (d) => d[chartConfig.xField]));
    this.scales.y.domain(d3.extent(data, (d) => d[chartConfig.yField]));

    if (chartConfig.sizeField) {
      this.scales.size.domain(d3.extent(data, (d) => d[chartConfig.sizeField]));
    }

    // Draw axes
    this.drawAxes();

    // Draw points
    const circles = this.chartArea
      .selectAll(".data-point")
      .data(data)
      .enter()
      .append("circle")
      .attr("class", "data-point")
      .attr("cx", (d) => this.scales.x(d[chartConfig.xField]))
      .attr("cy", (d) => this.scales.y(d[chartConfig.yField]))
      .attr("r", (d) =>
        chartConfig.sizeField ? this.scales.size(d[chartConfig.sizeField]) : 4,
      )
      .attr("fill", (d) =>
        this.scales.color(d[chartConfig.colorField] || "default"),
      )
      .attr("opacity", 0.7)
      .style("cursor", "pointer");

    // Add interactions
    if (this.options.enableTooltip) {
      circles
        .on("mouseover", (event, d) => {
          this.showTooltip(event, d, chartConfig);
        })
        .on("mouseout", () => {
          this.hideTooltip();
        });
    }

    // Add trendline if requested
    if (chartConfig.showTrendline) {
      this.addTrendline(data, chartConfig);
    }

    // Add legend
    if (this.options.enableLegend && chartConfig.colorField) {
      this.drawLegend(data, chartConfig.colorField);
    }

    this.charts.set("scatter", { data, config: chartConfig });
    return this;
  }

  createTimeSeries(data, config = {}) {
    const chartConfig = {
      timeField: "timestamp",
      valueField: "value",
      categoryField: null,
      lineType: "line", // line, area, step
      showPoints: false,
      showConfidenceBand: false,
      aggregation: null, // sum, avg, max, min
      ...config,
    };

    this.clearChart();
    this.data = data;

    // Process data for time series
    const processedData = this.processTimeSeriesData(data, chartConfig);

    // Update scales
    this.scales.x.domain(d3.extent(processedData, (d) => d.time));
    this.scales.y.domain(d3.extent(processedData, (d) => d.value));

    // Draw axes
    this.drawAxes(true); // time axis

    // Create line generator
    const line = d3
      .line()
      .x((d) => this.scales.x(d.time))
      .y((d) => this.scales.y(d.value))
      .curve(
        chartConfig.lineType === "step" ? d3.curveStepAfter : d3.curveMonotoneX,
      );

    // Group by category if specified
    const dataGroups = chartConfig.categoryField
      ? d3.group(processedData, (d) => d[chartConfig.categoryField])
      : new Map([["default", processedData]]);

    // Draw lines
    dataGroups.forEach((groupData, category) => {
      const path = this.chartArea
        .append("path")
        .datum(groupData)
        .attr("class", `time-series-line category-${category}`)
        .attr(
          "fill",
          chartConfig.lineType === "area"
            ? this.scales.color(category)
            : "none",
        )
        .attr("stroke", this.scales.color(category))
        .attr("stroke-width", 2)
        .attr("opacity", chartConfig.lineType === "area" ? 0.6 : 0.8)
        .attr(
          "d",
          chartConfig.lineType === "area"
            ? d3
                .area()
                .x((d) => this.scales.x(d.time))
                .y0(this.scales.y(0))
                .y1((d) => this.scales.y(d.value))
                .curve(d3.curveMonotoneX)
            : line,
        );

      // Animate path drawing
      const totalLength = path.node().getTotalLength();
      path
        .attr("stroke-dasharray", totalLength + " " + totalLength)
        .attr("stroke-dashoffset", totalLength)
        .transition()
        .duration(this.options.animationDuration)
        .attr("stroke-dashoffset", 0);

      // Add points if requested
      if (chartConfig.showPoints) {
        this.chartArea
          .selectAll(`.points-${category}`)
          .data(groupData)
          .enter()
          .append("circle")
          .attr("class", `data-point points-${category}`)
          .attr("cx", (d) => this.scales.x(d.time))
          .attr("cy", (d) => this.scales.y(d.value))
          .attr("r", 3)
          .attr("fill", this.scales.color(category))
          .style("cursor", "pointer")
          .on("mouseover", (event, d) => {
            this.showTooltip(event, d, chartConfig);
          })
          .on("mouseout", () => {
            this.hideTooltip();
          });
      }
    });

    // Add confidence band if requested
    if (chartConfig.showConfidenceBand) {
      this.addConfidenceBand(processedData, chartConfig);
    }

    // Add brush for time selection
    if (this.options.enableBrush) {
      this.chartArea.append("g").attr("class", "brush").call(this.brush);
    }

    // Add legend
    if (this.options.enableLegend && chartConfig.categoryField) {
      this.drawLegend(
        Array.from(dataGroups.keys()).map((key) => ({
          [chartConfig.categoryField]: key,
        })),
        chartConfig.categoryField,
      );
    }

    this.charts.set("timeSeries", { data: processedData, config: chartConfig });
    return this;
  }

  createHeatmap(data, config = {}) {
    const chartConfig = {
      xField: "x",
      yField: "y",
      valueField: "value",
      colorScheme: "interpolateViridis",
      showValues: false,
      cellPadding: 1,
      ...config,
    };

    this.clearChart();
    this.data = data;

    // Get unique x and y values
    const xValues = [...new Set(data.map((d) => d[chartConfig.xField]))].sort();
    const yValues = [...new Set(data.map((d) => d[chartConfig.yField]))].sort();

    // Update scales
    this.scales.x = d3
      .scaleBand()
      .domain(xValues)
      .range([0, this.innerWidth])
      .padding(0.1);
    this.scales.y = d3
      .scaleBand()
      .domain(yValues)
      .range([0, this.innerHeight])
      .padding(0.1);

    // Color scale
    const colorScale = d3
      .scaleSequential(d3[chartConfig.colorScheme])
      .domain(d3.extent(data, (d) => d[chartConfig.valueField]));

    // Draw axes
    this.drawAxes(false, true); // categorical axes

    // Draw heatmap cells
    const cells = this.chartArea
      .selectAll(".heatmap-cell")
      .data(data)
      .enter()
      .append("rect")
      .attr("class", "heatmap-cell")
      .attr("x", (d) => this.scales.x(d[chartConfig.xField]))
      .attr("y", (d) => this.scales.y(d[chartConfig.yField]))
      .attr("width", this.scales.x.bandwidth())
      .attr("height", this.scales.y.bandwidth())
      .attr("fill", (d) => colorScale(d[chartConfig.valueField]))
      .attr("opacity", 0)
      .style("cursor", "pointer");

    // Animate cells
    cells
      .transition()
      .duration(this.options.animationDuration)
      .delay((d, i) => i * 10)
      .attr("opacity", 0.8);

    // Add value labels if requested
    if (chartConfig.showValues) {
      this.chartArea
        .selectAll(".cell-label")
        .data(data)
        .enter()
        .append("text")
        .attr("class", "cell-label")
        .attr(
          "x",
          (d) =>
            this.scales.x(d[chartConfig.xField]) +
            this.scales.x.bandwidth() / 2,
        )
        .attr(
          "y",
          (d) =>
            this.scales.y(d[chartConfig.yField]) +
            this.scales.y.bandwidth() / 2,
        )
        .attr("text-anchor", "middle")
        .attr("dominant-baseline", "central")
        .attr("fill", (d) =>
          colorScale(d[chartConfig.valueField]) > 0.5 ? "white" : "black",
        )
        .attr("font-size", "10px")
        .text((d) => d[chartConfig.valueField].toFixed(2));
    }

    // Add interactions
    if (this.options.enableTooltip) {
      cells
        .on("mouseover", (event, d) => {
          this.showTooltip(event, d, chartConfig);
        })
        .on("mouseout", () => {
          this.hideTooltip();
        });
    }

    // Add color legend
    this.drawColorLegend(colorScale, chartConfig.valueField);

    this.charts.set("heatmap", { data, config: chartConfig });
    return this;
  }

  createHistogram(data, config = {}) {
    const chartConfig = {
      valueField: "value",
      bins: 20,
      showDensity: false,
      showStats: true,
      overlayDistribution: null, // normal, exponential, etc.
      ...config,
    };

    this.clearChart();
    this.data = data;

    const values = data.map((d) => d[chartConfig.valueField]);

    // Create histogram
    const histogram = d3
      .histogram()
      .domain(d3.extent(values))
      .thresholds(chartConfig.bins);

    const bins = histogram(values);

    // Update scales
    this.scales.x.domain(d3.extent(values));
    this.scales.y.domain([0, d3.max(bins, (d) => d.length)]);

    // Draw axes
    this.drawAxes();

    // Draw bars
    const bars = this.chartArea
      .selectAll(".histogram-bar")
      .data(bins)
      .enter()
      .append("rect")
      .attr("class", "histogram-bar")
      .attr("x", (d) => this.scales.x(d.x0))
      .attr("y", this.innerHeight)
      .attr("width", (d) =>
        Math.max(0, this.scales.x(d.x1) - this.scales.x(d.x0) - 1),
      )
      .attr("height", 0)
      .attr("fill", this.scales.color("histogram"))
      .attr("opacity", 0.7)
      .style("cursor", "pointer");

    // Animate bars
    bars
      .transition()
      .duration(this.options.animationDuration)
      .attr("y", (d) => this.scales.y(d.length))
      .attr("height", (d) => this.innerHeight - this.scales.y(d.length));

    // Add interactions
    if (this.options.enableTooltip) {
      bars
        .on("mouseover", (event, d) => {
          this.showTooltip(
            event,
            {
              range: `${d.x0.toFixed(2)} - ${d.x1.toFixed(2)}`,
              count: d.length,
              percentage: ((d.length / values.length) * 100).toFixed(1) + "%",
            },
            { type: "histogram" },
          );
        })
        .on("mouseout", () => {
          this.hideTooltip();
        });
    }

    // Add statistics
    if (chartConfig.showStats) {
      this.addStatistics(values);
    }

    // Add overlay distribution
    if (chartConfig.overlayDistribution) {
      this.addDistributionOverlay(values, chartConfig.overlayDistribution);
    }

    this.charts.set("histogram", { data: bins, config: chartConfig });
    return this;
  }

  // Utility methods
  processTimeSeriesData(data, config) {
    return data
      .map((d) => ({
        time: new Date(d[config.timeField]),
        value: +d[config.valueField],
        ...d,
      }))
      .sort((a, b) => a.time - b.time);
  }

  drawAxes(isTimeAxis = false, isCategorical = false) {
    this.axesGroup.selectAll("*").remove();

    // X axis
    const xAxis = isCategorical
      ? d3.axisBottom(this.scales.x)
      : isTimeAxis
        ? d3.axisBottom(this.scales.x).tickFormat(d3.timeFormat("%H:%M"))
        : d3.axisBottom(this.scales.x);

    this.axesGroup
      .append("g")
      .attr("class", "x-axis")
      .attr("transform", `translate(0, ${this.innerHeight})`)
      .call(xAxis);

    // Y axis
    const yAxis = isCategorical
      ? d3.axisLeft(this.scales.y)
      : d3.axisLeft(this.scales.y);

    this.axesGroup.append("g").attr("class", "y-axis").call(yAxis);
  }

  drawLegend(data, field) {
    this.legendGroup.selectAll("*").remove();

    const categories = [...new Set(data.map((d) => d[field]))];
    const legendItems = this.legendGroup
      .selectAll(".legend-item")
      .data(categories)
      .enter()
      .append("g")
      .attr("class", "legend-item")
      .attr(
        "transform",
        (d, i) => `translate(${this.innerWidth + 10}, ${i * 20})`,
      );

    legendItems
      .append("rect")
      .attr("width", 12)
      .attr("height", 12)
      .attr("fill", (d) => this.scales.color(d));

    legendItems
      .append("text")
      .attr("x", 16)
      .attr("y", 9)
      .attr("font-size", "12px")
      .text((d) => d);
  }

  drawColorLegend(colorScale, label) {
    const legendWidth = 200;
    const legendHeight = 20;

    const legend = this.legendGroup
      .append("g")
      .attr("class", "color-legend")
      .attr(
        "transform",
        `translate(${(this.innerWidth - legendWidth) / 2}, ${this.innerHeight + 40})`,
      );

    // Create gradient
    const defs = this.svg.append("defs");
    const gradient = defs
      .append("linearGradient")
      .attr("id", "legend-gradient");

    const domain = colorScale.domain();
    const steps = 10;
    for (let i = 0; i <= steps; i++) {
      const value = domain[0] + ((domain[1] - domain[0]) * i) / steps;
      gradient
        .append("stop")
        .attr("offset", `${(i / steps) * 100}%`)
        .attr("stop-color", colorScale(value));
    }

    // Draw legend rectangle
    legend
      .append("rect")
      .attr("width", legendWidth)
      .attr("height", legendHeight)
      .attr("fill", "url(#legend-gradient)");

    // Add scale
    const legendScale = d3.scaleLinear().domain(domain).range([0, legendWidth]);

    legend
      .append("g")
      .attr("transform", `translate(0, ${legendHeight})`)
      .call(d3.axisBottom(legendScale).ticks(5));

    // Add label
    legend
      .append("text")
      .attr("x", legendWidth / 2)
      .attr("y", -5)
      .attr("text-anchor", "middle")
      .attr("font-size", "12px")
      .text(label);
  }

  addTrendline(data, config) {
    const xValues = data.map((d) => d[config.xField]);
    const yValues = data.map((d) => d[config.yField]);

    // Calculate linear regression
    const n = data.length;
    const sumX = d3.sum(xValues);
    const sumY = d3.sum(yValues);
    const sumXY = d3.sum(data, (d) => d[config.xField] * d[config.yField]);
    const sumXX = d3.sum(xValues, (d) => d * d);

    const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
    const intercept = (sumY - slope * sumX) / n;

    // Draw trendline
    const xExtent = d3.extent(xValues);
    const trendData = [
      { x: xExtent[0], y: slope * xExtent[0] + intercept },
      { x: xExtent[1], y: slope * xExtent[1] + intercept },
    ];

    const line = d3
      .line()
      .x((d) => this.scales.x(d.x))
      .y((d) => this.scales.y(d.y));

    this.chartArea
      .append("path")
      .datum(trendData)
      .attr("class", "trendline")
      .attr("fill", "none")
      .attr("stroke", "red")
      .attr("stroke-width", 2)
      .attr("stroke-dasharray", "5,5")
      .attr("d", line);
  }

  addConfidenceBand(data, config) {
    // Simple confidence band implementation
    const windowSize = Math.floor(data.length / 10);
    const confidenceData = [];

    for (let i = windowSize; i < data.length - windowSize; i++) {
      const window = data.slice(i - windowSize, i + windowSize);
      const values = window.map((d) => d.value);
      const mean = d3.mean(values);
      const std = d3.deviation(values);

      confidenceData.push({
        time: data[i].time,
        lower: mean - 1.96 * std,
        upper: mean + 1.96 * std,
      });
    }

    const area = d3
      .area()
      .x((d) => this.scales.x(d.time))
      .y0((d) => this.scales.y(d.lower))
      .y1((d) => this.scales.y(d.upper))
      .curve(d3.curveMonotoneX);

    this.chartArea
      .append("path")
      .datum(confidenceData)
      .attr("class", "confidence-band")
      .attr("fill", "steelblue")
      .attr("opacity", 0.2)
      .attr("d", area);
  }

  addStatistics(values) {
    const stats = {
      mean: d3.mean(values),
      median: d3.median(values),
      std: d3.deviation(values),
      min: d3.min(values),
      max: d3.max(values),
    };

    // Add vertical lines for mean and median
    this.chartArea
      .append("line")
      .attr("class", "stat-line mean")
      .attr("x1", this.scales.x(stats.mean))
      .attr("x2", this.scales.x(stats.mean))
      .attr("y1", 0)
      .attr("y2", this.innerHeight)
      .attr("stroke", "red")
      .attr("stroke-width", 2)
      .attr("stroke-dasharray", "3,3");

    this.chartArea
      .append("line")
      .attr("class", "stat-line median")
      .attr("x1", this.scales.x(stats.median))
      .attr("x2", this.scales.x(stats.median))
      .attr("y1", 0)
      .attr("y2", this.innerHeight)
      .attr("stroke", "blue")
      .attr("stroke-width", 2)
      .attr("stroke-dasharray", "3,3");

    // Add statistics text
    const statsText = this.chartArea
      .append("g")
      .attr("class", "statistics")
      .attr("transform", `translate(10, 20)`);

    Object.entries(stats).forEach(([key, value], i) => {
      statsText
        .append("text")
        .attr("x", 0)
        .attr("y", i * 15)
        .attr("font-size", "12px")
        .attr("fill", this.options.theme === "dark" ? "white" : "black")
        .text(`${key}: ${value.toFixed(3)}`);
    });
  }

  addDistributionOverlay(values, distributionType) {
    const mean = d3.mean(values);
    const std = d3.deviation(values);

    if (distributionType === "normal") {
      const normalData = [];
      const xExtent = d3.extent(values);
      const step = (xExtent[1] - xExtent[0]) / 100;

      for (let x = xExtent[0]; x <= xExtent[1]; x += step) {
        const y =
          (1 / (std * Math.sqrt(2 * Math.PI))) *
          Math.exp(-0.5 * Math.pow((x - mean) / std, 2));
        normalData.push({ x, y: y * values.length * step });
      }

      const line = d3
        .line()
        .x((d) => this.scales.x(d.x))
        .y((d) => this.scales.y(d.y))
        .curve(d3.curveMonotoneX);

      this.chartArea
        .append("path")
        .datum(normalData)
        .attr("class", "distribution-overlay")
        .attr("fill", "none")
        .attr("stroke", "orange")
        .attr("stroke-width", 2)
        .attr("d", line);
    }
  }

  showTooltip(event, data, config) {
    if (!this.options.enableTooltip) return;

    let content = "";
    if (config.type === "histogram") {
      content = `Range: ${data.range}<br>Count: ${data.count}<br>Percentage: ${data.percentage}`;
    } else {
      content = Object.entries(data)
        .filter(([key, value]) => key !== "index")
        .map(
          ([key, value]) =>
            `${key}: ${typeof value === "number" ? value.toFixed(3) : value}`,
        )
        .join("<br>");
    }

    this.tooltip
      .style("opacity", 1)
      .html(content)
      .style("left", event.pageX + 10 + "px")
      .style("top", event.pageY - 10 + "px");
  }

  hideTooltip() {
    this.tooltip.style("opacity", 0);
  }

  onZoom(event) {
    const newScaleX = event.transform.rescaleX(this.scales.x);

    // Update chart elements based on zoom
    this.chartArea
      .selectAll(".data-point")
      .attr("cx", (d) => newScaleX(d.timestamp || d.x || d.time));

    this.chartArea
      .selectAll(".time-series-line")
      .attr("transform", event.transform);

    // Update x-axis
    this.axesGroup.select(".x-axis").call(d3.axisBottom(newScaleX));
  }

  onBrush(event) {
    if (!event.selection) return;

    const [x0, x1] = event.selection.map(this.scales.x.invert);

    // Emit brush event
    this.container.dispatchEvent(
      new CustomEvent("brush", {
        detail: { selection: [x0, x1] },
      }),
    );
  }

  clearChart() {
    this.chartArea.selectAll("*").remove();
    this.axesGroup.selectAll("*").remove();
    this.legendGroup.selectAll("*").remove();
  }

  resize(width, height) {
    this.options.width = width;
    this.options.height = height;

    this.svg.attr("width", width).attr("height", height);

    this.innerWidth =
      width - this.options.margin.left - this.options.margin.right;
    this.innerHeight =
      height - this.options.margin.top - this.options.margin.bottom;

    this.scales.x.range([0, this.innerWidth]);
    this.scales.y.range([this.innerHeight, 0]);

    // Redraw current chart
    const currentChart = this.charts.values().next().value;
    if (currentChart) {
      // Re-render with current data and config
      this.redraw();
    }
  }

  redraw() {
    const currentChart = this.charts.values().next().value;
    if (!currentChart) return;

    // Re-render based on chart type
    const chartType = Array.from(this.charts.keys())[0];
    switch (chartType) {
      case "scatter":
        this.createScatterPlot(currentChart.data, currentChart.config);
        break;
      case "timeSeries":
        this.createTimeSeries(this.data, currentChart.config);
        break;
      case "heatmap":
        this.createHeatmap(currentChart.data, currentChart.config);
        break;
      case "histogram":
        this.createHistogram(this.data, currentChart.config);
        break;
    }
  }

  exportChart(format = "png") {
    const svgElement = this.svg.node();

    if (format === "svg") {
      const serializer = new XMLSerializer();
      return serializer.serializeToString(svgElement);
    }

    // For raster formats, we'd need to use canvas conversion
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");
    const img = new Image();

    const svgData = new XMLSerializer().serializeToString(svgElement);
    const svgBlob = new Blob([svgData], {
      type: "image/svg+xml;charset=utf-8",
    });
    const url = URL.createObjectURL(svgBlob);

    return new Promise((resolve) => {
      img.onload = () => {
        canvas.width = this.options.width;
        canvas.height = this.options.height;
        ctx.drawImage(img, 0, 0);
        URL.revokeObjectURL(url);

        if (format === "png") {
          resolve(canvas.toDataURL("image/png"));
        } else if (format === "jpeg") {
          resolve(canvas.toDataURL("image/jpeg"));
        }
      };
      img.src = url;
    });
  }

  destroy() {
    if (this.tooltip) {
      this.tooltip.remove();
    }

    this.charts.clear();
    this.brushes.clear();

    if (this.container) {
      this.container.innerHTML = "";
    }
  }
}

export default AnalyticsCharts;
\n\n// anomaly-detector.js\n// Anomaly Detector Component - Production-ready anomaly detection interface
import { ChartComponent } from "./chart-components.js";

export class AnomalyDetector {
  constructor(element) {
    this.element = element;
    this.config = this.getConfig();
    this.state = {
      isProcessing: false,
      currentDataset: null,
      selectedAlgorithm: null,
      results: null,
      realTimeMode: false,
    };

    this.charts = new Map();
    this.websocket = null;

    this.init();
  }

  // Initialize the detector component
  init() {
    console.log("🔍 Initializing Anomaly Detector component");

    this.createInterface();
    this.bindEvents();
    this.loadAlgorithms();
    this.setupRealTimeCapabilities();

    // Announce component ready
    this.element.dispatchEvent(
      new CustomEvent("component:ready", {
        detail: { component: "anomaly-detector", element: this.element },
      }),
    );
  }

  // Get component configuration from data attributes
  getConfig() {
    const element = this.element;
    return {
      apiEndpoint: element.dataset.apiEndpoint || "/api/anomaly-detection",
      websocketUrl: element.dataset.websocketUrl || "/ws/anomaly-detection",
      autoDetect: element.dataset.autoDetect === "true",
      realTime: element.dataset.realTime === "true",
      maxFileSize: parseInt(element.dataset.maxFileSize) || 10 * 1024 * 1024, // 10MB
      allowedFormats: (
        element.dataset.allowedFormats || "csv,json,parquet"
      ).split(","),
      algorithms: element.dataset.algorithms
        ? JSON.parse(element.dataset.algorithms)
        : null,
    };
  }

  // Create the detector interface
  createInterface() {
    this.element.innerHTML = `
      <div class="anomaly-detector-container">
        <!-- Header -->
        <div class="detector-header card-header">
          <h3 class="text-lg font-semibold text-neutral-900">Anomaly Detection</h3>
          <div class="detector-controls flex items-center space-x-2">
            <button class="btn btn-sm btn-outline" data-action="toggle-realtime">
              <span class="realtime-icon">📡</span>
              <span class="realtime-text">Real-time</span>
            </button>
            <button class="btn btn-sm btn-ghost" data-action="toggle-settings">
              <span>⚙️</span>
            </button>
          </div>
        </div>

        <!-- Main Content -->
        <div class="detector-content">
          <!-- Step 1: Data Input -->
          <div class="detection-step" data-step="data-input">
            <div class="step-header">
              <h4 class="text-md font-medium">1. Select Data Source</h4>
              <div class="step-status status-normal" data-status="pending">Pending</div>
            </div>

            <div class="data-input-section">
              <div class="input-tabs">
                <button class="tab-button active" data-tab="upload">Upload File</button>
                <button class="tab-button" data-tab="dataset">Existing Dataset</button>
                <button class="tab-button" data-tab="stream">Real-time Stream</button>
              </div>

              <!-- File Upload Tab -->
              <div class="tab-content active" data-tab-content="upload">
                <div class="file-upload-area" data-drop-zone>
                  <div class="upload-icon">📁</div>
                  <div class="upload-text">
                    <p class="text-sm font-medium">Drop your data file here or click to browse</p>
                    <p class="text-xs text-neutral-500">Supported formats: CSV, JSON, Parquet (max ${this.config.maxFileSize / 1024 / 1024}MB)</p>
                  </div>
                  <input type="file" class="file-input" accept=".csv,.json,.parquet" hidden>
                </div>
                <div class="file-info hidden" data-file-info></div>
              </div>

              <!-- Existing Dataset Tab -->
              <div class="tab-content" data-tab-content="dataset">
                <div class="dataset-selector">
                  <select class="form-select" data-dataset-select>
                    <option value="">Select a dataset...</option>
                  </select>
                  <button class="btn btn-sm btn-outline" data-action="refresh-datasets">
                    <span>🔄</span> Refresh
                  </button>
                </div>
                <div class="dataset-info hidden" data-dataset-info></div>
              </div>

              <!-- Real-time Stream Tab -->
              <div class="tab-content" data-tab-content="stream">
                <div class="stream-config">
                  <div class="form-group">
                    <label class="form-label">Stream Source</label>
                    <select class="form-select" data-stream-source>
                      <option value="">Select stream source...</option>
                      <option value="kafka">Kafka Topic</option>
                      <option value="mqtt">MQTT</option>
                      <option value="websocket">WebSocket</option>
                      <option value="api">REST API Polling</option>
                    </select>
                  </div>
                  <div class="stream-connection-config hidden" data-stream-config></div>
                  <button class="btn btn-primary" data-action="connect-stream">Connect Stream</button>
                </div>
              </div>
            </div>
          </div>

          <!-- Step 2: Algorithm Selection -->
          <div class="detection-step" data-step="algorithm-selection">
            <div class="step-header">
              <h4 class="text-md font-medium">2. Choose Detection Algorithm</h4>
              <div class="step-status status-normal" data-status="pending">Pending</div>
            </div>

            <div class="algorithm-selection">
              <div class="algorithm-tabs">
                <button class="tab-button active" data-algo-tab="recommended">Recommended</button>
                <button class="tab-button" data-algo-tab="all">All Algorithms</button>
                <button class="tab-button" data-algo-tab="ensemble">Ensemble</button>
                <button class="tab-button" data-algo-tab="custom">Custom</button>
              </div>

              <div class="algorithm-grid" data-algorithm-grid></div>

              <div class="algorithm-params hidden" data-algorithm-params>
                <h5 class="text-sm font-medium mb-2">Algorithm Parameters</h5>
                <div class="params-form" data-params-form></div>
              </div>
            </div>
          </div>

          <!-- Step 3: Detection Execution -->
          <div class="detection-step" data-step="execution">
            <div class="step-header">
              <h4 class="text-md font-medium">3. Run Detection</h4>
              <div class="step-status status-normal" data-status="pending">Pending</div>
            </div>

            <div class="execution-controls">
              <button class="btn btn-primary btn-lg" data-action="start-detection" disabled>
                <span class="btn-icon">🚀</span>
                <span class="btn-text">Start Detection</span>
              </button>

              <div class="execution-options">
                <label class="checkbox-label">
                  <input type="checkbox" data-option="explain-results">
                  <span class="checkmark"></span>
                  Generate explanations
                </label>
                <label class="checkbox-label">
                  <input type="checkbox" data-option="save-model">
                  <span class="checkmark"></span>
                  Save trained model
                </label>
                <label class="checkbox-label">
                  <input type="checkbox" data-option="auto-threshold">
                  <span class="checkmark"></span>
                  Auto-optimize threshold
                </label>
              </div>
            </div>

            <div class="execution-progress hidden" data-execution-progress>
              <div class="progress-bar">
                <div class="progress-fill" data-progress-fill></div>
              </div>
              <div class="progress-text" data-progress-text>Initializing...</div>
              <button class="btn btn-sm btn-accent" data-action="cancel-detection">Cancel</button>
            </div>
          </div>

          <!-- Step 4: Results Visualization -->
          <div class="detection-step" data-step="results">
            <div class="step-header">
              <h4 class="text-md font-medium">4. Detection Results</h4>
              <div class="step-status status-normal" data-status="pending">Pending</div>
            </div>

            <div class="results-container hidden" data-results-container>
              <!-- Results Summary -->
              <div class="results-summary">
                <div class="metric-cards">
                  <div class="metric-card">
                    <div class="metric-value" data-metric="total-samples">-</div>
                    <div class="metric-label">Total Samples</div>
                  </div>
                  <div class="metric-card">
                    <div class="metric-value text-accent-600" data-metric="anomalies-detected">-</div>
                    <div class="metric-label">Anomalies Detected</div>
                  </div>
                  <div class="metric-card">
                    <div class="metric-value" data-metric="anomaly-rate">-</div>
                    <div class="metric-label">Anomaly Rate</div>
                  </div>
                  <div class="metric-card">
                    <div class="metric-value" data-metric="confidence-score">-</div>
                    <div class="metric-label">Confidence Score</div>
                  </div>
                </div>
              </div>

              <!-- Results Visualization -->
              <div class="results-visualization">
                <div class="chart-tabs">
                  <button class="tab-button active" data-chart-tab="scatter">Scatter Plot</button>
                  <button class="tab-button" data-chart-tab="timeline">Timeline</button>
                  <button class="tab-button" data-chart-tab="distribution">Distribution</button>
                  <button class="tab-button" data-chart-tab="heatmap">Feature Heatmap</button>
                </div>

                <div class="chart-container" data-chart-container>
                  <div class="chart-loading skeleton h-64"></div>
                </div>

                <div class="chart-controls">
                  <div class="threshold-control">
                    <label class="form-label">Anomaly Threshold</label>
                    <input type="range" class="threshold-slider" data-threshold-slider min="0" max="1" step="0.01" value="0.5">
                    <span class="threshold-value" data-threshold-value>0.5</span>
                  </div>

                  <div class="filter-controls">
                    <button class="btn btn-sm btn-outline" data-filter="show-all">Show All</button>
                    <button class="btn btn-sm btn-outline" data-filter="show-anomalies">Anomalies Only</button>
                    <button class="btn btn-sm btn-outline" data-filter="show-normal">Normal Only</button>
                  </div>
                </div>
              </div>

              <!-- Results Actions -->
              <div class="results-actions">
                <button class="btn btn-primary" data-action="export-results">
                  <span>📊</span> Export Results
                </button>
                <button class="btn btn-secondary" data-action="save-model">
                  <span>💾</span> Save Model
                </button>
                <button class="btn btn-outline" data-action="generate-report">
                  <span>📄</span> Generate Report
                </button>
                <button class="btn btn-ghost" data-action="explain-results">
                  <span>🔍</span> Explain Results
                </button>
              </div>
            </div>
          </div>
        </div>

        <!-- Settings Panel -->
        <div class="settings-panel hidden" data-settings-panel>
          <div class="settings-header">
            <h4 class="text-md font-medium">Detection Settings</h4>
            <button class="btn btn-sm btn-ghost" data-action="close-settings">✕</button>
          </div>

          <div class="settings-content">
            <div class="setting-group">
              <label class="form-label">Default Algorithm</label>
              <select class="form-select" data-setting="default-algorithm">
                <option value="auto">Auto-select</option>
                <option value="isolation-forest">Isolation Forest</option>
                <option value="one-class-svm">One-Class SVM</option>
                <option value="lof">Local Outlier Factor</option>
              </select>
            </div>

            <div class="setting-group">
              <label class="form-label">Contamination Rate</label>
              <input type="number" class="form-input" data-setting="contamination-rate" min="0" max="1" step="0.01" value="0.1">
            </div>

            <div class="setting-group">
              <label class="checkbox-label">
                <input type="checkbox" data-setting="auto-preprocess">
                <span class="checkmark"></span>
                Auto-preprocess data
              </label>
            </div>

            <div class="setting-group">
              <label class="checkbox-label">
                <input type="checkbox" data-setting="enable-notifications">
                <span class="checkmark"></span>
                Enable notifications
              </label>
            </div>
          </div>
        </div>
      </div>
    `;
  }

  // Bind event listeners
  bindEvents() {
    const element = this.element;

    // Tab switching
    element.addEventListener("click", (e) => {
      if (e.target.matches(".tab-button")) {
        this.switchTab(e.target);
      }
    });

    // File upload handling
    const fileInput = element.querySelector(".file-input");
    const dropZone = element.querySelector("[data-drop-zone]");

    fileInput?.addEventListener("change", (e) => {
      this.handleFileUpload(e.target.files[0]);
    });

    dropZone?.addEventListener("click", () => fileInput?.click());
    dropZone?.addEventListener("dragover", (e) => {
      e.preventDefault();
      dropZone.classList.add("drag-over");
    });

    dropZone?.addEventListener("dragleave", () => {
      dropZone.classList.remove("drag-over");
    });

    dropZone?.addEventListener("drop", (e) => {
      e.preventDefault();
      dropZone.classList.remove("drag-over");
      this.handleFileUpload(e.dataTransfer.files[0]);
    });

    // Action button handlers
    element.addEventListener("click", (e) => {
      const action = e.target.closest("[data-action]")?.dataset.action;
      if (action) {
        this.handleAction(action, e.target);
      }
    });

    // Algorithm selection
    element.addEventListener("click", (e) => {
      if (e.target.matches(".algorithm-card")) {
        this.selectAlgorithm(e.target);
      }
    });

    // Real-time threshold adjustment
    const thresholdSlider = element.querySelector("[data-threshold-slider]");
    thresholdSlider?.addEventListener("input", (e) => {
      this.updateThreshold(parseFloat(e.target.value));
    });

    // Keyboard shortcuts
    element.addEventListener("keydown", (e) => {
      if (e.ctrlKey || e.metaKey) {
        switch (e.key) {
          case "Enter":
            e.preventDefault();
            this.startDetection();
            break;
          case "r":
            e.preventDefault();
            this.resetDetector();
            break;
        }
      }
    });
  }

  // Load available algorithms
  async loadAlgorithms() {
    try {
      const response = await fetch("/api/algorithms");
      const algorithms = await response.json();

      this.renderAlgorithmGrid(algorithms);
      this.updateAlgorithmRecommendations(algorithms);
    } catch (error) {
      console.error("Failed to load algorithms:", error);
      this.showError("Failed to load detection algorithms");
    }
  }

  // Render algorithm selection grid
  renderAlgorithmGrid(algorithms) {
    const grid = this.element.querySelector("[data-algorithm-grid]");
    if (!grid) return;

    grid.innerHTML = algorithms
      .map(
        (algo) => `
      <div class="algorithm-card" data-algorithm="${algo.id}">
        <div class="algorithm-header">
          <h5 class="algorithm-name">${algo.name}</h5>
          <div class="algorithm-type">${algo.type}</div>
        </div>
        <div class="algorithm-description">${algo.description}</div>
        <div class="algorithm-metrics">
          <div class="metric">
            <span class="metric-label">Accuracy</span>
            <div class="metric-bar">
              <div class="metric-fill" style="width: ${algo.accuracy * 100}%"></div>
            </div>
          </div>
          <div class="metric">
            <span class="metric-label">Speed</span>
            <div class="metric-bar">
              <div class="metric-fill" style="width: ${algo.speed * 100}%"></div>
            </div>
          </div>
        </div>
        <div class="algorithm-tags">
          ${algo.tags.map((tag) => `<span class="tag">${tag}</span>`).join("")}
        </div>
      </div>
    `,
      )
      .join("");
  }

  // Handle file upload
  handleFileUpload(file) {
    if (!file) return;

    // Validate file type
    const fileExtension = file.name.split(".").pop().toLowerCase();
    if (!this.config.allowedFormats.includes(fileExtension)) {
      this.showError(`Unsupported file format: ${fileExtension}`);
      return;
    }

    // Validate file size
    if (file.size > this.config.maxFileSize) {
      this.showError(
        `File too large: ${(file.size / 1024 / 1024).toFixed(1)}MB (max: ${this.config.maxFileSize / 1024 / 1024}MB)`,
      );
      return;
    }

    // Show file info
    this.showFileInfo(file);

    // Upload file
    this.uploadFile(file);
  }

  // Upload file to server
  async uploadFile(file) {
    const formData = new FormData();
    formData.append("file", file);

    try {
      this.updateStepStatus("data-input", "processing");

      const response = await fetch("/api/datasets/upload", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Upload failed: ${response.statusText}`);
      }

      const result = await response.json();
      this.state.currentDataset = result.dataset;

      this.updateStepStatus("data-input", "completed");
      this.enableStep("algorithm-selection");

      this.announceToScreenReader(`File uploaded successfully: ${file.name}`);
    } catch (error) {
      console.error("Upload failed:", error);
      this.updateStepStatus("data-input", "error");
      this.showError(`Upload failed: ${error.message}`);
    }
  }

  // Start anomaly detection
  async startDetection() {
    if (!this.state.currentDataset || !this.state.selectedAlgorithm) {
      this.showError("Please select data and algorithm first");
      return;
    }

    this.state.isProcessing = true;
    this.updateStepStatus("execution", "processing");

    const progressContainer = this.element.querySelector(
      "[data-execution-progress]",
    );
    progressContainer?.classList.remove("hidden");

    try {
      const params = this.getAlgorithmParams();
      const options = this.getDetectionOptions();

      const response = await fetch("/api/anomaly-detection/detect", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          dataset_id: this.state.currentDataset.id,
          algorithm: this.state.selectedAlgorithm,
          parameters: params,
          options: options,
        }),
      });

      if (!response.ok) {
        throw new Error(`Detection failed: ${response.statusText}`);
      }

      const result = await response.json();
      this.state.results = result;

      this.updateStepStatus("execution", "completed");
      this.updateStepStatus("results", "completed");

      this.showResults(result);
      this.announceToScreenReader(
        `Detection completed. Found ${result.anomaly_count} anomalies.`,
      );
    } catch (error) {
      console.error("Detection failed:", error);
      this.updateStepStatus("execution", "error");
      this.showError(`Detection failed: ${error.message}`);
    } finally {
      this.state.isProcessing = false;
      progressContainer?.classList.add("hidden");
    }
  }

  // Show detection results
  showResults(results) {
    const container = this.element.querySelector("[data-results-container]");
    container?.classList.remove("hidden");

    // Update metrics
    this.updateMetric("total-samples", results.total_samples.toLocaleString());
    this.updateMetric(
      "anomalies-detected",
      results.anomaly_count.toLocaleString(),
    );
    this.updateMetric(
      "anomaly-rate",
      `${(results.anomaly_rate * 100).toFixed(2)}%`,
    );
    this.updateMetric("confidence-score", results.confidence_score.toFixed(3));

    // Create visualizations
    this.createResultsCharts(results);
  }

  // Create results charts
  createResultsCharts(results) {
    const chartContainer = this.element.querySelector("[data-chart-container]");
    if (!chartContainer) return;

    // Create scatter plot
    const scatterChart = new ChartComponent(chartContainer, {
      type: "scatter",
      data: results.visualization_data.scatter,
      options: {
        title: "Anomaly Detection Results",
        xAxis: { title: "Feature 1" },
        yAxis: { title: "Feature 2" },
        colorScale: {
          normal: "#22c55e",
          anomaly: "#ef4444",
        },
      },
    });

    this.charts.set("scatter", scatterChart);
  }

  // Utility methods
  switchTab(button) {
    const tabGroup = button.closest(
      ".input-tabs, .algorithm-tabs, .chart-tabs",
    );
    const tabName =
      button.dataset.tab || button.dataset.algoTab || button.dataset.chartTab;

    // Update button states
    tabGroup.querySelectorAll(".tab-button").forEach((btn) => {
      btn.classList.remove("active");
    });
    button.classList.add("active");

    // Update content visibility
    const contentContainer = tabGroup.nextElementSibling;
    contentContainer.querySelectorAll(".tab-content").forEach((content) => {
      content.classList.remove("active");
    });

    const targetContent = contentContainer.querySelector(
      `[data-tab-content="${tabName}"]`,
    );
    targetContent?.classList.add("active");
  }

  updateStepStatus(step, status) {
    const stepElement = this.element.querySelector(`[data-step="${step}"]`);
    const statusElement = stepElement?.querySelector("[data-status]");

    if (statusElement) {
      statusElement.className = `step-status status-${status}`;
      statusElement.textContent =
        status.charAt(0).toUpperCase() + status.slice(1);
      statusElement.dataset.status = status;
    }
  }

  enableStep(step) {
    const stepElement = this.element.querySelector(`[data-step="${step}"]`);
    stepElement?.classList.remove("disabled");
  }

  updateMetric(metric, value) {
    const metricElement = this.element.querySelector(
      `[data-metric="${metric}"]`,
    );
    if (metricElement) {
      metricElement.textContent = value;
    }
  }

  showError(message) {
    this.element.dispatchEvent(
      new CustomEvent("component:error", {
        detail: { component: "anomaly-detector", message },
      }),
    );
  }

  announceToScreenReader(message) {
    // Use the global app's announce method if available
    if (window.PynomalyApp) {
      window.PynomalyApp.announceToScreenReader(message);
    }
  }

  // Setup real-time capabilities
  setupRealTimeCapabilities() {
    if (this.config.realTime) {
      this.initWebSocket();
    }
  }

  // Initialize WebSocket for real-time updates
  initWebSocket() {
    if (!this.config.websocketUrl) return;

    try {
      this.websocket = new WebSocket(this.config.websocketUrl);

      this.websocket.onmessage = (event) => {
        const data = JSON.parse(event.data);
        this.handleRealTimeUpdate(data);
      };

      this.websocket.onerror = (error) => {
        console.error("WebSocket error:", error);
      };
    } catch (error) {
      console.error("Failed to initialize WebSocket:", error);
    }
  }

  // Handle real-time updates
  handleRealTimeUpdate(data) {
    if (data.type === "anomaly_detected") {
      this.showAnomalyAlert(data.anomaly);
    } else if (data.type === "model_updated") {
      this.refreshResults();
    }
  }

  // Handle action button clicks
  handleAction(action, button) {
    switch (action) {
      case "start-detection":
        this.startDetection();
        break;
      case "toggle-realtime":
        this.toggleRealTimeMode();
        break;
      case "export-results":
        this.exportResults();
        break;
      case "generate-report":
        this.generateReport();
        break;
      default:
        console.warn(`Unknown action: ${action}`);
    }
  }

  // Get algorithm parameters from form
  getAlgorithmParams() {
    const paramsForm = this.element.querySelector("[data-params-form]");
    if (!paramsForm) return {};

    const formData = new FormData(paramsForm);
    const params = {};

    for (const [key, value] of formData.entries()) {
      params[key] = value;
    }

    return params;
  }

  // Get detection options
  getDetectionOptions() {
    return {
      explain_results:
        this.element.querySelector('[data-option="explain-results"]')
          ?.checked || false,
      save_model:
        this.element.querySelector('[data-option="save-model"]')?.checked ||
        false,
      auto_threshold:
        this.element.querySelector('[data-option="auto-threshold"]')?.checked ||
        false,
    };
  }

  // Cleanup
  destroy() {
    if (this.websocket) {
      this.websocket.close();
    }

    this.charts.forEach((chart) => {
      if (chart.destroy) {
        chart.destroy();
      }
    });

    this.charts.clear();
  }
}

// Auto-initialize components
document.addEventListener("DOMContentLoaded", () => {
  document
    .querySelectorAll('[data-component="anomaly-detector"]')
    .forEach((element) => {
      new AnomalyDetector(element);
    });
});
\n\n// anomaly-timeline.js\n/**
 * Advanced Anomaly Timeline Component
 *
 * Interactive timeline visualization for anomaly detection events with D3.js
 * Features real-time updates, zooming, filtering, and detailed event inspection
 */

import * as d3 from "d3";

export class AnomalyTimeline {
  constructor(container, options = {}) {
    this.container = d3.select(container);
    this.options = {
      width: 800,
      height: 400,
      margin: { top: 20, right: 30, bottom: 40, left: 50 },
      animationDuration: 500,
      showTooltip: true,
      enableZoom: true,
      enableBrush: true,
      colorScheme: "category10",
      severityColors: {
        low: "#28a745",
        medium: "#ffc107",
        high: "#fd7e14",
        critical: "#dc3545",
      },
      ...options,
    };

    this.data = [];
    this.filteredData = [];
    this.selectedTimeRange = null;
    this.brushSelection = null;

    this.init();
  }

  init() {
    this.setupDimensions();
    this.createSVG();
    this.createScales();
    this.createAxes();
    this.createTooltip();
    this.createBrush();
    this.createZoom();
    this.setupEventListeners();
  }

  setupDimensions() {
    this.width =
      this.options.width - this.options.margin.left - this.options.margin.right;
    this.height =
      this.options.height -
      this.options.margin.top -
      this.options.margin.bottom;
  }

  createSVG() {
    // Clear existing content
    this.container.selectAll("*").remove();

    this.svg = this.container
      .append("svg")
      .attr("width", this.options.width)
      .attr("height", this.options.height)
      .attr("class", "anomaly-timeline");

    this.g = this.svg
      .append("g")
      .attr(
        "transform",
        `translate(${this.options.margin.left},${this.options.margin.top})`,
      );

    // Create clipping path for chart area
    this.svg
      .append("defs")
      .append("clipPath")
      .attr("id", "timeline-clip")
      .append("rect")
      .attr("width", this.width)
      .attr("height", this.height);

    this.chartArea = this.g
      .append("g")
      .attr("clip-path", "url(#timeline-clip)");
  }

  createScales() {
    this.xScale = d3.scaleTime().range([0, this.width]);

    this.yScale = d3.scaleLinear().range([this.height, 0]);

    this.colorScale = d3.scaleOrdinal(d3.schemeCategory10);

    this.sizeScale = d3.scaleSqrt().range([3, 15]);
  }

  createAxes() {
    this.xAxis = d3.axisBottom(this.xScale).tickFormat(d3.timeFormat("%H:%M"));

    this.yAxis = d3.axisLeft(this.yScale).tickFormat(d3.format(".2f"));

    this.g
      .append("g")
      .attr("class", "x-axis")
      .attr("transform", `translate(0,${this.height})`);

    this.g.append("g").attr("class", "y-axis");

    // Add axis labels
    this.g
      .append("text")
      .attr("class", "axis-label")
      .attr("transform", "rotate(-90)")
      .attr("y", 0 - this.options.margin.left)
      .attr("x", 0 - this.height / 2)
      .attr("dy", "1em")
      .style("text-anchor", "middle")
      .text("Anomaly Score");

    this.g
      .append("text")
      .attr("class", "axis-label")
      .attr(
        "transform",
        `translate(${this.width / 2}, ${this.height + this.options.margin.bottom})`,
      )
      .style("text-anchor", "middle")
      .text("Time");
  }

  createTooltip() {
    if (!this.options.showTooltip) return;

    this.tooltip = d3
      .select("body")
      .append("div")
      .attr("class", "anomaly-timeline-tooltip")
      .style("opacity", 0)
      .style("position", "absolute")
      .style("background", "rgba(0, 0, 0, 0.8)")
      .style("color", "white")
      .style("padding", "8px")
      .style("border-radius", "4px")
      .style("pointer-events", "none")
      .style("font-size", "12px")
      .style("z-index", "1000");
  }

  createBrush() {
    if (!this.options.enableBrush) return;

    this.brush = d3
      .brushX()
      .extent([
        [0, 0],
        [this.width, this.height],
      ])
      .on("start brush end", this.onBrush.bind(this));

    this.brushGroup = this.g
      .append("g")
      .attr("class", "brush")
      .call(this.brush);
  }

  createZoom() {
    if (!this.options.enableZoom) return;

    this.zoom = d3
      .zoom()
      .scaleExtent([1, 10])
      .translateExtent([
        [0, 0],
        [this.width, this.height],
      ])
      .on("zoom", this.onZoom.bind(this));

    this.svg.call(this.zoom);
  }

  setupEventListeners() {
    // Resize listener
    window.addEventListener(
      "resize",
      this.debounce(this.resize.bind(this), 250),
    );
  }

  setData(data) {
    this.data = data.map((d) => ({
      ...d,
      timestamp: new Date(d.timestamp),
      score: +d.score,
      severity: d.severity || this.getSeverityFromScore(d.score),
    }));

    this.filteredData = [...this.data];
    this.updateScales();
    this.render();
  }

  getSeverityFromScore(score) {
    if (score >= 0.8) return "critical";
    if (score >= 0.6) return "high";
    if (score >= 0.4) return "medium";
    return "low";
  }

  updateScales() {
    if (this.filteredData.length === 0) return;

    const timeExtent = d3.extent(this.filteredData, (d) => d.timestamp);
    const scoreExtent = d3.extent(this.filteredData, (d) => d.score);

    this.xScale.domain(timeExtent);
    this.yScale.domain([0, Math.max(1, scoreExtent[1])]);
    this.sizeScale.domain([0, scoreExtent[1]]);
  }

  render() {
    this.renderAxes();
    this.renderAnomalies();
    this.renderTrend();
    this.renderLegend();
  }

  renderAxes() {
    this.g
      .select(".x-axis")
      .transition()
      .duration(this.options.animationDuration)
      .call(this.xAxis);

    this.g
      .select(".y-axis")
      .transition()
      .duration(this.options.animationDuration)
      .call(this.yAxis);
  }

  renderAnomalies() {
    const circles = this.chartArea
      .selectAll(".anomaly-point")
      .data(this.filteredData, (d) => d.id || d.timestamp);

    // Enter
    const enter = circles
      .enter()
      .append("circle")
      .attr("class", "anomaly-point")
      .attr("cx", (d) => this.xScale(d.timestamp))
      .attr("cy", this.height)
      .attr("r", 0)
      .style("fill", (d) => this.options.severityColors[d.severity])
      .style("opacity", 0.7)
      .style("cursor", "pointer");

    // Update
    enter
      .merge(circles)
      .transition()
      .duration(this.options.animationDuration)
      .attr("cx", (d) => this.xScale(d.timestamp))
      .attr("cy", (d) => this.yScale(d.score))
      .attr("r", (d) => this.sizeScale(d.score))
      .style("fill", (d) => this.options.severityColors[d.severity]);

    // Exit
    circles
      .exit()
      .transition()
      .duration(this.options.animationDuration)
      .attr("r", 0)
      .style("opacity", 0)
      .remove();

    // Add event listeners
    this.chartArea
      .selectAll(".anomaly-point")
      .on("mouseover", this.showTooltip.bind(this))
      .on("mouseout", this.hideTooltip.bind(this))
      .on("click", this.onAnomalyClick.bind(this));
  }

  renderTrend() {
    if (this.filteredData.length < 2) return;

    const line = d3
      .line()
      .x((d) => this.xScale(d.timestamp))
      .y((d) => this.yScale(d.movingAverage || d.score))
      .curve(d3.curveMonotoneX);

    // Calculate moving average
    const windowSize = Math.max(3, Math.floor(this.filteredData.length / 10));
    const dataWithMA = this.calculateMovingAverage(
      this.filteredData,
      windowSize,
    );

    const trendPath = this.chartArea
      .selectAll(".trend-line")
      .data([dataWithMA]);

    trendPath
      .enter()
      .append("path")
      .attr("class", "trend-line")
      .style("fill", "none")
      .style("stroke", "#666")
      .style("stroke-width", 2)
      .style("opacity", 0.5)
      .merge(trendPath)
      .transition()
      .duration(this.options.animationDuration)
      .attr("d", line);

    trendPath.exit().remove();
  }

  renderLegend() {
    const legend = this.g.selectAll(".legend").data([null]);

    const legendEnter = legend
      .enter()
      .append("g")
      .attr("class", "legend")
      .attr("transform", `translate(${this.width - 120}, 20)`);

    const severities = Object.keys(this.options.severityColors);
    const legendItems = legendEnter.selectAll(".legend-item").data(severities);

    const legendItem = legendItems
      .enter()
      .append("g")
      .attr("class", "legend-item")
      .attr("transform", (d, i) => `translate(0, ${i * 20})`);

    legendItem
      .append("circle")
      .attr("r", 6)
      .style("fill", (d) => this.options.severityColors[d]);

    legendItem
      .append("text")
      .attr("x", 12)
      .attr("y", 5)
      .style("font-size", "12px")
      .text((d) => d.charAt(0).toUpperCase() + d.slice(1));
  }

  calculateMovingAverage(data, windowSize) {
    return data.map((item, index) => {
      const start = Math.max(0, index - windowSize + 1);
      const window = data.slice(start, index + 1);
      const average =
        window.reduce((sum, d) => sum + d.score, 0) / window.length;
      return { ...item, movingAverage: average };
    });
  }

  showTooltip(event, d) {
    if (!this.tooltip) return;

    this.tooltip.transition().duration(200).style("opacity", 0.9);

    this.tooltip
      .html(
        `
            <strong>Anomaly Details</strong><br/>
            Time: ${d3.timeFormat("%Y-%m-%d %H:%M:%S")(d.timestamp)}<br/>
            Score: ${d.score.toFixed(3)}<br/>
            Severity: ${d.severity}<br/>
            ${d.description ? `Description: ${d.description}<br/>` : ""}
            ${d.features ? `Features: ${d.features.join(", ")}<br/>` : ""}
        `,
      )
      .style("left", event.pageX + 10 + "px")
      .style("top", event.pageY - 28 + "px");
  }

  hideTooltip() {
    if (!this.tooltip) return;

    this.tooltip.transition().duration(500).style("opacity", 0);
  }

  onAnomalyClick(event, d) {
    // Emit custom event for anomaly selection
    this.container.node().dispatchEvent(
      new CustomEvent("anomalySelected", {
        detail: { anomaly: d, timeline: this },
      }),
    );
  }

  onBrush(event) {
    if (!event.selection) {
      this.brushSelection = null;
      this.filteredData = [...this.data];
    } else {
      this.brushSelection = event.selection.map(this.xScale.invert);
      this.filteredData = this.data.filter(
        (d) =>
          d.timestamp >= this.brushSelection[0] &&
          d.timestamp <= this.brushSelection[1],
      );
    }

    this.updateScales();
    this.renderAnomalies();
    this.renderTrend();

    // Emit filter event
    this.container.node().dispatchEvent(
      new CustomEvent("timeRangeFiltered", {
        detail: { range: this.brushSelection, data: this.filteredData },
      }),
    );
  }

  onZoom(event) {
    const newXScale = event.transform.rescaleX(this.xScale);

    this.g.select(".x-axis").call(this.xAxis.scale(newXScale));

    this.chartArea
      .selectAll(".anomaly-point")
      .attr("cx", (d) => newXScale(d.timestamp));

    this.chartArea.selectAll(".trend-line").attr(
      "d",
      d3
        .line()
        .x((d) => newXScale(d.timestamp))
        .y((d) => this.yScale(d.movingAverage || d.score))
        .curve(d3.curveMonotoneX),
    );
  }

  resize() {
    const containerNode = this.container.node();
    const newWidth = containerNode.getBoundingClientRect().width;

    if (newWidth !== this.options.width) {
      this.options.width = newWidth;
      this.setupDimensions();
      this.createSVG();
      this.createScales();
      this.createAxes();
      this.updateScales();
      this.render();
    }
  }

  // Public methods for external control
  filterByTimeRange(startTime, endTime) {
    this.filteredData = this.data.filter(
      (d) => d.timestamp >= startTime && d.timestamp <= endTime,
    );
    this.updateScales();
    this.render();
  }

  filterBySeverity(severities) {
    this.filteredData = this.data.filter((d) =>
      severities.includes(d.severity),
    );
    this.updateScales();
    this.render();
  }

  highlightAnomalies(anomalyIds) {
    this.chartArea
      .selectAll(".anomaly-point")
      .style("stroke", (d) => (anomalyIds.includes(d.id) ? "#000" : "none"))
      .style("stroke-width", (d) => (anomalyIds.includes(d.id) ? 2 : 0));
  }

  addRealTimeData(newData) {
    const processedData = newData.map((d) => ({
      ...d,
      timestamp: new Date(d.timestamp),
      score: +d.score,
      severity: d.severity || this.getSeverityFromScore(d.score),
    }));

    this.data.push(...processedData);

    // Keep only recent data if too many points
    const maxPoints = 1000;
    if (this.data.length > maxPoints) {
      this.data = this.data.slice(-maxPoints);
    }

    this.filteredData = [...this.data];
    this.updateScales();
    this.render();
  }

  exportData() {
    return {
      data: this.data,
      filteredData: this.filteredData,
      timeRange: this.brushSelection,
      config: this.options,
    };
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

  destroy() {
    if (this.tooltip) {
      this.tooltip.remove();
    }
    window.removeEventListener("resize", this.resize);
    this.container.selectAll("*").remove();
  }
}

// CSS styles (to be included in stylesheet)
export const timelineStyles = `
.anomaly-timeline {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}

.anomaly-timeline .axis-label {
    font-size: 12px;
    font-weight: 500;
}

.anomaly-timeline .legend-item text {
    font-size: 11px;
    fill: #666;
}

.anomaly-timeline .anomaly-point {
    transition: all 0.2s ease;
}

.anomaly-timeline .anomaly-point:hover {
    stroke: #000;
    stroke-width: 2px;
    transform: scale(1.1);
}

.anomaly-timeline .trend-line {
    filter: drop-shadow(0 0 2px rgba(102, 102, 102, 0.3));
}

.anomaly-timeline .brush .selection {
    fill: rgba(59, 130, 246, 0.3);
    stroke: #3b82f6;
}

.anomaly-timeline-tooltip {
    max-width: 300px;
    word-wrap: break-word;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}
`;
\n\n// automl-interface.js\n/**
 * AutoML Interface Component
 * User interface for automated model training and hyperparameter optimization
 * Provides intuitive workflow for enterprise machine learning automation
 */

/**
 * AutoML Configuration Wizard
 * Guides users through AutoML setup with intelligent defaults
 */
class AutoMLConfigWizard {
  constructor(container, options = {}) {
    this.container = container;
    this.options = {
      enableAdvancedOptions: true,
      showExpertMode: false,
      enableTemplates: true,
      ...options,
    };

    this.currentStep = 0;
    this.totalSteps = 5;
    this.config = {};
    this.templates = this.getConfigTemplates();

    this.eventListeners = new Map();
    this.init();
  }

  init() {
    this.createWizardStructure();
    this.setupEventListeners();
    this.showStep(0);
  }

  createWizardStructure() {
    this.container.innerHTML = `
      <div class="automl-wizard">
        <div class="wizard-header">
          <h2 class="wizard-title">AutoML Configuration Wizard</h2>
          <div class="wizard-progress">
            <div class="progress-bar">
              <div class="progress-fill" style="width: 0%"></div>
            </div>
            <span class="progress-text">Step 1 of ${this.totalSteps}</span>
          </div>
        </div>

        <div class="wizard-content">
          <!-- Step content will be dynamically inserted here -->
        </div>

        <div class="wizard-actions">
          <button class="btn btn-secondary wizard-prev" disabled>Previous</button>
          <button class="btn btn-primary wizard-next">Next</button>
          <button class="btn btn-success wizard-finish" style="display: none;">Start AutoML</button>
        </div>
      </div>
    `;

    this.wizardContent = this.container.querySelector(".wizard-content");
    this.prevButton = this.container.querySelector(".wizard-prev");
    this.nextButton = this.container.querySelector(".wizard-next");
    this.finishButton = this.container.querySelector(".wizard-finish");
    this.progressFill = this.container.querySelector(".progress-fill");
    this.progressText = this.container.querySelector(".progress-text");
  }

  setupEventListeners() {
    this.prevButton.addEventListener("click", () => this.previousStep());
    this.nextButton.addEventListener("click", () => this.nextStep());
    this.finishButton.addEventListener("click", () => this.finishWizard());
  }

  showStep(stepIndex) {
    this.currentStep = stepIndex;
    this.updateProgress();
    this.updateButtons();

    switch (stepIndex) {
      case 0:
        this.showDatasetStep();
        break;
      case 1:
        this.showTemplateStep();
        break;
      case 2:
        this.showAlgorithmStep();
        break;
      case 3:
        this.showOptimizationStep();
        break;
      case 4:
        this.showSummaryStep();
        break;
    }
  }

  showDatasetStep() {
    this.wizardContent.innerHTML = `
      <div class="wizard-step" data-step="0">
        <h3>Dataset Configuration</h3>
        <p>Configure your dataset and target variable for anomaly detection.</p>

        <div class="form-group">
          <label for="dataset-source">Data Source</label>
          <select id="dataset-source" class="form-control">
            <option value="upload">Upload CSV File</option>
            <option value="database">Database Connection</option>
            <option value="api">API Endpoint</option>
            <option value="streaming">Real-time Stream</option>
          </select>
        </div>

        <div class="form-group" id="file-upload-group">
          <label for="dataset-file">Upload Dataset</label>
          <input type="file" id="dataset-file" class="form-control" accept=".csv,.json,.parquet">
          <small class="form-text text-muted">Supported formats: CSV, JSON, Parquet</small>
        </div>

        <div class="form-group">
          <label for="target-column">Target Column (Optional)</label>
          <select id="target-column" class="form-control">
            <option value="">Auto-detect anomalies (unsupervised)</option>
            <option value="is_anomaly">is_anomaly</option>
            <option value="label">label</option>
            <option value="target">target</option>
            <option value="outlier">outlier</option>
          </select>
          <small class="form-text text-muted">Leave empty for unsupervised anomaly detection</small>
        </div>

        <div class="form-group">
          <label for="data-preview">Data Preview</label>
          <div id="data-preview" class="data-preview-container">
            <div class="preview-placeholder">
              <i class="fas fa-upload"></i>
              <p>Upload a dataset to see preview</p>
            </div>
          </div>
        </div>
      </div>
    `;

    this.setupDatasetStepListeners();
  }

  setupDatasetStepListeners() {
    const fileInput = this.wizardContent.querySelector("#dataset-file");
    const dataPreview = this.wizardContent.querySelector("#data-preview");

    fileInput?.addEventListener("change", (event) => {
      const file = event.target.files[0];
      if (file) {
        this.loadDatasetPreview(file, dataPreview);
      }
    });
  }

  async loadDatasetPreview(file, container) {
    try {
      container.innerHTML =
        '<div class="loading-spinner">Loading preview...</div>';

      // Simulate file reading and preview generation
      await new Promise((resolve) => setTimeout(resolve, 1000));

      const mockPreview = this.generateMockDataPreview();
      container.innerHTML = `
        <div class="data-preview">
          <div class="preview-stats">
            <div class="stat">
              <span class="stat-value">${mockPreview.rows}</span>
              <span class="stat-label">Rows</span>
            </div>
            <div class="stat">
              <span class="stat-value">${mockPreview.columns}</span>
              <span class="stat-label">Columns</span>
            </div>
            <div class="stat">
              <span class="stat-value">${mockPreview.missing}%</span>
              <span class="stat-label">Missing</span>
            </div>
          </div>
          <div class="preview-table">
            <table class="table table-sm">
              <thead>
                <tr>
                  ${mockPreview.headers.map((h) => `<th>${h}</th>`).join("")}
                </tr>
              </thead>
              <tbody>
                ${mockPreview.rows_data
                  .map(
                    (row) =>
                      `<tr>${row.map((cell) => `<td>${cell}</td>`).join("")}</tr>`,
                  )
                  .join("")}
              </tbody>
            </table>
          </div>
        </div>
      `;

      // Store dataset info in config
      this.config.dataset = {
        filename: file.name,
        size: file.size,
        rows: mockPreview.rows,
        columns: mockPreview.columns,
        preview: mockPreview,
      };
    } catch (error) {
      container.innerHTML = `<div class="error-message">Error loading file: ${error.message}</div>`;
    }
  }

  showTemplateStep() {
    this.wizardContent.innerHTML = `
      <div class="wizard-step" data-step="1">
        <h3>Configuration Template</h3>
        <p>Choose a pre-configured template or start with custom settings.</p>

        <div class="template-grid">
          ${this.templates
            .map(
              (template) => `
            <div class="template-card" data-template="${template.id}">
              <div class="template-header">
                <h4>${template.name}</h4>
                <span class="template-badge ${template.complexity}">${template.complexity}</span>
              </div>
              <p class="template-description">${template.description}</p>
              <div class="template-features">
                <h5>Features:</h5>
                <ul>
                  ${template.features.map((feature) => `<li>${feature}</li>`).join("")}
                </ul>
              </div>
              <div class="template-specs">
                <div class="spec">
                  <span class="spec-label">Training Time:</span>
                  <span class="spec-value">${template.estimatedTime}</span>
                </div>
                <div class="spec">
                  <span class="spec-label">Algorithms:</span>
                  <span class="spec-value">${template.algorithms.length}</span>
                </div>
              </div>
            </div>
          `,
            )
            .join("")}
        </div>

        <div class="custom-option">
          <div class="template-card custom-template" data-template="custom">
            <div class="template-header">
              <h4>Custom Configuration</h4>
              <span class="template-badge expert">Expert</span>
            </div>
            <p class="template-description">Create a custom configuration with full control over all parameters.</p>
            <div class="template-features">
              <h5>Features:</h5>
              <ul>
                <li>Full parameter control</li>
                <li>Advanced optimization</li>
                <li>Custom metrics</li>
                <li>Expert tuning</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    `;

    this.setupTemplateStepListeners();
  }

  setupTemplateStepListeners() {
    const templateCards = this.wizardContent.querySelectorAll(".template-card");

    templateCards.forEach((card) => {
      card.addEventListener("click", () => {
        // Remove previous selection
        templateCards.forEach((c) => c.classList.remove("selected"));

        // Select current template
        card.classList.add("selected");

        const templateId = card.dataset.template;
        if (templateId === "custom") {
          this.config.template = "custom";
        } else {
          const template = this.templates.find((t) => t.id === templateId);
          this.config.template = template;
          this.applyTemplate(template);
        }
      });
    });
  }

  showAlgorithmStep() {
    this.wizardContent.innerHTML = `
      <div class="wizard-step" data-step="2">
        <h3>Algorithm Selection</h3>
        <p>Choose which anomaly detection algorithms to include in the AutoML search.</p>

        <div class="algorithm-categories">
          <div class="category">
            <h4>
              <input type="checkbox" id="cat-statistical" checked>
              <label for="cat-statistical">Statistical Methods</label>
            </h4>
            <div class="algorithm-list" data-category="statistical">
              <div class="algorithm-item">
                <input type="checkbox" id="alg-isolation-forest" checked>
                <label for="alg-isolation-forest">
                  <span class="algorithm-name">Isolation Forest</span>
                  <span class="algorithm-description">Tree-based ensemble for outlier detection</span>
                  <span class="algorithm-complexity">Fast</span>
                </label>
              </div>
              <div class="algorithm-item">
                <input type="checkbox" id="alg-lof" checked>
                <label for="alg-lof">
                  <span class="algorithm-name">Local Outlier Factor</span>
                  <span class="algorithm-description">Density-based local outlier detection</span>
                  <span class="algorithm-complexity">Medium</span>
                </label>
              </div>
              <div class="algorithm-item">
                <input type="checkbox" id="alg-elliptic-envelope">
                <label for="alg-elliptic-envelope">
                  <span class="algorithm-name">Elliptic Envelope</span>
                  <span class="algorithm-description">Gaussian distribution assumption</span>
                  <span class="algorithm-complexity">Fast</span>
                </label>
              </div>
            </div>
          </div>

          <div class="category">
            <h4>
              <input type="checkbox" id="cat-neural" checked>
              <label for="cat-neural">Neural Networks</label>
            </h4>
            <div class="algorithm-list" data-category="neural">
              <div class="algorithm-item">
                <input type="checkbox" id="alg-autoencoder" checked>
                <label for="alg-autoencoder">
                  <span class="algorithm-name">Autoencoder</span>
                  <span class="algorithm-description">Neural network reconstruction error</span>
                  <span class="algorithm-complexity">Slow</span>
                </label>
              </div>
              <div class="algorithm-item">
                <input type="checkbox" id="alg-deep-svdd">
                <label for="alg-deep-svdd">
                  <span class="algorithm-name">Deep SVDD</span>
                  <span class="algorithm-description">Deep one-class classification</span>
                  <span class="algorithm-complexity">Slow</span>
                </label>
              </div>
            </div>
          </div>

          <div class="category">
            <h4>
              <input type="checkbox" id="cat-ensemble" checked>
              <label for="cat-ensemble">Ensemble Methods</label>
            </h4>
            <div class="algorithm-list" data-category="ensemble">
              <div class="algorithm-item">
                <input type="checkbox" id="alg-feature-bagging" checked>
                <label for="alg-feature-bagging">
                  <span class="algorithm-name">Feature Bagging</span>
                  <span class="algorithm-description">Ensemble of base detectors</span>
                  <span class="algorithm-complexity">Medium</span>
                </label>
              </div>
              <div class="algorithm-item">
                <input type="checkbox" id="alg-copod">
                <label for="alg-copod">
                  <span class="algorithm-name">COPOD</span>
                  <span class="algorithm-description">Copula-based outlier detection</span>
                  <span class="algorithm-complexity">Fast</span>
                </label>
              </div>
            </div>
          </div>
        </div>

        <div class="algorithm-summary">
          <h4>Selection Summary</h4>
          <div class="summary-stats">
            <div class="stat">
              <span class="stat-value" id="selected-algorithms">0</span>
              <span class="stat-label">Selected Algorithms</span>
            </div>
            <div class="stat">
              <span class="stat-value" id="estimated-time">0</span>
              <span class="stat-label">Est. Training Time</span>
            </div>
          </div>
        </div>
      </div>
    `;

    this.setupAlgorithmStepListeners();
  }

  setupAlgorithmStepListeners() {
    const algorithmCheckboxes = this.wizardContent.querySelectorAll(
      '.algorithm-item input[type="checkbox"]',
    );
    const categoryCheckboxes = this.wizardContent.querySelectorAll(
      '.category > h4 input[type="checkbox"]',
    );

    // Update selection summary
    const updateSummary = () => {
      const selectedCount = this.wizardContent.querySelectorAll(
        '.algorithm-item input[type="checkbox"]:checked',
      ).length;
      const estimatedTime = selectedCount * 5; // 5 minutes per algorithm estimate

      this.wizardContent.querySelector("#selected-algorithms").textContent =
        selectedCount;
      this.wizardContent.querySelector("#estimated-time").textContent =
        `${estimatedTime}min`;

      // Store selected algorithms in config
      const selectedAlgorithms = Array.from(algorithmCheckboxes)
        .filter((cb) => cb.checked)
        .map((cb) => cb.id.replace("alg-", "").replace("-", "_"));

      this.config.algorithms = selectedAlgorithms;
    };

    // Category checkbox handlers
    categoryCheckboxes.forEach((catCheckbox) => {
      catCheckbox.addEventListener("change", () => {
        const category = catCheckbox.id.replace("cat-", "");
        const categoryAlgorithms = this.wizardContent.querySelectorAll(
          `[data-category="${category}"] input[type="checkbox"]`,
        );

        categoryAlgorithms.forEach((algCheckbox) => {
          algCheckbox.checked = catCheckbox.checked;
        });

        updateSummary();
      });
    });

    // Algorithm checkbox handlers
    algorithmCheckboxes.forEach((algCheckbox) => {
      algCheckbox.addEventListener("change", updateSummary);
    });

    // Initial summary update
    updateSummary();
  }

  showOptimizationStep() {
    this.wizardContent.innerHTML = `
      <div class="wizard-step" data-step="3">
        <h3>Optimization Settings</h3>
        <p>Configure hyperparameter optimization and resource limits.</p>

        <div class="optimization-grid">
          <div class="optimization-section">
            <h4>Hyperparameter Optimization</h4>

            <div class="form-group">
              <label for="optimization-algorithm">Optimization Algorithm</label>
              <select id="optimization-algorithm" class="form-control">
                <option value="bayesian" selected>Bayesian Optimization (Recommended)</option>
                <option value="random_search">Random Search</option>
                <option value="grid_search">Grid Search</option>
                <option value="evolutionary">Evolutionary Algorithm</option>
                <option value="optuna">Optuna TPE</option>
              </select>
            </div>

            <div class="form-group">
              <label for="max-evaluations">Maximum Evaluations</label>
              <input type="range" id="max-evaluations" class="form-control-range"
                     min="10" max="500" value="100" step="10">
              <div class="range-labels">
                <span>10 (Fast)</span>
                <span id="eval-value">100</span>
                <span>500 (Thorough)</span>
              </div>
            </div>

            <div class="form-group">
              <label for="optimization-timeout">Timeout (minutes)</label>
              <input type="number" id="optimization-timeout" class="form-control"
                     value="60" min="5" max="480">
            </div>
          </div>

          <div class="optimization-section">
            <h4>Cross-Validation</h4>

            <div class="form-group">
              <label for="cv-folds">Cross-Validation Folds</label>
              <select id="cv-folds" class="form-control">
                <option value="3">3-Fold (Fast)</option>
                <option value="5" selected>5-Fold (Recommended)</option>
                <option value="10">10-Fold (Thorough)</option>
              </select>
            </div>

            <div class="form-group">
              <label for="scoring-metric">Scoring Metric</label>
              <select id="scoring-metric" class="form-control">
                <option value="roc_auc" selected>ROC AUC</option>
                <option value="average_precision">Average Precision</option>
                <option value="f1_score">F1 Score</option>
                <option value="precision">Precision</option>
                <option value="recall">Recall</option>
              </select>
            </div>
          </div>

          <div class="optimization-section">
            <h4>Resource Limits</h4>

            <div class="form-group">
              <label for="max-training-time">Max Training Time (minutes)</label>
              <input type="number" id="max-training-time" class="form-control"
                     value="120" min="10" max="1440">
            </div>

            <div class="form-group">
              <label for="memory-limit">Memory Limit (GB)</label>
              <input type="number" id="memory-limit" class="form-control"
                     value="8" min="1" max="64" step="1">
            </div>

            <div class="form-group">
              <div class="form-check">
                <input type="checkbox" id="gpu-enabled" class="form-check-input">
                <label for="gpu-enabled" class="form-check-label">
                  Enable GPU Acceleration (if available)
                </label>
              </div>
            </div>

            <div class="form-group">
              <label for="n-jobs">Parallel Jobs</label>
              <select id="n-jobs" class="form-control">
                <option value="1">1 (Single-threaded)</option>
                <option value="2">2 cores</option>
                <option value="4">4 cores</option>
                <option value="-1" selected>All available cores</option>
              </select>
            </div>
          </div>
        </div>

        <div class="estimation-panel">
          <h4>Training Estimation</h4>
          <div class="estimation-grid">
            <div class="estimation-item">
              <span class="estimation-label">Estimated Duration:</span>
              <span class="estimation-value" id="estimated-duration">~2-3 hours</span>
            </div>
            <div class="estimation-item">
              <span class="estimation-label">Memory Usage:</span>
              <span class="estimation-value" id="estimated-memory">~4-6 GB</span>
            </div>
            <div class="estimation-item">
              <span class="estimation-label">Total Trials:</span>
              <span class="estimation-value" id="estimated-trials">~500</span>
            </div>
          </div>
        </div>
      </div>
    `;

    this.setupOptimizationStepListeners();
  }

  setupOptimizationStepListeners() {
    const maxEvaluations = this.wizardContent.querySelector("#max-evaluations");
    const evalValue = this.wizardContent.querySelector("#eval-value");

    maxEvaluations.addEventListener("input", () => {
      evalValue.textContent = maxEvaluations.value;
      this.updateEstimations();
    });

    // Store optimization settings in config
    const formElements = this.wizardContent.querySelectorAll("input, select");
    formElements.forEach((element) => {
      element.addEventListener("change", () => {
        this.updateOptimizationConfig();
        this.updateEstimations();
      });
    });

    this.updateOptimizationConfig();
    this.updateEstimations();
  }

  updateOptimizationConfig() {
    const getValue = (id) => {
      const element = this.wizardContent.querySelector(`#${id}`);
      if (element.type === "checkbox") return element.checked;
      if (element.type === "number" || element.type === "range")
        return parseInt(element.value);
      return element.value;
    };

    this.config.optimization = {
      algorithm: getValue("optimization-algorithm"),
      max_evaluations: getValue("max-evaluations"),
      timeout_minutes: getValue("optimization-timeout"),
      cv_folds: getValue("cv-folds"),
      scoring_metric: getValue("scoring-metric"),
      max_training_time: getValue("max-training-time"),
      memory_limit: getValue("memory-limit"),
      gpu_enabled: getValue("gpu-enabled"),
      n_jobs: getValue("n-jobs"),
    };
  }

  updateEstimations() {
    const algorithms = this.config.algorithms?.length || 5;
    const evaluations = this.config.optimization?.max_evaluations || 100;
    const parallelJobs =
      this.config.optimization?.n_jobs === "-1"
        ? 4
        : parseInt(this.config.optimization?.n_jobs || 1);

    // Rough estimation formulas
    const totalTrials = algorithms * evaluations;
    const estimatedMinutes = Math.ceil((totalTrials * 2) / parallelJobs); // 2 minutes per trial
    const estimatedHours = Math.floor(estimatedMinutes / 60);
    const remainingMinutes = estimatedMinutes % 60;

    const durationText =
      estimatedHours > 0
        ? `~${estimatedHours}h ${remainingMinutes}m`
        : `~${estimatedMinutes}m`;

    const memoryUsage = Math.min(
      this.config.optimization?.memory_limit || 8,
      algorithms * 1.5,
    );

    this.wizardContent.querySelector("#estimated-duration").textContent =
      durationText;
    this.wizardContent.querySelector("#estimated-memory").textContent =
      `~${memoryUsage.toFixed(1)} GB`;
    this.wizardContent.querySelector("#estimated-trials").textContent =
      `~${totalTrials}`;
  }

  showSummaryStep() {
    const configSummary = this.generateConfigSummary();

    this.wizardContent.innerHTML = `
      <div class="wizard-step" data-step="4">
        <h3>Configuration Summary</h3>
        <p>Review your AutoML configuration before starting the training process.</p>

        <div class="summary-grid">
          <div class="summary-section">
            <h4>Dataset</h4>
            <div class="summary-content">
              <div class="summary-item">
                <span class="summary-label">Source:</span>
                <span class="summary-value">${configSummary.dataset.source}</span>
              </div>
              <div class="summary-item">
                <span class="summary-label">Samples:</span>
                <span class="summary-value">${configSummary.dataset.samples}</span>
              </div>
              <div class="summary-item">
                <span class="summary-label">Features:</span>
                <span class="summary-value">${configSummary.dataset.features}</span>
              </div>
              <div class="summary-item">
                <span class="summary-label">Target:</span>
                <span class="summary-value">${configSummary.dataset.target}</span>
              </div>
            </div>
          </div>

          <div class="summary-section">
            <h4>Algorithms</h4>
            <div class="summary-content">
              <div class="algorithm-chips">
                ${configSummary.algorithms
                  .map((alg) => `<span class="algorithm-chip">${alg}</span>`)
                  .join("")}
              </div>
              <div class="summary-item">
                <span class="summary-label">Total:</span>
                <span class="summary-value">${configSummary.algorithms.length} algorithms</span>
              </div>
            </div>
          </div>

          <div class="summary-section">
            <h4>Optimization</h4>
            <div class="summary-content">
              <div class="summary-item">
                <span class="summary-label">Algorithm:</span>
                <span class="summary-value">${configSummary.optimization.algorithm}</span>
              </div>
              <div class="summary-item">
                <span class="summary-label">Max Evaluations:</span>
                <span class="summary-value">${configSummary.optimization.evaluations}</span>
              </div>
              <div class="summary-item">
                <span class="summary-label">CV Folds:</span>
                <span class="summary-value">${configSummary.optimization.cv_folds}</span>
              </div>
              <div class="summary-item">
                <span class="summary-label">Scoring:</span>
                <span class="summary-value">${configSummary.optimization.scoring}</span>
              </div>
            </div>
          </div>

          <div class="summary-section">
            <h4>Resources</h4>
            <div class="summary-content">
              <div class="summary-item">
                <span class="summary-label">Max Time:</span>
                <span class="summary-value">${configSummary.resources.max_time}</span>
              </div>
              <div class="summary-item">
                <span class="summary-label">Memory Limit:</span>
                <span class="summary-value">${configSummary.resources.memory}</span>
              </div>
              <div class="summary-item">
                <span class="summary-label">Parallel Jobs:</span>
                <span class="summary-value">${configSummary.resources.parallel_jobs}</span>
              </div>
              <div class="summary-item">
                <span class="summary-label">GPU:</span>
                <span class="summary-value">${configSummary.resources.gpu ? "Enabled" : "Disabled"}</span>
              </div>
            </div>
          </div>
        </div>

        <div class="estimation-summary">
          <h4>Training Estimation</h4>
          <div class="estimation-highlight">
            <div class="estimation-main">
              <span class="estimation-duration">${configSummary.estimation.duration}</span>
              <span class="estimation-label">Estimated Training Time</span>
            </div>
            <div class="estimation-details">
              <span>~${configSummary.estimation.trials} total trials</span>
              <span>~${configSummary.estimation.memory} memory usage</span>
            </div>
          </div>
        </div>

        <div class="configuration-export">
          <h4>Configuration Export</h4>
          <div class="export-actions">
            <button class="btn btn-outline-secondary" id="export-config">
              <i class="fas fa-download"></i> Export Configuration
            </button>
            <button class="btn btn-outline-secondary" id="save-template">
              <i class="fas fa-save"></i> Save as Template
            </button>
          </div>
        </div>
      </div>
    `;

    this.setupSummaryStepListeners();
  }

  setupSummaryStepListeners() {
    const exportButton = this.wizardContent.querySelector("#export-config");
    const saveTemplateButton =
      this.wizardContent.querySelector("#save-template");

    exportButton?.addEventListener("click", () => {
      this.exportConfiguration();
    });

    saveTemplateButton?.addEventListener("click", () => {
      this.saveAsTemplate();
    });
  }

  generateConfigSummary() {
    return {
      dataset: {
        source: this.config.dataset?.filename || "Not specified",
        samples: this.config.dataset?.rows || "Unknown",
        features: this.config.dataset?.columns || "Unknown",
        target: "Auto-detect (unsupervised)",
      },
      algorithms: this.config.algorithms || [],
      optimization: {
        algorithm: this.config.optimization?.algorithm || "bayesian",
        evaluations: this.config.optimization?.max_evaluations || 100,
        cv_folds: this.config.optimization?.cv_folds || 5,
        scoring: this.config.optimization?.scoring_metric || "roc_auc",
      },
      resources: {
        max_time: `${this.config.optimization?.max_training_time || 120} minutes`,
        memory: `${this.config.optimization?.memory_limit || 8} GB`,
        parallel_jobs:
          this.config.optimization?.n_jobs === "-1"
            ? "All cores"
            : this.config.optimization?.n_jobs || 1,
        gpu: this.config.optimization?.gpu_enabled || false,
      },
      estimation: {
        duration: "~2-3 hours",
        trials: "500",
        memory: "4-6 GB",
      },
    };
  }

  updateProgress() {
    const progressPercent = ((this.currentStep + 1) / this.totalSteps) * 100;
    this.progressFill.style.width = `${progressPercent}%`;
    this.progressText.textContent = `Step ${this.currentStep + 1} of ${this.totalSteps}`;
  }

  updateButtons() {
    this.prevButton.disabled = this.currentStep === 0;
    this.nextButton.style.display =
      this.currentStep === this.totalSteps - 1 ? "none" : "inline-block";
    this.finishButton.style.display =
      this.currentStep === this.totalSteps - 1 ? "inline-block" : "none";
  }

  nextStep() {
    if (this.validateCurrentStep() && this.currentStep < this.totalSteps - 1) {
      this.showStep(this.currentStep + 1);
    }
  }

  previousStep() {
    if (this.currentStep > 0) {
      this.showStep(this.currentStep - 1);
    }
  }

  validateCurrentStep() {
    // Add validation logic for each step
    switch (this.currentStep) {
      case 0: // Dataset step
        return this.config.dataset || true; // Allow proceeding even without file for demo
      case 1: // Template step
        return this.config.template !== undefined;
      case 2: // Algorithm step
        return this.config.algorithms && this.config.algorithms.length > 0;
      case 3: // Optimization step
        return this.config.optimization !== undefined;
      default:
        return true;
    }
  }

  finishWizard() {
    const finalConfig = this.buildFinalConfig();
    this.emit("wizard-complete", { config: finalConfig });
  }

  buildFinalConfig() {
    // Build complete AutoML configuration
    return {
      dataset: this.config.dataset,
      model_search: {
        algorithms: this.config.algorithms,
        max_trials: 50,
        early_stopping: true,
      },
      hyperparameter_optimization: {
        algorithm: this.config.optimization?.algorithm || "bayesian",
        max_evaluations: this.config.optimization?.max_evaluations || 100,
        timeout_minutes: this.config.optimization?.timeout_minutes || 60,
        cv_folds: this.config.optimization?.cv_folds || 5,
        scoring_metric: this.config.optimization?.scoring_metric || "roc_auc",
      },
      performance: {
        max_training_time_minutes:
          this.config.optimization?.max_training_time || 120,
        memory_limit_gb: this.config.optimization?.memory_limit || 8,
        gpu_enabled: this.config.optimization?.gpu_enabled || false,
        n_jobs: this.config.optimization?.n_jobs || -1,
      },
      ensemble: {
        enable: true,
        strategy: "ensemble",
        max_models: 5,
      },
      validation: {
        test_size: 0.2,
        cross_validation: true,
        cv_folds: this.config.optimization?.cv_folds || 5,
      },
    };
  }

  getConfigTemplates() {
    return [
      {
        id: "quick",
        name: "Quick Start",
        complexity: "beginner",
        description:
          "Fast anomaly detection with basic algorithms and minimal tuning.",
        features: [
          "Fast training (~30 minutes)",
          "Basic algorithms",
          "Minimal resource usage",
          "Good baseline performance",
        ],
        algorithms: ["isolation_forest", "local_outlier_factor"],
        estimatedTime: "30 minutes",
      },
      {
        id: "balanced",
        name: "Balanced Performance",
        complexity: "intermediate",
        description:
          "Balance between training time and model performance with moderate tuning.",
        features: [
          "Moderate training (~2 hours)",
          "Multiple algorithms",
          "Hyperparameter optimization",
          "Ensemble methods",
        ],
        algorithms: [
          "isolation_forest",
          "local_outlier_factor",
          "one_class_svm",
          "autoencoder",
        ],
        estimatedTime: "2 hours",
      },
      {
        id: "comprehensive",
        name: "Comprehensive Search",
        complexity: "advanced",
        description:
          "Exhaustive search across all algorithms for maximum performance.",
        features: [
          "Extensive training (~6 hours)",
          "All available algorithms",
          "Advanced optimization",
          "Neural networks included",
        ],
        algorithms: [
          "isolation_forest",
          "local_outlier_factor",
          "one_class_svm",
          "autoencoder",
          "deep_svdd",
          "feature_bagging",
          "copod",
        ],
        estimatedTime: "6 hours",
      },
      {
        id: "neural",
        name: "Neural Network Focus",
        complexity: "advanced",
        description:
          "Focus on deep learning approaches for complex pattern detection.",
        features: [
          "GPU acceleration",
          "Deep learning algorithms",
          "Advanced feature learning",
          "Complex pattern detection",
        ],
        algorithms: ["autoencoder", "deep_svdd"],
        estimatedTime: "4 hours",
      },
    ];
  }

  generateMockDataPreview() {
    return {
      rows: 10000 + Math.floor(Math.random() * 50000),
      columns: 8 + Math.floor(Math.random() * 15),
      missing: Math.floor(Math.random() * 15),
      headers: [
        "timestamp",
        "sensor_1",
        "sensor_2",
        "temperature",
        "pressure",
        "flow_rate",
      ],
      rows_data: [
        ["2024-01-01 10:00:00", "0.823", "1.234", "25.4", "101.3", "15.2"],
        ["2024-01-01 10:01:00", "0.801", "1.189", "25.7", "101.2", "15.8"],
        ["2024-01-01 10:02:00", "0.856", "1.267", "25.1", "101.4", "14.9"],
        ["2024-01-01 10:03:00", "0.798", "1.145", "25.9", "101.1", "16.1"],
      ],
    };
  }

  applyTemplate(template) {
    this.config.algorithms = [...template.algorithms];
    // Apply other template-specific configurations
  }

  exportConfiguration() {
    const config = this.buildFinalConfig();
    const blob = new Blob([JSON.stringify(config, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `automl-config-${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  saveAsTemplate() {
    // Implementation for saving custom template
    const templateName = prompt("Enter template name:");
    if (templateName) {
      // Save to local storage or send to server
      console.log("Saving template:", templateName, this.buildFinalConfig());
    }
  }

  // Event system
  on(event, listener) {
    if (!this.eventListeners.has(event)) {
      this.eventListeners.set(event, new Set());
    }
    this.eventListeners.get(event).add(listener);
    return () => this.off(event, listener);
  }

  off(event, listener) {
    if (this.eventListeners.has(event)) {
      this.eventListeners.get(event).delete(listener);
    }
  }

  emit(event, data) {
    if (this.eventListeners.has(event)) {
      this.eventListeners.get(event).forEach((listener) => {
        try {
          listener(data);
        } catch (error) {
          console.error("AutoML wizard event error:", error);
        }
      });
    }
  }
}

// Export class
if (typeof module !== "undefined" && module.exports) {
  module.exports = {
    AutoMLConfigWizard,
  };
} else {
  // Browser environment
  window.AutoMLConfigWizard = AutoMLConfigWizard;
}
\n\n// chart-components.js\n// Chart Components - D3.js and ECharts integration for anomaly visualization
export class ChartComponent {
  constructor(container, config = {}) {
    this.container = container;
    this.config = {
      type: "scatter",
      width: 800,
      height: 400,
      margin: { top: 20, right: 20, bottom: 40, left: 40 },
      ...config,
    };

    this.chart = null;
    this.data = config.data || [];

    this.init();
  }

  init() {
    this.setupContainer();
    this.createChart();
  }

  setupContainer() {
    this.container.innerHTML = "";
    this.container.style.width = "100%";
    this.container.style.height = `${this.config.height}px`;
  }

  createChart() {
    switch (this.config.type) {
      case "scatter":
        this.createScatterPlot();
        break;
      case "timeline":
        this.createTimeline();
        break;
      case "distribution":
        this.createDistribution();
        break;
      default:
        console.warn(`Unknown chart type: ${this.config.type}`);
    }
  }

  createScatterPlot() {
    // Placeholder for D3.js scatter plot
    this.container.innerHTML = `
      <div class="chart-placeholder">
        <div class="placeholder-icon">📊</div>
        <div class="placeholder-text">Scatter Plot Visualization</div>
        <div class="placeholder-description">D3.js scatter plot will be rendered here</div>
      </div>
    `;
  }

  createTimeline() {
    // Placeholder for timeline chart
    this.container.innerHTML = `
      <div class="chart-placeholder">
        <div class="placeholder-icon">📈</div>
        <div class="placeholder-text">Timeline Visualization</div>
        <div class="placeholder-description">Time series chart will be rendered here</div>
      </div>
    `;
  }

  createDistribution() {
    // Placeholder for distribution chart
    this.container.innerHTML = `
      <div class="chart-placeholder">
        <div class="placeholder-icon">📊</div>
        <div class="placeholder-text">Distribution Visualization</div>
        <div class="placeholder-description">Distribution chart will be rendered here</div>
      </div>
    `;
  }

  updateData(newData) {
    this.data = newData;
    this.createChart();
  }

  destroy() {
    if (this.chart && this.chart.dispose) {
      this.chart.dispose();
    }
    this.container.innerHTML = "";
  }
}
\n\n// d3-chart-library.js\n/**
 * Advanced D3.js Chart Library for Pynomaly
 * Provides interactive, real-time data visualization components
 * with accessibility, responsive design, and performance optimization
 */

class D3ChartLibrary {
  constructor() {
    this.charts = new Map();
    this.themes = {
      light: {
        background: "#ffffff",
        text: "#1e293b",
        grid: "#e2e8f0",
        primary: "#0ea5e9",
        secondary: "#22c55e",
        danger: "#dc2626",
        warning: "#d97706",
      },
      dark: {
        background: "#0f172a",
        text: "#f1f5f9",
        grid: "#334155",
        primary: "#38bdf8",
        secondary: "#4ade80",
        danger: "#f87171",
        warning: "#fbbf24",
      },
    };
    this.currentTheme = "light";
    this.setupEventListeners();
  }

  setupEventListeners() {
    // Theme change listener
    document.addEventListener("theme-changed", (event) => {
      this.currentTheme = event.detail.theme;
      this.updateAllChartsTheme();
    });

    // Resize listener for responsive charts
    window.addEventListener(
      "resize",
      this.debounce(() => {
        this.resizeAllCharts();
      }, 250),
    );
  }

  // Utility function for debouncing
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

  // Get current theme colors
  getTheme() {
    return this.themes[this.currentTheme];
  }

  // Register a chart instance
  registerChart(id, chart) {
    this.charts.set(id, chart);
  }

  // Remove a chart instance
  removeChart(id) {
    const chart = this.charts.get(id);
    if (chart && chart.destroy) {
      chart.destroy();
    }
    this.charts.delete(id);
  }

  // Update all charts theme
  updateAllChartsTheme() {
    this.charts.forEach((chart) => {
      if (chart.updateTheme) {
        chart.updateTheme(this.getTheme());
      }
    });
  }

  // Resize all charts
  resizeAllCharts() {
    this.charts.forEach((chart) => {
      if (chart.resize) {
        chart.resize();
      }
    });
  }
}

/**
 * Base Chart Class
 * Provides common functionality for all chart types
 */
class BaseChart {
  constructor(container, options = {}) {
    this.container = d3.select(container);
    this.containerElement = container;
    this.options = {
      width: 800,
      height: 400,
      margin: { top: 20, right: 30, bottom: 40, left: 50 },
      animated: true,
      responsive: true,
      accessibility: true,
      ...options,
    };

    this.data = [];
    this.svg = null;
    this.tooltip = null;
    this.theme = chartLibrary.getTheme();

    this.setupChart();
    this.setupAccessibility();

    // Register with chart library
    const chartId = this.containerElement.id || `chart-${Date.now()}`;
    chartLibrary.registerChart(chartId, this);
  }

  setupChart() {
    // Clear existing content
    this.container.selectAll("*").remove();

    // Calculate dimensions
    this.updateDimensions();

    // Create main SVG
    this.svg = this.container
      .append("svg")
      .attr("width", this.width)
      .attr("height", this.height)
      .attr("role", "img")
      .attr("aria-labelledby", `${this.containerElement.id}-title`)
      .attr("aria-describedby", `${this.containerElement.id}-desc`);

    // Create main group for chart content
    this.chartGroup = this.svg
      .append("g")
      .attr(
        "transform",
        `translate(${this.options.margin.left},${this.options.margin.top})`,
      );

    // Setup tooltip
    this.setupTooltip();
  }

  setupAccessibility() {
    if (!this.options.accessibility) return;

    // Add title and description elements
    const titleId = `${this.containerElement.id}-title`;
    const descId = `${this.containerElement.id}-desc`;

    if (!document.getElementById(titleId)) {
      const titleElement = document.createElement("h3");
      titleElement.id = titleId;
      titleElement.className = "sr-only";
      titleElement.textContent = this.options.title || "Data Visualization";
      this.containerElement.insertBefore(
        titleElement,
        this.containerElement.firstChild,
      );
    }

    if (!document.getElementById(descId)) {
      const descElement = document.createElement("div");
      descElement.id = descId;
      descElement.className = "sr-only";
      descElement.textContent =
        this.options.description ||
        "Interactive chart showing data trends and patterns";
      this.containerElement.appendChild(descElement);
    }
  }

  setupTooltip() {
    this.tooltip = d3
      .select("body")
      .append("div")
      .attr("class", "chart-tooltip")
      .style("opacity", 0)
      .style("position", "absolute")
      .style("background", "rgba(0, 0, 0, 0.8)")
      .style("color", "white")
      .style("padding", "8px 12px")
      .style("border-radius", "4px")
      .style("font-size", "12px")
      .style("pointer-events", "none")
      .style("z-index", "1000");
  }

  updateDimensions() {
    if (this.options.responsive) {
      const containerWidth =
        this.containerElement.clientWidth || this.options.width;
      this.width = Math.max(300, containerWidth);
      this.height = Math.max(200, this.width * 0.5); // Maintain aspect ratio
    } else {
      this.width = this.options.width;
      this.height = this.options.height;
    }

    this.innerWidth =
      this.width - this.options.margin.left - this.options.margin.right;
    this.innerHeight =
      this.height - this.options.margin.top - this.options.margin.bottom;
  }

  showTooltip(content, event) {
    this.tooltip
      .html(content)
      .style("opacity", 1)
      .style("left", event.pageX + 10 + "px")
      .style("top", event.pageY - 10 + "px");
  }

  hideTooltip() {
    this.tooltip.style("opacity", 0);
  }

  updateTheme(newTheme) {
    this.theme = newTheme;
    this.render();
  }

  resize() {
    this.updateDimensions();
    this.svg.attr("width", this.width).attr("height", this.height);
    this.render();
  }

  setData(data) {
    this.data = data;
    this.render();
  }

  render() {
    // Override in subclasses
    throw new Error("render() method must be implemented by subclass");
  }

  destroy() {
    if (this.tooltip) {
      this.tooltip.remove();
    }
    this.container.selectAll("*").remove();
  }
}

/**
 * Time Series Chart
 * Interactive line chart for time-based anomaly detection data
 */
class TimeSeriesChart extends BaseChart {
  constructor(container, options = {}) {
    super(container, {
      xAccessor: (d) => d.timestamp,
      yAccessor: (d) => d.value,
      anomalyAccessor: (d) => d.isAnomaly,
      confidenceAccessor: (d) => d.confidence,
      showAnomalies: true,
      showConfidenceBands: false,
      interpolation: d3.curveMonotoneX,
      ...options,
    });

    this.xScale = d3.scaleTime();
    this.yScale = d3.scaleLinear();
    this.line = d3
      .line()
      .x((d) => this.xScale(this.options.xAccessor(d)))
      .y((d) => this.yScale(this.options.yAccessor(d)))
      .curve(this.options.interpolation);
  }

  render() {
    if (!this.data || this.data.length === 0) return;

    // Clear previous content
    this.chartGroup.selectAll("*").remove();

    // Update scales
    this.xScale
      .domain(d3.extent(this.data, this.options.xAccessor))
      .range([0, this.innerWidth]);

    this.yScale
      .domain(d3.extent(this.data, this.options.yAccessor))
      .nice()
      .range([this.innerHeight, 0]);

    // Add axes
    this.addAxes();

    // Add main line
    this.addMainLine();

    // Add anomaly markers
    if (this.options.showAnomalies) {
      this.addAnomalyMarkers();
    }

    // Add confidence bands
    if (this.options.showConfidenceBands) {
      this.addConfidenceBands();
    }

    // Add interaction
    this.addInteraction();
  }

  addAxes() {
    // X-axis
    const xAxis = d3.axisBottom(this.xScale).tickFormat(d3.timeFormat("%H:%M"));

    this.chartGroup
      .append("g")
      .attr("class", "x-axis")
      .attr("transform", `translate(0,${this.innerHeight})`)
      .call(xAxis)
      .selectAll("text")
      .style("fill", this.theme.text);

    // Y-axis
    const yAxis = d3.axisLeft(this.yScale);

    this.chartGroup
      .append("g")
      .attr("class", "y-axis")
      .call(yAxis)
      .selectAll("text")
      .style("fill", this.theme.text);

    // Style axes
    this.chartGroup
      .selectAll(".domain, .tick line")
      .style("stroke", this.theme.grid);
  }

  addMainLine() {
    const path = this.chartGroup
      .append("path")
      .datum(this.data)
      .attr("class", "main-line")
      .attr("fill", "none")
      .attr("stroke", this.theme.primary)
      .attr("stroke-width", 2)
      .attr("d", this.line);

    // Animate path drawing
    if (this.options.animated) {
      const totalLength = path.node().getTotalLength();
      path
        .attr("stroke-dasharray", totalLength + " " + totalLength)
        .attr("stroke-dashoffset", totalLength)
        .transition()
        .duration(1500)
        .ease(d3.easeLinear)
        .attr("stroke-dashoffset", 0);
    }
  }

  addAnomalyMarkers() {
    const anomalies = this.data.filter(this.options.anomalyAccessor);

    const markers = this.chartGroup
      .selectAll(".anomaly-marker")
      .data(anomalies)
      .enter()
      .append("circle")
      .attr("class", "anomaly-marker")
      .attr("cx", (d) => this.xScale(this.options.xAccessor(d)))
      .attr("cy", (d) => this.yScale(this.options.yAccessor(d)))
      .attr("r", 0)
      .attr("fill", this.theme.danger)
      .attr("stroke", this.theme.background)
      .attr("stroke-width", 2)
      .attr("opacity", 0.8)
      .style("cursor", "pointer");

    // Animate markers
    if (this.options.animated) {
      markers
        .transition()
        .delay((d, i) => i * 100)
        .duration(300)
        .attr("r", 5);
    } else {
      markers.attr("r", 5);
    }

    // Add tooltip interaction
    markers
      .on("mouseover", (event, d) => {
        const confidence = this.options.confidenceAccessor(d);
        const timestamp = d3.timeFormat("%Y-%m-%d %H:%M:%S")(
          this.options.xAccessor(d),
        );
        const value = this.options.yAccessor(d).toFixed(2);

        const content = `
          <strong>Anomaly Detected</strong><br/>
          Time: ${timestamp}<br/>
          Value: ${value}<br/>
          Confidence: ${(confidence * 100).toFixed(1)}%
        `;

        this.showTooltip(content, event);

        // Enlarge marker on hover
        d3.select(event.target).transition().duration(200).attr("r", 7);
      })
      .on("mouseout", (event) => {
        this.hideTooltip();

        // Reset marker size
        d3.select(event.target).transition().duration(200).attr("r", 5);
      })
      .on("click", (event, d) => {
        // Emit custom event for anomaly selection
        this.containerElement.dispatchEvent(
          new CustomEvent("anomaly-selected", {
            detail: { data: d, chart: this },
          }),
        );
      });

    // Add accessibility labels
    markers.attr("aria-label", (d) => {
      const timestamp = d3.timeFormat("%Y-%m-%d %H:%M:%S")(
        this.options.xAccessor(d),
      );
      const value = this.options.yAccessor(d).toFixed(2);
      const confidence = this.options.confidenceAccessor(d);
      return `Anomaly at ${timestamp}, value ${value}, confidence ${(confidence * 100).toFixed(1)}%`;
    });
  }

  addConfidenceBands() {
    // Calculate confidence intervals (simplified implementation)
    const confidenceData = this.data.map((d) => {
      const confidence = this.options.confidenceAccessor(d) || 0.5;
      const value = this.options.yAccessor(d);
      const margin = value * (1 - confidence) * 0.5;

      return {
        ...d,
        upper: value + margin,
        lower: value - margin,
      };
    });

    // Create area generator for confidence band
    const area = d3
      .area()
      .x((d) => this.xScale(this.options.xAccessor(d)))
      .y0((d) => this.yScale(d.lower))
      .y1((d) => this.yScale(d.upper))
      .curve(this.options.interpolation);

    // Add confidence band
    this.chartGroup
      .append("path")
      .datum(confidenceData)
      .attr("class", "confidence-band")
      .attr("fill", this.theme.primary)
      .attr("fill-opacity", 0.2)
      .attr("d", area);
  }

  addInteraction() {
    // Add invisible overlay for mouse tracking
    const overlay = this.chartGroup
      .append("rect")
      .attr("class", "overlay")
      .attr("width", this.innerWidth)
      .attr("height", this.innerHeight)
      .attr("fill", "none")
      .attr("pointer-events", "all")
      .style("cursor", "crosshair");

    // Add vertical line for cursor tracking
    const verticalLine = this.chartGroup
      .append("line")
      .attr("class", "cursor-line")
      .attr("stroke", this.theme.text)
      .attr("stroke-width", 1)
      .attr("stroke-dasharray", "3,3")
      .attr("opacity", 0);

    // Add mouse interaction
    overlay
      .on("mousemove", (event) => {
        const [mouseX] = d3.pointer(event);
        const xValue = this.xScale.invert(mouseX);

        // Find closest data point
        const bisector = d3.bisector(this.options.xAccessor).left;
        const index = bisector(this.data, xValue);
        const d0 = this.data[index - 1];
        const d1 = this.data[index];

        if (d0 && d1) {
          const d =
            xValue - this.options.xAccessor(d0) >
            this.options.xAccessor(d1) - xValue
              ? d1
              : d0;

          // Update vertical line
          verticalLine
            .attr("x1", this.xScale(this.options.xAccessor(d)))
            .attr("x2", this.xScale(this.options.xAccessor(d)))
            .attr("y1", 0)
            .attr("y2", this.innerHeight)
            .attr("opacity", 0.5);

          // Show tooltip with data point info
          const timestamp = d3.timeFormat("%Y-%m-%d %H:%M:%S")(
            this.options.xAccessor(d),
          );
          const value = this.options.yAccessor(d).toFixed(2);

          const content = `
            <strong>Data Point</strong><br/>
            Time: ${timestamp}<br/>
            Value: ${value}
          `;

          this.showTooltip(content, event);
        }
      })
      .on("mouseleave", () => {
        verticalLine.attr("opacity", 0);
        this.hideTooltip();
      });
  }

  // Real-time data update method
  updateData(newData, animate = true) {
    this.data = newData;

    if (animate) {
      // Animate the update
      this.render();
    } else {
      this.render();
    }

    // Announce update to screen readers
    if (this.options.accessibility) {
      const announcement = `Chart updated with ${newData.length} data points`;
      this.announceToScreenReader(announcement);
    }
  }

  // Add a single data point (for real-time streaming)
  addDataPoint(point, maxPoints = 1000) {
    this.data.push(point);

    // Keep only the last maxPoints
    if (this.data.length > maxPoints) {
      this.data = this.data.slice(-maxPoints);
    }

    this.render();
  }

  announceToScreenReader(message) {
    const announcer = document.getElementById("chart-announcer");
    if (announcer) {
      announcer.textContent = message;
    }
  }
}

/**
 * Scatter Plot Chart
 * Interactive scatter plot for anomaly detection in 2D space
 */
class ScatterPlotChart extends BaseChart {
  constructor(container, options = {}) {
    super(container, {
      xAccessor: (d) => d.x,
      yAccessor: (d) => d.y,
      colorAccessor: (d) => d.anomalyScore,
      sizeAccessor: (d) => d.confidence,
      anomalyAccessor: (d) => d.isAnomaly,
      showDensity: false,
      enableBrushing: true,
      enableZoom: true,
      ...options,
    });

    this.xScale = d3.scaleLinear();
    this.yScale = d3.scaleLinear();
    this.colorScale = d3.scaleSequential(d3.interpolateViridis);
    this.sizeScale = d3.scaleSqrt().range([3, 12]);

    this.brush = null;
    this.zoom = null;
  }

  render() {
    if (!this.data || this.data.length === 0) return;

    // Clear previous content
    this.chartGroup.selectAll("*").remove();

    // Update scales
    this.updateScales();

    // Add axes
    this.addAxes();

    // Add density background if enabled
    if (this.options.showDensity) {
      this.addDensityBackground();
    }

    // Add scatter points
    this.addScatterPoints();

    // Add brushing if enabled
    if (this.options.enableBrushing) {
      this.addBrushing();
    }

    // Add zoom if enabled
    if (this.options.enableZoom) {
      this.addZoom();
    }

    // Add legend
    this.addLegend();
  }

  updateScales() {
    this.xScale
      .domain(d3.extent(this.data, this.options.xAccessor))
      .nice()
      .range([0, this.innerWidth]);

    this.yScale
      .domain(d3.extent(this.data, this.options.yAccessor))
      .nice()
      .range([this.innerHeight, 0]);

    this.colorScale.domain(d3.extent(this.data, this.options.colorAccessor));

    this.sizeScale.domain(d3.extent(this.data, this.options.sizeAccessor));
  }

  addAxes() {
    // X-axis
    this.chartGroup
      .append("g")
      .attr("class", "x-axis")
      .attr("transform", `translate(0,${this.innerHeight})`)
      .call(d3.axisBottom(this.xScale))
      .selectAll("text")
      .style("fill", this.theme.text);

    // Y-axis
    this.chartGroup
      .append("g")
      .attr("class", "y-axis")
      .call(d3.axisLeft(this.yScale))
      .selectAll("text")
      .style("fill", this.theme.text);

    // Axis labels
    this.chartGroup
      .append("text")
      .attr("class", "x-label")
      .attr("text-anchor", "middle")
      .attr("x", this.innerWidth / 2)
      .attr("y", this.innerHeight + 35)
      .style("fill", this.theme.text)
      .text(this.options.xLabel || "X Value");

    this.chartGroup
      .append("text")
      .attr("class", "y-label")
      .attr("text-anchor", "middle")
      .attr("transform", "rotate(-90)")
      .attr("x", -this.innerHeight / 2)
      .attr("y", -35)
      .style("fill", this.theme.text)
      .text(this.options.yLabel || "Y Value");

    // Style axes
    this.chartGroup
      .selectAll(".domain, .tick line")
      .style("stroke", this.theme.grid);
  }

  addScatterPoints() {
    const points = this.chartGroup
      .selectAll(".data-point")
      .data(this.data)
      .enter()
      .append("circle")
      .attr("class", "data-point")
      .attr("cx", (d) => this.xScale(this.options.xAccessor(d)))
      .attr("cy", (d) => this.yScale(this.options.yAccessor(d)))
      .attr("r", 0)
      .attr("fill", (d) => {
        if (this.options.anomalyAccessor(d)) {
          return this.theme.danger;
        }
        return this.colorScale(this.options.colorAccessor(d));
      })
      .attr("stroke", (d) =>
        this.options.anomalyAccessor(d) ? this.theme.background : "none",
      )
      .attr("stroke-width", (d) => (this.options.anomalyAccessor(d) ? 2 : 0))
      .attr("opacity", 0.7)
      .style("cursor", "pointer");

    // Animate points
    if (this.options.animated) {
      points
        .transition()
        .delay((d, i) => i * 10)
        .duration(500)
        .attr("r", (d) => this.sizeScale(this.options.sizeAccessor(d)));
    } else {
      points.attr("r", (d) => this.sizeScale(this.options.sizeAccessor(d)));
    }

    // Add interaction
    points
      .on("mouseover", (event, d) => {
        const content = `
          <strong>${this.options.anomalyAccessor(d) ? "Anomaly" : "Normal Point"}</strong><br/>
          X: ${this.options.xAccessor(d).toFixed(2)}<br/>
          Y: ${this.options.yAccessor(d).toFixed(2)}<br/>
          Score: ${this.options.colorAccessor(d).toFixed(3)}<br/>
          Confidence: ${(this.options.sizeAccessor(d) * 100).toFixed(1)}%
        `;

        this.showTooltip(content, event);

        // Highlight point
        d3.select(event.target)
          .transition()
          .duration(200)
          .attr("r", (d) => this.sizeScale(this.options.sizeAccessor(d)) * 1.5)
          .attr("opacity", 1);
      })
      .on("mouseout", (event, d) => {
        this.hideTooltip();

        // Reset point
        d3.select(event.target)
          .transition()
          .duration(200)
          .attr("r", (d) => this.sizeScale(this.options.sizeAccessor(d)))
          .attr("opacity", 0.7);
      })
      .on("click", (event, d) => {
        // Emit selection event
        this.containerElement.dispatchEvent(
          new CustomEvent("point-selected", {
            detail: { data: d, chart: this },
          }),
        );
      });
  }

  addLegend() {
    const legendGroup = this.svg
      .append("g")
      .attr("class", "legend")
      .attr("transform", `translate(${this.width - 120}, 20)`);

    // Color legend
    const colorLegend = legendGroup.append("g").attr("class", "color-legend");

    const colorGradient = this.svg
      .append("defs")
      .append("linearGradient")
      .attr("id", "color-gradient")
      .attr("gradientUnits", "userSpaceOnUse")
      .attr("x1", 0)
      .attr("y1", 0)
      .attr("x2", 0)
      .attr("y2", 100);

    colorGradient
      .selectAll("stop")
      .data(d3.range(0, 1.1, 0.1))
      .enter()
      .append("stop")
      .attr("offset", (d) => `${d * 100}%`)
      .attr("stop-color", (d) =>
        this.colorScale(
          d3.quantile(this.data.map(this.options.colorAccessor), d),
        ),
      );

    colorLegend
      .append("rect")
      .attr("width", 20)
      .attr("height", 100)
      .style("fill", "url(#color-gradient)");

    colorLegend
      .append("text")
      .attr("x", 25)
      .attr("y", 10)
      .style("fill", this.theme.text)
      .style("font-size", "12px")
      .text("Anomaly Score");

    // Size legend
    const sizeLegend = legendGroup
      .append("g")
      .attr("class", "size-legend")
      .attr("transform", "translate(0, 120)");

    const sizeValues = [0.2, 0.5, 0.8];
    sizeLegend
      .selectAll(".size-legend-item")
      .data(sizeValues)
      .enter()
      .append("g")
      .attr("class", "size-legend-item")
      .attr("transform", (d, i) => `translate(0, ${i * 25})`)
      .each(function (d) {
        const group = d3.select(this);

        group
          .append("circle")
          .attr("cx", 10)
          .attr("cy", 0)
          .attr("r", (d) => this.sizeScale(d))
          .attr("fill", this.theme.primary)
          .attr("opacity", 0.7);

        group
          .append("text")
          .attr("x", 25)
          .attr("y", 4)
          .style("fill", this.theme.text)
          .style("font-size", "12px")
          .text(`${(d * 100).toFixed(0)}%`);
      });

    sizeLegend
      .append("text")
      .attr("x", 0)
      .attr("y", -10)
      .style("fill", this.theme.text)
      .style("font-size", "12px")
      .text("Confidence");
  }

  addBrushing() {
    this.brush = d3
      .brush()
      .extent([
        [0, 0],
        [this.innerWidth, this.innerHeight],
      ])
      .on("end", (event) => {
        const selection = event.selection;
        if (!selection) {
          // Clear selection
          this.chartGroup
            .selectAll(".data-point")
            .classed("selected", false)
            .attr("opacity", 0.7);
          return;
        }

        const [[x0, y0], [x1, y1]] = selection;

        // Select points within brush
        const selectedData = [];
        this.chartGroup
          .selectAll(".data-point")
          .classed("selected", (d) => {
            const x = this.xScale(this.options.xAccessor(d));
            const y = this.yScale(this.options.yAccessor(d));
            const isSelected = x >= x0 && x <= x1 && y >= y0 && y <= y1;

            if (isSelected) {
              selectedData.push(d);
            }

            return isSelected;
          })
          .attr("opacity", (d) => {
            const x = this.xScale(this.options.xAccessor(d));
            const y = this.yScale(this.options.yAccessor(d));
            return x >= x0 && x <= x1 && y >= y0 && y <= y1 ? 1 : 0.3;
          });

        // Emit selection event
        this.containerElement.dispatchEvent(
          new CustomEvent("points-selected", {
            detail: { data: selectedData, chart: this },
          }),
        );
      });

    this.chartGroup.append("g").attr("class", "brush").call(this.brush);
  }

  addZoom() {
    this.zoom = d3
      .zoom()
      .scaleExtent([1, 10])
      .on("zoom", (event) => {
        const { transform } = event;

        // Create new scales based on zoom transform
        const newXScale = transform.rescaleX(this.xScale);
        const newYScale = transform.rescaleY(this.yScale);

        // Update axes
        this.chartGroup.select(".x-axis").call(d3.axisBottom(newXScale));

        this.chartGroup.select(".y-axis").call(d3.axisLeft(newYScale));

        // Update points
        this.chartGroup
          .selectAll(".data-point")
          .attr("cx", (d) => newXScale(this.options.xAccessor(d)))
          .attr("cy", (d) => newYScale(this.options.yAccessor(d)));
      });

    this.svg.call(this.zoom);
  }
}

/**
 * Heatmap Chart
 * Interactive heatmap for correlation and anomaly density visualization
 */
class HeatmapChart extends BaseChart {
  constructor(container, options = {}) {
    super(container, {
      xAccessor: (d) => d.x,
      yAccessor: (d) => d.y,
      valueAccessor: (d) => d.value,
      gridSize: 20,
      colorScheme: d3.interpolateViridis,
      showLabels: true,
      enableZoom: false,
      ...options,
    });

    this.xScale = d3.scaleBand();
    this.yScale = d3.scaleBand();
    this.colorScale = d3.scaleSequential();
  }

  render() {
    if (!this.data || this.data.length === 0) return;

    // Clear previous content
    this.chartGroup.selectAll("*").remove();

    // Process data into grid format
    this.processData();

    // Update scales
    this.updateScales();

    // Add axes
    this.addAxes();

    // Add heatmap cells
    this.addHeatmapCells();

    // Add color legend
    this.addColorLegend();
  }

  processData() {
    // Create grid from data
    const xValues = [...new Set(this.data.map(this.options.xAccessor))].sort();
    const yValues = [...new Set(this.data.map(this.options.yAccessor))].sort();

    this.gridData = [];

    for (let y of yValues) {
      for (let x of xValues) {
        const dataPoint = this.data.find(
          (d) =>
            this.options.xAccessor(d) === x && this.options.yAccessor(d) === y,
        );

        this.gridData.push({
          x: x,
          y: y,
          value: dataPoint ? this.options.valueAccessor(dataPoint) : 0,
          original: dataPoint,
        });
      }
    }

    this.xValues = xValues;
    this.yValues = yValues;
  }

  updateScales() {
    this.xScale.domain(this.xValues).range([0, this.innerWidth]).padding(0.1);

    this.yScale.domain(this.yValues).range([this.innerHeight, 0]).padding(0.1);

    this.colorScale
      .interpolator(this.options.colorScheme)
      .domain(d3.extent(this.gridData, (d) => d.value));
  }

  addAxes() {
    // X-axis
    this.chartGroup
      .append("g")
      .attr("class", "x-axis")
      .attr("transform", `translate(0,${this.innerHeight})`)
      .call(d3.axisBottom(this.xScale))
      .selectAll("text")
      .style("fill", this.theme.text)
      .style("text-anchor", "end")
      .attr("dx", "-.8em")
      .attr("dy", ".15em")
      .attr("transform", "rotate(-65)");

    // Y-axis
    this.chartGroup
      .append("g")
      .attr("class", "y-axis")
      .call(d3.axisLeft(this.yScale))
      .selectAll("text")
      .style("fill", this.theme.text);

    // Style axes
    this.chartGroup
      .selectAll(".domain, .tick line")
      .style("stroke", this.theme.grid);
  }

  addHeatmapCells() {
    const cells = this.chartGroup
      .selectAll(".heatmap-cell")
      .data(this.gridData)
      .enter()
      .append("rect")
      .attr("class", "heatmap-cell")
      .attr("x", (d) => this.xScale(d.x))
      .attr("y", (d) => this.yScale(d.y))
      .attr("width", this.xScale.bandwidth())
      .attr("height", this.yScale.bandwidth())
      .attr("fill", (d) => this.colorScale(d.value))
      .attr("stroke", this.theme.background)
      .attr("stroke-width", 1)
      .attr("opacity", 0)
      .style("cursor", "pointer");

    // Animate cells
    if (this.options.animated) {
      cells
        .transition()
        .delay((d, i) => i * 20)
        .duration(500)
        .attr("opacity", 1);
    } else {
      cells.attr("opacity", 1);
    }

    // Add interaction
    cells
      .on("mouseover", (event, d) => {
        const content = `
          <strong>Cell (${d.x}, ${d.y})</strong><br/>
          Value: ${d.value.toFixed(3)}
        `;

        this.showTooltip(content, event);

        // Highlight cell
        d3.select(event.target)
          .attr("stroke-width", 3)
          .attr("stroke", this.theme.text);
      })
      .on("mouseout", (event) => {
        this.hideTooltip();

        // Reset cell
        d3.select(event.target)
          .attr("stroke-width", 1)
          .attr("stroke", this.theme.background);
      })
      .on("click", (event, d) => {
        // Emit selection event
        this.containerElement.dispatchEvent(
          new CustomEvent("cell-selected", {
            detail: { data: d, chart: this },
          }),
        );
      });

    // Add labels if enabled
    if (this.options.showLabels) {
      this.chartGroup
        .selectAll(".cell-label")
        .data(this.gridData.filter((d) => d.value > 0))
        .enter()
        .append("text")
        .attr("class", "cell-label")
        .attr("x", (d) => this.xScale(d.x) + this.xScale.bandwidth() / 2)
        .attr("y", (d) => this.yScale(d.y) + this.yScale.bandwidth() / 2)
        .attr("text-anchor", "middle")
        .attr("dy", ".35em")
        .style("fill", (d) =>
          d.value > this.colorScale.domain()[1] * 0.6 ? "white" : "black",
        )
        .style("font-size", "10px")
        .style("pointer-events", "none")
        .text((d) => d.value.toFixed(2));
    }
  }

  addColorLegend() {
    const legendGroup = this.svg
      .append("g")
      .attr("class", "color-legend")
      .attr("transform", `translate(${this.width - 100}, 20)`);

    // Create gradient
    const gradient = this.svg
      .append("defs")
      .append("linearGradient")
      .attr("id", "heatmap-gradient")
      .attr("gradientUnits", "userSpaceOnUse")
      .attr("x1", 0)
      .attr("y1", 100)
      .attr("x2", 0)
      .attr("y2", 0);

    gradient
      .selectAll("stop")
      .data(d3.range(0, 1.1, 0.1))
      .enter()
      .append("stop")
      .attr("offset", (d) => `${d * 100}%`)
      .attr("stop-color", (d) =>
        this.colorScale(d3.quantile(this.colorScale.domain(), d)),
      );

    // Legend rectangle
    legendGroup
      .append("rect")
      .attr("width", 20)
      .attr("height", 100)
      .style("fill", "url(#heatmap-gradient)")
      .style("stroke", this.theme.grid);

    // Legend scale
    const legendScale = d3
      .scaleLinear()
      .domain(this.colorScale.domain())
      .range([100, 0]);

    const legendAxis = d3
      .axisRight(legendScale)
      .tickSize(6)
      .tickFormat(d3.format(".2f"));

    legendGroup
      .append("g")
      .attr("class", "legend-axis")
      .attr("transform", "translate(20, 0)")
      .call(legendAxis)
      .selectAll("text")
      .style("fill", this.theme.text)
      .style("font-size", "10px");

    // Legend title
    legendGroup
      .append("text")
      .attr("x", 0)
      .attr("y", -5)
      .style("fill", this.theme.text)
      .style("font-size", "12px")
      .text("Value");
  }
}

// Initialize the chart library
const chartLibrary = new D3ChartLibrary();

// Export classes for use in other modules
if (typeof module !== "undefined" && module.exports) {
  module.exports = {
    D3ChartLibrary,
    BaseChart,
    TimeSeriesChart,
    ScatterPlotChart,
    HeatmapChart,
  };
} else {
  // Browser environment
  window.D3ChartLibrary = D3ChartLibrary;
  window.BaseChart = BaseChart;
  window.TimeSeriesChart = TimeSeriesChart;
  window.ScatterPlotChart = ScatterPlotChart;
  window.HeatmapChart = HeatmapChart;
  window.chartLibrary = chartLibrary;
}
\n\n// d3-charts-demo.js\n/**
 * D3.js Chart Library Demo and Usage Examples
 * Interactive demonstrations of all chart types with real-time data
 */

class D3ChartsDemo {
  constructor() {
    this.charts = new Map();
    this.demoData = this.generateDemoData();
    this.realTimeInterval = null;
    this.isRealTimeEnabled = false;

    this.init();
  }

  init() {
    this.setupDemoControls();
    this.createAllDemos();
    this.setupEventListeners();
  }

  generateDemoData() {
    const now = new Date();
    const timeSeriesData = [];
    const scatterData = [];
    const heatmapData = [];

    // Generate time series data with anomalies
    for (let i = 0; i < 100; i++) {
      const timestamp = new Date(now.getTime() - (100 - i) * 60000);
      const baseValue = 50 + 20 * Math.sin(i * 0.1) + Math.random() * 10;

      // Inject some anomalies
      const isAnomaly = Math.random() < 0.05;
      const value = isAnomaly
        ? baseValue + (Math.random() - 0.5) * 80
        : baseValue;
      const confidence = isAnomaly
        ? 0.7 + Math.random() * 0.3
        : 0.1 + Math.random() * 0.3;

      timeSeriesData.push({
        timestamp: timestamp,
        value: value,
        isAnomaly: isAnomaly,
        confidence: confidence,
      });
    }

    // Generate scatter plot data
    for (let i = 0; i < 200; i++) {
      const x = Math.random() * 100;
      const y = Math.random() * 100;

      // Create some clusters and anomalies
      const isCluster1 = Math.sqrt((x - 25) ** 2 + (y - 25) ** 2) < 15;
      const isCluster2 = Math.sqrt((x - 75) ** 2 + (y - 75) ** 2) < 15;
      const isAnomaly = !isCluster1 && !isCluster2 && Math.random() < 0.1;

      const anomalyScore = isAnomaly
        ? 0.7 + Math.random() * 0.3
        : Math.random() * 0.3;
      const confidence = isAnomaly
        ? 0.8 + Math.random() * 0.2
        : 0.3 + Math.random() * 0.4;

      scatterData.push({
        x: x,
        y: y,
        anomalyScore: anomalyScore,
        confidence: confidence,
        isAnomaly: isAnomaly,
      });
    }

    // Generate heatmap data
    const features = ["CPU", "Memory", "Disk", "Network", "Response Time"];
    const timeSlots = ["00:00", "04:00", "08:00", "12:00", "16:00", "20:00"];

    for (let feature of features) {
      for (let time of timeSlots) {
        heatmapData.push({
          x: time,
          y: feature,
          value: Math.random(),
        });
      }
    }

    return {
      timeSeries: timeSeriesData,
      scatter: scatterData,
      heatmap: heatmapData,
    };
  }

  setupDemoControls() {
    const controlsContainer = document.getElementById("demo-controls");
    if (!controlsContainer) return;

    controlsContainer.innerHTML = `
      <div class="demo-controls-grid">
        <div class="control-group">
          <h3>Real-Time Simulation</h3>
          <button id="realtime-toggle" class="btn btn-primary">Start Real-Time</button>
          <label>
            Update Interval:
            <select id="update-interval">
              <option value="1000">1 second</option>
              <option value="2000" selected>2 seconds</option>
              <option value="5000">5 seconds</option>
            </select>
          </label>
        </div>

        <div class="control-group">
          <h3>Theme</h3>
          <button id="theme-toggle" class="btn btn-secondary">Switch to Dark</button>
        </div>

        <div class="control-group">
          <h3>Chart Controls</h3>
          <button id="refresh-data" class="btn btn-secondary">Refresh Data</button>
          <button id="export-charts" class="btn btn-secondary">Export Charts</button>
        </div>

        <div class="control-group">
          <h3>Accessibility</h3>
          <button id="announce-data" class="btn btn-secondary">Announce Data</button>
          <label>
            <input type="checkbox" id="high-contrast" /> High Contrast
          </label>
        </div>
      </div>

      <div id="chart-announcer" aria-live="polite" class="sr-only"></div>
    `;
  }

  createAllDemos() {
    this.createTimeSeriesDemo();
    this.createScatterPlotDemo();
    this.createHeatmapDemo();
    this.createInteractiveDemo();
  }

  createTimeSeriesDemo() {
    const container = document.getElementById("timeseries-demo");
    if (!container) return;

    container.innerHTML = `
      <div class="demo-section">
        <h2>Time Series Chart - Anomaly Detection Timeline</h2>
        <p>Interactive time series visualization showing anomaly detection results over time with real-time updates.</p>

        <div class="chart-controls">
          <label>
            <input type="checkbox" id="show-confidence" checked /> Show Confidence Bands
          </label>
          <label>
            <input type="checkbox" id="show-anomalies" checked /> Show Anomaly Markers
          </label>
          <button id="zoom-reset-ts" class="btn btn-sm">Reset Zoom</button>
        </div>

        <div id="timeseries-chart" class="chart-container"></div>

        <div class="chart-info">
          <h4>Features:</h4>
          <ul>
            <li>Real-time data updates</li>
            <li>Interactive tooltips</li>
            <li>Anomaly highlighting</li>
            <li>Confidence bands</li>
            <li>Keyboard navigation</li>
            <li>Screen reader support</li>
          </ul>
        </div>
      </div>
    `;

    const chart = new TimeSeriesChart("#timeseries-chart", {
      title: "Anomaly Detection Timeline",
      description:
        "Time series chart showing detected anomalies with confidence intervals",
      showConfidenceBands: true,
      animated: true,
      responsive: true,
    });

    chart.setData(this.demoData.timeSeries);
    this.charts.set("timeseries", chart);

    // Setup controls
    document
      .getElementById("show-confidence")
      ?.addEventListener("change", (e) => {
        chart.options.showConfidenceBands = e.target.checked;
        chart.render();
      });

    document
      .getElementById("show-anomalies")
      ?.addEventListener("change", (e) => {
        chart.options.showAnomalies = e.target.checked;
        chart.render();
      });

    // Setup chart event listeners
    container.addEventListener("anomaly-selected", (e) => {
      const { data } = e.detail;
      this.showAnomalyDetails(data, "Time Series");
    });
  }

  createScatterPlotDemo() {
    const container = document.getElementById("scatter-demo");
    if (!container) return;

    container.innerHTML = `
      <div class="demo-section">
        <h2>Scatter Plot - 2D Anomaly Detection</h2>
        <p>Interactive scatter plot for detecting anomalies in two-dimensional data space with brushing and zoom.</p>

        <div class="chart-controls">
          <label>
            <input type="checkbox" id="enable-brushing" checked /> Enable Brushing
          </label>
          <label>
            <input type="checkbox" id="enable-zoom" checked /> Enable Zoom
          </label>
          <button id="clear-selection" class="btn btn-sm">Clear Selection</button>
        </div>

        <div id="scatter-chart" class="chart-container"></div>

        <div class="selection-info">
          <h4>Selection Info:</h4>
          <div id="selection-details">No points selected</div>
        </div>

        <div class="chart-info">
          <h4>Features:</h4>
          <ul>
            <li>Brush selection</li>
            <li>Zoom and pan</li>
            <li>Color-coded anomaly scores</li>
            <li>Size-coded confidence</li>
            <li>Interactive legends</li>
          </ul>
        </div>
      </div>
    `;

    const chart = new ScatterPlotChart("#scatter-chart", {
      title: "2D Anomaly Detection",
      description:
        "Scatter plot showing anomalies in two-dimensional feature space",
      xLabel: "Feature 1",
      yLabel: "Feature 2",
      enableBrushing: true,
      enableZoom: true,
      animated: true,
    });

    chart.setData(this.demoData.scatter);
    this.charts.set("scatter", chart);

    // Setup controls
    document
      .getElementById("enable-brushing")
      ?.addEventListener("change", (e) => {
        chart.options.enableBrushing = e.target.checked;
        chart.render();
      });

    document.getElementById("enable-zoom")?.addEventListener("change", (e) => {
      chart.options.enableZoom = e.target.checked;
      chart.render();
    });

    // Setup chart event listeners
    container.addEventListener("points-selected", (e) => {
      const { data } = e.detail;
      this.updateSelectionInfo(data);
    });

    container.addEventListener("point-selected", (e) => {
      const { data } = e.detail;
      this.showAnomalyDetails(data, "Scatter Plot");
    });
  }

  createHeatmapDemo() {
    const container = document.getElementById("heatmap-demo");
    if (!container) return;

    container.innerHTML = `
      <div class="demo-section">
        <h2>Heatmap - Feature Correlation Matrix</h2>
        <p>Interactive heatmap showing correlations between features and anomaly densities across time periods.</p>

        <div class="chart-controls">
          <label>
            <input type="checkbox" id="show-labels" checked /> Show Value Labels
          </label>
          <select id="color-scheme">
            <option value="interpolateViridis">Viridis</option>
            <option value="interpolatePlasma">Plasma</option>
            <option value="interpolateInferno">Inferno</option>
            <option value="interpolateTurbo">Turbo</option>
          </select>
        </div>

        <div id="heatmap-chart" class="chart-container"></div>

        <div class="chart-info">
          <h4>Features:</h4>
          <ul>
            <li>Interactive cells</li>
            <li>Color legends</li>
            <li>Value labels</li>
            <li>Multiple color schemes</li>
            <li>Responsive design</li>
          </ul>
        </div>
      </div>
    `;

    const chart = new HeatmapChart("#heatmap-chart", {
      title: "Feature Correlation Heatmap",
      description:
        "Heatmap showing correlation values between different system features",
      showLabels: true,
      animated: true,
    });

    chart.setData(this.demoData.heatmap);
    this.charts.set("heatmap", chart);

    // Setup controls
    document.getElementById("show-labels")?.addEventListener("change", (e) => {
      chart.options.showLabels = e.target.checked;
      chart.render();
    });

    document.getElementById("color-scheme")?.addEventListener("change", (e) => {
      chart.options.colorScheme = d3[e.target.value];
      chart.render();
    });

    // Setup chart event listeners
    container.addEventListener("cell-selected", (e) => {
      const { data } = e.detail;
      this.showCellDetails(data);
    });
  }

  createInteractiveDemo() {
    const container = document.getElementById("interactive-demo");
    if (!container) return;

    container.innerHTML = `
      <div class="demo-section">
        <h2>Interactive Dashboard Demo</h2>
        <p>Combined visualization dashboard showing real-time anomaly detection across multiple chart types.</p>

        <div class="dashboard-grid">
          <div class="dashboard-item">
            <h3>Real-Time Stream</h3>
            <div id="realtime-chart" class="mini-chart"></div>
          </div>

          <div class="dashboard-item">
            <h3>Anomaly Distribution</h3>
            <div id="distribution-chart" class="mini-chart"></div>
          </div>

          <div class="dashboard-item">
            <h3>Feature Correlation</h3>
            <div id="correlation-chart" class="mini-chart"></div>
          </div>

          <div class="dashboard-item stats-panel">
            <h3>Statistics</h3>
            <div id="stats-content">
              <div class="stat-item">
                <span class="stat-label">Total Anomalies:</span>
                <span class="stat-value" id="total-anomalies">-</span>
              </div>
              <div class="stat-item">
                <span class="stat-label">Avg Confidence:</span>
                <span class="stat-value" id="avg-confidence">-</span>
              </div>
              <div class="stat-item">
                <span class="stat-label">Last Updated:</span>
                <span class="stat-value" id="last-updated">-</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    `;

    // Create mini charts for dashboard
    const realtimeChart = new TimeSeriesChart("#realtime-chart", {
      title: "Real-time Anomaly Stream",
      height: 200,
      margin: { top: 10, right: 20, bottom: 30, left: 40 },
      showConfidenceBands: false,
      animated: false,
    });

    const distributionChart = new ScatterPlotChart("#distribution-chart", {
      title: "Anomaly Distribution",
      height: 200,
      margin: { top: 10, right: 20, bottom: 30, left: 40 },
      enableBrushing: false,
      enableZoom: false,
      animated: false,
    });

    const correlationChart = new HeatmapChart("#correlation-chart", {
      title: "Feature Correlation",
      height: 200,
      margin: { top: 10, right: 20, bottom: 30, left: 40 },
      showLabels: false,
      animated: false,
    });

    // Set initial data
    realtimeChart.setData(this.demoData.timeSeries.slice(-20));
    distributionChart.setData(this.demoData.scatter.slice(0, 50));
    correlationChart.setData(this.demoData.heatmap);

    this.charts.set("realtime", realtimeChart);
    this.charts.set("distribution", distributionChart);
    this.charts.set("correlation", correlationChart);

    this.updateStatistics();
  }

  setupEventListeners() {
    // Real-time toggle
    document
      .getElementById("realtime-toggle")
      ?.addEventListener("click", (e) => {
        this.toggleRealTime();
        e.target.textContent = this.isRealTimeEnabled
          ? "Stop Real-Time"
          : "Start Real-Time";
      });

    // Theme toggle
    document.getElementById("theme-toggle")?.addEventListener("click", (e) => {
      const newTheme = chartLibrary.currentTheme === "light" ? "dark" : "light";
      this.switchTheme(newTheme);
      e.target.textContent = `Switch to ${newTheme === "light" ? "Dark" : "Light"}`;
    });

    // Refresh data
    document.getElementById("refresh-data")?.addEventListener("click", () => {
      this.refreshAllData();
    });

    // Export charts
    document.getElementById("export-charts")?.addEventListener("click", () => {
      this.exportCharts();
    });

    // Announce data
    document.getElementById("announce-data")?.addEventListener("click", () => {
      this.announceDataSummary();
    });

    // High contrast
    document
      .getElementById("high-contrast")
      ?.addEventListener("change", (e) => {
        document.body.classList.toggle("high-contrast", e.target.checked);
      });
  }

  toggleRealTime() {
    if (this.isRealTimeEnabled) {
      clearInterval(this.realTimeInterval);
      this.isRealTimeEnabled = false;
    } else {
      const interval = parseInt(
        document.getElementById("update-interval")?.value || "2000",
      );
      this.realTimeInterval = setInterval(() => {
        this.updateRealTimeData();
      }, interval);
      this.isRealTimeEnabled = true;
    }
  }

  updateRealTimeData() {
    // Add new data point to time series
    const lastPoint =
      this.demoData.timeSeries[this.demoData.timeSeries.length - 1];
    const now = new Date();
    const baseValue =
      50 + 20 * Math.sin(Date.now() * 0.0001) + Math.random() * 10;
    const isAnomaly = Math.random() < 0.05;
    const value = isAnomaly
      ? baseValue + (Math.random() - 0.5) * 80
      : baseValue;

    const newPoint = {
      timestamp: now,
      value: value,
      isAnomaly: isAnomaly,
      confidence: isAnomaly
        ? 0.7 + Math.random() * 0.3
        : 0.1 + Math.random() * 0.3,
    };

    // Update main time series chart
    const timeSeriesChart = this.charts.get("timeseries");
    if (timeSeriesChart) {
      timeSeriesChart.addDataPoint(newPoint);
    }

    // Update real-time mini chart
    const realtimeChart = this.charts.get("realtime");
    if (realtimeChart) {
      realtimeChart.addDataPoint(newPoint, 20); // Keep only last 20 points
    }

    // Update statistics
    this.updateStatistics();

    // Announce significant anomalies
    if (isAnomaly && newPoint.confidence > 0.8) {
      this.announceToScreenReader(
        `High confidence anomaly detected: value ${value.toFixed(2)}, confidence ${(newPoint.confidence * 100).toFixed(1)}%`,
      );
    }
  }

  switchTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);

    // Emit theme change event
    document.dispatchEvent(
      new CustomEvent("theme-changed", {
        detail: { theme },
      }),
    );
  }

  refreshAllData() {
    this.demoData = this.generateDemoData();

    this.charts.forEach((chart, key) => {
      if (key === "timeseries" || key === "realtime") {
        chart.setData(this.demoData.timeSeries);
      } else if (key === "scatter" || key === "distribution") {
        chart.setData(this.demoData.scatter);
      } else if (key === "heatmap" || key === "correlation") {
        chart.setData(this.demoData.heatmap);
      }
    });

    this.updateStatistics();
    this.announceToScreenReader("All chart data refreshed");
  }

  exportCharts() {
    const exportData = {
      timestamp: new Date().toISOString(),
      charts: {},
    };

    this.charts.forEach((chart, key) => {
      if (chart.data) {
        exportData.charts[key] = {
          type: chart.constructor.name,
          data: chart.data,
          options: chart.options,
        };
      }
    });

    // Create downloadable JSON file
    const blob = new Blob([JSON.stringify(exportData, null, 2)], {
      type: "application/json",
    });

    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `pynomaly-charts-export-${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    this.announceToScreenReader("Chart data exported successfully");
  }

  updateStatistics() {
    const allAnomalies = this.demoData.timeSeries.filter((d) => d.isAnomaly);
    const totalAnomalies = allAnomalies.length;
    const avgConfidence =
      totalAnomalies > 0
        ? allAnomalies.reduce((sum, d) => sum + d.confidence, 0) /
          totalAnomalies
        : 0;

    document.getElementById("total-anomalies").textContent = totalAnomalies;
    document.getElementById("avg-confidence").textContent =
      (avgConfidence * 100).toFixed(1) + "%";
    document.getElementById("last-updated").textContent =
      new Date().toLocaleTimeString();
  }

  updateSelectionInfo(selectedData) {
    const detailsElement = document.getElementById("selection-details");
    if (!detailsElement) return;

    if (selectedData.length === 0) {
      detailsElement.textContent = "No points selected";
      return;
    }

    const anomalies = selectedData.filter((d) => d.isAnomaly).length;
    const avgScore =
      selectedData.reduce((sum, d) => sum + d.anomalyScore, 0) /
      selectedData.length;

    detailsElement.innerHTML = `
      <strong>${selectedData.length} points selected</strong><br/>
      Anomalies: ${anomalies}<br/>
      Average Score: ${avgScore.toFixed(3)}
    `;
  }

  showAnomalyDetails(data, chartType) {
    const modal = document.createElement("div");
    modal.className = "anomaly-modal";
    modal.innerHTML = `
      <div class="modal-content">
        <div class="modal-header">
          <h3>Anomaly Details - ${chartType}</h3>
          <button class="modal-close" aria-label="Close">&times;</button>
        </div>
        <div class="modal-body">
          <div class="detail-grid">
            ${data.timestamp ? `<div><strong>Timestamp:</strong> ${data.timestamp.toLocaleString()}</div>` : ""}
            ${data.value !== undefined ? `<div><strong>Value:</strong> ${data.value.toFixed(3)}</div>` : ""}
            ${data.x !== undefined ? `<div><strong>X:</strong> ${data.x.toFixed(3)}</div>` : ""}
            ${data.y !== undefined ? `<div><strong>Y:</strong> ${data.y.toFixed(3)}</div>` : ""}
            ${data.confidence !== undefined ? `<div><strong>Confidence:</strong> ${(data.confidence * 100).toFixed(1)}%</div>` : ""}
            ${data.anomalyScore !== undefined ? `<div><strong>Anomaly Score:</strong> ${data.anomalyScore.toFixed(3)}</div>` : ""}
          </div>
        </div>
        <div class="modal-footer">
          <button class="btn btn-primary modal-close">Close</button>
        </div>
      </div>
    `;

    document.body.appendChild(modal);

    // Setup close handlers
    modal.querySelectorAll(".modal-close").forEach((btn) => {
      btn.addEventListener("click", () => {
        document.body.removeChild(modal);
      });
    });

    // Close on backdrop click
    modal.addEventListener("click", (e) => {
      if (e.target === modal) {
        document.body.removeChild(modal);
      }
    });

    // Focus management
    const firstButton = modal.querySelector("button");
    firstButton?.focus();
  }

  showCellDetails(data) {
    this.announceToScreenReader(
      `Heatmap cell selected: ${data.x}, ${data.y}, value ${data.value.toFixed(3)}`,
    );
  }

  announceDataSummary() {
    const totalPoints = this.demoData.timeSeries.length;
    const anomalies = this.demoData.timeSeries.filter(
      (d) => d.isAnomaly,
    ).length;
    const anomalyRate = ((anomalies / totalPoints) * 100).toFixed(1);

    const message = `Data summary: ${totalPoints} total data points, ${anomalies} anomalies detected, ${anomalyRate}% anomaly rate`;
    this.announceToScreenReader(message);
  }

  announceToScreenReader(message) {
    const announcer = document.getElementById("chart-announcer");
    if (announcer) {
      announcer.textContent = message;
    }
  }

  destroy() {
    if (this.realTimeInterval) {
      clearInterval(this.realTimeInterval);
    }

    this.charts.forEach((chart) => {
      chart.destroy();
    });

    this.charts.clear();
  }
}

// Auto-initialize when DOM is ready
document.addEventListener("DOMContentLoaded", () => {
  if (document.querySelector(".d3-charts-demo")) {
    window.d3ChartsDemo = new D3ChartsDemo();
  }
});

// Export for module usage
if (typeof module !== "undefined" && module.exports) {
  module.exports = D3ChartsDemo;
} else {
  window.D3ChartsDemo = D3ChartsDemo;
}
\n\n// dashboard-layout.js\n/**
 * Advanced Dashboard Layout System
 *
 * Drag-and-drop dashboard configuration with responsive grid layouts
 * Features widget management, layout persistence, and real-time updates
 */

export class DashboardLayout {
  constructor(container, options = {}) {
    this.container =
      typeof container === "string"
        ? document.querySelector(container)
        : container;
    this.options = {
      columns: 12,
      rowHeight: 60,
      margin: [10, 10],
      containerPadding: [10, 10],
      maxRows: Infinity,
      isDraggable: true,
      isResizable: true,
      preventCollision: false,
      autoSize: true,
      compactType: "vertical",
      layouts: {},
      breakpoints: {
        lg: 1200,
        md: 996,
        sm: 768,
        xs: 480,
        xxs: 0,
      },
      responsiveLayouts: {
        lg: [],
        md: [],
        sm: [],
        xs: [],
        xxs: [],
      },
      ...options,
    };

    this.widgets = new Map();
    this.layout = [];
    this.currentBreakpoint = "lg";
    this.isDragging = false;
    this.isResizing = false;
    this.draggedWidget = null;
    this.placeholder = null;

    this.init();
  }

  init() {
    this.setupContainer();
    this.detectBreakpoint();
    this.bindEvents();
    this.render();
  }

  setupContainer() {
    this.container.classList.add("dashboard-layout");
    this.container.innerHTML = "";

    // Create grid overlay for visual guidance
    this.gridOverlay = document.createElement("div");
    this.gridOverlay.className = "grid-overlay";
    this.gridOverlay.style.display = "none";
    this.container.appendChild(this.gridOverlay);

    // Set container styles
    Object.assign(this.container.style, {
      position: "relative",
      minHeight: "100vh",
      padding: `${this.options.containerPadding[1]}px ${this.options.containerPadding[0]}px`,
    });
  }

  detectBreakpoint() {
    const width = this.container.clientWidth;

    for (const [breakpoint, minWidth] of Object.entries(
      this.options.breakpoints,
    )) {
      if (width >= minWidth) {
        this.currentBreakpoint = breakpoint;
        break;
      }
    }

    // Load layout for current breakpoint
    this.loadLayout(this.currentBreakpoint);
  }

  bindEvents() {
    // Resize observer for responsive behavior
    if (window.ResizeObserver) {
      this.resizeObserver = new ResizeObserver(() => {
        this.detectBreakpoint();
        this.render();
      });
      this.resizeObserver.observe(this.container);
    } else {
      window.addEventListener(
        "resize",
        this.debounce(() => {
          this.detectBreakpoint();
          this.render();
        }, 250),
      );
    }

    // Drag and drop events
    this.container.addEventListener("mousedown", this.onMouseDown.bind(this));
    this.container.addEventListener("mousemove", this.onMouseMove.bind(this));
    this.container.addEventListener("mouseup", this.onMouseUp.bind(this));

    // Touch events for mobile
    this.container.addEventListener(
      "touchstart",
      this.onTouchStart.bind(this),
      { passive: false },
    );
    this.container.addEventListener("touchmove", this.onTouchMove.bind(this), {
      passive: false,
    });
    this.container.addEventListener("touchend", this.onTouchEnd.bind(this));

    // Keyboard navigation
    this.container.addEventListener("keydown", this.onKeyDown.bind(this));
  }

  addWidget(widgetConfig) {
    const widget = {
      id: widgetConfig.id || this.generateId(),
      type: widgetConfig.type || "default",
      title: widgetConfig.title || "Widget",
      content: widgetConfig.content || "",
      component: widgetConfig.component || null,
      props: widgetConfig.props || {},
      x: widgetConfig.x || 0,
      y: widgetConfig.y || 0,
      w: widgetConfig.w || 2,
      h: widgetConfig.h || 2,
      minW: widgetConfig.minW || 1,
      minH: widgetConfig.minH || 1,
      maxW: widgetConfig.maxW || Infinity,
      maxH: widgetConfig.maxH || Infinity,
      static: widgetConfig.static || false,
      isDraggable: widgetConfig.isDraggable !== false,
      isResizable: widgetConfig.isResizable !== false,
      moved: false,
      resizeHandles: widgetConfig.resizeHandles || ["se"],
      ...widgetConfig,
    };

    // Find suitable position if not specified
    if (widgetConfig.x === undefined || widgetConfig.y === undefined) {
      const position = this.findSuitablePosition(widget.w, widget.h);
      widget.x = position.x;
      widget.y = position.y;
    }

    this.widgets.set(widget.id, widget);
    this.layout.push(widget);

    // Compact layout if needed
    if (this.options.compactType) {
      this.compactLayout();
    }

    this.render();
    this.saveLayout();

    // Emit event
    this.emitEvent("widgetAdded", { widget });

    return widget;
  }

  removeWidget(widgetId) {
    const widget = this.widgets.get(widgetId);
    if (!widget) return false;

    this.widgets.delete(widgetId);
    this.layout = this.layout.filter((w) => w.id !== widgetId);

    // Remove DOM element
    const element = this.container.querySelector(
      `[data-widget-id="${widgetId}"]`,
    );
    if (element) {
      element.remove();
    }

    // Compact layout
    if (this.options.compactType) {
      this.compactLayout();
    }

    this.render();
    this.saveLayout();

    // Emit event
    this.emitEvent("widgetRemoved", { widget });

    return true;
  }

  updateWidget(widgetId, updates) {
    const widget = this.widgets.get(widgetId);
    if (!widget) return false;

    Object.assign(widget, updates);

    // Update in layout array
    const layoutIndex = this.layout.findIndex((w) => w.id === widgetId);
    if (layoutIndex >= 0) {
      this.layout[layoutIndex] = widget;
    }

    this.render();
    this.saveLayout();

    // Emit event
    this.emitEvent("widgetUpdated", { widget, updates });

    return true;
  }

  findSuitablePosition(width, height) {
    const columns = this.options.columns;

    // Try to find a position without collisions
    for (let y = 0; y < this.options.maxRows; y++) {
      for (let x = 0; x <= columns - width; x++) {
        if (!this.hasCollision({ x, y, w: width, h: height })) {
          return { x, y };
        }
      }
    }

    // If no space found, place at bottom
    const maxY = Math.max(0, ...this.layout.map((w) => w.y + w.h));
    return { x: 0, y: maxY };
  }

  hasCollision(widget, excludeId = null) {
    return this.layout.some((w) => {
      if (w.id === excludeId) return false;

      return !(
        widget.x >= w.x + w.w ||
        widget.x + widget.w <= w.x ||
        widget.y >= w.y + w.h ||
        widget.y + widget.h <= w.y
      );
    });
  }

  compactLayout() {
    if (this.options.compactType === "vertical") {
      this.compactVertical();
    } else if (this.options.compactType === "horizontal") {
      this.compactHorizontal();
    }
  }

  compactVertical() {
    // Sort by y position, then x
    const sortedLayout = [...this.layout].sort((a, b) => {
      if (a.y === b.y) return a.x - b.x;
      return a.y - b.y;
    });

    sortedLayout.forEach((widget) => {
      if (widget.static) return;

      // Find the highest position this widget can move to
      let targetY = 0;
      for (let y = 0; y < widget.y; y++) {
        const testWidget = { ...widget, y };
        if (!this.hasCollision(testWidget, widget.id)) {
          targetY = y;
        } else {
          break;
        }
      }

      widget.y = targetY;
    });
  }

  compactHorizontal() {
    // Similar to vertical but moves widgets left
    const sortedLayout = [...this.layout].sort((a, b) => {
      if (a.x === b.x) return a.y - b.y;
      return a.x - b.x;
    });

    sortedLayout.forEach((widget) => {
      if (widget.static) return;

      let targetX = 0;
      for (let x = 0; x < widget.x; x++) {
        const testWidget = { ...widget, x };
        if (!this.hasCollision(testWidget, widget.id)) {
          targetX = x;
        } else {
          break;
        }
      }

      widget.x = targetX;
    });
  }

  render() {
    // Clear existing widgets (except overlay)
    const existingWidgets =
      this.container.querySelectorAll(".dashboard-widget");
    existingWidgets.forEach((w) => w.remove());

    // Calculate grid dimensions
    const containerWidth =
      this.container.clientWidth - this.options.containerPadding[0] * 2;
    const colWidth =
      (containerWidth - this.options.margin[0] * (this.options.columns - 1)) /
      this.options.columns;

    // Render each widget
    this.layout.forEach((widget) => {
      const element = this.createWidgetElement(widget, colWidth);
      this.container.appendChild(element);
    });

    // Update container height
    if (this.options.autoSize) {
      const maxY = Math.max(0, ...this.layout.map((w) => w.y + w.h));
      const containerHeight =
        maxY * (this.options.rowHeight + this.options.margin[1]) +
        this.options.containerPadding[1] * 2;
      this.container.style.minHeight = `${containerHeight}px`;
    }
  }

  createWidgetElement(widget, colWidth) {
    const element = document.createElement("div");
    element.className = "dashboard-widget";
    element.setAttribute("data-widget-id", widget.id);
    element.setAttribute("tabindex", "0");
    element.setAttribute("role", "article");
    element.setAttribute("aria-label", `Widget: ${widget.title}`);

    // Calculate position and size
    const x = widget.x * (colWidth + this.options.margin[0]);
    const y = widget.y * (this.options.rowHeight + this.options.margin[1]);
    const width = widget.w * colWidth + (widget.w - 1) * this.options.margin[0];
    const height =
      widget.h * this.options.rowHeight +
      (widget.h - 1) * this.options.margin[1];

    // Apply styles
    Object.assign(element.style, {
      position: "absolute",
      left: `${x}px`,
      top: `${y}px`,
      width: `${width}px`,
      height: `${height}px`,
      backgroundColor: "white",
      border: "1px solid #e2e8f0",
      borderRadius: "8px",
      boxShadow: "0 1px 3px rgba(0, 0, 0, 0.1)",
      overflow: "hidden",
      transition: this.isDragging || this.isResizing ? "none" : "all 0.2s ease",
      cursor: widget.isDraggable ? "move" : "default",
      zIndex: widget.static ? 1 : 2,
    });

    // Create widget content
    const header = document.createElement("div");
    header.className = "widget-header";
    header.style.cssText = `
            padding: 12px 16px;
            border-bottom: 1px solid #e2e8f0;
            background: #f8fafc;
            display: flex;
            justify-content: space-between;
            align-items: center;
            min-height: 44px;
        `;

    const title = document.createElement("h3");
    title.className = "widget-title";
    title.textContent = widget.title;
    title.style.cssText = `
            margin: 0;
            font-size: 14px;
            font-weight: 600;
            color: #1f2937;
        `;

    const actions = document.createElement("div");
    actions.className = "widget-actions";
    actions.style.cssText = `
            display: flex;
            gap: 8px;
        `;

    // Add action buttons
    if (!widget.static) {
      const removeBtn = this.createActionButton("×", () =>
        this.removeWidget(widget.id),
      );
      removeBtn.setAttribute("aria-label", "Remove widget");
      actions.appendChild(removeBtn);
    }

    header.appendChild(title);
    header.appendChild(actions);

    const content = document.createElement("div");
    content.className = "widget-content";
    content.style.cssText = `
            padding: 16px;
            height: calc(100% - 44px);
            overflow: auto;
        `;

    // Render widget content
    if (widget.component && typeof widget.component === "function") {
      const componentElement = widget.component(widget.props);
      content.appendChild(componentElement);
    } else if (widget.content) {
      content.innerHTML = widget.content;
    } else {
      content.innerHTML = `<p style="color: #6b7280;">No content available</p>`;
    }

    element.appendChild(header);
    element.appendChild(content);

    // Add resize handles if resizable
    if (widget.isResizable && !widget.static) {
      this.addResizeHandles(element, widget);
    }

    return element;
  }

  createActionButton(text, onClick) {
    const button = document.createElement("button");
    button.textContent = text;
    button.style.cssText = `
            background: transparent;
            border: none;
            color: #6b7280;
            cursor: pointer;
            font-size: 16px;
            width: 24px;
            height: 24px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 4px;
            transition: all 0.2s ease;
        `;

    button.addEventListener("mouseenter", () => {
      button.style.backgroundColor = "#e5e7eb";
      button.style.color = "#374151";
    });

    button.addEventListener("mouseleave", () => {
      button.style.backgroundColor = "transparent";
      button.style.color = "#6b7280";
    });

    button.addEventListener("click", (e) => {
      e.stopPropagation();
      onClick();
    });

    return button;
  }

  addResizeHandles(element, widget) {
    widget.resizeHandles.forEach((position) => {
      const handle = document.createElement("div");
      handle.className = `resize-handle resize-handle-${position}`;
      handle.style.cssText = this.getResizeHandleStyles(position);
      handle.setAttribute("data-resize-direction", position);

      handle.addEventListener("mousedown", (e) => {
        e.stopPropagation();
        this.startResize(e, widget, position);
      });

      element.appendChild(handle);
    });
  }

  getResizeHandleStyles(position) {
    const baseStyles = `
            position: absolute;
            background: #3b82f6;
            opacity: 0;
            transition: opacity 0.2s ease;
            cursor: ${this.getResizeCursor(position)};
        `;

    switch (position) {
      case "se":
        return (
          baseStyles +
          `
                    bottom: 0;
                    right: 0;
                    width: 12px;
                    height: 12px;
                    border-radius: 12px 0 0 0;
                `
        );
      case "s":
        return (
          baseStyles +
          `
                    bottom: 0;
                    left: 50%;
                    transform: translateX(-50%);
                    width: 24px;
                    height: 4px;
                `
        );
      case "e":
        return (
          baseStyles +
          `
                    right: 0;
                    top: 50%;
                    transform: translateY(-50%);
                    width: 4px;
                    height: 24px;
                `
        );
      default:
        return baseStyles;
    }
  }

  getResizeCursor(position) {
    const cursors = {
      se: "nw-resize",
      s: "ns-resize",
      e: "ew-resize",
      ne: "ne-resize",
      nw: "nw-resize",
      sw: "sw-resize",
      n: "ns-resize",
      w: "ew-resize",
    };
    return cursors[position] || "default";
  }

  // Event handlers
  onMouseDown(e) {
    const widgetElement = e.target.closest(".dashboard-widget");
    if (!widgetElement) return;

    const widgetId = widgetElement.getAttribute("data-widget-id");
    const widget = this.widgets.get(widgetId);

    if (!widget || widget.static || !widget.isDraggable) return;

    // Check if clicking on resize handle
    if (e.target.classList.contains("resize-handle")) return;

    this.startDrag(e, widget);
  }

  startDrag(e, widget) {
    this.isDragging = true;
    this.draggedWidget = widget;

    const widgetElement = this.container.querySelector(
      `[data-widget-id="${widget.id}"]`,
    );
    const rect = widgetElement.getBoundingClientRect();
    const containerRect = this.container.getBoundingClientRect();

    this.dragOffset = {
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
    };

    // Show grid overlay
    this.showGridOverlay();

    // Add dragging class
    widgetElement.classList.add("dragging");

    // Create placeholder
    this.createPlaceholder(widget);

    this.emitEvent("dragStart", { widget });
  }

  startResize(e, widget, direction) {
    this.isResizing = true;
    this.draggedWidget = widget;
    this.resizeDirection = direction;

    const widgetElement = this.container.querySelector(
      `[data-widget-id="${widget.id}"]`,
    );
    widgetElement.classList.add("resizing");

    this.emitEvent("resizeStart", { widget, direction });
  }

  onMouseMove(e) {
    if (this.isDragging) {
      this.handleDrag(e);
    } else if (this.isResizing) {
      this.handleResize(e);
    }
  }

  handleDrag(e) {
    if (!this.draggedWidget) return;

    const containerRect = this.container.getBoundingClientRect();
    const colWidth =
      (this.container.clientWidth -
        this.options.containerPadding[0] * 2 -
        this.options.margin[0] * (this.options.columns - 1)) /
      this.options.columns;

    // Calculate grid position
    const x = Math.round(
      (e.clientX - containerRect.left - this.dragOffset.x) /
        (colWidth + this.options.margin[0]),
    );
    const y = Math.round(
      (e.clientY - containerRect.top - this.dragOffset.y) /
        (this.options.rowHeight + this.options.margin[1]),
    );

    // Constrain to grid bounds
    const constrainedX = Math.max(
      0,
      Math.min(x, this.options.columns - this.draggedWidget.w),
    );
    const constrainedY = Math.max(0, y);

    // Update placeholder position
    if (this.placeholder) {
      this.placeholder.x = constrainedX;
      this.placeholder.y = constrainedY;
      this.updatePlaceholderPosition();
    }
  }

  handleResize(e) {
    if (!this.draggedWidget) return;

    const containerRect = this.container.getBoundingClientRect();
    const colWidth =
      (this.container.clientWidth -
        this.options.containerPadding[0] * 2 -
        this.options.margin[0] * (this.options.columns - 1)) /
      this.options.columns;

    const widget = this.draggedWidget;
    const direction = this.resizeDirection;

    // Calculate new size based on direction
    let newW = widget.w;
    let newH = widget.h;

    if (direction.includes("e")) {
      const newWidth =
        e.clientX -
        containerRect.left -
        widget.x * (colWidth + this.options.margin[0]);
      newW = Math.max(
        widget.minW,
        Math.min(
          widget.maxW,
          Math.round(newWidth / (colWidth + this.options.margin[0])),
        ),
      );
    }

    if (direction.includes("s")) {
      const newHeight =
        e.clientY -
        containerRect.top -
        widget.y * (this.options.rowHeight + this.options.margin[1]);
      newH = Math.max(
        widget.minH,
        Math.min(
          widget.maxH,
          Math.round(
            newHeight / (this.options.rowHeight + this.options.margin[1]),
          ),
        ),
      );
    }

    // Update widget size temporarily for visual feedback
    const widgetElement = this.container.querySelector(
      `[data-widget-id="${widget.id}"]`,
    );
    const width = newW * colWidth + (newW - 1) * this.options.margin[0];
    const height =
      newH * this.options.rowHeight + (newH - 1) * this.options.margin[1];

    widgetElement.style.width = `${width}px`;
    widgetElement.style.height = `${height}px`;
  }

  onMouseUp(e) {
    if (this.isDragging) {
      this.endDrag();
    } else if (this.isResizing) {
      this.endResize();
    }
  }

  endDrag() {
    if (!this.draggedWidget || !this.placeholder) return;

    const widget = this.draggedWidget;

    // Update widget position
    widget.x = this.placeholder.x;
    widget.y = this.placeholder.y;
    widget.moved = true;

    // Remove visual feedback
    const widgetElement = this.container.querySelector(
      `[data-widget-id="${widget.id}"]`,
    );
    widgetElement.classList.remove("dragging");

    this.removePlaceholder();
    this.hideGridOverlay();

    // Handle collisions if needed
    if (!this.options.preventCollision) {
      this.resolveCollisions(widget);
    }

    // Compact layout
    if (this.options.compactType) {
      this.compactLayout();
    }

    // Re-render and save
    this.render();
    this.saveLayout();

    this.emitEvent("dragEnd", { widget });

    this.isDragging = false;
    this.draggedWidget = null;
  }

  endResize() {
    if (!this.draggedWidget) return;

    const widget = this.draggedWidget;
    const widgetElement = this.container.querySelector(
      `[data-widget-id="${widget.id}"]`,
    );

    // Get final size from element
    const containerWidth =
      this.container.clientWidth - this.options.containerPadding[0] * 2;
    const colWidth =
      (containerWidth - this.options.margin[0] * (this.options.columns - 1)) /
      this.options.columns;

    const elementWidth = parseInt(widgetElement.style.width);
    const elementHeight = parseInt(widgetElement.style.height);

    widget.w = Math.round(elementWidth / (colWidth + this.options.margin[0]));
    widget.h = Math.round(
      elementHeight / (this.options.rowHeight + this.options.margin[1]),
    );

    widgetElement.classList.remove("resizing");

    // Handle collisions
    if (!this.options.preventCollision) {
      this.resolveCollisions(widget);
    }

    // Compact and re-render
    if (this.options.compactType) {
      this.compactLayout();
    }

    this.render();
    this.saveLayout();

    this.emitEvent("resizeEnd", { widget });

    this.isResizing = false;
    this.draggedWidget = null;
    this.resizeDirection = null;
  }

  resolveCollisions(movedWidget) {
    // Find colliding widgets and move them
    const collisions = this.layout.filter(
      (w) =>
        w.id !== movedWidget.id &&
        !w.static &&
        this.hasCollision(movedWidget, w.id),
    );

    collisions.forEach((widget) => {
      // Move widget down to resolve collision
      widget.y = movedWidget.y + movedWidget.h;
      widget.moved = true;
    });
  }

  createPlaceholder(widget) {
    this.placeholder = {
      x: widget.x,
      y: widget.y,
      w: widget.w,
      h: widget.h,
    };

    const element = document.createElement("div");
    element.className = "widget-placeholder";
    element.style.cssText = `
            position: absolute;
            background: rgba(59, 130, 246, 0.2);
            border: 2px dashed #3b82f6;
            border-radius: 8px;
            pointer-events: none;
            z-index: 1000;
        `;

    this.placeholderElement = element;
    this.container.appendChild(element);
    this.updatePlaceholderPosition();
  }

  updatePlaceholderPosition() {
    if (!this.placeholderElement || !this.placeholder) return;

    const containerWidth =
      this.container.clientWidth - this.options.containerPadding[0] * 2;
    const colWidth =
      (containerWidth - this.options.margin[0] * (this.options.columns - 1)) /
      this.options.columns;

    const x = this.placeholder.x * (colWidth + this.options.margin[0]);
    const y =
      this.placeholder.y * (this.options.rowHeight + this.options.margin[1]);
    const width =
      this.placeholder.w * colWidth +
      (this.placeholder.w - 1) * this.options.margin[0];
    const height =
      this.placeholder.h * this.options.rowHeight +
      (this.placeholder.h - 1) * this.options.margin[1];

    Object.assign(this.placeholderElement.style, {
      left: `${x}px`,
      top: `${y}px`,
      width: `${width}px`,
      height: `${height}px`,
    });
  }

  removePlaceholder() {
    if (this.placeholderElement) {
      this.placeholderElement.remove();
      this.placeholderElement = null;
    }
    this.placeholder = null;
  }

  showGridOverlay() {
    // Implementation for grid overlay
    this.gridOverlay.style.display = "block";
  }

  hideGridOverlay() {
    this.gridOverlay.style.display = "none";
  }

  // Touch event handlers
  onTouchStart(e) {
    if (e.touches.length === 1) {
      e.preventDefault();
      const touch = e.touches[0];
      this.onMouseDown({
        ...touch,
        target: touch.target,
        stopPropagation: () => e.stopPropagation(),
        preventDefault: () => e.preventDefault(),
      });
    }
  }

  onTouchMove(e) {
    if (e.touches.length === 1 && (this.isDragging || this.isResizing)) {
      e.preventDefault();
      const touch = e.touches[0];
      this.onMouseMove(touch);
    }
  }

  onTouchEnd(e) {
    this.onMouseUp(e);
  }

  // Keyboard navigation
  onKeyDown(e) {
    const widgetElement = e.target.closest(".dashboard-widget");
    if (!widgetElement) return;

    const widgetId = widgetElement.getAttribute("data-widget-id");
    const widget = this.widgets.get(widgetId);

    if (!widget || widget.static) return;

    let moved = false;
    const step = e.shiftKey ? 5 : 1;

    switch (e.key) {
      case "ArrowLeft":
        widget.x = Math.max(0, widget.x - step);
        moved = true;
        break;
      case "ArrowRight":
        widget.x = Math.min(this.options.columns - widget.w, widget.x + step);
        moved = true;
        break;
      case "ArrowUp":
        widget.y = Math.max(0, widget.y - step);
        moved = true;
        break;
      case "ArrowDown":
        widget.y = widget.y + step;
        moved = true;
        break;
      case "Delete":
      case "Backspace":
        this.removeWidget(widgetId);
        e.preventDefault();
        return;
    }

    if (moved) {
      e.preventDefault();

      // Check for collisions
      if (this.hasCollision(widget, widget.id)) {
        // Revert move
        switch (e.key) {
          case "ArrowLeft":
            widget.x += step;
            break;
          case "ArrowRight":
            widget.x -= step;
            break;
          case "ArrowUp":
            widget.y += step;
            break;
          case "ArrowDown":
            widget.y -= step;
            break;
        }
        return;
      }

      this.render();
      this.saveLayout();
      this.emitEvent("widgetMoved", { widget });
    }
  }

  // Layout persistence
  saveLayout() {
    const layoutData = {
      [this.currentBreakpoint]: this.layout.map((w) => ({
        i: w.id,
        x: w.x,
        y: w.y,
        w: w.w,
        h: w.h,
        static: w.static,
      })),
    };

    try {
      localStorage.setItem("dashboard-layout", JSON.stringify(layoutData));
      this.emitEvent("layoutSaved", { layout: layoutData });
    } catch (error) {
      console.error("Failed to save layout:", error);
    }
  }

  loadLayout(breakpoint) {
    try {
      const saved = localStorage.getItem("dashboard-layout");
      if (saved) {
        const layoutData = JSON.parse(saved);
        const breakpointLayout = layoutData[breakpoint];

        if (breakpointLayout) {
          breakpointLayout.forEach((item) => {
            const widget = this.widgets.get(item.i);
            if (widget) {
              widget.x = item.x;
              widget.y = item.y;
              widget.w = item.w;
              widget.h = item.h;
              widget.static = item.static;
            }
          });

          this.emitEvent("layoutLoaded", { layout: breakpointLayout });
        }
      }
    } catch (error) {
      console.error("Failed to load layout:", error);
    }
  }

  // Utility methods
  generateId() {
    return "widget_" + Math.random().toString(36).substr(2, 9);
  }

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

  emitEvent(eventName, detail) {
    const event = new CustomEvent(`dashboard:${eventName}`, {
      detail,
      bubbles: true,
      cancelable: true,
    });
    this.container.dispatchEvent(event);
  }

  // Public API methods
  getLayout() {
    return this.layout.map((w) => ({ ...w }));
  }

  setLayout(layout) {
    this.layout = layout.map((w) => ({ ...w }));
    this.widgets.clear();

    this.layout.forEach((widget) => {
      this.widgets.set(widget.id, widget);
    });

    this.render();
    this.saveLayout();
  }

  exportLayout() {
    return {
      layout: this.getLayout(),
      breakpoint: this.currentBreakpoint,
      options: { ...this.options },
    };
  }

  importLayout(data) {
    if (data.layout) {
      this.setLayout(data.layout);
    }
  }

  destroy() {
    if (this.resizeObserver) {
      this.resizeObserver.disconnect();
    }

    this.container.innerHTML = "";
    this.widgets.clear();
    this.layout = [];
  }
}

// Widget registry for reusable components
export class WidgetRegistry {
  constructor() {
    this.widgets = new Map();
  }

  register(type, config) {
    this.widgets.set(type, config);
  }

  create(type, props = {}) {
    const config = this.widgets.get(type);
    if (!config) {
      throw new Error(`Widget type '${type}' not found`);
    }

    return {
      ...config,
      props: { ...config.defaultProps, ...props },
    };
  }

  getTypes() {
    return Array.from(this.widgets.keys());
  }
}

// Default widget types
export const defaultWidgets = {
  metric: {
    type: "metric",
    title: "Metric Widget",
    w: 2,
    h: 2,
    component: (props) => {
      const div = document.createElement("div");
      div.className = "metric-widget";
      div.innerHTML = `
                <div class="metric-value">${props.value || 0}</div>
                <div class="metric-label">${props.label || "Metric"}</div>
                <div class="metric-change ${props.change >= 0 ? "positive" : "negative"}">
                    ${props.change >= 0 ? "+" : ""}${props.change || 0}%
                </div>
            `;
      return div;
    },
    defaultProps: {
      value: 0,
      label: "Metric",
      change: 0,
    },
  },

  chart: {
    type: "chart",
    title: "Chart Widget",
    w: 4,
    h: 3,
    component: (props) => {
      const div = document.createElement("div");
      div.className = "chart-widget";
      div.innerHTML = `
                <div class="chart-placeholder">
                    <div class="chart-icon">📊</div>
                    <p>${props.chartType || "Chart"} visualization</p>
                </div>
            `;
      return div;
    },
    defaultProps: {
      chartType: "Line",
    },
  },

  table: {
    type: "table",
    title: "Data Table",
    w: 6,
    h: 4,
    component: (props) => {
      const div = document.createElement("div");
      div.className = "table-widget";

      const table = document.createElement("table");
      table.className = "widget-table";

      // Create header
      if (props.columns) {
        const thead = document.createElement("thead");
        const headerRow = document.createElement("tr");
        props.columns.forEach((col) => {
          const th = document.createElement("th");
          th.textContent = col;
          headerRow.appendChild(th);
        });
        thead.appendChild(headerRow);
        table.appendChild(thead);
      }

      // Create body
      const tbody = document.createElement("tbody");
      if (props.data && props.data.length > 0) {
        props.data.forEach((row) => {
          const tr = document.createElement("tr");
          Object.values(row).forEach((cell) => {
            const td = document.createElement("td");
            td.textContent = cell;
            tr.appendChild(td);
          });
          tbody.appendChild(tr);
        });
      } else {
        const tr = document.createElement("tr");
        const td = document.createElement("td");
        td.colSpan = props.columns?.length || 1;
        td.textContent = "No data available";
        td.style.textAlign = "center";
        td.style.color = "#6b7280";
        tr.appendChild(td);
        tbody.appendChild(tr);
      }

      table.appendChild(tbody);
      div.appendChild(table);

      return div;
    },
    defaultProps: {
      columns: ["Column 1", "Column 2"],
      data: [],
    },
  },
};

// CSS styles
export const dashboardStyles = `
.dashboard-layout {
    position: relative;
    background: #f8fafc;
}

.dashboard-widget {
    box-sizing: border-box;
    user-select: none;
}

.dashboard-widget:hover .resize-handle {
    opacity: 1;
}

.dashboard-widget.dragging {
    z-index: 1000;
    opacity: 0.8;
}

.dashboard-widget.resizing {
    z-index: 1000;
}

.widget-header {
    cursor: move;
}

.widget-content {
    position: relative;
}

.widget-placeholder {
    animation: pulse 1s ease-in-out infinite alternate;
}

@keyframes pulse {
    from { opacity: 0.3; }
    to { opacity: 0.7; }
}

.metric-widget {
    text-align: center;
    padding: 20px;
}

.metric-value {
    font-size: 2rem;
    font-weight: bold;
    color: #1f2937;
}

.metric-label {
    font-size: 0.875rem;
    color: #6b7280;
    margin: 8px 0;
}

.metric-change {
    font-size: 0.875rem;
    font-weight: 500;
}

.metric-change.positive {
    color: #059669;
}

.metric-change.negative {
    color: #dc2626;
}

.chart-widget {
    display: flex;
    align-items: center;
    justify-content: center;
    height: 100%;
}

.chart-placeholder {
    text-align: center;
    color: #6b7280;
}

.chart-icon {
    font-size: 3rem;
    margin-bottom: 1rem;
}

.widget-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.875rem;
}

.widget-table th,
.widget-table td {
    padding: 8px 12px;
    text-align: left;
    border-bottom: 1px solid #e5e7eb;
}

.widget-table th {
    background: #f9fafb;
    font-weight: 600;
    color: #374151;
}

.grid-overlay {
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    pointer-events: none;
    z-index: 999;
    background-image:
        linear-gradient(to right, rgba(0,0,0,0.1) 1px, transparent 1px),
        linear-gradient(to bottom, rgba(0,0,0,0.1) 1px, transparent 1px);
    background-size: 60px 60px;
}
`;
\n\n// data-uploader.js\n// Data Uploader Component - File upload with validation and preview
export class DataUploader {
  constructor(element) {
    this.element = element;
    this.config = this.getConfig();
    this.uploadQueue = [];

    this.init();
  }

  getConfig() {
    const element = this.element;
    return {
      maxFileSize: parseInt(element.dataset.maxFileSize) || 10 * 1024 * 1024,
      allowedFormats: (
        element.dataset.allowedFormats || "csv,json,parquet"
      ).split(","),
      multiple: element.dataset.multiple === "true",
      autoUpload: element.dataset.autoUpload === "true",
    };
  }

  init() {
    this.createInterface();
    this.bindEvents();
  }

  createInterface() {
    this.element.innerHTML = `
      <div class="data-uploader">
        <div class="upload-zone" data-drop-zone>
          <div class="upload-icon">📁</div>
          <div class="upload-text">
            <p>Drop files here or click to browse</p>
            <p class="text-sm text-neutral-500">Max size: ${this.config.maxFileSize / 1024 / 1024}MB</p>
          </div>
          <input type="file" class="file-input" ${this.config.multiple ? "multiple" : ""} hidden>
        </div>
        <div class="upload-queue" data-upload-queue></div>
      </div>
    `;
  }

  bindEvents() {
    const fileInput = this.element.querySelector(".file-input");
    const dropZone = this.element.querySelector("[data-drop-zone]");

    dropZone.addEventListener("click", () => fileInput.click());
    fileInput.addEventListener("change", (e) =>
      this.handleFiles(e.target.files),
    );

    dropZone.addEventListener("dragover", (e) => {
      e.preventDefault();
      dropZone.classList.add("drag-over");
    });

    dropZone.addEventListener("dragleave", () => {
      dropZone.classList.remove("drag-over");
    });

    dropZone.addEventListener("drop", (e) => {
      e.preventDefault();
      dropZone.classList.remove("drag-over");
      this.handleFiles(e.dataTransfer.files);
    });
  }

  handleFiles(files) {
    Array.from(files).forEach((file) => {
      if (this.validateFile(file)) {
        this.addToQueue(file);
      }
    });
  }

  validateFile(file) {
    const extension = file.name.split(".").pop().toLowerCase();

    if (!this.config.allowedFormats.includes(extension)) {
      this.showError(`Unsupported format: ${extension}`);
      return false;
    }

    if (file.size > this.config.maxFileSize) {
      this.showError(`File too large: ${file.name}`);
      return false;
    }

    return true;
  }

  addToQueue(file) {
    this.uploadQueue.push(file);
    this.renderQueue();

    if (this.config.autoUpload) {
      this.uploadFile(file);
    }
  }

  renderQueue() {
    const queueContainer = this.element.querySelector("[data-upload-queue]");
    queueContainer.innerHTML = this.uploadQueue
      .map(
        (file) => `
      <div class="upload-item" data-file="${file.name}">
        <div class="file-info">
          <div class="file-name">${file.name}</div>
          <div class="file-size">${(file.size / 1024).toFixed(1)} KB</div>
        </div>
        <div class="upload-progress">
          <div class="progress-bar">
            <div class="progress-fill" style="width: 0%"></div>
          </div>
        </div>
      </div>
    `,
      )
      .join("");
  }

  async uploadFile(file) {
    // Placeholder upload implementation
    console.log("Uploading file:", file.name);
  }

  showError(message) {
    console.error("Upload error:", message);
  }
}
\n\n// echarts-dashboard.js\n/**
 * ECharts Dashboard Components for Pynomaly
 * Production-ready statistical charts, performance metrics, and anomaly visualization
 * with responsive design and accessibility features
 */

class EChartsDashboard {
  constructor() {
    this.charts = new Map();
    this.themes = {
      light: {
        backgroundColor: "#ffffff",
        textStyle: {
          color: "#1e293b",
        },
        title: {
          textStyle: {
            color: "#1e293b",
          },
        },
        legend: {
          textStyle: {
            color: "#64748b",
          },
        },
        grid: {
          borderColor: "#e2e8f0",
        },
        categoryAxis: {
          axisLine: { lineStyle: { color: "#e2e8f0" } },
          axisTick: { lineStyle: { color: "#e2e8f0" } },
          axisLabel: { color: "#64748b" },
          splitLine: { lineStyle: { color: "#f1f5f9" } },
        },
        valueAxis: {
          axisLine: { lineStyle: { color: "#e2e8f0" } },
          axisTick: { lineStyle: { color: "#e2e8f0" } },
          axisLabel: { color: "#64748b" },
          splitLine: { lineStyle: { color: "#f1f5f9" } },
        },
        colorBy: "series",
        color: [
          "#0ea5e9",
          "#22c55e",
          "#f59e0b",
          "#ef4444",
          "#8b5cf6",
          "#06b6d4",
        ],
      },
      dark: {
        backgroundColor: "#1e293b",
        textStyle: {
          color: "#f1f5f9",
        },
        title: {
          textStyle: {
            color: "#f1f5f9",
          },
        },
        legend: {
          textStyle: {
            color: "#94a3b8",
          },
        },
        grid: {
          borderColor: "#4b5563",
        },
        categoryAxis: {
          axisLine: { lineStyle: { color: "#4b5563" } },
          axisTick: { lineStyle: { color: "#4b5563" } },
          axisLabel: { color: "#94a3b8" },
          splitLine: { lineStyle: { color: "#374151" } },
        },
        valueAxis: {
          axisLine: { lineStyle: { color: "#4b5563" } },
          axisTick: { lineStyle: { color: "#4b5563" } },
          axisLabel: { color: "#94a3b8" },
          splitLine: { lineStyle: { color: "#374151" } },
        },
        colorBy: "series",
        color: [
          "#38bdf8",
          "#4ade80",
          "#fbbf24",
          "#f87171",
          "#a78bfa",
          "#22d3ee",
        ],
      },
    };
    this.currentTheme = "light";
    this.setupEventListeners();
  }

  setupEventListeners() {
    // Theme change listener
    document.addEventListener("theme-changed", (event) => {
      this.currentTheme = event.detail.theme;
      this.updateAllChartsTheme();
    });

    // Resize listener
    window.addEventListener(
      "resize",
      this.debounce(() => {
        this.resizeAllCharts();
      }, 250),
    );
  }

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

  getTheme() {
    return this.themes[this.currentTheme];
  }

  registerChart(id, chart) {
    this.charts.set(id, chart);
  }

  removeChart(id) {
    const chart = this.charts.get(id);
    if (chart) {
      chart.dispose();
    }
    this.charts.delete(id);
  }

  updateAllChartsTheme() {
    this.charts.forEach((chart) => {
      if (chart && !chart.isDisposed()) {
        const theme = this.getTheme();
        chart.setOption(theme, true);
        chart.resize();
      }
    });
  }

  resizeAllCharts() {
    this.charts.forEach((chart) => {
      if (chart && !chart.isDisposed()) {
        chart.resize();
      }
    });
  }
}

/**
 * Base ECharts Component
 * Provides common functionality for all chart types
 */
class BaseEChart {
  constructor(container, options = {}) {
    this.container =
      typeof container === "string"
        ? document.querySelector(container)
        : container;
    this.options = {
      responsive: true,
      accessibility: true,
      animation: true,
      theme: "light",
      ...options,
    };

    this.chart = null;
    this.data = [];

    this.init();
    this.setupAccessibility();

    // Register with dashboard
    const chartId = this.container.id || `chart-${Date.now()}`;
    echartsManager.registerChart(chartId, this.chart);
  }

  init() {
    if (!this.container) {
      throw new Error("Container element not found");
    }

    // Initialize ECharts instance
    this.chart = echarts.init(this.container, null, {
      renderer: "canvas",
      useDirtyRect: true, // Performance optimization
    });

    // Apply theme
    this.applyTheme();

    // Setup resize observer
    if (this.options.responsive) {
      this.setupResizeObserver();
    }

    // Setup event listeners
    this.setupEventListeners();
    this.enhanceInteractivity();
  }

  enhanceInteractivity() {
    this.chart.setOption({
      dataZoom: [
        {
          type: 'inside',
          start: 0,
          end: 100
        },
        {
          start: 0,
          end: 100
        }
      ],
      brush: {
        toolbox: ['rect', 'polygon', 'keep', 'clear'],
        xAxisIndex: 'all'
      }
    });
  }

  setupAccessibility() {
    if (!this.options.accessibility) return;

    // Add ARIA labels
    this.container.setAttribute("role", "img");
    this.container.setAttribute(
      "aria-label",
      this.options.title || "Chart visualization",
    );

    // Add description if provided
    if (this.options.description) {
      const descId = `${this.container.id || "chart"}-desc`;
      let descElement = document.getElementById(descId);

      if (!descElement) {
        descElement = document.createElement("div");
        descElement.id = descId;
        descElement.className = "sr-only";
        descElement.textContent = this.options.description;
        this.container.parentNode.insertBefore(
          descElement,
          this.container.nextSibling,
        );
      }

      this.container.setAttribute("aria-describedby", descId);
    }

    // Make container focusable
    this.container.setAttribute("tabindex", "0");
  }

  setupResizeObserver() {
    if ("ResizeObserver" in window) {
      this.resizeObserver = new ResizeObserver(() => {
        if (this.chart && !this.chart.isDisposed()) {
          this.chart.resize();
        }
      });
      this.resizeObserver.observe(this.container);
    }
  }

  setupEventListeners() {
    // Chart click events
    this.chart.on("click", (params) => {
      this.container.dispatchEvent(
        new CustomEvent("chart-click", {
          detail: { params, chart: this },
        }),
      );
    });

    // Chart hover events
    this.chart.on("mouseover", (params) => {
      this.container.dispatchEvent(
        new CustomEvent("chart-hover", {
          detail: { params, chart: this },
        }),
      );
    });

    // Keyboard navigation
    this.container.addEventListener("keydown", (event) => {
      this.handleKeyboardNavigation(event);
    });
  }

  handleKeyboardNavigation(event) {
    // Basic keyboard navigation for accessibility
    switch (event.key) {
      case "Enter":
      case " ":
        event.preventDefault();
        this.announceChartData();
        break;
      case "Escape":
        this.container.blur();
        break;
    }
  }

  announceChartData() {
    if (!this.options.accessibility) return;

    const announcement = this.generateDataAnnouncement();
    this.announceToScreenReader(announcement);
  }

  generateDataAnnouncement() {
    // Override in subclasses for specific announcements
    return `Chart with ${this.data.length} data points`;
  }

  announceToScreenReader(message) {
    const announcer =
      document.getElementById("chart-announcer") ||
      document.querySelector('[aria-live="polite"]');
    if (announcer) {
      announcer.textContent = message;
    }
  }

  applyTheme() {
    const theme = echartsManager.getTheme();
    this.chart.setOption(theme, true);
  }

  setData(data) {
    this.data = data;
    this.updateChart();
  }

  updateChart() {
    // Override in subclasses
    throw new Error("updateChart() method must be implemented by subclass");
  }

  resize() {
    if (this.chart && !this.chart.isDisposed()) {
      this.chart.resize();
    }
  }

  dispose() {
    if (this.resizeObserver) {
      this.resizeObserver.disconnect();
    }

    if (this.chart && !this.chart.isDisposed()) {
      this.chart.dispose();
    }
  }
}

/**
 * Performance Metrics Chart
 * Real-time system performance visualization
 */
class PerformanceMetricsChart extends BaseEChart {
  constructor(container, options = {}) {
    super(container, {
      title: "System Performance Metrics",
      description:
        "Real-time visualization of CPU, memory, and network utilization",
      metrics: ["cpu", "memory", "network"],
      updateInterval: 2000,
      maxDataPoints: 100,
      ...options,
    });

    this.metricColors = {
      cpu: "#ef4444",
      memory: "#f59e0b",
      network: "#22c55e",
      disk: "#8b5cf6",
      response_time: "#06b6d4",
    };

    this.setupRealTimeUpdates();
  }

  updateChart() {
    const option = {
      title: {
        text: this.options.title,
        left: "center",
        textStyle: {
          fontSize: 16,
          fontWeight: "bold",
        },
      },
      tooltip: {
        trigger: "axis",
        axisPointer: {
          type: "cross",
          label: {
            backgroundColor: "#6a7985",
          },
        },
        formatter: (params) => {
          let tooltip = `<strong>${params[0].axisValue}</strong><br/>`;
          params.forEach((param) => {
            const value =
              typeof param.value === "number"
                ? param.value.toFixed(2)
                : param.value;
            tooltip += `${param.marker} ${param.seriesName}: ${value}%<br/>`;
          });
          return tooltip;
        },
      },
      legend: {
        data: this.options.metrics.map((metric) => metric.toUpperCase()),
        top: 30,
        textStyle: {
          fontSize: 12,
        },
      },
      grid: {
        left: "3%",
        right: "4%",
        bottom: "3%",
        top: 80,
        containLabel: true,
      },
      xAxis: {
        type: "category",
        boundaryGap: false,
        data: this.data.map((d) => d.timestamp),
        axisLabel: {
          formatter: (value) => {
            const date = new Date(value);
            return date.toLocaleTimeString();
          },
        },
      },
      yAxis: {
        type: "value",
        min: 0,
        max: 100,
        axisLabel: {
          formatter: "{value}%",
        },
      },
      series: this.options.metrics.map((metric) => ({
        name: metric.toUpperCase(),
        type: "line",
        stack: false,
        smooth: true,
        symbol: "circle",
        symbolSize: 4,
        lineStyle: {
          width: 2,
        },
        areaStyle: {
          opacity: 0.1,
        },
        data: this.data.map((d) => d[metric] || 0),
        itemStyle: {
          color: this.metricColors[metric] || "#0ea5e9",
        },
      })),
      animation: this.options.animation,
      animationDuration: 1000,
      animationEasing: "cubicOut",
    };

    this.chart.setOption(option);
  }

  setupRealTimeUpdates() {
    if (this.options.updateInterval > 0) {
      this.updateTimer = setInterval(() => {
        this.addRandomDataPoint();
      }, this.options.updateInterval);
    }
  }

  addRandomDataPoint() {
    const now = new Date();
    const newPoint = {
      timestamp: now.toISOString(),
    };

    // Generate realistic performance data
    this.options.metrics.forEach((metric) => {
      switch (metric) {
        case "cpu":
          newPoint[metric] = Math.max(
            0,
            Math.min(
              100,
              (this.data.length > 0
                ? this.data[this.data.length - 1][metric]
                : 50) +
                (Math.random() - 0.5) * 20,
            ),
          );
          break;
        case "memory":
          newPoint[metric] = Math.max(
            0,
            Math.min(
              100,
              (this.data.length > 0
                ? this.data[this.data.length - 1][metric]
                : 60) +
                (Math.random() - 0.5) * 10,
            ),
          );
          break;
        case "network":
          newPoint[metric] = Math.max(
            0,
            Math.min(100, Math.random() * 30 + 10),
          );
          break;
        default:
          newPoint[metric] = Math.random() * 100;
      }
    });

    this.data.push(newPoint);

    // Keep only recent data points
    if (this.data.length > this.options.maxDataPoints) {
      this.data = this.data.slice(-this.options.maxDataPoints);
    }

    this.updateChart();

    // Announce significant changes
    if (this.options.accessibility) {
      const highUsage = this.options.metrics.find(
        (metric) => newPoint[metric] > 90,
      );
      if (highUsage) {
        this.announceToScreenReader(
          `High ${highUsage} usage detected: ${newPoint[highUsage].toFixed(1)}%`,
        );
      }
    }
  }

  generateDataAnnouncement() {
    if (this.data.length === 0) return "No performance data available";

    const latest = this.data[this.data.length - 1];
    const metrics = this.options.metrics
      .map((metric) => `${metric}: ${(latest[metric] || 0).toFixed(1)}%`)
      .join(", ");

    return `Current performance metrics: ${metrics}`;
  }

  dispose() {
    if (this.updateTimer) {
      clearInterval(this.updateTimer);
    }
    super.dispose();
  }
}

/**
 * Anomaly Distribution Chart
 * Statistical visualization of anomaly patterns
 */
class AnomalyDistributionChart extends BaseEChart {
  constructor(container, options = {}) {
    super(container, {
      title: "Anomaly Distribution Analysis",
      description:
        "Statistical distribution of detected anomalies by type and confidence",
      chartType: "pie", // 'pie', 'bar', 'histogram'
      showConfidenceRanges: true,
      ...options,
    });
  }

  updateChart() {
    if (this.options.chartType === "pie") {
      this.updatePieChart();
    } else if (this.options.chartType === "bar") {
      this.updateBarChart();
    } else if (this.options.chartType === "histogram") {
      this.updateHistogramChart();
    }
  }

  updatePieChart() {
    // Group anomalies by type
    const typeDistribution = this.data.reduce((acc, anomaly) => {
      const type = anomaly.type || "Unknown";
      acc[type] = (acc[type] || 0) + 1;
      return acc;
    }, {});

    const pieData = Object.entries(typeDistribution).map(([type, count]) => ({
      name: type,
      value: count,
      itemStyle: {
        color: this.getAnomalyTypeColor(type),
      },
    }));

    const option = {
      title: {
        text: this.options.title,
        left: "center",
        textStyle: {
          fontSize: 16,
          fontWeight: "bold",
        },
      },
      tooltip: {
        trigger: "item",
        formatter: (params) => {
          const percentage = ((params.value / this.data.length) * 100).toFixed(
            1,
          );
          return `<strong>${params.name}</strong><br/>
                  Count: ${params.value}<br/>
                  Percentage: ${percentage}%`;
        },
      },
      legend: {
        orient: "vertical",
        left: "left",
        top: "center",
        textStyle: {
          fontSize: 12,
        },
      },
      series: [
        {
          name: "Anomaly Types",
          type: "pie",
          radius: ["40%", "70%"],
          center: ["60%", "50%"],
          avoidLabelOverlap: false,
          itemStyle: {
            borderRadius: 4,
            borderColor: "#fff",
            borderWidth: 2,
          },
          label: {
            show: false,
            position: "center",
          },
          emphasis: {
            label: {
              show: true,
              fontSize: "16",
              fontWeight: "bold",
            },
            itemStyle: {
              shadowBlur: 10,
              shadowOffsetX: 0,
              shadowColor: "rgba(0, 0, 0, 0.5)",
            },
          },
          labelLine: {
            show: false,
          },
          data: pieData,
        },
      ],
      animation: this.options.animation,
      animationType: "expansion",
      animationEasing: "elasticOut",
      animationDelay: (idx) => idx * 100,
    };

    this.chart.setOption(option);
  }

  updateBarChart() {
    // Group by confidence ranges
    const confidenceRanges = {
      "Low (0-0.5)": 0,
      "Medium (0.5-0.8)": 0,
      "High (0.8-0.95)": 0,
      "Very High (0.95+)": 0,
    };

    this.data.forEach((anomaly) => {
      const confidence = anomaly.confidence || 0;
      if (confidence < 0.5) {
        confidenceRanges["Low (0-0.5)"]++;
      } else if (confidence < 0.8) {
        confidenceRanges["Medium (0.5-0.8)"]++;
      } else if (confidence < 0.95) {
        confidenceRanges["High (0.8-0.95)"]++;
      } else {
        confidenceRanges["Very High (0.95+)"]++;
      }
    });

    const option = {
      title: {
        text: "Anomalies by Confidence Level",
        left: "center",
        textStyle: {
          fontSize: 16,
          fontWeight: "bold",
        },
      },
      tooltip: {
        trigger: "axis",
        axisPointer: {
          type: "shadow",
        },
        formatter: (params) => {
          const param = params[0];
          const percentage = ((param.value / this.data.length) * 100).toFixed(
            1,
          );
          return `<strong>${param.name}</strong><br/>
                  Count: ${param.value}<br/>
                  Percentage: ${percentage}%`;
        },
      },
      grid: {
        left: "3%",
        right: "4%",
        bottom: "3%",
        top: 60,
        containLabel: true,
      },
      xAxis: {
        type: "category",
        data: Object.keys(confidenceRanges),
        axisLabel: {
          interval: 0,
          rotate: 45,
        },
      },
      yAxis: {
        type: "value",
        name: "Count",
        nameLocation: "middle",
        nameGap: 50,
      },
      series: [
        {
          name: "Anomalies",
          type: "bar",
          data: Object.values(confidenceRanges),
          itemStyle: {
            color: (params) => {
              const colors = ["#fbbf24", "#f59e0b", "#ef4444", "#dc2626"];
              return colors[params.dataIndex] || "#0ea5e9";
            },
            borderRadius: [4, 4, 0, 0],
          },
          emphasis: {
            itemStyle: {
              shadowBlur: 10,
              shadowColor: "rgba(0, 0, 0, 0.3)",
            },
          },
        },
      ],
      animation: this.options.animation,
      animationDelay: (idx) => idx * 100,
    };

    this.chart.setOption(option);
  }

  updateHistogramChart() {
    // Create histogram of anomaly scores
    const scores = this.data.map((d) => d.score || d.confidence || 0);
    const bins = this.createHistogramBins(scores, 20);

    const option = {
      title: {
        text: "Anomaly Score Distribution",
        left: "center",
        textStyle: {
          fontSize: 16,
          fontWeight: "bold",
        },
      },
      tooltip: {
        trigger: "axis",
        axisPointer: {
          type: "shadow",
        },
        formatter: (params) => {
          const param = params[0];
          return `<strong>Score Range: ${param.name}</strong><br/>
                  Count: ${param.value}<br/>
                  Density: ${(param.value / this.data.length).toFixed(3)}`;
        },
      },
      grid: {
        left: "3%",
        right: "4%",
        bottom: "3%",
        top: 60,
        containLabel: true,
      },
      xAxis: {
        type: "category",
        data: bins.map((bin) => bin.range),
        name: "Anomaly Score",
        nameLocation: "middle",
        nameGap: 30,
      },
      yAxis: {
        type: "value",
        name: "Frequency",
        nameLocation: "middle",
        nameGap: 50,
      },
      series: [
        {
          name: "Frequency",
          type: "bar",
          data: bins.map((bin) => bin.count),
          itemStyle: {
            color: "#0ea5e9",
            borderRadius: [2, 2, 0, 0],
          },
          emphasis: {
            itemStyle: {
              color: "#0284c7",
            },
          },
        },
      ],
      animation: this.options.animation,
    };

    this.chart.setOption(option);
  }

  createHistogramBins(data, numBins) {
    const min = Math.min(...data);
    const max = Math.max(...data);
    const binWidth = (max - min) / numBins;

    const bins = Array.from({ length: numBins }, (_, i) => ({
      range: `${(min + i * binWidth).toFixed(2)}-${(min + (i + 1) * binWidth).toFixed(2)}`,
      count: 0,
      min: min + i * binWidth,
      max: min + (i + 1) * binWidth,
    }));

    data.forEach((value) => {
      const binIndex = Math.min(
        Math.floor((value - min) / binWidth),
        numBins - 1,
      );
      bins[binIndex].count++;
    });

    return bins;
  }

  getAnomalyTypeColor(type) {
    const colors = {
      "Statistical Outlier": "#ef4444",
      "Temporal Anomaly": "#f59e0b",
      "Pattern Deviation": "#22c55e",
      "Threshold Violation": "#8b5cf6",
      "Trend Anomaly": "#06b6d4",
      Unknown: "#6b7280",
    };
    return colors[type] || "#0ea5e9";
  }

  generateDataAnnouncement() {
    if (this.data.length === 0) return "No anomaly data available";

    const types = [...new Set(this.data.map((d) => d.type || "Unknown"))];
    const avgConfidence =
      this.data.reduce((sum, d) => sum + (d.confidence || 0), 0) /
      this.data.length;

    return `${this.data.length} anomalies detected across ${types.length} types. Average confidence: ${(avgConfidence * 100).toFixed(1)}%`;
  }
}

/**
 * Detection Timeline Chart
 * Time-based visualization of anomaly detection events
 */
class DetectionTimelineChart extends BaseEChart {
  constructor(container, options = {}) {
    super(container, {
      title: "Anomaly Detection Timeline",
      description:
        "Chronological view of anomaly detection events with severity indicators",
      timeRange: "24h", // '1h', '6h', '24h', '7d', '30d'
      showSeverityLevels: true,
      groupByHour: false,
      ...options,
    });
  }

  updateChart() {
    // Process data based on time range and grouping
    const processedData = this.processTimelineData();

    const option = {
      title: {
        text: this.options.title,
        left: "center",
        textStyle: {
          fontSize: 16,
          fontWeight: "bold",
        },
      },
      tooltip: {
        trigger: "axis",
        axisPointer: {
          type: "cross",
          animation: false,
        },
        formatter: (params) => {
          let tooltip = `<strong>${params[0].axisValue}</strong><br/>`;
          params.forEach((param) => {
            tooltip += `${param.marker} ${param.seriesName}: ${param.value}<br/>`;
          });
          return tooltip;
        },
      },
      legend: {
        data: ["Low Severity", "Medium Severity", "High Severity", "Critical"],
        top: 30,
        textStyle: {
          fontSize: 12,
        },
      },
      grid: {
        left: "3%",
        right: "4%",
        bottom: "3%",
        top: 80,
        containLabel: true,
      },
      xAxis: {
        type: "category",
        data: processedData.timeline,
        axisLabel: {
          formatter: (value) => {
            const date = new Date(value);
            if (
              this.options.timeRange === "1h" ||
              this.options.timeRange === "6h"
            ) {
              return date.toLocaleTimeString();
            } else if (this.options.timeRange === "24h") {
              return `${date.getHours()}:00`;
            } else {
              return date.toLocaleDateString();
            }
          },
        },
      },
      yAxis: {
        type: "value",
        name: "Count",
        nameLocation: "middle",
        nameGap: 50,
        min: 0,
      },
      series: [
        {
          name: "Low Severity",
          type: "bar",
          stack: "severity",
          data: processedData.low,
          itemStyle: { color: "#22c55e" },
        },
        {
          name: "Medium Severity",
          type: "bar",
          stack: "severity",
          data: processedData.medium,
          itemStyle: { color: "#f59e0b" },
        },
        {
          name: "High Severity",
          type: "bar",
          stack: "severity",
          data: processedData.high,
          itemStyle: { color: "#ef4444" },
        },
        {
          name: "Critical",
          type: "bar",
          stack: "severity",
          data: processedData.critical,
          itemStyle: { color: "#dc2626" },
        },
      ],
      animation: this.options.animation,
      animationDelay: (idx) => idx * 50,
    };

    this.chart.setOption(option);
  }

  processTimelineData() {
    // Create time buckets based on time range
    const buckets = this.createTimeBuckets();

    // Initialize data structure
    const result = {
      timeline: buckets.map((bucket) => bucket.label),
      low: new Array(buckets.length).fill(0),
      medium: new Array(buckets.length).fill(0),
      high: new Array(buckets.length).fill(0),
      critical: new Array(buckets.length).fill(0),
    };

    // Categorize and count anomalies
    this.data.forEach((anomaly) => {
      const timestamp = new Date(anomaly.timestamp);
      const bucketIndex = this.findTimeBucket(timestamp, buckets);

      if (bucketIndex >= 0) {
        const severity = this.getSeverityLevel(anomaly);
        result[severity][bucketIndex]++;
      }
    });

    return result;
  }

  createTimeBuckets() {
    const now = new Date();
    const buckets = [];

    switch (this.options.timeRange) {
      case "1h":
        for (let i = 59; i >= 0; i--) {
          const time = new Date(now.getTime() - i * 60000);
          buckets.push({
            start: time,
            end: new Date(time.getTime() + 60000),
            label: time.toISOString(),
          });
        }
        break;

      case "6h":
        for (let i = 35; i >= 0; i--) {
          const time = new Date(now.getTime() - i * 600000); // 10-minute buckets
          buckets.push({
            start: time,
            end: new Date(time.getTime() + 600000),
            label: time.toISOString(),
          });
        }
        break;

      case "24h":
        for (let i = 23; i >= 0; i--) {
          const time = new Date(now.getTime() - i * 3600000); // 1-hour buckets
          buckets.push({
            start: time,
            end: new Date(time.getTime() + 3600000),
            label: time.toISOString(),
          });
        }
        break;

      case "7d":
        for (let i = 6; i >= 0; i--) {
          const time = new Date(now.getTime() - i * 86400000); // 1-day buckets
          buckets.push({
            start: time,
            end: new Date(time.getTime() + 86400000),
            label: time.toISOString(),
          });
        }
        break;

      default: // 30d
        for (let i = 29; i >= 0; i--) {
          const time = new Date(now.getTime() - i * 86400000); // 1-day buckets
          buckets.push({
            start: time,
            end: new Date(time.getTime() + 86400000),
            label: time.toISOString(),
          });
        }
    }

    return buckets;
  }

  findTimeBucket(timestamp, buckets) {
    return buckets.findIndex(
      (bucket) => timestamp >= bucket.start && timestamp < bucket.end,
    );
  }

  getSeverityLevel(anomaly) {
    const confidence = anomaly.confidence || 0;
    const score = anomaly.score || confidence;

    if (score >= 0.95) return "critical";
    if (score >= 0.8) return "high";
    if (score >= 0.5) return "medium";
    return "low";
  }

  generateDataAnnouncement() {
    if (this.data.length === 0) return "No detection events in timeline";

    const severityCounts = {
      low: 0,
      medium: 0,
      high: 0,
      critical: 0,
    };

    this.data.forEach((anomaly) => {
      const severity = this.getSeverityLevel(anomaly);
      severityCounts[severity]++;
    });

    const total = this.data.length;
    const critical = severityCounts.critical;
    const high = severityCounts.high;

    return `Timeline shows ${total} detection events. ${critical} critical and ${high} high severity anomalies detected.`;
  }
}

// Initialize the ECharts dashboard manager
const echartsManager = new EChartsDashboard();

// Export classes for use in other modules
if (typeof module !== "undefined" && module.exports) {
  module.exports = {
    EChartsDashboard,
    BaseEChart,
    PerformanceMetricsChart,
    AnomalyDistributionChart,
    DetectionTimelineChart,
  };
} else {
  // Browser environment
  window.EChartsDashboard = EChartsDashboard;
  window.BaseEChart = BaseEChart;
  window.PerformanceMetricsChart = PerformanceMetricsChart;
  window.AnomalyDistributionChart = AnomalyDistributionChart;
  window.DetectionTimelineChart = DetectionTimelineChart;
  window.echartsManager = echartsManager;
}
\n\n// echarts-demo.js\n/**
 * ECharts Demo Application
 * Interactive demonstration of all ECharts dashboard components
 */

class EChartsDemo {
  constructor() {
    this.charts = new Map();
    this.demoData = this.generateDemoData();
    this.realTimeInterval = null;
    this.isRealTimeEnabled = false;

    this.init();
  }

  init() {
    this.setupDemoControls();
    this.createAllDemos();
    this.setupEventListeners();
  }

  generateDemoData() {
    const now = new Date();

    // Performance metrics data
    const performanceData = [];
    for (let i = 0; i < 50; i++) {
      const timestamp = new Date(now.getTime() - (50 - i) * 60000);
      performanceData.push({
        timestamp: timestamp.toISOString(),
        cpu: Math.random() * 80 + 10,
        memory: Math.random() * 70 + 20,
        network: Math.random() * 50 + 5,
        disk: Math.random() * 60 + 10,
      });
    }

    // Anomaly distribution data
    const anomalyTypes = [
      "Statistical Outlier",
      "Temporal Anomaly",
      "Pattern Deviation",
      "Threshold Violation",
      "Trend Anomaly",
    ];

    const anomalyData = [];
    for (let i = 0; i < 100; i++) {
      const confidence = Math.random();
      anomalyData.push({
        id: i,
        type: anomalyTypes[Math.floor(Math.random() * anomalyTypes.length)],
        confidence: confidence,
        score: confidence,
        timestamp: new Date(
          now.getTime() - Math.random() * 86400000 * 7,
        ).toISOString(),
        severity:
          confidence > 0.8 ? "high" : confidence > 0.5 ? "medium" : "low",
      });
    }

    // Detection timeline data
    const timelineData = [];
    for (let i = 0; i < 200; i++) {
      const confidence = Math.random();
      timelineData.push({
        timestamp: new Date(
          now.getTime() - Math.random() * 86400000,
        ).toISOString(),
        confidence: confidence,
        score: confidence,
        type: anomalyTypes[Math.floor(Math.random() * anomalyTypes.length)],
      });
    }

    return {
      performance: performanceData,
      anomalies: anomalyData,
      timeline: timelineData,
    };
  }

  setupDemoControls() {
    const controlsContainer = document.getElementById("echarts-demo-controls");
    if (!controlsContainer) return;

    controlsContainer.innerHTML = `
      <div class="demo-controls-grid">
        <div class="control-group">
          <h3>Real-Time Simulation</h3>
          <button id="echarts-realtime-toggle" class="btn btn-primary">Start Real-Time</button>
          <label>
            Update Interval:
            <select id="echarts-update-interval">
              <option value="1000">1 second</option>
              <option value="2000" selected>2 seconds</option>
              <option value="5000">5 seconds</option>
            </select>
          </label>
        </div>

        <div class="control-group">
          <h3>Theme</h3>
          <button id="echarts-theme-toggle" class="btn btn-secondary">Switch to Dark</button>
          <label>
            <input type="checkbox" id="echarts-high-contrast" /> High Contrast
          </label>
        </div>

        <div class="control-group">
          <h3>Chart Controls</h3>
          <button id="echarts-refresh-data" class="btn btn-secondary">Refresh Data</button>
          <button id="echarts-export-charts" class="btn btn-secondary">Export Charts</button>
        </div>

        <div class="control-group">
          <h3>Performance</h3>
          <label>
            Metrics:
            <select id="performance-metrics" multiple>
              <option value="cpu" selected>CPU</option>
              <option value="memory" selected>Memory</option>
              <option value="network" selected>Network</option>
              <option value="disk">Disk I/O</option>
            </select>
          </label>
        </div>

        <div class="control-group">
          <h3>Anomaly Charts</h3>
          <label>
            Distribution Type:
            <select id="distribution-type">
              <option value="pie" selected>Pie Chart</option>
              <option value="bar">Bar Chart</option>
              <option value="histogram">Histogram</option>
            </select>
          </label>
        </div>

        <div class="control-group">
          <h3>Timeline</h3>
          <label>
            Time Range:
            <select id="timeline-range">
              <option value="1h">1 Hour</option>
              <option value="6h">6 Hours</option>
              <option value="24h" selected>24 Hours</option>
              <option value="7d">7 Days</option>
              <option value="30d">30 Days</option>
            </select>
          </label>
        </div>
      </div>

      <div id="echarts-announcer" aria-live="polite" class="sr-only"></div>
    `;
  }

  createAllDemos() {
    this.createPerformanceDemo();
    this.createAnomalyDistributionDemo();
    this.createTimelineDemo();
    this.createDashboardDemo();
  }

  createPerformanceDemo() {
    const container = document.getElementById("performance-demo");
    if (!container) return;

    container.innerHTML = `
      <div class="demo-section">
        <h2>Performance Metrics Chart</h2>
        <p>Real-time system performance monitoring with CPU, memory, network, and disk utilization metrics.</p>

        <div class="chart-controls">
          <label>
            <input type="checkbox" id="performance-animation" checked /> Enable Animations
          </label>
          <label>
            Max Data Points:
            <select id="performance-max-points">
              <option value="50">50 points</option>
              <option value="100" selected>100 points</option>
              <option value="200">200 points</option>
            </select>
          </label>
          <button id="performance-reset" class="btn btn-sm">Reset Data</button>
        </div>

        <div id="performance-chart" class="chart-container" style="height: 400px;"></div>

        <div class="chart-info">
          <h4>Features:</h4>
          <ul>
            <li>Real-time data streaming</li>
            <li>Multiple metric tracking</li>
            <li>Smooth line animations</li>
            <li>Interactive tooltips</li>
            <li>Responsive design</li>
            <li>Alert notifications</li>
          </ul>
        </div>
      </div>
    `;

    // Initialize chart
    const chart = new PerformanceMetricsChart("#performance-chart", {
      title: "System Performance Metrics",
      description:
        "Real-time monitoring of system CPU, memory, and network utilization",
      metrics: ["cpu", "memory", "network"],
      updateInterval: 0, // We'll control updates manually
      maxDataPoints: 100,
      animation: true,
    });

    chart.setData(this.demoData.performance);
    this.charts.set("performance", chart);

    // Setup controls
    this.setupPerformanceControls();
  }

  setupPerformanceControls() {
    document
      .getElementById("performance-animation")
      ?.addEventListener("change", (e) => {
        const chart = this.charts.get("performance");
        if (chart) {
          chart.options.animation = e.target.checked;
          chart.updateChart();
        }
      });

    document
      .getElementById("performance-max-points")
      ?.addEventListener("change", (e) => {
        const chart = this.charts.get("performance");
        if (chart) {
          chart.options.maxDataPoints = parseInt(e.target.value);
          chart.setData(
            this.demoData.performance.slice(-chart.options.maxDataPoints),
          );
        }
      });

    document
      .getElementById("performance-reset")
      ?.addEventListener("click", () => {
        this.demoData.performance = this.generateDemoData().performance;
        const chart = this.charts.get("performance");
        if (chart) {
          chart.setData(this.demoData.performance);
        }
      });

    document
      .getElementById("performance-metrics")
      ?.addEventListener("change", (e) => {
        const selectedMetrics = Array.from(e.target.selectedOptions).map(
          (option) => option.value,
        );
        const chart = this.charts.get("performance");
        if (chart) {
          chart.options.metrics = selectedMetrics;
          chart.updateChart();
        }
      });
  }

  createAnomalyDistributionDemo() {
    const container = document.getElementById("anomaly-distribution-demo");
    if (!container) return;

    container.innerHTML = `
      <div class="demo-section">
        <h2>Anomaly Distribution Analysis</h2>
        <p>Statistical visualization of detected anomalies by type, confidence level, and distribution patterns.</p>

        <div class="chart-controls">
          <label>
            Chart Type:
            <select id="distribution-chart-type">
              <option value="pie" selected>Pie Chart</option>
              <option value="bar">Bar Chart</option>
              <option value="histogram">Histogram</option>
            </select>
          </label>
          <label>
            <input type="checkbox" id="distribution-animation" checked /> Enable Animations
          </label>
          <button id="distribution-randomize" class="btn btn-sm">Randomize Data</button>
        </div>

        <div id="anomaly-distribution-chart" class="chart-container" style="height: 400px;"></div>

        <div class="stats-panel">
          <h4>Statistics:</h4>
          <div id="distribution-stats">
            <div class="stat-item">
              <span class="stat-label">Total Anomalies:</span>
              <span class="stat-value" id="total-anomalies-count">-</span>
            </div>
            <div class="stat-item">
              <span class="stat-label">Average Confidence:</span>
              <span class="stat-value" id="avg-confidence-value">-</span>
            </div>
            <div class="stat-item">
              <span class="stat-label">High Confidence:</span>
              <span class="stat-value" id="high-confidence-count">-</span>
            </div>
          </div>
        </div>

        <div class="chart-info">
          <h4>Chart Types:</h4>
          <ul>
            <li><strong>Pie Chart</strong> - Distribution by anomaly type</li>
            <li><strong>Bar Chart</strong> - Confidence level ranges</li>
            <li><strong>Histogram</strong> - Score distribution density</li>
          </ul>
        </div>
      </div>
    `;

    // Initialize chart
    const chart = new AnomalyDistributionChart("#anomaly-distribution-chart", {
      title: "Anomaly Distribution by Type",
      description:
        "Pie chart showing distribution of anomaly types detected in the system",
      chartType: "pie",
      animation: true,
    });

    chart.setData(this.demoData.anomalies);
    this.charts.set("distribution", chart);

    // Update statistics
    this.updateDistributionStats();

    // Setup controls
    this.setupDistributionControls();
  }

  setupDistributionControls() {
    document
      .getElementById("distribution-chart-type")
      ?.addEventListener("change", (e) => {
        const chart = this.charts.get("distribution");
        if (chart) {
          chart.options.chartType = e.target.value;

          // Update title based on chart type
          const titles = {
            pie: "Anomaly Distribution by Type",
            bar: "Anomalies by Confidence Level",
            histogram: "Anomaly Score Distribution",
          };
          chart.options.title = titles[e.target.value];
          chart.updateChart();
        }
      });

    document
      .getElementById("distribution-animation")
      ?.addEventListener("change", (e) => {
        const chart = this.charts.get("distribution");
        if (chart) {
          chart.options.animation = e.target.checked;
          chart.updateChart();
        }
      });

    document
      .getElementById("distribution-randomize")
      ?.addEventListener("click", () => {
        this.demoData.anomalies = this.generateDemoData().anomalies;
        const chart = this.charts.get("distribution");
        if (chart) {
          chart.setData(this.demoData.anomalies);
          this.updateDistributionStats();
        }
      });

    document
      .getElementById("distribution-type")
      ?.addEventListener("change", (e) => {
        const chart = this.charts.get("distribution");
        if (chart) {
          chart.options.chartType = e.target.value;
          chart.updateChart();
        }
      });
  }

  updateDistributionStats() {
    const data = this.demoData.anomalies;
    const total = data.length;
    const avgConfidence =
      data.reduce((sum, d) => sum + d.confidence, 0) / total;
    const highConfidence = data.filter((d) => d.confidence > 0.8).length;

    document.getElementById("total-anomalies-count").textContent = total;
    document.getElementById("avg-confidence-value").textContent =
      (avgConfidence * 100).toFixed(1) + "%";
    document.getElementById("high-confidence-count").textContent =
      highConfidence;
  }

  createTimelineDemo() {
    const container = document.getElementById("timeline-demo");
    if (!container) return;

    container.innerHTML = `
      <div class="demo-section">
        <h2>Detection Timeline</h2>
        <p>Chronological visualization of anomaly detection events with severity indicators and time-based analysis.</p>

        <div class="chart-controls">
          <label>
            Time Range:
            <select id="timeline-time-range">
              <option value="1h">1 Hour</option>
              <option value="6h">6 Hours</option>
              <option value="24h" selected>24 Hours</option>
              <option value="7d">7 Days</option>
              <option value="30d">30 Days</option>
            </select>
          </label>
          <label>
            <input type="checkbox" id="timeline-severity" checked /> Show Severity Levels
          </label>
          <button id="timeline-refresh" class="btn btn-sm">Refresh Timeline</button>
        </div>

        <div id="detection-timeline-chart" class="chart-container" style="height: 400px;"></div>

        <div class="timeline-summary">
          <h4>Timeline Summary:</h4>
          <div id="timeline-summary-content">
            <div class="summary-item">
              <span class="severity-indicator critical"></span>
              <span>Critical: <strong id="critical-count">0</strong></span>
            </div>
            <div class="summary-item">
              <span class="severity-indicator high"></span>
              <span>High: <strong id="high-count">0</strong></span>
            </div>
            <div class="summary-item">
              <span class="severity-indicator medium"></span>
              <span>Medium: <strong id="medium-count">0</strong></span>
            </div>
            <div class="summary-item">
              <span class="severity-indicator low"></span>
              <span>Low: <strong id="low-count">0</strong></span>
            </div>
          </div>
        </div>

        <div class="chart-info">
          <h4>Features:</h4>
          <ul>
            <li>Multiple time range options</li>
            <li>Severity level stacking</li>
            <li>Interactive timeline navigation</li>
            <li>Real-time updates</li>
            <li>Trend analysis</li>
          </ul>
        </div>
      </div>
    `;

    // Initialize chart
    const chart = new DetectionTimelineChart("#detection-timeline-chart", {
      title: "Anomaly Detection Timeline - 24 Hours",
      description:
        "Stacked bar chart showing anomaly detection events over time by severity level",
      timeRange: "24h",
      showSeverityLevels: true,
      animation: true,
    });

    chart.setData(this.demoData.timeline);
    this.charts.set("timeline", chart);

    // Update summary
    this.updateTimelineSummary();

    // Setup controls
    this.setupTimelineControls();
  }

  setupTimelineControls() {
    document
      .getElementById("timeline-time-range")
      ?.addEventListener("change", (e) => {
        const chart = this.charts.get("timeline");
        if (chart) {
          chart.options.timeRange = e.target.value;
          chart.options.title = `Anomaly Detection Timeline - ${e.target.value.toUpperCase()}`;
          chart.updateChart();
        }
      });

    document
      .getElementById("timeline-severity")
      ?.addEventListener("change", (e) => {
        const chart = this.charts.get("timeline");
        if (chart) {
          chart.options.showSeverityLevels = e.target.checked;
          chart.updateChart();
        }
      });

    document
      .getElementById("timeline-refresh")
      ?.addEventListener("click", () => {
        this.demoData.timeline = this.generateDemoData().timeline;
        const chart = this.charts.get("timeline");
        if (chart) {
          chart.setData(this.demoData.timeline);
          this.updateTimelineSummary();
        }
      });

    document
      .getElementById("timeline-range")
      ?.addEventListener("change", (e) => {
        const chart = this.charts.get("timeline");
        if (chart) {
          chart.options.timeRange = e.target.value;
          chart.updateChart();
        }
      });
  }

  updateTimelineSummary() {
    const data = this.demoData.timeline;
    const counts = { critical: 0, high: 0, medium: 0, low: 0 };

    data.forEach((item) => {
      const confidence = item.confidence || 0;
      if (confidence >= 0.95) counts.critical++;
      else if (confidence >= 0.8) counts.high++;
      else if (confidence >= 0.5) counts.medium++;
      else counts.low++;
    });

    document.getElementById("critical-count").textContent = counts.critical;
    document.getElementById("high-count").textContent = counts.high;
    document.getElementById("medium-count").textContent = counts.medium;
    document.getElementById("low-count").textContent = counts.low;
  }

  createDashboardDemo() {
    const container = document.getElementById("echarts-dashboard-demo");
    if (!container) return;

    container.innerHTML = `
      <div class="demo-section">
        <h2>Integrated Dashboard</h2>
        <p>Comprehensive anomaly detection dashboard combining multiple chart types for complete system overview.</p>

        <div class="dashboard-grid">
          <div class="dashboard-widget">
            <h3>Performance Overview</h3>
            <div id="dashboard-performance" class="mini-chart" style="height: 200px;"></div>
          </div>

          <div class="dashboard-widget">
            <h3>Anomaly Types</h3>
            <div id="dashboard-distribution" class="mini-chart" style="height: 200px;"></div>
          </div>

          <div class="dashboard-widget">
            <h3>Detection Events</h3>
            <div id="dashboard-timeline" class="mini-chart" style="height: 200px;"></div>
          </div>

          <div class="dashboard-widget stats-summary">
            <h3>Key Metrics</h3>
            <div class="metrics-grid">
              <div class="metric">
                <div class="metric-value" id="dashboard-total-anomalies">0</div>
                <div class="metric-label">Total Anomalies</div>
              </div>
              <div class="metric">
                <div class="metric-value" id="dashboard-avg-confidence">0%</div>
                <div class="metric-label">Avg Confidence</div>
              </div>
              <div class="metric">
                <div class="metric-value" id="dashboard-critical-alerts">0</div>
                <div class="metric-label">Critical Alerts</div>
              </div>
              <div class="metric">
                <div class="metric-value" id="dashboard-system-health">Good</div>
                <div class="metric-label">System Health</div>
              </div>
            </div>
          </div>
        </div>

        <div class="dashboard-controls">
          <button id="dashboard-refresh" class="btn btn-primary">Refresh Dashboard</button>
          <button id="dashboard-export" class="btn btn-secondary">Export Data</button>
          <label>
            <input type="checkbox" id="dashboard-realtime" /> Real-time Updates
          </label>
        </div>
      </div>
    `;

    // Initialize mini charts
    this.createDashboardCharts();
    this.updateDashboardMetrics();
  }

  createDashboardCharts() {
    // Mini performance chart
    const perfChart = new PerformanceMetricsChart("#dashboard-performance", {
      title: "",
      metrics: ["cpu", "memory"],
      updateInterval: 0,
      maxDataPoints: 20,
      animation: false,
    });
    perfChart.setData(this.demoData.performance.slice(-20));
    this.charts.set("dashboard-performance", perfChart);

    // Mini distribution chart
    const distChart = new AnomalyDistributionChart("#dashboard-distribution", {
      title: "",
      chartType: "pie",
      animation: false,
    });
    distChart.setData(this.demoData.anomalies.slice(0, 50));
    this.charts.set("dashboard-distribution", distChart);

    // Mini timeline chart
    const timelineChart = new DetectionTimelineChart("#dashboard-timeline", {
      title: "",
      timeRange: "6h",
      animation: false,
    });
    timelineChart.setData(this.demoData.timeline.slice(-50));
    this.charts.set("dashboard-timeline", timelineChart);

    // Setup dashboard controls
    this.setupDashboardControls();
  }

  setupDashboardControls() {
    document
      .getElementById("dashboard-refresh")
      ?.addEventListener("click", () => {
        this.refreshDashboard();
      });

    document
      .getElementById("dashboard-export")
      ?.addEventListener("click", () => {
        this.exportDashboardData();
      });

    document
      .getElementById("dashboard-realtime")
      ?.addEventListener("change", (e) => {
        if (e.target.checked) {
          this.startDashboardRealTime();
        } else {
          this.stopDashboardRealTime();
        }
      });
  }

  updateDashboardMetrics() {
    const totalAnomalies = this.demoData.anomalies.length;
    const avgConfidence =
      this.demoData.anomalies.reduce((sum, d) => sum + d.confidence, 0) /
      totalAnomalies;
    const criticalAlerts = this.demoData.anomalies.filter(
      (d) => d.confidence > 0.95,
    ).length;
    const systemHealth =
      criticalAlerts > 5 ? "Critical" : criticalAlerts > 2 ? "Warning" : "Good";

    document.getElementById("dashboard-total-anomalies").textContent =
      totalAnomalies;
    document.getElementById("dashboard-avg-confidence").textContent =
      (avgConfidence * 100).toFixed(1) + "%";
    document.getElementById("dashboard-critical-alerts").textContent =
      criticalAlerts;
    document.getElementById("dashboard-system-health").textContent =
      systemHealth;
  }

  refreshDashboard() {
    this.demoData = this.generateDemoData();

    // Update all dashboard charts
    const perfChart = this.charts.get("dashboard-performance");
    if (perfChart) {
      perfChart.setData(this.demoData.performance.slice(-20));
    }

    const distChart = this.charts.get("dashboard-distribution");
    if (distChart) {
      distChart.setData(this.demoData.anomalies.slice(0, 50));
    }

    const timelineChart = this.charts.get("dashboard-timeline");
    if (timelineChart) {
      timelineChart.setData(this.demoData.timeline.slice(-50));
    }

    this.updateDashboardMetrics();
    this.announceToScreenReader("Dashboard refreshed with new data");
  }

  startDashboardRealTime() {
    if (this.dashboardTimer) return;

    this.dashboardTimer = setInterval(() => {
      // Add new performance data
      const now = new Date();
      const newPerfPoint = {
        timestamp: now.toISOString(),
        cpu: Math.random() * 80 + 10,
        memory: Math.random() * 70 + 20,
        network: Math.random() * 50 + 5,
      };

      this.demoData.performance.push(newPerfPoint);
      this.demoData.performance = this.demoData.performance.slice(-100);

      // Update performance chart
      const perfChart = this.charts.get("dashboard-performance");
      if (perfChart) {
        perfChart.setData(this.demoData.performance.slice(-20));
      }

      // Occasionally add new anomaly
      if (Math.random() < 0.1) {
        const newAnomaly = {
          id: Date.now(),
          type: [
            "Statistical Outlier",
            "Temporal Anomaly",
            "Pattern Deviation",
          ][Math.floor(Math.random() * 3)],
          confidence: Math.random(),
          timestamp: now.toISOString(),
        };

        this.demoData.anomalies.push(newAnomaly);
        this.demoData.timeline.push(newAnomaly);

        this.updateDashboardMetrics();
      }
    }, 3000);
  }

  stopDashboardRealTime() {
    if (this.dashboardTimer) {
      clearInterval(this.dashboardTimer);
      this.dashboardTimer = null;
    }
  }

  exportDashboardData() {
    const exportData = {
      timestamp: new Date().toISOString(),
      performance: this.demoData.performance,
      anomalies: this.demoData.anomalies,
      timeline: this.demoData.timeline,
      metrics: {
        totalAnomalies: this.demoData.anomalies.length,
        avgConfidence:
          this.demoData.anomalies.reduce((sum, d) => sum + d.confidence, 0) /
          this.demoData.anomalies.length,
        criticalAlerts: this.demoData.anomalies.filter(
          (d) => d.confidence > 0.95,
        ).length,
      },
    };

    const blob = new Blob([JSON.stringify(exportData, null, 2)], {
      type: "application/json",
    });

    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `echarts-dashboard-export-${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    this.announceToScreenReader("Dashboard data exported successfully");
  }

  setupEventListeners() {
    // Real-time toggle
    document
      .getElementById("echarts-realtime-toggle")
      ?.addEventListener("click", (e) => {
        this.toggleRealTime();
        e.target.textContent = this.isRealTimeEnabled
          ? "Stop Real-Time"
          : "Start Real-Time";
      });

    // Theme toggle
    document
      .getElementById("echarts-theme-toggle")
      ?.addEventListener("click", (e) => {
        const newTheme =
          echartsManager.currentTheme === "light" ? "dark" : "light";
        this.switchTheme(newTheme);
        e.target.textContent = `Switch to ${newTheme === "light" ? "Dark" : "Light"}`;
      });

    // Refresh data
    document
      .getElementById("echarts-refresh-data")
      ?.addEventListener("click", () => {
        this.refreshAllData();
      });

    // Export charts
    document
      .getElementById("echarts-export-charts")
      ?.addEventListener("click", () => {
        this.exportCharts();
      });
  }

  toggleRealTime() {
    if (this.isRealTimeEnabled) {
      clearInterval(this.realTimeInterval);
      this.isRealTimeEnabled = false;
    } else {
      const interval = parseInt(
        document.getElementById("echarts-update-interval")?.value || "2000",
      );
      this.realTimeInterval = setInterval(() => {
        this.updateRealTimeData();
      }, interval);
      this.isRealTimeEnabled = true;
    }
  }

  updateRealTimeData() {
    // Update performance chart
    const perfChart = this.charts.get("performance");
    if (perfChart) {
      perfChart.addRandomDataPoint();
    }

    // Occasionally add new anomaly
    if (Math.random() < 0.05) {
      const now = new Date();
      const newAnomaly = {
        id: Date.now(),
        type: ["Statistical Outlier", "Temporal Anomaly", "Pattern Deviation"][
          Math.floor(Math.random() * 3)
        ],
        confidence: Math.random(),
        timestamp: now.toISOString(),
      };

      this.demoData.anomalies.push(newAnomaly);
      this.demoData.timeline.push(newAnomaly);

      // Update distribution chart
      const distChart = this.charts.get("distribution");
      if (distChart) {
        distChart.setData(this.demoData.anomalies);
        this.updateDistributionStats();
      }

      // Update timeline chart
      const timelineChart = this.charts.get("timeline");
      if (timelineChart) {
        timelineChart.setData(this.demoData.timeline);
        this.updateTimelineSummary();
      }

      if (newAnomaly.confidence > 0.9) {
        this.announceToScreenReader(
          `High confidence anomaly detected: ${newAnomaly.type}, confidence ${(newAnomaly.confidence * 100).toFixed(1)}%`,
        );
      }
    }
  }

  switchTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);

    // Emit theme change event
    document.dispatchEvent(
      new CustomEvent("theme-changed", {
        detail: { theme },
      }),
    );
  }

  refreshAllData() {
    this.demoData = this.generateDemoData();

    this.charts.forEach((chart, key) => {
      if (key.includes("performance")) {
        chart.setData(this.demoData.performance);
      } else if (key.includes("distribution")) {
        chart.setData(this.demoData.anomalies);
        this.updateDistributionStats();
      } else if (key.includes("timeline")) {
        chart.setData(this.demoData.timeline);
        this.updateTimelineSummary();
      }
    });

    this.updateDashboardMetrics();
    this.announceToScreenReader("All chart data refreshed");
  }

  exportCharts() {
    const exportData = {
      timestamp: new Date().toISOString(),
      chartTypes: ["performance", "distribution", "timeline"],
      data: this.demoData,
      chartConfigurations: {},
    };

    this.charts.forEach((chart, key) => {
      exportData.chartConfigurations[key] = {
        type: chart.constructor.name,
        options: chart.options,
      };
    });

    const blob = new Blob([JSON.stringify(exportData, null, 2)], {
      type: "application/json",
    });

    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `echarts-demo-export-${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    this.announceToScreenReader("ECharts data exported successfully");
  }

  announceToScreenReader(message) {
    const announcer = document.getElementById("echarts-announcer");
    if (announcer) {
      announcer.textContent = message;
    }
  }

  destroy() {
    if (this.realTimeInterval) {
      clearInterval(this.realTimeInterval);
    }

    if (this.dashboardTimer) {
      clearInterval(this.dashboardTimer);
    }

    this.charts.forEach((chart) => {
      chart.dispose();
    });

    this.charts.clear();
  }
}

// Auto-initialize when DOM is ready
document.addEventListener("DOMContentLoaded", () => {
  if (document.querySelector(".echarts-demo")) {
    window.echartsDemo = new EChartsDemo();
  }
});

// Export for module usage
if (typeof module !== "undefined" && module.exports) {
  module.exports = EChartsDemo;
} else {
  window.EChartsDemo = EChartsDemo;
}
\n\n// mobile-ui.js\n/**
 * Mobile-Optimized UI Components
 * Touch-friendly interfaces with gesture support and responsive dashboard layouts
 * Provides native-like mobile experience for anomaly detection platform
 */

/**
 * Touch Gesture Recognition System
 * Handles swipe, pinch, pan, and tap gestures for mobile interactions
 */
class TouchGestureManager {
  constructor(element, options = {}) {
    this.element = element;
    this.options = {
      swipeThreshold: 50, // Minimum distance for swipe
      tapTimeout: 300, // Maximum time for tap
      doubleTapTimeout: 300, // Time between taps for double tap
      pinchThreshold: 10, // Minimum distance change for pinch
      enablePinch: true,
      enableSwipe: true,
      enableTap: true,
      enablePan: true,
      preventDefault: true,
      ...options,
    };

    // Touch state tracking
    this.touches = new Map();
    this.lastTap = null;
    this.isGesturing = false;
    this.gestureStartDistance = 0;
    this.gestureStartScale = 1;

    // Event listeners
    this.listeners = new Map();

    this.init();
  }

  init() {
    this.bindEvents();
  }

  bindEvents() {
    // Touch events
    this.element.addEventListener(
      "touchstart",
      this.handleTouchStart.bind(this),
      { passive: false },
    );
    this.element.addEventListener(
      "touchmove",
      this.handleTouchMove.bind(this),
      { passive: false },
    );
    this.element.addEventListener("touchend", this.handleTouchEnd.bind(this), {
      passive: false,
    });
    this.element.addEventListener(
      "touchcancel",
      this.handleTouchCancel.bind(this),
      { passive: false },
    );

    // Mouse events for desktop testing
    this.element.addEventListener("mousedown", this.handleMouseDown.bind(this));
    this.element.addEventListener("mousemove", this.handleMouseMove.bind(this));
    this.element.addEventListener("mouseup", this.handleMouseUp.bind(this));

    // Prevent default context menu on long press
    this.element.addEventListener("contextmenu", (e) => {
      if (this.options.preventDefault) {
        e.preventDefault();
      }
    });
  }

  handleTouchStart(event) {
    if (this.options.preventDefault) {
      event.preventDefault();
    }

    const touches = Array.from(event.changedTouches);
    touches.forEach((touch) => {
      this.touches.set(touch.identifier, {
        id: touch.identifier,
        startX: touch.clientX,
        startY: touch.clientY,
        currentX: touch.clientX,
        currentY: touch.clientY,
        startTime: Date.now(),
        element: touch.target,
      });
    });

    if (event.touches.length === 2 && this.options.enablePinch) {
      this.startPinchGesture(event.touches);
    }

    this.emit("touchstart", {
      touches: Array.from(this.touches.values()),
      originalEvent: event,
    });
  }

  handleTouchMove(event) {
    if (this.options.preventDefault) {
      event.preventDefault();
    }

    const touches = Array.from(event.changedTouches);
    touches.forEach((touch) => {
      const touchData = this.touches.get(touch.identifier);
      if (touchData) {
        touchData.currentX = touch.clientX;
        touchData.currentY = touch.clientY;
      }
    });

    if (event.touches.length === 2 && this.options.enablePinch) {
      this.handlePinchGesture(event.touches);
    } else if (event.touches.length === 1 && this.options.enablePan) {
      this.handlePanGesture(Array.from(this.touches.values())[0]);
    }

    this.emit("touchmove", {
      touches: Array.from(this.touches.values()),
      originalEvent: event,
    });
  }

  handleTouchEnd(event) {
    const touches = Array.from(event.changedTouches);

    touches.forEach((touch) => {
      const touchData = this.touches.get(touch.identifier);
      if (touchData) {
        this.processTouchEnd(touchData);
        this.touches.delete(touch.identifier);
      }
    });

    if (event.touches.length === 0) {
      this.isGesturing = false;
    }

    this.emit("touchend", {
      touches: Array.from(this.touches.values()),
      originalEvent: event,
    });
  }

  handleTouchCancel(event) {
    event.changedTouches.forEach((touch) => {
      this.touches.delete(touch.identifier);
    });
    this.isGesturing = false;
  }

  processTouchEnd(touchData) {
    const duration = Date.now() - touchData.startTime;
    const deltaX = touchData.currentX - touchData.startX;
    const deltaY = touchData.currentY - touchData.startY;
    const distance = Math.sqrt(deltaX * deltaX + deltaY * deltaY);

    // Check for tap
    if (
      this.options.enableTap &&
      duration < this.options.tapTimeout &&
      distance < 10
    ) {
      this.handleTap(touchData);
    }

    // Check for swipe
    if (this.options.enableSwipe && distance > this.options.swipeThreshold) {
      this.handleSwipe(touchData, deltaX, deltaY, distance);
    }
  }

  handleTap(touchData) {
    const now = Date.now();
    const tapEvent = {
      x: touchData.currentX,
      y: touchData.currentY,
      element: touchData.element,
      timestamp: now,
    };

    // Check for double tap
    if (
      this.lastTap &&
      now - this.lastTap.timestamp < this.options.doubleTapTimeout &&
      Math.abs(this.lastTap.x - tapEvent.x) < 25 &&
      Math.abs(this.lastTap.y - tapEvent.y) < 25
    ) {
      this.emit("doubletap", tapEvent);
      this.lastTap = null;
    } else {
      this.emit("tap", tapEvent);
      this.lastTap = tapEvent;

      // Clear last tap after timeout
      setTimeout(() => {
        if (this.lastTap === tapEvent) {
          this.lastTap = null;
        }
      }, this.options.doubleTapTimeout);
    }
  }

  handleSwipe(touchData, deltaX, deltaY, distance) {
    const direction = this.getSwipeDirection(deltaX, deltaY);

    this.emit("swipe", {
      direction,
      deltaX,
      deltaY,
      distance,
      velocity: distance / (Date.now() - touchData.startTime),
      startX: touchData.startX,
      startY: touchData.startY,
      endX: touchData.currentX,
      endY: touchData.currentY,
    });
  }

  getSwipeDirection(deltaX, deltaY) {
    const absDeltaX = Math.abs(deltaX);
    const absDeltaY = Math.abs(deltaY);

    if (absDeltaX > absDeltaY) {
      return deltaX > 0 ? "right" : "left";
    } else {
      return deltaY > 0 ? "down" : "up";
    }
  }

  startPinchGesture(touches) {
    const touch1 = touches[0];
    const touch2 = touches[1];

    this.gestureStartDistance = this.calculateDistance(
      touch1.clientX,
      touch1.clientY,
      touch2.clientX,
      touch2.clientY,
    );
    this.isGesturing = true;
  }

  handlePinchGesture(touches) {
    if (!this.isGesturing) return;

    const touch1 = touches[0];
    const touch2 = touches[1];

    const currentDistance = this.calculateDistance(
      touch1.clientX,
      touch1.clientY,
      touch2.clientX,
      touch2.clientY,
    );

    const scale = currentDistance / this.gestureStartDistance;
    const centerX = (touch1.clientX + touch2.clientX) / 2;
    const centerY = (touch1.clientY + touch2.clientY) / 2;

    this.emit("pinch", {
      scale,
      centerX,
      centerY,
      distance: currentDistance,
      startDistance: this.gestureStartDistance,
    });
  }

  handlePanGesture(touchData) {
    const deltaX = touchData.currentX - touchData.startX;
    const deltaY = touchData.currentY - touchData.startY;

    this.emit("pan", {
      deltaX,
      deltaY,
      currentX: touchData.currentX,
      currentY: touchData.currentY,
      startX: touchData.startX,
      startY: touchData.startY,
    });
  }

  calculateDistance(x1, y1, x2, y2) {
    return Math.sqrt(Math.pow(x2 - x1, 2) + Math.pow(y2 - y1, 2));
  }

  // Mouse event handlers for desktop testing
  handleMouseDown(event) {
    this.touches.set("mouse", {
      id: "mouse",
      startX: event.clientX,
      startY: event.clientY,
      currentX: event.clientX,
      currentY: event.clientY,
      startTime: Date.now(),
      element: event.target,
    });
  }

  handleMouseMove(event) {
    const touchData = this.touches.get("mouse");
    if (touchData) {
      touchData.currentX = event.clientX;
      touchData.currentY = event.clientY;
      this.handlePanGesture(touchData);
    }
  }

  handleMouseUp(event) {
    const touchData = this.touches.get("mouse");
    if (touchData) {
      this.processTouchEnd(touchData);
      this.touches.delete("mouse");
    }
  }

  // Event system
  on(event, callback) {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, new Set());
    }
    this.listeners.get(event).add(callback);
    return () => this.off(event, callback);
  }

  off(event, callback) {
    if (this.listeners.has(event)) {
      this.listeners.get(event).delete(callback);
    }
  }

  emit(event, data) {
    if (this.listeners.has(event)) {
      this.listeners.get(event).forEach((callback) => {
        try {
          callback(data);
        } catch (error) {
          console.error("Touch gesture callback error:", error);
        }
      });
    }
  }

  destroy() {
    this.touches.clear();
    this.listeners.clear();
    this.lastTap = null;
  }
}

/**
 * Mobile Dashboard Layout Manager
 * Responsive layout system optimized for mobile screens
 */
class MobileDashboardManager {
  constructor(container, options = {}) {
    this.container = container;
    this.options = {
      enableSwipeNavigation: true,
      enablePullToRefresh: true,
      enableCollapsiblePanels: true,
      tabBarHeight: 60,
      headerHeight: 56,
      minPanelHeight: 200,
      maxColumns: { mobile: 1, tablet: 2, desktop: 3 },
      breakpoints: {
        mobile: 768,
        tablet: 1024,
        desktop: 1200,
      },
      ...options,
    };

    this.currentLayout = "mobile";
    this.widgets = new Map();
    this.panels = new Map();
    this.activeTab = 0;
    this.isRefreshing = false;

    // UI elements
    this.header = null;
    this.tabBar = null;
    this.contentArea = null;
    this.pullToRefreshIndicator = null;

    this.init();
  }

  init() {
    this.detectLayout();
    this.createMobileStructure();
    this.setupGestureHandling();
    this.setupResizeListener();
    this.setupPullToRefresh();
  }

  detectLayout() {
    const width = window.innerWidth;
    if (width <= this.options.breakpoints.mobile) {
      this.currentLayout = "mobile";
    } else if (width <= this.options.breakpoints.tablet) {
      this.currentLayout = "tablet";
    } else {
      this.currentLayout = "desktop";
    }
  }

  createMobileStructure() {
    this.container.className = `mobile-dashboard ${this.currentLayout}`;

    this.container.innerHTML = `
      <header class="mobile-header">
        <div class="header-content">
          <button class="menu-button" aria-label="Menu">
            <span class="hamburger"></span>
          </button>
          <h1 class="header-title">Pynomaly</h1>
          <div class="header-actions">
            <button class="refresh-button" aria-label="Refresh">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                <path d="M17.65 6.35A7.958 7.958 0 0012 4c-4.42 0-7.99 3.58-7.99 8s3.57 8 7.99 8c3.73 0 6.84-2.55 7.73-6h-2.08A5.99 5.99 0 0112 18c-3.31 0-6-2.69-6-6s2.69-6 6-6c1.66 0 3.14.69 4.22 1.78L13 11h7V4l-2.35 2.35z"/>
              </svg>
            </button>
            <button class="settings-button" aria-label="Settings">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                <path d="M19.14,12.94c0.04-0.3,0.06-0.61,0.06-0.94c0-0.32-0.02-0.64-0.07-0.94l2.03-1.58c0.18-0.14,0.23-0.41,0.12-0.61 l-1.92-3.32c-0.12-0.22-0.37-0.29-0.59-0.22l-2.39,0.96c-0.5-0.38-1.03-0.7-1.62-0.94L14.4,2.81c-0.04-0.24-0.24-0.41-0.48-0.41 h-3.84c-0.24,0-0.43,0.17-0.47,0.41L9.25,5.35C8.66,5.59,8.12,5.92,7.63,6.29L5.24,5.33c-0.22-0.08-0.47,0-0.59,0.22L2.74,8.87 C2.62,9.08,2.66,9.34,2.86,9.48l2.03,1.58C4.84,11.36,4.8,11.69,4.8,12s0.02,0.64,0.07,0.94l-2.03,1.58 c-0.18,0.14-0.23,0.41-0.12,0.61l1.92,3.32c0.12,0.22,0.37,0.29,0.59,0.22l2.39-0.96c0.5,0.38,1.03,0.7,1.62,0.94l0.36,2.54 c0.05,0.24,0.24,0.41,0.48,0.41h3.84c0.24,0,0.44-0.17,0.47-0.41l0.36-2.54c0.59-0.24,1.13-0.56,1.62-0.94l2.39,0.96 c0.22,0.08,0.47,0,0.59-0.22l1.92-3.32c0.12-0.22,0.07-0.47-0.12-0.61L19.14,12.94z M12,15.6c-1.98,0-3.6-1.62-3.6-3.6 s1.62-3.6,3.6-3.6s3.6,1.62,3.6,3.6S13.98,15.6,12,15.6z"/>
              </svg>
            </button>
          </div>
        </div>
      </header>

      <div class="pull-to-refresh-indicator">
        <div class="refresh-spinner"></div>
        <span class="refresh-text">Pull to refresh</span>
      </div>

      <main class="content-area">
        <div class="dashboard-tabs" role="tablist">
          <!-- Tabs will be dynamically generated -->
        </div>

        <div class="tab-panels">
          <!-- Panel content will be dynamically generated -->
        </div>
      </main>

      <nav class="tab-bar" role="tablist">
        <button class="tab-button active" data-tab="0" role="tab" aria-selected="true">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor">
            <path d="M3 13h8V3H3v10zm0 8h8v-6H3v6zm10 0h8V11h-8v10zm0-18v6h8V3h-8z"/>
          </svg>
          <span>Dashboard</span>
        </button>
        <button class="tab-button" data-tab="1" role="tab" aria-selected="false">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor">
            <path d="M16 6l2.29 2.29-4.88 4.88-4-4L2 16.59 3.41 18l6-6 4 4 6.3-6.29L22 12V6z"/>
          </svg>
          <span>Analytics</span>
        </button>
        <button class="tab-button" data-tab="2" role="tab" aria-selected="false">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor">
            <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/>
          </svg>
          <span>Alerts</span>
        </button>
        <button class="tab-button" data-tab="3" role="tab" aria-selected="false">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor">
            <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/>
          </svg>
          <span>Models</span>
        </button>
      </nav>
    `;

    // Cache DOM elements
    this.header = this.container.querySelector(".mobile-header");
    this.tabBar = this.container.querySelector(".tab-bar");
    this.contentArea = this.container.querySelector(".content-area");
    this.pullToRefreshIndicator = this.container.querySelector(
      ".pull-to-refresh-indicator",
    );

    this.setupTabNavigation();
  }

  setupTabNavigation() {
    const tabButtons = this.tabBar.querySelectorAll(".tab-button");

    tabButtons.forEach((button, index) => {
      button.addEventListener("click", () => {
        this.switchTab(index);
      });
    });

    // Initialize first tab
    this.switchTab(0);
  }

  switchTab(tabIndex) {
    const tabButtons = this.tabBar.querySelectorAll(".tab-button");
    const panels = this.contentArea.querySelectorAll(".tab-panel");

    // Update tab buttons
    tabButtons.forEach((button, index) => {
      const isActive = index === tabIndex;
      button.classList.toggle("active", isActive);
      button.setAttribute("aria-selected", isActive);
    });

    // Update panels
    panels.forEach((panel, index) => {
      panel.classList.toggle("active", index === tabIndex);
    });

    this.activeTab = tabIndex;
    this.emit("tab-changed", { activeTab: tabIndex });
  }

  setupGestureHandling() {
    const gestureManager = new TouchGestureManager(this.contentArea, {
      enableSwipe: this.options.enableSwipeNavigation,
      enablePinch: false, // Disable pinch on main container
    });

    // Swipe navigation between tabs
    if (this.options.enableSwipeNavigation) {
      gestureManager.on("swipe", (gesture) => {
        if (Math.abs(gesture.deltaY) < 50) {
          // Horizontal swipes only
          if (gesture.direction === "left" && this.activeTab < 3) {
            this.switchTab(this.activeTab + 1);
          } else if (gesture.direction === "right" && this.activeTab > 0) {
            this.switchTab(this.activeTab - 1);
          }
        }
      });
    }
  }

  setupPullToRefresh() {
    if (!this.options.enablePullToRefresh) return;

    let pullStartY = 0;
    let pullCurrentY = 0;
    let pullDeltaY = 0;
    let isPulling = false;

    const gestureManager = new TouchGestureManager(this.contentArea);

    gestureManager.on("touchstart", (event) => {
      if (this.contentArea.scrollTop === 0) {
        pullStartY = event.touches[0].currentY;
        isPulling = true;
      }
    });

    gestureManager.on("touchmove", (event) => {
      if (!isPulling) return;

      pullCurrentY = event.touches[0].currentY;
      pullDeltaY = pullCurrentY - pullStartY;

      if (pullDeltaY > 0 && this.contentArea.scrollTop === 0) {
        const pullDistance = Math.min(pullDeltaY, 100);
        const opacity = Math.min(pullDistance / 60, 1);

        this.pullToRefreshIndicator.style.transform = `translateY(${pullDistance}px)`;
        this.pullToRefreshIndicator.style.opacity = opacity;

        if (pullDistance > 60) {
          this.pullToRefreshIndicator.classList.add("ready");
          this.pullToRefreshIndicator.querySelector(
            ".refresh-text",
          ).textContent = "Release to refresh";
        } else {
          this.pullToRefreshIndicator.classList.remove("ready");
          this.pullToRefreshIndicator.querySelector(
            ".refresh-text",
          ).textContent = "Pull to refresh";
        }
      }
    });

    gestureManager.on("touchend", () => {
      if (!isPulling) return;

      isPulling = false;

      if (pullDeltaY > 60 && !this.isRefreshing) {
        this.triggerRefresh();
      } else {
        this.resetPullToRefresh();
      }
    });
  }

  triggerRefresh() {
    this.isRefreshing = true;
    this.pullToRefreshIndicator.classList.add("refreshing");
    this.pullToRefreshIndicator.querySelector(".refresh-text").textContent =
      "Refreshing...";

    this.emit("refresh-requested");

    // Auto-reset after 3 seconds if not manually reset
    setTimeout(() => {
      if (this.isRefreshing) {
        this.resetPullToRefresh();
      }
    }, 3000);
  }

  resetPullToRefresh() {
    this.isRefreshing = false;
    this.pullToRefreshIndicator.classList.remove("ready", "refreshing");
    this.pullToRefreshIndicator.style.transform = "translateY(-100%)";
    this.pullToRefreshIndicator.style.opacity = "0";
    this.pullToRefreshIndicator.querySelector(".refresh-text").textContent =
      "Pull to refresh";
  }

  setupResizeListener() {
    let resizeTimeout;
    window.addEventListener("resize", () => {
      clearTimeout(resizeTimeout);
      resizeTimeout = setTimeout(() => {
        const oldLayout = this.currentLayout;
        this.detectLayout();

        if (oldLayout !== this.currentLayout) {
          this.container.className = `mobile-dashboard ${this.currentLayout}`;
          this.updateLayoutForScreen();
        }
      }, 250);
    });
  }

  updateLayoutForScreen() {
    const maxCols = this.options.maxColumns[this.currentLayout];

    // Update widget layouts based on screen size
    this.panels.forEach((panel) => {
      panel.updateLayout(maxCols);
    });

    this.emit("layout-changed", { layout: this.currentLayout });
  }

  /**
   * Widget and Panel Management
   */
  createPanel(id, title, content, tabIndex = 0) {
    const panelElement = document.createElement("div");
    panelElement.className = `tab-panel ${tabIndex === this.activeTab ? "active" : ""}`;
    panelElement.setAttribute("role", "tabpanel");
    panelElement.innerHTML = `
      <div class="panel-header">
        <h2 class="panel-title">${title}</h2>
        <div class="panel-actions">
          <button class="panel-collapse-btn" aria-label="Collapse panel">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
              <path d="M7.41 8.84L12 13.42l4.59-4.58L18 10.25l-6 6-6-6z"/>
            </svg>
          </button>
        </div>
      </div>
      <div class="panel-content">
        ${content}
      </div>
    `;

    const panel = {
      id,
      element: panelElement,
      title,
      tabIndex,
      isCollapsed: false,
      updateLayout: (maxCols) => {
        panelElement.style.gridColumn = `span ${Math.min(1, maxCols)}`;
      },
    };

    // Add collapse functionality
    const collapseBtn = panelElement.querySelector(".panel-collapse-btn");
    const panelContent = panelElement.querySelector(".panel-content");

    collapseBtn.addEventListener("click", () => {
      panel.isCollapsed = !panel.isCollapsed;
      panelElement.classList.toggle("collapsed", panel.isCollapsed);

      if (panel.isCollapsed) {
        panelContent.style.height = "0";
        collapseBtn.style.transform = "rotate(-90deg)";
      } else {
        panelContent.style.height = "auto";
        collapseBtn.style.transform = "rotate(0deg)";
      }
    });

    this.panels.set(id, panel);

    // Add to appropriate tab
    const tabPanels = this.contentArea.querySelector(".tab-panels");
    tabPanels.appendChild(panelElement);

    return panel;
  }

  createWidget(id, type, config, panelId) {
    const widget = {
      id,
      type,
      config,
      panelId,
      element: null,
      touchOptimized: true,
    };

    // Create widget element based on type
    switch (type) {
      case "chart":
        widget.element = this.createChartWidget(config);
        break;
      case "metric":
        widget.element = this.createMetricWidget(config);
        break;
      case "list":
        widget.element = this.createListWidget(config);
        break;
      case "form":
        widget.element = this.createFormWidget(config);
        break;
      default:
        widget.element = this.createDefaultWidget(config);
    }

    // Add touch optimizations
    this.optimizeWidgetForTouch(widget);

    this.widgets.set(id, widget);

    // Add to panel
    const panel = this.panels.get(panelId);
    if (panel) {
      panel.element.querySelector(".panel-content").appendChild(widget.element);
    }

    return widget;
  }

  createChartWidget(config) {
    const element = document.createElement("div");
    element.className = "widget chart-widget touch-optimized";
    element.innerHTML = `
      <div class="widget-header">
        <h3 class="widget-title">${config.title || "Chart"}</h3>
        <div class="widget-controls">
          <button class="zoom-out-btn" aria-label="Zoom out">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
              <path d="M15.5 14h-.79l-.28-.27C15.41 12.59 16 11.11 16 9.5 16 5.91 13.09 3 9.5 3S3 5.91 3 9.5 5.91 16 9.5 16c1.61 0 3.09-.59 4.23-1.57l.27.28v.79l5 4.99L20.49 19l-4.99-5zm-6 0C7.01 14 5 11.99 5 9.5S7.01 5 9.5 5 14 7.01 14 9.5 11.99 14 9.5 14z"/>
              <path d="M7 9h5v1H7z"/>
            </svg>
          </button>
          <button class="fullscreen-btn" aria-label="Fullscreen">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
              <path d="M7 14H5v5h5v-2H7v-3zm-2-4h2V7h3V5H5v5zm12 7h-3v2h5v-5h-2v3zM14 5v2h3v3h2V5h-5z"/>
            </svg>
          </button>
        </div>
      </div>
      <div class="widget-content chart-container" style="height: ${config.height || "250px"};">
        <!-- Chart will be rendered here -->
      </div>
    `;

    return element;
  }

  createMetricWidget(config) {
    const element = document.createElement("div");
    element.className = "widget metric-widget touch-optimized";
    element.innerHTML = `
      <div class="metric-display">
        <div class="metric-value ${config.trend || ""}">${config.value || "0"}</div>
        <div class="metric-label">${config.label || "Metric"}</div>
        <div class="metric-change">${config.change || "+0%"}</div>
      </div>
    `;

    return element;
  }

  createListWidget(config) {
    const element = document.createElement("div");
    element.className = "widget list-widget touch-optimized";
    element.innerHTML = `
      <div class="widget-header">
        <h3 class="widget-title">${config.title || "List"}</h3>
        <button class="refresh-widget-btn" aria-label="Refresh">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
            <path d="M17.65 6.35A7.958 7.958 0 0012 4c-4.42 0-7.99 3.58-7.99 8s3.57 8 7.99 8c3.73 0 6.84-2.55 7.73-6h-2.08A5.99 5.99 0 0112 18c-3.31 0-6-2.69-6-6s2.69-6 6-6c1.66 0 3.14.69 4.22 1.78L13 11h7V4l-2.35 2.35z"/>
          </svg>
        </button>
      </div>
      <div class="widget-content">
        <div class="list-container">
          <!-- List items will be dynamically added -->
        </div>
      </div>
    `;

    return element;
  }

  createFormWidget(config) {
    const element = document.createElement("div");
    element.className = "widget form-widget touch-optimized";
    element.innerHTML = `
      <div class="widget-header">
        <h3 class="widget-title">${config.title || "Form"}</h3>
      </div>
      <div class="widget-content">
        <form class="mobile-form">
          <!-- Form fields will be dynamically added -->
        </form>
      </div>
    `;

    return element;
  }

  createDefaultWidget(config) {
    const element = document.createElement("div");
    element.className = "widget default-widget touch-optimized";
    element.innerHTML = `
      <div class="widget-content">
        ${config.content || "Widget content"}
      </div>
    `;

    return element;
  }

  optimizeWidgetForTouch(widget) {
    const element = widget.element;

    // Add touch-friendly tap targets
    const buttons = element.querySelectorAll("button");
    buttons.forEach((button) => {
      button.style.minHeight = "44px";
      button.style.minWidth = "44px";
      button.classList.add("touch-target");
    });

    // Add gesture support for charts
    if (widget.type === "chart") {
      const chartContainer = element.querySelector(".chart-container");
      const gestureManager = new TouchGestureManager(chartContainer, {
        enablePinch: true,
        enablePan: true,
      });

      gestureManager.on("pinch", (gesture) => {
        // Handle chart zoom
        this.emit("chart-zoom", {
          widgetId: widget.id,
          scale: gesture.scale,
          centerX: gesture.centerX,
          centerY: gesture.centerY,
        });
      });

      gestureManager.on("pan", (gesture) => {
        // Handle chart pan
        this.emit("chart-pan", {
          widgetId: widget.id,
          deltaX: gesture.deltaX,
          deltaY: gesture.deltaY,
        });
      });

      gestureManager.on("doubletap", () => {
        // Reset chart zoom
        this.emit("chart-reset", { widgetId: widget.id });
      });
    }

    // Add touch feedback
    element.addEventListener("touchstart", () => {
      element.classList.add("touch-active");
    });

    element.addEventListener("touchend", () => {
      setTimeout(() => {
        element.classList.remove("touch-active");
      }, 150);
    });
  }

  /**
   * Mobile-specific features
   */
  enableHapticFeedback() {
    if ("vibrate" in navigator) {
      return {
        light: () => navigator.vibrate(10),
        medium: () => navigator.vibrate(20),
        heavy: () => navigator.vibrate(50),
        success: () => navigator.vibrate([50, 50, 50]),
        error: () => navigator.vibrate([100, 50, 100]),
      };
    }
    return {
      light: () => {},
      medium: () => {},
      heavy: () => {},
      success: () => {},
      error: () => {},
    };
  }

  showToast(message, type = "info", duration = 3000) {
    const toast = document.createElement("div");
    toast.className = `mobile-toast ${type}`;
    toast.innerHTML = `
      <div class="toast-content">
        <span class="toast-message">${message}</span>
        <button class="toast-close" aria-label="Close">×</button>
      </div>
    `;

    document.body.appendChild(toast);

    // Add touch handling
    const gestureManager = new TouchGestureManager(toast);
    gestureManager.on("swipe", (gesture) => {
      if (gesture.direction === "up" || gesture.direction === "right") {
        this.dismissToast(toast);
      }
    });

    toast.querySelector(".toast-close").addEventListener("click", () => {
      this.dismissToast(toast);
    });

    // Auto-dismiss
    setTimeout(() => {
      this.dismissToast(toast);
    }, duration);

    // Show animation
    requestAnimationFrame(() => {
      toast.classList.add("show");
    });
  }

  dismissToast(toast) {
    toast.classList.add("dismiss");
    setTimeout(() => {
      if (toast.parentNode) {
        toast.parentNode.removeChild(toast);
      }
    }, 300);
  }

  // Event system
  on(event, callback) {
    if (!this.listeners) {
      this.listeners = new Map();
    }
    if (!this.listeners.has(event)) {
      this.listeners.set(event, new Set());
    }
    this.listeners.get(event).add(callback);
    return () => this.off(event, callback);
  }

  off(event, callback) {
    if (this.listeners && this.listeners.has(event)) {
      this.listeners.get(event).delete(callback);
    }
  }

  emit(event, data) {
    if (this.listeners && this.listeners.has(event)) {
      this.listeners.get(event).forEach((callback) => {
        try {
          callback(data);
        } catch (error) {
          console.error("Mobile dashboard event error:", error);
        }
      });
    }
  }

  /**
   * Cleanup
   */
  destroy() {
    if (this.listeners) {
      this.listeners.clear();
    }
    this.widgets.clear();
    this.panels.clear();
  }
}

// Export classes
if (typeof module !== "undefined" && module.exports) {
  module.exports = {
    TouchGestureManager,
    MobileDashboardManager,
  };
} else {
  // Browser environment
  window.TouchGestureManager = TouchGestureManager;
  window.MobileDashboardManager = MobileDashboardManager;
}
\n\n// multi-step-form.js\n/**
 * Advanced Multi-Step Form Component
 *
 * Comprehensive form system with validation, file upload, progress tracking,
 * and dynamic field generation for anomaly detection workflows
 */

export class MultiStepForm {
  constructor(container, options = {}) {
    this.container =
      typeof container === "string"
        ? document.querySelector(container)
        : container;
    this.options = {
      showProgress: true,
      showStepNumbers: true,
      allowStepNavigation: true,
      validateOnStepChange: true,
      saveProgress: true,
      progressKey: "multi-step-form-progress",
      submitUrl: null,
      submitMethod: "POST",
      fileUploadUrl: "/api/upload",
      maxFileSize: 10 * 1024 * 1024, // 10MB
      allowedFileTypes: [".csv", ".json", ".xlsx", ".parquet"],
      ...options,
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
    this.container.classList.add("multi-step-form");
    this.container.innerHTML = "";

    // Create form structure
    this.form = document.createElement("form");
    this.form.className = "form-container";
    this.form.setAttribute("novalidate", "");

    // Progress indicator
    if (this.options.showProgress) {
      this.progressContainer = document.createElement("div");
      this.progressContainer.className = "form-progress";
      this.container.appendChild(this.progressContainer);
    }

    // Steps container
    this.stepsContainer = document.createElement("div");
    this.stepsContainer.className = "form-steps";
    this.form.appendChild(this.stepsContainer);

    // Navigation
    this.navigationContainer = document.createElement("div");
    this.navigationContainer.className = "form-navigation";
    this.form.appendChild(this.navigationContainer);

    this.container.appendChild(this.form);
  }

  addStep(stepConfig) {
    const step = {
      id: stepConfig.id || `step_${this.steps.length}`,
      title: stepConfig.title || `Step ${this.steps.length + 1}`,
      description: stepConfig.description || "",
      fields: stepConfig.fields || [],
      validation: stepConfig.validation || null,
      onEnter: stepConfig.onEnter || null,
      onExit: stepConfig.onExit || null,
      conditional: stepConfig.conditional || null,
      ...stepConfig,
    };

    this.steps.push(step);
    this.renderProgress();

    return step;
  }

  renderProgress() {
    if (!this.options.showProgress || !this.progressContainer) return;

    this.progressContainer.innerHTML = "";

    const progressBar = document.createElement("div");
    progressBar.className = "progress-bar";

    const progressFill = document.createElement("div");
    progressFill.className = "progress-fill";
    progressFill.style.width = `${((this.currentStep + 1) / this.steps.length) * 100}%`;

    progressBar.appendChild(progressFill);
    this.progressContainer.appendChild(progressBar);

    // Step indicators
    if (this.options.showStepNumbers) {
      const stepsIndicator = document.createElement("div");
      stepsIndicator.className = "steps-indicator";

      this.steps.forEach((step, index) => {
        const stepIndicator = document.createElement("div");
        stepIndicator.className = "step-indicator";
        stepIndicator.setAttribute("data-step", index);

        if (index < this.currentStep) {
          stepIndicator.classList.add("completed");
        } else if (index === this.currentStep) {
          stepIndicator.classList.add("active");
        }

        if (this.options.allowStepNavigation && index <= this.currentStep) {
          stepIndicator.classList.add("clickable");
          stepIndicator.addEventListener("click", () => this.goToStep(index));
        }

        const stepNumber = document.createElement("span");
        stepNumber.className = "step-number";
        stepNumber.textContent = index + 1;

        const stepTitle = document.createElement("span");
        stepTitle.className = "step-title";
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
    this.stepsContainer.innerHTML = "";

    // Create step container
    const stepElement = document.createElement("div");
    stepElement.className = "form-step active";
    stepElement.setAttribute("data-step-id", step.id);

    // Step header
    const stepHeader = document.createElement("div");
    stepHeader.className = "step-header";

    const stepTitle = document.createElement("h2");
    stepTitle.className = "step-title";
    stepTitle.textContent = step.title;
    stepHeader.appendChild(stepTitle);

    if (step.description) {
      const stepDescription = document.createElement("p");
      stepDescription.className = "step-description";
      stepDescription.textContent = step.description;
      stepHeader.appendChild(stepDescription);
    }

    stepElement.appendChild(stepHeader);

    // Step content
    const stepContent = document.createElement("div");
    stepContent.className = "step-content";

    // Render fields
    step.fields.forEach((fieldConfig) => {
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
      type: "text",
      name: fieldConfig.name || "",
      label: fieldConfig.label || "",
      placeholder: fieldConfig.placeholder || "",
      required: fieldConfig.required || false,
      validation: fieldConfig.validation || [],
      options: fieldConfig.options || [],
      defaultValue: fieldConfig.defaultValue,
      conditional: fieldConfig.conditional || null,
      ...fieldConfig,
    };

    // Check conditional display
    if (field.conditional && !this.evaluateCondition(field.conditional)) {
      const hiddenDiv = document.createElement("div");
      hiddenDiv.style.display = "none";
      hiddenDiv.setAttribute("data-field-name", field.name);
      return hiddenDiv;
    }

    // Get field component
    const fieldComponent =
      this.fieldComponents.get(field.type) || this.fieldComponents.get("text");
    return fieldComponent(field, this);
  }

  renderNavigation() {
    this.navigationContainer.innerHTML = "";

    const navContainer = document.createElement("div");
    navContainer.className = "nav-buttons";

    // Previous button
    if (this.currentStep > 0) {
      const prevButton = document.createElement("button");
      prevButton.type = "button";
      prevButton.className = "btn btn-secondary prev-btn";
      prevButton.textContent = "Previous";
      prevButton.addEventListener("click", () => this.previousStep());
      navContainer.appendChild(prevButton);
    }

    // Next/Submit button
    const nextButton = document.createElement("button");
    nextButton.type = "button";
    nextButton.className = "btn btn-primary next-btn";

    if (this.currentStep === this.steps.length - 1) {
      nextButton.textContent = this.isSubmitting ? "Submitting..." : "Submit";
      nextButton.disabled = this.isSubmitting;
      nextButton.addEventListener("click", () => this.submitForm());
    } else {
      nextButton.textContent = "Next";
      nextButton.addEventListener("click", () => this.nextStep());
    }

    navContainer.appendChild(nextButton);

    // Cancel button
    const cancelButton = document.createElement("button");
    cancelButton.type = "button";
    cancelButton.className = "btn btn-ghost cancel-btn";
    cancelButton.textContent = "Cancel";
    cancelButton.addEventListener("click", () => this.cancel());
    navContainer.appendChild(cancelButton);

    this.navigationContainer.appendChild(navContainer);
  }

  registerDefaultFieldTypes() {
    // Text input
    this.fieldComponents.set("text", (field, form) => {
      const container = document.createElement("div");
      container.className = "field-group";
      container.setAttribute("data-field-name", field.name);

      const label = document.createElement("label");
      label.className = "field-label";
      label.textContent = field.label;
      if (field.required) {
        label.innerHTML += ' <span class="required">*</span>';
      }

      const input = document.createElement("input");
      input.type = field.subtype || "text";
      input.name = field.name;
      input.className = "field-input";
      input.placeholder = field.placeholder;
      input.required = field.required;
      input.value = form.formData[field.name] || field.defaultValue || "";

      if (field.pattern) {
        input.pattern = field.pattern;
      }

      input.addEventListener("input", (e) => {
        form.updateFieldValue(field.name, e.target.value);
      });

      input.addEventListener("blur", () => {
        form.validateField(field.name);
      });

      container.appendChild(label);
      container.appendChild(input);

      if (field.help) {
        const help = document.createElement("div");
        help.className = "field-help";
        help.textContent = field.help;
        container.appendChild(help);
      }

      return container;
    });

    // Select dropdown
    this.fieldComponents.set("select", (field, form) => {
      const container = document.createElement("div");
      container.className = "field-group";
      container.setAttribute("data-field-name", field.name);

      const label = document.createElement("label");
      label.className = "field-label";
      label.textContent = field.label;
      if (field.required) {
        label.innerHTML += ' <span class="required">*</span>';
      }

      const select = document.createElement("select");
      select.name = field.name;
      select.className = "field-select";
      select.required = field.required;

      // Add default option
      if (field.placeholder) {
        const defaultOption = document.createElement("option");
        defaultOption.value = "";
        defaultOption.textContent = field.placeholder;
        defaultOption.disabled = true;
        defaultOption.selected = !form.formData[field.name];
        select.appendChild(defaultOption);
      }

      // Add options
      field.options.forEach((option) => {
        const optionElement = document.createElement("option");

        if (typeof option === "string") {
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

      select.addEventListener("change", (e) => {
        form.updateFieldValue(field.name, e.target.value);
      });

      container.appendChild(label);
      container.appendChild(select);

      return container;
    });

    // Textarea
    this.fieldComponents.set("textarea", (field, form) => {
      const container = document.createElement("div");
      container.className = "field-group";
      container.setAttribute("data-field-name", field.name);

      const label = document.createElement("label");
      label.className = "field-label";
      label.textContent = field.label;
      if (field.required) {
        label.innerHTML += ' <span class="required">*</span>';
      }

      const textarea = document.createElement("textarea");
      textarea.name = field.name;
      textarea.className = "field-textarea";
      textarea.placeholder = field.placeholder;
      textarea.required = field.required;
      textarea.rows = field.rows || 4;
      textarea.value = form.formData[field.name] || field.defaultValue || "";

      textarea.addEventListener("input", (e) => {
        form.updateFieldValue(field.name, e.target.value);
      });

      container.appendChild(label);
      container.appendChild(textarea);

      return container;
    });

    // Checkbox
    this.fieldComponents.set("checkbox", (field, form) => {
      const container = document.createElement("div");
      container.className = "field-group field-checkbox";
      container.setAttribute("data-field-name", field.name);

      const checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      checkbox.name = field.name;
      checkbox.className = "field-input-checkbox";
      checkbox.required = field.required;
      checkbox.checked =
        form.formData[field.name] || field.defaultValue || false;

      const label = document.createElement("label");
      label.className = "field-label-checkbox";
      label.appendChild(checkbox);

      const labelText = document.createElement("span");
      labelText.textContent = field.label;
      if (field.required) {
        labelText.innerHTML += ' <span class="required">*</span>';
      }
      label.appendChild(labelText);

      checkbox.addEventListener("change", (e) => {
        form.updateFieldValue(field.name, e.target.checked);
      });

      container.appendChild(label);

      return container;
    });

    // File upload
    this.fieldComponents.set("file", (field, form) => {
      const container = document.createElement("div");
      container.className = "field-group field-file";
      container.setAttribute("data-field-name", field.name);

      const label = document.createElement("label");
      label.className = "field-label";
      label.textContent = field.label;
      if (field.required) {
        label.innerHTML += ' <span class="required">*</span>';
      }

      // File input
      const fileInput = document.createElement("input");
      fileInput.type = "file";
      fileInput.name = field.name;
      fileInput.className = "field-file-input";
      fileInput.required = field.required;
      fileInput.style.display = "none";

      if (field.multiple) {
        fileInput.multiple = true;
      }

      if (field.accept) {
        fileInput.accept = field.accept;
      } else {
        fileInput.accept = form.options.allowedFileTypes.join(",");
      }

      // Custom file upload button
      const uploadButton = document.createElement("button");
      uploadButton.type = "button";
      uploadButton.className = "btn btn-outline file-upload-btn";
      uploadButton.innerHTML = `
                <svg class="upload-icon" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                    <polyline points="7,10 12,15 17,10"></polyline>
                    <line x1="12" y1="15" x2="12" y2="3"></line>
                </svg>
                Choose File${field.multiple ? "s" : ""}
            `;

      uploadButton.addEventListener("click", () => fileInput.click());

      // File preview area
      const previewArea = document.createElement("div");
      previewArea.className = "file-preview-area";

      fileInput.addEventListener("change", (e) => {
        form.handleFileUpload(field.name, e.target.files);
      });

      container.appendChild(label);
      container.appendChild(fileInput);
      container.appendChild(uploadButton);
      container.appendChild(previewArea);

      if (field.help) {
        const help = document.createElement("div");
        help.className = "field-help";
        help.textContent = field.help;
        container.appendChild(help);
      }

      return container;
    });

    // Range slider
    this.fieldComponents.set("range", (field, form) => {
      const container = document.createElement("div");
      container.className = "field-group field-range";
      container.setAttribute("data-field-name", field.name);

      const label = document.createElement("label");
      label.className = "field-label";
      label.textContent = field.label;
      if (field.required) {
        label.innerHTML += ' <span class="required">*</span>';
      }

      const rangeContainer = document.createElement("div");
      rangeContainer.className = "range-container";

      const range = document.createElement("input");
      range.type = "range";
      range.name = field.name;
      range.className = "field-range-input";
      range.min = field.min || 0;
      range.max = field.max || 100;
      range.step = field.step || 1;
      range.value =
        form.formData[field.name] || field.defaultValue || field.min || 0;

      const valueDisplay = document.createElement("span");
      valueDisplay.className = "range-value";
      valueDisplay.textContent = range.value;

      range.addEventListener("input", (e) => {
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
    this.validators.set("required", (value, field) => {
      if (field.required) {
        if (value === null || value === undefined || value === "") {
          return `${field.label} is required`;
        }
      }
      return null;
    });

    this.validators.set("email", (value, field) => {
      if (value && field.type === "email") {
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        if (!emailRegex.test(value)) {
          return "Please enter a valid email address";
        }
      }
      return null;
    });

    this.validators.set("minLength", (value, field) => {
      if (value && field.minLength && value.length < field.minLength) {
        return `${field.label} must be at least ${field.minLength} characters`;
      }
      return null;
    });

    this.validators.set("maxLength", (value, field) => {
      if (value && field.maxLength && value.length > field.maxLength) {
        return `${field.label} must be no more than ${field.maxLength} characters`;
      }
      return null;
    });

    this.validators.set("pattern", (value, field) => {
      if (value && field.pattern) {
        const regex = new RegExp(field.pattern);
        if (!regex.test(value)) {
          return field.patternMessage || `${field.label} format is invalid`;
        }
      }
      return null;
    });

    this.validators.set("fileSize", (value, field) => {
      if (field.type === "file" && value) {
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
    const field = step.fields.find((f) => f.name === fieldName);

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
      if (typeof field.validation === "function") {
        const result = field.validation(value, this.formData);
        if (result && typeof result === "string") {
          errors.push(result);
        }
      } else if (Array.isArray(field.validation)) {
        field.validation.forEach((validator) => {
          if (typeof validator === "function") {
            const result = validator(value, this.formData);
            if (result && typeof result === "string") {
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
    step.fields.forEach((field) => {
      if (!this.validateField(field.name)) {
        isValid = false;
      }
    });

    // Run step-level validation
    if (step.validation && typeof step.validation === "function") {
      const result = step.validation(this.formData);
      if (result !== true) {
        isValid = false;
        if (typeof result === "string") {
          this.showStepError(result);
        }
      }
    }

    return isValid;
  }

  setFieldError(fieldName, message) {
    this.validationErrors[fieldName] = message;

    const fieldContainer = this.container.querySelector(
      `[data-field-name="${fieldName}"]`,
    );
    if (fieldContainer) {
      fieldContainer.classList.add("field-error");

      // Remove existing error message
      const existingError = fieldContainer.querySelector(
        ".field-error-message",
      );
      if (existingError) {
        existingError.remove();
      }

      // Add error message
      const errorElement = document.createElement("div");
      errorElement.className = "field-error-message";
      errorElement.textContent = message;
      fieldContainer.appendChild(errorElement);
    }
  }

  clearFieldError(fieldName) {
    delete this.validationErrors[fieldName];

    const fieldContainer = this.container.querySelector(
      `[data-field-name="${fieldName}"]`,
    );
    if (fieldContainer) {
      fieldContainer.classList.remove("field-error");

      const errorElement = fieldContainer.querySelector(".field-error-message");
      if (errorElement) {
        errorElement.remove();
      }
    }
  }

  showStepError(message) {
    // Implementation for step-level errors
    const existingError = this.stepsContainer.querySelector(".step-error");
    if (existingError) {
      existingError.remove();
    }

    const errorElement = document.createElement("div");
    errorElement.className = "step-error alert alert-danger";
    errorElement.textContent = message;

    this.stepsContainer.insertBefore(
      errorElement,
      this.stepsContainer.firstChild,
    );
  }

  updateConditionalFields() {
    // Re-render current step to show/hide conditional fields
    this.renderCurrentStep();
  }

  evaluateCondition(condition) {
    if (typeof condition === "function") {
      return condition(this.formData);
    }

    if (typeof condition === "object") {
      const { field, operator, value } = condition;
      const fieldValue = this.formData[field];

      switch (operator) {
        case "equals":
          return fieldValue === value;
        case "not_equals":
          return fieldValue !== value;
        case "greater_than":
          return fieldValue > value;
        case "less_than":
          return fieldValue < value;
        case "contains":
          return fieldValue && fieldValue.includes(value);
        case "in":
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
        this.setFieldError(
          fieldName,
          `File "${file.name}" is too large. Maximum size is ${this.formatFileSize(this.options.maxFileSize)}`,
        );
        continue;
      }

      // Validate file type
      const fileExtension = "." + file.name.split(".").pop().toLowerCase();
      if (!this.options.allowedFileTypes.includes(fileExtension)) {
        this.setFieldError(
          fieldName,
          `File type "${fileExtension}" is not allowed`,
        );
        continue;
      }

      try {
        const uploadedFile = await this.uploadFile(file);
        uploadedFiles.push(uploadedFile);

        // Show file preview
        this.showFilePreview(fieldName, uploadedFile);
      } catch (error) {
        this.setFieldError(
          fieldName,
          `Failed to upload "${file.name}": ${error.message}`,
        );
      }
    }

    if (uploadedFiles.length > 0) {
      this.updateFieldValue(fieldName, uploadedFiles);
    }
  }

  async uploadFile(file) {
    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch(this.options.fileUploadUrl, {
      method: "POST",
      body: formData,
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
      uploadedAt: new Date().toISOString(),
    };
  }

  showFilePreview(fieldName, file) {
    const fieldContainer = this.container.querySelector(
      `[data-field-name="${fieldName}"]`,
    );
    const previewArea = fieldContainer.querySelector(".file-preview-area");

    if (!previewArea) return;

    const preview = document.createElement("div");
    preview.className = "file-preview";
    preview.innerHTML = `
            <div class="file-info">
                <span class="file-name">${file.name}</span>
                <span class="file-size">${this.formatFileSize(file.size)}</span>
            </div>
            <button type="button" class="btn btn-ghost btn-sm remove-file" data-file-id="${file.id}">
                Remove
            </button>
        `;

    preview.querySelector(".remove-file").addEventListener("click", () => {
      this.removeFile(fieldName, file.id);
      preview.remove();
    });

    previewArea.appendChild(preview);
  }

  removeFile(fieldName, fileId) {
    const currentFiles = this.formData[fieldName] || [];
    const updatedFiles = currentFiles.filter((file) => file.id !== fileId);
    this.updateFieldValue(fieldName, updatedFiles);
  }

  formatFileSize(bytes) {
    if (bytes === 0) return "0 Bytes";

    const k = 1024;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));

    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
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

      this.emitEvent("stepChanged", {
        currentStep: this.currentStep,
        totalSteps: this.steps.length,
        formData: this.formData,
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

      this.emitEvent("stepChanged", {
        currentStep: this.currentStep,
        totalSteps: this.steps.length,
        formData: this.formData,
      });

      return true;
    }

    return false;
  }

  goToStep(stepIndex) {
    if (
      stepIndex >= 0 &&
      stepIndex < this.steps.length &&
      stepIndex <= this.currentStep
    ) {
      this.currentStep = stepIndex;
      this.renderCurrentStep();
      this.saveProgress();

      this.emitEvent("stepChanged", {
        currentStep: this.currentStep,
        totalSteps: this.steps.length,
        formData: this.formData,
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
      this.emitEvent("submitStart", { formData: this.formData });

      let result;
      if (this.options.submitUrl) {
        const response = await fetch(this.options.submitUrl, {
          method: this.options.submitMethod,
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(this.formData),
        });

        if (!response.ok) {
          throw new Error(`Submit failed: ${response.statusText}`);
        }

        result = await response.json();
      } else {
        result = { success: true, data: this.formData };
      }

      this.emitEvent("submitSuccess", { result, formData: this.formData });
      this.clearSavedProgress();

      return result;
    } catch (error) {
      this.emitEvent("submitError", { error, formData: this.formData });
      throw error;
    } finally {
      this.isSubmitting = false;
      this.renderNavigation();
    }
  }

  cancel() {
    this.emitEvent("cancel", { formData: this.formData });
    this.clearSavedProgress();
  }

  reset() {
    this.currentStep = 0;
    this.formData = {};
    this.validationErrors = {};
    this.uploadedFiles.clear();
    this.clearSavedProgress();
    this.renderCurrentStep();

    this.emitEvent("reset", {});
  }

  saveProgress() {
    if (!this.options.saveProgress) return;

    try {
      const progress = {
        currentStep: this.currentStep,
        formData: this.formData,
        timestamp: Date.now(),
      };

      localStorage.setItem(this.options.progressKey, JSON.stringify(progress));
    } catch (error) {
      console.warn("Failed to save form progress:", error);
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
      console.warn("Failed to load saved progress:", error);
    }
  }

  clearSavedProgress() {
    if (!this.options.saveProgress) return;

    try {
      localStorage.removeItem(this.options.progressKey);
    } catch (error) {
      console.warn("Failed to clear saved progress:", error);
    }
  }

  bindEvents() {
    // Form submission
    this.form.addEventListener("submit", (e) => {
      e.preventDefault();
      this.submitForm();
    });

    // Keyboard navigation
    this.container.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && e.ctrlKey) {
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
      cancelable: true,
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
      percentage: ((this.currentStep + 1) / this.steps.length) * 100,
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
      throw new Error("No steps defined. Add steps before starting the form.");
    }

    this.renderCurrentStep();
    this.emitEvent("started", { totalSteps: this.steps.length });
  }

  destroy() {
    this.clearSavedProgress();
    this.container.innerHTML = "";
    this.steps = [];
    this.formData = {};
    this.validationErrors = {};
    this.uploadedFiles.clear();
  }
}

// Form configuration examples
export const anomalyDetectionFormSteps = [
  {
    id: "dataset",
    title: "Dataset Configuration",
    description: "Upload and configure your dataset for anomaly detection",
    fields: [
      {
        type: "text",
        name: "dataset_name",
        label: "Dataset Name",
        placeholder: "Enter a name for your dataset",
        required: true,
        help: "Choose a descriptive name that will help you identify this dataset later",
      },
      {
        type: "file",
        name: "dataset_file",
        label: "Dataset File",
        required: true,
        accept: ".csv,.json,.xlsx,.parquet",
        help: "Upload your dataset in CSV, JSON, Excel, or Parquet format",
      },
      {
        type: "checkbox",
        name: "has_header",
        label: "Dataset has header row",
        defaultValue: true,
      },
    ],
  },
  {
    id: "algorithm",
    title: "Algorithm Selection",
    description:
      "Choose the anomaly detection algorithm and configure parameters",
    fields: [
      {
        type: "select",
        name: "algorithm",
        label: "Detection Algorithm",
        required: true,
        options: [
          { value: "isolation_forest", label: "Isolation Forest" },
          { value: "one_class_svm", label: "One-Class SVM" },
          { value: "local_outlier_factor", label: "Local Outlier Factor" },
          { value: "elliptic_envelope", label: "Elliptic Envelope" },
        ],
      },
      {
        type: "range",
        name: "contamination",
        label: "Contamination Rate",
        min: 0.01,
        max: 0.5,
        step: 0.01,
        defaultValue: 0.1,
        help: "Expected proportion of outliers in the dataset",
      },
      {
        type: "select",
        name: "features",
        label: "Feature Selection",
        options: [
          { value: "all", label: "Use all features" },
          { value: "select", label: "Select specific features" },
          { value: "auto", label: "Auto-select features" },
        ],
        defaultValue: "all",
      },
    ],
  },
  {
    id: "execution",
    title: "Execution Settings",
    description: "Configure how the anomaly detection will be executed",
    fields: [
      {
        type: "select",
        name: "execution_mode",
        label: "Execution Mode",
        required: true,
        options: [
          { value: "immediate", label: "Run immediately" },
          { value: "scheduled", label: "Schedule for later" },
          { value: "batch", label: "Batch processing" },
        ],
      },
      {
        type: "checkbox",
        name: "save_model",
        label: "Save trained model",
        defaultValue: true,
        help: "Save the model for future use and comparison",
      },
      {
        type: "checkbox",
        name: "generate_report",
        label: "Generate detailed report",
        defaultValue: true,
      },
    ],
  },
];

export default MultiStepForm;
\n\n// offline-dashboard.js\n/**
 * Offline Dashboard Component
 * Provides interactive dashboard functionality using cached data
 */
export class OfflineDashboard {
  constructor() {
    this.charts = new Map();
    this.cachedData = {
      datasets: [],
      results: [],
      stats: {},
      algorithms: [],
    };
    this.isInitialized = false;

    this.init();
  }

  async init() {
    await this.loadCachedData();
    this.setupEventListeners();
    this.renderDashboard();
    this.isInitialized = true;
  }

  /**
   * Load cached data from IndexedDB via service worker
   */
  async loadCachedData() {
    try {
      if ("serviceWorker" in navigator) {
        const registration = await navigator.serviceWorker.getRegistration();
        if (registration?.active) {
          // Request all cached data
          registration.active.postMessage({
            type: "GET_OFFLINE_DASHBOARD_DATA",
          });

          return new Promise((resolve) => {
            navigator.serviceWorker.addEventListener(
              "message",
              function handler(event) {
                if (event.data.type === "OFFLINE_DASHBOARD_DATA") {
                  navigator.serviceWorker.removeEventListener(
                    "message",
                    handler,
                  );
                  this.cachedData = {
                    ...this.cachedData,
                    ...event.data.data,
                  };
                  resolve(event.data.data);
                }
              }.bind(this),
            );
          });
        }
      }
    } catch (error) {
      console.error("[OfflineDashboard] Failed to load cached data:", error);
    }
  }

  /**
   * Setup event listeners for user interactions
   */
  setupEventListeners() {
    // Dataset selection
    document.addEventListener("change", (event) => {
      if (event.target.matches(".dataset-selector")) {
        this.onDatasetChange(event.target.value);
      }
    });

    // Algorithm selection
    document.addEventListener("change", (event) => {
      if (event.target.matches(".algorithm-selector")) {
        this.onAlgorithmChange(event.target.value);
      }
    });

    // Refresh button
    document.addEventListener("click", (event) => {
      if (event.target.matches(".refresh-dashboard")) {
        this.refreshDashboard();
      }
    });

    // Export buttons
    document.addEventListener("click", (event) => {
      if (event.target.matches(".export-chart")) {
        this.exportChart(event.target.dataset.chartId);
      }
    });
  }

  /**
   * Render the complete dashboard
   */
  renderDashboard() {
    this.renderOverviewCards();
    this.renderDatasetChart();
    this.renderAlgorithmPerformanceChart();
    this.renderAnomalyTimelineChart();
    this.renderRecentActivity();
  }

  /**
   * Render overview statistic cards
   */
  renderOverviewCards() {
    const container = document.getElementById("overview-cards");
    if (!container) return;

    const stats = this.calculateStats();

    container.innerHTML = `
      <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <div class="card">
          <div class="card-body">
            <div class="flex items-center justify-between">
              <div>
                <p class="text-text-secondary text-sm">Total Datasets</p>
                <p class="text-2xl font-bold">${stats.totalDatasets}</p>
              </div>
              <div class="text-3xl">📊</div>
            </div>
            <div class="mt-2 text-sm text-green-600">
              ${stats.datasetsLastWeek} added this week
            </div>
          </div>
        </div>

        <div class="card">
          <div class="card-body">
            <div class="flex items-center justify-between">
              <div>
                <p class="text-text-secondary text-sm">Detections Run</p>
                <p class="text-2xl font-bold">${stats.totalDetections}</p>
              </div>
              <div class="text-3xl">🔍</div>
            </div>
            <div class="mt-2 text-sm text-blue-600">
              ${stats.detectionsToday} today
            </div>
          </div>
        </div>

        <div class="card">
          <div class="card-body">
            <div class="flex items-center justify-between">
              <div>
                <p class="text-text-secondary text-sm">Anomalies Found</p>
                <p class="text-2xl font-bold">${stats.totalAnomalies}</p>
              </div>
              <div class="text-3xl">⚠️</div>
            </div>
            <div class="mt-2 text-sm ${stats.anomalyRate > 0.1 ? "text-red-600" : "text-gray-600"}">
              ${(stats.anomalyRate * 100).toFixed(1)}% anomaly rate
            </div>
          </div>
        </div>

        <div class="card">
          <div class="card-body">
            <div class="flex items-center justify-between">
              <div>
                <p class="text-text-secondary text-sm">Cached Data</p>
                <p class="text-2xl font-bold">${this.formatBytes(stats.cacheSize)}</p>
              </div>
              <div class="text-3xl">💾</div>
            </div>
            <div class="mt-2 text-sm text-purple-600">
              Available offline
            </div>
          </div>
        </div>
      </div>
    `;
  }

  /**
   * Render dataset distribution chart
   */
  renderDatasetChart() {
    const container = document.getElementById("dataset-chart");
    if (!container) return;

    const datasets = this.cachedData.datasets || [];
    const typeDistribution = datasets.reduce((acc, dataset) => {
      const type = dataset.type || "unknown";
      acc[type] = (acc[type] || 0) + 1;
      return acc;
    }, {});

    const chartData = Object.entries(typeDistribution).map(([name, value]) => ({
      name: name.charAt(0).toUpperCase() + name.slice(1),
      value,
    }));

    // Use ECharts for visualization
    const chart = echarts.init(container);
    const option = {
      title: {
        text: "Dataset Distribution",
        left: "center",
      },
      tooltip: {
        trigger: "item",
        formatter: "{a} <br/>{b}: {c} ({d}%)",
      },
      legend: {
        orient: "vertical",
        left: "left",
      },
      series: [
        {
          name: "Datasets",
          type: "pie",
          radius: "50%",
          data: chartData,
          emphasis: {
            itemStyle: {
              shadowBlur: 10,
              shadowOffsetX: 0,
              shadowColor: "rgba(0, 0, 0, 0.5)",
            },
          },
        },
      ],
    };

    chart.setOption(option);
    this.charts.set("dataset-chart", chart);

    // Make chart responsive
    window.addEventListener("resize", () => chart.resize());
  }

  /**
   * Render algorithm performance comparison chart
   */
  renderAlgorithmPerformanceChart() {
    const container = document.getElementById("algorithm-performance-chart");
    if (!container) return;

    const results = this.cachedData.results || [];
    const algorithmStats = results.reduce((acc, result) => {
      const algo = result.algorithm || "unknown";
      if (!acc[algo]) {
        acc[algo] = { count: 0, totalTime: 0, totalAnomalies: 0 };
      }
      acc[algo].count++;
      acc[algo].totalTime += result.processingTime || 0;
      acc[algo].totalAnomalies += result.anomalies?.length || 0;
      return acc;
    }, {});

    const algorithms = Object.keys(algorithmStats);
    const avgTimes = algorithms.map(
      (algo) => algorithmStats[algo].totalTime / algorithmStats[algo].count,
    );
    const totalAnomalies = algorithms.map(
      (algo) => algorithmStats[algo].totalAnomalies,
    );

    const chart = echarts.init(container);
    const option = {
      title: {
        text: "Algorithm Performance",
        left: "center",
      },
      tooltip: {
        trigger: "axis",
        axisPointer: {
          type: "cross",
        },
      },
      legend: {
        data: ["Average Processing Time (ms)", "Total Anomalies Found"],
        bottom: 0,
      },
      xAxis: {
        type: "category",
        data: algorithms,
        axisPointer: {
          type: "shadow",
        },
      },
      yAxis: [
        {
          type: "value",
          name: "Time (ms)",
          position: "left",
        },
        {
          type: "value",
          name: "Anomalies",
          position: "right",
        },
      ],
      series: [
        {
          name: "Average Processing Time (ms)",
          type: "bar",
          yAxisIndex: 0,
          data: avgTimes,
          itemStyle: {
            color: "#3b82f6",
          },
        },
        {
          name: "Total Anomalies Found",
          type: "line",
          yAxisIndex: 1,
          data: totalAnomalies,
          itemStyle: {
            color: "#ef4444",
          },
        },
      ],
    };

    chart.setOption(option);
    this.charts.set("algorithm-performance-chart", chart);

    window.addEventListener("resize", () => chart.resize());
  }

  /**
   * Render anomaly detection timeline chart
   */
  renderAnomalyTimelineChart() {
    const container = document.getElementById("anomaly-timeline-chart");
    if (!container) return;

    const results = this.cachedData.results || [];

    // Group results by day
    const timelineData = results.reduce((acc, result) => {
      const date = new Date(result.timestamp).toISOString().split("T")[0];
      if (!acc[date]) {
        acc[date] = { detections: 0, anomalies: 0 };
      }
      acc[date].detections++;
      acc[date].anomalies += result.anomalies?.length || 0;
      return acc;
    }, {});

    const dates = Object.keys(timelineData).sort();
    const detections = dates.map((date) => timelineData[date].detections);
    const anomalies = dates.map((date) => timelineData[date].anomalies);

    const chart = echarts.init(container);
    const option = {
      title: {
        text: "Detection Activity Timeline",
        left: "center",
      },
      tooltip: {
        trigger: "axis",
        axisPointer: {
          type: "cross",
        },
      },
      legend: {
        data: ["Detections Run", "Anomalies Found"],
        bottom: 0,
      },
      grid: {
        left: "3%",
        right: "4%",
        bottom: "15%",
        containLabel: true,
      },
      xAxis: {
        type: "category",
        boundaryGap: false,
        data: dates,
      },
      yAxis: {
        type: "value",
      },
      series: [
        {
          name: "Detections Run",
          type: "line",
          stack: "Total",
          areaStyle: {},
          data: detections,
          itemStyle: {
            color: "#10b981",
          },
        },
        {
          name: "Anomalies Found",
          type: "line",
          stack: "Total",
          areaStyle: {},
          data: anomalies,
          itemStyle: {
            color: "#f59e0b",
          },
        },
      ],
    };

    chart.setOption(option);
    this.charts.set("anomaly-timeline-chart", chart);

    window.addEventListener("resize", () => chart.resize());
  }

  /**
   * Render recent activity feed
   */
  renderRecentActivity() {
    const container = document.getElementById("recent-activity");
    if (!container) return;

    const results = this.cachedData.results || [];
    const recentResults = results
      .sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp))
      .slice(0, 10);

    const activityHtml = recentResults
      .map((result) => {
        const timeAgo = this.timeAgo(new Date(result.timestamp));
        const anomalyCount = result.anomalies?.length || 0;
        const statusColor =
          anomalyCount > 0 ? "text-orange-600" : "text-green-600";
        const statusIcon = anomalyCount > 0 ? "⚠️" : "✅";

        return `
        <div class="flex items-start gap-3 p-3 border-b border-border last:border-b-0">
          <div class="text-xl">${statusIcon}</div>
          <div class="flex-grow">
            <div class="flex items-center justify-between">
              <h4 class="font-medium">${result.dataset || "Unknown Dataset"}</h4>
              <span class="text-sm text-text-secondary">${timeAgo}</span>
            </div>
            <p class="text-sm text-text-secondary">
              Algorithm: ${result.algorithm || "Unknown"}
            </p>
            <p class="text-sm ${statusColor}">
              ${anomalyCount} anomalies detected
            </p>
          </div>
        </div>
      `;
      })
      .join("");

    container.innerHTML = `
      <div class="card">
        <div class="card-header">
          <h3 class="card-title">Recent Activity</h3>
          <button class="btn-base btn-sm refresh-dashboard">
            <svg class="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
              <path fill-rule="evenodd" d="M4 2a1 1 0 011 1v2.101a7.002 7.002 0 0111.601 2.566 1 1 0 11-1.885.666A5.002 5.002 0 005.999 7H9a1 1 0 010 2H4a1 1 0 01-1-1V3a1 1 0 011-1zm.008 9.057a1 1 0 011.276.61A5.002 5.002 0 0014.001 13H11a1 1 0 110-2h5a1 1 0 011 1v5a1 1 0 11-2 0v-2.101a7.002 7.002 0 01-11.601-2.566 1 1 0 01.61-1.276z" clip-rule="evenodd"></path>
            </svg>
            Refresh
          </button>
        </div>
        <div class="card-body p-0">
          ${activityHtml || '<div class="p-4 text-center text-text-secondary">No recent activity</div>'}
        </div>
      </div>
    `;
  }

  /**
   * Handle dataset selection change
   */
  onDatasetChange(datasetId) {
    const dataset = this.cachedData.datasets.find((d) => d.id === datasetId);
    if (dataset) {
      this.renderDatasetDetails(dataset);
    }
  }

  /**
   * Handle algorithm selection change
   */
  onAlgorithmChange(algorithmId) {
    const algorithm = this.cachedData.algorithms.find(
      (a) => a.id === algorithmId,
    );
    if (algorithm) {
      this.renderAlgorithmDetails(algorithm);
    }
  }

  /**
   * Refresh dashboard data
   */
  async refreshDashboard() {
    const refreshButton = document.querySelector(".refresh-dashboard");
    if (refreshButton) {
      refreshButton.disabled = true;
      refreshButton.innerHTML = "Refreshing...";
    }

    try {
      await this.loadCachedData();
      this.renderDashboard();
    } catch (error) {
      console.error("[OfflineDashboard] Failed to refresh:", error);
    } finally {
      if (refreshButton) {
        refreshButton.disabled = false;
        refreshButton.innerHTML = `
          <svg class="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
            <path fill-rule="evenodd" d="M4 2a1 1 0 011 1v2.101a7.002 7.002 0 0111.601 2.566 1 1 0 11-1.885.666A5.002 5.002 0 005.999 7H9a1 1 0 010 2H4a1 1 0 01-1-1V3a1 1 0 011-1zm.008 9.057a1 1 0 011.276.61A5.002 5.002 0 0014.001 13H11a1 1 0 110-2h5a1 1 0 011 1v5a1 1 0 11-2 0v-2.101a7.002 7.002 0 01-11.601-2.566 1 1 0 01.61-1.276z" clip-rule="evenodd"></path>
          </svg>
          Refresh
        `;
      }
    }
  }

  /**
   * Export chart as image
   */
  exportChart(chartId) {
    const chart = this.charts.get(chartId);
    if (chart) {
      const dataURL = chart.getDataURL({
        type: "png",
        pixelRatio: 2,
        backgroundColor: "#fff",
      });

      const link = document.createElement("a");
      link.download = `${chartId}-${Date.now()}.png`;
      link.href = dataURL;
      link.click();
    }
  }

  /**
   * Calculate dashboard statistics
   */
  calculateStats() {
    const datasets = this.cachedData.datasets || [];
    const results = this.cachedData.results || [];

    const now = new Date();
    const weekAgo = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);
    const dayAgo = new Date(now.getTime() - 24 * 60 * 60 * 1000);

    const datasetsLastWeek = datasets.filter(
      (d) => new Date(d.timestamp) > weekAgo,
    ).length;

    const detectionsToday = results.filter(
      (r) => new Date(r.timestamp) > dayAgo,
    ).length;

    const totalAnomalies = results.reduce(
      (sum, r) => sum + (r.anomalies?.length || 0),
      0,
    );

    const totalSamples = results.reduce(
      (sum, r) => sum + (r.totalSamples || 0),
      0,
    );

    return {
      totalDatasets: datasets.length,
      datasetsLastWeek,
      totalDetections: results.length,
      detectionsToday,
      totalAnomalies,
      anomalyRate: totalSamples > 0 ? totalAnomalies / totalSamples : 0,
      cacheSize: this.estimateCacheSize(),
    };
  }

  /**
   * Estimate cache size
   */
  estimateCacheSize() {
    const jsonSize = JSON.stringify(this.cachedData).length;
    return jsonSize * 2; // Rough estimate including IndexedDB overhead
  }

  /**
   * Format bytes to human readable
   */
  formatBytes(bytes) {
    if (bytes === 0) return "0 B";
    const k = 1024;
    const sizes = ["B", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
  }

  /**
   * Format time ago
   */
  timeAgo(date) {
    const now = new Date();
    const diffInSeconds = Math.floor((now - date) / 1000);

    if (diffInSeconds < 60) return "Just now";
    if (diffInSeconds < 3600) return `${Math.floor(diffInSeconds / 60)}m ago`;
    if (diffInSeconds < 86400)
      return `${Math.floor(diffInSeconds / 3600)}h ago`;
    return `${Math.floor(diffInSeconds / 86400)}d ago`;
  }
}

// Initialize and expose globally
if (typeof window !== "undefined") {
  window.OfflineDashboard = new OfflineDashboard();
}
\n\n// offline-detector.js\n/**
 * Offline Anomaly Detection Component
 * Provides anomaly detection capabilities using cached data and local algorithms
 */
export class OfflineDetector {
  constructor() {
    this.algorithms = new Map();
    this.cachedDatasets = new Map();
    this.cachedModels = new Map();
    this.isInitialized = false;

    this.initializeAlgorithms();
  }

  /**
   * Initialize local anomaly detection algorithms
   */
  initializeAlgorithms() {
    // Simple statistical algorithms that can run in the browser
    this.algorithms.set("zscore", {
      name: "Z-Score Detection",
      description: "Statistical outlier detection using Z-scores",
      parameters: { threshold: 3.0 },
      detect: this.zScoreDetection.bind(this),
    });

    this.algorithms.set("iqr", {
      name: "Interquartile Range",
      description: "Outlier detection using IQR method",
      parameters: { factor: 1.5 },
      detect: this.iqrDetection.bind(this),
    });

    this.algorithms.set("isolation", {
      name: "Simple Isolation Detection",
      description: "Basic isolation-based anomaly detection",
      parameters: { contamination: 0.1 },
      detect: this.isolationDetection.bind(this),
    });

    this.algorithms.set("mad", {
      name: "Median Absolute Deviation",
      description: "Robust outlier detection using MAD",
      parameters: { threshold: 3.5 },
      detect: this.madDetection.bind(this),
    });

    this.isInitialized = true;
  }

  /**
   * Load cached datasets from IndexedDB
   */
  async loadCachedDatasets() {
    try {
      if ("serviceWorker" in navigator) {
        const registration = await navigator.serviceWorker.getRegistration();
        if (registration?.active) {
          // Request cached datasets from service worker
          registration.active.postMessage({ type: "GET_OFFLINE_DATASETS" });

          // Listen for response
          return new Promise((resolve) => {
            navigator.serviceWorker.addEventListener(
              "message",
              function handler(event) {
                if (event.data.type === "OFFLINE_DATASETS") {
                  navigator.serviceWorker.removeEventListener(
                    "message",
                    handler,
                  );
                  event.data.datasets.forEach((dataset) => {
                    this.cachedDatasets.set(dataset.id, dataset);
                  });
                  resolve(event.data.datasets);
                }
              }.bind(this),
            );
          });
        }
      }
    } catch (error) {
      console.error("[OfflineDetector] Failed to load cached datasets:", error);
      return [];
    }
  }

  /**
   * Get available algorithms
   */
  getAlgorithms() {
    return Array.from(this.algorithms.entries()).map(([id, algo]) => ({
      id,
      name: algo.name,
      description: algo.description,
      parameters: algo.parameters,
    }));
  }

  /**
   * Get cached datasets
   */
  getCachedDatasets() {
    return Array.from(this.cachedDatasets.values());
  }

  /**
   * Run anomaly detection on cached data
   */
  async detectAnomalies(datasetId, algorithmId, parameters = {}) {
    if (!this.isInitialized) {
      throw new Error("Offline detector not initialized");
    }

    const dataset = this.cachedDatasets.get(datasetId);
    if (!dataset) {
      throw new Error(`Dataset ${datasetId} not found in cache`);
    }

    const algorithm = this.algorithms.get(algorithmId);
    if (!algorithm) {
      throw new Error(`Algorithm ${algorithmId} not available`);
    }

    const startTime = performance.now();

    try {
      // Prepare data
      const data = this.prepareData(dataset.data);

      // Run detection
      const config = { ...algorithm.parameters, ...parameters };
      const result = algorithm.detect(data, config);

      const endTime = performance.now();
      const processingTime = endTime - startTime;

      // Create detection result
      const detectionResult = {
        id: `offline_${Date.now()}`,
        datasetId: datasetId,
        algorithmId: algorithmId,
        timestamp: new Date().toISOString(),
        processingTimeMs: processingTime,
        anomalies: result.anomalies,
        scores: result.scores,
        statistics: result.statistics,
        parameters: config,
        isOffline: true,
      };

      // Save to offline storage
      await this.saveResult(detectionResult);

      return detectionResult;
    } catch (error) {
      console.error("[OfflineDetector] Detection failed:", error);
      throw error;
    }
  }

  /**
   * Prepare data for analysis
   */
  prepareData(rawData) {
    // Convert to numeric matrix if needed
    if (Array.isArray(rawData)) {
      return rawData.map((row) => {
        if (typeof row === "object") {
          return Object.values(row).map((val) => {
            const num = parseFloat(val);
            return isNaN(num) ? 0 : num;
          });
        }
        return Array.isArray(row) ? row : [row];
      });
    }
    return rawData;
  }

  /**
   * Z-Score based anomaly detection
   */
  zScoreDetection(data, config) {
    const { threshold = 3.0 } = config;
    const anomalies = [];
    const scores = [];

    // Calculate statistics for each feature
    const features = data[0].length;
    const featureStats = [];

    for (let f = 0; f < features; f++) {
      const values = data.map((row) => row[f]);
      const mean = values.reduce((a, b) => a + b, 0) / values.length;
      const variance =
        values.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / values.length;
      const std = Math.sqrt(variance);

      featureStats.push({ mean, std });
    }

    // Calculate Z-scores and detect anomalies
    data.forEach((row, index) => {
      let maxZScore = 0;

      row.forEach((value, featureIndex) => {
        const { mean, std } = featureStats[featureIndex];
        const zScore = std > 0 ? Math.abs((value - mean) / std) : 0;
        maxZScore = Math.max(maxZScore, zScore);
      });

      scores.push(maxZScore);

      if (maxZScore > threshold) {
        anomalies.push({
          index,
          score: maxZScore,
          values: row,
        });
      }
    });

    return {
      anomalies,
      scores,
      statistics: {
        totalSamples: data.length,
        totalAnomalies: anomalies.length,
        anomalyRate: anomalies.length / data.length,
        averageScore: scores.reduce((a, b) => a + b, 0) / scores.length,
        maxScore: Math.max(...scores),
        threshold,
      },
    };
  }

  /**
   * IQR based anomaly detection
   */
  iqrDetection(data, config) {
    const { factor = 1.5 } = config;
    const anomalies = [];
    const scores = [];

    // Calculate IQR for each feature
    const features = data[0].length;
    const featureBounds = [];

    for (let f = 0; f < features; f++) {
      const values = data.map((row) => row[f]).sort((a, b) => a - b);
      const q1Index = Math.floor(values.length * 0.25);
      const q3Index = Math.floor(values.length * 0.75);
      const q1 = values[q1Index];
      const q3 = values[q3Index];
      const iqr = q3 - q1;

      featureBounds.push({
        lower: q1 - factor * iqr,
        upper: q3 + factor * iqr,
        iqr,
      });
    }

    // Detect anomalies
    data.forEach((row, index) => {
      let anomalyScore = 0;
      let isAnomaly = false;

      row.forEach((value, featureIndex) => {
        const bounds = featureBounds[featureIndex];
        if (value < bounds.lower || value > bounds.upper) {
          isAnomaly = true;
          const deviation = Math.min(
            Math.abs(value - bounds.lower),
            Math.abs(value - bounds.upper),
          );
          anomalyScore = Math.max(
            anomalyScore,
            deviation / Math.max(bounds.iqr, 1),
          );
        }
      });

      scores.push(anomalyScore);

      if (isAnomaly) {
        anomalies.push({
          index,
          score: anomalyScore,
          values: row,
        });
      }
    });

    return {
      anomalies,
      scores,
      statistics: {
        totalSamples: data.length,
        totalAnomalies: anomalies.length,
        anomalyRate: anomalies.length / data.length,
        averageScore: scores.reduce((a, b) => a + b, 0) / scores.length,
        maxScore: Math.max(...scores),
        factor,
      },
    };
  }

  /**
   * Simple isolation-based detection
   */
  isolationDetection(data, config) {
    const { contamination = 0.1 } = config;

    // Simple implementation: random feature selection and isolation
    const scores = data.map(() => 0);
    const numTrees = 100;
    const maxDepth = Math.ceil(Math.log2(data.length));

    for (let tree = 0; tree < numTrees; tree++) {
      const pathLengths = this.isolationTree(data, maxDepth);
      pathLengths.forEach((length, index) => {
        scores[index] += length;
      });
    }

    // Normalize scores
    const avgPathLength = scores.reduce((a, b) => a + b, 0) / scores.length;
    const normalizedScores = scores.map((score) =>
      Math.pow(2, -(score / numTrees) / avgPathLength),
    );

    // Determine threshold based on contamination rate
    const sortedScores = [...normalizedScores].sort((a, b) => b - a);
    const thresholdIndex = Math.floor(data.length * contamination);
    const threshold = sortedScores[thresholdIndex] || 0.5;

    const anomalies = [];
    normalizedScores.forEach((score, index) => {
      if (score > threshold) {
        anomalies.push({
          index,
          score,
          values: data[index],
        });
      }
    });

    return {
      anomalies,
      scores: normalizedScores,
      statistics: {
        totalSamples: data.length,
        totalAnomalies: anomalies.length,
        anomalyRate: anomalies.length / data.length,
        averageScore:
          normalizedScores.reduce((a, b) => a + b, 0) / normalizedScores.length,
        maxScore: Math.max(...normalizedScores),
        threshold,
        contamination,
      },
    };
  }

  /**
   * Simple isolation tree implementation
   */
  isolationTree(data, maxDepth, currentDepth = 0) {
    if (currentDepth >= maxDepth || data.length <= 1) {
      return data.map(() => currentDepth);
    }

    // Random feature selection
    const featureIndex = Math.floor(Math.random() * data[0].length);
    const values = data.map((row) => row[featureIndex]);
    const minVal = Math.min(...values);
    const maxVal = Math.max(...values);

    if (minVal === maxVal) {
      return data.map(() => currentDepth);
    }

    // Random split point
    const splitPoint = minVal + Math.random() * (maxVal - minVal);

    // Split data
    const leftData = [];
    const rightData = [];
    const leftIndices = [];
    const rightIndices = [];

    data.forEach((row, index) => {
      if (row[featureIndex] < splitPoint) {
        leftData.push(row);
        leftIndices.push(index);
      } else {
        rightData.push(row);
        rightIndices.push(index);
      }
    });

    // Recursive calls
    const leftPaths =
      leftData.length > 0
        ? this.isolationTree(leftData, maxDepth, currentDepth + 1)
        : [];
    const rightPaths =
      rightData.length > 0
        ? this.isolationTree(rightData, maxDepth, currentDepth + 1)
        : [];

    // Combine results
    const result = new Array(data.length);
    leftIndices.forEach((originalIndex, i) => {
      result[i] = leftPaths[i];
    });
    rightIndices.forEach((originalIndex, i) => {
      result[i + leftData.length] = rightPaths[i];
    });

    return result;
  }

  /**
   * MAD (Median Absolute Deviation) based detection
   */
  madDetection(data, config) {
    const { threshold = 3.5 } = config;
    const anomalies = [];
    const scores = [];

    // Calculate MAD for each feature
    const features = data[0].length;
    const featureStats = [];

    for (let f = 0; f < features; f++) {
      const values = data.map((row) => row[f]).sort((a, b) => a - b);
      const median = values[Math.floor(values.length / 2)];
      const deviations = values
        .map((val) => Math.abs(val - median))
        .sort((a, b) => a - b);
      const mad = deviations[Math.floor(deviations.length / 2)];

      featureStats.push({ median, mad });
    }

    // Calculate modified Z-scores and detect anomalies
    data.forEach((row, index) => {
      let maxScore = 0;

      row.forEach((value, featureIndex) => {
        const { median, mad } = featureStats[featureIndex];
        const modifiedZScore = mad > 0 ? (0.6745 * (value - median)) / mad : 0;
        maxScore = Math.max(maxScore, Math.abs(modifiedZScore));
      });

      scores.push(maxScore);

      if (maxScore > threshold) {
        anomalies.push({
          index,
          score: maxScore,
          values: row,
        });
      }
    });

    return {
      anomalies,
      scores,
      statistics: {
        totalSamples: data.length,
        totalAnomalies: anomalies.length,
        anomalyRate: anomalies.length / data.length,
        averageScore: scores.reduce((a, b) => a + b, 0) / scores.length,
        maxScore: Math.max(...scores),
        threshold,
      },
    };
  }

  /**
   * Save detection result to offline storage
   */
  async saveResult(result) {
    try {
      if ("serviceWorker" in navigator) {
        const registration = await navigator.serviceWorker.getRegistration();
        if (registration?.active) {
          registration.active.postMessage({
            type: "SAVE_DETECTION_RESULT",
            payload: result,
          });
        }
      }
    } catch (error) {
      console.error("[OfflineDetector] Failed to save result:", error);
    }
  }

  /**
   * Get detection history from offline storage
   */
  async getDetectionHistory() {
    try {
      if ("serviceWorker" in navigator) {
        const registration = await navigator.serviceWorker.getRegistration();
        if (registration?.active) {
          registration.active.postMessage({ type: "GET_OFFLINE_RESULTS" });

          return new Promise((resolve) => {
            navigator.serviceWorker.addEventListener(
              "message",
              function handler(event) {
                if (event.data.type === "OFFLINE_RESULTS") {
                  navigator.serviceWorker.removeEventListener(
                    "message",
                    handler,
                  );
                  resolve(event.data.results);
                }
              },
            );
          });
        }
      }
    } catch (error) {
      console.error(
        "[OfflineDetector] Failed to get detection history:",
        error,
      );
      return [];
    }
  }
}

// Initialize and expose globally
if (typeof window !== "undefined") {
  window.OfflineDetector = new OfflineDetector();
}
\n\n// offline-visualizer.js\n/**
 * Offline Data Visualizer Component
 * Provides advanced data visualization capabilities using cached data
 */
export class OfflineVisualizer {
  constructor() {
    this.charts = new Map();
    this.datasets = new Map();
    this.results = new Map();
    this.currentDataset = null;
    this.currentResult = null;
    this.isInitialized = false;

    this.init();
  }

  async init() {
    await this.loadCachedData();
    this.setupEventListeners();
    this.isInitialized = true;
  }

  /**
   * Load cached data from IndexedDB
   */
  async loadCachedData() {
    try {
      if ("serviceWorker" in navigator) {
        const registration = await navigator.serviceWorker.getRegistration();
        if (registration?.active) {
          registration.active.postMessage({ type: "GET_OFFLINE_DATASETS" });
          registration.active.postMessage({ type: "GET_OFFLINE_RESULTS" });

          return new Promise((resolve) => {
            let datasetsLoaded = false;
            let resultsLoaded = false;

            navigator.serviceWorker.addEventListener("message", (event) => {
              if (event.data.type === "OFFLINE_DATASETS") {
                event.data.datasets.forEach((dataset) => {
                  this.datasets.set(dataset.id, dataset);
                });
                datasetsLoaded = true;
              } else if (event.data.type === "OFFLINE_RESULTS") {
                event.data.results.forEach((result) => {
                  this.results.set(result.id, result);
                });
                resultsLoaded = true;
              }

              if (datasetsLoaded && resultsLoaded) {
                resolve();
              }
            });
          });
        }
      }
    } catch (error) {
      console.error("[OfflineVisualizer] Failed to load cached data:", error);
    }
  }

  /**
   * Setup event listeners
   */
  setupEventListeners() {
    // Dataset selection
    document.addEventListener("change", (event) => {
      if (event.target.matches(".dataset-selector")) {
        this.selectDataset(event.target.value);
      }
    });

    // Result selection
    document.addEventListener("change", (event) => {
      if (event.target.matches(".result-selector")) {
        this.selectResult(event.target.value);
      }
    });

    // Visualization type selection
    document.addEventListener("change", (event) => {
      if (event.target.matches(".viz-type-selector")) {
        this.changeVisualizationType(event.target.value);
      }
    });

    // Export functionality
    document.addEventListener("click", (event) => {
      if (event.target.matches(".export-viz")) {
        this.exportVisualization(event.target.dataset.format);
      }
    });
  }

  /**
   * Select and load a dataset for visualization
   */
  async selectDataset(datasetId) {
    const dataset = this.datasets.get(datasetId);
    if (!dataset) return;

    this.currentDataset = dataset;
    await this.renderDatasetVisualization();
    this.updateResultSelector();
  }

  /**
   * Select and load a result for visualization
   */
  async selectResult(resultId) {
    const result = this.results.get(resultId);
    if (!result) return;

    this.currentResult = result;
    await this.renderResultVisualization();
  }

  /**
   * Render dataset visualization
   */
  async renderDatasetVisualization() {
    if (!this.currentDataset) return;

    const container = document.getElementById("dataset-visualization");
    if (!container) return;

    const data = this.currentDataset.data;
    const features = this.extractFeatures(data);

    // Clear existing charts
    this.clearCharts();

    // Create visualization container
    container.innerHTML = `
      <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div class="card">
          <div class="card-header">
            <h3 class="card-title">Data Distribution</h3>
            <select class="form-select form-select-sm viz-type-selector" data-target="distribution">
              <option value="histogram">Histogram</option>
              <option value="boxplot">Box Plot</option>
              <option value="violin">Violin Plot</option>
            </select>
          </div>
          <div class="card-body">
            <div id="distribution-chart" style="height: 300px;"></div>
          </div>
        </div>

        <div class="card">
          <div class="card-header">
            <h3 class="card-title">Feature Correlation</h3>
            <select class="form-select form-select-sm viz-type-selector" data-target="correlation">
              <option value="heatmap">Heatmap</option>
              <option value="scatter">Scatter Matrix</option>
            </select>
          </div>
          <div class="card-body">
            <div id="correlation-chart" style="height: 300px;"></div>
          </div>
        </div>

        <div class="card">
          <div class="card-header">
            <h3 class="card-title">Feature Statistics</h3>
          </div>
          <div class="card-body">
            <div id="statistics-table"></div>
          </div>
        </div>

        <div class="card">
          <div class="card-header">
            <h3 class="card-title">Data Quality</h3>
          </div>
          <div class="card-body">
            <div id="quality-chart" style="height: 300px;"></div>
          </div>
        </div>
      </div>
    `;

    // Render individual charts
    await Promise.all([
      this.renderDistributionChart(features),
      this.renderCorrelationChart(features),
      this.renderStatisticsTable(features),
      this.renderQualityChart(features),
    ]);
  }

  /**
   * Render result visualization (anomaly detection results)
   */
  async renderResultVisualization() {
    if (!this.currentResult) return;

    const container = document.getElementById("result-visualization");
    if (!container) return;

    container.innerHTML = `
      <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div class="card">
          <div class="card-header">
            <h3 class="card-title">Anomaly Distribution</h3>
          </div>
          <div class="card-body">
            <div id="anomaly-distribution-chart" style="height: 300px;"></div>
          </div>
        </div>

        <div class="card">
          <div class="card-header">
            <h3 class="card-title">Score Distribution</h3>
          </div>
          <div class="card-body">
            <div id="score-distribution-chart" style="height: 300px;"></div>
          </div>
        </div>

        <div class="card">
          <div class="card-header">
            <h3 class="card-title">Anomaly Scatter Plot</h3>
          </div>
          <div class="card-body">
            <div id="anomaly-scatter-chart" style="height: 400px;"></div>
          </div>
        </div>

        <div class="card">
          <div class="card-header">
            <h3 class="card-title">Detection Summary</h3>
          </div>
          <div class="card-body">
            <div id="detection-summary"></div>
          </div>
        </div>
      </div>
    `;

    await Promise.all([
      this.renderAnomalyDistributionChart(),
      this.renderScoreDistributionChart(),
      this.renderAnomalyScatterPlot(),
      this.renderDetectionSummary(),
    ]);
  }

  /**
   * Render distribution chart (histogram/boxplot)
   */
  async renderDistributionChart(features) {
    const container = document.getElementById("distribution-chart");
    if (!container || !features.length) return;

    const chart = echarts.init(container);

    // Use first numeric feature for histogram
    const feature = features.find((f) => f.type === "numeric");
    if (!feature) return;

    const values = feature.values;
    const bins = this.calculateHistogramBins(values, 20);

    const option = {
      title: {
        text: `Distribution: ${feature.name}`,
        textStyle: { fontSize: 14 },
      },
      tooltip: {
        trigger: "axis",
        axisPointer: { type: "shadow" },
      },
      xAxis: {
        type: "category",
        data: bins.map((bin) => bin.range),
      },
      yAxis: {
        type: "value",
        name: "Frequency",
      },
      series: [
        {
          name: "Frequency",
          type: "bar",
          data: bins.map((bin) => bin.count),
          itemStyle: {
            color: "#3b82f6",
          },
        },
      ],
    };

    chart.setOption(option);
    this.charts.set("distribution-chart", chart);
  }

  /**
   * Render correlation heatmap
   */
  async renderCorrelationChart(features) {
    const container = document.getElementById("correlation-chart");
    if (!container) return;

    const numericFeatures = features.filter((f) => f.type === "numeric");
    if (numericFeatures.length < 2) return;

    const correlationMatrix = this.calculateCorrelationMatrix(numericFeatures);
    const chart = echarts.init(container);

    const option = {
      title: {
        text: "Feature Correlation Matrix",
        textStyle: { fontSize: 14 },
      },
      tooltip: {
        position: "top",
        formatter: (params) => {
          return `${params.name}<br/>Correlation: ${params.value[2].toFixed(3)}`;
        },
      },
      grid: {
        height: "50%",
        top: "10%",
      },
      xAxis: {
        type: "category",
        data: numericFeatures.map((f) => f.name),
        splitArea: { show: true },
      },
      yAxis: {
        type: "category",
        data: numericFeatures.map((f) => f.name),
        splitArea: { show: true },
      },
      visualMap: {
        min: -1,
        max: 1,
        calculable: true,
        orient: "horizontal",
        left: "center",
        bottom: "15%",
        inRange: {
          color: [
            "#313695",
            "#4575b4",
            "#74add1",
            "#abd9e9",
            "#e0f3f8",
            "#ffffbf",
            "#fee090",
            "#fdae61",
            "#f46d43",
            "#d73027",
            "#a50026",
          ],
        },
      },
      series: [
        {
          name: "Correlation",
          type: "heatmap",
          data: correlationMatrix,
          label: {
            show: true,
            formatter: (params) => params.value[2].toFixed(2),
          },
          emphasis: {
            itemStyle: {
              shadowBlur: 10,
              shadowColor: "rgba(0, 0, 0, 0.5)",
            },
          },
        },
      ],
    };

    chart.setOption(option);
    this.charts.set("correlation-chart", chart);
  }

  /**
   * Render statistics table
   */
  renderStatisticsTable(features) {
    const container = document.getElementById("statistics-table");
    if (!container) return;

    const numericFeatures = features.filter((f) => f.type === "numeric");

    const tableRows = numericFeatures
      .map((feature) => {
        const stats = this.calculateBasicStats(feature.values);
        return `
        <tr>
          <td class="font-medium">${feature.name}</td>
          <td>${stats.mean.toFixed(3)}</td>
          <td>${stats.std.toFixed(3)}</td>
          <td>${stats.min.toFixed(3)}</td>
          <td>${stats.max.toFixed(3)}</td>
          <td>${stats.median.toFixed(3)}</td>
        </tr>
      `;
      })
      .join("");

    container.innerHTML = `
      <div class="overflow-x-auto">
        <table class="table table-striped">
          <thead>
            <tr>
              <th>Feature</th>
              <th>Mean</th>
              <th>Std Dev</th>
              <th>Min</th>
              <th>Max</th>
              <th>Median</th>
            </tr>
          </thead>
          <tbody>
            ${tableRows}
          </tbody>
        </table>
      </div>
    `;
  }

  /**
   * Render data quality chart
   */
  async renderQualityChart(features) {
    const container = document.getElementById("quality-chart");
    if (!container) return;

    const qualityMetrics = features.map((feature) => {
      const totalValues = feature.values.length;
      const missingValues = feature.values.filter(
        (v) => v === null || v === undefined || v === "",
      ).length;
      const completeness = ((totalValues - missingValues) / totalValues) * 100;

      return {
        name: feature.name,
        completeness: completeness,
        uniqueness: this.calculateUniqueness(feature.values),
        validity: this.calculateValidity(feature.values, feature.type),
      };
    });

    const chart = echarts.init(container);
    const option = {
      title: {
        text: "Data Quality Metrics",
        textStyle: { fontSize: 14 },
      },
      tooltip: {
        trigger: "axis",
        axisPointer: { type: "shadow" },
      },
      legend: {
        data: ["Completeness", "Uniqueness", "Validity"],
        bottom: 0,
      },
      xAxis: {
        type: "category",
        data: qualityMetrics.map((m) => m.name),
        axisLabel: {
          rotate: 45,
        },
      },
      yAxis: {
        type: "value",
        name: "Percentage",
        max: 100,
      },
      series: [
        {
          name: "Completeness",
          type: "bar",
          data: qualityMetrics.map((m) => m.completeness),
          itemStyle: { color: "#10b981" },
        },
        {
          name: "Uniqueness",
          type: "bar",
          data: qualityMetrics.map((m) => m.uniqueness),
          itemStyle: { color: "#3b82f6" },
        },
        {
          name: "Validity",
          type: "bar",
          data: qualityMetrics.map((m) => m.validity),
          itemStyle: { color: "#f59e0b" },
        },
      ],
    };

    chart.setOption(option);
    this.charts.set("quality-chart", chart);
  }

  /**
   * Render anomaly distribution chart
   */
  async renderAnomalyDistributionChart() {
    const container = document.getElementById("anomaly-distribution-chart");
    if (!container || !this.currentResult) return;

    const result = this.currentResult;
    const totalSamples = result.statistics?.totalSamples || 0;
    const totalAnomalies = result.statistics?.totalAnomalies || 0;
    const normalSamples = totalSamples - totalAnomalies;

    const chart = echarts.init(container);
    const option = {
      title: {
        text: "Normal vs Anomalous Data",
        textStyle: { fontSize: 14 },
      },
      tooltip: {
        trigger: "item",
        formatter: "{a} <br/>{b}: {c} ({d}%)",
      },
      series: [
        {
          name: "Data Distribution",
          type: "pie",
          radius: ["40%", "70%"],
          data: [
            {
              value: normalSamples,
              name: "Normal",
              itemStyle: { color: "#10b981" },
            },
            {
              value: totalAnomalies,
              name: "Anomalous",
              itemStyle: { color: "#ef4444" },
            },
          ],
          emphasis: {
            itemStyle: {
              shadowBlur: 10,
              shadowOffsetX: 0,
              shadowColor: "rgba(0, 0, 0, 0.5)",
            },
          },
        },
      ],
    };

    chart.setOption(option);
    this.charts.set("anomaly-distribution-chart", chart);
  }

  /**
   * Render score distribution chart
   */
  async renderScoreDistributionChart() {
    const container = document.getElementById("score-distribution-chart");
    if (!container || !this.currentResult) return;

    const scores = this.currentResult.scores || [];
    if (!scores.length) return;

    const bins = this.calculateHistogramBins(scores, 30);
    const chart = echarts.init(container);

    const option = {
      title: {
        text: "Anomaly Score Distribution",
        textStyle: { fontSize: 14 },
      },
      tooltip: {
        trigger: "axis",
        axisPointer: { type: "shadow" },
      },
      xAxis: {
        type: "category",
        data: bins.map((bin) => bin.range),
        name: "Anomaly Score",
      },
      yAxis: {
        type: "value",
        name: "Frequency",
      },
      series: [
        {
          name: "Frequency",
          type: "bar",
          data: bins.map((bin) => bin.count),
          itemStyle: {
            color: "#8b5cf6",
          },
        },
      ],
    };

    chart.setOption(option);
    this.charts.set("score-distribution-chart", chart);
  }

  /**
   * Render anomaly scatter plot
   */
  async renderAnomalyScatterPlot() {
    const container = document.getElementById("anomaly-scatter-chart");
    if (!container || !this.currentResult || !this.currentDataset) return;

    const data = this.currentDataset.data;
    const anomalies = this.currentResult.anomalies || [];
    const scores = this.currentResult.scores || [];

    // Use first two numeric features for scatter plot
    const features = this.extractFeatures(data);
    const numericFeatures = features
      .filter((f) => f.type === "numeric")
      .slice(0, 2);

    if (numericFeatures.length < 2) return;

    const normalData = [];
    const anomalyData = [];
    const anomalyIndices = new Set(anomalies.map((a) => a.index));

    data.forEach((row, index) => {
      const point = [
        row[numericFeatures[0].name] || 0,
        row[numericFeatures[1].name] || 0,
        scores[index] || 0,
      ];

      if (anomalyIndices.has(index)) {
        anomalyData.push(point);
      } else {
        normalData.push(point);
      }
    });

    const chart = echarts.init(container);
    const option = {
      title: {
        text: `${numericFeatures[0].name} vs ${numericFeatures[1].name}`,
        textStyle: { fontSize: 14 },
      },
      tooltip: {
        trigger: "item",
        formatter: (params) => {
          const [x, y, score] = params.data;
          return `${params.seriesName}<br/>
                  ${numericFeatures[0].name}: ${x.toFixed(3)}<br/>
                  ${numericFeatures[1].name}: ${y.toFixed(3)}<br/>
                  Score: ${score.toFixed(3)}`;
        },
      },
      legend: {
        data: ["Normal", "Anomaly"],
        bottom: 0,
      },
      xAxis: {
        type: "value",
        name: numericFeatures[0].name,
        scale: true,
      },
      yAxis: {
        type: "value",
        name: numericFeatures[1].name,
        scale: true,
      },
      series: [
        {
          name: "Normal",
          type: "scatter",
          data: normalData,
          itemStyle: {
            color: "#10b981",
            opacity: 0.7,
          },
          symbolSize: 6,
        },
        {
          name: "Anomaly",
          type: "scatter",
          data: anomalyData,
          itemStyle: {
            color: "#ef4444",
            opacity: 0.9,
          },
          symbolSize: 10,
        },
      ],
    };

    chart.setOption(option);
    this.charts.set("anomaly-scatter-chart", chart);
  }

  /**
   * Render detection summary
   */
  renderDetectionSummary() {
    const container = document.getElementById("detection-summary");
    if (!container || !this.currentResult) return;

    const result = this.currentResult;
    const stats = result.statistics || {};

    container.innerHTML = `
      <div class="space-y-4">
        <div class="grid grid-cols-2 gap-4">
          <div class="bg-gray-50 p-3 rounded">
            <div class="text-sm text-gray-600">Algorithm</div>
            <div class="font-semibold">${result.algorithmId || "Unknown"}</div>
          </div>
          <div class="bg-gray-50 p-3 rounded">
            <div class="text-sm text-gray-600">Processing Time</div>
            <div class="font-semibold">${result.processingTimeMs || 0}ms</div>
          </div>
          <div class="bg-gray-50 p-3 rounded">
            <div class="text-sm text-gray-600">Total Samples</div>
            <div class="font-semibold">${stats.totalSamples || 0}</div>
          </div>
          <div class="bg-gray-50 p-3 rounded">
            <div class="text-sm text-gray-600">Anomaly Rate</div>
            <div class="font-semibold">${((stats.anomalyRate || 0) * 100).toFixed(2)}%</div>
          </div>
        </div>

        <div class="bg-blue-50 p-4 rounded border border-blue-200">
          <h4 class="font-medium text-blue-900 mb-2">Detection Parameters</h4>
          <div class="text-sm text-blue-800">
            ${Object.entries(result.parameters || {})
              .map(
                ([key, value]) =>
                  `<div><strong>${key}:</strong> ${value}</div>`,
              )
              .join("")}
          </div>
        </div>

        <div class="flex gap-2">
          <button class="btn-base btn-sm btn-primary export-viz" data-format="png">
            Export as PNG
          </button>
          <button class="btn-base btn-sm btn-secondary export-viz" data-format="pdf">
            Export as PDF
          </button>
        </div>
      </div>
    `;
  }

  // Helper methods...

  /**
   * Extract features from dataset
   */
  extractFeatures(data) {
    if (!data || !data.length) return [];

    const features = [];
    const firstRow = data[0];

    Object.keys(firstRow).forEach((key) => {
      const values = data.map((row) => row[key]);
      const type = this.inferDataType(values);

      features.push({
        name: key,
        type,
        values: values.filter((v) => v !== null && v !== undefined),
      });
    });

    return features;
  }

  /**
   * Infer data type from values
   */
  inferDataType(values) {
    const nonNullValues = values.filter(
      (v) => v !== null && v !== undefined && v !== "",
    );
    if (!nonNullValues.length) return "unknown";

    const numericCount = nonNullValues.filter(
      (v) => !isNaN(parseFloat(v)),
    ).length;
    const ratio = numericCount / nonNullValues.length;

    return ratio > 0.8 ? "numeric" : "categorical";
  }

  /**
   * Calculate histogram bins
   */
  calculateHistogramBins(values, numBins) {
    const min = Math.min(...values);
    const max = Math.max(...values);
    const binWidth = (max - min) / numBins;

    const bins = [];
    for (let i = 0; i < numBins; i++) {
      const start = min + i * binWidth;
      const end = min + (i + 1) * binWidth;
      const count = values.filter(
        (v) => v >= start && (i === numBins - 1 ? v <= end : v < end),
      ).length;

      bins.push({
        range: `${start.toFixed(2)}-${end.toFixed(2)}`,
        count,
      });
    }

    return bins;
  }

  /**
   * Calculate correlation matrix
   */
  calculateCorrelationMatrix(features) {
    const matrix = [];

    for (let i = 0; i < features.length; i++) {
      for (let j = 0; j < features.length; j++) {
        const correlation = this.calculateCorrelation(
          features[i].values,
          features[j].values,
        );
        matrix.push([i, j, correlation]);
      }
    }

    return matrix;
  }

  /**
   * Calculate Pearson correlation coefficient
   */
  calculateCorrelation(x, y) {
    const n = Math.min(x.length, y.length);
    if (n < 2) return 0;

    const sumX = x.slice(0, n).reduce((a, b) => a + b, 0);
    const sumY = y.slice(0, n).reduce((a, b) => a + b, 0);
    const sumXY = x.slice(0, n).reduce((sum, xi, i) => sum + xi * y[i], 0);
    const sumX2 = x.slice(0, n).reduce((sum, xi) => sum + xi * xi, 0);
    const sumY2 = y.slice(0, n).reduce((sum, yi) => sum + yi * yi, 0);

    const numerator = n * sumXY - sumX * sumY;
    const denominator = Math.sqrt(
      (n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY),
    );

    return denominator === 0 ? 0 : numerator / denominator;
  }

  /**
   * Calculate basic statistics
   */
  calculateBasicStats(values) {
    const sorted = [...values].sort((a, b) => a - b);
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const variance =
      values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) /
      values.length;

    return {
      mean,
      std: Math.sqrt(variance),
      min: Math.min(...values),
      max: Math.max(...values),
      median: sorted[Math.floor(sorted.length / 2)],
    };
  }

  /**
   * Calculate data uniqueness percentage
   */
  calculateUniqueness(values) {
    const unique = new Set(values).size;
    return (unique / values.length) * 100;
  }

  /**
   * Calculate data validity percentage
   */
  calculateValidity(values, type) {
    if (type === "numeric") {
      const validCount = values.filter((v) => !isNaN(parseFloat(v))).length;
      return (validCount / values.length) * 100;
    }
    // For categorical, assume all non-null values are valid
    const validCount = values.filter(
      (v) => v !== null && v !== undefined && v !== "",
    ).length;
    return (validCount / values.length) * 100;
  }

  /**
   * Update result selector based on current dataset
   */
  updateResultSelector() {
    const selector = document.querySelector(".result-selector");
    if (!selector || !this.currentDataset) return;

    const relevantResults = Array.from(this.results.values()).filter(
      (result) => result.datasetId === this.currentDataset.id,
    );

    selector.innerHTML = `
      <option value="">Select a result...</option>
      ${relevantResults
        .map(
          (result) => `
        <option value="${result.id}">
          ${result.algorithmId} - ${new Date(result.timestamp).toLocaleDateString()}
        </option>
      `,
        )
        .join("")}
    `;
  }

  /**
   * Change visualization type
   */
  changeVisualizationType(type) {
    // Implementation for different visualization types
    console.log("Changing visualization type to:", type);
  }

  /**
   * Export visualization
   */
  exportVisualization(format) {
    // Implementation for exporting visualizations
    console.log("Exporting visualization as:", format);
  }

  /**
   * Clear all charts
   */
  clearCharts() {
    this.charts.forEach((chart) => {
      chart.dispose();
    });
    this.charts.clear();
  }

  /**
   * Get available datasets for selection
   */
  getAvailableDatasets() {
    return Array.from(this.datasets.values());
  }

  /**
   * Get available results for selection
   */
  getAvailableResults() {
    return Array.from(this.results.values());
  }
}

// Initialize and expose globally
if (typeof window !== "undefined") {
  window.OfflineVisualizer = new OfflineVisualizer();
}
\n\n// real-time-dashboard.js\n/**
 * Real-Time Dashboard Component
 *
 * Advanced dashboard for live anomaly detection monitoring with
 * real-time charts, alerts, and system metrics visualization
 */

import { AnomalyWebSocketClient } from "../services/websocket-service.js";
import { AnomalyTimeline } from "./anomaly-timeline.js";
import { DashboardLayout } from "./dashboard-layout.js";

export class RealTimeDashboard {
  constructor(container, options = {}) {
    this.container =
      typeof container === "string"
        ? document.querySelector(container)
        : container;
    this.options = {
      enableRealTime: true,
      updateInterval: 1000,
      maxDataPoints: 1000,
      alertThreshold: 0.7,
      enableSound: false,
      enableNotifications: true,
      autoScroll: true,
      theme: "light",
      ...options,
    };

    this.wsClient = null;
    this.dashboardLayout = null;
    this.timeline = null;
    this.isActive = false;
    this.lastUpdate = null;

    this.data = {
      realTimeData: [],
      alerts: [],
      systemMetrics: {},
      connectionStatus: "disconnected",
      statistics: {
        totalAnomalies: 0,
        criticalAlerts: 0,
        avgScore: 0,
        detectionRate: 0,
      },
    };

    this.charts = new Map();
    this.widgets = new Map();
    this.updateTimers = new Map();

    this.init();
  }

  init() {
    this.setupContainer();
    this.initializeWebSocket();
    this.createDashboardLayout();
    this.bindEvents();

    if (this.options.enableRealTime) {
      this.start();
    }
  }

  setupContainer() {
    this.container.classList.add("real-time-dashboard");
    this.container.innerHTML = "";

    // Add dashboard header
    this.header = document.createElement("div");
    this.header.className = "dashboard-header";
    this.header.innerHTML = `
            <div class="dashboard-title">
                <h2>Real-Time Anomaly Detection</h2>
                <div class="connection-status" id="connection-status">
                    <span class="status-indicator"></span>
                    <span class="status-text">Connecting...</span>
                </div>
            </div>
            <div class="dashboard-controls">
                <button class="btn btn-primary" id="start-btn">Start Monitoring</button>
                <button class="btn btn-secondary" id="pause-btn" disabled>Pause</button>
                <button class="btn btn-ghost" id="settings-btn">Settings</button>
                <button class="btn btn-ghost" id="fullscreen-btn">⛶</button>
            </div>
        `;

    // Dashboard content area
    this.content = document.createElement("div");
    this.content.className = "dashboard-content";

    this.container.appendChild(this.header);
    this.container.appendChild(this.content);
  }

  initializeWebSocket() {
    this.wsClient = new AnomalyWebSocketClient({
      enableLogging: true,
      autoConnect: false,
      enableRealTimeDetection: true,
      enableAlerts: true,
      enableMetrics: true,
    });

    // Set up WebSocket event handlers
    this.wsClient.on("connected", () => {
      this.updateConnectionStatus("connected");
      this.onWebSocketConnected();
    });

    this.wsClient.on("disconnected", () => {
      this.updateConnectionStatus("disconnected");
      this.onWebSocketDisconnected();
    });

    this.wsClient.on("anomaly_detected", (anomaly) => {
      this.onAnomalyDetected(anomaly);
    });

    this.wsClient.on("real_time_data", (data) => {
      this.onRealTimeData(data);
    });

    this.wsClient.on("alert", (alert) => {
      this.onAlert(alert);
    });

    this.wsClient.on("system_metrics", (metrics) => {
      this.onSystemMetrics(metrics);
    });
  }

  createDashboardLayout() {
    this.dashboardLayout = new DashboardLayout(this.content, {
      columns: 12,
      rowHeight: 80,
      margin: [10, 10],
      isDraggable: true,
      isResizable: true,
    });

    // Add default widgets
    this.addWidget({
      id: "connection-info",
      title: "Connection Status",
      type: "status",
      x: 0,
      y: 0,
      w: 3,
      h: 2,
      component: () => this.createConnectionWidget(),
    });

    this.addWidget({
      id: "live-timeline",
      title: "Live Anomaly Timeline",
      type: "chart",
      x: 3,
      y: 0,
      w: 9,
      h: 6,
      component: () => this.createTimelineWidget(),
    });

    this.addWidget({
      id: "statistics",
      title: "Detection Statistics",
      type: "metrics",
      x: 0,
      y: 2,
      w: 3,
      h: 4,
      component: () => this.createStatisticsWidget(),
    });

    this.addWidget({
      id: "live-alerts",
      title: "Live Alerts",
      type: "alerts",
      x: 0,
      y: 6,
      w: 6,
      h: 4,
      component: () => this.createAlertsWidget(),
    });

    this.addWidget({
      id: "system-metrics",
      title: "System Metrics",
      type: "metrics",
      x: 6,
      y: 6,
      w: 6,
      h: 4,
      component: () => this.createSystemMetricsWidget(),
    });

    this.addWidget({
      id: "data-stream",
      title: "Data Stream",
      type: "stream",
      x: 0,
      y: 10,
      w: 12,
      h: 3,
      component: () => this.createDataStreamWidget(),
    });
  }

  addWidget(config) {
    this.dashboardLayout.addWidget(config);
    return config.id;
  }

  createConnectionWidget() {
    const widget = document.createElement("div");
    widget.className = "connection-widget";

    const update = () => {
      const info = this.wsClient.getConnectionInfo();
      widget.innerHTML = `
                <div class="connection-info">
                    <div class="status-row">
                        <span class="label">Status:</span>
                        <span class="value status-${info.state}">${info.state}</span>
                    </div>
                    <div class="status-row">
                        <span class="label">Connection ID:</span>
                        <span class="value">${info.connectionId || "N/A"}</span>
                    </div>
                    <div class="status-row">
                        <span class="label">Subscriptions:</span>
                        <span class="value">${info.subscriptions.length}</span>
                    </div>
                    <div class="status-row">
                        <span class="label">Queued:</span>
                        <span class="value">${info.queuedMessages}</span>
                    </div>
                    ${
                      info.state === "connecting"
                        ? `
                        <div class="status-row">
                            <span class="label">Attempts:</span>
                            <span class="value">${info.reconnectAttempts}</span>
                        </div>
                    `
                        : ""
                    }
                </div>
            `;
    };

    update();
    this.updateTimers.set("connection", setInterval(update, 2000));

    return widget;
  }

  createTimelineWidget() {
    const widget = document.createElement("div");
    widget.className = "timeline-widget";
    widget.style.height = "100%";

    // Initialize timeline component
    this.timeline = new AnomalyTimeline(widget, {
      width: widget.clientWidth || 800,
      height: 400,
      enableRealTime: true,
      enableZoom: true,
      enableBrush: true,
      maxDataPoints: this.options.maxDataPoints,
    });

    return widget;
  }

  createStatisticsWidget() {
    const widget = document.createElement("div");
    widget.className = "statistics-widget";

    const update = () => {
      const stats = this.data.statistics;
      widget.innerHTML = `
                <div class="stats-grid">
                    <div class="stat-item">
                        <div class="stat-value">${stats.totalAnomalies}</div>
                        <div class="stat-label">Total Anomalies</div>
                    </div>
                    <div class="stat-item critical">
                        <div class="stat-value">${stats.criticalAlerts}</div>
                        <div class="stat-label">Critical Alerts</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">${stats.avgScore.toFixed(3)}</div>
                        <div class="stat-label">Avg Score</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">${stats.detectionRate.toFixed(1)}/min</div>
                        <div class="stat-label">Detection Rate</div>
                    </div>
                </div>
            `;
    };

    update();
    this.updateTimers.set("statistics", setInterval(update, 1000));

    return widget;
  }

  createAlertsWidget() {
    const widget = document.createElement("div");
    widget.className = "alerts-widget";

    widget.innerHTML = `
            <div class="alerts-header">
                <span class="alerts-count">0 active alerts</span>
                <button class="btn btn-sm btn-ghost" id="clear-alerts">Clear All</button>
            </div>
            <div class="alerts-list" id="alerts-list"></div>
        `;

    widget.querySelector("#clear-alerts").onclick = () => {
      this.clearAlerts();
    };

    const update = () => {
      const alertsList = widget.querySelector("#alerts-list");
      const alertsCount = widget.querySelector(".alerts-count");
      const unacknowledged = this.wsClient.getUnacknowledgedAlerts();

      alertsCount.textContent = `${unacknowledged.length} active alerts`;

      alertsList.innerHTML = unacknowledged
        .slice(0, 10)
        .map(
          (alert) => `
                <div class="alert-item severity-${alert.severity}" data-alert-id="${alert.id}">
                    <div class="alert-content">
                        <div class="alert-message">${alert.message}</div>
                        <div class="alert-time">${new Date(alert.timestamp).toLocaleTimeString()}</div>
                    </div>
                    <button class="alert-dismiss" onclick="dashboard.acknowledgeAlert('${alert.id}')">×</button>
                </div>
            `,
        )
        .join("");
    };

    update();
    this.updateTimers.set("alerts", setInterval(update, 500));

    return widget;
  }

  createSystemMetricsWidget() {
    const widget = document.createElement("div");
    widget.className = "system-metrics-widget";

    const update = () => {
      const metrics = this.data.systemMetrics;
      widget.innerHTML = `
                <div class="metrics-grid">
                    <div class="metric-item">
                        <div class="metric-label">CPU Usage</div>
                        <div class="metric-value">${(metrics.cpu * 100).toFixed(1)}%</div>
                        <div class="metric-bar">
                            <div class="metric-fill" style="width: ${metrics.cpu * 100}%"></div>
                        </div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-label">Memory Usage</div>
                        <div class="metric-value">${(metrics.memory * 100).toFixed(1)}%</div>
                        <div class="metric-bar">
                            <div class="metric-fill" style="width: ${metrics.memory * 100}%"></div>
                        </div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-label">Active Models</div>
                        <div class="metric-value">${metrics.activeModels || 0}</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-label">Queue Size</div>
                        <div class="metric-value">${metrics.queueSize || 0}</div>
                    </div>
                </div>
            `;
    };

    update();
    this.updateTimers.set("system-metrics", setInterval(update, 2000));

    return widget;
  }

  createDataStreamWidget() {
    const widget = document.createElement("div");
    widget.className = "data-stream-widget";

    widget.innerHTML = `
            <div class="stream-header">
                <span>Live Data Stream</span>
                <div class="stream-controls">
                    <button class="btn btn-sm" id="pause-stream">Pause</button>
                    <button class="btn btn-sm" id="clear-stream">Clear</button>
                </div>
            </div>
            <div class="stream-content" id="stream-content"></div>
        `;

    const streamContent = widget.querySelector("#stream-content");
    let isPaused = false;

    widget.querySelector("#pause-stream").onclick = (e) => {
      isPaused = !isPaused;
      e.target.textContent = isPaused ? "Resume" : "Pause";
    };

    widget.querySelector("#clear-stream").onclick = () => {
      streamContent.innerHTML = "";
    };

    const update = () => {
      if (isPaused) return;

      const recentData = this.data.realTimeData.slice(-50);
      streamContent.innerHTML = recentData
        .map(
          (item) => `
                <div class="stream-item">
                    <span class="stream-time">${new Date(item.timestamp).toLocaleTimeString()}</span>
                    <span class="stream-data">${JSON.stringify(item.data).substring(0, 100)}</span>
                    ${item.anomaly ? '<span class="stream-anomaly">⚠️</span>' : ""}
                </div>
            `,
        )
        .join("");

      if (this.options.autoScroll) {
        streamContent.scrollTop = streamContent.scrollHeight;
      }
    };

    this.updateTimers.set("stream", setInterval(update, 1000));

    return widget;
  }

  bindEvents() {
    // Header controls
    this.header.querySelector("#start-btn").onclick = () => this.start();
    this.header.querySelector("#pause-btn").onclick = () => this.pause();
    this.header.querySelector("#settings-btn").onclick = () =>
      this.showSettings();
    this.header.querySelector("#fullscreen-btn").onclick = () =>
      this.toggleFullscreen();

    // Global keyboard shortcuts
    document.addEventListener("keydown", (e) => {
      if (e.ctrlKey || e.metaKey) {
        switch (e.key) {
          case "p":
            e.preventDefault();
            this.isActive ? this.pause() : this.start();
            break;
          case "r":
            e.preventDefault();
            this.refresh();
            break;
          case "f":
            e.preventDefault();
            this.toggleFullscreen();
            break;
        }
      }
    });
  }

  // Show notification for real-time alerts
  onAlert(alert) {
    this.showNotification(`Alert: ${alert.message}`, alert.severity === 'critical' ? 'error' : 'warning');
    // Update state and UI elements...
  }

  // WebSocket event handlers
  onWebSocketConnected() {
    console.log("Real-time dashboard connected");
    this.updateStatistics();
  }

  onWebSocketDisconnected() {
    console.log("Real-time dashboard disconnected");
  }

  onAnomalyDetected(anomaly) {
    // Add to timeline
    if (this.timeline) {
      this.timeline.addRealTimeData([anomaly]);
    }

    // Update statistics
    this.data.statistics.totalAnomalies++;
    if (anomaly.severity === "critical") {
      this.data.statistics.criticalAlerts++;
    }

    // Calculate rolling average score
    const recentScores = this.data.realTimeData
      .filter((item) => item.anomaly)
      .slice(-100)
      .map((item) => item.anomaly.score);

    if (recentScores.length > 0) {
      this.data.statistics.avgScore =
        recentScores.reduce((a, b) => a + b) / recentScores.length;
    }

    // Play notification sound
    if (this.options.enableSound && anomaly.severity === "critical") {
      this.playNotificationSound();
    }

    // Show browser notification
    if (this.options.enableNotifications && anomaly.severity === "critical") {
      this.showBrowserNotification(anomaly);
    }
  }

  onRealTimeData(data) {
    this.data.realTimeData.push({
      ...data,
      timestamp: Date.now(),
    });

    // Maintain buffer size
    if (this.data.realTimeData.length > this.options.maxDataPoints) {
      this.data.realTimeData.shift();
    }

    this.lastUpdate = Date.now();
    this.updateDetectionRate();
  }

  onAlert(alert) {
    this.data.alerts.unshift(alert);

    // Maintain alerts buffer
    if (this.data.alerts.length > 1000) {
      this.data.alerts.pop();
    }

    // Update critical alerts count
    if (alert.severity === "critical") {
      this.data.statistics.criticalAlerts++;
    }
  }

  onSystemMetrics(metrics) {
    this.data.systemMetrics = {
      ...this.data.systemMetrics,
      ...metrics,
      lastUpdate: Date.now(),
    };
  }

  // Control methods
  start() {
    if (this.isActive) return;

    this.isActive = true;
    this.wsClient.connect();

    this.header.querySelector("#start-btn").disabled = true;
    this.header.querySelector("#pause-btn").disabled = false;

    this.updateConnectionStatus("connecting");
  }

  pause() {
    if (!this.isActive) return;

    this.isActive = false;
    this.wsClient.disconnect();

    this.header.querySelector("#start-btn").disabled = false;
    this.header.querySelector("#pause-btn").disabled = true;

    this.updateConnectionStatus("paused");
  }

  refresh() {
    this.data.realTimeData = [];
    this.data.alerts = [];
    this.data.statistics = {
      totalAnomalies: 0,
      criticalAlerts: 0,
      avgScore: 0,
      detectionRate: 0,
    };

    if (this.timeline) {
      this.timeline.setData([]);
    }
  }

  showSettings() {
    // Implementation for settings modal
    console.log("Settings modal would open here");
  }

  toggleFullscreen() {
    if (document.fullscreenElement) {
      document.exitFullscreen();
    } else {
      this.container.requestFullscreen();
    }
  }

  // Utility methods
  updateConnectionStatus(status) {
    this.data.connectionStatus = status;

    const statusElement = this.header.querySelector("#connection-status");
    const indicator = statusElement.querySelector(".status-indicator");
    const text = statusElement.querySelector(".status-text");

    indicator.className = `status-indicator status-${status}`;
    text.textContent = status.charAt(0).toUpperCase() + status.slice(1);
  }

  updateStatistics() {
    // Calculate detection rate (detections per minute)
    const now = Date.now();
    const oneMinuteAgo = now - 60000;
    const recentDetections = this.data.realTimeData.filter(
      (item) => item.timestamp > oneMinuteAgo,
    ).length;

    this.data.statistics.detectionRate = recentDetections;
  }

  updateDetectionRate() {
    // Update detection rate every 10 seconds
    if (!this.detectionRateTimer) {
      this.detectionRateTimer = setInterval(() => {
        this.updateStatistics();
      }, 10000);
    }
  }

  acknowledgeAlert(alertId) {
    this.wsClient.acknowledgeAlert(alertId);

    // Remove from UI
    const alertElement = document.querySelector(`[data-alert-id="${alertId}"]`);
    if (alertElement) {
      alertElement.remove();
    }
  }

  clearAlerts() {
    const unacknowledged = this.wsClient.getUnacknowledgedAlerts();
    unacknowledged.forEach((alert) => {
      this.wsClient.acknowledgeAlert(alert.id);
    });
  }

  playNotificationSound() {
    if ("AudioContext" in window) {
      const audioContext = new AudioContext();
      const oscillator = audioContext.createOscillator();
      const gainNode = audioContext.createGain();

      oscillator.connect(gainNode);
      gainNode.connect(audioContext.destination);

      oscillator.frequency.value = 800;
      oscillator.type = "square";
      gainNode.gain.value = 0.1;

      oscillator.start();
      oscillator.stop(audioContext.currentTime + 0.2);
    }
  }

  showBrowserNotification(anomaly) {
    if ("Notification" in window && Notification.permission === "granted") {
      new Notification("Critical Anomaly Detected", {
        body: `Anomaly score: ${anomaly.score.toFixed(3)}`,
        icon: "/static/icons/alert.png",
        tag: "anomaly-alert",
      });
    }
  }

  // Data export methods
  exportData(format = "json") {
    const data = {
      timestamp: new Date().toISOString(),
      statistics: this.data.statistics,
      recentData: this.data.realTimeData.slice(-1000),
      alerts: this.data.alerts.slice(0, 100),
      systemMetrics: this.data.systemMetrics,
    };

    switch (format) {
      case "json":
        return JSON.stringify(data, null, 2);
      case "csv":
        return this.convertToCSV(data.recentData);
      default:
        return data;
    }
  }

  convertToCSV(data) {
    if (data.length === 0) return "";

    const headers = Object.keys(data[0]);
    const rows = data.map((item) =>
      headers.map((header) => JSON.stringify(item[header] || "")).join(","),
    );

    return [headers.join(","), ...rows].join("\n");
  }

  // Cleanup
  destroy() {
    // Clear all timers
    this.updateTimers.forEach((timer) => clearInterval(timer));
    this.updateTimers.clear();

    if (this.detectionRateTimer) {
      clearInterval(this.detectionRateTimer);
    }

    // Disconnect WebSocket
    if (this.wsClient) {
      this.wsClient.destroy();
    }

    // Destroy components
    if (this.timeline) {
      this.timeline.destroy();
    }

    if (this.dashboardLayout) {
      this.dashboardLayout.destroy();
    }

    // Clear data
    this.data = null;
    this.charts.clear();
    this.widgets.clear();
  }
}

// Make globally available for onclick handlers
if (typeof window !== "undefined") {
  window.dashboard = null;
}

export default RealTimeDashboard;
\n\n// realtime-chart-optimizer.js\n/**
 * Real-Time Chart Optimization
 * 60 FPS chart rendering with efficient data buffering and smooth animations
 * Performance-optimized rendering engine for live data visualization
 */

/**
 * Frame Rate Controller
 * Manages frame rate optimization and smooth animations
 */
class FrameRateController {
  constructor(targetFPS = 60) {
    this.targetFPS = targetFPS;
    this.frameInterval = 1000 / targetFPS;
    this.lastFrameTime = 0;
    this.animationFrame = null;
    this.updateCallbacks = new Set();
    this.isRunning = false;

    // Performance monitoring
    this.frameCount = 0;
    this.actualFPS = 0;
    this.frameTimeHistory = [];
    this.maxHistorySize = 100;
  }

  start() {
    if (this.isRunning) return;

    this.isRunning = true;
    this.lastFrameTime = performance.now();
    this.frameLoop();
  }

  stop() {
    this.isRunning = false;
    if (this.animationFrame) {
      cancelAnimationFrame(this.animationFrame);
      this.animationFrame = null;
    }
  }

  frameLoop() {
    if (!this.isRunning) return;

    const currentTime = performance.now();
    const deltaTime = currentTime - this.lastFrameTime;

    if (deltaTime >= this.frameInterval) {
      // Update performance metrics
      this.frameCount++;
      this.frameTimeHistory.push(deltaTime);
      if (this.frameTimeHistory.length > this.maxHistorySize) {
        this.frameTimeHistory.shift();
      }

      // Calculate actual FPS
      this.actualFPS =
        1000 /
        (this.frameTimeHistory.reduce((a, b) => a + b, 0) /
          this.frameTimeHistory.length);

      // Execute update callbacks
      this.updateCallbacks.forEach((callback) => {
        try {
          callback(deltaTime, currentTime);
        } catch (error) {
          console.error("Frame update callback error:", error);
        }
      });

      this.lastFrameTime = currentTime - (deltaTime % this.frameInterval);
    }

    this.animationFrame = requestAnimationFrame(() => this.frameLoop());
  }

  addUpdateCallback(callback) {
    this.updateCallbacks.add(callback);
    return () => this.updateCallbacks.delete(callback);
  }

  getPerformanceMetrics() {
    return {
      targetFPS: this.targetFPS,
      actualFPS: Math.round(this.actualFPS),
      frameCount: this.frameCount,
      averageFrameTime:
        this.frameTimeHistory.reduce((a, b) => a + b, 0) /
        this.frameTimeHistory.length,
      isRunning: this.isRunning,
    };
  }
}

/**
 * Data Buffer Manager
 * Efficient circular buffer for real-time data with automatic memory management
 */
class DataBufferManager {
  constructor(maxSize = 10000, compressionThreshold = 0.8) {
    this.maxSize = maxSize;
    this.compressionThreshold = compressionThreshold;
    this.buffer = [];
    this.writeIndex = 0;
    this.readIndex = 0;
    this.size = 0;
    this.isCircular = false;

    // Compression settings
    this.compressionRatio = 0.5; // Compress to 50% when threshold reached
    this.lastCompressionTime = 0;

    // Performance tracking
    this.totalWrites = 0;
    this.totalReads = 0;
    this.compressionCount = 0;
  }

  add(item) {
    const timestamp = performance.now();
    const wrappedItem = {
      data: item,
      timestamp,
      id: this.totalWrites++,
    };

    if (this.size < this.maxSize) {
      this.buffer[this.writeIndex] = wrappedItem;
      this.size++;
    } else {
      // Buffer is full, overwrite oldest item
      this.buffer[this.writeIndex] = wrappedItem;
      this.readIndex = (this.readIndex + 1) % this.maxSize;
      this.isCircular = true;
    }

    this.writeIndex = (this.writeIndex + 1) % this.maxSize;

    // Check if compression is needed
    if (this.size / this.maxSize > this.compressionThreshold) {
      this.compress();
    }

    return wrappedItem.id;
  }

  getLast(count = 1) {
    if (count <= 0 || this.size === 0) return [];

    const result = [];
    const actualCount = Math.min(count, this.size);

    for (let i = 0; i < actualCount; i++) {
      const index = this.isCircular
        ? (this.writeIndex - 1 - i + this.maxSize) % this.maxSize
        : this.writeIndex - 1 - i;

      if (index >= 0 && this.buffer[index]) {
        result.unshift(this.buffer[index]);
      }
    }

    this.totalReads += result.length;
    return result;
  }

  getRange(startTime, endTime) {
    const result = [];
    const items = this.getLast(this.size);

    for (const item of items) {
      if (item.timestamp >= startTime && item.timestamp <= endTime) {
        result.push(item);
      }
    }

    return result;
  }

  compress() {
    if (performance.now() - this.lastCompressionTime < 1000) {
      return; // Avoid frequent compression
    }

    const targetSize = Math.floor(this.maxSize * this.compressionRatio);
    const items = this.getLast(this.size);

    // Simple compression: keep every nth item
    const compressionFactor = Math.ceil(items.length / targetSize);
    const compressed = items.filter(
      (_, index) => index % compressionFactor === 0,
    );

    // Rebuild buffer
    this.buffer = new Array(this.maxSize);
    this.writeIndex = 0;
    this.readIndex = 0;
    this.size = 0;
    this.isCircular = false;

    // Add compressed items back
    compressed.forEach((item) => this.add(item.data));

    this.compressionCount++;
    this.lastCompressionTime = performance.now();
  }

  clear() {
    this.buffer = [];
    this.writeIndex = 0;
    this.readIndex = 0;
    this.size = 0;
    this.isCircular = false;
  }

  getStats() {
    return {
      size: this.size,
      maxSize: this.maxSize,
      utilization: ((this.size / this.maxSize) * 100).toFixed(1) + "%",
      totalWrites: this.totalWrites,
      totalReads: this.totalReads,
      compressionCount: this.compressionCount,
      isCircular: this.isCircular,
    };
  }
}

/**
 * Animation Manager
 * Smooth animations and transitions for real-time charts
 */
class AnimationManager {
  constructor() {
    this.animations = new Map();
    this.easingFunctions = {
      linear: (t) => t,
      easeInOut: (t) => (t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t),
      easeOut: (t) => 1 - Math.pow(1 - t, 3),
      easeIn: (t) => t * t * t,
      elastic: (t) =>
        t === 0
          ? 0
          : t === 1
            ? 1
            : -Math.pow(2, 10 * (t - 1)) * Math.sin((t - 1.1) * 5 * Math.PI),
    };
  }

  animate(
    id,
    fromValue,
    toValue,
    duration,
    easing = "easeInOut",
    onUpdate,
    onComplete,
  ) {
    const animation = {
      id,
      fromValue,
      toValue,
      duration,
      easing: this.easingFunctions[easing] || this.easingFunctions.easeInOut,
      onUpdate,
      onComplete,
      startTime: performance.now(),
      completed: false,
    };

    this.animations.set(id, animation);
    return id;
  }

  update(currentTime) {
    for (const [id, animation] of this.animations) {
      const elapsed = currentTime - animation.startTime;
      const progress = Math.min(elapsed / animation.duration, 1);

      if (progress >= 1) {
        // Animation complete
        const finalValue = animation.toValue;
        animation.onUpdate?.(finalValue, 1);
        animation.onComplete?.(finalValue);
        this.animations.delete(id);
      } else {
        // Animation in progress
        const easedProgress = animation.easing(progress);
        const currentValue =
          animation.fromValue +
          (animation.toValue - animation.fromValue) * easedProgress;
        animation.onUpdate?.(currentValue, progress);
      }
    }
  }

  cancel(id) {
    return this.animations.delete(id);
  }

  cancelAll() {
    this.animations.clear();
  }

  isAnimating(id) {
    return this.animations.has(id);
  }
}

/**
 * Real-Time Chart Optimizer
 * Main optimization engine for real-time chart rendering
 */
class RealTimeChartOptimizer {
  constructor(options = {}) {
    this.options = {
      targetFPS: 60,
      maxDataPoints: 1000,
      bufferSize: 10000,
      enableAnimations: true,
      animationDuration: 300,
      adaptiveQuality: true,
      performanceThreshold: 30, // FPS threshold for quality adjustment
      ...options,
    };

    // Core components
    this.frameController = new FrameRateController(this.options.targetFPS);
    this.dataBuffer = new DataBufferManager(this.options.bufferSize);
    this.animationManager = new AnimationManager();

    // Chart registry
    this.charts = new Map();
    this.chartUpdateQueue = new Set();

    // Performance monitoring
    this.performanceMonitor = {
      frameDrops: 0,
      averageFPS: 0,
      memoryUsage: 0,
      renderTime: 0,
      lastQualityAdjustment: 0,
    };

    // Quality settings
    this.qualityLevel = 1.0; // 0.0 to 1.0
    this.qualitySettings = {
      high: { pointSize: 4, lineWidth: 2, antiAliasing: true, shadows: true },
      medium: {
        pointSize: 3,
        lineWidth: 1.5,
        antiAliasing: true,
        shadows: false,
      },
      low: { pointSize: 2, lineWidth: 1, antiAliasing: false, shadows: false },
    };

    this.init();
  }

  init() {
    // Set up frame controller callback
    this.frameController.addUpdateCallback((deltaTime, currentTime) => {
      this.update(deltaTime, currentTime);
    });

    // Monitor performance
    setInterval(() => {
      this.updatePerformanceMetrics();
    }, 1000);

    // Start the optimization engine
    this.start();
  }

  start() {
    this.frameController.start();
  }

  stop() {
    this.frameController.stop();
  }

  /**
   * Chart Registration and Management
   */
  registerChart(chartId, chartInstance, options = {}) {
    const chartConfig = {
      instance: chartInstance,
      lastUpdate: 0,
      updateInterval: options.updateInterval || 16, // ~60 FPS
      isDirty: false,
      dataBuffer: new DataBufferManager(
        options.bufferSize || this.options.bufferSize,
      ),
      renderMode: options.renderMode || "canvas", // 'canvas', 'svg', 'webgl'
      priority: options.priority || 1, // Higher priority = more frequent updates
      ...options,
    };

    this.charts.set(chartId, chartConfig);
    return chartId;
  }

  unregisterChart(chartId) {
    const chart = this.charts.get(chartId);
    if (chart) {
      chart.dataBuffer.clear();
      this.charts.delete(chartId);
      this.chartUpdateQueue.delete(chartId);
    }
  }

  /**
   * Data Management
   */
  addDataPoint(chartId, dataPoint) {
    const chart = this.charts.get(chartId);
    if (!chart) return;

    const itemId = chart.dataBuffer.add(dataPoint);
    chart.isDirty = true;
    this.chartUpdateQueue.add(chartId);

    return itemId;
  }

  addBatchData(chartId, dataPoints) {
    const chart = this.charts.get(chartId);
    if (!chart) return;

    const itemIds = dataPoints.map((point) => chart.dataBuffer.add(point));
    chart.isDirty = true;
    this.chartUpdateQueue.add(chartId);

    return itemIds;
  }

  getChartData(chartId, count = null) {
    const chart = this.charts.get(chartId);
    if (!chart) return [];

    return chart.dataBuffer.getLast(count || this.options.maxDataPoints);
  }

  /**
   * Update Loop
   */
  update(deltaTime, currentTime) {
    // Update animations
    this.animationManager.update(currentTime);

    // Update charts that need rendering
    this.updateCharts(currentTime);

    // Adaptive quality adjustment
    if (this.options.adaptiveQuality) {
      this.adjustQuality();
    }
  }

  updateCharts(currentTime) {
    const renderStartTime = performance.now();
    let chartsUpdated = 0;

    // Sort charts by priority
    const sortedCharts = Array.from(this.chartUpdateQueue)
      .map((id) => ({ id, chart: this.charts.get(id) }))
      .filter(({ chart }) => chart)
      .sort((a, b) => b.chart.priority - a.chart.priority);

    for (const { id, chart } of sortedCharts) {
      if (currentTime - chart.lastUpdate >= chart.updateInterval) {
        try {
          this.renderChart(id, chart, currentTime);
          chart.lastUpdate = currentTime;
          chart.isDirty = false;
          chartsUpdated++;
        } catch (error) {
          console.error(`Error updating chart ${id}:`, error);
        }
      }
    }

    // Clear update queue for updated charts
    for (const { id, chart } of sortedCharts) {
      if (!chart.isDirty) {
        this.chartUpdateQueue.delete(id);
      }
    }

    this.performanceMonitor.renderTime = performance.now() - renderStartTime;
  }

  renderChart(chartId, chartConfig, currentTime) {
    const { instance, dataBuffer, renderMode } = chartConfig;

    // Get current data
    const data = dataBuffer.getLast(this.options.maxDataPoints);
    const processedData = this.processDataForRendering(data);

    // Apply quality settings
    const qualitySettings = this.getCurrentQualitySettings();

    // Render based on mode
    switch (renderMode) {
      case "canvas":
        this.renderCanvasChart(instance, processedData, qualitySettings);
        break;
      case "svg":
        this.renderSVGChart(instance, processedData, qualitySettings);
        break;
      case "webgl":
        this.renderWebGLChart(instance, processedData, qualitySettings);
        break;
      default:
        this.renderDefaultChart(instance, processedData, qualitySettings);
    }

    // Emit update event
    instance.dispatchEvent?.(
      new CustomEvent("chart-updated", {
        detail: {
          chartId,
          dataPoints: processedData.length,
          timestamp: currentTime,
        },
      }),
    );
  }

  processDataForRendering(data) {
    // Convert wrapped data back to raw format
    return data.map((item) => ({
      ...item.data,
      timestamp: item.timestamp,
      id: item.id,
    }));
  }

  renderCanvasChart(chartInstance, data, quality) {
    if (chartInstance.updateData) {
      chartInstance.updateData(data, {
        pointSize: quality.pointSize,
        lineWidth: quality.lineWidth,
        antiAliasing: quality.antiAliasing,
        enableAnimations: this.options.enableAnimations && quality.antiAliasing,
      });
    }
  }

  renderSVGChart(chartInstance, data, quality) {
    if (chartInstance.updateData) {
      chartInstance.updateData(data, {
        strokeWidth: quality.lineWidth,
        pointRadius: quality.pointSize,
        enableTransitions: this.options.enableAnimations,
      });
    }
  }

  renderWebGLChart(chartInstance, data, quality) {
    if (chartInstance.updateData) {
      chartInstance.updateData(data, {
        pointSize: quality.pointSize,
        lineWidth: quality.lineWidth,
        enableShaders: quality.antiAliasing,
      });
    }
  }

  renderDefaultChart(chartInstance, data, quality) {
    if (chartInstance.setData) {
      chartInstance.setData(data);
    } else if (chartInstance.updateChart) {
      chartInstance.updateChart();
    }
  }

  /**
   * Quality Management
   */
  adjustQuality() {
    const now = performance.now();
    if (now - this.performanceMonitor.lastQualityAdjustment < 2000) {
      return; // Don't adjust too frequently
    }

    const metrics = this.frameController.getPerformanceMetrics();
    const currentFPS = metrics.actualFPS;

    if (
      currentFPS < this.options.performanceThreshold &&
      this.qualityLevel > 0.3
    ) {
      // Decrease quality
      this.qualityLevel = Math.max(0.3, this.qualityLevel - 0.1);
      console.log(
        `Quality decreased to ${this.qualityLevel.toFixed(1)} (FPS: ${currentFPS})`,
      );
    } else if (
      currentFPS > this.options.performanceThreshold + 10 &&
      this.qualityLevel < 1.0
    ) {
      // Increase quality
      this.qualityLevel = Math.min(1.0, this.qualityLevel + 0.05);
      console.log(
        `Quality increased to ${this.qualityLevel.toFixed(1)} (FPS: ${currentFPS})`,
      );
    }

    this.performanceMonitor.lastQualityAdjustment = now;
  }

  getCurrentQualitySettings() {
    if (this.qualityLevel >= 0.8) {
      return this.qualitySettings.high;
    } else if (this.qualityLevel >= 0.5) {
      return this.qualitySettings.medium;
    } else {
      return this.qualitySettings.low;
    }
  }

  setQualityLevel(level) {
    this.qualityLevel = Math.max(0.1, Math.min(1.0, level));
  }

  /**
   * Animation Support
   */
  animateChartTransition(
    chartId,
    property,
    fromValue,
    toValue,
    duration = null,
  ) {
    const animationId = `${chartId}_${property}`;
    const animationDuration = duration || this.options.animationDuration;

    return this.animationManager.animate(
      animationId,
      fromValue,
      toValue,
      animationDuration,
      "easeInOut",
      (value, progress) => {
        const chart = this.charts.get(chartId);
        if (chart && chart.instance.setProperty) {
          chart.instance.setProperty(property, value);
        }
      },
      (finalValue) => {
        const chart = this.charts.get(chartId);
        if (chart) {
          chart.isDirty = true;
          this.chartUpdateQueue.add(chartId);
        }
      },
    );
  }

  /**
   * Performance Monitoring
   */
  updatePerformanceMetrics() {
    const frameMetrics = this.frameController.getPerformanceMetrics();
    this.performanceMonitor.averageFPS = frameMetrics.actualFPS;

    // Detect frame drops
    if (frameMetrics.actualFPS < this.options.targetFPS * 0.8) {
      this.performanceMonitor.frameDrops++;
    }

    // Memory usage (if available)
    if (performance.memory) {
      this.performanceMonitor.memoryUsage =
        performance.memory.usedJSHeapSize / 1024 / 1024; // MB
    }
  }

  getPerformanceReport() {
    const frameMetrics = this.frameController.getPerformanceMetrics();

    return {
      fps: {
        target: frameMetrics.targetFPS,
        actual: frameMetrics.actualFPS,
        frameCount: frameMetrics.frameCount,
      },
      quality: {
        level: this.qualityLevel,
        settings: this.getCurrentQualitySettings(),
      },
      charts: {
        registered: this.charts.size,
        updateQueue: this.chartUpdateQueue.size,
      },
      performance: {
        ...this.performanceMonitor,
        renderTime: this.performanceMonitor.renderTime.toFixed(2) + "ms",
      },
      buffers: Array.from(this.charts.entries()).map(([id, chart]) => ({
        chartId: id,
        ...chart.dataBuffer.getStats(),
      })),
    };
  }

  /**
   * Cleanup
   */
  destroy() {
    this.stop();
    this.animationManager.cancelAll();

    // Clear all chart buffers
    for (const [id, chart] of this.charts) {
      chart.dataBuffer.clear();
    }

    this.charts.clear();
    this.chartUpdateQueue.clear();
  }
}

/**
 * Global optimizer instance
 */
const globalChartOptimizer = new RealTimeChartOptimizer();

// Export classes and global instance
if (typeof module !== "undefined" && module.exports) {
  module.exports = {
    RealTimeChartOptimizer,
    FrameRateController,
    DataBufferManager,
    AnimationManager,
    globalChartOptimizer,
  };
} else {
  // Browser environment
  window.RealTimeChartOptimizer = RealTimeChartOptimizer;
  window.FrameRateController = FrameRateController;
  window.DataBufferManager = DataBufferManager;
  window.AnimationManager = AnimationManager;
  window.globalChartOptimizer = globalChartOptimizer;
}
\n\n// training-monitor.js\n/**
 * Training Monitor Component
 *
 * Real-time monitoring interface for automated training pipelines with:
 * - Live progress tracking with WebSocket updates
 * - Training metrics visualization using D3.js
 * - Resource usage monitoring (CPU, memory)
 * - Training history and comparison
 * - Interactive controls for starting/stopping training
 */

import { WebSocketService } from "../services/websocket-service.js";

export class TrainingMonitor {
  constructor(container, options = {}) {
    this.container =
      typeof container === "string"
        ? document.querySelector(container)
        : container;
    this.options = {
      enableWebSocket: true,
      enableAutoRefresh: true,
      refreshInterval: 5000,
      showResourceMonitoring: true,
      showTrainingHistory: true,
      maxHistoryItems: 50,
      ...options,
    };

    // State management
    this.activeTrainings = new Map();
    this.trainingHistory = [];
    this.websocketService = null;

    // UI components
    this.components = {
      toolbar: null,
      activeTrainings: null,
      trainingDetails: null,
      metricsChart: null,
      resourceChart: null,
      historyTable: null,
    };

    // Event listeners
    this.listeners = new Map();

    this.init();
  }

  init() {
    this.createLayout();
    this.setupWebSocket();
    this.loadInitialData();
    this.bindEvents();

    if (this.options.enableAutoRefresh) {
      this.startAutoRefresh();
    }
  }

  createLayout() {
    this.container.innerHTML = `
            <div class="training-monitor">
                <!-- Header and Controls -->
                <div class="training-monitor__header">
                    <div class="training-monitor__title">
                        <h2>Automated Training Monitor</h2>
                        <div class="training-monitor__status">
                            <span class="status-indicator" data-status="disconnected">
                                <span class="status-dot"></span>
                                <span class="status-text">Connecting...</span>
                            </span>
                        </div>
                    </div>

                    <div class="training-monitor__toolbar">
                        <button class="btn btn--primary" data-action="start-training">
                            <i class="icon-play"></i> Start Training
                        </button>
                        <button class="btn btn--secondary" data-action="refresh">
                            <i class="icon-refresh"></i> Refresh
                        </button>
                        <button class="btn btn--secondary" data-action="settings">
                            <i class="icon-settings"></i> Settings
                        </button>
                    </div>
                </div>

                <!-- Active Trainings Grid -->
                <div class="training-monitor__content">
                    <div class="training-monitor__grid">
                        <!-- Active Trainings Panel -->
                        <div class="training-panel">
                            <div class="panel-header">
                                <h3>Active Trainings</h3>
                                <span class="badge badge--info" data-count="active-count">0</span>
                            </div>
                            <div class="panel-content">
                                <div class="training-list" data-component="active-trainings">
                                    <div class="empty-state">
                                        <i class="icon-training"></i>
                                        <p>No active trainings</p>
                                        <button class="btn btn--primary btn--sm" data-action="start-training">
                                            Start New Training
                                        </button>
                                    </div>
                                </div>
                            </div>
                        </div>

                        <!-- Training Details Panel -->
                        <div class="training-panel">
                            <div class="panel-header">
                                <h3>Training Details</h3>
                                <div class="panel-actions">
                                    <button class="btn btn--icon" data-action="expand-details" title="Expand">
                                        <i class="icon-expand"></i>
                                    </button>
                                </div>
                            </div>
                            <div class="panel-content">
                                <div class="training-details" data-component="training-details">
                                    <div class="empty-state">
                                        <p>Select a training to view details</p>
                                    </div>
                                </div>
                            </div>
                        </div>

                        <!-- Metrics Visualization Panel -->
                        <div class="training-panel training-panel--wide">
                            <div class="panel-header">
                                <h3>Training Metrics</h3>
                                <div class="panel-controls">
                                    <select class="form-select form-select--sm" data-control="metric-type">
                                        <option value="score">Score Progress</option>
                                        <option value="loss">Loss Curve</option>
                                        <option value="trials">Trial Results</option>
                                        <option value="resource">Resource Usage</option>
                                    </select>
                                </div>
                            </div>
                            <div class="panel-content">
                                <div class="metrics-chart" data-component="metrics-chart">
                                    <svg class="chart-svg"></svg>
                                    <div class="chart-legend"></div>
                                </div>
                            </div>
                        </div>

                        <!-- Training History Panel -->
                        <div class="training-panel training-panel--full">
                            <div class="panel-header">
                                <h3>Training History</h3>
                                <div class="panel-controls">
                                    <input type="search" class="form-input form-input--sm"
                                           placeholder="Search trainings..." data-control="history-search">
                                    <select class="form-select form-select--sm" data-control="history-filter">
                                        <option value="">All Statuses</option>
                                        <option value="completed">Completed</option>
                                        <option value="failed">Failed</option>
                                        <option value="cancelled">Cancelled</option>
                                    </select>
                                </div>
                            </div>
                            <div class="panel-content">
                                <div class="training-history" data-component="training-history">
                                    <div class="table-container">
                                        <table class="data-table">
                                            <thead>
                                                <tr>
                                                    <th>Training ID</th>
                                                    <th>Detector</th>
                                                    <th>Algorithm</th>
                                                    <th>Score</th>
                                                    <th>Duration</th>
                                                    <th>Status</th>
                                                    <th>Started</th>
                                                    <th>Actions</th>
                                                </tr>
                                            </thead>
                                            <tbody data-target="history-rows">
                                                <!-- Dynamic content -->
                                            </tbody>
                                        </table>
                                    </div>
                                    <div class="table-pagination">
                                        <button class="btn btn--sm" data-action="prev-page" disabled>Previous</button>
                                        <span class="pagination-info">Page 1 of 1</span>
                                        <button class="btn btn--sm" data-action="next-page" disabled>Next</button>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Start Training Modal -->
                <div class="modal" data-modal="start-training">
                    <div class="modal-content">
                        <div class="modal-header">
                            <h3>Start New Training</h3>
                            <button class="btn btn--icon" data-action="close-modal">
                                <i class="icon-close"></i>
                            </button>
                        </div>
                        <div class="modal-body">
                            <form class="training-form" data-form="start-training">
                                <div class="form-grid">
                                    <div class="form-group">
                                        <label>Detector</label>
                                        <select class="form-select" name="detector_id" required>
                                            <option value="">Select detector...</option>
                                        </select>
                                    </div>

                                    <div class="form-group">
                                        <label>Dataset</label>
                                        <select class="form-select" name="dataset_id" required>
                                            <option value="">Select dataset...</option>
                                        </select>
                                    </div>

                                    <div class="form-group">
                                        <label>Experiment Name</label>
                                        <input type="text" class="form-input" name="experiment_name"
                                               placeholder="Optional experiment name">
                                    </div>

                                    <div class="form-group">
                                        <label>Optimization Objective</label>
                                        <select class="form-select" name="optimization_objective">
                                            <option value="auc">AUC</option>
                                            <option value="precision">Precision</option>
                                            <option value="recall">Recall</option>
                                            <option value="f1_score">F1 Score</option>
                                        </select>
                                    </div>

                                    <div class="form-group">
                                        <label>Max Algorithms</label>
                                        <input type="number" class="form-input" name="max_algorithms"
                                               value="3" min="1" max="10">
                                    </div>

                                    <div class="form-group">
                                        <label>Max Optimization Time (minutes)</label>
                                        <input type="number" class="form-input" name="max_optimization_time"
                                               value="60" min="1" max="1440">
                                    </div>
                                </div>

                                <div class="form-group">
                                    <div class="form-checkboxes">
                                        <label class="checkbox-label">
                                            <input type="checkbox" name="enable_automl" checked>
                                            <span class="checkbox-custom"></span>
                                            Enable AutoML optimization
                                        </label>
                                        <label class="checkbox-label">
                                            <input type="checkbox" name="enable_ensemble" checked>
                                            <span class="checkbox-custom"></span>
                                            Enable ensemble creation
                                        </label>
                                        <label class="checkbox-label">
                                            <input type="checkbox" name="enable_early_stopping" checked>
                                            <span class="checkbox-custom"></span>
                                            Enable early stopping
                                        </label>
                                    </div>
                                </div>
                            </form>
                        </div>
                        <div class="modal-footer">
                            <button class="btn btn--secondary" data-action="close-modal">Cancel</button>
                            <button class="btn btn--primary" data-action="submit-training">Start Training</button>
                        </div>
                    </div>
                </div>
            </div>
        `;

    // Store component references
    this.components.toolbar = this.container.querySelector(
      ".training-monitor__toolbar",
    );
    this.components.activeTrainings = this.container.querySelector(
      '[data-component="active-trainings"]',
    );
    this.components.trainingDetails = this.container.querySelector(
      '[data-component="training-details"]',
    );
    this.components.metricsChart = this.container.querySelector(
      '[data-component="metrics-chart"]',
    );
    this.components.historyTable = this.container.querySelector(
      '[data-component="training-history"]',
    );

    this.setupCharts();
  }

  setupCharts() {
    // Initialize D3.js charts for metrics visualization
    this.initializeMetricsChart();
    this.initializeResourceChart();
  }

  initializeMetricsChart() {
    const chartContainer =
      this.components.metricsChart.querySelector(".chart-svg");
    const width = chartContainer.clientWidth || 600;
    const height = 300;
    const margin = { top: 20, right: 30, bottom: 40, left: 50 };

    const svg = d3
      .select(chartContainer)
      .attr("width", width)
      .attr("height", height);

    // Clear existing content
    svg.selectAll("*").remove();

    // Create chart group
    const chartGroup = svg
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    // Setup scales
    this.chartScales = {
      x: d3.scaleLinear().range([0, width - margin.left - margin.right]),
      y: d3.scaleLinear().range([height - margin.top - margin.bottom, 0]),
    };

    // Setup axes
    this.chartAxes = {
      x: d3.axisBottom(this.chartScales.x),
      y: d3.axisLeft(this.chartScales.y),
    };

    // Add axes to chart
    chartGroup
      .append("g")
      .attr("class", "axis axis--x")
      .attr("transform", `translate(0,${height - margin.top - margin.bottom})`);

    chartGroup.append("g").attr("class", "axis axis--y");

    // Add chart title
    chartGroup
      .append("text")
      .attr("class", "chart-title")
      .attr("x", (width - margin.left - margin.right) / 2)
      .attr("y", -5)
      .attr("text-anchor", "middle")
      .text("Training Progress");

    this.chartGroup = chartGroup;
  }

  initializeResourceChart() {
    // Resource usage chart initialization
    // This would be a smaller chart for CPU/memory monitoring
  }

  setupWebSocket() {
    if (!this.options.enableWebSocket) return;

    this.websocketService = new WebSocketService({
      url: this.getWebSocketUrl(),
      enableLogging: true,
    });

    // Setup WebSocket event handlers
    this.websocketService.on("connected", () => {
      this.updateConnectionStatus("connected");
      this.subscribeToTrainingUpdates();
    });

    this.websocketService.on("disconnected", () => {
      this.updateConnectionStatus("disconnected");
    });

    this.websocketService.on("error", (error) => {
      console.error("Training WebSocket error:", error);
      this.updateConnectionStatus("error");
    });

    this.websocketService.on("training_update", (data) => {
      this.handleTrainingUpdate(data);
    });

    this.websocketService.on("training_progress", (data) => {
      this.handleTrainingProgress(data.data);
    });
  }

  getWebSocketUrl() {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const host = window.location.host;
    return `${protocol}//${host}/ws/training`;
  }

  subscribeToTrainingUpdates() {
    if (this.websocketService) {
      this.websocketService.send({
        type: "subscribe_training_updates",
      });
    }
  }

  updateConnectionStatus(status) {
    const statusIndicator = this.container.querySelector(".status-indicator");
    const statusText = statusIndicator.querySelector(".status-text");

    statusIndicator.setAttribute("data-status", status);

    switch (status) {
      case "connected":
        statusText.textContent = "Connected";
        break;
      case "disconnected":
        statusText.textContent = "Disconnected";
        break;
      case "error":
        statusText.textContent = "Connection Error";
        break;
      default:
        statusText.textContent = "Connecting...";
    }
  }

  handleTrainingUpdate(data) {
    console.log("Training update received:", data);

    if (data.training_id) {
      // Update specific training
      this.updateTrainingItem(data.training_id, data);
    } else {
      // General update, refresh all
      this.refreshActiveTrainings();
    }
  }

  handleTrainingProgress(progressData) {
    console.log("Training progress:", progressData);

    // Update active training item
    this.updateActiveTraining(progressData);

    // Update charts if this training is selected
    const selectedTrainingId = this.getSelectedTrainingId();
    if (selectedTrainingId === progressData.training_id) {
      this.updateTrainingDetails(progressData);
      this.updateMetricsChart(progressData);
    }
  }

  updateActiveTraining(progressData) {
    this.activeTrainings.set(progressData.training_id, progressData);
    this.renderActiveTrainings();
  }

  renderActiveTrainings() {
    const container = this.components.activeTrainings;
    const trainings = Array.from(this.activeTrainings.values());

    if (trainings.length === 0) {
      container.innerHTML = `
                <div class="empty-state">
                    <i class="icon-training"></i>
                    <p>No active trainings</p>
                    <button class="btn btn--primary btn--sm" data-action="start-training">
                        Start New Training
                    </button>
                </div>
            `;
      this.updateActiveCount(0);
      return;
    }

    const html = trainings
      .map((training) => this.renderTrainingItem(training))
      .join("");
    container.innerHTML = html;
    this.updateActiveCount(trainings.length);
  }

  renderTrainingItem(training) {
    const statusClass = this.getStatusClass(training.status);
    const progressWidth = Math.round(training.progress_percentage);

    return `
            <div class="training-item" data-training-id="${training.training_id}">
                <div class="training-item__header">
                    <div class="training-item__title">
                        <strong>${training.training_id.substring(0, 8)}...</strong>
                        <span class="status-badge status-badge--${statusClass}">${training.status}</span>
                    </div>
                    <div class="training-item__actions">
                        <button class="btn btn--icon btn--sm" data-action="view-training"
                                data-training-id="${training.training_id}" title="View Details">
                            <i class="icon-eye"></i>
                        </button>
                        ${
                          training.status === "running"
                            ? `
                            <button class="btn btn--icon btn--sm btn--danger" data-action="cancel-training"
                                    data-training-id="${training.training_id}" title="Cancel">
                                <i class="icon-stop"></i>
                            </button>
                        `
                            : ""
                        }
                    </div>
                </div>

                <div class="training-item__progress">
                    <div class="progress-bar">
                        <div class="progress-bar__fill" style="width: ${progressWidth}%"></div>
                    </div>
                    <div class="progress-text">
                        <span>${training.current_step}</span>
                        <span>${progressWidth}%</span>
                    </div>
                </div>

                <div class="training-item__details">
                    ${
                      training.current_algorithm
                        ? `
                        <div class="detail-item">
                            <span class="detail-label">Algorithm:</span>
                            <span class="detail-value">${training.current_algorithm}</span>
                        </div>
                    `
                        : ""
                    }
                    ${
                      training.best_score
                        ? `
                        <div class="detail-item">
                            <span class="detail-label">Best Score:</span>
                            <span class="detail-value">${training.best_score.toFixed(4)}</span>
                        </div>
                    `
                        : ""
                    }
                    ${
                      training.current_message
                        ? `
                        <div class="detail-item detail-item--full">
                            <span class="detail-message">${training.current_message}</span>
                        </div>
                    `
                        : ""
                    }
                </div>

                ${
                  training.warnings && training.warnings.length > 0
                    ? `
                    <div class="training-item__warnings">
                        ${training.warnings
                          .map(
                            (warning) => `
                            <div class="warning-item">
                                <i class="icon-warning"></i>
                                <span>${warning}</span>
                            </div>
                        `,
                          )
                          .join("")}
                    </div>
                `
                    : ""
                }
            </div>
        `;
  }

  getStatusClass(status) {
    const statusMap = {
      idle: "secondary",
      scheduled: "info",
      running: "primary",
      optimizing: "warning",
      evaluating: "info",
      completed: "success",
      failed: "danger",
      cancelled: "secondary",
    };
    return statusMap[status] || "secondary";
  }

  updateActiveCount(count) {
    const countElement = this.container.querySelector(
      '[data-count="active-count"]',
    );
    if (countElement) {
      countElement.textContent = count;
    }
  }

  updateTrainingDetails(trainingData) {
    const container = this.components.trainingDetails;

    if (!trainingData) {
      container.innerHTML = `
                <div class="empty-state">
                    <p>Select a training to view details</p>
                </div>
            `;
      return;
    }

    container.innerHTML = `
            <div class="training-details-content">
                <div class="details-header">
                    <h4>Training ${trainingData.training_id.substring(0, 8)}...</h4>
                    <span class="status-badge status-badge--${this.getStatusClass(trainingData.status)}">
                        ${trainingData.status}
                    </span>
                </div>

                <div class="details-grid">
                    <div class="detail-group">
                        <h5>Progress</h5>
                        <div class="detail-item">
                            <span class="detail-label">Current Step:</span>
                            <span class="detail-value">${trainingData.current_step}</span>
                        </div>
                        <div class="detail-item">
                            <span class="detail-label">Progress:</span>
                            <span class="detail-value">${Math.round(trainingData.progress_percentage)}%</span>
                        </div>
                        <div class="detail-item">
                            <span class="detail-label">Started:</span>
                            <span class="detail-value">${new Date(trainingData.start_time).toLocaleString()}</span>
                        </div>
                        ${
                          trainingData.estimated_completion
                            ? `
                            <div class="detail-item">
                                <span class="detail-label">ETA:</span>
                                <span class="detail-value">${new Date(trainingData.estimated_completion).toLocaleString()}</span>
                            </div>
                        `
                            : ""
                        }
                    </div>

                    ${
                      trainingData.current_algorithm
                        ? `
                        <div class="detail-group">
                            <h5>Algorithm</h5>
                            <div class="detail-item">
                                <span class="detail-label">Current:</span>
                                <span class="detail-value">${trainingData.current_algorithm}</span>
                            </div>
                            ${
                              trainingData.current_trial &&
                              trainingData.total_trials
                                ? `
                                <div class="detail-item">
                                    <span class="detail-label">Trial:</span>
                                    <span class="detail-value">${trainingData.current_trial} / ${trainingData.total_trials}</span>
                                </div>
                            `
                                : ""
                            }
                            ${
                              trainingData.best_score
                                ? `
                                <div class="detail-item">
                                    <span class="detail-label">Best Score:</span>
                                    <span class="detail-value">${trainingData.best_score.toFixed(4)}</span>
                                </div>
                            `
                                : ""
                            }
                            ${
                              trainingData.current_score
                                ? `
                                <div class="detail-item">
                                    <span class="detail-label">Current Score:</span>
                                    <span class="detail-value">${trainingData.current_score.toFixed(4)}</span>
                                </div>
                            `
                                : ""
                            }
                        </div>
                    `
                        : ""
                    }

                    ${
                      this.options.showResourceMonitoring
                        ? `
                        <div class="detail-group">
                            <h5>Resources</h5>
                            ${
                              trainingData.memory_usage_mb
                                ? `
                                <div class="detail-item">
                                    <span class="detail-label">Memory:</span>
                                    <span class="detail-value">${Math.round(trainingData.memory_usage_mb)} MB</span>
                                </div>
                            `
                                : ""
                            }
                            ${
                              trainingData.cpu_usage_percent
                                ? `
                                <div class="detail-item">
                                    <span class="detail-label">CPU:</span>
                                    <span class="detail-value">${Math.round(trainingData.cpu_usage_percent)}%</span>
                                </div>
                            `
                                : ""
                            }
                        </div>
                    `
                        : ""
                    }
                </div>

                ${
                  trainingData.current_message
                    ? `
                    <div class="detail-message">
                        <h5>Status Message</h5>
                        <p>${trainingData.current_message}</p>
                    </div>
                `
                    : ""
                }
            </div>
        `;
  }

  updateMetricsChart(trainingData) {
    if (!this.chartGroup || !trainingData.best_score) return;

    // For now, just update with current score
    // In a real implementation, we'd track score history over time
    const data = [
      {
        trial: trainingData.current_trial || 1,
        score: trainingData.current_score || trainingData.best_score,
      },
    ];

    // Update scales
    this.chartScales.x.domain([0, trainingData.total_trials || 100]);
    this.chartScales.y.domain([0, 1]);

    // Update axes
    this.chartGroup.select(".axis--x").call(this.chartAxes.x);

    this.chartGroup.select(".axis--y").call(this.chartAxes.y);

    // Update data points
    const points = this.chartGroup.selectAll(".data-point").data(data);

    points
      .enter()
      .append("circle")
      .attr("class", "data-point")
      .attr("r", 4)
      .attr("fill", "#3b82f6")
      .merge(points)
      .attr("cx", (d) => this.chartScales.x(d.trial))
      .attr("cy", (d) => this.chartScales.y(d.score));

    points.exit().remove();
  }

  bindEvents() {
    // Toolbar actions
    this.container.addEventListener("click", (event) => {
      const action = event.target.closest("[data-action]")?.dataset.action;

      switch (action) {
        case "start-training":
          this.showStartTrainingModal();
          break;
        case "refresh":
          this.refreshAll();
          break;
        case "settings":
          this.showSettings();
          break;
        case "view-training":
          const trainingId =
            event.target.closest("[data-training-id]")?.dataset.trainingId;
          this.selectTraining(trainingId);
          break;
        case "cancel-training":
          const cancelId =
            event.target.closest("[data-training-id]")?.dataset.trainingId;
          this.cancelTraining(cancelId);
          break;
        case "close-modal":
          this.closeModal();
          break;
        case "submit-training":
          this.submitTraining();
          break;
      }
    });

    // Form submissions
    this.container.addEventListener("submit", (event) => {
      event.preventDefault();

      if (event.target.matches('[data-form="start-training"]')) {
        this.submitTraining();
      }
    });
  }

  async loadInitialData() {
    try {
      // Load active trainings
      await this.refreshActiveTrainings();

      // Load training history
      await this.refreshTrainingHistory();

      // Load dropdown options
      await this.loadFormOptions();
    } catch (error) {
      console.error("Failed to load initial data:", error);
      this.showError("Failed to load training data");
    }
  }

  async refreshActiveTrainings() {
    try {
      const response = await fetch("/api/training/active");
      const trainings = await response.json();

      this.activeTrainings.clear();
      trainings.forEach((training) => {
        this.activeTrainings.set(training.training_id, training);
      });

      this.renderActiveTrainings();
    } catch (error) {
      console.error("Failed to refresh active trainings:", error);
    }
  }

  async refreshTrainingHistory() {
    try {
      const response = await fetch("/api/training/history");
      const data = await response.json();

      this.trainingHistory = data.trainings || [];
      this.renderTrainingHistory();
    } catch (error) {
      console.error("Failed to refresh training history:", error);
    }
  }

  renderTrainingHistory() {
    const tbody = this.container.querySelector('[data-target="history-rows"]');

    if (this.trainingHistory.length === 0) {
      tbody.innerHTML = `
                <tr>
                    <td colspan="8" class="empty-cell">No training history available</td>
                </tr>
            `;
      return;
    }

    tbody.innerHTML = this.trainingHistory
      .map(
        (training) => `
            <tr>
                <td><code>${training.training_id.substring(0, 8)}...</code></td>
                <td>${training.detector_id.substring(0, 8)}...</td>
                <td>${training.best_algorithm || "N/A"}</td>
                <td>${training.best_score ? training.best_score.toFixed(4) : "N/A"}</td>
                <td>${training.training_time_seconds ? this.formatDuration(training.training_time_seconds) : "N/A"}</td>
                <td><span class="status-badge status-badge--${this.getStatusClass(training.status)}">${training.status}</span></td>
                <td>${training.start_time ? new Date(training.start_time).toLocaleDateString() : "N/A"}</td>
                <td>
                    <button class="btn btn--icon btn--sm" data-action="view-result"
                            data-training-id="${training.training_id}" title="View Details">
                        <i class="icon-eye"></i>
                    </button>
                </td>
            </tr>
        `,
      )
      .join("");
  }

  formatDuration(seconds) {
    if (seconds < 60) return `${Math.round(seconds)}s`;
    if (seconds < 3600) return `${Math.round(seconds / 60)}m`;
    return `${Math.round(seconds / 3600)}h`;
  }

  async loadFormOptions() {
    try {
      // Load detectors
      const detectorsResponse = await fetch("/api/detectors");
      const detectors = await detectorsResponse.json();

      const detectorSelect = this.container.querySelector(
        'select[name="detector_id"]',
      );
      detectorSelect.innerHTML =
        '<option value="">Select detector...</option>' +
        detectors
          .map((d) => `<option value="${d.id}">${d.name}</option>`)
          .join("");

      // Load datasets
      const datasetsResponse = await fetch("/api/datasets");
      const datasets = await datasetsResponse.json();

      const datasetSelect = this.container.querySelector(
        'select[name="dataset_id"]',
      );
      datasetSelect.innerHTML =
        '<option value="">Select dataset...</option>' +
        datasets
          .map((d) => `<option value="${d.id}">${d.name}</option>`)
          .join("");
    } catch (error) {
      console.error("Failed to load form options:", error);
    }
  }

  showStartTrainingModal() {
    const modal = this.container.querySelector('[data-modal="start-training"]');
    modal.classList.add("modal--active");
  }

  closeModal() {
    const modals = this.container.querySelectorAll(".modal");
    modals.forEach((modal) => modal.classList.remove("modal--active"));
  }

  async submitTraining() {
    const form = this.container.querySelector('[data-form="start-training"]');
    const formData = new FormData(form);

    const requestData = {
      detector_id: formData.get("detector_id"),
      dataset_id: formData.get("dataset_id"),
      experiment_name: formData.get("experiment_name") || null,
      optimization_objective: formData.get("optimization_objective"),
      max_algorithms: parseInt(formData.get("max_algorithms")),
      max_optimization_time:
        parseInt(formData.get("max_optimization_time")) * 60, // Convert to seconds
      enable_automl: formData.has("enable_automl"),
      enable_ensemble: formData.has("enable_ensemble"),
      enable_early_stopping: formData.has("enable_early_stopping"),
    };

    try {
      const response = await fetch("/api/training/start", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(requestData),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const result = await response.json();

      this.closeModal();
      this.showSuccess(`Training started: ${result.training_id}`);

      // Refresh active trainings
      await this.refreshActiveTrainings();
    } catch (error) {
      console.error("Failed to start training:", error);
      this.showError(`Failed to start training: ${error.message}`);
    }
  }

  async cancelTraining(trainingId) {
    if (!confirm("Are you sure you want to cancel this training?")) return;

    try {
      const response = await fetch(`/api/training/cancel/${trainingId}`, {
        method: "POST",
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      this.showSuccess("Training cancelled successfully");
      await this.refreshActiveTrainings();
    } catch (error) {
      console.error("Failed to cancel training:", error);
      this.showError(`Failed to cancel training: ${error.message}`);
    }
  }

  selectTraining(trainingId) {
    const training = this.activeTrainings.get(trainingId);
    if (training) {
      this.updateTrainingDetails(training);
      this.updateMetricsChart(training);

      // Update UI selection
      this.container.querySelectorAll(".training-item").forEach((item) => {
        item.classList.toggle(
          "training-item--selected",
          item.dataset.trainingId === trainingId,
        );
      });
    }
  }

  getSelectedTrainingId() {
    const selected = this.container.querySelector(".training-item--selected");
    return selected?.dataset.trainingId || null;
  }

  startAutoRefresh() {
    setInterval(() => {
      if (document.visibilityState === "visible") {
        this.refreshActiveTrainings();
      }
    }, this.options.refreshInterval);
  }

  async refreshAll() {
    await Promise.all([
      this.refreshActiveTrainings(),
      this.refreshTrainingHistory(),
    ]);
  }

  showSuccess(message) {
    // Implementation for success notifications
    console.log("Success:", message);
  }

  showError(message) {
    // Implementation for error notifications
    console.error("Error:", message);
  }

  showSettings() {
    // Implementation for settings modal
    console.log("Show settings");
  }

  destroy() {
    if (this.websocketService) {
      this.websocketService.disconnect();
    }

    this.listeners.clear();
    this.activeTrainings.clear();
  }
}

export default TrainingMonitor;
\n\n// analytics-engine.js\n/**
 * Advanced Analytics Engine
 * Real-time statistical analysis with trend detection and smart alerting system
 * Provides comprehensive anomaly analysis, pattern recognition, and predictive insights
 */

/**
 * Statistical Analysis Library
 * Core statistical functions for real-time analysis
 */
class StatisticalAnalysis {
  static mean(values) {
    return values.length === 0
      ? 0
      : values.reduce((sum, val) => sum + val, 0) / values.length;
  }

  static median(values) {
    if (values.length === 0) return 0;
    const sorted = [...values].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 === 0
      ? (sorted[mid - 1] + sorted[mid]) / 2
      : sorted[mid];
  }

  static standardDeviation(values) {
    if (values.length < 2) return 0;
    const mean = this.mean(values);
    const variance =
      values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) /
      (values.length - 1);
    return Math.sqrt(variance);
  }

  static percentile(values, p) {
    if (values.length === 0) return 0;
    const sorted = [...values].sort((a, b) => a - b);
    const index = (p / 100) * (sorted.length - 1);
    const lower = Math.floor(index);
    const upper = Math.ceil(index);

    if (lower === upper) return sorted[lower];
    return sorted[lower] * (upper - index) + sorted[upper] * (index - lower);
  }

  static zScore(value, mean, stdDev) {
    return stdDev === 0 ? 0 : (value - mean) / stdDev;
  }

  static correlationCoefficient(x, y) {
    if (x.length !== y.length || x.length < 2) return 0;

    const meanX = this.mean(x);
    const meanY = this.mean(y);

    let numerator = 0;
    let denomX = 0;
    let denomY = 0;

    for (let i = 0; i < x.length; i++) {
      const xDiff = x[i] - meanX;
      const yDiff = y[i] - meanY;
      numerator += xDiff * yDiff;
      denomX += xDiff * xDiff;
      denomY += yDiff * yDiff;
    }

    const denominator = Math.sqrt(denomX * denomY);
    return denominator === 0 ? 0 : numerator / denominator;
  }

  static linearRegression(x, y) {
    if (x.length !== y.length || x.length < 2) {
      return { slope: 0, intercept: 0, r2: 0 };
    }

    const n = x.length;
    const sumX = x.reduce((a, b) => a + b, 0);
    const sumY = y.reduce((a, b) => a + b, 0);
    const sumXY = x.reduce((acc, xi, i) => acc + xi * y[i], 0);
    const sumXX = x.reduce((acc, xi) => acc + xi * xi, 0);
    const sumYY = y.reduce((acc, yi) => acc + yi * yi, 0);

    const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
    const intercept = (sumY - slope * sumX) / n;

    // Calculate R-squared
    const yMean = sumY / n;
    const ssReg = x.reduce((acc, xi, i) => {
      const predicted = slope * xi + intercept;
      return acc + Math.pow(predicted - yMean, 2);
    }, 0);
    const ssTot = y.reduce((acc, yi) => acc + Math.pow(yi - yMean, 2), 0);
    const r2 = ssTot === 0 ? 1 : ssReg / ssTot;

    return { slope, intercept, r2 };
  }

  static exponentialSmoothing(values, alpha = 0.3) {
    if (values.length === 0) return [];

    const smoothed = [values[0]];
    for (let i = 1; i < values.length; i++) {
      smoothed[i] = alpha * values[i] + (1 - alpha) * smoothed[i - 1];
    }
    return smoothed;
  }

  static movingAverage(values, window) {
    if (values.length < window) return values.slice();

    const result = [];
    for (let i = 0; i <= values.length - window; i++) {
      const windowValues = values.slice(i, i + window);
      result.push(this.mean(windowValues));
    }
    return result;
  }
}

/**
 * Trend Detection Engine
 * Identifies patterns and trends in time series data
 */
class TrendDetector {
  constructor(options = {}) {
    this.options = {
      minDataPoints: 10,
      trendThreshold: 0.05, // Minimum slope for trend detection
      significanceLevel: 0.05, // Statistical significance threshold
      seasonalityWindow: 24, // Hours for seasonal pattern detection
      changePointSensitivity: 2.0, // Standard deviations for change point detection
      ...options,
    };

    this.trendHistory = [];
    this.seasonalPatterns = new Map();
    this.changePoints = [];
  }

  analyzeTrend(data, timestamps = null) {
    if (data.length < this.options.minDataPoints) {
      return {
        type: "insufficient_data",
        confidence: 0,
        message: "Not enough data points for trend analysis",
      };
    }

    // Create time indices if timestamps not provided
    const timeIndices = timestamps
      ? timestamps.map((t) => new Date(t).getTime())
      : data.map((_, i) => i);

    // Perform linear regression
    const regression = StatisticalAnalysis.linearRegression(timeIndices, data);

    // Determine trend type and confidence
    const trendAnalysis = this.classifyTrend(regression, data);

    // Detect seasonal patterns
    const seasonality = this.detectSeasonality(data, timestamps);

    // Detect change points
    const changePoints = this.detectChangePoints(data);

    // Calculate overall trend strength
    const trendStrength = this.calculateTrendStrength(data, regression);

    const result = {
      ...trendAnalysis,
      regression,
      seasonality,
      changePoints,
      trendStrength,
      dataPoints: data.length,
      timespan: timestamps
        ? {
            start: new Date(Math.min(...timestamps)),
            end: new Date(Math.max(...timestamps)),
            duration: Math.max(...timestamps) - Math.min(...timestamps),
          }
        : null,
    };

    // Store in history
    this.trendHistory.push({
      timestamp: Date.now(),
      analysis: result,
    });

    return result;
  }

  classifyTrend(regression, data) {
    const { slope, r2 } = regression;
    const absSlope = Math.abs(slope);
    const dataRange = Math.max(...data) - Math.min(...data);
    const normalizedSlope = dataRange > 0 ? absSlope / dataRange : 0;

    let type, confidence, direction, strength;

    // Determine direction
    if (Math.abs(slope) < this.options.trendThreshold) {
      direction = "stable";
      type = "stable";
    } else if (slope > 0) {
      direction = "increasing";
      type = "upward";
    } else {
      direction = "decreasing";
      type = "downward";
    }

    // Determine strength
    if (normalizedSlope < 0.01) {
      strength = "weak";
    } else if (normalizedSlope < 0.05) {
      strength = "moderate";
    } else {
      strength = "strong";
    }

    // Calculate confidence based on R-squared and data consistency
    confidence = Math.min(r2 * 100, 95); // Cap at 95%

    return {
      type,
      direction,
      strength,
      confidence: Math.round(confidence),
      slope,
      normalizedSlope,
      message: this.generateTrendMessage(type, strength, confidence),
    };
  }

  detectSeasonality(data, timestamps) {
    if (!timestamps || data.length < this.options.seasonalityWindow * 2) {
      return {
        detected: false,
        message: "Insufficient data for seasonality detection",
      };
    }

    // Group data by hour of day
    const hourlyPatterns = {};
    timestamps.forEach((timestamp, index) => {
      const hour = new Date(timestamp).getHours();
      if (!hourlyPatterns[hour]) hourlyPatterns[hour] = [];
      hourlyPatterns[hour].push(data[index]);
    });

    // Calculate average for each hour
    const hourlyAverages = {};
    Object.keys(hourlyPatterns).forEach((hour) => {
      hourlyAverages[hour] = StatisticalAnalysis.mean(hourlyPatterns[hour]);
    });

    // Check for significant variation across hours
    const averageValues = Object.values(hourlyAverages);
    const overallMean = StatisticalAnalysis.mean(averageValues);
    const variance = StatisticalAnalysis.standardDeviation(averageValues);
    const coefficientOfVariation = variance / Math.abs(overallMean);

    const seasonalityDetected = coefficientOfVariation > 0.1; // 10% variation threshold

    return {
      detected: seasonalityDetected,
      patterns: hourlyAverages,
      variation: coefficientOfVariation,
      peak: seasonalityDetected ? this.findPeakHour(hourlyAverages) : null,
      message: seasonalityDetected
        ? `Seasonal pattern detected with ${(coefficientOfVariation * 100).toFixed(1)}% variation`
        : "No significant seasonal pattern detected",
    };
  }

  findPeakHour(hourlyAverages) {
    let maxHour = null;
    let maxValue = -Infinity;

    Object.entries(hourlyAverages).forEach(([hour, value]) => {
      if (value > maxValue) {
        maxValue = value;
        maxHour = parseInt(hour);
      }
    });

    return { hour: maxHour, value: maxValue };
  }

  detectChangePoints(data) {
    if (data.length < 10) return [];

    const changePoints = [];
    const windowSize = Math.max(3, Math.floor(data.length / 10));

    for (let i = windowSize; i < data.length - windowSize; i++) {
      const leftWindow = data.slice(i - windowSize, i);
      const rightWindow = data.slice(i, i + windowSize);

      const leftMean = StatisticalAnalysis.mean(leftWindow);
      const rightMean = StatisticalAnalysis.mean(rightWindow);
      const leftStd = StatisticalAnalysis.standardDeviation(leftWindow);
      const rightStd = StatisticalAnalysis.standardDeviation(rightWindow);

      // Calculate change magnitude
      const meanDifference = Math.abs(rightMean - leftMean);
      const pooledStd = Math.sqrt(
        (leftStd * leftStd + rightStd * rightStd) / 2,
      );

      if (pooledStd > 0) {
        const changeMagnitude = meanDifference / pooledStd;

        if (changeMagnitude > this.options.changePointSensitivity) {
          changePoints.push({
            index: i,
            magnitude: changeMagnitude,
            before: leftMean,
            after: rightMean,
            change: rightMean - leftMean,
            changePercent:
              leftMean !== 0 ? ((rightMean - leftMean) / leftMean) * 100 : 0,
          });
        }
      }
    }

    return changePoints;
  }

  calculateTrendStrength(data, regression) {
    const { r2 } = regression;
    const variance = StatisticalAnalysis.standardDeviation(data);
    const mean = StatisticalAnalysis.mean(data);
    const coefficientOfVariation =
      Math.abs(mean) > 0 ? variance / Math.abs(mean) : 0;

    return {
      r_squared: r2,
      coefficient_of_variation: coefficientOfVariation,
      strength_score: r2 / (1 + coefficientOfVariation), // Combined metric
      interpretation: this.interpretTrendStrength(r2, coefficientOfVariation),
    };
  }

  interpretTrendStrength(r2, cv) {
    if (r2 > 0.8 && cv < 0.2) return "Very Strong";
    if (r2 > 0.6 && cv < 0.4) return "Strong";
    if (r2 > 0.4 && cv < 0.6) return "Moderate";
    if (r2 > 0.2) return "Weak";
    return "Very Weak";
  }

  generateTrendMessage(type, strength, confidence) {
    const strengthAdj = strength === "weak" ? "slight" : strength;
    return `${strengthAdj} ${type} trend detected with ${confidence}% confidence`;
  }

  getTrendHistory(limit = 10) {
    return this.trendHistory.slice(-limit);
  }

  clearHistory() {
    this.trendHistory = [];
    this.seasonalPatterns.clear();
    this.changePoints = [];
  }
}

/**
 * Alert System
 * Smart alerting based on statistical analysis and trend detection
 */
class SmartAlertSystem {
  constructor(options = {}) {
    this.options = {
      alertThresholds: {
        critical: { zScore: 3.0, trendChange: 50 },
        high: { zScore: 2.5, trendChange: 30 },
        medium: { zScore: 2.0, trendChange: 20 },
        low: { zScore: 1.5, trendChange: 10 },
      },
      cooldownPeriod: 5 * 60 * 1000, // 5 minutes
      maxAlertsPerHour: 10,
      enableAdaptiveThresholds: true,
      ...options,
    };

    this.alerts = [];
    this.alertHistory = [];
    this.suppressedAlerts = new Set();
    this.adaptiveThresholds = new Map();
  }

  analyzeForAlerts(data, metadata = {}) {
    const timestamp = Date.now();
    const alerts = [];

    // Statistical outlier detection
    const outlierAlerts = this.detectOutlierAlerts(data, metadata);
    alerts.push(...outlierAlerts);

    // Trend-based alerts
    if (data.length >= 10) {
      const trendAlerts = this.detectTrendAlerts(data, metadata);
      alerts.push(...trendAlerts);
    }

    // Volume-based alerts
    const volumeAlerts = this.detectVolumeAlerts(data, metadata);
    alerts.push(...volumeAlerts);

    // Rate of change alerts
    const changeAlerts = this.detectChangeRateAlerts(data, metadata);
    alerts.push(...changeAlerts);

    // Filter and process alerts
    const processedAlerts = this.processAlerts(alerts, timestamp);

    // Store in history
    this.alertHistory.push({
      timestamp,
      alerts: processedAlerts,
      dataPoints: data.length,
      metadata,
    });

    return processedAlerts;
  }

  detectOutlierAlerts(data, metadata) {
    if (data.length < 3) return [];

    const alerts = [];
    const mean = StatisticalAnalysis.mean(data);
    const stdDev = StatisticalAnalysis.standardDeviation(data);

    // Check most recent values
    const recentValues = data.slice(-5);

    recentValues.forEach((value, index) => {
      const zScore = StatisticalAnalysis.zScore(value, mean, stdDev);
      const severity = this.classifyZScoreSeverity(Math.abs(zScore));

      if (severity) {
        alerts.push({
          type: "statistical_outlier",
          severity,
          value,
          zScore,
          mean,
          stdDev,
          message: `Statistical outlier detected: value ${value.toFixed(3)} is ${Math.abs(zScore).toFixed(2)} standard deviations from mean`,
          metadata: {
            index: data.length - recentValues.length + index,
            ...metadata,
          },
        });
      }
    });

    return alerts;
  }

  detectTrendAlerts(data, metadata) {
    const trendDetector = new TrendDetector();
    const trendAnalysis = trendDetector.analyzeTrend(data);
    const alerts = [];

    // Alert on significant trend changes
    if (trendAnalysis.confidence > 70) {
      if (trendAnalysis.strength === "strong") {
        alerts.push({
          type: "trend_change",
          severity: trendAnalysis.type === "stable" ? "low" : "medium",
          trend: trendAnalysis,
          message: `Strong ${trendAnalysis.direction} trend detected with ${trendAnalysis.confidence}% confidence`,
          metadata,
        });
      }

      // Alert on change points
      if (trendAnalysis.changePoints && trendAnalysis.changePoints.length > 0) {
        const significantChanges = trendAnalysis.changePoints.filter(
          (cp) => Math.abs(cp.changePercent) > 20,
        );

        significantChanges.forEach((changePoint) => {
          alerts.push({
            type: "change_point",
            severity:
              Math.abs(changePoint.changePercent) > 50 ? "high" : "medium",
            changePoint,
            message: `Significant change detected: ${changePoint.changePercent.toFixed(1)}% change at data point ${changePoint.index}`,
            metadata,
          });
        });
      }
    }

    return alerts;
  }

  detectVolumeAlerts(data, metadata) {
    const alerts = [];
    const recentWindow = Math.min(20, Math.floor(data.length * 0.2));

    if (data.length < recentWindow * 2) return alerts;

    const recentData = data.slice(-recentWindow);
    const historicalData = data.slice(0, -recentWindow);

    const recentMean = StatisticalAnalysis.mean(recentData);
    const historicalMean = StatisticalAnalysis.mean(historicalData);

    const changePercent =
      historicalMean !== 0
        ? ((recentMean - historicalMean) / historicalMean) * 100
        : 0;

    if (Math.abs(changePercent) > 25) {
      alerts.push({
        type: "volume_change",
        severity: Math.abs(changePercent) > 50 ? "high" : "medium",
        changePercent,
        recentMean,
        historicalMean,
        message: `Significant volume change: ${changePercent > 0 ? "increase" : "decrease"} of ${Math.abs(changePercent).toFixed(1)}%`,
        metadata,
      });
    }

    return alerts;
  }

  detectChangeRateAlerts(data, metadata) {
    if (data.length < 5) return [];

    const alerts = [];
    const recentChanges = [];

    // Calculate rate of change for recent points
    for (let i = data.length - 4; i < data.length; i++) {
      if (i > 0) {
        const changeRate =
          Math.abs((data[i] - data[i - 1]) / data[i - 1]) * 100;
        recentChanges.push(changeRate);
      }
    }

    const avgChangeRate = StatisticalAnalysis.mean(recentChanges);
    const maxChangeRate = Math.max(...recentChanges);

    if (maxChangeRate > 20) {
      alerts.push({
        type: "rapid_change",
        severity:
          maxChangeRate > 50
            ? "critical"
            : maxChangeRate > 30
              ? "high"
              : "medium",
        maxChangeRate,
        avgChangeRate,
        message: `Rapid change detected: maximum ${maxChangeRate.toFixed(1)}% change between consecutive points`,
        metadata,
      });
    }

    return alerts;
  }

  classifyZScoreSeverity(absZScore) {
    if (absZScore >= this.options.alertThresholds.critical.zScore)
      return "critical";
    if (absZScore >= this.options.alertThresholds.high.zScore) return "high";
    if (absZScore >= this.options.alertThresholds.medium.zScore)
      return "medium";
    if (absZScore >= this.options.alertThresholds.low.zScore) return "low";
    return null;
  }

  processAlerts(alerts, timestamp) {
    // Filter out suppressed alerts
    const activeAlerts = alerts.filter(
      (alert) => !this.isAlertSuppressed(alert),
    );

    // Apply rate limiting
    const rateLimitedAlerts = this.applyRateLimit(activeAlerts, timestamp);

    // Add timestamp and IDs
    const processedAlerts = rateLimitedAlerts.map((alert) => ({
      ...alert,
      id: this.generateAlertId(),
      timestamp,
      acknowledged: false,
    }));

    // Store active alerts
    this.alerts.push(...processedAlerts);

    // Clean up old alerts
    this.cleanupOldAlerts();

    return processedAlerts;
  }

  isAlertSuppressed(alert) {
    const alertKey = `${alert.type}_${alert.severity}`;
    const suppressEntry = this.suppressedAlerts.get(alertKey);

    if (!suppressEntry) return false;

    // Check if cooldown period has passed
    if (Date.now() - suppressEntry.timestamp > this.options.cooldownPeriod) {
      this.suppressedAlerts.delete(alertKey);
      return false;
    }

    return true;
  }

  applyRateLimit(alerts, timestamp) {
    const hourAgo = timestamp - 60 * 60 * 1000;
    const recentAlerts = this.alertHistory.filter(
      (entry) => entry.timestamp > hourAgo,
    );
    const recentAlertCount = recentAlerts.reduce(
      (count, entry) => count + entry.alerts.length,
      0,
    );

    if (recentAlertCount >= this.options.maxAlertsPerHour) {
      // Only allow critical alerts when rate limited
      return alerts.filter((alert) => alert.severity === "critical");
    }

    return alerts;
  }

  acknowledgeAlert(alertId) {
    const alert = this.alerts.find((a) => a.id === alertId);
    if (alert) {
      alert.acknowledged = true;
      alert.acknowledgedAt = Date.now();

      // Add to suppressed alerts to prevent duplicates
      const alertKey = `${alert.type}_${alert.severity}`;
      this.suppressedAlerts.set(alertKey, {
        timestamp: Date.now(),
        alertId,
      });

      return true;
    }
    return false;
  }

  getActiveAlerts(severityFilter = null) {
    let alerts = this.alerts.filter((alert) => !alert.acknowledged);

    if (severityFilter) {
      alerts = alerts.filter((alert) => alert.severity === severityFilter);
    }

    return alerts.sort((a, b) => {
      const severityOrder = { critical: 4, high: 3, medium: 2, low: 1 };
      return severityOrder[b.severity] - severityOrder[a.severity];
    });
  }

  getAlertHistory(limit = 100) {
    return this.alertHistory.slice(-limit);
  }

  generateAlertId() {
    return `alert_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  cleanupOldAlerts() {
    const cutoffTime = Date.now() - 24 * 60 * 60 * 1000; // 24 hours
    this.alerts = this.alerts.filter((alert) => alert.timestamp > cutoffTime);
    this.alertHistory = this.alertHistory.filter(
      (entry) => entry.timestamp > cutoffTime,
    );
  }

  getAlertStatistics() {
    const activeAlerts = this.getActiveAlerts();
    const severityCounts = { critical: 0, high: 0, medium: 0, low: 0 };

    activeAlerts.forEach((alert) => {
      severityCounts[alert.severity]++;
    });

    return {
      total: activeAlerts.length,
      by_severity: severityCounts,
      acknowledged: this.alerts.filter((a) => a.acknowledged).length,
      suppressed: this.suppressedAlerts.size,
      recent_history: this.alertHistory.slice(-10),
    };
  }
}

/**
 * Advanced Analytics Engine
 * Main orchestrator for real-time analytics and alerting
 */
class AdvancedAnalyticsEngine {
  constructor(options = {}) {
    this.options = {
      analysisInterval: 30000, // 30 seconds
      trendAnalysisWindow: 100, // Number of data points
      enableRealTimeAnalysis: true,
      enableAlerts: true,
      enablePrediction: true,
      ...options,
    };

    // Core components
    this.trendDetector = new TrendDetector();
    this.alertSystem = new SmartAlertSystem();

    // Data storage
    this.dataStreams = new Map();
    this.analysisResults = new Map();

    // Timers
    this.analysisTimer = null;

    // Event system
    this.eventListeners = new Map();

    this.init();
  }

  init() {
    if (this.options.enableRealTimeAnalysis) {
      this.startRealTimeAnalysis();
    }
  }

  /**
   * Data Stream Management
   */
  registerDataStream(streamId, options = {}) {
    this.dataStreams.set(streamId, {
      data: [],
      timestamps: [],
      maxSize: options.maxSize || 1000,
      metadata: options.metadata || {},
      lastAnalysis: 0,
      analysisInterval:
        options.analysisInterval || this.options.analysisInterval,
    });
  }

  addDataPoint(streamId, value, timestamp = null) {
    const stream = this.dataStreams.get(streamId);
    if (!stream) {
      console.warn(`Data stream ${streamId} not registered`);
      return;
    }

    const ts = timestamp || Date.now();
    stream.data.push(value);
    stream.timestamps.push(ts);

    // Maintain max size
    if (stream.data.length > stream.maxSize) {
      stream.data.shift();
      stream.timestamps.shift();
    }

    // Trigger analysis if needed
    if (ts - stream.lastAnalysis > stream.analysisInterval) {
      this.analyzeStream(streamId);
    }
  }

  addBatchData(streamId, values, timestamps = null) {
    values.forEach((value, index) => {
      const timestamp = timestamps ? timestamps[index] : null;
      this.addDataPoint(streamId, value, timestamp);
    });
  }

  /**
   * Analysis Execution
   */
  analyzeStream(streamId) {
    const stream = this.dataStreams.get(streamId);
    if (!stream || stream.data.length === 0) return null;

    const analysisStartTime = performance.now();

    // Trend analysis
    const trendAnalysis = this.trendDetector.analyzeTrend(
      stream.data,
      stream.timestamps,
    );

    // Statistical summary
    const statistics = this.calculateStatistics(stream.data);

    // Alert analysis
    const alerts = this.options.enableAlerts
      ? this.alertSystem.analyzeForAlerts(stream.data, {
          streamId,
          ...stream.metadata,
        })
      : [];

    // Prediction (if enabled)
    const prediction = this.options.enablePrediction
      ? this.generatePrediction(stream.data, stream.timestamps)
      : null;

    const analysisResult = {
      streamId,
      timestamp: Date.now(),
      dataPoints: stream.data.length,
      timespan:
        stream.timestamps.length > 1
          ? {
              start: Math.min(...stream.timestamps),
              end: Math.max(...stream.timestamps),
              duration:
                Math.max(...stream.timestamps) - Math.min(...stream.timestamps),
            }
          : null,
      trend: trendAnalysis,
      statistics,
      alerts,
      prediction,
      analysisTime: performance.now() - analysisStartTime,
    };

    // Store results
    this.analysisResults.set(streamId, analysisResult);
    stream.lastAnalysis = Date.now();

    // Emit events
    this.emit("analysis_complete", { streamId, results: analysisResult });

    if (alerts.length > 0) {
      this.emit("alerts_generated", { streamId, alerts });
    }

    return analysisResult;
  }

  analyzeAllStreams() {
    const results = {};

    for (const streamId of this.dataStreams.keys()) {
      results[streamId] = this.analyzeStream(streamId);
    }

    return results;
  }

  calculateStatistics(data) {
    if (data.length === 0) return null;

    return {
      count: data.length,
      mean: StatisticalAnalysis.mean(data),
      median: StatisticalAnalysis.median(data),
      std_dev: StatisticalAnalysis.standardDeviation(data),
      min: Math.min(...data),
      max: Math.max(...data),
      range: Math.max(...data) - Math.min(...data),
      percentiles: {
        p25: StatisticalAnalysis.percentile(data, 25),
        p50: StatisticalAnalysis.percentile(data, 50),
        p75: StatisticalAnalysis.percentile(data, 75),
        p90: StatisticalAnalysis.percentile(data, 90),
        p95: StatisticalAnalysis.percentile(data, 95),
        p99: StatisticalAnalysis.percentile(data, 99),
      },
      recent_stats:
        data.length >= 10
          ? {
              recent_mean: StatisticalAnalysis.mean(data.slice(-10)),
              recent_std: StatisticalAnalysis.standardDeviation(
                data.slice(-10),
              ),
            }
          : null,
    };
  }

  generatePrediction(data, timestamps = null) {
    if (data.length < 10) return null;

    try {
      // Simple linear prediction for next few points
      const timeIndices = timestamps
        ? timestamps.map((t) => new Date(t).getTime())
        : data.map((_, i) => i);

      const regression = StatisticalAnalysis.linearRegression(
        timeIndices,
        data,
      );

      if (regression.r2 < 0.1) {
        return {
          method: "linear_regression",
          confidence: "low",
          message: "Low predictive confidence due to high variance",
        };
      }

      // Predict next 5 points
      const lastTime = Math.max(...timeIndices);
      const timeStep =
        timeIndices.length > 1
          ? timeIndices[timeIndices.length - 1] -
            timeIndices[timeIndices.length - 2]
          : 1;

      const predictions = [];
      for (let i = 1; i <= 5; i++) {
        const futureTime = lastTime + timeStep * i;
        const predictedValue =
          regression.slope * futureTime + regression.intercept;
        predictions.push({
          time: futureTime,
          value: predictedValue,
          confidence: regression.r2,
        });
      }

      return {
        method: "linear_regression",
        predictions,
        confidence:
          regression.r2 > 0.7 ? "high" : regression.r2 > 0.4 ? "medium" : "low",
        regression_stats: regression,
      };
    } catch (error) {
      return {
        method: "linear_regression",
        error: error.message,
        confidence: "none",
      };
    }
  }

  /**
   * Real-time Analysis
   */
  startRealTimeAnalysis() {
    if (this.analysisTimer) return;

    this.analysisTimer = setInterval(() => {
      const results = this.analyzeAllStreams();
      this.emit("realtime_analysis", { results, timestamp: Date.now() });
    }, this.options.analysisInterval);
  }

  stopRealTimeAnalysis() {
    if (this.analysisTimer) {
      clearInterval(this.analysisTimer);
      this.analysisTimer = null;
    }
  }

  /**
   * Results and Alerts
   */
  getStreamAnalysis(streamId) {
    return this.analysisResults.get(streamId);
  }

  getAllAnalysisResults() {
    const results = {};
    for (const [streamId, result] of this.analysisResults) {
      results[streamId] = result;
    }
    return results;
  }

  getActiveAlerts(severityFilter = null) {
    return this.alertSystem.getActiveAlerts(severityFilter);
  }

  acknowledgeAlert(alertId) {
    return this.alertSystem.acknowledgeAlert(alertId);
  }

  getAlertStatistics() {
    return this.alertSystem.getAlertStatistics();
  }

  /**
   * Event System
   */
  on(event, listener) {
    if (!this.eventListeners.has(event)) {
      this.eventListeners.set(event, new Set());
    }
    this.eventListeners.get(event).add(listener);
    return () => this.off(event, listener);
  }

  off(event, listener) {
    if (this.eventListeners.has(event)) {
      this.eventListeners.get(event).delete(listener);
    }
  }

  emit(event, data) {
    if (this.eventListeners.has(event)) {
      this.eventListeners.get(event).forEach((listener) => {
        try {
          listener(data);
        } catch (error) {
          console.error("Analytics event listener error:", error);
        }
      });
    }
  }

  /**
   * Utilities and Reports
   */
  generateAnalyticsReport() {
    const report = {
      timestamp: Date.now(),
      summary: {
        total_streams: this.dataStreams.size,
        active_alerts: this.getActiveAlerts().length,
        analysis_results: this.analysisResults.size,
      },
      streams: {},
      alerts: this.getAlertStatistics(),
      system_health: this.getSystemHealth(),
    };

    // Add stream summaries
    for (const [streamId, stream] of this.dataStreams) {
      const analysis = this.analysisResults.get(streamId);
      report.streams[streamId] = {
        data_points: stream.data.length,
        last_analysis: stream.lastAnalysis,
        trend: analysis ? analysis.trend.type : "unknown",
        alerts: analysis ? analysis.alerts.length : 0,
      };
    }

    return report;
  }

  getSystemHealth() {
    const activeStreams = Array.from(this.dataStreams.values()).filter(
      (s) => s.data.length > 0,
    );
    const recentAnalyses = Array.from(this.analysisResults.values()).filter(
      (r) => Date.now() - r.timestamp < 5 * 60 * 1000,
    ); // Last 5 minutes

    return {
      status: activeStreams.length > 0 ? "active" : "idle",
      active_streams: activeStreams.length,
      recent_analyses: recentAnalyses.length,
      avg_analysis_time:
        recentAnalyses.length > 0
          ? recentAnalyses.reduce((sum, r) => sum + r.analysisTime, 0) /
            recentAnalyses.length
          : 0,
      memory_usage: this.getMemoryUsage(),
    };
  }

  getMemoryUsage() {
    const totalDataPoints = Array.from(this.dataStreams.values()).reduce(
      (sum, stream) => sum + stream.data.length,
      0,
    );

    return {
      total_data_points: totalDataPoints,
      estimated_memory_mb: (totalDataPoints * 16) / 1024 / 1024, // Rough estimate
      streams: this.dataStreams.size,
      results: this.analysisResults.size,
    };
  }

  /**
   * Cleanup
   */
  destroy() {
    this.stopRealTimeAnalysis();
    this.dataStreams.clear();
    this.analysisResults.clear();
    this.eventListeners.clear();
  }
}

// Export classes
if (typeof module !== "undefined" && module.exports) {
  module.exports = {
    AdvancedAnalyticsEngine,
    TrendDetector,
    SmartAlertSystem,
    StatisticalAnalysis,
  };
} else {
  // Browser environment
  window.AdvancedAnalyticsEngine = AdvancedAnalyticsEngine;
  window.TrendDetector = TrendDetector;
  window.SmartAlertSystem = SmartAlertSystem;
  window.StatisticalAnalysis = StatisticalAnalysis;
}
\n\n// auth-service.js\n/**
 * Authentication Service
 *
 * Comprehensive authentication and authorization service with JWT tokens,
 * role-based access control, and session management
 */

export class AuthService {
  constructor(options = {}) {
    this.options = {
      apiBaseUrl: options.apiBaseUrl || "/api/auth",
      tokenKey: options.tokenKey || "pynomaly_token",
      refreshTokenKey: options.refreshTokenKey || "pynomaly_refresh_token",
      userKey: options.userKey || "pynomaly_user",
      autoRefresh: options.autoRefresh !== false,
      refreshThreshold: options.refreshThreshold || 300000, // 5 minutes
      sessionTimeout: options.sessionTimeout || 3600000, // 1 hour
      enableLogging: options.enableLogging || false,
      ...options,
    };

    this.currentUser = null;
    this.token = null;
    this.refreshToken = null;
    this.refreshTimer = null;
    this.sessionTimer = null;
    this.listeners = new Map();

    this.init();
  }

  init() {
    this.loadStoredAuth();
    this.startTokenRefreshTimer();
    this.bindEvents();
  }

  bindEvents() {
    // Listen for storage changes in other tabs
    window.addEventListener("storage", (e) => {
      if (e.key === this.options.tokenKey || e.key === this.options.userKey) {
        this.loadStoredAuth();
      }
    });

    // Listen for visibility changes to refresh tokens when tab becomes active
    document.addEventListener("visibilitychange", () => {
      if (
        !document.hidden &&
        this.isAuthenticated() &&
        this.shouldRefreshToken()
      ) {
        this.refreshAccessToken();
      }
    });
  }

  // Authentication methods
  async login(credentials) {
    try {
      this.log("Attempting login for user:", credentials.username);

      const response = await this.makeRequest("/login", {
        method: "POST",
        body: JSON.stringify(credentials),
      });

      if (response.success) {
        await this.handleAuthResponse(response);
        this.emit("login", { user: this.currentUser });
        return { success: true, user: this.currentUser };
      } else {
        throw new Error(response.message || "Login failed");
      }
    } catch (error) {
      this.log("Login error:", error);
      this.emit("login_error", { error: error.message });
      throw error;
    }
  }

  async register(userData) {
    try {
      this.log("Attempting registration for user:", userData.username);

      const response = await this.makeRequest("/register", {
        method: "POST",
        body: JSON.stringify(userData),
      });

      if (response.success) {
        // Don't auto-login after registration, require email verification
        this.emit("registration_success", { message: response.message });
        return { success: true, message: response.message };
      } else {
        throw new Error(response.message || "Registration failed");
      }
    } catch (error) {
      this.log("Registration error:", error);
      this.emit("registration_error", { error: error.message });
      throw error;
    }
  }

  async logout() {
    try {
      if (this.token) {
        // Notify server about logout
        await this.makeRequest("/logout", {
          method: "POST",
          headers: {
            Authorization: `Bearer ${this.token}`,
          },
        });
      }
    } catch (error) {
      this.log("Logout API error (continuing with local logout):", error);
    } finally {
      this.clearAuth();
      this.emit("logout");
    }
  }

  async refreshAccessToken() {
    if (!this.refreshToken) {
      throw new Error("No refresh token available");
    }

    try {
      this.log("Refreshing access token");

      const response = await this.makeRequest("/refresh", {
        method: "POST",
        body: JSON.stringify({ refreshToken: this.refreshToken }),
      });

      if (response.success) {
        this.setToken(response.accessToken);
        if (response.refreshToken) {
          this.setRefreshToken(response.refreshToken);
        }
        this.emit("token_refreshed");
        return response.accessToken;
      } else {
        throw new Error(response.message || "Token refresh failed");
      }
    } catch (error) {
      this.log("Token refresh error:", error);
      this.clearAuth();
      this.emit("token_refresh_error", { error: error.message });
      throw error;
    }
  }

  async verifyEmail(token) {
    try {
      const response = await this.makeRequest("/verify-email", {
        method: "POST",
        body: JSON.stringify({ token }),
      });

      if (response.success) {
        this.emit("email_verified", { message: response.message });
        return { success: true, message: response.message };
      } else {
        throw new Error(response.message || "Email verification failed");
      }
    } catch (error) {
      this.log("Email verification error:", error);
      this.emit("email_verification_error", { error: error.message });
      throw error;
    }
  }

  async requestPasswordReset(email) {
    try {
      const response = await this.makeRequest("/request-password-reset", {
        method: "POST",
        body: JSON.stringify({ email }),
      });

      if (response.success) {
        this.emit("password_reset_requested", { message: response.message });
        return { success: true, message: response.message };
      } else {
        throw new Error(response.message || "Password reset request failed");
      }
    } catch (error) {
      this.log("Password reset request error:", error);
      this.emit("password_reset_error", { error: error.message });
      throw error;
    }
  }

  async resetPassword(token, newPassword) {
    try {
      const response = await this.makeRequest("/reset-password", {
        method: "POST",
        body: JSON.stringify({ token, newPassword }),
      });

      if (response.success) {
        this.emit("password_reset_success", { message: response.message });
        return { success: true, message: response.message };
      } else {
        throw new Error(response.message || "Password reset failed");
      }
    } catch (error) {
      this.log("Password reset error:", error);
      this.emit("password_reset_error", { error: error.message });
      throw error;
    }
  }

  async changePassword(currentPassword, newPassword) {
    try {
      const response = await this.makeAuthenticatedRequest("/change-password", {
        method: "POST",
        body: JSON.stringify({ currentPassword, newPassword }),
      });

      if (response.success) {
        this.emit("password_changed", { message: response.message });
        return { success: true, message: response.message };
      } else {
        throw new Error(response.message || "Password change failed");
      }
    } catch (error) {
      this.log("Password change error:", error);
      this.emit("password_change_error", { error: error.message });
      throw error;
    }
  }

  async updateProfile(profileData) {
    try {
      const response = await this.makeAuthenticatedRequest("/profile", {
        method: "PUT",
        body: JSON.stringify(profileData),
      });

      if (response.success) {
        this.currentUser = { ...this.currentUser, ...response.user };
        this.storeUser(this.currentUser);
        this.emit("profile_updated", { user: this.currentUser });
        return { success: true, user: this.currentUser };
      } else {
        throw new Error(response.message || "Profile update failed");
      }
    } catch (error) {
      this.log("Profile update error:", error);
      this.emit("profile_update_error", { error: error.message });
      throw error;
    }
  }

  // Authorization methods
  hasRole(role) {
    return (
      this.currentUser &&
      this.currentUser.roles &&
      this.currentUser.roles.includes(role)
    );
  }

  hasPermission(permission) {
    return (
      this.currentUser &&
      this.currentUser.permissions &&
      this.currentUser.permissions.includes(permission)
    );
  }

  hasAnyRole(roles) {
    return roles.some((role) => this.hasRole(role));
  }

  hasAllRoles(roles) {
    return roles.every((role) => this.hasRole(role));
  }

  hasAnyPermission(permissions) {
    return permissions.some((permission) => this.hasPermission(permission));
  }

  hasAllPermissions(permissions) {
    return permissions.every((permission) => this.hasPermission(permission));
  }

  canAccess(resource, action = "read") {
    // Basic RBAC implementation
    const resourcePermissions = this.getResourcePermissions(resource);
    return (
      resourcePermissions.includes(`${resource}:${action}`) ||
      resourcePermissions.includes(`${resource}:*`) ||
      this.hasRole("admin")
    );
  }

  getResourcePermissions(resource) {
    if (!this.currentUser || !this.currentUser.permissions) return [];

    return this.currentUser.permissions.filter((permission) =>
      permission.startsWith(resource + ":"),
    );
  }

  // Token management
  setToken(token) {
    this.token = token;
    this.storeToken(token);
    this.startTokenRefreshTimer();
    this.startSessionTimer();
  }

  setRefreshToken(refreshToken) {
    this.refreshToken = refreshToken;
    this.storeRefreshToken(refreshToken);
  }

  getToken() {
    return this.token;
  }

  isAuthenticated() {
    return !!this.token && !!this.currentUser && !this.isTokenExpired();
  }

  isTokenExpired() {
    if (!this.token) return true;

    try {
      const payload = this.parseJWT(this.token);
      return payload.exp * 1000 < Date.now();
    } catch (error) {
      this.log("Error parsing token:", error);
      return true;
    }
  }

  shouldRefreshToken() {
    if (!this.token) return false;

    try {
      const payload = this.parseJWT(this.token);
      const timeUntilExpiry = payload.exp * 1000 - Date.now();
      return timeUntilExpiry < this.options.refreshThreshold;
    } catch (error) {
      return false;
    }
  }

  parseJWT(token) {
    const base64Url = token.split(".")[1];
    const base64 = base64Url.replace(/-/g, "+").replace(/_/g, "/");
    const jsonPayload = decodeURIComponent(
      atob(base64)
        .split("")
        .map(function (c) {
          return "%" + ("00" + c.charCodeAt(0).toString(16)).slice(-2);
        })
        .join(""),
    );

    return JSON.parse(jsonPayload);
  }

  // Session management
  startSessionTimer() {
    if (this.sessionTimer) {
      clearTimeout(this.sessionTimer);
    }

    this.sessionTimer = setTimeout(() => {
      this.emit("session_expired");
      this.clearAuth();
    }, this.options.sessionTimeout);
  }

  extendSession() {
    this.startSessionTimer();
    this.emit("session_extended");
  }

  startTokenRefreshTimer() {
    if (!this.options.autoRefresh || this.refreshTimer) {
      return;
    }

    if (this.shouldRefreshToken()) {
      this.refreshAccessToken().catch((error) => {
        this.log("Auto-refresh failed:", error);
      });
    }

    this.refreshTimer = setInterval(() => {
      if (this.isAuthenticated() && this.shouldRefreshToken()) {
        this.refreshAccessToken().catch((error) => {
          this.log("Auto-refresh failed:", error);
        });
      }
    }, 60000); // Check every minute
  }

  stopTokenRefreshTimer() {
    if (this.refreshTimer) {
      clearInterval(this.refreshTimer);
      this.refreshTimer = null;
    }
  }

  // Storage methods
  storeToken(token) {
    try {
      localStorage.setItem(this.options.tokenKey, token);
    } catch (error) {
      this.log("Error storing token:", error);
    }
  }

  storeRefreshToken(refreshToken) {
    try {
      localStorage.setItem(this.options.refreshTokenKey, refreshToken);
    } catch (error) {
      this.log("Error storing refresh token:", error);
    }
  }

  storeUser(user) {
    try {
      localStorage.setItem(this.options.userKey, JSON.stringify(user));
    } catch (error) {
      this.log("Error storing user:", error);
    }
  }

  loadStoredAuth() {
    try {
      const token = localStorage.getItem(this.options.tokenKey);
      const refreshToken = localStorage.getItem(this.options.refreshTokenKey);
      const userJson = localStorage.getItem(this.options.userKey);

      if (token && userJson) {
        this.token = token;
        this.refreshToken = refreshToken;
        this.currentUser = JSON.parse(userJson);

        if (this.isTokenExpired()) {
          if (this.refreshToken) {
            this.refreshAccessToken().catch(() => {
              this.clearAuth();
            });
          } else {
            this.clearAuth();
          }
        } else {
          this.startTokenRefreshTimer();
          this.startSessionTimer();
          this.emit("auth_restored", { user: this.currentUser });
        }
      }
    } catch (error) {
      this.log("Error loading stored auth:", error);
      this.clearAuth();
    }
  }

  clearAuth() {
    this.currentUser = null;
    this.token = null;
    this.refreshToken = null;

    this.stopTokenRefreshTimer();

    if (this.sessionTimer) {
      clearTimeout(this.sessionTimer);
      this.sessionTimer = null;
    }

    try {
      localStorage.removeItem(this.options.tokenKey);
      localStorage.removeItem(this.options.refreshTokenKey);
      localStorage.removeItem(this.options.userKey);
    } catch (error) {
      this.log("Error clearing auth storage:", error);
    }
  }

  // HTTP request methods
  async makeRequest(endpoint, options = {}) {
    const url = `${this.options.apiBaseUrl}${endpoint}`;

    const defaultOptions = {
      headers: {
        "Content-Type": "application/json",
        ...options.headers,
      },
    };

    const requestOptions = { ...defaultOptions, ...options };

    const response = await fetch(url, requestOptions);

    if (!response.ok) {
      if (response.status === 401) {
        this.clearAuth();
        this.emit("unauthorized");
      }
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    return await response.json();
  }

  async makeAuthenticatedRequest(endpoint, options = {}) {
    if (!this.isAuthenticated()) {
      throw new Error("Not authenticated");
    }

    const authOptions = {
      ...options,
      headers: {
        Authorization: `Bearer ${this.token}`,
        ...options.headers,
      },
    };

    try {
      return await this.makeRequest(endpoint, authOptions);
    } catch (error) {
      if (error.message.includes("401") && this.refreshToken) {
        // Try to refresh token and retry request
        await this.refreshAccessToken();
        authOptions.headers["Authorization"] = `Bearer ${this.token}`;
        return await this.makeRequest(endpoint, authOptions);
      }
      throw error;
    }
  }

  // User management methods
  async getUsers(filters = {}) {
    const queryParams = new URLSearchParams(filters).toString();
    const endpoint = `/users${queryParams ? `?${queryParams}` : ""}`;

    const response = await this.makeAuthenticatedRequest(endpoint);
    return response.users || [];
  }

  async createUser(userData) {
    const response = await this.makeAuthenticatedRequest("/users", {
      method: "POST",
      body: JSON.stringify(userData),
    });

    this.emit("user_created", { user: response.user });
    return response.user;
  }

  async updateUser(userId, userData) {
    const response = await this.makeAuthenticatedRequest(`/users/${userId}`, {
      method: "PUT",
      body: JSON.stringify(userData),
    });

    this.emit("user_updated", { user: response.user });
    return response.user;
  }

  async deleteUser(userId) {
    await this.makeAuthenticatedRequest(`/users/${userId}`, {
      method: "DELETE",
    });

    this.emit("user_deleted", { userId });
  }

  async assignRole(userId, role) {
    const response = await this.makeAuthenticatedRequest(
      `/users/${userId}/roles`,
      {
        method: "POST",
        body: JSON.stringify({ role }),
      },
    );

    this.emit("role_assigned", { userId, role });
    return response;
  }

  async removeRole(userId, role) {
    const response = await this.makeAuthenticatedRequest(
      `/users/${userId}/roles/${role}`,
      {
        method: "DELETE",
      },
    );

    this.emit("role_removed", { userId, role });
    return response;
  }

  // Utility methods
  async handleAuthResponse(response) {
    if (response.accessToken) {
      this.setToken(response.accessToken);
    }

    if (response.refreshToken) {
      this.setRefreshToken(response.refreshToken);
    }

    if (response.user) {
      this.currentUser = response.user;
      this.storeUser(this.currentUser);
    }
  }

  getCurrentUser() {
    return this.currentUser;
  }

  getUserRoles() {
    return this.currentUser ? this.currentUser.roles || [] : [];
  }

  getUserPermissions() {
    return this.currentUser ? this.currentUser.permissions || [] : [];
  }

  // Event management
  on(event, listener) {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, new Set());
    }
    this.listeners.get(event).add(listener);

    return () => this.off(event, listener);
  }

  off(event, listener) {
    const eventListeners = this.listeners.get(event);
    if (eventListeners) {
      eventListeners.delete(listener);
      if (eventListeners.size === 0) {
        this.listeners.delete(event);
      }
    }
  }

  emit(event, data) {
    const eventListeners = this.listeners.get(event);
    if (eventListeners) {
      eventListeners.forEach((listener) => {
        try {
          listener(data);
        } catch (error) {
          this.log("Error in event listener:", error);
        }
      });
    }
  }

  log(...args) {
    if (this.options.enableLogging) {
      console.log("[AuthService]", ...args);
    }
  }

  // Cleanup
  destroy() {
    this.clearAuth();
    this.listeners.clear();

    window.removeEventListener("storage", this.loadStoredAuth);
    document.removeEventListener(
      "visibilitychange",
      this.handleVisibilityChange,
    );
  }
}

/**
 * Role-Based Access Control Helper
 */
export class RBACHelper {
  constructor(authService) {
    this.authService = authService;
  }

  // UI element access control
  showIfAuthorized(element, requirements) {
    const isAuthorized = this.checkAuthorization(requirements);
    if (element) {
      element.style.display = isAuthorized ? "" : "none";
    }
    return isAuthorized;
  }

  enableIfAuthorized(element, requirements) {
    const isAuthorized = this.checkAuthorization(requirements);
    if (element) {
      element.disabled = !isAuthorized;
    }
    return isAuthorized;
  }

  checkAuthorization(requirements) {
    if (!this.authService.isAuthenticated()) {
      return false;
    }

    if (requirements.roles) {
      if (requirements.requireAll) {
        if (!this.authService.hasAllRoles(requirements.roles)) {
          return false;
        }
      } else {
        if (!this.authService.hasAnyRole(requirements.roles)) {
          return false;
        }
      }
    }

    if (requirements.permissions) {
      if (requirements.requireAll) {
        if (!this.authService.hasAllPermissions(requirements.permissions)) {
          return false;
        }
      } else {
        if (!this.authService.hasAnyPermission(requirements.permissions)) {
          return false;
        }
      }
    }

    if (requirements.custom) {
      return requirements.custom(this.authService.getCurrentUser());
    }

    return true;
  }

  // Route protection
  protectRoute(routeHandler, requirements) {
    return (...args) => {
      if (this.checkAuthorization(requirements)) {
        return routeHandler(...args);
      } else {
        throw new Error("Access denied");
      }
    };
  }

  // API request protection
  protectApiCall(apiCall, requirements) {
    return async (...args) => {
      if (this.checkAuthorization(requirements)) {
        return await apiCall(...args);
      } else {
        throw new Error("Access denied");
      }
    };
  }
}

// Default export
export default AuthService;
\n\n// automl-service.js\n/**
 * AutoML Service for Pynomaly
 * Automated model training, hyperparameter optimization, and model selection
 * Provides enterprise-grade machine learning automation capabilities
 */

/**
 * Hyperparameter Optimization Algorithms
 */
const OPTIMIZATION_ALGORITHMS = {
  GRID_SEARCH: "grid_search",
  RANDOM_SEARCH: "random_search",
  BAYESIAN: "bayesian",
  EVOLUTIONARY: "evolutionary",
  OPTUNA: "optuna",
  HYPEROPT: "hyperopt",
};

/**
 * Model Selection Strategies
 */
const MODEL_SELECTION_STRATEGIES = {
  BEST_SINGLE: "best_single",
  ENSEMBLE: "ensemble",
  STACKING: "stacking",
  VOTING: "voting",
  WEIGHTED_AVERAGE: "weighted_average",
};

/**
 * AutoML Pipeline Status
 */
const PIPELINE_STATUS = {
  INITIALIZING: "initializing",
  DATA_PREPROCESSING: "data_preprocessing",
  FEATURE_ENGINEERING: "feature_engineering",
  MODEL_SEARCH: "model_search",
  HYPERPARAMETER_OPTIMIZATION: "hyperparameter_optimization",
  MODEL_VALIDATION: "model_validation",
  ENSEMBLE_CREATION: "ensemble_creation",
  FINAL_TRAINING: "final_training",
  COMPLETED: "completed",
  FAILED: "failed",
  CANCELLED: "cancelled",
};

/**
 * AutoML Configuration Manager
 * Manages AutoML pipeline configurations and defaults
 */
class AutoMLConfig {
  constructor() {
    this.defaultConfig = {
      // Data preprocessing
      preprocessing: {
        handle_missing: "auto", // 'auto', 'drop', 'impute'
        scaling: "auto", // 'auto', 'standard', 'minmax', 'robust', 'none'
        feature_selection: true,
        max_features: 1000,
        categorical_encoding: "auto", // 'auto', 'onehot', 'label', 'target'
      },

      // Feature engineering
      feature_engineering: {
        polynomial_features: false,
        interaction_features: true,
        statistical_features: true,
        time_features: true, // For time series data
        max_polynomial_degree: 2,
      },

      // Model search
      model_search: {
        algorithms: [
          "isolation_forest",
          "local_outlier_factor",
          "one_class_svm",
          "elliptic_envelope",
          "autoencoder",
          "deep_svdd",
          "copod",
          "ecod",
          "feature_bagging",
          "histogram_based",
        ],
        max_trials: 50,
        early_stopping: true,
        early_stopping_patience: 10,
      },

      // Hyperparameter optimization
      hyperparameter_optimization: {
        algorithm: OPTIMIZATION_ALGORITHMS.BAYESIAN,
        max_evaluations: 100,
        timeout_minutes: 60,
        n_jobs: -1,
        cv_folds: 5,
        scoring_metric: "roc_auc",
        optimization_direction: "maximize",
      },

      // Model validation
      validation: {
        test_size: 0.2,
        validation_size: 0.1,
        cross_validation: true,
        cv_folds: 5,
        stratify: false, // For anomaly detection, usually false
        shuffle: true,
        random_state: 42,
      },

      // Ensemble methods
      ensemble: {
        enable: true,
        strategy: MODEL_SELECTION_STRATEGIES.ENSEMBLE,
        max_models: 5,
        ensemble_method: "voting", // 'voting', 'stacking', 'blending'
        meta_learner: "logistic_regression",
      },

      // Performance and resources
      performance: {
        max_training_time_minutes: 120,
        memory_limit_gb: 8,
        gpu_enabled: false,
        distributed: false,
        n_jobs: -1,
      },

      // Monitoring and logging
      monitoring: {
        log_level: "INFO",
        save_intermediate_results: true,
        progress_reporting_interval: 30, // seconds
        checkpoint_frequency: 10, // trials
      },
    };
  }

  createConfig(userConfig = {}) {
    return this.deepMerge(this.defaultConfig, userConfig);
  }

  validateConfig(config) {
    const errors = [];

    // Validate required fields
    if (!config.model_search?.algorithms?.length) {
      errors.push("At least one algorithm must be specified");
    }

    if (config.hyperparameter_optimization?.max_evaluations <= 0) {
      errors.push("max_evaluations must be positive");
    }

    if (
      config.validation?.test_size <= 0 ||
      config.validation?.test_size >= 1
    ) {
      errors.push("test_size must be between 0 and 1");
    }

    // Validate algorithm availability
    const availableAlgorithms = [
      "isolation_forest",
      "local_outlier_factor",
      "one_class_svm",
      "elliptic_envelope",
      "autoencoder",
      "deep_svdd",
      "copod",
      "ecod",
      "feature_bagging",
      "histogram_based",
    ];

    const invalidAlgorithms = config.model_search.algorithms.filter(
      (alg) => !availableAlgorithms.includes(alg),
    );

    if (invalidAlgorithms.length > 0) {
      errors.push(`Invalid algorithms: ${invalidAlgorithms.join(", ")}`);
    }

    return {
      isValid: errors.length === 0,
      errors,
    };
  }

  deepMerge(target, source) {
    const result = { ...target };

    for (const key in source) {
      if (
        source[key] &&
        typeof source[key] === "object" &&
        !Array.isArray(source[key])
      ) {
        result[key] = this.deepMerge(target[key] || {}, source[key]);
      } else {
        result[key] = source[key];
      }
    }

    return result;
  }
}

/**
 * AutoML Pipeline Manager
 * Orchestrates the entire AutoML process
 */
class AutoMLPipeline {
  constructor(config = {}) {
    this.configManager = new AutoMLConfig();
    this.config = this.configManager.createConfig(config);

    // Pipeline state
    this.pipelineId = this.generatePipelineId();
    this.status = PIPELINE_STATUS.INITIALIZING;
    this.currentStep = 0;
    this.totalSteps = 7;
    this.startTime = null;
    this.endTime = null;

    // Results storage
    this.results = {
      preprocessing: null,
      feature_engineering: null,
      model_trials: [],
      best_models: [],
      ensemble_model: null,
      final_model: null,
      evaluation_metrics: null,
      training_history: [],
    };

    // Progress tracking
    this.progress = {
      overall: 0,
      step_progress: 0,
      current_trial: 0,
      total_trials: 0,
      estimated_time_remaining: null,
    };

    // Event system
    this.eventListeners = new Map();

    // Cancellation support
    this.isCancelled = false;
    this.cancelToken = null;
  }

  /**
   * Main pipeline execution
   */
  async run(dataset, target_column = null) {
    try {
      this.startTime = Date.now();
      this.status = PIPELINE_STATUS.INITIALIZING;
      this.emit("pipeline_started", {
        pipelineId: this.pipelineId,
        config: this.config,
      });

      // Validate configuration
      const configValidation = this.configManager.validateConfig(this.config);
      if (!configValidation.isValid) {
        throw new Error(
          `Configuration validation failed: ${configValidation.errors.join(", ")}`,
        );
      }

      // Step 1: Data preprocessing
      await this.executeStep(PIPELINE_STATUS.DATA_PREPROCESSING, async () => {
        this.results.preprocessing = await this.preprocessData(
          dataset,
          target_column,
        );
      });

      // Step 2: Feature engineering
      await this.executeStep(PIPELINE_STATUS.FEATURE_ENGINEERING, async () => {
        this.results.feature_engineering = await this.engineerFeatures(
          this.results.preprocessing.processed_data,
        );
      });

      // Step 3: Model search and hyperparameter optimization
      await this.executeStep(PIPELINE_STATUS.MODEL_SEARCH, async () => {
        this.results.model_trials = await this.searchModels(
          this.results.feature_engineering.features,
          this.results.preprocessing.target,
        );
      });

      // Step 4: Model validation
      await this.executeStep(PIPELINE_STATUS.MODEL_VALIDATION, async () => {
        this.results.best_models = await this.validateModels(
          this.results.model_trials,
        );
      });

      // Step 5: Ensemble creation (if enabled)
      if (this.config.ensemble.enable) {
        await this.executeStep(PIPELINE_STATUS.ENSEMBLE_CREATION, async () => {
          this.results.ensemble_model = await this.createEnsemble(
            this.results.best_models,
          );
        });
      }

      // Step 6: Final training
      await this.executeStep(PIPELINE_STATUS.FINAL_TRAINING, async () => {
        this.results.final_model = await this.finalTraining();
      });

      // Complete pipeline
      this.status = PIPELINE_STATUS.COMPLETED;
      this.endTime = Date.now();
      this.progress.overall = 100;

      const finalResults = this.compileFinalResults();
      this.emit("pipeline_completed", finalResults);

      return finalResults;
    } catch (error) {
      this.status = PIPELINE_STATUS.FAILED;
      this.endTime = Date.now();
      this.emit("pipeline_failed", {
        error: error.message,
        pipelineId: this.pipelineId,
      });
      throw error;
    }
  }

  async executeStep(stepStatus, stepFunction) {
    if (this.isCancelled) {
      throw new Error("Pipeline was cancelled");
    }

    this.status = stepStatus;
    this.currentStep++;
    this.progress.overall = Math.round(
      (this.currentStep / this.totalSteps) * 100,
    );

    this.emit("step_started", {
      step: stepStatus,
      progress: this.progress.overall,
      step_number: this.currentStep,
      total_steps: this.totalSteps,
    });

    const stepStartTime = Date.now();

    try {
      await stepFunction();

      const stepDuration = Date.now() - stepStartTime;
      this.emit("step_completed", {
        step: stepStatus,
        duration: stepDuration,
        progress: this.progress.overall,
      });
    } catch (error) {
      this.emit("step_failed", {
        step: stepStatus,
        error: error.message,
      });
      throw error;
    }
  }

  /**
   * Data Preprocessing
   */
  async preprocessData(dataset, target_column) {
    this.emit("preprocessing_started", {
      dataset_shape: this.getDatasetShape(dataset),
    });

    const preprocessingResult = {
      original_shape: this.getDatasetShape(dataset),
      processed_data: null,
      target: null,
      preprocessing_steps: [],
      feature_names: [],
      statistics: {},
    };

    // Simulate preprocessing steps
    const steps = [
      "Analyzing data structure",
      "Handling missing values",
      "Feature scaling",
      "Outlier detection",
      "Feature selection",
    ];

    for (let i = 0; i < steps.length; i++) {
      if (this.isCancelled) return;

      this.progress.step_progress = Math.round(((i + 1) / steps.length) * 100);
      this.emit("preprocessing_progress", {
        step: steps[i],
        progress: this.progress.step_progress,
      });

      // Simulate processing time
      await this.sleep(1000);

      preprocessingResult.preprocessing_steps.push({
        step: steps[i],
        completed: true,
        timestamp: Date.now(),
      });
    }

    // Generate mock processed data
    preprocessingResult.processed_data =
      this.generateMockProcessedData(dataset);
    preprocessingResult.feature_names = this.generateFeatureNames(
      preprocessingResult.processed_data,
    );
    preprocessingResult.statistics = this.calculateDataStatistics(
      preprocessingResult.processed_data,
    );

    this.emit("preprocessing_completed", preprocessingResult);
    return preprocessingResult;
  }

  /**
   * Feature Engineering
   */
  async engineerFeatures(processedData) {
    this.emit("feature_engineering_started");

    const featureResult = {
      original_features: processedData.length,
      engineered_features: null,
      feature_importance: {},
      feature_transformations: [],
      new_feature_count: 0,
    };

    const engineeringSteps = [
      "Statistical feature generation",
      "Polynomial features",
      "Interaction features",
      "Time-based features",
      "Feature selection",
    ];

    for (let i = 0; i < engineeringSteps.length; i++) {
      if (this.isCancelled) return;

      this.progress.step_progress = Math.round(
        ((i + 1) / engineeringSteps.length) * 100,
      );
      this.emit("feature_engineering_progress", {
        step: engineeringSteps[i],
        progress: this.progress.step_progress,
      });

      await this.sleep(800);

      featureResult.feature_transformations.push({
        transformation: engineeringSteps[i],
        features_added: Math.floor(Math.random() * 10) + 1,
        timestamp: Date.now(),
      });
    }

    // Generate mock feature engineering results
    featureResult.engineered_features =
      this.generateMockEngineeredFeatures(processedData);
    featureResult.new_feature_count =
      featureResult.engineered_features.length -
      featureResult.original_features;
    featureResult.feature_importance = this.generateMockFeatureImportance(
      featureResult.engineered_features,
    );

    this.emit("feature_engineering_completed", featureResult);
    return featureResult;
  }

  /**
   * Model Search and Hyperparameter Optimization
   */
  async searchModels(features, target) {
    this.emit("model_search_started", {
      algorithms: this.config.model_search.algorithms,
      max_trials: this.config.hyperparameter_optimization.max_evaluations,
    });

    const modelTrials = [];
    const algorithms = this.config.model_search.algorithms;
    const trialsPerAlgorithm = Math.floor(
      this.config.hyperparameter_optimization.max_evaluations /
        algorithms.length,
    );

    this.progress.total_trials = algorithms.length * trialsPerAlgorithm;
    this.progress.current_trial = 0;

    for (const algorithm of algorithms) {
      if (this.isCancelled) return modelTrials;

      this.emit("algorithm_started", { algorithm });

      // Perform hyperparameter optimization for this algorithm
      const algorithmTrials = await this.optimizeHyperparameters(
        algorithm,
        features,
        target,
        trialsPerAlgorithm,
      );
      modelTrials.push(...algorithmTrials);

      this.emit("algorithm_completed", {
        algorithm,
        trials: algorithmTrials.length,
        best_score: Math.max(...algorithmTrials.map((t) => t.score)),
      });
    }

    // Sort trials by score
    modelTrials.sort((a, b) => b.score - a.score);

    this.emit("model_search_completed", {
      total_trials: modelTrials.length,
      best_score: modelTrials[0]?.score,
      best_algorithm: modelTrials[0]?.algorithm,
    });

    return modelTrials;
  }

  async optimizeHyperparameters(algorithm, features, target, maxTrials) {
    const trials = [];
    const hyperparameterSpace = this.getHyperparameterSpace(algorithm);

    for (let i = 0; i < maxTrials; i++) {
      if (this.isCancelled) return trials;

      this.progress.current_trial++;
      this.progress.step_progress = Math.round(
        (this.progress.current_trial / this.progress.total_trials) * 100,
      );

      // Generate hyperparameters based on optimization algorithm
      const hyperparameters = this.sampleHyperparameters(
        hyperparameterSpace,
        this.config.hyperparameter_optimization.algorithm,
      );

      // Simulate model training and evaluation
      const trial = await this.evaluateModel(
        algorithm,
        hyperparameters,
        features,
        target,
      );
      trials.push(trial);

      this.emit("trial_completed", {
        trial_number: this.progress.current_trial,
        algorithm,
        score: trial.score,
        hyperparameters: trial.hyperparameters,
        progress: this.progress.step_progress,
      });

      // Simulate training time
      await this.sleep(500);
    }

    return trials;
  }

  async evaluateModel(algorithm, hyperparameters, features, target) {
    // Simulate cross-validation
    const cvScores = [];
    for (let fold = 0; fold < this.config.validation.cv_folds; fold++) {
      // Generate realistic but mock score
      const baseScore = this.getBaseScoreForAlgorithm(algorithm);
      const noise = (Math.random() - 0.5) * 0.1;
      cvScores.push(Math.max(0, Math.min(1, baseScore + noise)));
    }

    const score = cvScores.reduce((sum, s) => sum + s, 0) / cvScores.length;
    const std = Math.sqrt(
      cvScores.reduce((sum, s) => sum + Math.pow(s - score, 2), 0) /
        cvScores.length,
    );

    return {
      trial_id: this.generateTrialId(),
      algorithm,
      hyperparameters,
      score,
      cv_scores: cvScores,
      cv_std: std,
      training_time: Math.random() * 60 + 10, // 10-70 seconds
      evaluation_metrics: this.generateMockEvaluationMetrics(score),
      timestamp: Date.now(),
    };
  }

  /**
   * Model Validation
   */
  async validateModels(modelTrials) {
    this.emit("model_validation_started", { total_models: modelTrials.length });

    // Select top models for validation
    const topModels = modelTrials.slice(0, Math.min(10, modelTrials.length));
    const validatedModels = [];

    for (let i = 0; i < topModels.length; i++) {
      if (this.isCancelled) return validatedModels;

      const model = topModels[i];
      this.progress.step_progress = Math.round(
        ((i + 1) / topModels.length) * 100,
      );

      this.emit("model_validation_progress", {
        model_index: i + 1,
        total_models: topModels.length,
        algorithm: model.algorithm,
        progress: this.progress.step_progress,
      });

      // Perform detailed validation
      const validationResult = await this.performDetailedValidation(model);
      validatedModels.push({
        ...model,
        validation: validationResult,
      });

      await this.sleep(600);
    }

    // Sort by validation score
    validatedModels.sort((a, b) => b.validation.score - a.validation.score);

    this.emit("model_validation_completed", {
      validated_models: validatedModels.length,
      best_validation_score: validatedModels[0]?.validation.score,
    });

    return validatedModels;
  }

  async performDetailedValidation(model) {
    // Simulate detailed validation including various metrics
    const baseScore = model.score;
    const noise = (Math.random() - 0.5) * 0.05;

    return {
      score: Math.max(0, Math.min(1, baseScore + noise)),
      precision: Math.max(
        0,
        Math.min(1, baseScore + (Math.random() - 0.5) * 0.1),
      ),
      recall: Math.max(0, Math.min(1, baseScore + (Math.random() - 0.5) * 0.1)),
      f1_score: Math.max(
        0,
        Math.min(1, baseScore + (Math.random() - 0.5) * 0.08),
      ),
      roc_auc: Math.max(
        0,
        Math.min(1, baseScore + (Math.random() - 0.5) * 0.06),
      ),
      confusion_matrix: this.generateMockConfusionMatrix(),
      feature_importance: this.generateMockFeatureImportance(),
      validation_time: Math.random() * 30 + 5,
    };
  }

  /**
   * Ensemble Creation
   */
  async createEnsemble(validatedModels) {
    if (!this.config.ensemble.enable || validatedModels.length < 2) {
      return null;
    }

    this.emit("ensemble_creation_started", {
      strategy: this.config.ensemble.strategy,
      num_models: Math.min(
        this.config.ensemble.max_models,
        validatedModels.length,
      ),
    });

    const selectedModels = validatedModels.slice(
      0,
      this.config.ensemble.max_models,
    );

    // Simulate ensemble creation
    const ensembleSteps = [
      "Model selection for ensemble",
      "Weight optimization",
      "Ensemble training",
      "Ensemble validation",
    ];

    for (let i = 0; i < ensembleSteps.length; i++) {
      if (this.isCancelled) return null;

      this.progress.step_progress = Math.round(
        ((i + 1) / ensembleSteps.length) * 100,
      );
      this.emit("ensemble_progress", {
        step: ensembleSteps[i],
        progress: this.progress.step_progress,
      });

      await this.sleep(800);
    }

    const ensembleResult = {
      ensemble_id: this.generateEnsembleId(),
      strategy: this.config.ensemble.strategy,
      models: selectedModels.map((m) => ({
        algorithm: m.algorithm,
        weight: Math.random(),
        model_id: m.trial_id,
      })),
      ensemble_score:
        Math.max(...selectedModels.map((m) => m.validation.score)) + 0.02,
      creation_time: Date.now(),
    };

    // Normalize weights
    const totalWeight = ensembleResult.models.reduce(
      (sum, m) => sum + m.weight,
      0,
    );
    ensembleResult.models.forEach((m) => {
      m.weight = m.weight / totalWeight;
    });

    this.emit("ensemble_creation_completed", ensembleResult);
    return ensembleResult;
  }

  /**
   * Final Training
   */
  async finalTraining() {
    this.emit("final_training_started");

    const finalModel =
      this.results.ensemble_model || this.results.best_models[0];

    const trainingSteps = [
      "Preparing full dataset",
      "Training final model",
      "Model serialization",
      "Performance validation",
    ];

    for (let i = 0; i < trainingSteps.length; i++) {
      if (this.isCancelled) return null;

      this.progress.step_progress = Math.round(
        ((i + 1) / trainingSteps.length) * 100,
      );
      this.emit("final_training_progress", {
        step: trainingSteps[i],
        progress: this.progress.step_progress,
      });

      await this.sleep(1000);
    }

    const finalResult = {
      model_id: this.generateModelId(),
      model_type: finalModel.ensemble_id ? "ensemble" : "single",
      algorithm: finalModel.algorithm || "ensemble",
      final_score: finalModel.ensemble_score || finalModel.validation.score,
      training_time: Date.now() - this.startTime,
      model_size_mb: Math.random() * 50 + 10,
      deployment_ready: true,
      timestamp: Date.now(),
    };

    this.emit("final_training_completed", finalResult);
    return finalResult;
  }

  /**
   * Results Compilation
   */
  compileFinalResults() {
    const totalDuration = this.endTime - this.startTime;

    return {
      pipeline_id: this.pipelineId,
      status: this.status,
      duration_ms: totalDuration,
      duration_human: this.formatDuration(totalDuration),

      // Data insights
      data_insights: {
        original_features:
          this.results.preprocessing?.original_shape?.features || 0,
        engineered_features:
          this.results.feature_engineering?.new_feature_count || 0,
        preprocessing_steps:
          this.results.preprocessing?.preprocessing_steps?.length || 0,
      },

      // Model performance
      model_performance: {
        total_trials: this.results.model_trials?.length || 0,
        best_single_model_score:
          this.results.best_models?.[0]?.validation?.score || 0,
        ensemble_score: this.results.ensemble_model?.ensemble_score || null,
        final_model_score: this.results.final_model?.final_score || 0,
        algorithms_tested: [
          ...new Set(this.results.model_trials?.map((t) => t.algorithm) || []),
        ],
      },

      // Resource utilization
      resource_utilization: {
        total_training_time: totalDuration,
        memory_usage_peak: Math.random() * 4 + 2, // GB
        cpu_utilization_avg: Math.random() * 80 + 20, // %
        gpu_utilization: this.config.performance.gpu_enabled
          ? Math.random() * 90 + 10
          : 0,
      },

      // Final model
      final_model: this.results.final_model,

      // Recommendations
      recommendations: this.generateRecommendations(),

      // Full results for detailed analysis
      detailed_results: this.results,

      // Configuration used
      configuration: this.config,

      timestamp: Date.now(),
    };
  }

  generateRecommendations() {
    const recommendations = [];

    if (this.results.model_trials?.length > 0) {
      const bestScore = Math.max(
        ...this.results.model_trials.map((t) => t.score),
      );

      if (bestScore < 0.7) {
        recommendations.push({
          type: "data_quality",
          message:
            "Consider improving data quality or collecting more diverse samples",
          priority: "high",
        });
      }

      if (this.results.feature_engineering?.new_feature_count < 5) {
        recommendations.push({
          type: "feature_engineering",
          message:
            "Additional feature engineering may improve model performance",
          priority: "medium",
        });
      }

      if (this.config.hyperparameter_optimization.max_evaluations < 50) {
        recommendations.push({
          type: "hyperparameter_tuning",
          message:
            "Increase hyperparameter optimization budget for better results",
          priority: "low",
        });
      }
    }

    return recommendations;
  }

  /**
   * Utility Methods
   */
  getHyperparameterSpace(algorithm) {
    const spaces = {
      isolation_forest: {
        n_estimators: [50, 100, 200, 300],
        contamination: [0.05, 0.1, 0.15, 0.2],
        max_features: [0.5, 0.75, 1.0],
      },
      local_outlier_factor: {
        n_neighbors: [5, 10, 20, 35, 50],
        contamination: [0.05, 0.1, 0.15, 0.2],
        algorithm: ["auto", "ball_tree", "kd_tree"],
      },
      one_class_svm: {
        kernel: ["rbf", "linear", "poly"],
        gamma: ["scale", "auto", 0.001, 0.01, 0.1],
        nu: [0.05, 0.1, 0.15, 0.2],
      },
    };

    return spaces[algorithm] || {};
  }

  sampleHyperparameters(space, algorithm) {
    const params = {};

    for (const [param, values] of Object.entries(space)) {
      if (Array.isArray(values)) {
        params[param] = values[Math.floor(Math.random() * values.length)];
      } else {
        params[param] = values;
      }
    }

    return params;
  }

  getBaseScoreForAlgorithm(algorithm) {
    const baseScores = {
      isolation_forest: 0.75,
      local_outlier_factor: 0.72,
      one_class_svm: 0.7,
      elliptic_envelope: 0.68,
      autoencoder: 0.78,
      deep_svdd: 0.76,
      copod: 0.74,
      ecod: 0.73,
    };

    return baseScores[algorithm] || 0.7;
  }

  generateMockProcessedData(originalData) {
    // Generate mock processed data
    return Array.from({ length: 1000 }, (_, i) => ({
      id: i,
      features: Array.from({ length: 10 }, () => Math.random()),
      label: Math.random() > 0.9 ? 1 : 0, // 10% anomalies
    }));
  }

  generateFeatureNames(data) {
    const baseNames = ["feature", "sensor", "metric", "signal", "value"];
    return Array.from(
      { length: 10 },
      (_, i) => `${baseNames[i % baseNames.length]}_${i + 1}`,
    );
  }

  calculateDataStatistics(data) {
    return {
      total_samples: data.length,
      anomaly_rate: data.filter((d) => d.label === 1).length / data.length,
      feature_count: 10,
      missing_values: Math.floor(Math.random() * 100),
      data_quality_score: Math.random() * 0.3 + 0.7,
    };
  }

  generateMockEngineeredFeatures(data) {
    return Array.from({ length: 15 }, (_, i) => `engineered_feature_${i + 1}`);
  }

  generateMockFeatureImportance(features = []) {
    const importance = {};
    features.forEach((feature, i) => {
      importance[feature] = Math.random() * (1 - i * 0.05);
    });
    return importance;
  }

  generateMockEvaluationMetrics(score) {
    return {
      accuracy: score,
      precision: Math.max(0, Math.min(1, score + (Math.random() - 0.5) * 0.1)),
      recall: Math.max(0, Math.min(1, score + (Math.random() - 0.5) * 0.1)),
      f1_score: Math.max(0, Math.min(1, score + (Math.random() - 0.5) * 0.08)),
    };
  }

  generateMockConfusionMatrix() {
    const tp = Math.floor(Math.random() * 50) + 10;
    const fp = Math.floor(Math.random() * 20) + 5;
    const tn = Math.floor(Math.random() * 200) + 100;
    const fn = Math.floor(Math.random() * 15) + 3;

    return { tp, fp, tn, fn };
  }

  getDatasetShape(dataset) {
    return {
      samples: Array.isArray(dataset) ? dataset.length : 1000,
      features: 10, // Mock feature count
    };
  }

  generatePipelineId() {
    return `pipeline_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  generateTrialId() {
    return `trial_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  generateEnsembleId() {
    return `ensemble_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  generateModelId() {
    return `model_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  formatDuration(ms) {
    const seconds = Math.floor(ms / 1000);
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);

    if (hours > 0) {
      return `${hours}h ${minutes % 60}m ${seconds % 60}s`;
    } else if (minutes > 0) {
      return `${minutes}m ${seconds % 60}s`;
    } else {
      return `${seconds}s`;
    }
  }

  sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  /**
   * Pipeline Control
   */
  cancel() {
    this.isCancelled = true;
    this.status = PIPELINE_STATUS.CANCELLED;
    this.endTime = Date.now();
    this.emit("pipeline_cancelled", { pipelineId: this.pipelineId });
  }

  pause() {
    // Implementation for pausing pipeline
    this.emit("pipeline_paused", { pipelineId: this.pipelineId });
  }

  resume() {
    // Implementation for resuming pipeline
    this.emit("pipeline_resumed", { pipelineId: this.pipelineId });
  }

  /**
   * Event System
   */
  on(event, listener) {
    if (!this.eventListeners.has(event)) {
      this.eventListeners.set(event, new Set());
    }
    this.eventListeners.get(event).add(listener);
    return () => this.off(event, listener);
  }

  off(event, listener) {
    if (this.eventListeners.has(event)) {
      this.eventListeners.get(event).delete(listener);
    }
  }

  emit(event, data) {
    if (this.eventListeners.has(event)) {
      this.eventListeners.get(event).forEach((listener) => {
        try {
          listener({ type: event, data, timestamp: Date.now() });
        } catch (error) {
          console.error("AutoML event listener error:", error);
        }
      });
    }
  }
}

/**
 * AutoML Service Manager
 * High-level interface for managing AutoML pipelines
 */
class AutoMLService {
  constructor() {
    this.activePipelines = new Map();
    this.pipelineHistory = [];
    this.defaultConfig = new AutoMLConfig().defaultConfig;
  }

  async startPipeline(dataset, config = {}, target_column = null) {
    const pipeline = new AutoMLPipeline(config);
    this.activePipelines.set(pipeline.pipelineId, pipeline);

    // Set up pipeline event forwarding
    pipeline.on("pipeline_completed", (event) => {
      this.pipelineHistory.push(event.data);
      this.activePipelines.delete(pipeline.pipelineId);
    });

    pipeline.on("pipeline_failed", (event) => {
      this.pipelineHistory.push(event.data);
      this.activePipelines.delete(pipeline.pipelineId);
    });

    pipeline.on("pipeline_cancelled", (event) => {
      this.activePipelines.delete(pipeline.pipelineId);
    });

    try {
      const results = await pipeline.run(dataset, target_column);
      return { pipelineId: pipeline.pipelineId, results };
    } catch (error) {
      this.activePipelines.delete(pipeline.pipelineId);
      throw error;
    }
  }

  cancelPipeline(pipelineId) {
    const pipeline = this.activePipelines.get(pipelineId);
    if (pipeline) {
      pipeline.cancel();
      return true;
    }
    return false;
  }

  getPipelineStatus(pipelineId) {
    const pipeline = this.activePipelines.get(pipelineId);
    if (pipeline) {
      return {
        pipelineId,
        status: pipeline.status,
        progress: pipeline.progress,
        startTime: pipeline.startTime,
        currentStep: pipeline.currentStep,
        totalSteps: pipeline.totalSteps,
      };
    }
    return null;
  }

  getActivePipelines() {
    return Array.from(this.activePipelines.keys()).map((id) =>
      this.getPipelineStatus(id),
    );
  }

  getPipelineHistory(limit = 10) {
    return this.pipelineHistory.slice(-limit);
  }

  getDefaultConfig() {
    return { ...this.defaultConfig };
  }

  validateConfig(config) {
    const configManager = new AutoMLConfig();
    return configManager.validateConfig(config);
  }
}

// Export classes
if (typeof module !== "undefined" && module.exports) {
  module.exports = {
    AutoMLService,
    AutoMLPipeline,
    AutoMLConfig,
    OPTIMIZATION_ALGORITHMS,
    MODEL_SELECTION_STRATEGIES,
    PIPELINE_STATUS,
  };
} else {
  // Browser environment
  window.AutoMLService = AutoMLService;
  window.AutoMLPipeline = AutoMLPipeline;
  window.AutoMLConfig = AutoMLConfig;
  window.OPTIMIZATION_ALGORITHMS = OPTIMIZATION_ALGORITHMS;
  window.MODEL_SELECTION_STRATEGIES = MODEL_SELECTION_STRATEGIES;
  window.PIPELINE_STATUS = PIPELINE_STATUS;
}
\n\n// pwa-service.js\n/**
 * Progressive Web App Service
 *
 * Comprehensive PWA service with background sync, push notifications,
 * offline capabilities, and app installation management
 */

export class PWAService {
  constructor(options = {}) {
    this.options = {
      enableServiceWorker: true,
      enablePushNotifications: true,
      enableBackgroundSync: true,
      enablePeriodicBackgroundSync: false,
      swPath: "/sw.js",
      vapidPublicKey: options.vapidPublicKey || null,
      notificationIcon: "/static/icons/notification.png",
      notificationBadge: "/static/icons/badge.png",
      enableLogging: false,
      syncTags: {
        backgroundSync: "background-sync",
        periodicSync: "periodic-background-sync",
      },
      ...options,
    };

    this.serviceWorker = null;
    this.pushSubscription = null;
    this.isOnline = navigator.onLine;
    this.isInstallPromptAvailable = false;
    this.installPrompt = null;
    this.listeners = new Map();
    this.syncQueue = [];
    this.notificationPermission = "default";

    this.init();
  }

  async init() {
    this.log("Initializing PWA Service...");

    // Check browser support
    this.checkBrowserSupport();

    // Register service worker
    if (this.options.enableServiceWorker && "serviceWorker" in navigator) {
      await this.registerServiceWorker();
    }

    // Setup event listeners
    this.bindEvents();

    // Initialize push notifications
    if (this.options.enablePushNotifications) {
      await this.initializePushNotifications();
    }

    // Setup background sync
    if (this.options.enableBackgroundSync && this.serviceWorker) {
      this.initializeBackgroundSync();
    }

    // Check for app install prompt
    this.setupInstallPrompt();

    this.log("PWA Service initialized successfully");
  }

  checkBrowserSupport() {
    const support = {
      serviceWorker: "serviceWorker" in navigator,
      pushManager: "PushManager" in window,
      notifications: "Notification" in window,
      backgroundSync:
        "serviceWorker" in navigator &&
        "sync" in window.ServiceWorkerRegistration.prototype,
      periodicBackgroundSync:
        "serviceWorker" in navigator &&
        "periodicSync" in window.ServiceWorkerRegistration.prototype,
      indexedDB: "indexedDB" in window,
      cacheAPI: "caches" in window,
    };

    this.log("Browser support:", support);
    this.emit("support_check", support);

    return support;
  }

  async registerServiceWorker() {
    try {
      this.log("Registering service worker...");

      const registration = await navigator.serviceWorker.register(
        this.options.swPath,
        {
          scope: "/",
        },
      );

      this.serviceWorker = registration;

      // Handle service worker updates
      registration.addEventListener("updatefound", () => {
        this.handleServiceWorkerUpdate(registration);
      });

      // Check for existing service worker
      if (registration.active) {
        this.log("Service worker already active");
        this.emit("sw_ready", registration);
      }

      // Listen for service worker messages
      navigator.serviceWorker.addEventListener("message", (event) => {
        this.handleServiceWorkerMessage(event);
      });

      this.log("Service worker registered successfully");
      this.emit("sw_registered", registration);

      return registration;
    } catch (error) {
      this.log("Service worker registration failed:", error);
      this.emit("sw_registration_failed", error);
      throw error;
    }
  }

  handleServiceWorkerUpdate(registration) {
    const newWorker = registration.installing;

    newWorker.addEventListener("statechange", () => {
      if (
        newWorker.state === "installed" &&
        navigator.serviceWorker.controller
      ) {
        // New service worker is available
        this.log("New service worker available");
        this.emit("sw_update_available", newWorker);

        this.showUpdateNotification();
      }
    });
  }

  showUpdateNotification() {
    const notification = {
      title: "App Update Available",
      message: "A new version of the app is available. Refresh to update.",
      actions: [
        { text: "Update Now", action: "update" },
        { text: "Later", action: "dismiss" },
      ],
    };

    this.emit("update_notification", notification);
  }

  async updateServiceWorker() {
    if (this.serviceWorker && this.serviceWorker.waiting) {
      this.serviceWorker.waiting.postMessage({ type: "SKIP_WAITING" });
      window.location.reload();
    }
  }

  handleServiceWorkerMessage(event) {
    const { type, payload } = event.data;

    switch (type) {
      case "SYNC_COMPLETE":
        this.handleSyncComplete(payload);
        break;

      case "PUSH_RECEIVED":
        this.handlePushReceived(payload);
        break;

      case "NOTIFICATION_CLICK":
        this.handleNotificationClick(payload);
        break;

      case "CACHE_UPDATE":
        this.handleCacheUpdate(payload);
        break;

      case "ERROR":
        this.handleServiceWorkerError(payload);
        break;

      default:
        this.log("Unknown service worker message:", type, payload);
    }
  }

  bindEvents() {
    // Network status
    window.addEventListener("online", () => {
      this.isOnline = true;
      this.log("App is online");
      this.emit("online");
      this.processSyncQueue();
    });

    window.addEventListener("offline", () => {
      this.isOnline = false;
      this.log("App is offline");
      this.emit("offline");
    });

    // App install prompt
    window.addEventListener("beforeinstallprompt", (event) => {
      event.preventDefault();
      this.installPrompt = event;
      this.isInstallPromptAvailable = true;
      this.emit("install_prompt_available");
    });

    // App installed
    window.addEventListener("appinstalled", () => {
      this.log("App installed successfully");
      this.installPrompt = null;
      this.isInstallPromptAvailable = false;
      this.emit("app_installed");
    });

    // Visibility change
    document.addEventListener("visibilitychange", () => {
      if (!document.hidden && this.isOnline) {
        this.syncInBackground();
      }
    });
  }

  // Push Notifications
  async initializePushNotifications() {
    if (!("PushManager" in window) || !("Notification" in window)) {
      this.log("Push notifications not supported");
      return;
    }

    this.notificationPermission = Notification.permission;

    if (this.notificationPermission === "granted") {
      await this.subscribeToPush();
    }

    this.log("Push notifications initialized");
  }

  async requestNotificationPermission() {
    if (!("Notification" in window)) {
      throw new Error("Notifications not supported");
    }

    if (this.notificationPermission === "granted") {
      return "granted";
    }

    const permission = await Notification.requestPermission();
    this.notificationPermission = permission;

    if (permission === "granted") {
      await this.subscribeToPush();
    }

    this.emit("notification_permission", permission);
    return permission;
  }

  async subscribeToPush() {
    if (!this.serviceWorker || !this.options.vapidPublicKey) {
      this.log("Cannot subscribe to push: missing service worker or VAPID key");
      return;
    }

    try {
      const subscription = await this.serviceWorker.pushManager.subscribe({
        userVisibleOnly: true,
        applicationServerKey: this.urlBase64ToUint8Array(
          this.options.vapidPublicKey,
        ),
      });

      this.pushSubscription = subscription;
      this.log("Push subscription created:", subscription);

      // Send subscription to server
      await this.sendSubscriptionToServer(subscription);

      this.emit("push_subscribed", subscription);
      return subscription;
    } catch (error) {
      this.log("Push subscription failed:", error);
      this.emit("push_subscription_failed", error);
      throw error;
    }
  }

  async sendSubscriptionToServer(subscription) {
    try {
      const response = await fetch("/api/push/subscribe", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          subscription,
          userAgent: navigator.userAgent,
          timestamp: Date.now(),
        }),
      });

      if (!response.ok) {
        throw new Error(`Server responded with ${response.status}`);
      }

      this.log("Subscription sent to server successfully");
    } catch (error) {
      this.log("Failed to send subscription to server:", error);
      throw error;
    }
  }

  async unsubscribeFromPush() {
    if (!this.pushSubscription) {
      return;
    }

    try {
      await this.pushSubscription.unsubscribe();

      // Notify server
      await fetch("/api/push/unsubscribe", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          endpoint: this.pushSubscription.endpoint,
        }),
      });

      this.pushSubscription = null;
      this.log("Unsubscribed from push notifications");
      this.emit("push_unsubscribed");
    } catch (error) {
      this.log("Failed to unsubscribe from push:", error);
      throw error;
    }
  }

  showLocalNotification(title, options = {}) {
    if (this.notificationPermission !== "granted") {
      this.log("Cannot show notification: permission not granted");
      return;
    }

    const notificationOptions = {
      body: options.body || "",
      icon: options.icon || this.options.notificationIcon,
      badge: options.badge || this.options.notificationBadge,
      tag: options.tag || "pynomaly-notification",
      data: options.data || {},
      actions: options.actions || [],
      requireInteraction: options.requireInteraction || false,
      silent: options.silent || false,
      ...options,
    };

    if (this.serviceWorker && this.serviceWorker.active) {
      // Show notification through service worker
      this.serviceWorker.active.postMessage({
        type: "SHOW_NOTIFICATION",
        payload: { title, options: notificationOptions },
      });
    } else {
      // Fallback to regular notification
      const notification = new Notification(title, notificationOptions);

      notification.onclick = () => {
        this.handleNotificationClick({
          notification: {
            tag: notificationOptions.tag,
            data: notificationOptions.data,
          },
          action: "click",
        });
      };
    }
  }

  handlePushReceived(payload) {
    this.log("Push notification received:", payload);
    this.emit("push_received", payload);
  }

  handleNotificationClick(payload) {
    this.log("Notification clicked:", payload);
    this.emit("notification_click", payload);

    // Focus app window
    if ("clients" in self) {
      // This runs in service worker context
      self.clients.openWindow("/");
    } else {
      window.focus();
    }
  }

  // Background Sync
  initializeBackgroundSync() {
    if (!("sync" in window.ServiceWorkerRegistration.prototype)) {
      this.log("Background sync not supported");
      return;
    }

    this.log("Background sync initialized");
    this.emit("background_sync_ready");
  }

  async scheduleBackgroundSync(tag = null, data = null) {
    if (!this.serviceWorker) {
      this.log("Cannot schedule background sync: no service worker");
      return;
    }

    const syncTag = tag || this.options.syncTags.backgroundSync;

    try {
      await this.serviceWorker.sync.register(syncTag);
      this.log("Background sync scheduled:", syncTag);

      if (data) {
        // Store data for sync
        await this.storeSyncData(syncTag, data);
      }

      this.emit("sync_scheduled", { tag: syncTag, data });
    } catch (error) {
      this.log("Failed to schedule background sync:", error);
      this.addToSyncQueue({ tag: syncTag, data });
    }
  }

  async schedulePeriodicBackgroundSync(
    tag = null,
    minInterval = 24 * 60 * 60 * 1000,
  ) {
    if (
      !this.options.enablePeriodicBackgroundSync ||
      !("periodicSync" in window.ServiceWorkerRegistration.prototype)
    ) {
      this.log("Periodic background sync not supported or disabled");
      return;
    }

    const syncTag = tag || this.options.syncTags.periodicSync;

    try {
      await this.serviceWorker.periodicSync.register(syncTag, {
        minInterval,
      });

      this.log("Periodic background sync scheduled:", syncTag, minInterval);
      this.emit("periodic_sync_scheduled", { tag: syncTag, minInterval });
    } catch (error) {
      this.log("Failed to schedule periodic background sync:", error);
      throw error;
    }
  }

  async storeSyncData(tag, data) {
    try {
      if ("indexedDB" in window) {
        // Store in IndexedDB
        const db = await this.openSyncDatabase();
        const transaction = db.transaction(["sync_data"], "readwrite");
        const store = transaction.objectStore("sync_data");

        await store.put({
          tag,
          data,
          timestamp: Date.now(),
        });

        this.log("Sync data stored:", tag);
      } else {
        // Fallback to localStorage
        localStorage.setItem(
          `sync_data_${tag}`,
          JSON.stringify({
            data,
            timestamp: Date.now(),
          }),
        );
      }
    } catch (error) {
      this.log("Failed to store sync data:", error);
    }
  }

  async openSyncDatabase() {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open("PWASyncDB", 1);

      request.onerror = () => reject(request.error);
      request.onsuccess = () => resolve(request.result);

      request.onupgradeneeded = (event) => {
        const db = event.target.result;
        if (!db.objectStoreNames.contains("sync_data")) {
          const store = db.createObjectStore("sync_data", { keyPath: "tag" });
          store.createIndex("timestamp", "timestamp");
        }
      };
    });
  }

  addToSyncQueue(syncData) {
    this.syncQueue.push({
      ...syncData,
      timestamp: Date.now(),
    });

    this.log("Added to sync queue:", syncData);
  }

  async processSyncQueue() {
    if (!this.isOnline || this.syncQueue.length === 0) {
      return;
    }

    this.log("Processing sync queue:", this.syncQueue.length, "items");

    const queue = [...this.syncQueue];
    this.syncQueue = [];

    for (const item of queue) {
      try {
        await this.scheduleBackgroundSync(item.tag, item.data);
      } catch (error) {
        this.log("Failed to process sync queue item:", error);
        this.syncQueue.push(item); // Re-add failed items
      }
    }
  }

  handleSyncComplete(payload) {
    this.log("Background sync completed:", payload);
    this.emit("sync_complete", payload);
  }

  async syncInBackground() {
    if (!this.isOnline || !this.serviceWorker) {
      return;
    }

    this.log("Syncing in background...");

    // Schedule sync for various data types
    await this.scheduleBackgroundSync("anomaly_data");
    await this.scheduleBackgroundSync("user_preferences");
    await this.scheduleBackgroundSync("offline_actions");

    this.emit("background_sync_initiated");
  }

  // App Installation
  setupInstallPrompt() {
    this.log("Setting up install prompt...");
  }

  async promptAppInstall() {
    if (!this.isInstallPromptAvailable || !this.installPrompt) {
      throw new Error("Install prompt not available");
    }

    try {
      const result = await this.installPrompt.prompt();
      this.log("Install prompt result:", result);

      const choiceResult = await result.userChoice;
      this.log("User choice:", choiceResult);

      if (choiceResult.outcome === "accepted") {
        this.emit("install_accepted");
      } else {
        this.emit("install_dismissed");
      }

      this.installPrompt = null;
      this.isInstallPromptAvailable = false;

      return choiceResult;
    } catch (error) {
      this.log("Install prompt failed:", error);
      this.emit("install_failed", error);
      throw error;
    }
  }

  isAppInstalled() {
    return (
      window.matchMedia("(display-mode: standalone)").matches ||
      window.navigator.standalone === true
    );
  }

  // Cache Management
  async clearCache(cacheName = null) {
    if (!("caches" in window)) {
      this.log("Cache API not supported");
      return;
    }

    try {
      if (cacheName) {
        await caches.delete(cacheName);
        this.log("Cache cleared:", cacheName);
      } else {
        const cacheNames = await caches.keys();
        await Promise.all(cacheNames.map((name) => caches.delete(name)));
        this.log("All caches cleared");
      }

      this.emit("cache_cleared", { cacheName });
    } catch (error) {
      this.log("Failed to clear cache:", error);
      throw error;
    }
  }

  async getCacheSize() {
    if (
      !("caches" in window) ||
      !("storage" in navigator) ||
      !("estimate" in navigator.storage)
    ) {
      return null;
    }

    try {
      const estimate = await navigator.storage.estimate();
      return {
        usage: estimate.usage,
        quota: estimate.quota,
        usageDetails: estimate.usageDetails,
      };
    } catch (error) {
      this.log("Failed to get cache size:", error);
      return null;
    }
  }

  handleCacheUpdate(payload) {
    this.log("Cache updated:", payload);
    this.emit("cache_updated", payload);
  }

  // Offline Support
  async enableOfflineMode() {
    if (!this.serviceWorker) {
      throw new Error("Service worker required for offline mode");
    }

    this.serviceWorker.active.postMessage({
      type: "ENABLE_OFFLINE_MODE",
    });

    this.log("Offline mode enabled");
    this.emit("offline_mode_enabled");
  }

  async disableOfflineMode() {
    if (!this.serviceWorker) {
      return;
    }

    this.serviceWorker.active.postMessage({
      type: "DISABLE_OFFLINE_MODE",
    });

    this.log("Offline mode disabled");
    this.emit("offline_mode_disabled");
  }

  // Utility methods
  urlBase64ToUint8Array(base64String) {
    const padding = "=".repeat((4 - (base64String.length % 4)) % 4);
    const base64 = (base64String + padding)
      .replace(/-/g, "+")
      .replace(/_/g, "/");

    const rawData = window.atob(base64);
    const outputArray = new Uint8Array(rawData.length);

    for (let i = 0; i < rawData.length; ++i) {
      outputArray[i] = rawData.charCodeAt(i);
    }

    return outputArray;
  }

  handleServiceWorkerError(payload) {
    this.log("Service worker error:", payload);
    this.emit("sw_error", payload);
  }

  // Status and Information
  getStatus() {
    return {
      isOnline: this.isOnline,
      isInstalled: this.isAppInstalled(),
      isInstallPromptAvailable: this.isInstallPromptAvailable,
      serviceWorkerRegistered: !!this.serviceWorker,
      pushSubscribed: !!this.pushSubscription,
      notificationPermission: this.notificationPermission,
      syncQueueLength: this.syncQueue.length,
    };
  }

  async getCapabilities() {
    const support = this.checkBrowserSupport();
    const cacheSize = await this.getCacheSize();

    return {
      ...support,
      cacheSize,
      isOnline: this.isOnline,
      isInstalled: this.isAppInstalled(),
    };
  }

  // Event management
  on(event, listener) {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, new Set());
    }
    this.listeners.get(event).add(listener);

    return () => this.off(event, listener);
  }

  off(event, listener) {
    const eventListeners = this.listeners.get(event);
    if (eventListeners) {
      eventListeners.delete(listener);
      if (eventListeners.size === 0) {
        this.listeners.delete(event);
      }
    }
  }

  emit(event, data) {
    const eventListeners = this.listeners.get(event);
    if (eventListeners) {
      eventListeners.forEach((listener) => {
        try {
          listener(data);
        } catch (error) {
          this.log("Error in event listener:", error);
        }
      });
    }
  }

  log(...args) {
    if (this.options.enableLogging) {
      console.log("[PWAService]", ...args);
    }
  }

  // Cleanup
  destroy() {
    this.listeners.clear();
    this.syncQueue = [];

    if (this.pushSubscription) {
      this.unsubscribeFromPush().catch(() => {});
    }

    // Remove event listeners
    window.removeEventListener("online", this.handleOnline);
    window.removeEventListener("offline", this.handleOffline);
    window.removeEventListener(
      "beforeinstallprompt",
      this.handleBeforeInstallPrompt,
    );
    window.removeEventListener("appinstalled", this.handleAppInstalled);
    document.removeEventListener(
      "visibilitychange",
      this.handleVisibilityChange,
    );
  }
}

/**
 * PWA Installation Helper Component
 */
export class PWAInstallPrompt {
  constructor(pwaService, options = {}) {
    this.pwaService = pwaService;
    this.options = {
      showBanner: true,
      showButton: true,
      bannerDismissible: true,
      buttonText: "Install App",
      bannerText: "Install this app for a better experience",
      position: "bottom",
      ...options,
    };

    this.isVisible = false;
    this.banner = null;
    this.button = null;

    this.init();
  }

  init() {
    this.pwaService.on("install_prompt_available", () => {
      this.show();
    });

    this.pwaService.on("app_installed", () => {
      this.hide();
    });

    if (this.options.showBanner) {
      this.createBanner();
    }

    if (this.options.showButton) {
      this.createButton();
    }
  }

  createBanner() {
    this.banner = document.createElement("div");
    this.banner.className = `pwa-install-banner pwa-banner-${this.options.position}`;
    this.banner.style.display = "none";
    this.banner.innerHTML = `
            <div class="pwa-banner-content">
                <div class="pwa-banner-icon">📱</div>
                <div class="pwa-banner-text">${this.options.bannerText}</div>
                <div class="pwa-banner-actions">
                    <button class="pwa-install-btn">Install</button>
                    ${this.options.bannerDismissible ? '<button class="pwa-dismiss-btn">×</button>' : ""}
                </div>
            </div>
        `;

    // Bind events
    this.banner.querySelector(".pwa-install-btn").onclick = () => {
      this.install();
    };

    if (this.options.bannerDismissible) {
      this.banner.querySelector(".pwa-dismiss-btn").onclick = () => {
        this.hide();
      };
    }

    document.body.appendChild(this.banner);
  }

  createButton() {
    this.button = document.createElement("button");
    this.button.className = "pwa-install-button";
    this.button.textContent = this.options.buttonText;
    this.button.style.display = "none";
    this.button.onclick = () => this.install();

    // You can append this button to any container
    // For example: document.querySelector('.header-actions').appendChild(this.button);
  }

  show() {
    if (this.pwaService.isAppInstalled()) {
      return;
    }

    this.isVisible = true;

    if (this.banner) {
      this.banner.style.display = "block";
      setTimeout(() => {
        this.banner.classList.add("pwa-banner-visible");
      }, 100);
    }

    if (this.button) {
      this.button.style.display = "inline-block";
    }
  }

  hide() {
    this.isVisible = false;

    if (this.banner) {
      this.banner.classList.remove("pwa-banner-visible");
      setTimeout(() => {
        this.banner.style.display = "none";
      }, 300);
    }

    if (this.button) {
      this.button.style.display = "none";
    }
  }

  async install() {
    try {
      await this.pwaService.promptAppInstall();
    } catch (error) {
      console.error("Installation failed:", error);
    }
  }

  getButton() {
    return this.button;
  }

  destroy() {
    if (this.banner && this.banner.parentNode) {
      this.banner.parentNode.removeChild(this.banner);
    }

    if (this.button && this.button.parentNode) {
      this.button.parentNode.removeChild(this.button);
    }
  }
}

export default PWAService;
\n\n// websocket-service.js\n/**
 * WebSocket Service for Real-Time Data
 * Production-ready WebSocket client for live anomaly detection updates
 * with connection management, error handling, and message routing
 */

/**
 * WebSocket Connection States
 */
const CONNECTION_STATES = {
  CONNECTING: "connecting",
  CONNECTED: "connected",
  DISCONNECTED: "disconnected",
  RECONNECTING: "reconnecting",
  ERROR: "error",
};

/**
 * Message Types for WebSocket Communication
 */
const MESSAGE_TYPES = {
  // System messages
  PING: "ping",
  PONG: "pong",
  CONNECT: "connect",
  DISCONNECT: "disconnect",
  ERROR: "error",

  // Data messages
  ANOMALY_DETECTED: "anomaly_detected",
  PERFORMANCE_UPDATE: "performance_update",
  SYSTEM_ALERT: "system_alert",
  DATA_UPDATE: "data_update",
  BATCH_UPDATE: "batch_update",

  // Control messages
  SUBSCRIBE: "subscribe",
  UNSUBSCRIBE: "unsubscribe",
  REQUEST_HISTORY: "request_history",

  // Authentication
  AUTHENTICATE: "authenticate",
  AUTH_SUCCESS: "auth_success",
  AUTH_FAILED: "auth_failed",
};

export class WebSocketService {
  constructor(options = {}) {
    this.options = {
      url: options.url || this.getWebSocketUrl(),
      protocols: options.protocols || ["anomaly-detection-v1"],
      maxReconnectAttempts: options.maxReconnectAttempts || 10,
      reconnectInterval: options.reconnectInterval || 3000,
      maxReconnectDelay: options.maxReconnectDelay || 30000,
      heartbeatInterval: options.heartbeatInterval || 30000,
      messageQueueSize: options.messageQueueSize || 1000,
      enableMessageQueue: options.enableMessageQueue !== false,
      enableCompression: options.enableCompression !== false,
      enableLogging: options.enableLogging || false,
      autoConnect: options.autoConnect !== false,
      authentication: options.authentication || null,
      ...options,
    };

    this.ws = null;
    this.isConnected = false;
    this.reconnectAttempts = 0;
    this.listeners = new Map();
    this.subscriptions = new Set();
    this.heartbeatTimer = null;
    this.reconnectTimer = null;
    this.messageQueue = [];
    this.connectionId = null;

    this.bindMethods();

    if (this.options.autoConnect) {
      this.connect();
    }
  }

  bindMethods() {
    this.handleOpen = this.handleOpen.bind(this);
    this.handleMessage = this.handleMessage.bind(this);
    this.handleError = this.handleError.bind(this);
    this.handleClose = this.handleClose.bind(this);
    this.sendHeartbeat = this.sendHeartbeat.bind(this);
  }

  getWebSocketUrl() {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const host = window.location.host;
    return `${protocol}//${host}/ws/anomaly-detection`;
  }

  connect() {
    if (
      this.isConnected ||
      (this.ws && this.ws.readyState === WebSocket.CONNECTING)
    ) {
      return Promise.resolve();
    }

    return new Promise((resolve, reject) => {
      try {
        this.log("Connecting to WebSocket...", this.options.url);

        this.ws = new WebSocket(this.options.url, this.options.protocols);

        // Configure WebSocket properties
        if (this.options.enableCompression && this.ws.extensions) {
          this.ws.extensions = "permessage-deflate";
        }

        const connectTimeout = setTimeout(() => {
          if (this.ws.readyState === WebSocket.CONNECTING) {
            this.ws.close();
            reject(new Error("Connection timeout"));
          }
        }, 10000);

        this.ws.addEventListener("open", (event) => {
          clearTimeout(connectTimeout);
          this.handleOpen(event);
          resolve();
        });

        this.ws.addEventListener("message", this.handleMessage);
        this.ws.addEventListener("error", (error) => {
          clearTimeout(connectTimeout);
          this.handleError(error);
          reject(error);
        });
        this.ws.addEventListener("close", this.handleClose);
      } catch (error) {
        this.log("Connection error:", error);
        reject(error);
      }
    });
  }

  handleOpen(event) {
    this.log("WebSocket connected");
    this.isConnected = true;
    this.reconnectAttempts = 0;

    // Start heartbeat
    this.startHeartbeat();

    // Send queued messages
    this.processMessageQueue();

    // Re-subscribe to previous subscriptions
    this.resubscribe();

    // Emit connected event
    this.emit("connected", { event, connectionId: this.connectionId });
  }

  handleMessage(event) {
    try {
      const message = JSON.parse(event.data);
      this.log("Received message:", message);

      // Handle system messages
      if (message.type === "system") {
        this.handleSystemMessage(message);
        return;
      }

      // Handle heartbeat responses
      if (message.type === "pong") {
        this.log("Heartbeat acknowledged");
        return;
      }

      // Emit message to listeners
      this.emit("message", message);

      // Emit specific message type
      if (message.type) {
        this.emit(message.type, message.data || message);
      }

      // Handle subscription messages
      if (message.subscription) {
        this.emit(
          `subscription:${message.subscription}`,
          message.data || message,
        );
      }
    } catch (error) {
      this.log("Error parsing message:", error, event.data);
      this.emit("error", { type: "parse_error", error, rawData: event.data });
    }
  }

  handleSystemMessage(message) {
    switch (message.action) {
      case "connection_established":
        this.connectionId = message.connectionId;
        this.log("Connection ID received:", this.connectionId);
        break;

      case "subscription_confirmed":
        this.log("Subscription confirmed:", message.subscription);
        this.emit("subscription_confirmed", message);
        break;

      case "subscription_error":
        this.log("Subscription error:", message.error);
        this.emit("subscription_error", message);
        break;

      case "rate_limit_exceeded":
        this.log("Rate limit exceeded");
        this.emit("rate_limit_exceeded", message);
        break;

      case "server_shutdown":
        this.log("Server shutdown notification");
        this.emit("server_shutdown", message);
        break;

      default:
        this.log("Unknown system message:", message);
    }
  }

  handleError(error) {
    this.log("WebSocket error:", error);
    this.emit("error", { type: "connection_error", error });
  }

  handleClose(event) {
    this.log("WebSocket closed:", event.code, event.reason);
    this.isConnected = false;
    this.stopHeartbeat();

    this.emit("disconnected", {
      code: event.code,
      reason: event.reason,
      wasClean: event.wasClean,
    });

    // Attempt reconnection if not a clean close
    if (
      !event.wasClean &&
      this.reconnectAttempts < this.options.maxReconnectAttempts
    ) {
      this.scheduleReconnect();
    }
  }

  scheduleReconnect() {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
    }

    this.reconnectAttempts++;
    const delay = Math.min(
      this.options.reconnectInterval * Math.pow(2, this.reconnectAttempts - 1),
      30000,
    );

    this.log(
      `Scheduling reconnect attempt ${this.reconnectAttempts} in ${delay}ms`,
    );

    this.reconnectTimer = setTimeout(() => {
      this.log(`Reconnect attempt ${this.reconnectAttempts}`);
      this.connect().catch((error) => {
        this.log("Reconnect failed:", error);
        if (this.reconnectAttempts < this.options.maxReconnectAttempts) {
          this.scheduleReconnect();
        } else {
          this.emit("max_reconnect_attempts_reached");
        }
      });
    }, delay);
  }

  startHeartbeat() {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
    }

    this.heartbeatTimer = setInterval(
      this.sendHeartbeat,
      this.options.heartbeatInterval,
    );
  }

  stopHeartbeat() {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
  }

  sendHeartbeat() {
    if (this.isConnected) {
      this.send({
        type: "ping",
        timestamp: Date.now(),
      });
    }
  }

  send(data) {
    if (!this.isConnected) {
      this.log("Queueing message (not connected):", data);
      this.messageQueue.push(data);
      return false;
    }

    try {
      const message = typeof data === "string" ? data : JSON.stringify(data);
      this.ws.send(message);
      this.log("Sent message:", data);
      return true;
    } catch (error) {
      this.log("Error sending message:", error);
      this.emit("error", { type: "send_error", error, data });
      return false;
    }
  }

  processMessageQueue() {
    while (this.messageQueue.length > 0) {
      const message = this.messageQueue.shift();
      this.send(message);
    }
  }

  // Subscription management
  subscribe(subscription, params = {}) {
    const subscriptionData = {
      type: "subscribe",
      subscription,
      params,
      timestamp: Date.now(),
    };

    this.subscriptions.add(subscription);
    this.send(subscriptionData);

    this.log("Subscribed to:", subscription, params);
    return () => this.unsubscribe(subscription);
  }

  unsubscribe(subscription) {
    this.subscriptions.delete(subscription);
    this.send({
      type: "unsubscribe",
      subscription,
      timestamp: Date.now(),
    });

    this.log("Unsubscribed from:", subscription);
  }

  resubscribe() {
    this.subscriptions.forEach((subscription) => {
      this.send({
        type: "subscribe",
        subscription,
        timestamp: Date.now(),
      });
    });
  }

  // Event management
  on(event, listener) {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, new Set());
    }
    this.listeners.get(event).add(listener);

    return () => this.off(event, listener);
  }

  off(event, listener) {
    const eventListeners = this.listeners.get(event);
    if (eventListeners) {
      eventListeners.delete(listener);
      if (eventListeners.size === 0) {
        this.listeners.delete(event);
      }
    }
  }

  emit(event, data) {
    const eventListeners = this.listeners.get(event);
    if (eventListeners) {
      eventListeners.forEach((listener) => {
        try {
          listener(data);
        } catch (error) {
          this.log("Error in event listener:", error);
        }
      });
    }
  }

  // Utility methods
  isConnected() {
    return this.isConnected && this.ws && this.ws.readyState === WebSocket.OPEN;
  }

  getConnectionState() {
    if (!this.ws) return "disconnected";

    switch (this.ws.readyState) {
      case WebSocket.CONNECTING:
        return "connecting";
      case WebSocket.OPEN:
        return "connected";
      case WebSocket.CLOSING:
        return "closing";
      case WebSocket.CLOSED:
        return "disconnected";
      default:
        return "unknown";
    }
  }

  getConnectionInfo() {
    return {
      connectionId: this.connectionId,
      state: this.getConnectionState(),
      reconnectAttempts: this.reconnectAttempts,
      subscriptions: Array.from(this.subscriptions),
      queuedMessages: this.messageQueue.length,
      url: this.options.url,
    };
  }

  disconnect() {
    this.log("Disconnecting WebSocket");

    this.stopHeartbeat();

    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }

    if (this.ws) {
      this.ws.close(1000, "Client disconnect");
    }

    this.isConnected = false;
    this.subscriptions.clear();
    this.messageQueue = [];
  }

  log(...args) {
    if (this.options.enableLogging) {
      console.log("[WebSocketService]", ...args);
    }
  }

  destroy() {
    this.disconnect();
    this.listeners.clear();

    if (this.resizeObserver) {
      this.resizeObserver.disconnect();
    }
  }
}

/**
 * Anomaly Detection WebSocket Client
 *
 * High-level client for anomaly detection specific operations
 */
export class AnomalyWebSocketClient {
  constructor(options = {}) {
    this.options = {
      enableRealTimeDetection: true,
      enableAlerts: true,
      enableMetrics: true,
      enableModelUpdates: true,
      bufferSize: 1000,
      ...options,
    };

    this.wsService = new WebSocketService(options);
    this.dataBuffer = [];
    this.alertsBuffer = [];
    this.callbacks = new Map();

    this.init();
  }

  init() {
    // Set up event handlers
    this.wsService.on("connected", () => {
      this.setupSubscriptions();
    });

    this.wsService.on("anomaly_detected", (data) => {
      this.handleAnomalyDetected(data);
    });

    this.wsService.on("real_time_data", (data) => {
      this.handleRealTimeData(data);
    });

    this.wsService.on("alert", (data) => {
      this.handleAlert(data);
    });

    this.wsService.on("model_update", (data) => {
      this.handleModelUpdate(data);
    });

    this.wsService.on("system_metrics", (data) => {
      this.handleSystemMetrics(data);
    });
  }

  setupSubscriptions() {
    if (this.options.enableRealTimeDetection) {
      this.wsService.subscribe("anomaly_detection");
      this.wsService.subscribe("real_time_data");
    }

    if (this.options.enableAlerts) {
      this.wsService.subscribe("alerts");
    }

    if (this.options.enableMetrics) {
      this.wsService.subscribe("system_metrics");
    }

    if (this.options.enableModelUpdates) {
      this.wsService.subscribe("model_updates");
    }
  }

  handleAnomalyDetected(data) {
    const anomaly = {
      id: data.id || Date.now(),
      timestamp: data.timestamp || new Date().toISOString(),
      score: data.score,
      severity: data.severity || this.calculateSeverity(data.score),
      features: data.features || [],
      metadata: data.metadata || {},
      datasetId: data.datasetId,
      modelId: data.modelId,
    };

    this.emit("anomaly_detected", anomaly);

    // Auto-alert for high severity anomalies
    if (anomaly.severity === "critical" || anomaly.severity === "high") {
      this.handleAlert({
        type: "anomaly",
        severity: anomaly.severity,
        message: `${anomaly.severity.toUpperCase()} anomaly detected with score ${anomaly.score.toFixed(3)}`,
        anomaly,
      });
    }
  }

  handleRealTimeData(data) {
    this.dataBuffer.push({
      ...data,
      receivedAt: Date.now(),
    });

    // Maintain buffer size
    if (this.dataBuffer.length > this.options.bufferSize) {
      this.dataBuffer.shift();
    }

    this.emit("real_time_data", data);
  }

  handleAlert(alert) {
    const enrichedAlert = {
      id: alert.id || Date.now(),
      timestamp: alert.timestamp || new Date().toISOString(),
      type: alert.type,
      severity: alert.severity || "info",
      message: alert.message,
      data: alert.data || alert.anomaly,
      acknowledged: false,
      receivedAt: Date.now(),
    };

    this.alertsBuffer.unshift(enrichedAlert);

    // Maintain alerts buffer
    if (this.alertsBuffer.length > this.options.bufferSize) {
      this.alertsBuffer.pop();
    }

    this.emit("alert", enrichedAlert);
  }

  handleModelUpdate(update) {
    this.emit("model_update", {
      modelId: update.modelId,
      version: update.version,
      performance: update.performance,
      status: update.status,
      timestamp: update.timestamp || new Date().toISOString(),
    });
  }

  handleSystemMetrics(metrics) {
    this.emit("system_metrics", {
      ...metrics,
      receivedAt: Date.now(),
    });
  }

  calculateSeverity(score) {
    if (score >= 0.9) return "critical";
    if (score >= 0.7) return "high";
    if (score >= 0.5) return "medium";
    return "low";
  }

  // Public API methods
  startRealTimeDetection(datasetId, config = {}) {
    return this.wsService.send({
      type: "start_detection",
      datasetId,
      config: {
        algorithm: config.algorithm || "isolation_forest",
        threshold: config.threshold || 0.5,
        windowSize: config.windowSize || 100,
        ...config,
      },
    });
  }

  stopRealTimeDetection(datasetId) {
    return this.wsService.send({
      type: "stop_detection",
      datasetId,
    });
  }

  sendDataPoint(dataPoint) {
    return this.wsService.send({
      type: "data_point",
      data: dataPoint,
      timestamp: Date.now(),
    });
  }

  sendBatchData(dataPoints) {
    return this.wsService.send({
      type: "batch_data",
      data: dataPoints,
      count: dataPoints.length,
      timestamp: Date.now(),
    });
  }

  acknowledgeAlert(alertId) {
    const alert = this.alertsBuffer.find((a) => a.id === alertId);
    if (alert) {
      alert.acknowledged = true;
      alert.acknowledgedAt = Date.now();
    }

    return this.wsService.send({
      type: "acknowledge_alert",
      alertId,
      timestamp: Date.now(),
    });
  }

  requestModelRetraining(modelId, config = {}) {
    return this.wsService.send({
      type: "request_retraining",
      modelId,
      config,
      timestamp: Date.now(),
    });
  }

  // Data access methods
  getRecentData(count = 100) {
    return this.dataBuffer.slice(-count);
  }

  getRecentAlerts(count = 50) {
    return this.alertsBuffer.slice(0, count);
  }

  getUnacknowledgedAlerts() {
    return this.alertsBuffer.filter((alert) => !alert.acknowledged);
  }

  // Event management
  on(event, callback) {
    return this.wsService.on(event, callback);
  }

  off(event, callback) {
    return this.wsService.off(event, callback);
  }

  emit(event, data) {
    return this.wsService.emit(event, data);
  }

  // Connection management
  connect() {
    return this.wsService.connect();
  }

  disconnect() {
    return this.wsService.disconnect();
  }

  isConnected() {
    return this.wsService.isConnected();
  }

  getConnectionInfo() {
    return {
      ...this.wsService.getConnectionInfo(),
      buffers: {
        data: this.dataBuffer.length,
        alerts: this.alertsBuffer.length,
        unacknowledgedAlerts: this.getUnacknowledgedAlerts().length,
      },
    };
  }

  destroy() {
    this.wsService.destroy();
    this.dataBuffer = [];
    this.alertsBuffer = [];
    this.callbacks.clear();
  }
}

// Export default
export default AnomalyWebSocketClient;
\n\n// accessibility.js\n// Accessibility Utilities
export class AccessibilityManager {
  constructor() {
    this.init();
  }

  init() {
    this.setupFocusManagement();
    this.setupKeyboardNavigation();
    this.setupScreenReaderSupport();
  }

  setupFocusManagement() {
    // Enhanced focus management
    document.addEventListener("keydown", (e) => {
      if (e.key === "Tab") {
        document.body.classList.add("using-keyboard");
      }
    });

    document.addEventListener("mousedown", () => {
      document.body.classList.remove("using-keyboard");
    });
  }

  setupKeyboardNavigation() {
    // Arrow key navigation for grids
    document.addEventListener("keydown", (e) => {
      const grid = e.target.closest('[role="grid"]');
      if (
        grid &&
        ["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"].includes(e.key)
      ) {
        this.handleGridNavigation(e, grid);
      }
    });
  }

  setupScreenReaderSupport() {
    // Dynamic content announcements
    const observer = new MutationObserver((mutations) => {
      mutations.forEach((mutation) => {
        if (mutation.type === "childList") {
          mutation.addedNodes.forEach((node) => {
            if (
              node.nodeType === Node.ELEMENT_NODE &&
              node.hasAttribute("aria-live")
            ) {
              this.announceChange(node);
            }
          });
        }
      });
    });

    observer.observe(document.body, {
      childList: true,
      subtree: true,
    });
  }

  handleGridNavigation(event, grid) {
    // Grid navigation implementation
    event.preventDefault();
    // Implementation would handle arrow key navigation
  }

  announceChange(element) {
    // Announce dynamic content changes
    console.log("Content changed:", element.textContent);
  }
}

// Initialize Accessibility Manager
if (typeof window !== "undefined") {
  window.AccessibilityManager = new AccessibilityManager();
}
\n\n// htmx-extensions.js\n// HTMX Extensions for Enhanced Functionality
if (typeof htmx !== "undefined") {
  // Loading indicator extension
  htmx.defineExtension("loading-states", {
    onEvent: function (name, evt) {
      if (name === "htmx:beforeRequest") {
        evt.target.classList.add("htmx-loading");
      } else if (name === "htmx:afterRequest") {
        evt.target.classList.remove("htmx-loading");
      }
    },
  });

  // Auto-retry extension for failed requests
  htmx.defineExtension("auto-retry", {
    onEvent: function (name, evt) {
      if (name === "htmx:responseError") {
        const retryCount = parseInt(evt.target.dataset.retryCount || "0");
        const maxRetries = parseInt(evt.target.dataset.maxRetries || "3");

        if (retryCount < maxRetries) {
          evt.target.dataset.retryCount = (retryCount + 1).toString();
          setTimeout(
            () => {
              htmx.trigger(evt.target, "click");
            },
            1000 * Math.pow(2, retryCount),
          ); // Exponential backoff
        }
      }
    },
  });
}
\n\n// performance-monitor.js\n// Performance Monitor - Real-time performance tracking
export class PerformanceMonitor {
  constructor() {
    this.metrics = {};
    this.observers = new Map();

    this.init();
  }

  init() {
    this.setupCoreWebVitals();
    this.setupResourceMonitoring();
    this.setupLongTaskMonitoring();
  }

  setupCoreWebVitals() {
    // LCP Observer
    if ("PerformanceObserver" in window) {
      const lcpObserver = new PerformanceObserver((entryList) => {
        for (const entry of entryList.getEntries()) {
          this.recordMetric("lcp", entry.startTime);
        }
      });
      lcpObserver.observe({ type: "largest-contentful-paint", buffered: true });
      this.observers.set("lcp", lcpObserver);

      // FID Observer
      const fidObserver = new PerformanceObserver((entryList) => {
        for (const entry of entryList.getEntries()) {
          const fid = entry.processingStart - entry.startTime;
          this.recordMetric("fid", fid);
        }
      });
      fidObserver.observe({ type: "first-input", buffered: true });
      this.observers.set("fid", fidObserver);

      // CLS Observer
      let clsValue = 0;
      const clsObserver = new PerformanceObserver((entryList) => {
        for (const entry of entryList.getEntries()) {
          if (!entry.hadRecentInput) {
            clsValue += entry.value;
          }
        }
        this.recordMetric("cls", clsValue);
      });
      clsObserver.observe({ type: "layout-shift", buffered: true });
      this.observers.set("cls", clsObserver);
    }
  }

  setupResourceMonitoring() {
    if ("PerformanceObserver" in window) {
      const resourceObserver = new PerformanceObserver((entryList) => {
        for (const entry of entryList.getEntries()) {
          if (entry.duration > 1000) {
            console.warn("Slow resource:", entry.name, `${entry.duration}ms`);
          }
        }
      });
      resourceObserver.observe({ type: "resource", buffered: true });
      this.observers.set("resource", resourceObserver);
    }
  }

  setupLongTaskMonitoring() {
    if ("PerformanceObserver" in window) {
      const longTaskObserver = new PerformanceObserver((entryList) => {
        for (const entry of entryList.getEntries()) {
          console.warn("Long task detected:", `${entry.duration}ms`);
          this.recordMetric("long-task", entry.duration);
        }
      });
      longTaskObserver.observe({ type: "longtask", buffered: true });
      this.observers.set("longtask", longTaskObserver);
    }
  }

  recordMetric(name, value) {
    this.metrics[name] = value;

    // Send to analytics
    if (navigator.sendBeacon) {
      const data = JSON.stringify({
        metric: name,
        value: value,
        timestamp: Date.now(),
        url: window.location.href,
      });
      navigator.sendBeacon("/api/analytics/performance", data);
    }
  }

  getMetrics() {
    return { ...this.metrics };
  }

  destroy() {
    this.observers.forEach((observer) => {
      observer.disconnect();
    });
    this.observers.clear();
  }
}

// Initialize Performance Monitor
if (typeof window !== "undefined") {
  window.PerformanceMonitor = new PerformanceMonitor();
}
\n\n// pwa-manager.js\n/**
 * Advanced PWA Manager - Enhanced Progressive Web App functionality
 * Features: Install prompts, offline data management, sync status, update handling
 */
export class PWAManager {
  constructor() {
    this.deferredPrompt = null;
    this.isInstalled = false;
    this.isOnline = navigator.onLine;
    this.syncQueue = [];
    this.offlineData = {
      datasets: [],
      results: [],
      preferences: {},
    };

    this.init();
  }

  async init() {
    await this.setupServiceWorker();
    this.setupInstallPrompt();
    this.setupUpdateHandling();
    this.setupOfflineHandling();
    this.checkInstallStatus();
    this.setupConnectionMonitoring();
    await this.loadOfflineData();
  }

  /**
   * Service Worker Setup and Communication
   */
  async setupServiceWorker() {
    if ("serviceWorker" in navigator) {
      try {
        const registration =
          await navigator.serviceWorker.register("/static/sw.js");
        console.log("[PWA] Service Worker registered:", registration.scope);

        // Listen for service worker messages
        navigator.serviceWorker.addEventListener("message", (event) => {
          this.handleServiceWorkerMessage(event.data);
        });

        // Handle updates
        registration.addEventListener("updatefound", () => {
          this.handleServiceWorkerUpdate(registration.installing);
        });

        this.registration = registration;
      } catch (error) {
        console.error("[PWA] Service Worker registration failed:", error);
      }
    }
  }

  handleServiceWorkerMessage(data) {
    const { type, payload } = data;

    switch (type) {
      case "DETECTION_COMPLETE":
        this.onDetectionComplete(payload);
        break;
      case "SYNC_QUEUE_STATUS":
        this.updateSyncStatus(payload);
        break;
      case "CACHE_STATUS":
        this.updateCacheStatus(payload);
        break;
      case "OFFLINE_DATA_UPDATED":
        this.loadOfflineData();
        break;
    }
  }

  handleServiceWorkerUpdate(worker) {
    worker.addEventListener("statechange", () => {
      if (worker.state === "installed") {
        this.showUpdateNotification();
      }
    });
  }

  /**
   * Install Prompt Management
   */
  setupInstallPrompt() {
    window.addEventListener("beforeinstallprompt", (e) => {
      e.preventDefault();
      this.deferredPrompt = e;
      this.showInstallButton();
    });

    // Handle successful installation
    window.addEventListener("appinstalled", () => {
      this.isInstalled = true;
      this.hideInstallButton();
      this.showInstallSuccessMessage();
      this.deferredPrompt = null;
    });
  }

  showInstallButton() {
    // Create install button if it doesn't exist
    if (!document.querySelector(".pwa-install-button")) {
      const installButton = this.createInstallButton();
      this.addInstallButtonToPage(installButton);
    }
  }

  createInstallButton() {
    const button = document.createElement("button");
    button.className =
      "pwa-install-button fixed bottom-4 right-4 bg-primary text-white px-4 py-2 rounded-lg shadow-lg hover:bg-blue-700 transition-colors z-50 flex items-center gap-2";
    button.innerHTML = `
      <svg class="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
        <path fill-rule="evenodd" d="M3 17a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm3.293-7.707a1 1 0 011.414 0L9 10.586V3a1 1 0 112 0v7.586l1.293-1.293a1 1 0 111.414 1.414l-3 3a1 1 0 01-1.414 0l-3-3a1 1 0 010-1.414z" clip-rule="evenodd"></path>
      </svg>
      Install App
    `;
    button.addEventListener("click", () => this.installPWA());
    return button;
  }

  addInstallButtonToPage(button) {
    document.body.appendChild(button);

    // Animate in
    setTimeout(() => {
      button.style.transform = "translateY(0)";
      button.style.opacity = "1";
    }, 100);
  }

  hideInstallButton() {
    const button = document.querySelector(".pwa-install-button");
    if (button) {
      button.style.transform = "translateY(100px)";
      button.style.opacity = "0";
      setTimeout(() => button.remove(), 300);
    }
  }

  async installPWA() {
    if (this.deferredPrompt) {
      this.deferredPrompt.prompt();
      const result = await this.deferredPrompt.userChoice;

      if (result.outcome === "accepted") {
        console.log("[PWA] User accepted the install prompt");
        this.trackEvent("pwa_install_accepted");
      } else {
        console.log("[PWA] User dismissed the install prompt");
        this.trackEvent("pwa_install_dismissed");
      }

      this.deferredPrompt = null;
    }
  }

  showInstallSuccessMessage() {
    this.showNotification(
      "📱 App installed successfully! You can now use Pynomaly offline.",
      "success",
    );
  }

  /**
   * Update Handling
   */
  setupUpdateHandling() {
    if ("serviceWorker" in navigator) {
      navigator.serviceWorker.addEventListener("controllerchange", () => {
        if (this.refreshing) return;
        this.refreshing = true;
        window.location.reload();
      });
    }
  }

  showUpdateNotification() {
    const notification = this.createNotification(
      "🔄 New version available! Click to update.",
      "info",
      [
        { text: "Update Now", action: () => this.updateApp() },
        { text: "Later", action: () => this.dismissUpdate() },
      ],
    );
    document.body.appendChild(notification);
  }

  async updateApp() {
    if (this.registration && this.registration.waiting) {
      this.registration.waiting.postMessage({ type: "SKIP_WAITING" });
    }
  }

  dismissUpdate() {
    const notification = document.querySelector(".pwa-update-notification");
    if (notification) notification.remove();
  }

  /**
   * Offline Data Management
   */
  setupOfflineHandling() {
    window.addEventListener("online", () => {
      this.isOnline = true;
      this.onConnectionRestore();
    });

    window.addEventListener("offline", () => {
      this.isOnline = false;
      this.onConnectionLost();
    });
  }

  async loadOfflineData() {
    try {
      if (this.registration && this.registration.active) {
        // Request data from service worker
        this.registration.active.postMessage({ type: "GET_OFFLINE_DATA" });
      }
    } catch (error) {
      console.error("[PWA] Failed to load offline data:", error);
    }
  }

  async saveDataOffline(type, data) {
    try {
      if (this.registration && this.registration.active) {
        this.registration.active.postMessage({
          type: "SAVE_OFFLINE_DATA",
          payload: { type, data },
        });
      }
    } catch (error) {
      console.error("[PWA] Failed to save offline data:", error);
    }
  }

  /**
   * Connection Monitoring
   */
  setupConnectionMonitoring() {
    // Update UI connection status
    this.updateConnectionUI();

    // Periodic connectivity check
    setInterval(() => {
      this.checkConnectivity();
    }, 30000); // Check every 30 seconds
  }

  updateConnectionUI() {
    const indicators = document.querySelectorAll(".connection-indicator");
    indicators.forEach((indicator) => {
      indicator.className = `connection-indicator ${this.isOnline ? "online" : "offline"}`;
    });

    const statusTexts = document.querySelectorAll(".connection-status");
    statusTexts.forEach((text) => {
      text.textContent = this.isOnline ? "Online" : "Offline";
      text.className = `connection-status ${this.isOnline ? "text-green-600" : "text-orange-600"}`;
    });
  }

  async checkConnectivity() {
    try {
      const response = await fetch("/api/health/ping", {
        method: "HEAD",
        mode: "no-cors",
        cache: "no-cache",
      });

      if (!this.isOnline && navigator.onLine) {
        this.isOnline = true;
        this.onConnectionRestore();
      }
    } catch (error) {
      if (this.isOnline && !navigator.onLine) {
        this.isOnline = false;
        this.onConnectionLost();
      }
    }
  }

  onConnectionRestore() {
    console.log("[PWA] Connection restored");
    this.updateConnectionUI();
    this.syncPendingData();
    this.showNotification("🌐 Connection restored! Syncing data...", "success");
  }

  onConnectionLost() {
    console.log("[PWA] Connection lost");
    this.updateConnectionUI();
    this.showNotification(
      "📡 You're now offline. Changes will sync when connection returns.",
      "warning",
    );
  }

  /**
   * Data Synchronization
   */
  async syncPendingData() {
    if (this.registration && this.registration.active) {
      // Trigger background sync for all queued requests
      this.registration.active.postMessage({ type: "SYNC_ALL_QUEUES" });

      // Request sync status update
      setTimeout(() => {
        this.registration.active.postMessage({ type: "GET_SYNC_STATUS" });
      }, 1000);
    }
  }

  updateSyncStatus(status) {
    const pendingCount = status.pending || 0;

    // Update sync indicators in UI
    const syncIndicators = document.querySelectorAll(".sync-indicator");
    syncIndicators.forEach((indicator) => {
      indicator.textContent =
        pendingCount > 0 ? `${pendingCount} pending` : "Up to date";
      indicator.className = `sync-indicator ${pendingCount > 0 ? "text-orange-600" : "text-green-600"}`;
    });
  }

  /**
   * Cache Management
   */
  async getCacheInfo() {
    if (this.registration && this.registration.active) {
      return new Promise((resolve) => {
        const channel = new MessageChannel();
        channel.port1.onmessage = (event) => resolve(event.data);

        this.registration.active.postMessage({ type: "GET_CACHE_STATUS" }, [
          channel.port2,
        ]);
      });
    }
    return null;
  }

  async clearCache(cacheName = null) {
    if (this.registration && this.registration.active) {
      this.registration.active.postMessage({
        type: "CLEAR_CACHE",
        payload: { cacheName },
      });
    }
  }

  /**
   * Installation Status
   */
  checkInstallStatus() {
    // Check if running in standalone mode
    if (window.matchMedia("(display-mode: standalone)").matches) {
      this.isInstalled = true;
    }

    // iOS Safari check
    if (window.navigator.standalone === true) {
      this.isInstalled = true;
    }

    // Android TWA check
    if (document.referrer.includes("android-app://")) {
      this.isInstalled = true;
    }

    if (this.isInstalled) {
      this.onInstallDetected();
    }
  }

  onInstallDetected() {
    document.body.classList.add("pwa-installed");
    this.trackEvent("pwa_running_installed");
  }

  /**
   * Event Handlers
   */
  onDetectionComplete(payload) {
    this.showNotification(
      `✅ Detection completed: ${payload.result.n_anomalies} anomalies found`,
      "success",
    );
  }

  /**
   * UI Utilities
   */
  showNotification(message, type = "info", actions = []) {
    const notification = this.createNotification(message, type, actions);
    document.body.appendChild(notification);

    // Auto-dismiss after 5 seconds if no actions
    if (actions.length === 0) {
      setTimeout(() => {
        if (notification.parentNode) {
          notification.remove();
        }
      }, 5000);
    }
  }

  createNotification(message, type, actions = []) {
    const notification = document.createElement("div");
    notification.className = `pwa-notification fixed top-4 right-4 max-w-sm bg-white border rounded-lg shadow-lg z-50 transform transition-all duration-300`;

    const colors = {
      success: "border-green-200 bg-green-50",
      warning: "border-orange-200 bg-orange-50",
      error: "border-red-200 bg-red-50",
      info: "border-blue-200 bg-blue-50",
    };

    notification.className += ` ${colors[type] || colors.info}`;

    const actionsHtml =
      actions.length > 0
        ? `
      <div class="mt-3 flex gap-2">
        ${actions
          .map(
            (action) => `
          <button class="px-3 py-1 text-sm bg-white border rounded hover:bg-gray-50 transition-colors"
                  onclick="this.closest('.pwa-notification').remove(); (${action.action.toString()})()">
            ${action.text}
          </button>
        `,
          )
          .join("")}
      </div>
    `
        : "";

    notification.innerHTML = `
      <div class="p-4">
        <div class="flex justify-between items-start">
          <div class="flex-1">
            <p class="text-sm font-medium text-gray-900">${message}</p>
            ${actionsHtml}
          </div>
          <button onclick="this.closest('.pwa-notification').remove()"
                  class="ml-3 text-gray-400 hover:text-gray-600">
            <svg class="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
              <path fill-rule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clip-rule="evenodd"></path>
            </svg>
          </button>
        </div>
      </div>
    `;

    // Animate in
    setTimeout(() => {
      notification.style.transform = "translateX(0)";
    }, 100);

    return notification;
  }

  /**
   * Analytics and Tracking
   */
  trackEvent(eventName, data = {}) {
    // Send analytics event
    if (typeof gtag !== "undefined") {
      gtag("event", eventName, data);
    }

    console.log(`[PWA] Event: ${eventName}`, data);
  }

  /**
   * Public API Methods
   */
  isAppInstalled() {
    return this.isInstalled;
  }

  isAppOnline() {
    return this.isOnline;
  }

  async getAppStatus() {
    const cacheInfo = await this.getCacheInfo();
    return {
      installed: this.isInstalled,
      online: this.isOnline,
      serviceWorkerActive: !!this.registration?.active,
      cacheInfo: cacheInfo,
    };
  }
}

// Initialize PWA Manager
if (typeof window !== "undefined") {
  window.PWAManager = new PWAManager();

  // Expose useful methods globally
  window.PWA = {
    install: () => window.PWAManager.installPWA(),
    getStatus: () => window.PWAManager.getAppStatus(),
    clearCache: (name) => window.PWAManager.clearCache(name),
    sync: () => window.PWAManager.syncPendingData(),
  };
}
\n\n// state-management.js\n/**
 * State Management System for Pynomaly
 * Zustand-based state management for complex component interactions
 * and data flow with persistence, middleware, and real-time updates
 */

/**
 * Simple Zustand-like state management implementation
 * Provides reactive state management with subscriptions and middleware
 */
class StateStore {
  constructor(createState) {
    this.state = {};
    this.listeners = new Set();
    this.middlewares = [];
    this.slices = new Map();
    this.persistConfig = null;

    // Initialize state
    const setState = this.createSetState();
    const getState = () => this.state;
    this.state = createState(setState, getState, this);

    // Load persisted state if configured
    this.loadPersistedState();
  }

  createSetState() {
    return (partial, replace = false) => {
      const nextState =
        typeof partial === "function" ? partial(this.state) : partial;

      const prevState = this.state;
      this.state = replace ? nextState : { ...this.state, ...nextState };

      // Apply middleware
      this.middlewares.forEach((middleware) => {
        middleware(this.state, prevState, this);
      });

      // Persist state if configured
      this.persistState();

      // Notify subscribers
      this.listeners.forEach((listener) => listener(this.state, prevState));
    };
  }

  subscribe(listener) {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  getState() {
    return this.state;
  }

  addMiddleware(middleware) {
    this.middlewares.push(middleware);
  }

  addSlice(name, slice) {
    this.slices.set(name, slice);
  }

  getSlice(name) {
    return this.slices.get(name);
  }

  configurePersistence(config) {
    this.persistConfig = config;
  }

  persistState() {
    if (!this.persistConfig) return;

    try {
      const {
        key,
        storage = localStorage,
        whitelist,
        blacklist,
      } = this.persistConfig;
      let stateToSave = this.state;

      if (whitelist) {
        stateToSave = Object.keys(this.state)
          .filter((key) => whitelist.includes(key))
          .reduce((obj, key) => {
            obj[key] = this.state[key];
            return obj;
          }, {});
      }

      if (blacklist) {
        stateToSave = Object.keys(this.state)
          .filter((key) => !blacklist.includes(key))
          .reduce((obj, key) => {
            obj[key] = this.state[key];
            return obj;
          }, {});
      }

      storage.setItem(key, JSON.stringify(stateToSave));
    } catch (error) {
      console.warn("Failed to persist state:", error);
    }
  }

  loadPersistedState() {
    if (!this.persistConfig) return;

    try {
      const { key, storage = localStorage } = this.persistConfig;
      const persistedState = storage.getItem(key);

      if (persistedState) {
        const parsed = JSON.parse(persistedState);
        this.state = { ...this.state, ...parsed };
      }
    } catch (error) {
      console.warn("Failed to load persisted state:", error);
    }
  }
}

/**
 * Main Application Store
 * Central state management for the entire Pynomaly application
 */
const createAppStore = (set, get) => ({
  // UI State
  ui: {
    theme: "light",
    sidebarOpen: true,
    loading: false,
    notifications: [],
    modal: null,
    activeView: "dashboard",
    layout: "default",
  },

  // User State
  user: {
    isAuthenticated: false,
    profile: null,
    preferences: {
      chartAnimations: true,
      realTimeUpdates: true,
      accessibilityMode: false,
      dataRefreshInterval: 30000,
    },
    permissions: [],
  },

  // Data State
  data: {
    datasets: [],
    models: [],
    detectionResults: [],
    performanceMetrics: [],
    realTimeData: {
      isConnected: false,
      lastUpdate: null,
      buffer: [],
    },
  },

  // Chart State
  charts: {
    instances: new Map(),
    configurations: {},
    themes: {
      light: {
        /* theme config */
      },
      dark: {
        /* theme config */
      },
    },
    activeFilters: {},
    selectedData: null,
  },

  // Dashboard State
  dashboard: {
    layout: [],
    widgets: {},
    filters: {},
    refreshInterval: 30000,
    autoRefresh: false,
  },

  // Actions
  setTheme: (theme) =>
    set((state) => ({
      ui: { ...state.ui, theme },
    })),

  toggleSidebar: () =>
    set((state) => ({
      ui: { ...state.ui, sidebarOpen: !state.ui.sidebarOpen },
    })),

  setLoading: (loading) =>
    set((state) => ({
      ui: { ...state.ui, loading },
    })),

  addNotification: (notification) =>
    set((state) => ({
      ui: {
        ...state.ui,
        notifications: [
          ...state.ui.notifications,
          {
            id: Date.now(),
            timestamp: new Date().toISOString(),
            ...notification,
          },
        ],
      },
    })),

  removeNotification: (id) =>
    set((state) => ({
      ui: {
        ...state.ui,
        notifications: state.ui.notifications.filter((n) => n.id !== id),
      },
    })),

  showModal: (modal) =>
    set((state) => ({
      ui: { ...state.ui, modal },
    })),

  hideModal: () =>
    set((state) => ({
      ui: { ...state.ui, modal: null },
    })),

  setActiveView: (view) =>
    set((state) => ({
      ui: { ...state.ui, activeView: view },
    })),

  // User Actions
  setUser: (user) =>
    set((state) => ({
      user: { ...state.user, ...user },
    })),

  updateUserPreferences: (preferences) =>
    set((state) => ({
      user: {
        ...state.user,
        preferences: { ...state.user.preferences, ...preferences },
      },
    })),

  // Data Actions
  setDatasets: (datasets) =>
    set((state) => ({
      data: { ...state.data, datasets },
    })),

  addDataset: (dataset) =>
    set((state) => ({
      data: {
        ...state.data,
        datasets: [...state.data.datasets, dataset],
      },
    })),

  updateDataset: (id, updates) =>
    set((state) => ({
      data: {
        ...state.data,
        datasets: state.data.datasets.map((d) =>
          d.id === id ? { ...d, ...updates } : d,
        ),
      },
    })),

  removeDataset: (id) =>
    set((state) => ({
      data: {
        ...state.data,
        datasets: state.data.datasets.filter((d) => d.id !== id),
      },
    })),

  setModels: (models) =>
    set((state) => ({
      data: { ...state.data, models },
    })),

  addDetectionResult: (result) =>
    set((state) => ({
      data: {
        ...state.data,
        detectionResults: [...state.data.detectionResults, result],
      },
    })),

  updatePerformanceMetrics: (metrics) =>
    set((state) => ({
      data: {
        ...state.data,
        performanceMetrics: [...state.data.performanceMetrics, metrics],
      },
    })),

  // Real-time Data Actions
  setRealTimeConnection: (isConnected) =>
    set((state) => ({
      data: {
        ...state.data,
        realTimeData: {
          ...state.data.realTimeData,
          isConnected,
          lastUpdate: isConnected
            ? new Date().toISOString()
            : state.data.realTimeData.lastUpdate,
        },
      },
    })),

  addRealTimeData: (data) =>
    set((state) => {
      const buffer = [...state.data.realTimeData.buffer, data];
      // Keep only last 1000 items
      const trimmedBuffer = buffer.slice(-1000);

      return {
        data: {
          ...state.data,
          realTimeData: {
            ...state.data.realTimeData,
            buffer: trimmedBuffer,
            lastUpdate: new Date().toISOString(),
          },
        },
      };
    }),

  clearRealTimeBuffer: () =>
    set((state) => ({
      data: {
        ...state.data,
        realTimeData: {
          ...state.data.realTimeData,
          buffer: [],
        },
      },
    })),

  // Chart Actions
  registerChart: (id, chart) =>
    set((state) => {
      const newInstances = new Map(state.charts.instances);
      newInstances.set(id, chart);
      return {
        charts: { ...state.charts, instances: newInstances },
      };
    }),

  unregisterChart: (id) =>
    set((state) => {
      const newInstances = new Map(state.charts.instances);
      newInstances.delete(id);
      return {
        charts: { ...state.charts, instances: newInstances },
      };
    }),

  updateChartConfiguration: (id, config) =>
    set((state) => ({
      charts: {
        ...state.charts,
        configurations: {
          ...state.charts.configurations,
          [id]: { ...state.charts.configurations[id], ...config },
        },
      },
    })),

  setActiveFilters: (filters) =>
    set((state) => ({
      charts: { ...state.charts, activeFilters: filters },
    })),

  setSelectedData: (data) =>
    set((state) => ({
      charts: { ...state.charts, selectedData: data },
    })),

  // Dashboard Actions
  updateDashboardLayout: (layout) =>
    set((state) => ({
      dashboard: { ...state.dashboard, layout },
    })),

  addWidget: (widget) =>
    set((state) => ({
      dashboard: {
        ...state.dashboard,
        widgets: { ...state.dashboard.widgets, [widget.id]: widget },
      },
    })),

  updateWidget: (id, updates) =>
    set((state) => ({
      dashboard: {
        ...state.dashboard,
        widgets: {
          ...state.dashboard.widgets,
          [id]: { ...state.dashboard.widgets[id], ...updates },
        },
      },
    })),

  removeWidget: (id) =>
    set((state) => {
      const newWidgets = { ...state.dashboard.widgets };
      delete newWidgets[id];
      return {
        dashboard: { ...state.dashboard, widgets: newWidgets },
      };
    }),

  setDashboardFilters: (filters) =>
    set((state) => ({
      dashboard: { ...state.dashboard, filters },
    })),

  setAutoRefresh: (autoRefresh) =>
    set((state) => ({
      dashboard: { ...state.dashboard, autoRefresh },
    })),

  setRefreshInterval: (interval) =>
    set((state) => ({
      dashboard: { ...state.dashboard, refreshInterval: interval },
    })),
});

// Create the main application store
const appStore = new StateStore(createAppStore);

// Configure persistence
appStore.configurePersistence({
  key: "pynomaly-app-state",
  storage: localStorage,
  whitelist: ["user", "dashboard", "ui"],
  blacklist: ["data.realTimeData"],
});

/**
 * Middleware for logging state changes
 */
const loggingMiddleware = (currentState, previousState, store) => {
  if (process.env.NODE_ENV === "development") {
    console.group("State Update");
    console.log("Previous State:", previousState);
    console.log("Current State:", currentState);
    console.groupEnd();
  }
};

/**
 * Middleware for analytics tracking
 */
const analyticsMiddleware = (currentState, previousState, store) => {
  // Track significant state changes
  if (currentState.ui.activeView !== previousState.ui.activeView) {
    // Track view changes
    if (window.gtag) {
      window.gtag("event", "page_view", {
        page_title: currentState.ui.activeView,
        page_location: window.location.href,
      });
    }
  }

  if (
    currentState.data.detectionResults.length >
    previousState.data.detectionResults.length
  ) {
    // Track new detections
    if (window.gtag) {
      window.gtag("event", "anomaly_detected", {
        custom_parameter: currentState.data.detectionResults.length,
      });
    }
  }
};

/**
 * Middleware for accessibility announcements
 */
const accessibilityMiddleware = (currentState, previousState, store) => {
  const announcer =
    document.getElementById("state-announcer") ||
    document.querySelector('[aria-live="polite"]');

  if (!announcer) return;

  // Announce loading state changes
  if (currentState.ui.loading !== previousState.ui.loading) {
    if (currentState.ui.loading) {
      announcer.textContent = "Loading data, please wait";
    } else {
      announcer.textContent = "Data loaded successfully";
    }
  }

  // Announce new notifications
  if (
    currentState.ui.notifications.length > previousState.ui.notifications.length
  ) {
    const newNotifications = currentState.ui.notifications.slice(
      previousState.ui.notifications.length,
    );
    newNotifications.forEach((notification) => {
      announcer.textContent = `${notification.type}: ${notification.message}`;
    });
  }

  // Announce real-time connection changes
  if (
    currentState.data.realTimeData.isConnected !==
    previousState.data.realTimeData.isConnected
  ) {
    announcer.textContent = currentState.data.realTimeData.isConnected
      ? "Real-time data connection established"
      : "Real-time data connection lost";
  }
};

// Add middleware to store
appStore.addMiddleware(loggingMiddleware);
appStore.addMiddleware(analyticsMiddleware);
appStore.addMiddleware(accessibilityMiddleware);

/**
 * Chart State Slice
 * Specialized state management for chart interactions
 */
const createChartSlice = (set, get) => ({
  selectedPoints: [],
  hoveredPoint: null,
  brushSelection: null,
  zoomLevel: 1,
  filters: {
    timeRange: null,
    confidenceThreshold: 0,
    anomalyTypes: [],
  },
  interactions: {
    brushEnabled: true,
    zoomEnabled: true,
    tooltipsEnabled: true,
  },

  // Actions
  selectPoints: (points) =>
    set((state) => ({
      selectedPoints: points,
    })),

  setHoveredPoint: (point) =>
    set((state) => ({
      hoveredPoint: point,
    })),

  setBrushSelection: (selection) =>
    set((state) => ({
      brushSelection: selection,
    })),

  setZoomLevel: (level) =>
    set((state) => ({
      zoomLevel: level,
    })),

  updateFilters: (filters) =>
    set((state) => ({
      filters: { ...state.filters, ...filters },
    })),

  resetFilters: () =>
    set((state) => ({
      filters: {
        timeRange: null,
        confidenceThreshold: 0,
        anomalyTypes: [],
      },
    })),

  setInteractions: (interactions) =>
    set((state) => ({
      interactions: { ...state.interactions, ...interactions },
    })),
});

const chartStore = new StateStore(createChartSlice);
appStore.addSlice("charts", chartStore);

/**
 * Form State Slice
 * Specialized state management for complex forms
 */
const createFormSlice = (set, get) => ({
  forms: {},
  validations: {},
  submissions: {},

  // Actions
  createForm: (formId, initialData = {}) =>
    set((state) => ({
      forms: {
        ...state.forms,
        [formId]: {
          id: formId,
          data: initialData,
          touched: {},
          errors: {},
          isValid: true,
          isSubmitting: false,
          isDirty: false,
        },
      },
    })),

  updateFormField: (formId, field, value) =>
    set((state) => {
      const form = state.forms[formId];
      if (!form) return state;

      const newData = { ...form.data, [field]: value };
      const touched = { ...form.touched, [field]: true };

      // Run validation
      const validator = state.validations[formId];
      const errors = validator ? validator(newData) : {};
      const isValid = Object.keys(errors).length === 0;

      return {
        forms: {
          ...state.forms,
          [formId]: {
            ...form,
            data: newData,
            touched,
            errors,
            isValid,
            isDirty: true,
          },
        },
      };
    }),

  setFormValidation: (formId, validator) =>
    set((state) => ({
      validations: {
        ...state.validations,
        [formId]: validator,
      },
    })),

  setFormSubmitting: (formId, isSubmitting) =>
    set((state) => ({
      forms: {
        ...state.forms,
        [formId]: {
          ...state.forms[formId],
          isSubmitting,
        },
      },
    })),

  resetForm: (formId) =>
    set((state) => {
      const form = state.forms[formId];
      if (!form) return state;

      return {
        forms: {
          ...state.forms,
          [formId]: {
            ...form,
            touched: {},
            errors: {},
            isValid: true,
            isSubmitting: false,
            isDirty: false,
          },
        },
      };
    }),

  removeForm: (formId) =>
    set((state) => {
      const newForms = { ...state.forms };
      const newValidations = { ...state.validations };
      const newSubmissions = { ...state.submissions };

      delete newForms[formId];
      delete newValidations[formId];
      delete newSubmissions[formId];

      return {
        forms: newForms,
        validations: newValidations,
        submissions: newSubmissions,
      };
    }),
});

const formStore = new StateStore(createFormSlice);
appStore.addSlice("forms", formStore);

/**
 * Real-time Data Manager
 * Handles WebSocket connections and real-time updates
 */
class RealTimeManager {
  constructor(store) {
    this.store = store;
    this.websocket = null;
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 5;
    this.reconnectDelay = 1000;
    this.heartbeatInterval = null;
  }

  connect(url = "ws://localhost:8000/ws") {
    try {
      this.websocket = new WebSocket(url);
      this.setupEventListeners();
    } catch (error) {
      console.error("Failed to connect to WebSocket:", error);
      this.store.getState().setRealTimeConnection(false);
    }
  }

  setupEventListeners() {
    this.websocket.onopen = () => {
      console.log("WebSocket connected");
      this.store.getState().setRealTimeConnection(true);
      this.reconnectAttempts = 0;
      this.startHeartbeat();
    };

    this.websocket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        this.handleMessage(data);
      } catch (error) {
        console.error("Failed to parse WebSocket message:", error);
      }
    };

    this.websocket.onclose = () => {
      console.log("WebSocket disconnected");
      this.store.getState().setRealTimeConnection(false);
      this.stopHeartbeat();
      this.attemptReconnect();
    };

    this.websocket.onerror = (error) => {
      console.error("WebSocket error:", error);
      this.store.getState().addNotification({
        type: "error",
        message: "Real-time connection error",
        duration: 5000,
      });
    };
  }

  handleMessage(data) {
    const { type, payload } = data;

    switch (type) {
      case "anomaly_detected":
        this.store.getState().addDetectionResult(payload);
        this.store.getState().addNotification({
          type: "warning",
          message: `Anomaly detected: ${payload.type}`,
          duration: 10000,
        });
        break;

      case "performance_update":
        this.store.getState().updatePerformanceMetrics(payload);
        this.store.getState().addRealTimeData(payload);
        break;

      case "system_alert":
        this.store.getState().addNotification({
          type: payload.severity,
          message: payload.message,
          duration: payload.severity === "error" ? 0 : 5000,
        });
        break;

      case "data_update":
        this.store.getState().addRealTimeData(payload);
        break;

      default:
        console.warn("Unknown message type:", type);
    }
  }

  startHeartbeat() {
    this.heartbeatInterval = setInterval(() => {
      if (this.websocket.readyState === WebSocket.OPEN) {
        this.websocket.send(JSON.stringify({ type: "ping" }));
      }
    }, 30000);
  }

  stopHeartbeat() {
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
      this.heartbeatInterval = null;
    }
  }

  attemptReconnect() {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      const delay =
        this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);

      setTimeout(() => {
        console.log(
          `Attempting to reconnect (${this.reconnectAttempts}/${this.maxReconnectAttempts})`,
        );
        this.connect();
      }, delay);
    } else {
      console.error("Max reconnection attempts reached");
      this.store.getState().addNotification({
        type: "error",
        message: "Real-time connection failed. Please refresh the page.",
        duration: 0,
      });
    }
  }

  disconnect() {
    if (this.websocket) {
      this.websocket.close();
      this.websocket = null;
    }
    this.stopHeartbeat();
  }

  send(message) {
    if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
      this.websocket.send(JSON.stringify(message));
    } else {
      console.warn("WebSocket not connected");
    }
  }
}

/**
 * React-like hooks for state management
 */
const useStore = (selector) => {
  const state = appStore.getState();
  return selector ? selector(state) : state;
};

const useStoreSubscription = (selector, callback) => {
  const unsubscribe = appStore.subscribe((state, prevState) => {
    const current = selector(state);
    const previous = selector(prevState);

    if (current !== previous) {
      callback(current, previous);
    }
  });

  return unsubscribe;
};

/**
 * Computed state selectors
 */
const selectors = {
  // UI Selectors
  getTheme: (state) => state.ui.theme,
  getLoading: (state) => state.ui.loading,
  getNotifications: (state) => state.ui.notifications,
  getActiveView: (state) => state.ui.activeView,

  // User Selectors
  getUser: (state) => state.user,
  getUserPreferences: (state) => state.user.preferences,
  isAuthenticated: (state) => state.user.isAuthenticated,

  // Data Selectors
  getDatasets: (state) => state.data.datasets,
  getModels: (state) => state.data.models,
  getDetectionResults: (state) => state.data.detectionResults,
  getPerformanceMetrics: (state) => state.data.performanceMetrics,
  getRealTimeData: (state) => state.data.realTimeData,

  // Computed Selectors
  getRecentAnomalies: (state) => {
    const oneDayAgo = new Date(Date.now() - 24 * 60 * 60 * 1000);
    return state.data.detectionResults.filter(
      (result) => new Date(result.timestamp) > oneDayAgo,
    );
  },

  getHighConfidenceAnomalies: (state) => {
    return state.data.detectionResults.filter(
      (result) => result.confidence > 0.8,
    );
  },

  getAnomalyTypeDistribution: (state) => {
    const distribution = {};
    state.data.detectionResults.forEach((result) => {
      const type = result.type || "Unknown";
      distribution[type] = (distribution[type] || 0) + 1;
    });
    return distribution;
  },

  getSystemHealth: (state) => {
    const recentAnomalies = selectors.getRecentAnomalies(state);
    const criticalCount = recentAnomalies.filter(
      (a) => a.confidence > 0.95,
    ).length;

    if (criticalCount > 5) return "critical";
    if (criticalCount > 2) return "warning";
    if (recentAnomalies.length > 10) return "caution";
    return "good";
  },

  // Chart Selectors
  getChartInstances: (state) => state.charts.instances,
  getActiveFilters: (state) => state.charts.activeFilters,
  getSelectedData: (state) => state.charts.selectedData,

  // Dashboard Selectors
  getDashboardLayout: (state) => state.dashboard.layout,
  getDashboardWidgets: (state) => state.dashboard.widgets,
  getDashboardFilters: (state) => state.dashboard.filters,
  isAutoRefreshEnabled: (state) => state.dashboard.autoRefresh,
};

// Initialize real-time manager
const realTimeManager = new RealTimeManager(appStore);

// Auto-connect on page load if user preferences allow
document.addEventListener("DOMContentLoaded", () => {
  const userPrefs = useStore(selectors.getUserPreferences);
  if (userPrefs.realTimeUpdates) {
    realTimeManager.connect();
  }
});

// Export for use in other modules
if (typeof module !== "undefined" && module.exports) {
  module.exports = {
    StateStore,
    appStore,
    chartStore,
    formStore,
    RealTimeManager,
    realTimeManager,
    useStore,
    useStoreSubscription,
    selectors,
  };
} else {
  // Browser environment
  window.StateStore = StateStore;
  window.appStore = appStore;
  window.chartStore = chartStore;
  window.formStore = formStore;
  window.RealTimeManager = RealTimeManager;
  window.realTimeManager = realTimeManager;
  window.useStore = useStore;
  window.useStoreSubscription = useStoreSubscription;
  window.selectors = selectors;
}
\n\n// state-manager.js\n/**
 * Advanced State Management System
 *
 * Lightweight state management inspired by Redux with reactive subscriptions
 * Features middleware support, time-travel debugging, and async action handling
 */

export class StateManager {
  constructor(initialState = {}, options = {}) {
    this.state = { ...initialState };
    this.listeners = new Set();
    this.middleware = [];
    this.history = [];
    this.historyIndex = -1;
    this.options = {
      maxHistorySize: 50,
      enableTimeTravel: true,
      enableLogging: false,
      persistState: false,
      persistKey: "pynomaly-state",
      ...options,
    };

    this.actionQueue = [];
    this.isDispatching = false;

    this.init();
  }

  init() {
    // Load persisted state
    if (this.options.persistState) {
      this.loadPersistedState();
    }

    // Add default middleware
    if (this.options.enableLogging) {
      this.use(this.createLoggingMiddleware());
    }

    // Save initial state to history
    if (this.options.enableTimeTravel) {
      this.saveToHistory({
        type: "@@INIT",
        payload: null,
      });
    }
  }

  /**
   * Dispatch an action to update state
   */
  dispatch(action) {
    if (typeof action !== "object" || action === null) {
      throw new Error("Action must be a plain object");
    }

    if (typeof action.type !== "string") {
      throw new Error("Action must have a type property");
    }

    // Handle async actions
    if (typeof action === "function") {
      return action(this.dispatch.bind(this), this.getState.bind(this));
    }

    // Queue actions if currently dispatching
    if (this.isDispatching) {
      this.actionQueue.push(action);
      return;
    }

    this.isDispatching = true;

    try {
      // Apply middleware
      const middlewareChain = this.middleware.slice();
      let dispatch = this.dispatchAction.bind(this);

      // Compose middleware
      for (let i = middlewareChain.length - 1; i >= 0; i--) {
        const middleware = middlewareChain[i];
        dispatch = middleware(this)(dispatch);
      }

      // Execute dispatch
      const result = dispatch(action);

      // Process queued actions
      this.processActionQueue();

      return result;
    } finally {
      this.isDispatching = false;
    }
  }

  dispatchAction(action) {
    const prevState = this.state;

    // Create new state
    this.state = this.reduce(this.state, action);

    // Save to history
    if (this.options.enableTimeTravel) {
      this.saveToHistory(action);
    }

    // Persist state
    if (this.options.persistState) {
      this.persistState();
    }

    // Notify listeners
    this.notifyListeners(prevState, this.state, action);

    return action;
  }

  processActionQueue() {
    while (this.actionQueue.length > 0) {
      const action = this.actionQueue.shift();
      this.dispatchAction(action);
    }
  }

  /**
   * Get current state
   */
  getState() {
    return { ...this.state };
  }

  /**
   * Subscribe to state changes
   */
  subscribe(listener) {
    if (typeof listener !== "function") {
      throw new Error("Listener must be a function");
    }

    this.listeners.add(listener);

    // Return unsubscribe function
    return () => {
      this.listeners.delete(listener);
    };
  }

  /**
   * Add middleware
   */
  use(middleware) {
    if (typeof middleware !== "function") {
      throw new Error("Middleware must be a function");
    }

    this.middleware.push(middleware);
    return this;
  }

  /**
   * Main reducer function
   */
  reduce(state, action) {
    // Handle built-in actions
    switch (action.type) {
      case "@@STATE/RESET":
        return action.payload || {};

      case "@@STATE/MERGE":
        return { ...state, ...action.payload };

      case "@@STATE/SET_PROPERTY":
        return this.setNestedProperty(
          state,
          action.payload.path,
          action.payload.value,
        );

      case "@@STATE/DELETE_PROPERTY":
        return this.deleteNestedProperty(state, action.payload.path);

      default:
        return this.handleCustomAction(state, action);
    }
  }

  /**
   * Handle custom actions - override in subclasses
   */
  handleCustomAction(state, action) {
    // Default: return state unchanged
    return state;
  }

  setNestedProperty(obj, path, value) {
    const keys = Array.isArray(path) ? path : path.split(".");
    const result = { ...obj };
    let current = result;

    for (let i = 0; i < keys.length - 1; i++) {
      const key = keys[i];
      current[key] = { ...current[key] };
      current = current[key];
    }

    current[keys[keys.length - 1]] = value;
    return result;
  }

  deleteNestedProperty(obj, path) {
    const keys = Array.isArray(path) ? path : path.split(".");
    const result = { ...obj };
    let current = result;

    for (let i = 0; i < keys.length - 1; i++) {
      const key = keys[i];
      current[key] = { ...current[key] };
      current = current[key];
    }

    delete current[keys[keys.length - 1]];
    return result;
  }

  notifyListeners(prevState, nextState, action) {
    this.listeners.forEach((listener) => {
      try {
        listener(nextState, prevState, action);
      } catch (error) {
        console.error("Error in state listener:", error);
      }
    });
  }

  saveToHistory(action) {
    // Remove future history if we're not at the end
    if (this.historyIndex < this.history.length - 1) {
      this.history = this.history.slice(0, this.historyIndex + 1);
    }

    // Add new state to history
    this.history.push({
      state: { ...this.state },
      action: { ...action },
      timestamp: Date.now(),
    });

    // Limit history size
    if (this.history.length > this.options.maxHistorySize) {
      this.history.shift();
    } else {
      this.historyIndex++;
    }
  }

  // Time travel methods
  undo() {
    if (!this.options.enableTimeTravel || this.historyIndex <= 0) {
      return false;
    }

    this.historyIndex--;
    const historyEntry = this.history[this.historyIndex];
    const prevState = this.state;

    this.state = { ...historyEntry.state };
    this.notifyListeners(prevState, this.state, {
      type: "@@HISTORY/UNDO",
      payload: { historyIndex: this.historyIndex },
    });

    return true;
  }

  redo() {
    if (
      !this.options.enableTimeTravel ||
      this.historyIndex >= this.history.length - 1
    ) {
      return false;
    }

    this.historyIndex++;
    const historyEntry = this.history[this.historyIndex];
    const prevState = this.state;

    this.state = { ...historyEntry.state };
    this.notifyListeners(prevState, this.state, {
      type: "@@HISTORY/REDO",
      payload: { historyIndex: this.historyIndex },
    });

    return true;
  }

  jumpToHistory(index) {
    if (
      !this.options.enableTimeTravel ||
      index < 0 ||
      index >= this.history.length
    ) {
      return false;
    }

    const historyEntry = this.history[index];
    const prevState = this.state;

    this.historyIndex = index;
    this.state = { ...historyEntry.state };
    this.notifyListeners(prevState, this.state, {
      type: "@@HISTORY/JUMP",
      payload: { historyIndex: index },
    });

    return true;
  }

  getHistory() {
    return this.history.map((entry, index) => ({
      ...entry,
      index,
      isCurrent: index === this.historyIndex,
    }));
  }

  // State persistence
  persistState() {
    try {
      localStorage.setItem(this.options.persistKey, JSON.stringify(this.state));
    } catch (error) {
      console.warn("Failed to persist state:", error);
    }
  }

  loadPersistedState() {
    try {
      const saved = localStorage.getItem(this.options.persistKey);
      if (saved) {
        this.state = { ...this.state, ...JSON.parse(saved) };
      }
    } catch (error) {
      console.warn("Failed to load persisted state:", error);
    }
  }

  clearPersistedState() {
    try {
      localStorage.removeItem(this.options.persistKey);
    } catch (error) {
      console.warn("Failed to clear persisted state:", error);
    }
  }

  // Middleware creators
  createLoggingMiddleware() {
    return (store) => (next) => (action) => {
      const prevState = store.getState();
      console.group(`🎯 Action: ${action.type}`);
      console.log("📤 Dispatching:", action);
      console.log("📋 Previous state:", prevState);

      const result = next(action);

      console.log("📋 Next state:", store.getState());
      console.groupEnd();

      return result;
    };
  }

  createAsyncMiddleware() {
    return (store) => (next) => (action) => {
      if (typeof action === "function") {
        return action(store.dispatch, store.getState);
      }

      if (action && typeof action.then === "function") {
        return action.then(store.dispatch);
      }

      return next(action);
    };
  }

  createValidationMiddleware(validators = {}) {
    return (store) => (next) => (action) => {
      const validator = validators[action.type];
      if (validator && !validator(action, store.getState())) {
        console.warn(`Validation failed for action: ${action.type}`);
        return action;
      }

      return next(action);
    };
  }

  // Action creators
  createActions() {
    return {
      reset: (state = {}) => ({
        type: "@@STATE/RESET",
        payload: state,
      }),

      merge: (updates) => ({
        type: "@@STATE/MERGE",
        payload: updates,
      }),

      set: (path, value) => ({
        type: "@@STATE/SET_PROPERTY",
        payload: { path, value },
      }),

      delete: (path) => ({
        type: "@@STATE/DELETE_PROPERTY",
        payload: { path },
      }),
    };
  }

  // Utility methods
  select(selector) {
    if (typeof selector === "string") {
      return this.getNestedProperty(this.state, selector);
    }

    if (typeof selector === "function") {
      return selector(this.state);
    }

    return this.state;
  }

  getNestedProperty(obj, path) {
    const keys = Array.isArray(path) ? path : path.split(".");
    return keys.reduce((current, key) => current && current[key], obj);
  }

  // Batching support
  batch(actions) {
    const prevState = this.state;
    let finalAction = {
      type: "@@BATCH",
      payload: actions,
    };

    // Apply all actions without notifying listeners
    const originalNotify = this.notifyListeners;
    this.notifyListeners = () => {}; // Temporarily disable notifications

    try {
      actions.forEach((action) => this.dispatch(action));
      finalAction.payload = { actions, count: actions.length };
    } finally {
      this.notifyListeners = originalNotify;
    }

    // Notify listeners once with all changes
    this.notifyListeners(prevState, this.state, finalAction);
  }

  // DevTools integration
  connectDevTools() {
    if (typeof window !== "undefined" && window.__REDUX_DEVTOOLS_EXTENSION__) {
      this.devTools = window.__REDUX_DEVTOOLS_EXTENSION__.connect({
        name: "Pynomaly State Manager",
      });

      this.devTools.init(this.state);

      this.subscribe((state, prevState, action) => {
        this.devTools.send(action, state);
      });

      this.devTools.subscribe((message) => {
        if (message.type === "DISPATCH") {
          switch (message.payload.type) {
            case "JUMP_TO_STATE":
            case "JUMP_TO_ACTION":
              this.state = JSON.parse(message.state);
              this.notifyListeners({}, this.state, { type: "@@DEVTOOLS" });
              break;
          }
        }
      });
    }
  }

  destroy() {
    this.listeners.clear();
    this.middleware = [];
    this.history = [];

    if (this.devTools) {
      this.devTools.disconnect();
    }
  }
}

/**
 * React-like hooks for state management
 */
export class StateHooks {
  constructor(stateManager) {
    this.stateManager = stateManager;
    this.components = new Map();
  }

  useState(component, selector = (state) => state) {
    if (!this.components.has(component)) {
      this.components.set(component, {
        selectors: new Set(),
        unsubscribe: null,
      });
    }

    const componentData = this.components.get(component);
    componentData.selectors.add(selector);

    // Subscribe to state changes if not already subscribed
    if (!componentData.unsubscribe) {
      componentData.unsubscribe = this.stateManager.subscribe(
        (state, prevState, action) => {
          // Check if any selector results changed
          const hasChanges = Array.from(componentData.selectors).some((sel) => {
            const current = sel(state);
            const previous = sel(prevState);
            return !this.shallowEqual(current, previous);
          });

          if (hasChanges && typeof component.forceUpdate === "function") {
            component.forceUpdate();
          } else if (hasChanges && typeof component.render === "function") {
            component.render();
          }
        },
      );
    }

    return [
      this.stateManager.select(selector),
      this.stateManager.dispatch.bind(this.stateManager),
    ];
  }

  useDispatch() {
    return this.stateManager.dispatch.bind(this.stateManager);
  }

  useSelector(selector) {
    return this.stateManager.select(selector);
  }

  useActions(actionCreators) {
    const dispatch = this.stateManager.dispatch.bind(this.stateManager);

    if (typeof actionCreators === "function") {
      return (...args) => dispatch(actionCreators(...args));
    }

    if (typeof actionCreators === "object") {
      const boundActions = {};
      Object.keys(actionCreators).forEach((key) => {
        boundActions[key] = (...args) => dispatch(actionCreators[key](...args));
      });
      return boundActions;
    }

    return dispatch;
  }

  cleanup(component) {
    const componentData = this.components.get(component);
    if (componentData && componentData.unsubscribe) {
      componentData.unsubscribe();
      this.components.delete(component);
    }
  }

  shallowEqual(obj1, obj2) {
    if (obj1 === obj2) return true;

    if (
      typeof obj1 !== "object" ||
      typeof obj2 !== "object" ||
      obj1 === null ||
      obj2 === null
    ) {
      return false;
    }

    const keys1 = Object.keys(obj1);
    const keys2 = Object.keys(obj2);

    if (keys1.length !== keys2.length) return false;

    for (let key of keys1) {
      if (obj1[key] !== obj2[key]) return false;
    }

    return true;
  }
}

/**
 * Anomaly Detection specific state manager
 */
export class AnomalyStateManager extends StateManager {
  constructor(initialState = {}) {
    const defaultState = {
      datasets: [],
      models: [],
      results: [],
      currentDataset: null,
      currentModel: null,
      isLoading: false,
      error: null,
      ui: {
        activeTab: "overview",
        sidebarOpen: true,
        notifications: [],
        theme: "light",
      },
      realTime: {
        connected: false,
        streamingData: [],
        alerts: [],
      },
      ...initialState,
    };

    super(defaultState, {
      enableTimeTravel: true,
      enableLogging: process.env.NODE_ENV === "development",
      persistState: true,
      persistKey: "pynomaly-app-state",
    });

    // Add async middleware
    this.use(this.createAsyncMiddleware());
  }

  handleCustomAction(state, action) {
    switch (action.type) {
      case "DATASETS/LOAD_REQUEST":
        return { ...state, isLoading: true, error: null };

      case "DATASETS/LOAD_SUCCESS":
        return {
          ...state,
          datasets: action.payload,
          isLoading: false,
          error: null,
        };

      case "DATASETS/LOAD_FAILURE":
        return {
          ...state,
          datasets: [],
          isLoading: false,
          error: action.payload,
        };

      case "DATASETS/SELECT":
        return { ...state, currentDataset: action.payload };

      case "MODELS/LOAD_SUCCESS":
        return { ...state, models: action.payload };

      case "MODELS/SELECT":
        return { ...state, currentModel: action.payload };

      case "RESULTS/ADD":
        return {
          ...state,
          results: [...state.results, action.payload],
        };

      case "RESULTS/UPDATE":
        return {
          ...state,
          results: state.results.map((result) =>
            result.id === action.payload.id
              ? { ...result, ...action.payload }
              : result,
          ),
        };

      case "UI/SET_ACTIVE_TAB":
        return {
          ...state,
          ui: { ...state.ui, activeTab: action.payload },
        };

      case "UI/TOGGLE_SIDEBAR":
        return {
          ...state,
          ui: { ...state.ui, sidebarOpen: !state.ui.sidebarOpen },
        };

      case "UI/ADD_NOTIFICATION":
        return {
          ...state,
          ui: {
            ...state.ui,
            notifications: [...state.ui.notifications, action.payload],
          },
        };

      case "UI/REMOVE_NOTIFICATION":
        return {
          ...state,
          ui: {
            ...state.ui,
            notifications: state.ui.notifications.filter(
              (notification) => notification.id !== action.payload,
            ),
          },
        };

      case "REALTIME/CONNECT":
        return {
          ...state,
          realTime: { ...state.realTime, connected: true },
        };

      case "REALTIME/DISCONNECT":
        return {
          ...state,
          realTime: { ...state.realTime, connected: false },
        };

      case "REALTIME/ADD_DATA":
        return {
          ...state,
          realTime: {
            ...state.realTime,
            streamingData: [...state.realTime.streamingData, action.payload],
          },
        };

      case "REALTIME/ADD_ALERT":
        return {
          ...state,
          realTime: {
            ...state.realTime,
            alerts: [...state.realTime.alerts, action.payload],
          },
        };

      default:
        return state;
    }
  }
}

/**
 * Action creators for anomaly detection
 */
export const anomalyActions = {
  // Dataset actions
  loadDatasets: () => async (dispatch, getState) => {
    dispatch({ type: "DATASETS/LOAD_REQUEST" });

    try {
      const response = await fetch("/api/datasets");
      const datasets = await response.json();
      dispatch({ type: "DATASETS/LOAD_SUCCESS", payload: datasets });
    } catch (error) {
      dispatch({ type: "DATASETS/LOAD_FAILURE", payload: error.message });
    }
  },

  selectDataset: (dataset) => ({
    type: "DATASETS/SELECT",
    payload: dataset,
  }),

  // Model actions
  loadModels: () => async (dispatch) => {
    try {
      const response = await fetch("/api/models");
      const models = await response.json();
      dispatch({ type: "MODELS/LOAD_SUCCESS", payload: models });
    } catch (error) {
      console.error("Failed to load models:", error);
    }
  },

  selectModel: (model) => ({
    type: "MODELS/SELECT",
    payload: model,
  }),

  // Result actions
  addResult: (result) => ({
    type: "RESULTS/ADD",
    payload: { ...result, id: Date.now(), timestamp: new Date().toISOString() },
  }),

  updateResult: (id, updates) => ({
    type: "RESULTS/UPDATE",
    payload: { id, ...updates },
  }),

  // UI actions
  setActiveTab: (tab) => ({
    type: "UI/SET_ACTIVE_TAB",
    payload: tab,
  }),

  toggleSidebar: () => ({
    type: "UI/TOGGLE_SIDEBAR",
  }),

  addNotification: (message, type = "info", duration = 5000) => ({
    type: "UI/ADD_NOTIFICATION",
    payload: {
      id: Date.now(),
      message,
      type,
      timestamp: Date.now(),
      duration,
    },
  }),

  removeNotification: (id) => ({
    type: "UI/REMOVE_NOTIFICATION",
    payload: id,
  }),

  // Real-time actions
  connectRealTime: () => ({
    type: "REALTIME/CONNECT",
  }),

  disconnectRealTime: () => ({
    type: "REALTIME/DISCONNECT",
  }),

  addStreamingData: (data) => ({
    type: "REALTIME/ADD_DATA",
    payload: data,
  }),

  addAlert: (alert) => ({
    type: "REALTIME/ADD_ALERT",
    payload: {
      ...alert,
      id: Date.now(),
      timestamp: new Date().toISOString(),
    },
  }),
};

// Default export
export default StateManager;
\n\n// sync-manager.js\n/**
 * Sync Manager - Handles data synchronization between offline and online modes
 * Manages background sync, conflict resolution, and data consistency
 */
export class SyncManager {
  constructor() {
    this.syncQueue = [];
    this.conflictQueue = [];
    this.isOnline = navigator.onLine;
    this.isSyncing = false;
    this.syncStrategy = "smart"; // 'immediate', 'smart', 'manual'
    this.retryAttempts = 3;
    this.retryDelay = 1000; // ms

    this.init();
  }

  async init() {
    this.setupEventListeners();
    await this.loadSyncQueue();
    this.startPeriodicSync();
  }

  /**
   * Setup event listeners for online/offline detection
   */
  setupEventListeners() {
    window.addEventListener("online", () => {
      this.isOnline = true;
      this.onConnectionRestore();
    });

    window.addEventListener("offline", () => {
      this.isOnline = false;
      this.onConnectionLost();
    });

    // Listen for service worker messages
    if ("serviceWorker" in navigator) {
      navigator.serviceWorker.addEventListener("message", (event) => {
        this.handleServiceWorkerMessage(event.data);
      });
    }
  }

  /**
   * Handle service worker messages
   */
  handleServiceWorkerMessage(data) {
    const { type, payload } = data;

    switch (type) {
      case "SYNC_COMPLETE":
        this.onSyncComplete(payload);
        break;
      case "SYNC_FAILED":
        this.onSyncFailed(payload);
        break;
      case "CONFLICT_DETECTED":
        this.onConflictDetected(payload);
        break;
      case "SYNC_QUEUE_UPDATE":
        this.onSyncQueueUpdate(payload);
        break;
    }
  }

  /**
   * Load sync queue from IndexedDB
   */
  async loadSyncQueue() {
    try {
      if ("serviceWorker" in navigator) {
        const registration = await navigator.serviceWorker.getRegistration();
        if (registration?.active) {
          registration.active.postMessage({ type: "GET_SYNC_QUEUE" });
        }
      }
    } catch (error) {
      console.error("[SyncManager] Failed to load sync queue:", error);
    }
  }

  /**
   * Queue data for synchronization
   */
  async queueForSync(operation, data, priority = "normal") {
    const syncItem = {
      id: this.generateSyncId(),
      operation, // 'create', 'update', 'delete'
      entityType: data.entityType, // 'dataset', 'result', 'model', etc.
      entityId: data.entityId,
      data: data,
      priority, // 'high', 'normal', 'low'
      timestamp: Date.now(),
      retryCount: 0,
      status: "pending", // 'pending', 'syncing', 'completed', 'failed'
      conflicts: [],
    };

    this.syncQueue.push(syncItem);
    await this.persistSyncQueue();

    // Trigger immediate sync for high priority items when online
    if (
      this.isOnline &&
      priority === "high" &&
      this.syncStrategy !== "manual"
    ) {
      this.processSyncQueue();
    }

    this.notifyUI("sync_queue_updated", {
      pendingCount: this.getPendingCount(),
    });
    return syncItem.id;
  }

  /**
   * Process sync queue
   */
  async processSyncQueue() {
    if (this.isSyncing || !this.isOnline) return;

    this.isSyncing = true;
    this.notifyUI("sync_started");

    try {
      // Sort by priority and timestamp
      const pendingItems = this.syncQueue
        .filter((item) => item.status === "pending")
        .sort((a, b) => {
          const priorityOrder = { high: 3, normal: 2, low: 1 };
          const priorityDiff =
            priorityOrder[b.priority] - priorityOrder[a.priority];
          return priorityDiff !== 0 ? priorityDiff : a.timestamp - b.timestamp;
        });

      const results = {
        completed: 0,
        failed: 0,
        conflicts: 0,
      };

      for (const item of pendingItems) {
        try {
          item.status = "syncing";
          await this.persistSyncQueue();

          const result = await this.syncItem(item);

          if (result.success) {
            item.status = "completed";
            item.completedAt = Date.now();
            results.completed++;
          } else if (result.conflict) {
            item.status = "conflict";
            item.conflicts.push(result.conflict);
            this.conflictQueue.push(item);
            results.conflicts++;
          } else {
            throw new Error(result.error || "Sync failed");
          }
        } catch (error) {
          console.error("[SyncManager] Failed to sync item:", item.id, error);
          item.retryCount++;

          if (item.retryCount >= this.retryAttempts) {
            item.status = "failed";
            item.error = error.message;
            results.failed++;
          } else {
            item.status = "pending";
            // Exponential backoff
            await this.delay(
              this.retryDelay * Math.pow(2, item.retryCount - 1),
            );
          }
        }

        await this.persistSyncQueue();
      }

      // Clean up completed items older than 24 hours
      this.cleanupCompletedItems();

      this.notifyUI("sync_completed", results);
    } catch (error) {
      console.error("[SyncManager] Sync processing failed:", error);
      this.notifyUI("sync_failed", { error: error.message });
    } finally {
      this.isSyncing = false;
    }
  }

  /**
   * Sync individual item
   */
  async syncItem(item) {
    const { operation, entityType, entityId, data } = item;

    try {
      let endpoint;
      let method;
      let payload;

      switch (operation) {
        case "create":
          endpoint = `/api/${entityType}s`;
          method = "POST";
          payload = data.payload;
          break;
        case "update":
          endpoint = `/api/${entityType}s/${entityId}`;
          method = "PUT";
          payload = data.payload;
          break;
        case "delete":
          endpoint = `/api/${entityType}s/${entityId}`;
          method = "DELETE";
          break;
        default:
          throw new Error(`Unknown operation: ${operation}`);
      }

      // Check for conflicts before syncing
      const conflictCheck = await this.checkForConflicts(item);
      if (conflictCheck.hasConflict) {
        return { success: false, conflict: conflictCheck.conflict };
      }

      const response = await fetch(endpoint, {
        method,
        headers: {
          "Content-Type": "application/json",
          Authorization: this.getAuthHeader(),
        },
        body: payload ? JSON.stringify(payload) : undefined,
      });

      if (!response.ok) {
        if (response.status === 409) {
          // Conflict detected
          const conflictData = await response.json();
          return { success: false, conflict: conflictData };
        }
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const result = await response.json();

      // Update local data with server response
      await this.updateLocalData(item, result);

      return { success: true, data: result };
    } catch (error) {
      return { success: false, error: error.message };
    }
  }

  /**
   * Check for conflicts before syncing
   */
  async checkForConflicts(item) {
    if (item.operation === "create") {
      return { hasConflict: false };
    }

    try {
      const response = await fetch(
        `/api/${item.entityType}s/${item.entityId}`,
        {
          method: "HEAD",
          headers: {
            Authorization: this.getAuthHeader(),
          },
        },
      );

      if (response.status === 404) {
        // Entity no longer exists on server
        if (item.operation === "update") {
          return {
            hasConflict: true,
            conflict: {
              type: "entity_deleted",
              message: "Entity was deleted on server",
              serverVersion: null,
              localVersion: item.data.version,
            },
          };
        }
      }

      const serverVersion =
        response.headers.get("etag") || response.headers.get("last-modified");
      const localVersion = item.data.version;

      if (serverVersion && localVersion && serverVersion !== localVersion) {
        return {
          hasConflict: true,
          conflict: {
            type: "version_mismatch",
            message: "Entity was modified on server",
            serverVersion,
            localVersion,
          },
        };
      }

      return { hasConflict: false };
    } catch (error) {
      console.warn("[SyncManager] Conflict check failed:", error);
      return { hasConflict: false }; // Proceed with sync
    }
  }

  /**
   * Resolve conflict using specified strategy
   */
  async resolveConflict(conflictId, strategy, resolution = null) {
    const conflict = this.conflictQueue.find((c) => c.id === conflictId);
    if (!conflict) {
      throw new Error("Conflict not found");
    }

    try {
      let resolvedData;

      switch (strategy) {
        case "server_wins":
          resolvedData = await this.fetchServerVersion(conflict);
          await this.updateLocalData(conflict, resolvedData);
          break;
        case "client_wins":
          // Force sync local version
          conflict.retryCount = 0;
          conflict.status = "pending";
          await this.forceSyncItem(conflict);
          break;
        case "merge":
          if (!resolution) {
            throw new Error("Merge resolution data required");
          }
          resolvedData = await this.mergeVersions(conflict, resolution);
          await this.updateLocalData(conflict, resolvedData);
          await this.syncMergedData(conflict, resolvedData);
          break;
        case "manual":
          // User will resolve manually
          conflict.status = "manual_resolution";
          break;
        default:
          throw new Error(`Unknown resolution strategy: ${strategy}`);
      }

      // Remove from conflict queue
      this.conflictQueue = this.conflictQueue.filter(
        (c) => c.id !== conflictId,
      );

      // Update sync queue
      const syncItem = this.syncQueue.find((s) => s.id === conflictId);
      if (syncItem && strategy !== "manual") {
        syncItem.status = "completed";
        syncItem.resolvedAt = Date.now();
        syncItem.resolutionStrategy = strategy;
      }

      await this.persistSyncQueue();
      this.notifyUI("conflict_resolved", { conflictId, strategy });

      return { success: true };
    } catch (error) {
      console.error("[SyncManager] Failed to resolve conflict:", error);
      return { success: false, error: error.message };
    }
  }

  /**
   * Start periodic background sync
   */
  startPeriodicSync() {
    if (this.syncStrategy === "manual") return;

    const interval = this.syncStrategy === "immediate" ? 30000 : 300000; // 30s or 5min

    setInterval(() => {
      if (this.isOnline && this.getPendingCount() > 0) {
        this.processSyncQueue();
      }
    }, interval);
  }

  /**
   * Force manual sync
   */
  async forceSyncAll() {
    if (!this.isOnline) {
      throw new Error("Cannot sync while offline");
    }

    await this.processSyncQueue();
  }

  /**
   * Get sync status
   */
  getSyncStatus() {
    const pending = this.syncQueue.filter(
      (item) => item.status === "pending",
    ).length;
    const syncing = this.syncQueue.filter(
      (item) => item.status === "syncing",
    ).length;
    const failed = this.syncQueue.filter(
      (item) => item.status === "failed",
    ).length;
    const conflicts = this.conflictQueue.length;

    return {
      isOnline: this.isOnline,
      isSyncing: this.isSyncing,
      pending,
      syncing,
      failed,
      conflicts,
      strategy: this.syncStrategy,
      lastSyncAt: this.getLastSyncTime(),
    };
  }

  /**
   * Set sync strategy
   */
  setSyncStrategy(strategy) {
    if (!["immediate", "smart", "manual"].includes(strategy)) {
      throw new Error("Invalid sync strategy");
    }

    this.syncStrategy = strategy;
    this.notifyUI("sync_strategy_changed", { strategy });
  }

  /**
   * Clear completed sync items
   */
  async clearCompleted() {
    this.syncQueue = this.syncQueue.filter(
      (item) => item.status !== "completed",
    );
    await this.persistSyncQueue();
    this.notifyUI("completed_cleared");
  }

  /**
   * Event handlers
   */
  onConnectionRestore() {
    console.log("[SyncManager] Connection restored");
    this.notifyUI("connection_restored");

    if (this.syncStrategy !== "manual" && this.getPendingCount() > 0) {
      setTimeout(() => this.processSyncQueue(), 1000);
    }
  }

  onConnectionLost() {
    console.log("[SyncManager] Connection lost");
    this.notifyUI("connection_lost");
  }

  onSyncComplete(payload) {
    this.notifyUI("sync_item_completed", payload);
  }

  onSyncFailed(payload) {
    this.notifyUI("sync_item_failed", payload);
  }

  onConflictDetected(payload) {
    this.conflictQueue.push(payload);
    this.notifyUI("conflict_detected", payload);
  }

  onSyncQueueUpdate(payload) {
    this.syncQueue = payload.queue || [];
    this.notifyUI("sync_queue_updated", {
      pendingCount: this.getPendingCount(),
    });
  }

  // Helper methods...

  generateSyncId() {
    return `sync_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  getPendingCount() {
    return this.syncQueue.filter((item) =>
      ["pending", "syncing"].includes(item.status),
    ).length;
  }

  getLastSyncTime() {
    const completed = this.syncQueue.filter(
      (item) => item.status === "completed",
    );
    if (!completed.length) return null;

    return Math.max(...completed.map((item) => item.completedAt));
  }

  async persistSyncQueue() {
    try {
      if ("serviceWorker" in navigator) {
        const registration = await navigator.serviceWorker.getRegistration();
        if (registration?.active) {
          registration.active.postMessage({
            type: "UPDATE_SYNC_QUEUE",
            payload: { queue: this.syncQueue },
          });
        }
      }
    } catch (error) {
      console.error("[SyncManager] Failed to persist sync queue:", error);
    }
  }

  cleanupCompletedItems() {
    const oneDayAgo = Date.now() - 24 * 60 * 60 * 1000;
    this.syncQueue = this.syncQueue.filter(
      (item) => item.status !== "completed" || item.completedAt > oneDayAgo,
    );
  }

  async delay(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  getAuthHeader() {
    // Get auth token from storage or context
    const token = localStorage.getItem("auth_token");
    return token ? `Bearer ${token}` : "";
  }

  async fetchServerVersion(item) {
    const response = await fetch(`/api/${item.entityType}s/${item.entityId}`, {
      headers: { Authorization: this.getAuthHeader() },
    });
    return await response.json();
  }

  async updateLocalData(item, serverData) {
    // Update local IndexedDB with server data
    if ("serviceWorker" in navigator) {
      const registration = await navigator.serviceWorker.getRegistration();
      if (registration?.active) {
        registration.active.postMessage({
          type: "UPDATE_LOCAL_DATA",
          payload: {
            entityType: item.entityType,
            entityId: item.entityId,
            data: serverData,
          },
        });
      }
    }
  }

  async forceSyncItem(item) {
    // Force sync without conflict checking
    item.forceSync = true;
    return await this.syncItem(item);
  }

  async mergeVersions(conflict, resolution) {
    // Implement merge logic based on entity type
    return resolution.mergedData;
  }

  async syncMergedData(conflict, mergedData) {
    // Sync merged data to server
    const endpoint = `/api/${conflict.entityType}s/${conflict.entityId}`;
    const response = await fetch(endpoint, {
      method: "PUT",
      headers: {
        "Content-Type": "application/json",
        Authorization: this.getAuthHeader(),
      },
      body: JSON.stringify(mergedData),
    });

    if (!response.ok) {
      throw new Error(`Failed to sync merged data: ${response.statusText}`);
    }

    return await response.json();
  }

  notifyUI(eventType, data = {}) {
    // Dispatch custom events for UI updates
    window.dispatchEvent(
      new CustomEvent("sync-manager", {
        detail: { type: eventType, data },
      }),
    );
  }

  /**
   * Public API methods
   */

  // Queue operations for different entity types
  async queueDatasetSync(operation, dataset, priority = "normal") {
    return await this.queueForSync(
      operation,
      {
        entityType: "dataset",
        entityId: dataset.id,
        payload: dataset,
        version: dataset.version,
      },
      priority,
    );
  }

  async queueResultSync(operation, result, priority = "normal") {
    return await this.queueForSync(
      operation,
      {
        entityType: "result",
        entityId: result.id,
        payload: result,
        version: result.version,
      },
      priority,
    );
  }

  async queueModelSync(operation, model, priority = "high") {
    return await this.queueForSync(
      operation,
      {
        entityType: "model",
        entityId: model.id,
        payload: model,
        version: model.version,
      },
      priority,
    );
  }

  // Get conflicts for UI
  getConflicts() {
    return this.conflictQueue.map((conflict) => ({
      id: conflict.id,
      entityType: conflict.entityType,
      entityId: conflict.entityId,
      operation: conflict.operation,
      conflicts: conflict.conflicts,
      timestamp: conflict.timestamp,
    }));
  }

  // Get pending sync items for UI
  getPendingItems() {
    return this.syncQueue
      .filter((item) => ["pending", "syncing", "failed"].includes(item.status))
      .map((item) => ({
        id: item.id,
        operation: item.operation,
        entityType: item.entityType,
        priority: item.priority,
        status: item.status,
        retryCount: item.retryCount,
        timestamp: item.timestamp,
      }));
  }
}

// Initialize and expose globally
if (typeof window !== "undefined") {
  window.SyncManager = new SyncManager();
}
\n\n