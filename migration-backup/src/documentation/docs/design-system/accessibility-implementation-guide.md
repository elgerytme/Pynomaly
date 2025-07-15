# Accessibility Implementation Guide

🍞 **Breadcrumb:** 🏠 [Home](../index.md) > 📁 Design-System

---


## Overview

This comprehensive guide provides detailed implementation strategies for achieving and maintaining WCAG 2.1 AA compliance across the Pynomaly platform. It covers practical implementation patterns, testing procedures, and ongoing maintenance for creating accessible anomaly detection interfaces.

## WCAG 2.1 AA Implementation Standards

### 1. Perceivable - Information and UI components must be presentable to users in ways they can perceive

#### 1.1 Text Alternatives

**1.1.1 Non-text Content (Level A)**

All non-text content must have text alternatives that serve the equivalent purpose.

**Implementation Examples:**

```html
<!-- Images with meaningful content -->
<img src="anomaly-trend-chart.png"
     alt="Anomaly detection trend showing 15 anomalies detected over the past 24 hours, with highest activity between 2-4 PM">

<!-- Decorative images -->
<img src="decorative-pattern.svg"
     alt=""
     role="presentation">

<!-- Complex charts and visualizations -->
<div class="chart-container"
     role="img"
     aria-labelledby="chart-title"
     aria-describedby="chart-summary">
  <h3 id="chart-title">Monthly Anomaly Detection Results</h3>
  <div id="chart-svg">
    <!-- D3.js or Canvas chart content -->
  </div>
  <div id="chart-summary" class="sr-only">
    This chart displays anomaly detection results for January through December 2025.
    January had 45 anomalies (highest), March had 38, and August had 12 (lowest).
    The overall trend shows higher anomaly rates in winter months.
  </div>
</div>

<!-- Alternative data table for charts -->
<details class="chart-data-table">
  <summary>View data table for Monthly Anomaly Results</summary>
  <table>
    <caption>Monthly anomaly detection results for 2025</caption>
    <thead>
      <tr>
        <th scope="col">Month</th>
        <th scope="col">Anomalies Detected</th>
        <th scope="col">Total Data Points</th>
        <th scope="col">Anomaly Rate</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <th scope="row">January</th>
        <td>45</td>
        <td>1,440</td>
        <td>3.1%</td>
      </tr>
      <!-- Additional rows -->
    </tbody>
  </table>
</details>

<!-- Interactive buttons with icons -->
<button type="button" class="btn btn--primary" aria-describedby="export-help">
  <svg class="btn__icon" aria-hidden="true" width="16" height="16">
    <use href="#icon-download"></use>
  </svg>
  <span class="btn__text">Export Data</span>
</button>
<div id="export-help" class="sr-only">
  Downloads the current dataset as a CSV file
</div>
```

#### 1.3 Adaptable Content

**1.3.1 Info and Relationships (Level A)**

Information, structure, and relationships must be programmatically determinable.

**Implementation Examples:**

```html
<!-- Proper heading hierarchy -->
<main>
  <h1>Anomaly Detection Dashboard</h1>

  <section>
    <h2>Recent Detections</h2>
    <article>
      <h3>Critical Anomalies</h3>
      <p>5 critical anomalies detected in the last hour</p>
    </article>
    <article>
      <h3>System Status</h3>
      <p>All detection algorithms running normally</p>
    </article>
  </section>

  <section>
    <h2>Detection History</h2>
    <!-- history content -->
  </section>
</main>

<!-- Form relationships -->
<form>
  <fieldset>
    <legend>Dataset Configuration</legend>

    <div class="form-group">
      <label for="dataset-name">Dataset Name <span class="required">*</span></label>
      <input type="text"
             id="dataset-name"
             name="dataset-name"
             required
             aria-describedby="name-help name-error">
      <div id="name-help" class="form-help">
        Enter a descriptive name for your dataset
      </div>
      <div id="name-error" class="form-error" role="alert" style="display: none;">
        Dataset name is required
      </div>
    </div>

    <div class="form-group">
      <fieldset>
        <legend>Detection Algorithm</legend>
        <div class="radio-group" role="radiogroup" aria-labelledby="algorithm-legend">
          <label class="radio-label">
            <input type="radio" name="algorithm" value="isolation-forest" checked>
            <span class="radio-text">Isolation Forest</span>
          </label>
          <label class="radio-label">
            <input type="radio" name="algorithm" value="one-class-svm">
            <span class="radio-text">One-Class SVM</span>
          </label>
        </div>
      </fieldset>
    </div>
  </fieldset>
</form>

<!-- Table relationships -->
<table class="data-table">
  <caption>
    Anomaly Detection Results
    <span class="table-summary">150 total results, sorted by timestamp</span>
  </caption>
  <thead>
    <tr>
      <th scope="col">
        <button type="button" aria-sort="ascending" class="sort-button">
          Timestamp
          <span class="sort-indicator" aria-hidden="true">↑</span>
        </button>
      </th>
      <th scope="col">Value</th>
      <th scope="col">Anomaly Score</th>
      <th scope="col">Status</th>
      <th scope="col">Actions</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>2025-06-26 14:30:00</td>
      <td>145.67</td>
      <td>0.85</td>
      <td>
        <span class="status-badge status-badge--anomaly" aria-label="Anomaly detected">
          Anomaly
        </span>
      </td>
      <td>
        <button type="button" class="btn btn--sm"
                aria-label="View details for anomaly at 2025-06-26 14:30:00">
          Details
        </button>
      </td>
    </tr>
  </tbody>
</table>
```

#### 1.4 Distinguishable Content

**1.4.3 Contrast (Minimum) (Level AA)**

Text must have a contrast ratio of at least 4.5:1 (3:1 for large text).

**Implementation:**

```css
/* WCAG AA compliant color combinations */
:root {
  /* High contrast text combinations */
  --color-text-primary: #1e293b;     /* 15.36:1 on white */
  --color-text-secondary: #475569;   /* 8.33:1 on white */
  --color-text-muted: #64748b;       /* 4.78:1 on white */

  /* Status colors with proper contrast */
  --color-success-text: #166534;     /* 5.85:1 on white */
  --color-warning-text: #92400e;     /* 4.56:1 on white */
  --color-danger-text: #991b1b;      /* 6.64:1 on white */

  /* Focus indicators */
  --color-focus-ring: #2563eb;       /* High contrast focus ring */
  --shadow-focus: 0 0 0 3px rgba(37, 99, 235, 0.5);
}

/* Status indicators with sufficient contrast */
.status-badge {
  font-weight: 600;
  font-size: 0.875rem;
  padding: 0.25rem 0.75rem;
  border-radius: 0.375rem;
}

.status-badge--normal {
  background-color: #dcfce7;  /* Light green background */
  color: #166534;             /* Dark green text: 5.85:1 contrast */
}

.status-badge--anomaly {
  background-color: #fecaca;  /* Light red background */
  color: #991b1b;             /* Dark red text: 6.64:1 contrast */
}

.status-badge--warning {
  background-color: #fef3c7;  /* Light yellow background */
  color: #92400e;             /* Dark brown text: 4.56:1 contrast */
}

/* Focus states with high contrast */
.btn:focus-visible {
  outline: none;
  box-shadow: var(--shadow-focus);
}

.form-input:focus {
  outline: none;
  border-color: var(--color-focus-ring);
  box-shadow: var(--shadow-focus);
}
```

### 2. Operable - UI components and navigation must be operable

#### 2.1 Keyboard Accessible

**2.1.1 Keyboard (Level A)**

All functionality must be available from a keyboard.

**Implementation Examples:**

```html
<!-- Custom interactive elements with keyboard support -->
<div class="chart-controls">
  <button type="button"
          class="chart-control"
          aria-label="Zoom in on chart"
          data-action="zoom-in">
    <svg aria-hidden="true"><use href="#icon-zoom-in"></use></svg>
  </button>

  <button type="button"
          class="chart-control"
          aria-label="Zoom out on chart"
          data-action="zoom-out">
    <svg aria-hidden="true"><use href="#icon-zoom-out"></use></svg>
  </button>

  <div class="chart-range-selector"
       role="slider"
       tabindex="0"
       aria-label="Select time range"
       aria-valuemin="0"
       aria-valuemax="100"
       aria-valuenow="50"
       data-action="range-select">
    <div class="range-track">
      <div class="range-fill" style="width: 50%;"></div>
      <div class="range-thumb" style="left: 50%;"></div>
    </div>
  </div>
</div>

<script>
// Keyboard event handling
class ChartControls {
  constructor(container) {
    this.container = container;
    this.bindEvents();
  }

  bindEvents() {
    this.container.addEventListener('keydown', this.handleKeyDown.bind(this));
    this.container.addEventListener('click', this.handleClick.bind(this));
  }

  handleKeyDown(event) {
    const target = event.target;
    const action = target.dataset.action;

    switch (event.key) {
      case 'Enter':
      case ' ':
        if (action === 'zoom-in' || action === 'zoom-out') {
          event.preventDefault();
          this.handleZoom(action);
        }
        break;

      case 'ArrowLeft':
      case 'ArrowRight':
        if (action === 'range-select') {
          event.preventDefault();
          this.handleRangeChange(event.key, target);
        }
        break;

      case 'Home':
        if (action === 'range-select') {
          event.preventDefault();
          this.setRangeValue(target, 0);
        }
        break;

      case 'End':
        if (action === 'range-select') {
          event.preventDefault();
          this.setRangeValue(target, 100);
        }
        break;
    }
  }

  handleRangeChange(direction, element) {
    const currentValue = parseInt(element.getAttribute('aria-valuenow'));
    const step = event.shiftKey ? 10 : 1;
    const newValue = direction === 'ArrowLeft'
      ? Math.max(0, currentValue - step)
      : Math.min(100, currentValue + step);

    this.setRangeValue(element, newValue);
  }

  setRangeValue(element, value) {
    element.setAttribute('aria-valuenow', value);
    const fill = element.querySelector('.range-fill');
    const thumb = element.querySelector('.range-thumb');

    fill.style.width = `${value}%`;
    thumb.style.left = `${value}%`;

    // Announce change to screen readers
    this.announceChange(`Range value: ${value}%`);
  }

  announceChange(message) {
    const announcement = document.createElement('div');
    announcement.setAttribute('aria-live', 'polite');
    announcement.setAttribute('aria-atomic', 'true');
    announcement.className = 'sr-only';
    announcement.textContent = message;

    document.body.appendChild(announcement);
    setTimeout(() => document.body.removeChild(announcement), 1000);
  }
}
</script>
```

#### 2.4 Navigable

**2.4.1 Bypass Blocks (Level A)**

Provide mechanisms to bypass blocks of content.

**Implementation:**

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Pynomaly Dashboard</title>
  <style>
    .skip-link {
      position: absolute;
      top: -40px;
      left: 6px;
      background: #000;
      color: #fff;
      padding: 8px;
      text-decoration: none;
      z-index: 1000;
      border-radius: 0 0 4px 4px;
    }

    .skip-link:focus {
      top: 6px;
    }
  </style>
</head>
<body>
  <!-- Skip links -->
  <a href="#main-content" class="skip-link">Skip to main content</a>
  <a href="#navigation" class="skip-link">Skip to navigation</a>
  <a href="#search" class="skip-link">Skip to search</a>

  <!-- Header with navigation -->
  <header role="banner">
    <nav id="navigation" role="navigation" aria-label="Main navigation">
      <ul>
        <li><a href="/dashboard">Dashboard</a></li>
        <li><a href="/datasets">Datasets</a></li>
        <li><a href="/models">Models</a></li>
        <li><a href="/analysis">Analysis</a></li>
      </ul>
    </nav>

    <div id="search" role="search">
      <label for="search-input" class="sr-only">Search datasets and models</label>
      <input type="search"
             id="search-input"
             placeholder="Search datasets and models..."
             aria-describedby="search-help">
      <div id="search-help" class="sr-only">
        Use this search to find datasets, models, and analysis results
      </div>
    </div>
  </header>

  <!-- Main content -->
  <main id="main-content" role="main">
    <h1>Anomaly Detection Dashboard</h1>
    <!-- Main content here -->
  </main>

  <!-- Footer -->
  <footer role="contentinfo">
    <!-- Footer content -->
  </footer>
</body>
</html>
```

**2.4.3 Focus Order (Level A)**

Components receive focus in an order that preserves meaning and operability.

**Implementation:**

```html
<!-- Logical tab order in forms -->
<form class="dataset-form">
  <fieldset>
    <legend>Basic Information</legend>

    <!-- Tab order: 1 -->
    <label for="name">Dataset Name</label>
    <input type="text" id="name" name="name" tabindex="1">

    <!-- Tab order: 2 -->
    <label for="description">Description</label>
    <textarea id="description" name="description" tabindex="2"></textarea>
  </fieldset>

  <fieldset>
    <legend>Configuration</legend>

    <!-- Tab order: 3 -->
    <label for="algorithm">Algorithm</label>
    <select id="algorithm" name="algorithm" tabindex="3">
      <option value="isolation-forest">Isolation Forest</option>
      <option value="one-class-svm">One-Class SVM</option>
    </select>

    <!-- Tab order: 4 -->
    <label for="threshold">Threshold</label>
    <input type="number" id="threshold" name="threshold" tabindex="4">
  </fieldset>

  <!-- Tab order: 5, 6 -->
  <div class="form-actions">
    <button type="button" tabindex="6">Cancel</button>
    <button type="submit" tabindex="5">Create Dataset</button>
  </div>
</form>

<!-- Modal with focus management -->
<div id="settings-modal"
     class="modal"
     role="dialog"
     aria-modal="true"
     aria-labelledby="modal-title"
     style="display: none;">
  <div class="modal-content">
    <header class="modal-header">
      <h2 id="modal-title">Detection Settings</h2>
      <button type="button"
              class="modal-close"
              aria-label="Close settings dialog"
              data-action="close-modal">
        ×
      </button>
    </header>

    <div class="modal-body">
      <!-- Form content -->
    </div>

    <footer class="modal-footer">
      <button type="button" data-action="save">Save</button>
      <button type="button" data-action="cancel">Cancel</button>
    </footer>
  </div>
</div>

<script>
class ModalManager {
  constructor() {
    this.activeModal = null;
    this.lastFocusedElement = null;
  }

  openModal(modalId) {
    // Store the currently focused element
    this.lastFocusedElement = document.activeElement;

    const modal = document.getElementById(modalId);
    modal.style.display = 'block';
    this.activeModal = modal;

    // Focus first focusable element in modal
    const firstFocusable = modal.querySelector('button, input, select, textarea, [tabindex]:not([tabindex="-1"])');
    if (firstFocusable) {
      firstFocusable.focus();
    }

    // Trap focus within modal
    modal.addEventListener('keydown', this.trapFocus.bind(this));
  }

  closeModal() {
    if (this.activeModal) {
      this.activeModal.style.display = 'none';
      this.activeModal = null;

      // Return focus to triggering element
      if (this.lastFocusedElement) {
        this.lastFocusedElement.focus();
        this.lastFocusedElement = null;
      }
    }
  }

  trapFocus(event) {
    if (event.key !== 'Tab') return;

    const focusableElements = this.activeModal.querySelectorAll(
      'button, input, select, textarea, [tabindex]:not([tabindex="-1"])'
    );

    const firstElement = focusableElements[0];
    const lastElement = focusableElements[focusableElements.length - 1];

    if (event.shiftKey) {
      // Shift + Tab
      if (document.activeElement === firstElement) {
        event.preventDefault();
        lastElement.focus();
      }
    } else {
      // Tab
      if (document.activeElement === lastElement) {
        event.preventDefault();
        firstElement.focus();
      }
    }
  }
}
</script>
```

### 3. Understandable - Information and UI operation must be understandable

#### 3.1 Readable

**3.1.1 Language of Page (Level A)**

The default human language of each web page can be programmatically determined.

**Implementation:**

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Pynomaly - Anomaly Detection Platform</title>
</head>
<body>
  <!-- Content in English -->

  <!-- Mixed language content -->
  <section>
    <h2>Documentation</h2>
    <p>The algorithm uses <span lang="fr">machine learning</span> techniques...</p>
    <p>For more information, see: <cite lang="es">Detección de Anomalías en Datos</cite></p>
  </section>
</body>
</html>
```

#### 3.2 Predictable

**3.2.1 On Focus (Level A)**

When a component receives focus, it does not initiate a change of context.

**Implementation:**

```html
<!-- Good: No context change on focus -->
<select id="algorithm-select" aria-describedby="algorithm-help">
  <option value="">Select an algorithm</option>
  <option value="isolation-forest">Isolation Forest</option>
  <option value="one-class-svm">One-Class SVM</option>
</select>
<div id="algorithm-help" class="form-help">
  Choose the anomaly detection algorithm for your dataset
</div>

<!-- Context change only on explicit user action -->
<form>
  <label for="filter-type">Filter by type:</label>
  <select id="filter-type" onchange="this.form.submit()">
    <option value="all">All Results</option>
    <option value="anomalies">Anomalies Only</option>
    <option value="normal">Normal Data Only</option>
  </select>
  <div class="form-note">
    Results will update automatically when you change the filter
  </div>
</form>

<script>
// Avoid automatic context changes
class FormManager {
  constructor() {
    this.bindEvents();
  }

  bindEvents() {
    // Use explicit submit instead of onChange
    document.querySelectorAll('.auto-submit-form').forEach(form => {
      const submitBtn = form.querySelector('[type="submit"]');
      const inputs = form.querySelectorAll('input, select, textarea');

      inputs.forEach(input => {
        input.addEventListener('change', () => {
          // Enable submit button when changes are made
          submitBtn.disabled = false;
          submitBtn.textContent = 'Apply Changes';
        });
      });
    });
  }
}
</script>
```

#### 3.3 Input Assistance

**3.3.1 Error Identification (Level A)**

If an input error is automatically detected, the item in error is identified and described to the user in text.

**Implementation:**

```html
<form class="dataset-upload-form" novalidate>
  <div class="form-group">
    <label for="file-input" class="form-label required">
      Dataset File
    </label>
    <input type="file"
           id="file-input"
           name="dataset-file"
           accept=".csv,.json,.xlsx"
           required
           aria-describedby="file-help file-error"
           aria-invalid="false">

    <div id="file-help" class="form-help">
      Upload a CSV, JSON, or Excel file (maximum 100MB)
    </div>

    <div id="file-error"
         class="form-error"
         role="alert"
         style="display: none;"
         aria-live="polite">
      <!-- Error message will be inserted here -->
    </div>
  </div>

  <div class="form-group">
    <label for="email-input" class="form-label required">
      Email for Notifications
    </label>
    <input type="email"
           id="email-input"
           name="email"
           required
           aria-describedby="email-help email-error"
           aria-invalid="false">

    <div id="email-help" class="form-help">
      We'll send you updates when analysis is complete
    </div>

    <div id="email-error"
         class="form-error"
         role="alert"
         style="display: none;"
         aria-live="polite">
      <!-- Error message will be inserted here -->
    </div>
  </div>

  <button type="submit" class="btn btn--primary">
    Upload Dataset
  </button>
</form>

<script>
class FormValidation {
  constructor(form) {
    this.form = form;
    this.bindEvents();
  }

  bindEvents() {
    this.form.addEventListener('submit', this.handleSubmit.bind(this));

    // Real-time validation
    this.form.querySelectorAll('input').forEach(input => {
      input.addEventListener('blur', () => this.validateField(input));
      input.addEventListener('input', () => this.clearErrors(input));
    });
  }

  handleSubmit(event) {
    event.preventDefault();

    const isValid = this.validateForm();
    if (isValid) {
      this.submitForm();
    } else {
      // Focus first invalid field
      const firstError = this.form.querySelector('[aria-invalid="true"]');
      if (firstError) {
        firstError.focus();
      }
    }
  }

  validateField(input) {
    const fieldName = input.name;
    let isValid = true;
    let errorMessage = '';

    // Required field validation
    if (input.required && !input.value.trim()) {
      isValid = false;
      errorMessage = `${this.getFieldLabel(input)} is required.`;
    }

    // Type-specific validation
    if (input.type === 'email' && input.value) {
      const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
      if (!emailRegex.test(input.value)) {
        isValid = false;
        errorMessage = 'Please enter a valid email address.';
      }
    }

    if (input.type === 'file' && input.files.length > 0) {
      const file = input.files[0];
      const maxSize = 100 * 1024 * 1024; // 100MB

      if (file.size > maxSize) {
        isValid = false;
        errorMessage = 'File size must be less than 100MB.';
      }

      const allowedTypes = ['.csv', '.json', '.xlsx'];
      const fileExtension = '.' + file.name.split('.').pop().toLowerCase();

      if (!allowedTypes.includes(fileExtension)) {
        isValid = false;
        errorMessage = 'Please upload a CSV, JSON, or Excel file.';
      }
    }

    this.updateFieldValidation(input, isValid, errorMessage);
    return isValid;
  }

  updateFieldValidation(input, isValid, errorMessage) {
    const errorElement = document.getElementById(input.getAttribute('aria-describedby').split(' ').find(id => id.includes('error')));

    input.setAttribute('aria-invalid', !isValid);

    if (isValid) {
      input.classList.remove('form-input--error');
      errorElement.style.display = 'none';
      errorElement.textContent = '';
    } else {
      input.classList.add('form-input--error');
      errorElement.style.display = 'block';
      errorElement.textContent = errorMessage;
    }
  }

  getFieldLabel(input) {
    const label = this.form.querySelector(`label[for="${input.id}"]`);
    return label ? label.textContent.replace('*', '').trim() : input.name;
  }

  clearErrors(input) {
    if (input.getAttribute('aria-invalid') === 'true') {
      this.validateField(input);
    }
  }

  validateForm() {
    const inputs = this.form.querySelectorAll('input[required]');
    let isFormValid = true;

    inputs.forEach(input => {
      const isFieldValid = this.validateField(input);
      if (!isFieldValid) {
        isFormValid = false;
      }
    });

    return isFormValid;
  }
}

// Initialize form validation
document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('form').forEach(form => {
    new FormValidation(form);
  });
});
</script>
```

### 4. Robust - Content must be robust enough for interpretation by assistive technologies

#### 4.1 Compatible

**4.1.2 Name, Role, Value (Level A)**

For all UI components, the name and role can be programmatically determined.

**Implementation:**

```html
<!-- Custom components with proper ARIA -->
<div class="anomaly-detector"
     role="region"
     aria-labelledby="detector-title"
     aria-describedby="detector-status">

  <h3 id="detector-title">Real-time Anomaly Detector</h3>

  <div id="detector-status" aria-live="polite" aria-atomic="true">
    Monitoring 1,247 data points - 3 anomalies detected
  </div>

  <div class="detector-controls">
    <button type="button"
            class="detector-control"
            aria-pressed="false"
            aria-describedby="start-help"
            data-action="start">
      <span class="control-icon" aria-hidden="true">▶</span>
      <span class="control-text">Start Detection</span>
    </button>
    <div id="start-help" class="sr-only">
      Begin real-time anomaly detection on the current dataset
    </div>

    <button type="button"
            class="detector-control"
            aria-pressed="false"
            aria-describedby="pause-help"
            data-action="pause"
            disabled>
      <span class="control-icon" aria-hidden="true">⏸</span>
      <span class="control-text">Pause Detection</span>
    </button>
    <div id="pause-help" class="sr-only">
      Pause the current detection process
    </div>
  </div>

  <!-- Progress indicator -->
  <div class="detection-progress">
    <div role="progressbar"
         aria-valuenow="0"
         aria-valuemin="0"
         aria-valuemax="100"
         aria-label="Detection progress">
      <div class="progress-bar" style="width: 0%;"></div>
    </div>
    <div class="progress-text" aria-live="polite">
      Ready to start detection
    </div>
  </div>
</div>

<!-- Data table with interactive features -->
<div class="data-table-container">
  <table class="data-table"
         role="table"
         aria-label="Anomaly detection results"
         aria-describedby="table-summary">

    <caption id="table-summary">
      Anomaly detection results showing 150 data points with 12 anomalies detected.
      Use arrow keys to navigate, Enter to select, and Escape to exit selection mode.
    </caption>

    <thead>
      <tr role="row">
        <th scope="col"
            role="columnheader"
            aria-sort="ascending"
            tabindex="0">
          <button type="button" class="sort-button">
            Timestamp
            <span class="sort-indicator" aria-hidden="true">↑</span>
          </button>
        </th>
        <th scope="col" role="columnheader">Value</th>
        <th scope="col" role="columnheader">Anomaly Score</th>
        <th scope="col" role="columnheader">Status</th>
        <th scope="col" role="columnheader">Actions</th>
      </tr>
    </thead>

    <tbody>
      <tr role="row"
          tabindex="0"
          aria-selected="false"
          data-anomaly="true">
        <td role="gridcell">2025-06-26 14:30:00</td>
        <td role="gridcell">145.67</td>
        <td role="gridcell">0.85</td>
        <td role="gridcell">
          <span class="status-badge status-badge--anomaly"
                role="status"
                aria-label="Anomaly detected with high confidence">
            Anomaly
          </span>
        </td>
        <td role="gridcell">
          <button type="button"
                  class="action-button"
                  aria-label="View detailed analysis for data point at 2025-06-26 14:30:00">
            Analyze
          </button>
        </td>
      </tr>
    </tbody>
  </table>
</div>

<script>
// Interactive table with keyboard navigation
class AccessibleDataTable {
  constructor(table) {
    this.table = table;
    this.currentRow = 0;
    this.currentCell = 0;
    this.selectedRows = new Set();

    this.bindEvents();
  }

  bindEvents() {
    this.table.addEventListener('keydown', this.handleKeyDown.bind(this));
    this.table.addEventListener('click', this.handleClick.bind(this));
  }

  handleKeyDown(event) {
    const rows = this.table.querySelectorAll('tbody tr');

    switch (event.key) {
      case 'ArrowDown':
        event.preventDefault();
        this.moveRow(1, rows);
        break;

      case 'ArrowUp':
        event.preventDefault();
        this.moveRow(-1, rows);
        break;

      case 'ArrowRight':
        event.preventDefault();
        this.moveCell(1);
        break;

      case 'ArrowLeft':
        event.preventDefault();
        this.moveCell(-1);
        break;

      case 'Space':
        event.preventDefault();
        this.toggleRowSelection(this.currentRow);
        break;

      case 'Enter':
        event.preventDefault();
        this.activateCurrentCell();
        break;

      case 'Home':
        event.preventDefault();
        this.goToFirstRow();
        break;

      case 'End':
        event.preventDefault();
        this.goToLastRow();
        break;
    }
  }

  moveRow(direction, rows) {
    const newRow = Math.max(0, Math.min(rows.length - 1, this.currentRow + direction));
    if (newRow !== this.currentRow) {
      this.focusRow(newRow);
    }
  }

  focusRow(rowIndex) {
    const rows = this.table.querySelectorAll('tbody tr');

    // Remove focus from current row
    if (rows[this.currentRow]) {
      rows[this.currentRow].classList.remove('table-row--focused');
    }

    // Focus new row
    this.currentRow = rowIndex;
    const newRow = rows[this.currentRow];

    if (newRow) {
      newRow.classList.add('table-row--focused');
      newRow.focus();

      // Announce row content to screen readers
      this.announceRowContent(newRow);
    }
  }

  toggleRowSelection(rowIndex) {
    const rows = this.table.querySelectorAll('tbody tr');
    const row = rows[rowIndex];

    if (this.selectedRows.has(rowIndex)) {
      this.selectedRows.delete(rowIndex);
      row.setAttribute('aria-selected', 'false');
      row.classList.remove('table-row--selected');
    } else {
      this.selectedRows.add(rowIndex);
      row.setAttribute('aria-selected', 'true');
      row.classList.add('table-row--selected');
    }

    this.announceSelection();
  }

  announceRowContent(row) {
    const cells = row.querySelectorAll('td');
    const content = Array.from(cells).map(cell => cell.textContent.trim()).join(', ');
    this.announceToScreenReader(`Row content: ${content}`);
  }

  announceSelection() {
    const count = this.selectedRows.size;
    const message = count === 0
      ? 'No rows selected'
      : `${count} row${count > 1 ? 's' : ''} selected`;

    this.announceToScreenReader(message);
  }

  announceToScreenReader(message) {
    const announcement = document.createElement('div');
    announcement.setAttribute('aria-live', 'polite');
    announcement.setAttribute('aria-atomic', 'true');
    announcement.className = 'sr-only';
    announcement.textContent = message;

    document.body.appendChild(announcement);
    setTimeout(() => document.body.removeChild(announcement), 1000);
  }
}
</script>
```

## Testing Implementation

### Automated Testing Setup

```javascript
// Playwright accessibility test configuration
const { test, expect } = require('@playwright/test');
const { injectAxe, checkA11y } = require('axe-playwright');

test.describe('Accessibility Tests', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/dashboard');
    await injectAxe(page);
  });

  test('should pass WCAG 2.1 AA standards', async ({ page }) => {
    await checkA11y(page, null, {
      detailedReport: true,
      detailedReportOptions: { html: true },
      rules: {
        'color-contrast': { enabled: true },
        'keyboard': { enabled: true },
        'landmark-one-main': { enabled: true },
        'region': { enabled: true },
      }
    });
  });

  test('should have proper heading hierarchy', async ({ page }) => {
    const headings = await page.$$eval('h1, h2, h3, h4, h5, h6', elements =>
      elements.map(el => ({ tag: el.tagName, text: el.textContent }))
    );

    // Verify h1 exists and is unique
    const h1Count = headings.filter(h => h.tag === 'H1').length;
    expect(h1Count).toBe(1);

    // Verify logical heading sequence
    let currentLevel = 0;
    for (const heading of headings) {
      const level = parseInt(heading.tag.charAt(1));
      expect(level).toBeLessThanOrEqual(currentLevel + 1);
      currentLevel = level;
    }
  });
});
```

This comprehensive accessibility implementation guide ensures that the Pynomaly platform meets and exceeds WCAG 2.1 AA standards while providing an exceptional user experience for all users, including those using assistive technologies.
