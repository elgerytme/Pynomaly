# Accessibility Guidelines & WCAG Compliance

## 🌟 **Overview**

This comprehensive guide provides accessibility implementation standards, WCAG 2.1 AA compliance procedures, and testing methodologies for the Pynomaly platform. Our commitment to accessibility ensures that all users, regardless of abilities or disabilities, can effectively use our anomaly detection platform.

## 📋 **Table of Contents**

- [🎯 Accessibility Strategy](#-accessibility-strategy)
- [📏 WCAG 2.1 AA Compliance](#-wcag-21-aa-compliance)
- [🔍 Testing Procedures](#-testing-procedures)
- [💻 Implementation Standards](#-implementation-standards)
- [🎨 Design Guidelines](#-design-guidelines)
- [⌨️ Keyboard Navigation](#️-keyboard-navigation)
- [🗣️ Screen Reader Support](#️-screen-reader-support)
- [🎨 Color & Contrast](#-color--contrast)
- [📱 Mobile Accessibility](#-mobile-accessibility)
- [🛠️ Development Tools](#️-development-tools)
- [📊 Accessibility Testing](#-accessibility-testing)
- [🚨 Common Issues & Solutions](#-common-issues--solutions)

## 🎯 **Accessibility Strategy**

### Core Principles

Our accessibility strategy follows the **POUR** principles:

1. **Perceivable**: Information must be presentable to users in ways they can perceive
2. **Operable**: Interface components must be operable by all users
3. **Understandable**: Information and UI operation must be understandable
4. **Robust**: Content must be robust enough for various user agents and assistive technologies

### Accessibility Goals

| **Level** | **Target** | **Status** | **Requirements** |
|-----------|------------|------------|------------------|
| **WCAG 2.1 A** | 100% | ✅ Complete | Basic accessibility requirements |
| **WCAG 2.1 AA** | 100% | 🎯 Target | Industry standard compliance |
| **WCAG 2.1 AAA** | 80% | 📈 Progressive | Enhanced accessibility features |

### User Groups

Our platform serves diverse users with various accessibility needs:

- **Vision Impairments**: Blindness, low vision, color blindness
- **Hearing Impairments**: Deafness, hard of hearing
- **Motor Impairments**: Limited fine motor control, paralysis
- **Cognitive Disabilities**: Dyslexia, attention disorders, memory issues
- **Temporary Disabilities**: Broken arm, eye strain, environmental constraints

## 📏 **WCAG 2.1 AA Compliance**

### Level A Requirements (Mandatory)

#### 1.1 Text Alternatives
```html
<!-- Images with meaningful content -->
<img src="anomaly-chart.png" 
     alt="Time series chart showing 3 anomalies detected between 2:00-4:00 PM">

<!-- Decorative images -->
<img src="decorative-line.svg" alt="" role="presentation">

<!-- Complex charts -->
<div role="img" aria-labelledby="chart-title" aria-describedby="chart-desc">
  <h3 id="chart-title">Anomaly Detection Results</h3>
  <p id="chart-desc">
    Chart displays 100 data points with 3 anomalies detected at timestamps 
    14:23, 15:45, and 16:12. Average confidence score: 0.87.
  </p>
  <!-- Chart visualization -->
</div>
```

#### 1.2 Time-Based Media
```html
<!-- Video with captions and transcripts -->
<video controls>
  <source src="tutorial.mp4" type="video/mp4">
  <track kind="captions" src="captions.vtt" srclang="en" label="English">
  <track kind="descriptions" src="descriptions.vtt" srclang="en" label="English Descriptions">
</video>

<!-- Audio alternatives -->
<audio controls aria-describedby="audio-transcript">
  <source src="detection-alert.mp3" type="audio/mpeg">
  <p id="audio-transcript">
    Audio alert: "Anomaly detected in dataset 'sales-data' with confidence 0.94"
  </p>
</audio>
```

#### 1.3 Adaptable Content
```html
<!-- Logical heading structure -->
<h1>Anomaly Detection Dashboard</h1>
  <h2>Recent Detections</h2>
    <h3>High Priority Anomalies</h3>
    <h3>Medium Priority Anomalies</h3>
  <h2>Model Performance</h2>
    <h3>Accuracy Metrics</h3>

<!-- Data tables with proper headers -->
<table>
  <caption>Detection Results Summary</caption>
  <thead>
    <tr>
      <th scope="col">Timestamp</th>
      <th scope="col">Dataset</th>
      <th scope="col">Anomaly Type</th>
      <th scope="col">Confidence</th>
      <th scope="col">Actions</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th scope="row">2025-06-26 14:23:15</th>
      <td>sales-data</td>
      <td>Statistical Outlier</td>
      <td>0.94</td>
      <td>
        <button aria-label="View details for anomaly at 14:23:15">
          View Details
        </button>
      </td>
    </tr>
  </tbody>
</table>
```

#### 1.4 Distinguishable Content
```css
/* High contrast color scheme */
:root {
  --contrast-ratio-normal: 4.5; /* AA standard */
  --contrast-ratio-large: 3.0;  /* AA large text */
  
  /* Primary colors with sufficient contrast */
  --primary-color: #0ea5e9;     /* Contrast: 4.52:1 on white */
  --primary-dark: #0284c7;      /* Contrast: 5.74:1 on white */
  --text-primary: #1e293b;      /* Contrast: 13.15:1 on white */
  --text-secondary: #475569;    /* Contrast: 7.07:1 on white */
  
  /* Error colors */
  --error-color: #dc2626;       /* Contrast: 5.93:1 on white */
  --error-bg: #fef2f2;         /* Contrast: 1.04:1 on white */
  
  /* Success colors */
  --success-color: #059669;     /* Contrast: 4.52:1 on white */
  --success-bg: #f0fdf4;       /* Contrast: 1.02:1 on white */
}

/* Focus indicators */
.focus-visible {
  outline: 2px solid var(--primary-color);
  outline-offset: 2px;
  border-radius: 4px;
}

/* Text sizing for readability */
.text-base {
  font-size: 1rem;           /* 16px */
  line-height: 1.5;          /* 24px */
}

.text-large {
  font-size: 1.125rem;       /* 18px */
  line-height: 1.556;        /* 28px */
}
```

#### 2.1 Keyboard Accessible
```html
<!-- Skip navigation -->
<a href="#main-content" class="skip-link">Skip to main content</a>

<!-- Proper tab order -->
<nav aria-label="Main navigation" tabindex="0">
  <ul>
    <li><a href="/dashboard" tabindex="1">Dashboard</a></li>
    <li><a href="/datasets" tabindex="2">Datasets</a></li>
    <li><a href="/models" tabindex="3">Models</a></li>
  </ul>
</nav>

<main id="main-content" tabindex="-1">
  <!-- Main content -->
</main>

<!-- Custom interactive components -->
<div role="button" 
     tabindex="0" 
     aria-pressed="false"
     onkeydown="handleKeyDown(event)"
     onclick="toggleButton()">
  Toggle Detection Mode
</div>

<script>
function handleKeyDown(event) {
  // Handle Enter and Space keys
  if (event.key === 'Enter' || event.key === ' ') {
    event.preventDefault();
    toggleButton();
  }
}
</script>
```

### Level AA Requirements (Target Standard)

#### 1.4.3 Contrast (Minimum)
- **Normal text**: 4.5:1 contrast ratio
- **Large text (18pt+ or 14pt+ bold)**: 3:1 contrast ratio
- **UI components and graphics**: 3:1 contrast ratio

#### 1.4.4 Resize Text
```css
/* Responsive typography that scales well */
html {
  font-size: 100%; /* 16px base */
}

/* Support up to 200% zoom without horizontal scrolling */
@media (max-width: 1200px) {
  .container {
    max-width: 100%;
    padding: 1rem;
  }
  
  .text-responsive {
    font-size: clamp(0.875rem, 2.5vw, 1.125rem);
  }
}

/* Ensure interactive elements maintain size */
.btn {
  min-height: 44px;  /* Touch target size */
  min-width: 44px;
  padding: 0.75rem 1rem;
}
```

#### 1.4.5 Images of Text
```html
<!-- Avoid images of text, use CSS typography instead -->
<h1 class="heading-hero">
  Anomaly Detection Platform
</h1>

<!-- If images of text are necessary, provide alternatives -->
<img src="logo-text.png" 
     alt="Pynomaly - Advanced Anomaly Detection">
```

#### 2.4.6 Headings and Labels
```html
<!-- Descriptive headings -->
<h2>Real-Time Anomaly Detection Results</h2>

<!-- Clear form labels -->
<label for="dataset-upload">
  Choose Dataset File (CSV, JSON, or Parquet)
</label>
<input type="file" 
       id="dataset-upload" 
       accept=".csv,.json,.parquet"
       aria-describedby="upload-help">
<div id="upload-help">
  Maximum file size: 100MB. Supported formats: CSV, JSON, Parquet.
</div>
```

#### 2.4.7 Focus Visible
```css
/* Enhanced focus indicators */
.focus-visible,
*:focus-visible {
  outline: 2px solid var(--focus-color, #0ea5e9);
  outline-offset: 2px;
  border-radius: 4px;
  box-shadow: 0 0 0 4px rgba(14, 165, 233, 0.2);
}

/* Different focus styles for different element types */
button:focus-visible {
  outline-color: var(--button-focus-color, #0284c7);
}

input:focus-visible,
textarea:focus-visible,
select:focus-visible {
  outline-color: var(--input-focus-color, #059669);
  border-color: var(--input-focus-border, #10b981);
}

a:focus-visible {
  outline-color: var(--link-focus-color, #7c3aed);
}
```

#### 3.2.3 Consistent Navigation
```html
<!-- Consistent navigation structure across pages -->
<nav aria-label="Main navigation" class="main-nav">
  <ul>
    <li><a href="/dashboard" aria-current="page">Dashboard</a></li>
    <li><a href="/datasets">Datasets</a></li>
    <li><a href="/models">Models</a></li>
    <li><a href="/settings">Settings</a></li>
  </ul>
</nav>

<!-- Consistent secondary navigation -->
<nav aria-label="Dashboard sections" class="section-nav">
  <ul>
    <li><a href="#overview">Overview</a></li>
    <li><a href="#detections">Recent Detections</a></li>
    <li><a href="#performance">Performance</a></li>
  </ul>
</nav>
```

## 🔍 **Testing Procedures**

### Automated Testing Tools

#### 1. Playwright Accessibility Testing
```typescript
// tests/accessibility/axe-accessibility.spec.ts
import { test, expect } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';

test.describe('Accessibility Testing', () => {
  test('Dashboard page meets WCAG 2.1 AA standards', async ({ page }) => {
    await page.goto('/dashboard');
    
    const accessibilityScanResults = await new AxeBuilder({ page })
      .withTags(['wcag2a', 'wcag2aa', 'wcag21aa'])
      .analyze();
    
    expect(accessibilityScanResults.violations).toEqual([]);
  });
  
  test('Form interactions are accessible', async ({ page }) => {
    await page.goto('/datasets/upload');
    
    // Test keyboard navigation
    await page.keyboard.press('Tab');
    await expect(page.locator('#dataset-upload')).toBeFocused();
    
    await page.keyboard.press('Tab');
    await expect(page.locator('#upload-button')).toBeFocused();
    
    // Test screen reader announcements
    await page.locator('#upload-button').click();
    await expect(page.locator('[aria-live="polite"]')).toContainText('Upload started');
  });
  
  test('Data visualizations have proper alternatives', async ({ page }) => {
    await page.goto('/dashboard');
    
    // Check chart accessibility
    const chart = page.locator('[role="img"]');
    await expect(chart).toHaveAttribute('aria-labelledby');
    await expect(chart).toHaveAttribute('aria-describedby');
    
    // Verify data table alternative
    const tableButton = page.locator('button:has-text("View as Table")');
    await expect(tableButton).toBeVisible();
  });
});
```

#### 2. Color Contrast Validation
```typescript
// tests/accessibility/color-contrast.spec.ts
import { test, expect } from '@playwright/test';

test('Color contrast meets WCAG standards', async ({ page }) => {
  await page.goto('/dashboard');
  
  // Test primary text contrast
  const textElement = page.locator('h1').first();
  const computedStyle = await textElement.evaluate((el) => {
    const style = window.getComputedStyle(el);
    return {
      color: style.color,
      backgroundColor: style.backgroundColor
    };
  });
  
  // Calculate contrast ratio (implementation depends on contrast library)
  const contrastRatio = calculateContrastRatio(
    computedStyle.color,
    computedStyle.backgroundColor
  );
  
  expect(contrastRatio).toBeGreaterThanOrEqual(4.5);
});
```

### Manual Testing Procedures

#### 1. Keyboard Navigation Testing
```markdown
### Keyboard Navigation Checklist

**Navigation Tests:**
- [ ] Tab order is logical and sequential
- [ ] All interactive elements are reachable via keyboard
- [ ] Skip links are provided and functional
- [ ] Focus indicators are clearly visible
- [ ] Trapped focus works correctly in modals/dialogs
- [ ] Escape key closes modals and dropdowns

**Interaction Tests:**
- [ ] Enter key activates buttons and links
- [ ] Space key activates buttons and toggles checkboxes
- [ ] Arrow keys navigate within component groups
- [ ] Home/End keys jump to first/last items in lists
- [ ] Page Up/Down scroll content appropriately

**Form Tests:**
- [ ] Tab moves between form fields
- [ ] Shift+Tab moves backwards
- [ ] Enter submits forms (when appropriate)
- [ ] Error messages are announced
- [ ] Required field indicators are accessible
```

#### 2. Screen Reader Testing
```markdown
### Screen Reader Testing Checklist

**Content Structure:**
- [ ] Headings create logical page outline
- [ ] Lists are properly marked up
- [ ] Tables have appropriate headers
- [ ] Form labels are associated correctly
- [ ] Error messages are announced

**Navigation:**
- [ ] Landmarks (nav, main, aside) are identified
- [ ] Skip links function correctly
- [ ] Page title describes content
- [ ] Breadcrumbs are announced properly

**Interactive Elements:**
- [ ] Button purposes are clear
- [ ] Link destinations are described
- [ ] Form field requirements are announced
- [ ] Status updates use live regions
- [ ] Progress indicators are announced
```

### Testing Tools Configuration

#### axe-core Configuration
```javascript
// .axerc.json
{
  "locale": "en",
  "tags": ["wcag2a", "wcag2aa", "wcag21aa"],
  "rules": {
    "color-contrast": {
      "enabled": true,
      "options": {
        "noScroll": false
      }
    },
    "focus-order-semantics": {
      "enabled": true
    },
    "keyboard-navigation": {
      "enabled": true
    }
  },
  "reporter": "v2",
  "resultTypes": ["violations", "incomplete"]
}
```

#### Lighthouse Accessibility Configuration
```javascript
// lighthouse-a11y.config.js
module.exports = {
  extends: 'lighthouse:default',
  settings: {
    onlyCategories: ['accessibility'],
    skipAudits: ['uses-http2'],
  },
  audits: [
    'accessibility/aria-allowed-attr',
    'accessibility/aria-required-attr',
    'accessibility/aria-required-children',
    'accessibility/aria-required-parent',
    'accessibility/aria-roles',
    'accessibility/aria-valid-attr',
    'accessibility/aria-valid-attr-value',
    'accessibility/color-contrast',
    'accessibility/document-title',
    'accessibility/duplicate-id',
    'accessibility/frame-title',
    'accessibility/html-has-lang',
    'accessibility/image-alt',
    'accessibility/input-image-alt',
    'accessibility/label',
    'accessibility/link-name',
    'accessibility/list',
    'accessibility/listitem',
    'accessibility/meta-refresh',
    'accessibility/meta-viewport',
    'accessibility/object-alt',
    'accessibility/tabindex',
    'accessibility/td-headers-attr',
    'accessibility/th-has-data-cells',
    'accessibility/valid-lang'
  ]
};
```

## 💻 **Implementation Standards**

### Semantic HTML
```html
<!-- Use semantic elements for structure -->
<header>
  <nav aria-label="Main navigation">
    <!-- Navigation content -->
  </nav>
</header>

<main>
  <section aria-labelledby="dashboard-heading">
    <h1 id="dashboard-heading">Anomaly Detection Dashboard</h1>
    
    <article aria-labelledby="recent-detections">
      <h2 id="recent-detections">Recent Detections</h2>
      <!-- Detection content -->
    </article>
    
    <aside aria-labelledby="model-info">
      <h2 id="model-info">Model Information</h2>
      <!-- Model details -->
    </aside>
  </section>
</main>

<footer>
  <!-- Footer content -->
</footer>
```

### ARIA Implementation
```html
<!-- Complex UI components -->
<div role="tablist" aria-label="Detection Analysis Tabs">
  <button role="tab" 
          aria-selected="true" 
          aria-controls="overview-panel" 
          id="overview-tab">
    Overview
  </button>
  <button role="tab" 
          aria-selected="false" 
          aria-controls="details-panel" 
          id="details-tab">
    Details
  </button>
</div>

<div role="tabpanel" 
     id="overview-panel" 
     aria-labelledby="overview-tab">
  <!-- Overview content -->
</div>

<!-- Live regions for dynamic content -->
<div aria-live="polite" aria-atomic="true" id="status-updates">
  <!-- Status messages appear here -->
</div>

<div aria-live="assertive" id="error-announcements">
  <!-- Critical error messages -->
</div>

<!-- Modal dialogs -->
<div role="dialog" 
     aria-labelledby="modal-title" 
     aria-describedby="modal-description"
     aria-modal="true">
  <h2 id="modal-title">Confirm Deletion</h2>
  <p id="modal-description">
    Are you sure you want to delete this dataset? This action cannot be undone.
  </p>
  
  <button type="button" onclick="confirmDelete()">Delete</button>
  <button type="button" onclick="closeModal()">Cancel</button>
</div>
```

### Form Accessibility
```html
<!-- Comprehensive form example -->
<form id="dataset-upload-form" novalidate>
  <fieldset>
    <legend>Dataset Upload Configuration</legend>
    
    <div class="form-group">
      <label for="dataset-name" class="required">
        Dataset Name
        <span class="required-indicator" aria-label="required">*</span>
      </label>
      <input type="text" 
             id="dataset-name" 
             required 
             aria-describedby="name-help name-error"
             autocomplete="off">
      <div id="name-help" class="help-text">
        Enter a descriptive name for your dataset
      </div>
      <div id="name-error" class="error-message" aria-live="polite">
        <!-- Error messages appear here -->
      </div>
    </div>
    
    <div class="form-group">
      <label for="file-upload">
        Choose File
      </label>
      <input type="file" 
             id="file-upload" 
             accept=".csv,.json,.parquet"
             aria-describedby="file-help">
      <div id="file-help" class="help-text">
        Supported formats: CSV, JSON, Parquet. Maximum size: 100MB.
      </div>
    </div>
    
    <fieldset>
      <legend>Detection Parameters</legend>
      
      <div class="form-group">
        <label for="contamination-rate">
          Contamination Rate
        </label>
        <input type="range" 
               id="contamination-rate" 
               min="0.01" 
               max="0.5" 
               step="0.01" 
               value="0.1"
               aria-describedby="contamination-help"
               aria-valuetext="10 percent">
        <output for="contamination-rate" aria-live="polite">0.1 (10%)</output>
        <div id="contamination-help" class="help-text">
          Expected proportion of anomalies in the dataset
        </div>
      </div>
    </fieldset>
  </fieldset>
  
  <div class="form-actions">
    <button type="submit" class="btn-primary">
      Upload and Analyze
    </button>
    <button type="reset" class="btn-secondary">
      Reset Form
    </button>
  </div>
</form>
```

## 🎨 **Design Guidelines**

### Color and Contrast
```css
/* Color palette with accessibility considerations */
:root {
  /* Primary colors - ensure 4.5:1 contrast on white */
  --blue-600: #2563eb;    /* 5.74:1 contrast */
  --blue-700: #1d4ed8;    /* 7.04:1 contrast */
  --blue-800: #1e40af;    /* 8.59:1 contrast */
  
  /* Success colors */
  --green-600: #059669;   /* 4.52:1 contrast */
  --green-700: #047857;   /* 5.85:1 contrast */
  
  /* Warning colors */
  --amber-600: #d97706;   /* 3.94:1 contrast - use with caution */
  --amber-700: #b45309;   /* 5.08:1 contrast */
  
  /* Error colors */
  --red-600: #dc2626;     /* 5.93:1 contrast */
  --red-700: #b91c1c;     /* 7.22:1 contrast */
  
  /* Neutral colors */
  --gray-600: #4b5563;    /* 6.87:1 contrast */
  --gray-700: #374151;    /* 9.26:1 contrast */
  --gray-800: #1f2937;    /* 12.63:1 contrast */
  --gray-900: #111827;    /* 15.56:1 contrast */
}

/* High contrast mode support */
@media (prefers-contrast: high) {
  :root {
    --primary-color: var(--blue-800);
    --text-color: var(--gray-900);
    --border-color: var(--gray-800);
  }
  
  .btn {
    border-width: 2px;
  }
  
  .focus-visible {
    outline-width: 3px;
    outline-offset: 3px;
  }
}
```

### Typography
```css
/* Accessible typography scale */
:root {
  --font-family-base: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  --font-family-mono: "JetBrains Mono", Consolas, "Liberation Mono", Menlo, Courier, monospace;
  
  /* Type scale with good line height ratios */
  --text-xs: 0.75rem;     /* 12px */
  --text-sm: 0.875rem;    /* 14px */
  --text-base: 1rem;      /* 16px */
  --text-lg: 1.125rem;    /* 18px */
  --text-xl: 1.25rem;     /* 20px */
  --text-2xl: 1.5rem;     /* 24px */
  --text-3xl: 1.875rem;   /* 30px */
  --text-4xl: 2.25rem;    /* 36px */
  
  /* Line heights for readability */
  --leading-tight: 1.25;
  --leading-normal: 1.5;
  --leading-relaxed: 1.625;
  --leading-loose: 2;
}

/* Responsive typography */
.text-responsive {
  font-size: clamp(var(--text-sm), 2.5vw, var(--text-lg));
  line-height: var(--leading-normal);
}

/* Reading optimized text */
.text-content {
  font-size: var(--text-lg);
  line-height: var(--leading-relaxed);
  max-width: 65ch; /* Optimal reading width */
  margin: 0 auto;
}

/* Support for user font size preferences */
@media (prefers-reduced-motion: no-preference) {
  html {
    scroll-behavior: smooth;
  }
}
```

### Interactive Elements
```css
/* Accessible button designs */
.btn {
  min-height: 44px;  /* Touch target size */
  min-width: 44px;
  padding: 0.75rem 1rem;
  font-size: var(--text-base);
  font-weight: 500;
  border-radius: 0.375rem;
  border: 2px solid transparent;
  cursor: pointer;
  transition: all 0.2s ease;
  position: relative;
}

.btn:focus-visible {
  outline: 2px solid var(--focus-color);
  outline-offset: 2px;
}

.btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

/* Loading states */
.btn[aria-busy="true"] {
  color: transparent;
}

.btn[aria-busy="true"]::after {
  content: "";
  position: absolute;
  top: 50%;
  left: 50%;
  width: 1rem;
  height: 1rem;
  margin: -0.5rem 0 0 -0.5rem;
  border: 2px solid currentColor;
  border-radius: 50%;
  border-top-color: transparent;
  animation: spin 1s linear infinite;
}

/* Form controls */
.form-input {
  min-height: 44px;
  padding: 0.75rem;
  font-size: var(--text-base);
  border: 2px solid var(--border-color);
  border-radius: 0.375rem;
  background-color: var(--input-bg);
  transition: border-color 0.2s ease, box-shadow 0.2s ease;
}

.form-input:focus {
  border-color: var(--primary-color);
  box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
  outline: none;
}

.form-input[aria-invalid="true"] {
  border-color: var(--error-color);
}
```

## ⌨️ **Keyboard Navigation**

### Navigation Patterns
```typescript
// Enhanced keyboard navigation handler
class KeyboardNavigationManager {
  private focusableElements: HTMLElement[] = [];
  private currentIndex: number = 0;
  
  constructor(container: HTMLElement) {
    this.updateFocusableElements(container);
    this.setupEventListeners();
  }
  
  private updateFocusableElements(container: HTMLElement): void {
    const selectors = [
      'a[href]',
      'button:not([disabled])',
      'input:not([disabled])',
      'select:not([disabled])',
      'textarea:not([disabled])',
      '[tabindex]:not([tabindex="-1"])',
      '[role="button"]:not([aria-disabled="true"])',
      '[role="link"]:not([aria-disabled="true"])'
    ].join(', ');
    
    this.focusableElements = Array.from(
      container.querySelectorAll(selectors)
    ) as HTMLElement[];
  }
  
  private setupEventListeners(): void {
    document.addEventListener('keydown', (event) => {
      switch (event.key) {
        case 'Tab':
          this.handleTabNavigation(event);
          break;
        case 'ArrowDown':
        case 'ArrowUp':
          this.handleArrowNavigation(event);
          break;
        case 'Home':
        case 'End':
          this.handleHomeEndNavigation(event);
          break;
        case 'Escape':
          this.handleEscapeKey(event);
          break;
      }
    });
  }
  
  private handleTabNavigation(event: KeyboardEvent): void {
    const activeElement = document.activeElement as HTMLElement;
    const currentIndex = this.focusableElements.indexOf(activeElement);
    
    if (event.shiftKey) {
      // Previous element
      const prevIndex = currentIndex > 0 ? currentIndex - 1 : this.focusableElements.length - 1;
      this.focusableElements[prevIndex]?.focus();
    } else {
      // Next element
      const nextIndex = currentIndex < this.focusableElements.length - 1 ? currentIndex + 1 : 0;
      this.focusableElements[nextIndex]?.focus();
    }
  }
  
  private announceNavigation(element: HTMLElement): void {
    const announcement = this.getElementAnnouncement(element);
    this.announceToScreenReader(announcement);
  }
  
  private announceToScreenReader(message: string): void {
    const announcer = document.getElementById('screen-reader-announcer');
    if (announcer) {
      announcer.textContent = message;
    }
  }
}
```

### Modal and Dialog Navigation
```typescript
// Focus trap for modal dialogs
class FocusTrap {
  private container: HTMLElement;
  private firstFocusableElement: HTMLElement | null = null;
  private lastFocusableElement: HTMLElement | null = null;
  private previousActiveElement: HTMLElement | null = null;
  
  constructor(container: HTMLElement) {
    this.container = container;
    this.setupFocusTrap();
  }
  
  activate(): void {
    this.previousActiveElement = document.activeElement as HTMLElement;
    this.updateFocusableElements();
    this.firstFocusableElement?.focus();
    
    document.addEventListener('keydown', this.handleKeyDown);
  }
  
  deactivate(): void {
    document.removeEventListener('keydown', this.handleKeyDown);
    this.previousActiveElement?.focus();
  }
  
  private handleKeyDown = (event: KeyboardEvent): void => {
    if (event.key === 'Tab') {
      if (event.shiftKey) {
        if (document.activeElement === this.firstFocusableElement) {
          event.preventDefault();
          this.lastFocusableElement?.focus();
        }
      } else {
        if (document.activeElement === this.lastFocusableElement) {
          event.preventDefault();
          this.firstFocusableElement?.focus();
        }
      }
    } else if (event.key === 'Escape') {
      event.preventDefault();
      this.deactivate();
      // Close modal
      this.container.dispatchEvent(new CustomEvent('modal-close'));
    }
  };
}
```

## 🗣️ **Screen Reader Support**

### ARIA Live Regions
```html
<!-- Status announcements -->
<div id="status-announcer" 
     aria-live="polite" 
     aria-atomic="true" 
     class="sr-only">
  <!-- Non-critical status updates -->
</div>

<div id="alert-announcer" 
     aria-live="assertive" 
     aria-atomic="true" 
     class="sr-only">
  <!-- Critical alerts and errors -->
</div>

<!-- Progress announcements -->
<div id="progress-announcer" 
     aria-live="polite" 
     aria-atomic="false" 
     class="sr-only">
  <!-- Progress updates -->
</div>
```

### JavaScript Announcements
```typescript
// Screen reader announcement utilities
class ScreenReaderAnnouncer {
  private statusElement: HTMLElement;
  private alertElement: HTMLElement;
  private progressElement: HTMLElement;
  
  constructor() {
    this.statusElement = document.getElementById('status-announcer')!;
    this.alertElement = document.getElementById('alert-announcer')!;
    this.progressElement = document.getElementById('progress-announcer')!;
  }
  
  announceStatus(message: string, priority: 'polite' | 'assertive' = 'polite'): void {
    const element = priority === 'assertive' ? this.alertElement : this.statusElement;
    
    // Clear previous message
    element.textContent = '';
    
    // Add new message after a brief delay
    setTimeout(() => {
      element.textContent = message;
    }, 100);
  }
  
  announceProgress(current: number, total: number, task: string): void {
    const percentage = Math.round((current / total) * 100);
    const message = `${task}: ${percentage}% complete. ${current} of ${total} items processed.`;
    
    this.progressElement.textContent = message;
  }
  
  announceDetectionResult(result: DetectionResult): void {
    const message = `Detection complete. Found ${result.anomalyCount} anomalies ` +
                   `with average confidence ${result.averageConfidence.toFixed(2)}.`;
    
    this.announceStatus(message, 'polite');
  }
  
  announceError(error: string): void {
    const message = `Error: ${error}. Please try again or contact support.`;
    this.announceStatus(message, 'assertive');
  }
}

// Usage examples
const announcer = new ScreenReaderAnnouncer();

// File upload progress
announcer.announceProgress(50, 100, 'File upload');

// Detection completion
announcer.announceDetectionResult({
  anomalyCount: 3,
  averageConfidence: 0.87
});

// Error handling
announcer.announceError('Failed to load dataset. Please check file format.');
```

### Chart and Visualization Descriptions
```html
<!-- Complex data visualization with full accessibility -->
<div class="chart-container">
  <div role="img" 
       aria-labelledby="chart-title" 
       aria-describedby="chart-description chart-data-table">
    
    <h3 id="chart-title">Anomaly Detection Timeline</h3>
    
    <div id="chart-description">
      Time series chart showing anomaly detection results over the past 24 hours.
      X-axis represents time from 00:00 to 23:59. Y-axis represents data values
      ranging from 0 to 100. Red markers indicate detected anomalies.
    </div>
    
    <!-- Visual chart (D3.js or ECharts) -->
    <div id="visual-chart" aria-hidden="true">
      <!-- Chart visualization -->
    </div>
    
    <!-- Alternative data representation -->
    <details>
      <summary>View chart data as table</summary>
      <table id="chart-data-table" class="data-table">
        <caption>Anomaly Detection Data</caption>
        <thead>
          <tr>
            <th scope="col">Time</th>
            <th scope="col">Value</th>
            <th scope="col">Status</th>
            <th scope="col">Confidence</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <th scope="row">14:23:15</th>
            <td>87.3</td>
            <td>Anomaly</td>
            <td>0.94</td>
          </tr>
          <!-- More data rows -->
        </tbody>
      </table>
    </details>
  </div>
</div>
```

## 📱 **Mobile Accessibility**

### Touch Target Guidelines
```css
/* Ensure adequate touch target sizes */
.touch-target {
  min-height: 44px;  /* iOS recommendation */
  min-width: 44px;
  margin: 2px;       /* Prevent accidental activation */
}

/* Larger targets for complex actions */
.primary-action {
  min-height: 48px;
  min-width: 88px;
}

/* Spacing between touch targets */
.button-group .btn {
  margin: 0 4px 8px 0;
}

.button-group .btn:last-child {
  margin-right: 0;
}
```

### Mobile Navigation
```html
<!-- Mobile-optimized navigation -->
<nav class="mobile-nav" aria-label="Main navigation">
  <button class="nav-toggle" 
          aria-expanded="false" 
          aria-controls="nav-menu"
          aria-label="Toggle navigation menu">
    <span class="hamburger"></span>
  </button>
  
  <ul id="nav-menu" 
      class="nav-menu" 
      aria-hidden="true">
    <li><a href="/dashboard">Dashboard</a></li>
    <li><a href="/datasets">Datasets</a></li>
    <li><a href="/models">Models</a></li>
    <li><a href="/settings">Settings</a></li>
  </ul>
</nav>

<script>
// Mobile navigation enhancement
class MobileNavigation {
  constructor() {
    this.toggle = document.querySelector('.nav-toggle');
    this.menu = document.querySelector('.nav-menu');
    this.setupEventListeners();
  }
  
  setupEventListeners() {
    this.toggle.addEventListener('click', () => this.toggleMenu());
    
    // Close menu when clicking outside
    document.addEventListener('click', (event) => {
      if (!this.toggle.contains(event.target) && 
          !this.menu.contains(event.target)) {
        this.closeMenu();
      }
    });
    
    // Handle escape key
    document.addEventListener('keydown', (event) => {
      if (event.key === 'Escape' && this.isMenuOpen()) {
        this.closeMenu();
        this.toggle.focus();
      }
    });
  }
  
  toggleMenu() {
    if (this.isMenuOpen()) {
      this.closeMenu();
    } else {
      this.openMenu();
    }
  }
  
  openMenu() {
    this.toggle.setAttribute('aria-expanded', 'true');
    this.menu.setAttribute('aria-hidden', 'false');
    this.menu.classList.add('open');
    
    // Focus first menu item
    const firstItem = this.menu.querySelector('a');
    firstItem?.focus();
  }
  
  closeMenu() {
    this.toggle.setAttribute('aria-expanded', 'false');
    this.menu.setAttribute('aria-hidden', 'true');
    this.menu.classList.remove('open');
  }
  
  isMenuOpen() {
    return this.toggle.getAttribute('aria-expanded') === 'true';
  }
}
</script>
```

## 🛠️ **Development Tools**

### VS Code Extensions
```json
// .vscode/extensions.json
{
  "recommendations": [
    "deque-systems.vscode-axe-linter",
    "streetsidesoftware.code-spell-checker",
    "ms-vscode.vscode-typescript-next",
    "bradlc.vscode-tailwindcss",
    "webhint.vscode-webhint"
  ]
}
```

### Accessibility Linting
```json
// .eslintrc.json
{
  "extends": [
    "plugin:jsx-a11y/recommended"
  ],
  "plugins": [
    "jsx-a11y"
  ],
  "rules": {
    "jsx-a11y/alt-text": "error",
    "jsx-a11y/aria-props": "error",
    "jsx-a11y/aria-proptypes": "error",
    "jsx-a11y/aria-unsupported-elements": "error",
    "jsx-a11y/role-has-required-aria-props": "error",
    "jsx-a11y/role-supports-aria-props": "error",
    "jsx-a11y/img-redundant-alt": "error",
    "jsx-a11y/label-has-associated-control": "error",
    "jsx-a11y/no-autofocus": "error",
    "jsx-a11y/click-events-have-key-events": "error",
    "jsx-a11y/interactive-supports-focus": "error"
  }
}
```

### Build Integration
```javascript
// scripts/accessibility-check.js
const { execSync } = require('child_process');
const fs = require('fs');

async function runAccessibilityChecks() {
  console.log('Running accessibility checks...');
  
  try {
    // Run axe-core tests
    console.log('🔍 Running axe-core accessibility tests...');
    execSync('npm run test:a11y', { stdio: 'inherit' });
    
    // Run Lighthouse accessibility audit
    console.log('🚨 Running Lighthouse accessibility audit...');
    execSync('npm run lighthouse:a11y', { stdio: 'inherit' });
    
    // Check color contrast
    console.log('🎨 Checking color contrast ratios...');
    execSync('npm run test:contrast', { stdio: 'inherit' });
    
    console.log('✅ All accessibility checks passed!');
    
  } catch (error) {
    console.error('❌ Accessibility checks failed!');
    process.exit(1);
  }
}

// Generate accessibility report
function generateAccessibilityReport() {
  const reportData = {
    timestamp: new Date().toISOString(),
    wcagLevel: 'AA',
    testResults: {
      automated: {
        axe: 'passed',
        lighthouse: 'passed',
        colorContrast: 'passed'
      },
      manual: {
        keyboardNavigation: 'pending',
        screenReader: 'pending'
      }
    },
    coverage: {
      pages: 15,
      components: 45,
      criticalPaths: 8
    }
  };
  
  fs.writeFileSync(
    'reports/accessibility-report.json',
    JSON.stringify(reportData, null, 2)
  );
  
  console.log('📊 Accessibility report generated: reports/accessibility-report.json');
}

if (require.main === module) {
  runAccessibilityChecks()
    .then(() => generateAccessibilityReport())
    .catch(console.error);
}

module.exports = { runAccessibilityChecks, generateAccessibilityReport };
```

## 🚨 **Common Issues & Solutions**

### Issue 1: Missing Focus Indicators
```css
/* Problem: Default browser focus styles removed */
button:focus { outline: none; } /* ❌ Don't do this */

/* Solution: Provide custom focus indicators */
button:focus-visible {
  outline: 2px solid var(--focus-color);
  outline-offset: 2px;
  border-radius: 4px;
}

/* Enhanced focus for better visibility */
.focus-enhanced:focus-visible {
  outline: 3px solid var(--focus-color);
  outline-offset: 3px;
  box-shadow: 0 0 0 6px rgba(59, 130, 246, 0.2);
}
```

### Issue 2: Insufficient Color Contrast
```css
/* Problem: Low contrast text */
.text-gray { color: #9ca3af; } /* ❌ 2.93:1 contrast - fails AA */

/* Solution: Use higher contrast colors */
.text-accessible { 
  color: #4b5563; /* ✅ 6.87:1 contrast - passes AA */
}

/* Provide high contrast alternative */
@media (prefers-contrast: high) {
  .text-accessible {
    color: #1f2937; /* 12.63:1 contrast */
  }
}
```

### Issue 3: Missing Alternative Text
```html
<!-- Problem: Decorative images with alt text -->
<img src="decorative-border.svg" alt="decorative border"> <!-- ❌ -->

<!-- Solution: Empty alt for decorative images -->
<img src="decorative-border.svg" alt="" role="presentation"> <!-- ✅ -->

<!-- Problem: Complex charts without alternatives -->
<div id="chart"></div> <!-- ❌ -->

<!-- Solution: Comprehensive chart accessibility -->
<div role="img" aria-labelledby="chart-title" aria-describedby="chart-desc">
  <h3 id="chart-title">Monthly Anomaly Trends</h3>
  <div id="chart-desc">
    Bar chart showing anomaly counts by month. January: 12, February: 8, 
    March: 15. Highest month was March with 15 anomalies.
  </div>
  <div id="chart"></div>
  
  <!-- Data table alternative -->
  <table class="sr-only">
    <caption>Monthly Anomaly Data</caption>
    <thead>
      <tr><th>Month</th><th>Count</th></tr>
    </thead>
    <tbody>
      <tr><td>January</td><td>12</td></tr>
      <tr><td>February</td><td>8</td></tr>
      <tr><td>March</td><td>15</td></tr>
    </tbody>
  </table>
</div>
```

### Issue 4: Keyboard Trap in Modals
```javascript
// Problem: Focus escapes modal dialog
function openModal() {
  document.getElementById('modal').style.display = 'block';
  // ❌ No focus management
}

// Solution: Implement proper focus trap
class AccessibleModal {
  constructor(modalElement) {
    this.modal = modalElement;
    this.focusTrap = new FocusTrap(modalElement);
    this.previousActiveElement = null;
  }
  
  open() {
    this.previousActiveElement = document.activeElement;
    this.modal.style.display = 'block';
    this.modal.setAttribute('aria-hidden', 'false');
    
    // Enable focus trap
    this.focusTrap.activate();
    
    // Focus first focusable element
    const firstFocusable = this.modal.querySelector(
      'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
    );
    firstFocusable?.focus();
  }
  
  close() {
    this.modal.style.display = 'none';
    this.modal.setAttribute('aria-hidden', 'true');
    
    // Disable focus trap
    this.focusTrap.deactivate();
    
    // Return focus to trigger element
    this.previousActiveElement?.focus();
  }
}
```

## 📊 **Accessibility Checklist**

### Pre-Launch Checklist
```markdown
## WCAG 2.1 AA Compliance Checklist

### Perceivable
- [ ] All images have appropriate alt text
- [ ] Videos have captions and transcripts
- [ ] Color is not the only means of conveying information
- [ ] Text contrast ratio meets AA standards (4.5:1 normal, 3:1 large)
- [ ] Content can be resized up to 200% without horizontal scrolling
- [ ] Content is structured with proper headings (H1-H6)

### Operable
- [ ] All functionality is keyboard accessible
- [ ] No keyboard traps exist
- [ ] Focus indicators are clearly visible
- [ ] Users can pause, stop, or hide moving content
- [ ] Page titles are descriptive and unique
- [ ] Skip links are provided
- [ ] Link purposes are clear from context

### Understandable
- [ ] Language of page and parts is identified
- [ ] Navigation is consistent across pages
- [ ] Form labels and instructions are clear
- [ ] Error messages are descriptive and helpful
- [ ] Help information is available

### Robust
- [ ] Content works with assistive technologies
- [ ] Valid HTML markup is used
- [ ] ARIA is used correctly
- [ ] Content works in multiple browsers
- [ ] Content works with JavaScript disabled (graceful degradation)

### Testing Completed
- [ ] Automated testing with axe-core
- [ ] Lighthouse accessibility audit
- [ ] Manual keyboard navigation testing
- [ ] Screen reader testing (NVDA/JAWS/VoiceOver)
- [ ] Color contrast validation
- [ ] Mobile accessibility testing
- [ ] Focus management verification
- [ ] Form accessibility validation
```

### Component-Level Checklist
```markdown
## Component Accessibility Checklist

### Buttons
- [ ] Proper button semantics (button element or role="button")
- [ ] Clear and descriptive labels
- [ ] Keyboard activation (Enter/Space keys)
- [ ] Focus indicators
- [ ] Disabled state properly indicated
- [ ] Loading states announced to screen readers

### Forms
- [ ] All inputs have associated labels
- [ ] Required fields are indicated
- [ ] Error messages are associated with fields
- [ ] Form validation is accessible
- [ ] Fieldsets group related fields
- [ ] Help text is properly associated

### Navigation
- [ ] Skip links are provided
- [ ] Navigation landmarks are used
- [ ] Current page/section is indicated
- [ ] Breadcrumbs are accessible
- [ ] Mobile navigation is keyboard accessible

### Data Tables
- [ ] Table headers are properly associated
- [ ] Complex tables use scope attributes
- [ ] Caption describes table purpose
- [ ] Sortable columns are announced
- [ ] Pagination is accessible

### Modal Dialogs
- [ ] Focus is trapped within modal
- [ ] Focus returns to trigger on close
- [ ] Escape key closes modal
- [ ] Modal has appropriate labels
- [ ] Background content is hidden from screen readers

### Charts and Visualizations
- [ ] Alternative text or data tables provided
- [ ] Complex charts have detailed descriptions
- [ ] Interactive elements are keyboard accessible
- [ ] Color is not the only means of conveying information
- [ ] Sonification or other alternatives considered
```

This comprehensive accessibility guidelines document provides the foundation for ensuring WCAG 2.1 AA compliance across the Pynomaly platform. Regular testing and adherence to these standards will create an inclusive experience for all users.