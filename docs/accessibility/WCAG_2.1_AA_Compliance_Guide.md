# WCAG 2.1 AA Compliance Guide for Pynomaly

🍞 **Breadcrumb:** 🏠 [Home](../index.md) > 📁 Accessibility

---


## Overview

This guide provides comprehensive documentation for maintaining WCAG 2.1 AA accessibility compliance in the Pynomaly anomaly detection platform. It covers implementation standards, testing procedures, and ongoing maintenance requirements.

## Table of Contents

1. [Accessibility Principles](#accessibility-principles)
2. [WCAG 2.1 AA Success Criteria](#wcag-21-aa-success-criteria)
3. [Implementation Standards](#implementation-standards)
4. [Testing Procedures](#testing-procedures)
5. [Common Accessibility Patterns](#common-accessibility-patterns)
6. [Automated Testing Framework](#automated-testing-framework)
7. [Manual Testing Checklist](#manual-testing-checklist)
8. [Accessibility Maintenance](#accessibility-maintenance)
9. [Resources and Tools](#resources-and-tools)

## Accessibility Principles

### 1. Perceivable
Information and user interface components must be presentable to users in ways they can perceive.

**Key Requirements:**
- Provide text alternatives for non-text content
- Provide captions and alternatives for multimedia
- Ensure sufficient color contrast (4.5:1 for normal text, 3:1 for large text)
- Make content adaptable to different presentations

### 2. Operable
User interface components and navigation must be operable.

**Key Requirements:**
- Make all functionality available via keyboard
- Give users enough time to read content
- Do not use content that causes seizures
- Help users navigate and find content

### 3. Understandable
Information and the operation of user interface must be understandable.

**Key Requirements:**
- Make text readable and understandable
- Make content appear and operate in predictable ways
- Help users avoid and correct mistakes

### 4. Robust
Content must be robust enough to be interpreted reliably by assistive technologies.

**Key Requirements:**
- Maximize compatibility with assistive technologies
- Use valid, semantic HTML
- Provide proper ARIA labels and roles

## WCAG 2.1 AA Success Criteria

### Level A Requirements

#### 1.1.1 Non-text Content
All non-text content has a text alternative that serves the equivalent purpose.

**Implementation:**
```html
<!-- Images -->
<img src="anomaly-chart.png" alt="Anomaly detection results showing 15 anomalies detected over the past 24 hours">

<!-- Charts and visualizations -->
<svg role="img" aria-labelledby="chart-title chart-desc">
  <title id="chart-title">Monthly Anomaly Trends</title>
  <desc id="chart-desc">Line chart showing anomaly detection rates from January to December, with highest rates in March (45 anomalies) and lowest in August (12 anomalies)</desc>
</svg>

<!-- Interactive controls -->
<button aria-label="Start anomaly detection analysis">
  <svg>...</svg>
</button>
```

#### 1.2.1 Audio-only and Video-only (Prerecorded)
Provide alternatives for time-based media.

**Implementation:**
- Provide transcripts for audio content
- Provide audio descriptions for video content
- Include captions for instructional videos

#### 1.3.1 Info and Relationships
Information, structure, and relationships conveyed through presentation can be programmatically determined.

**Implementation:**
```html
<!-- Proper heading hierarchy -->
<h1>Anomaly Detection Dashboard</h1>
  <h2>Recent Detections</h2>
    <h3>Critical Anomalies</h3>
    <h3>Moderate Anomalies</h3>
  <h2>System Status</h2>

<!-- Form labels -->
<label for="dataset-name">Dataset Name *</label>
<input type="text" id="dataset-name" required aria-describedby="name-help">
<div id="name-help">Enter a descriptive name for your dataset</div>

<!-- Table headers -->
<table>
  <thead>
    <tr>
      <th scope="col">Timestamp</th>
      <th scope="col">Value</th>
      <th scope="col">Anomaly Score</th>
      <th scope="col">Status</th>
    </tr>
  </thead>
  <tbody>
    <!-- table rows -->
  </tbody>
</table>
```

#### 1.3.2 Meaningful Sequence
Content can be presented in a meaningful sequence without losing meaning.

**Implementation:**
- Use logical DOM order
- Ensure CSS positioning doesn't break reading order
- Test with CSS disabled

#### 2.1.1 Keyboard
All functionality is available from a keyboard.

**Implementation:**
```html
<!-- Keyboard accessible controls -->
<button onclick="startDetection()" onkeydown="handleKeyDown(event)">Start Detection</button>

<!-- Custom interactive elements -->
<div role="button" tabindex="0" aria-label="Expand anomaly details" 
     onclick="toggleDetails()" onkeydown="handleToggleKeyDown(event)">
  Details
</div>
```

```javascript
function handleKeyDown(event) {
  if (event.key === 'Enter' || event.key === ' ') {
    event.preventDefault();
    startDetection();
  }
}

function handleToggleKeyDown(event) {
  if (event.key === 'Enter' || event.key === ' ') {
    event.preventDefault();
    toggleDetails();
  }
}
```

### Level AA Requirements

#### 1.4.3 Contrast (Minimum)
Text has a contrast ratio of at least 4.5:1 (3:1 for large text).

**Pynomaly Design System Colors:**
```css
/* WCAG AA compliant color combinations */
:root {
  /* Primary colors with proper contrast */
  --color-primary-500: #0ea5e9;    /* 4.52:1 on white background */
  --color-primary-600: #0284c7;    /* 5.77:1 on white background */
  --color-primary-700: #0369a1;    /* 7.25:1 on white background */
  
  /* Text colors */
  --color-text-primary: #1e293b;   /* 15.36:1 on white background */
  --color-text-secondary: #64748b; /* 4.78:1 on white background */
  --color-text-muted: #94a3b8;     /* 3.07:1 on white (large text only) */
  
  /* Status colors */
  --color-success: #16a34a;        /* 4.68:1 on white background */
  --color-warning: #ca8a04;        /* 4.51:1 on white background */
  --color-error: #dc2626;          /* 5.25:1 on white background */
}
```

#### 1.4.4 Resize Text
Text can be resized up to 200% without loss of content or functionality.

**Implementation:**
```css
/* Use relative units */
body {
  font-size: 1rem; /* 16px base */
  line-height: 1.5;
}

.heading-1 {
  font-size: 2.25rem; /* 36px at base size */
}

.body-text {
  font-size: 1rem; /* 16px at base size */
}

/* Responsive containers */
.container {
  max-width: 100%;
  padding: 1rem;
}

/* Avoid fixed heights */
.card {
  min-height: auto;
  padding: 1.5rem;
}
```

#### 1.4.5 Images of Text
Images of text are only used for decoration or when essential.

**Implementation:**
- Use actual text instead of text images
- Use CSS for styling text
- If text images are necessary, provide alternative text

#### 2.4.1 Bypass Blocks
Provide a mechanism to bypass blocks of content.

**Implementation:**
```html
<!-- Skip links -->
<a href="#main-content" class="skip-link">Skip to main content</a>
<a href="#navigation" class="skip-link">Skip to navigation</a>

<nav id="navigation">
  <!-- navigation content -->
</nav>

<main id="main-content">
  <!-- main content -->
</main>
```

```css
.skip-link {
  position: absolute;
  top: -40px;
  left: 6px;
  background: #000;
  color: #fff;
  padding: 8px;
  text-decoration: none;
  z-index: 1000;
}

.skip-link:focus {
  top: 6px;
}
```

#### 2.4.2 Page Titled
Web pages have titles that describe topic or purpose.

**Implementation:**
```html
<title>Anomaly Detection Dashboard - Pynomaly</title>
<title>Dataset Upload - Pynomaly</title>
<title>Model Configuration - Pynomaly</title>
```

#### 2.4.3 Focus Order
Components receive focus in an order that preserves meaning and operability.

**Implementation:**
- Use logical DOM order
- Use `tabindex="0"` for custom interactive elements
- Use `tabindex="-1"` to remove from tab order when appropriate
- Never use positive tabindex values

#### 3.1.1 Language of Page
Default human language of each web page can be programmatically determined.

**Implementation:**
```html
<html lang="en">
```

#### 3.2.1 On Focus
When a component receives focus, it does not initiate a change of context.

**Implementation:**
- Avoid auto-submitting forms on focus
- Don't automatically redirect on focus
- Don't open popups on focus

#### 3.2.2 On Input
Changing the setting of a user interface component does not automatically cause a change of context.

**Implementation:**
```html
<!-- Good: Explicit submit -->
<form>
  <select id="algorithm-type" onchange="updateForm()">
    <option>Isolation Forest</option>
    <option>One-Class SVM</option>
  </select>
  <button type="submit">Apply Algorithm</button>
</form>

<!-- Inform users of automatic changes -->
<select aria-describedby="auto-submit-notice">
  <option>Filter option</option>
</select>
<div id="auto-submit-notice">Results will update automatically when you make a selection</div>
```

## Implementation Standards

### HTML Semantic Structure

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Page Title - Pynomaly</title>
</head>
<body>
  <!-- Skip links -->
  <a href="#main" class="skip-link">Skip to main content</a>
  
  <!-- Header with navigation -->
  <header role="banner">
    <nav role="navigation" aria-label="Main navigation">
      <!-- navigation items -->
    </nav>
  </header>
  
  <!-- Main content -->
  <main id="main" role="main">
    <h1>Page Heading</h1>
    <!-- page content -->
  </main>
  
  <!-- Footer -->
  <footer role="contentinfo">
    <!-- footer content -->
  </footer>
</body>
</html>
```

### Form Accessibility

```html
<form>
  <fieldset>
    <legend>Dataset Configuration</legend>
    
    <div class="form-group">
      <label for="dataset-name" class="form-label required">
        Dataset Name
      </label>
      <input 
        type="text" 
        id="dataset-name" 
        class="form-input"
        required 
        aria-describedby="name-help name-error"
        aria-invalid="false"
      >
      <div id="name-help" class="form-helper">
        Enter a descriptive name for your dataset
      </div>
      <div id="name-error" class="form-error" style="display: none;">
        Dataset name is required
      </div>
    </div>
    
    <div class="form-group">
      <label for="algorithm-select" class="form-label">
        Detection Algorithm
      </label>
      <select id="algorithm-select" class="form-select">
        <option value="">Select an algorithm</option>
        <option value="isolation-forest">Isolation Forest</option>
        <option value="one-class-svm">One-Class SVM</option>
      </select>
    </div>
    
    <button type="submit" class="btn-primary">
      Create Dataset
    </button>
  </fieldset>
</form>
```

### Data Visualization Accessibility

```html
<!-- Chart with alternative text -->
<div class="chart-container">
  <h3 id="chart-title">Anomaly Detection Results</h3>
  <div 
    id="anomaly-chart" 
    role="img"
    aria-labelledby="chart-title"
    aria-describedby="chart-summary"
  >
    <!-- D3.js chart content -->
  </div>
  <div id="chart-summary" class="sr-only">
    Chart showing 150 data points over the last 24 hours, with 12 anomalies detected. 
    Peak anomaly activity occurred between 2 PM and 4 PM with 7 anomalies detected.
  </div>
  
  <!-- Alternative data table -->
  <details>
    <summary>View data table</summary>
    <table>
      <caption>Anomaly detection results data</caption>
      <thead>
        <tr>
          <th scope="col">Time</th>
          <th scope="col">Value</th>
          <th scope="col">Anomaly Score</th>
          <th scope="col">Status</th>
        </tr>
      </thead>
      <tbody>
        <!-- data rows -->
      </tbody>
    </table>
  </details>
</div>
```

### Dynamic Content and ARIA Live Regions

```html
<!-- Status updates -->
<div id="status-region" aria-live="polite" aria-atomic="true">
  Detection in progress...
</div>

<!-- Error messages -->
<div id="error-region" aria-live="assertive" role="alert">
  <!-- Error messages appear here -->
</div>

<!-- Progress indicator -->
<div role="progressbar" 
     aria-valuenow="45" 
     aria-valuemin="0" 
     aria-valuemax="100"
     aria-label="Dataset processing progress">
  <div class="progress-bar" style="width: 45%"></div>
</div>
```

### Interactive Components

```html
<!-- Modal dialog -->
<div 
  id="settings-modal" 
  role="dialog" 
  aria-modal="true"
  aria-labelledby="modal-title"
  aria-describedby="modal-desc"
  style="display: none;"
>
  <div class="modal-header">
    <h2 id="modal-title">Detection Settings</h2>
    <button 
      type="button" 
      class="modal-close"
      aria-label="Close settings dialog"
      onclick="closeModal()"
    >
      ×
    </button>
  </div>
  <div class="modal-body">
    <p id="modal-desc">Configure your anomaly detection parameters.</p>
    <!-- modal content -->
  </div>
</div>

<!-- Expandable content -->
<button 
  type="button"
  aria-expanded="false"
  aria-controls="advanced-options"
  onclick="toggleAdvanced()"
>
  Advanced Options
</button>
<div id="advanced-options" style="display: none;">
  <!-- advanced options content -->
</div>
```

## Testing Procedures

### Automated Testing

Run the comprehensive accessibility test suite:

```bash
# Smoke test (quick validation)
python tests/ui/accessibility/automated_accessibility_tests.py --scenario smoke

# Comprehensive test
python tests/ui/accessibility/automated_accessibility_tests.py --scenario comprehensive

# CI mode (fails on violations)
python tests/ui/accessibility/automated_accessibility_tests.py --scenario critical --ci
```

### Manual Testing Checklist

#### Keyboard Navigation
- [ ] All interactive elements are reachable via keyboard
- [ ] Tab order is logical and follows visual flow
- [ ] Focus indicators are visible and clear
- [ ] No keyboard traps exist
- [ ] Skip links work correctly

#### Screen Reader Testing
- [ ] Page content is read in logical order
- [ ] Headings provide proper document structure
- [ ] Images have appropriate alternative text
- [ ] Form labels are properly associated
- [ ] Dynamic content updates are announced

#### Color and Contrast
- [ ] All text meets contrast requirements (4.5:1 minimum)
- [ ] Information is not conveyed by color alone
- [ ] Focus indicators are visible with sufficient contrast
- [ ] UI components meet 3:1 contrast requirement

#### Responsive Design
- [ ] Content is usable at 200% zoom
- [ ] No horizontal scrolling at 320px width
- [ ] All functionality remains available on mobile
- [ ] Touch targets are at least 44x44 pixels

#### Forms
- [ ] All form controls have labels
- [ ] Required fields are clearly indicated
- [ ] Error messages are descriptive and helpful
- [ ] Form submission provides feedback

### Browser Testing

Test across multiple browsers and assistive technologies:

**Browsers:**
- Chrome with ChromeVox
- Firefox with NVDA (Windows)
- Safari with VoiceOver (macOS)
- Edge with Narrator (Windows)

**Screen Readers:**
- NVDA (Windows)
- JAWS (Windows)
- VoiceOver (macOS, iOS)
- TalkBack (Android)

## Common Accessibility Patterns

### Loading States

```html
<!-- Loading with appropriate feedback -->
<button type="button" disabled aria-describedby="loading-status">
  <span aria-hidden="true">Processing...</span>
  <span class="sr-only">Processing dataset, please wait</span>
</button>
<div id="loading-status" aria-live="polite">
  Dataset upload: 60% complete
</div>
```

### Error Handling

```html
<!-- Form with error state -->
<div class="form-group">
  <label for="file-input" class="form-label required">
    Dataset File
  </label>
  <input 
    type="file" 
    id="file-input"
    aria-describedby="file-error"
    aria-invalid="true"
    class="form-input error"
  >
  <div id="file-error" class="form-error" role="alert">
    File must be in CSV format and less than 100MB
  </div>
</div>
```

### Data Tables

```html
<table class="data-table">
  <caption>
    Anomaly Detection Results (150 items)
    <button type="button" class="btn-link">Sort by timestamp</button>
  </caption>
  <thead>
    <tr>
      <th scope="col">
        <button type="button" aria-sort="ascending">
          Timestamp
          <span aria-hidden="true">↑</span>
        </button>
      </th>
      <th scope="col">Value</th>
      <th scope="col">Anomaly Score</th>
      <th scope="col">Actions</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>2023-12-01 14:30:00</td>
      <td>145.67</td>
      <td>0.85</td>
      <td>
        <button type="button" aria-label="View details for anomaly at 2023-12-01 14:30:00">
          Details
        </button>
      </td>
    </tr>
  </tbody>
</table>
```

## Automated Testing Framework

### Running Tests

```bash
# Install dependencies
pip install playwright pytest-playwright pytest-axe

# Install browsers
playwright install

# Run accessibility tests
pytest tests/ui/accessibility/ -v --html=reports/accessibility.html
```

### Test Configuration

```python
# tests/conftest.py
import pytest
from playwright.sync_api import Page

@pytest.fixture
def accessibility_page(page: Page):
    """Configure page for accessibility testing"""
    # Set high contrast mode for testing
    page.emulate_media(color_scheme="no-preference")
    
    # Enable accessibility tree
    page.add_init_script("""
        window.accessibility = true;
        Object.defineProperty(navigator, 'webdriver', {
            get: () => false,
        });
    """)
    
    return page
```

### Custom Test Examples

```python
# tests/ui/accessibility/test_pynomaly_accessibility.py
import pytest
from playwright.sync_api import Page, expect

@pytest.mark.accessibility
class TestPynomaloAccessibility:
    
    def test_dashboard_keyboard_navigation(self, page: Page):
        """Test keyboard navigation on dashboard"""
        page.goto("/dashboard")
        
        # Test skip links
        page.keyboard.press("Tab")
        expect(page.locator(".skip-link")).to_be_focused()
        
        # Test main navigation
        page.keyboard.press("Tab")
        expect(page.locator("nav a:first-child")).to_be_focused()
        
        # Test main content area
        page.keyboard.press("Enter")  # Activate skip link
        expect(page.locator("main")).to_be_focused()
    
    def test_form_accessibility(self, page: Page):
        """Test form accessibility"""
        page.goto("/datasets/upload")
        
        # Check form labels
        form_inputs = page.locator("input, select, textarea")
        for i in range(form_inputs.count()):
            input_elem = form_inputs.nth(i)
            input_id = input_elem.get_attribute("id")
            
            if input_id:
                label = page.locator(f"label[for='{input_id}']")
                expect(label).to_be_visible()
    
    def test_chart_accessibility(self, page: Page):
        """Test chart accessibility"""
        page.goto("/dashboard")
        
        # Wait for chart to load
        page.wait_for_selector("[data-component='anomaly-chart']")
        
        # Check for alternative text
        chart = page.locator("[data-component='anomaly-chart']")
        expect(chart).to_have_attribute("aria-label")
        
        # Check for data table alternative
        page.click("text=View data table")
        expect(page.locator("table")).to_be_visible()
```

## Accessibility Maintenance

### Development Workflow

1. **Design Phase**
   - Include accessibility requirements in designs
   - Define color palettes with proper contrast
   - Plan keyboard navigation flows

2. **Implementation Phase**
   - Use semantic HTML elements
   - Implement ARIA labels and roles
   - Follow established patterns

3. **Testing Phase**
   - Run automated accessibility tests
   - Perform manual keyboard testing
   - Test with screen readers

4. **Code Review Phase**
   - Review accessibility implementation
   - Check for WCAG compliance
   - Validate test coverage

### Continuous Monitoring

```bash
# Add to CI/CD pipeline
- name: Accessibility Tests
  run: |
    npm start &
    sleep 10
    python tests/ui/accessibility/automated_accessibility_tests.py --scenario critical --ci
```

### Regular Audits

- Monthly comprehensive accessibility audits
- Quarterly user testing with people with disabilities
- Annual third-party accessibility assessment

## Resources and Tools

### Testing Tools
- **axe-core**: Automated accessibility testing engine
- **Lighthouse**: Performance and accessibility audits
- **WAVE**: Web accessibility evaluation tool
- **Color Oracle**: Color blindness simulator
- **High Contrast**: Browser extension for contrast testing

### Screen Readers
- **NVDA**: Free screen reader for Windows
- **VoiceOver**: Built-in screen reader for macOS/iOS
- **TalkBack**: Built-in screen reader for Android
- **JAWS**: Commercial screen reader for Windows

### Documentation
- [WCAG 2.1 Guidelines](https://www.w3.org/WAI/WCAG21/quickref/)
- [ARIA Authoring Practices](https://www.w3.org/WAI/ARIA/apg/)
- [WebAIM Resources](https://webaim.org/)
- [A11y Project Checklist](https://www.a11yproject.com/checklist/)

### Browser Extensions
- **axe DevTools**: Browser extension for accessibility testing
- **WAVE**: Web accessibility evaluation extension
- **Accessibility Insights**: Microsoft's accessibility testing tools
- **Colour Contrast Analyser**: Color contrast checking tool

## Conclusion

Maintaining WCAG 2.1 AA compliance requires ongoing commitment and systematic testing. By following these guidelines, using the automated testing framework, and conducting regular manual testing, the Pynomaly platform can provide an accessible experience for all users.

Remember that accessibility is not a one-time implementation but an ongoing process that should be integrated into all aspects of development and design.