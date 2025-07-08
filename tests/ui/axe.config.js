/**
 * Axe-core Configuration for Pynomaly Accessibility Testing
 * WCAG 2.1 AA compliance configuration with custom rules
 */

module.exports = {
  // Global configuration for axe-core
  rules: {
    // Core WCAG 2.1 AA rules
    'color-contrast': { enabled: true },
    'focus-order-semantics': { enabled: true },
    'heading-order': { enabled: true },
    'label': { enabled: true },
    'landmark-one-main': { enabled: true },
    'page-has-heading-one': { enabled: true },
    'region': { enabled: true },

    // Enhanced accessibility rules
    'aria-allowed-attr': { enabled: true },
    'aria-command-name': { enabled: true },
    'aria-hidden-body': { enabled: true },
    'aria-hidden-focus': { enabled: true },
    'aria-input-field-name': { enabled: true },
    'aria-meter-name': { enabled: true },
    'aria-progressbar-name': { enabled: true },
    'aria-required-attr': { enabled: true },
    'aria-required-children': { enabled: true },
    'aria-required-parent': { enabled: true },
    'aria-roledescription': { enabled: true },
    'aria-roles': { enabled: true },
    'aria-text': { enabled: true },
    'aria-toggle-field-name': { enabled: true },
    'aria-tooltip-name': { enabled: true },
    'aria-treeitem-name': { enabled: true },
    'aria-valid-attr': { enabled: true },
    'aria-valid-attr-value': { enabled: true },

    // Form accessibility
    'autocomplete-valid': { enabled: true },
    'form-field-multiple-labels': { enabled: true },
    'input-button-name': { enabled: true },
    'input-image-alt': { enabled: true },

    // Keyboard accessibility
    'accesskeys': { enabled: true },
    'focus-order-semantics': { enabled: true },
    'focusable-content': { enabled: true },
    'interactive-controls-focus': { enabled: true },
    'keyboard': { enabled: true },
    'no-focusable-content': { enabled: true },
    'tabindex': { enabled: true },

    // Image accessibility
    'image-alt': { enabled: true },
    'image-redundant-alt': { enabled: true },
    'object-alt': { enabled: true },
    'role-img-alt': { enabled: true },
    'svg-img-alt': { enabled: true },

    // Table accessibility
    'table-duplicate-name': { enabled: true },
    'table-fake-caption': { enabled: true },
    'td-has-header': { enabled: true },
    'td-headers-attr': { enabled: true },
    'th-has-data-cells': { enabled: true },

    // List accessibility
    'list': { enabled: true },
    'listitem': { enabled: true },
    'definition-list': { enabled: true },
    'dlitem': { enabled: true },

    // Link accessibility
    'link-in-text-block': { enabled: true },
    'link-name': { enabled: true },

    // Media accessibility
    'audio-caption': { enabled: true },
    'video-caption': { enabled: true },
    'video-description': { enabled: true },

    // Structural accessibility
    'bypass': { enabled: true },
    'duplicate-id': { enabled: true },
    'duplicate-id-active': { enabled: true },
    'duplicate-id-aria': { enabled: true },
    'frame-title': { enabled: true },
    'html-has-lang': { enabled: true },
    'html-lang-valid': { enabled: true },
    'html-xml-lang-mismatch': { enabled: true },
    'meta-refresh': { enabled: true },
    'meta-viewport': { enabled: true },
    'nested-interactive': { enabled: true },
    'server-side-image-map': { enabled: true },
    'valid-lang': { enabled: true },

    // Custom rules for Pynomaly
    'custom-button-name': {
      enabled: true,
      selector: 'button:not([aria-label]):not([aria-labelledby])',
      any: ['has-visible-text', 'aria-label', 'aria-labelledby'],
      metadata: {
        description: 'Ensures buttons have accessible names',
        help: 'Buttons must have discernible text'
      }
    }
  },

  // Tags to include in testing
  tags: ['wcag2a', 'wcag2aa', 'wcag21aa', 'best-practice'],

  // Element exclusions
  exclude: [
    // Exclude third-party widgets
    '.percy-hide',
    '[data-percy-hide]',
    '.gtm-',
    '#google_translate_element',

    // Exclude loading states that may not be accessible
    '.loading-spinner',
    '.skeleton-loader',

    // Exclude decorative elements
    '.decorative',
    '[role="presentation"]',
    '[aria-hidden="true"]'
  ],

  // Include only specific areas for focused testing
  include: [
    'main',
    'nav',
    'header',
    'footer',
    '[role="main"]',
    '[role="navigation"]',
    '[role="banner"]',
    '[role="contentinfo"]'
  ],

  // Reporting configuration
  reporter: 'v2',

  // Custom checks for Pynomaly-specific patterns
  checks: {
    'pynomaly-chart-accessible': {
      id: 'pynomaly-chart-accessible',
      evaluate: function(node, options) {
        // Check if chart containers have proper accessibility
        if (node.classList.contains('chart-container') ||
            node.getAttribute('data-testid') === 'chart') {

          const hasAriaLabel = node.getAttribute('aria-label');
          const hasAriaLabelledBy = node.getAttribute('aria-labelledby');
          const hasRole = node.getAttribute('role');
          const hasTitle = node.querySelector('title');

          return hasAriaLabel || hasAriaLabelledBy || hasRole === 'img' || hasTitle;
        }
        return true;
      },
      metadata: {
        impact: 'serious',
        messages: {
          pass: 'Chart has accessible name or description',
          fail: 'Chart must have aria-label, aria-labelledby, role="img", or title element'
        }
      }
    },

    'pynomaly-data-table-accessible': {
      id: 'pynomaly-data-table-accessible',
      evaluate: function(node, options) {
        // Check if data tables have proper headers
        if (node.tagName.toLowerCase() === 'table' ||
            node.getAttribute('role') === 'table') {

          const hasCaption = node.querySelector('caption');
          const hasAriaLabel = node.getAttribute('aria-label');
          const hasAriaLabelledBy = node.getAttribute('aria-labelledby');
          const hasHeaderCells = node.querySelectorAll('th').length > 0;

          return (hasCaption || hasAriaLabel || hasAriaLabelledBy) && hasHeaderCells;
        }
        return true;
      },
      metadata: {
        impact: 'serious',
        messages: {
          pass: 'Data table has accessible structure',
          fail: 'Data table must have caption or aria-label and header cells'
        }
      }
    },

    'pynomaly-form-error-accessible': {
      id: 'pynomaly-form-error-accessible',
      evaluate: function(node, options) {
        // Check if form errors are properly associated
        if (node.classList.contains('error') ||
            node.classList.contains('invalid') ||
            node.getAttribute('aria-invalid') === 'true') {

          const hasAriaDescribedBy = node.getAttribute('aria-describedby');
          const hasAriaLabel = node.getAttribute('aria-label');
          const errorElement = document.querySelector(`#${hasAriaDescribedBy}`);

          return hasAriaDescribedBy && errorElement || hasAriaLabel;
        }
        return true;
      },
      metadata: {
        impact: 'serious',
        messages: {
          pass: 'Form error is properly associated with field',
          fail: 'Form error must be associated with field via aria-describedby'
        }
      }
    }
  },

  // Accessibility testing levels
  levels: {
    // Level A compliance
    A: ['wcag2a'],

    // Level AA compliance (target)
    AA: ['wcag2a', 'wcag2aa'],

    // Level AAA compliance (aspirational)
    AAA: ['wcag2a', 'wcag2aa', 'wcag2aaa'],

    // WCAG 2.1 specific
    '2.1': ['wcag21a', 'wcag21aa']
  },

  // Browser-specific configurations
  browsers: {
    chrome: {
      // Chrome-specific accessibility features
      enableExperimentalFeatures: true
    },
    firefox: {
      // Firefox-specific accessibility features
      enableA11yEngine: true
    },
    safari: {
      // Safari-specific accessibility features
      enableVoiceOverTesting: false // Not available in automated testing
    }
  },

  // Performance configuration
  performance: {
    timeout: 30000,
    maxElements: 1000,
    skipHiddenElements: true
  },

  // Locale configuration for international accessibility
  locale: 'en-US',

  // Output configuration
  output: {
    format: ['json', 'html'],
    destination: 'test_reports/accessibility/',
    includeScreenshots: true,
    includeViolationDetails: true
  }
};
