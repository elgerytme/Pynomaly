# Pynomaly Design Tokens Specification

🍞 **Breadcrumb:** 🏠 [Home](../index.md) > 📁 Design-System

---


## Overview

Design tokens are the foundational elements of the Pynomaly design system, providing a consistent and scalable approach to design decisions. These tokens ensure visual consistency across all platforms and interfaces while enabling theme customization and maintenance.

## Token Categories

### 1. Color Tokens

#### Primary Color Palette
```css
:root {
  /* Primary Blues - Main brand colors */
  --color-primary-50: #eff6ff;   /* Very light blue */
  --color-primary-100: #dbeafe;  /* Light blue */
  --color-primary-200: #bfdbfe;  /* Lighter blue */
  --color-primary-300: #93c5fd;  /* Light blue */
  --color-primary-400: #60a5fa;  /* Medium light blue */
  --color-primary-500: #3b82f6;  /* Primary blue (default) */
  --color-primary-600: #2563eb;  /* Medium blue */
  --color-primary-700: #1d4ed8;  /* Dark blue */
  --color-primary-800: #1e40af;  /* Darker blue */
  --color-primary-900: #1e3a8a;  /* Darkest blue */
}
```

#### Semantic Color Tokens
```css
:root {
  /* Success - Green palette */
  --color-success-50: #ecfdf5;
  --color-success-100: #d1fae5;
  --color-success-200: #a7f3d0;
  --color-success-300: #6ee7b7;
  --color-success-400: #34d399;
  --color-success-500: #10b981;   /* Default success */
  --color-success-600: #059669;
  --color-success-700: #047857;
  --color-success-800: #065f46;
  --color-success-900: #064e3b;

  /* Warning - Amber palette */
  --color-warning-50: #fffbeb;
  --color-warning-100: #fef3c7;
  --color-warning-200: #fde68a;
  --color-warning-300: #fcd34d;
  --color-warning-400: #fbbf24;
  --color-warning-500: #f59e0b;   /* Default warning */
  --color-warning-600: #d97706;
  --color-warning-700: #b45309;
  --color-warning-800: #92400e;
  --color-warning-900: #78350f;

  /* Danger - Red palette */
  --color-danger-50: #fef2f2;
  --color-danger-100: #fee2e2;
  --color-danger-200: #fecaca;
  --color-danger-300: #fca5a5;
  --color-danger-400: #f87171;
  --color-danger-500: #ef4444;    /* Default danger */
  --color-danger-600: #dc2626;
  --color-danger-700: #b91c1c;
  --color-danger-800: #991b1b;
  --color-danger-900: #7f1d1d;
}
```

#### Neutral Colors
```css
:root {
  /* Grayscale - Text, borders, backgrounds */
  --color-gray-50: #f9fafb;     /* Lightest gray */
  --color-gray-100: #f3f4f6;    /* Very light gray */
  --color-gray-200: #e5e7eb;    /* Light gray */
  --color-gray-300: #d1d5db;    /* Light gray */
  --color-gray-400: #9ca3af;    /* Medium light gray */
  --color-gray-500: #6b7280;    /* Medium gray */
  --color-gray-600: #4b5563;    /* Medium dark gray */
  --color-gray-700: #374151;    /* Dark gray */
  --color-gray-800: #1f2937;    /* Darker gray */
  --color-gray-900: #111827;    /* Darkest gray */
}
```

#### Context-Specific Color Tokens
```css
:root {
  /* Text colors */
  --color-text-primary: var(--color-gray-900);
  --color-text-secondary: var(--color-gray-600);
  --color-text-muted: var(--color-gray-500);
  --color-text-inverse: #ffffff;
  --color-text-link: var(--color-primary-600);
  --color-text-link-hover: var(--color-primary-700);

  /* Background colors */
  --color-bg-primary: #ffffff;
  --color-bg-secondary: var(--color-gray-50);
  --color-bg-tertiary: var(--color-gray-100);
  --color-bg-inverse: var(--color-gray-900);
  --color-bg-overlay: rgba(0, 0, 0, 0.5);

  /* Border colors */
  --color-border-light: var(--color-gray-200);
  --color-border-medium: var(--color-gray-300);
  --color-border-dark: var(--color-gray-400);
  --color-border-focus: var(--color-primary-500);

  /* Anomaly-specific colors */
  --color-anomaly-high: var(--color-danger-500);
  --color-anomaly-medium: var(--color-warning-500);
  --color-anomaly-low: var(--color-warning-300);
  --color-normal: var(--color-success-500);
  --color-unknown: var(--color-gray-400);
}
```

### 2. Typography Tokens

#### Font Families
```css
:root {
  /* Primary font stack */
  --font-family-sans: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;

  /* Monospace font for code */
  --font-family-mono: 'JetBrains Mono', 'Fira Code', Consolas, 'Liberation Mono', Menlo, Courier, monospace;

  /* Display font for headlines */
  --font-family-display: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}
```

#### Font Sizes
```css
:root {
  /* Font size scale */
  --font-size-xs: 0.75rem;      /* 12px */
  --font-size-sm: 0.875rem;     /* 14px */
  --font-size-base: 1rem;       /* 16px - base size */
  --font-size-lg: 1.125rem;     /* 18px */
  --font-size-xl: 1.25rem;      /* 20px */
  --font-size-2xl: 1.5rem;      /* 24px */
  --font-size-3xl: 1.875rem;    /* 30px */
  --font-size-4xl: 2.25rem;     /* 36px */
  --font-size-5xl: 3rem;        /* 48px */
  --font-size-6xl: 3.75rem;     /* 60px */
  --font-size-7xl: 4.5rem;      /* 72px */
  --font-size-8xl: 6rem;        /* 96px */
  --font-size-9xl: 8rem;        /* 128px */
}
```

#### Font Weights
```css
:root {
  --font-weight-thin: 100;
  --font-weight-extralight: 200;
  --font-weight-light: 300;
  --font-weight-normal: 400;
  --font-weight-medium: 500;
  --font-weight-semibold: 600;
  --font-weight-bold: 700;
  --font-weight-extrabold: 800;
  --font-weight-black: 900;
}
```

#### Line Heights
```css
:root {
  --line-height-none: 1;
  --line-height-tight: 1.25;
  --line-height-snug: 1.375;
  --line-height-normal: 1.5;
  --line-height-relaxed: 1.625;
  --line-height-loose: 2;
}
```

#### Letter Spacing
```css
:root {
  --letter-spacing-tighter: -0.05em;
  --letter-spacing-tight: -0.025em;
  --letter-spacing-normal: 0;
  --letter-spacing-wide: 0.025em;
  --letter-spacing-wider: 0.05em;
  --letter-spacing-widest: 0.1em;
}
```

### 3. Spacing Tokens

#### Base Spacing Scale
```css
:root {
  /* 4px base unit scale */
  --spacing-0: 0;
  --spacing-px: 1px;
  --spacing-0-5: 0.125rem;    /* 2px */
  --spacing-1: 0.25rem;       /* 4px */
  --spacing-1-5: 0.375rem;    /* 6px */
  --spacing-2: 0.5rem;        /* 8px */
  --spacing-2-5: 0.625rem;    /* 10px */
  --spacing-3: 0.75rem;       /* 12px */
  --spacing-3-5: 0.875rem;    /* 14px */
  --spacing-4: 1rem;          /* 16px */
  --spacing-5: 1.25rem;       /* 20px */
  --spacing-6: 1.5rem;        /* 24px */
  --spacing-7: 1.75rem;       /* 28px */
  --spacing-8: 2rem;          /* 32px */
  --spacing-9: 2.25rem;       /* 36px */
  --spacing-10: 2.5rem;       /* 40px */
  --spacing-11: 2.75rem;      /* 44px */
  --spacing-12: 3rem;         /* 48px */
  --spacing-14: 3.5rem;       /* 56px */
  --spacing-16: 4rem;         /* 64px */
  --spacing-20: 5rem;         /* 80px */
  --spacing-24: 6rem;         /* 96px */
  --spacing-28: 7rem;         /* 112px */
  --spacing-32: 8rem;         /* 128px */
  --spacing-36: 9rem;         /* 144px */
  --spacing-40: 10rem;        /* 160px */
  --spacing-44: 11rem;        /* 176px */
  --spacing-48: 12rem;        /* 192px */
  --spacing-52: 13rem;        /* 208px */
  --spacing-56: 14rem;        /* 224px */
  --spacing-60: 15rem;        /* 240px */
  --spacing-64: 16rem;        /* 256px */
  --spacing-72: 18rem;        /* 288px */
  --spacing-80: 20rem;        /* 320px */
  --spacing-96: 24rem;        /* 384px */
}
```

#### Component-Specific Spacing
```css
:root {
  /* Button spacing */
  --spacing-btn-padding-x-sm: var(--spacing-3);
  --spacing-btn-padding-y-sm: var(--spacing-1-5);
  --spacing-btn-padding-x-base: var(--spacing-4);
  --spacing-btn-padding-y-base: var(--spacing-2);
  --spacing-btn-padding-x-lg: var(--spacing-6);
  --spacing-btn-padding-y-lg: var(--spacing-3);

  /* Form spacing */
  --spacing-form-gap: var(--spacing-4);
  --spacing-form-label-margin: var(--spacing-1);
  --spacing-form-input-padding: var(--spacing-3);

  /* Card spacing */
  --spacing-card-padding: var(--spacing-6);
  --spacing-card-gap: var(--spacing-4);

  /* Container spacing */
  --spacing-container-padding-x: var(--spacing-4);
  --spacing-container-padding-y: var(--spacing-6);
  --spacing-section-gap: var(--spacing-12);
}
```

### 4. Border Radius Tokens

```css
:root {
  --border-radius-none: 0;
  --border-radius-sm: 0.125rem;    /* 2px */
  --border-radius-base: 0.25rem;   /* 4px */
  --border-radius-md: 0.375rem;    /* 6px */
  --border-radius-lg: 0.5rem;      /* 8px */
  --border-radius-xl: 0.75rem;     /* 12px */
  --border-radius-2xl: 1rem;       /* 16px */
  --border-radius-3xl: 1.5rem;     /* 24px */
  --border-radius-full: 9999px;    /* Fully rounded */
}
```

### 5. Shadow Tokens

```css
:root {
  /* Box shadows */
  --shadow-xs: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
  --shadow-sm: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
  --shadow-base: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
  --shadow-md: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
  --shadow-lg: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
  --shadow-xl: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
  --shadow-2xl: 0 50px 100px -20px rgba(0, 0, 0, 0.25);
  --shadow-inner: inset 0 2px 4px 0 rgba(0, 0, 0, 0.06);

  /* Focus shadows */
  --shadow-focus: 0 0 0 3px rgba(59, 130, 246, 0.5);
  --shadow-focus-danger: 0 0 0 3px rgba(239, 68, 68, 0.5);
  --shadow-focus-success: 0 0 0 3px rgba(16, 185, 129, 0.5);
  --shadow-focus-warning: 0 0 0 3px rgba(245, 158, 11, 0.5);
}
```

### 6. Animation Tokens

```css
:root {
  /* Durations */
  --duration-fastest: 100ms;
  --duration-fast: 150ms;
  --duration-normal: 200ms;
  --duration-slow: 300ms;
  --duration-slowest: 500ms;

  /* Easing functions */
  --ease-linear: linear;
  --ease-in: cubic-bezier(0.4, 0, 1, 1);
  --ease-out: cubic-bezier(0, 0, 0.2, 1);
  --ease-in-out: cubic-bezier(0.4, 0, 0.2, 1);
  --ease-bounce: cubic-bezier(0.68, -0.55, 0.265, 1.55);

  /* Component-specific animations */
  --transition-colors: color var(--duration-normal) var(--ease-out),
                       background-color var(--duration-normal) var(--ease-out),
                       border-color var(--duration-normal) var(--ease-out);
  --transition-transform: transform var(--duration-normal) var(--ease-out);
  --transition-opacity: opacity var(--duration-fast) var(--ease-out);
  --transition-all: all var(--duration-normal) var(--ease-out);
}
```

### 7. Z-Index Tokens

```css
:root {
  --z-index-hide: -1;
  --z-index-auto: auto;
  --z-index-base: 0;
  --z-index-docked: 10;
  --z-index-dropdown: 1000;
  --z-index-sticky: 1020;
  --z-index-banner: 1030;
  --z-index-overlay: 1040;
  --z-index-modal: 1050;
  --z-index-popover: 1060;
  --z-index-skiplink: 1070;
  --z-index-toast: 1080;
  --z-index-tooltip: 1090;
}
```

## Dark Theme Overrides

```css
[data-theme="dark"] {
  /* Dark theme color overrides */
  --color-text-primary: #f9fafb;
  --color-text-secondary: #d1d5db;
  --color-text-muted: #9ca3af;
  --color-text-inverse: #111827;

  --color-bg-primary: #111827;
  --color-bg-secondary: #1f2937;
  --color-bg-tertiary: #374151;
  --color-bg-inverse: #ffffff;
  --color-bg-overlay: rgba(0, 0, 0, 0.8);

  --color-border-light: #374151;
  --color-border-medium: #4b5563;
  --color-border-dark: #6b7280;

  /* Adjust shadows for dark theme */
  --shadow-xs: 0 1px 2px 0 rgba(0, 0, 0, 0.3);
  --shadow-sm: 0 1px 3px 0 rgba(0, 0, 0, 0.4), 0 1px 2px 0 rgba(0, 0, 0, 0.3);
  --shadow-base: 0 4px 6px -1px rgba(0, 0, 0, 0.4), 0 2px 4px -1px rgba(0, 0, 0, 0.3);
}
```

## Usage Guidelines

### 1. Token Naming Convention

Tokens follow a hierarchical naming structure:
```
--{category}-{variant}-{scale}
```

Examples:
- `--color-primary-500` (category: color, variant: primary, scale: 500)
- `--spacing-4` (category: spacing, scale: 4)
- `--font-size-lg` (category: font-size, scale: lg)

### 2. Semantic vs. Descriptive Tokens

- **Descriptive tokens**: `--color-blue-500`, `--spacing-4`
- **Semantic tokens**: `--color-primary`, `--spacing-button-padding`

Use semantic tokens for component definitions, descriptive tokens for the underlying values.

### 3. Component-Specific Tokens

When creating component-specific tokens, use this pattern:
```css
--{component}-{property}-{variant}

/* Examples */
--button-padding-x-lg
--card-border-radius
--modal-backdrop-opacity
```

### 4. Accessibility Considerations

- All color tokens must meet WCAG 2.1 AA contrast requirements
- Touch targets should use minimum `--spacing-11` (44px)
- Focus indicators should use `--shadow-focus` tokens
- Animation tokens should respect `prefers-reduced-motion`

### 5. Maintenance Guidelines

- Test all token changes across light and dark themes
- Validate color contrast ratios before updating color tokens
- Ensure spacing tokens maintain consistent relationships
- Document any breaking changes to tokens

## Integration with CSS

### Direct CSS Usage
```css
.button {
  background-color: var(--color-primary-500);
  padding: var(--spacing-btn-padding-y-base) var(--spacing-btn-padding-x-base);
  border-radius: var(--border-radius-md);
  transition: var(--transition-colors);
}
```

### Tailwind CSS Integration
```javascript
// tailwind.config.js
module.exports = {
  theme: {
    extend: {
      colors: {
        primary: {
          50: 'var(--color-primary-50)',
          100: 'var(--color-primary-100)',
          // ... etc
        }
      },
      spacing: {
        'btn-x-sm': 'var(--spacing-btn-padding-x-sm)',
        // ... etc
      }
    }
  }
}
```

### JavaScript Integration
```javascript
// Get token values in JavaScript
const primaryColor = getComputedStyle(document.documentElement)
  .getPropertyValue('--color-primary-500').trim();

// Set token values dynamically
document.documentElement.style.setProperty('--color-primary-500', newColor);
```

## Token Validation

### Automated Checks
- Color contrast validation
- Token existence verification
- Naming convention compliance
- Cross-browser compatibility

### Manual Review Process
1. Visual design review
2. Accessibility audit
3. Developer experience testing
4. Documentation updates

This comprehensive design token system ensures consistency, maintainability, and accessibility across the entire Pynomaly platform while providing flexibility for customization and theming.
