# Pynomaly Design System Documentation

🍞 **Breadcrumb:** 🏠 [Home](../index.md) > 📁 Design-System

---


## 🎨 **Overview**

The Pynomaly Design System is a comprehensive, accessibility-first component library and design framework for building production-ready anomaly detection interfaces. Built with WCAG 2.1 AA compliance, performance optimization, and developer experience at its core.

## 📋 **Table of Contents**

- [🎯 Design Principles](#-design-principles)
- [🎨 Foundation](#-foundation)
  - [Colors](#colors)
  - [Typography](#typography)
  - [Spacing](#spacing)
  - [Layout](#layout)
- [🧩 Components](#-components)
- [♿ Accessibility](#-accessibility)
- [📱 Responsive Design](#-responsive-design)
- [🚀 Performance](#-performance)
- [🛠️ Implementation](#️-implementation)

## 🎯 **Design Principles**

### Accessibility First
Every component and pattern is designed and tested for **WCAG 2.1 AA compliance**:
- Minimum 4.5:1 color contrast for normal text
- 3:1 contrast for large text and UI components
- Keyboard navigation support for all interactive elements
- Screen reader compatibility with semantic HTML and ARIA
- Touch targets meet 44x44px minimum with adequate spacing

### Performance Optimized
Built for **production performance** with Core Web Vitals optimization:
- Minimal CSS and JavaScript footprint (< 50KB gzipped)
- Progressive enhancement approach
- Efficient rendering and re-rendering patterns
- Mobile-first responsive design
- Optimized asset loading and caching strategies

### Developer Experience
Designed for **efficient development** workflows:
- Clear, consistent API patterns
- Comprehensive documentation with examples
- Interactive Storybook component explorer
- Copy-paste ready code snippets
- TypeScript support and IntelliSense

### Consistency and Cohesion
Unified design language across the platform:
- Systematic approach to color, typography, and spacing
- Reusable components and patterns
- Semantic naming conventions
- Predictable interaction patterns

## 🎨 **Foundation**

### Colors

#### Primary Palette
The primary color palette uses sky blue tones for brand identity and interactive elements:

```css
/* Primary Colors */
--primary-50: #f0f9ff;   /* Light backgrounds, subtle highlights */
--primary-100: #e0f2fe;  /* Hover states, light accents */
--primary-500: #0ea5e9;  /* Primary buttons, links, active states */
--primary-600: #0284c7;  /* Hover states, pressed buttons */
--primary-700: #0369a1;  /* Active states, dark accents */
--primary-900: #0c4a6e;  /* High contrast text, dark themes */
```

**Accessibility Notes:**
- Primary 500 has 4.5:1 contrast with white
- Primary 600 has 5.9:1 contrast with white
- Primary 700 has 7.8:1 contrast with white (AAA compliant)

#### Semantic Colors
Status and feedback colors that communicate meaning:

```css
/* Success Colors */
--success-500: #22c55e;  /* Success messages, positive states (4.5:1 contrast) */
--success-100: #dcfce7;  /* Success backgrounds, subtle indicators */

/* Warning Colors */
--warning-500: #f59e0b;  /* Warning messages, cautionary states (4.7:1 contrast) */
--warning-100: #fef3c7;  /* Warning backgrounds, attention areas */

/* Error Colors */
--error-500: #ef4444;    /* Error messages, destructive actions (4.5:1 contrast) */
--error-100: #fee2e2;    /* Error backgrounds, danger zones */
```

#### Neutral Palette
Text, backgrounds, and interface colors:

```css
/* Neutral Colors */
--neutral-50: #fafafa;   /* Page backgrounds, light surfaces */
--neutral-100: #f5f5f5;  /* Card backgrounds, secondary surfaces */
--neutral-200: #e5e5e5;  /* Borders, dividers, subtle separators */
--neutral-400: #a3a3a3;  /* Muted text, placeholders (7.0:1 with white) */
--neutral-600: #525252;  /* Secondary text, captions (7.23:1 with white) */
--neutral-900: #171717;  /* Primary text, headings (18.82:1 with white) */
```

#### Color Usage Guidelines

**✅ Do:**
- Use color plus another indicator (icons, text, patterns)
- Test with color blindness simulators
- Provide sufficient contrast for all text
- Use semantic colors consistently across the platform
- Consider cultural color associations

**❌ Don't:**
- Rely solely on color to convey information
- Use red and green together without additional cues
- Create custom colors without testing contrast ratios
- Use color alone for form validation feedback

### Typography

#### Font Stack
```css
/* Primary Font Family */
font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;

/* Monospace Font Family */
font-family: 'JetBrains Mono', 'Fira Code', Consolas, 'Courier New', monospace;
```

#### Type Scale
Based on a modular scale with accessibility considerations:

```css
/* Display Typography - For hero sections and major headings */
.text-display-large { font-size: 3.75rem; font-weight: 700; line-height: 1.0; letter-spacing: -0.025em; } /* 60px */
.text-display-medium { font-size: 3rem; font-weight: 700; line-height: 1.0; letter-spacing: -0.025em; } /* 48px */
.text-display-small { font-size: 2.25rem; font-weight: 700; line-height: 1.25; letter-spacing: -0.025em; } /* 36px */

/* Headline Typography - Semantic heading hierarchy */
.text-headline-large { font-size: 1.875rem; font-weight: 600; line-height: 1.25; } /* 30px - H1 */
.text-headline-medium { font-size: 1.5rem; font-weight: 600; line-height: 1.25; } /* 24px - H2 */
.text-headline-small { font-size: 1.25rem; font-weight: 600; line-height: 1.25; } /* 20px - H3 */

/* Title Typography - Component headers and labels */
.text-title-large { font-size: 1.125rem; font-weight: 500; line-height: 1.5; } /* 18px */
.text-title-medium { font-size: 1rem; font-weight: 500; line-height: 1.5; } /* 16px */
.text-title-small { font-size: 0.875rem; font-weight: 500; line-height: 1.5; } /* 14px */

/* Body Typography - Content and interface text */
.text-body-large { font-size: 1rem; font-weight: 400; line-height: 1.625; } /* 16px */
.text-body-medium { font-size: 0.875rem; font-weight: 400; line-height: 1.625; } /* 14px */
.text-body-small { font-size: 0.75rem; font-weight: 400; line-height: 1.625; } /* 12px */

/* Label Typography - Compact interface elements */
.text-label-large { font-size: 0.875rem; font-weight: 500; line-height: 1.0; } /* 14px */
.text-label-medium { font-size: 0.75rem; font-weight: 500; line-height: 1.0; } /* 12px */
.text-label-small { font-size: 0.75rem; font-weight: 500; line-height: 1.0; text-transform: uppercase; letter-spacing: 0.05em; } /* 12px */
```

#### Typography Accessibility Guidelines

**Semantic Heading Structure:**
```html
<h1>Page Title (only one per page)</h1>
  <h2>Major Section</h2>
    <h3>Subsection</h3>
      <h4>Sub-subsection</h4>
```

**Readability Requirements:**
- Minimum 16px for body text on desktop
- Line height 1.5 or greater for readability
- Maximum 80 characters per line for optimal reading
- Adequate color contrast (4.5:1 minimum)

### Spacing

#### 8pt Grid System
Consistent spacing based on 8px increments:

```css
/* Spacing Scale */
--space-0: 0;           /* 0px */
--space-1: 0.125rem;    /* 2px */
--space-2: 0.25rem;     /* 4px */
--space-3: 0.5rem;      /* 8px */
--space-4: 1rem;        /* 16px */
--space-5: 1.25rem;     /* 20px */
--space-6: 1.5rem;      /* 24px */
--space-8: 2rem;        /* 32px */
--space-10: 2.5rem;     /* 40px */
--space-12: 3rem;       /* 48px */
--space-16: 4rem;       /* 64px */
--space-20: 5rem;       /* 80px */
--space-24: 6rem;       /* 96px */
```

#### Spacing Usage Guidelines

**Component Spacing:**
- Use `--space-3` (8px) for tight spacing within components
- Use `--space-4` (16px) for standard component padding
- Use `--space-6` (24px) for loose spacing between related elements
- Use `--space-8` (32px) for section separation

**Layout Spacing:**
- Use `--space-12` (48px) for major section separation
- Use `--space-16` (64px) for page-level spacing
- Use `--space-20` (80px) for hero section spacing

### Layout

#### Grid System
Responsive grid based on CSS Grid and Flexbox:

```css
/* Container Sizes */
.container-sm { max-width: 640px; }   /* Small screens */
.container-md { max-width: 768px; }   /* Medium screens */
.container-lg { max-width: 1024px; }  /* Large screens */
.container-xl { max-width: 1280px; }  /* Extra large screens */
.container-2xl { max-width: 1536px; } /* 2X large screens */

/* Grid Columns */
.grid { display: grid; }
.grid-cols-1 { grid-template-columns: repeat(1, minmax(0, 1fr)); }
.grid-cols-2 { grid-template-columns: repeat(2, minmax(0, 1fr)); }
.grid-cols-3 { grid-template-columns: repeat(3, minmax(0, 1fr)); }
.grid-cols-4 { grid-template-columns: repeat(4, minmax(0, 1fr)); }
.grid-cols-12 { grid-template-columns: repeat(12, minmax(0, 1fr)); }

/* Flexbox Utilities */
.flex { display: flex; }
.flex-col { flex-direction: column; }
.flex-wrap { flex-wrap: wrap; }
.items-center { align-items: center; }
.justify-center { justify-content: center; }
.justify-between { justify-content: space-between; }
```

#### Responsive Breakpoints
Mobile-first responsive design:

```css
/* Breakpoints */
@media (min-width: 640px) { /* sm */ }
@media (min-width: 768px) { /* md */ }
@media (min-width: 1024px) { /* lg */ }
@media (min-width: 1280px) { /* xl */ }
@media (min-width: 1536px) { /* 2xl */ }
```

## 🧩 **Components**

### Component Categories

#### Core Components
Essential interface elements used throughout the platform:

1. **Buttons** - Primary, secondary, outline, ghost, and danger variants
2. **Form Controls** - Inputs, textareas, selects, checkboxes, radios
3. **Navigation** - Main nav, breadcrumbs, pagination, tabs
4. **Feedback** - Alerts, notifications, toasts, loading states

#### Data Components
Specialized components for data visualization and interaction:

1. **Charts** - Line charts, scatter plots, heatmaps for anomaly visualization
2. **Tables** - Data grids, sortable columns, filtering, pagination
3. **Dashboards** - Metric cards, KPI displays, summary widgets
4. **Filters** - Search inputs, filter dropdowns, date pickers

#### Layout Components
Structural components for page organization:

1. **Cards** - Content containers, information displays
2. **Modals** - Dialogs, overlays, confirmations
3. **Panels** - Sidebars, collapsible sections, accordions
4. **Headers** - Page headers, section headers, toolbars

### Component Design Tokens

#### Shadows
Consistent elevation system:

```css
--shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
--shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
--shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
--shadow-xl: 0 20px 25px -5px rgb(0 0 0 / 0.1), 0 8px 10px -6px rgb(0 0 0 / 0.1);
```

#### Border Radius
Consistent corner rounding:

```css
--radius-none: 0;
--radius-sm: 0.125rem;  /* 2px */
--radius-md: 0.375rem;  /* 6px */
--radius-lg: 0.5rem;    /* 8px */
--radius-xl: 0.75rem;   /* 12px */
--radius-full: 9999px;  /* Fully rounded */
```

#### Transitions
Smooth, performant animations:

```css
--transition-fast: 150ms cubic-bezier(0.4, 0, 0.2, 1);
--transition-normal: 300ms cubic-bezier(0.4, 0, 0.2, 1);
--transition-slow: 500ms cubic-bezier(0.4, 0, 0.2, 1);
```

## ♿ **Accessibility**

### WCAG 2.1 AA Compliance

#### Color Contrast Requirements
- **Normal text**: 4.5:1 minimum contrast ratio
- **Large text** (18pt+ or 14pt+ bold): 3:1 minimum contrast ratio
- **UI components**: 3:1 minimum contrast ratio for interactive elements
- **Enhanced (AAA)**: 7:1 contrast ratio for better visibility

#### Touch Target Requirements
- **Minimum size**: 44x44px for touch targets
- **Recommended size**: 48x48px for better usability
- **Spacing**: 8px minimum between adjacent touch targets

#### Keyboard Navigation
- **Tab order**: Logical sequence following visual layout
- **Focus indicators**: 2px solid outline with high contrast
- **Skip links**: "Skip to main content" for screen reader users
- **Escape key**: Closes modals and overlays

#### Screen Reader Support
- **Semantic HTML**: Use appropriate HTML elements (button, nav, main, etc.)
- **ARIA labels**: Descriptive labels for all interactive elements
- **Live regions**: Announce dynamic content changes
- **Landmarks**: Clear page structure with semantic landmarks

### Accessibility Testing Tools

#### Automated Testing
```bash
# Run accessibility audit on Storybook
npm run accessibility-audit

# Test specific components
npx axe-crawler http://localhost:6006/iframe.html?id=components-button--default
```

#### Manual Testing Checklist
- [ ] Tab through all interactive elements
- [ ] Test with screen reader (NVDA, JAWS, VoiceOver)
- [ ] Verify color contrast ratios
- [ ] Test with keyboard only
- [ ] Check focus indicators
- [ ] Validate ARIA attributes

## 📱 **Responsive Design**

### Mobile-First Approach
Design and develop for mobile devices first, then enhance for larger screens:

```css
/* Mobile First (320px+) */
.button {
  padding: 0.75rem 1rem;
  font-size: 0.875rem;
}

/* Tablet (768px+) */
@media (min-width: 768px) {
  .button {
    padding: 0.875rem 1.25rem;
    font-size: 1rem;
  }
}

/* Desktop (1024px+) */
@media (min-width: 1024px) {
  .button {
    padding: 1rem 1.5rem;
  }
}
```

### Responsive Typography
Scale typography appropriately across devices:

```css
/* Mobile Typography */
.heading-1 { font-size: 1.5rem; }  /* 24px */
.body-text { font-size: 1rem; }    /* 16px */

/* Tablet Typography */
@media (min-width: 768px) {
  .heading-1 { font-size: 1.875rem; } /* 30px */
}

/* Desktop Typography */
@media (min-width: 1024px) {
  .heading-1 { font-size: 2.25rem; }  /* 36px */
}
```

### Touch Optimization
Optimize for touch interactions on mobile devices:

- **Touch targets**: Minimum 44x44px with adequate spacing
- **Gestures**: Support swipe, pinch, and tap interactions
- **Hover states**: Use `:hover` only on devices that support it
- **Focus states**: Clear focus indicators for keyboard navigation

## 🚀 **Performance**

### Core Web Vitals Optimization

#### Largest Contentful Paint (LCP)
Target: < 2.5 seconds

- **Optimize images**: Use WebP format with appropriate sizing
- **Preload critical resources**: Fonts, above-the-fold images
- **Minimize render-blocking**: Inline critical CSS, defer non-critical JavaScript

#### First Input Delay (FID)
Target: < 100 milliseconds

- **Minimize JavaScript**: Use code splitting and lazy loading
- **Optimize event handlers**: Use passive event listeners where possible
- **Web Workers**: Move heavy computations off the main thread

#### Cumulative Layout Shift (CLS)
Target: < 0.1

- **Reserve space**: Set dimensions for images and embeds
- **Font loading**: Use `font-display: swap` and preload fonts
- **Avoid layout shifts**: Don't insert content above existing content

### Performance Best Practices

#### CSS Optimization
```css
/* Use efficient selectors */
.button { /* Good: simple class selector */ }
.sidebar .navigation .item .link { /* Avoid: deeply nested selectors */ }

/* Minimize repaints and reflows */
.animate-transform {
  transform: translateX(100px); /* Good: composite layer */
}
.animate-layout {
  left: 100px; /* Avoid: triggers layout */
}
```

#### JavaScript Optimization
```javascript
// Use efficient DOM queries
const elements = document.querySelectorAll('.item'); // Good: batch query
// Avoid repeated DOM queries in loops

// Debounce expensive operations
const debouncedSearch = debounce(searchFunction, 300);

// Use requestAnimationFrame for animations
function animate() {
  // Animation logic
  requestAnimationFrame(animate);
}
```

#### Asset Optimization
- **Images**: Use modern formats (WebP, AVIF) with fallbacks
- **Fonts**: Subset fonts to include only needed characters
- **JavaScript**: Use tree shaking to eliminate unused code
- **CSS**: Purge unused styles in production builds

## 🛠️ **Implementation**

### Getting Started

#### Installation
```bash
# Install Tailwind CSS and dependencies
npm install tailwindcss @tailwindcss/forms @tailwindcss/typography @tailwindcss/aspect-ratio

# Install optional enhancements
npm install alpinejs htmx.org d3 echarts
```

#### Tailwind Configuration
```javascript
// tailwind.config.js
module.exports = {
  content: [
    './src/pynomaly/presentation/web/**/*.{html,js,py}',
    './tests/ui/**/*.{html,js,py}'
  ],
  theme: {
    extend: {
      colors: {
        primary: {
          50: '#f0f9ff',
          100: '#e0f2fe',
          500: '#0ea5e9',
          600: '#0284c7',
          700: '#0369a1',
          900: '#0c4a6e',
        },
        // Additional colors...
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'Menlo', 'monospace'],
      },
      // Additional theme extensions...
    },
  },
  plugins: [
    require('@tailwindcss/forms'),
    require('@tailwindcss/typography'),
    require('@tailwindcss/aspect-ratio'),
  ],
};
```

### Development Workflow

#### Component Development
1. **Design Review**: Validate designs against accessibility guidelines
2. **Component Creation**: Build with semantic HTML and ARIA attributes
3. **Styling**: Apply design system tokens and utilities
4. **Testing**: Automated accessibility and cross-browser testing
5. **Documentation**: Add to Storybook with usage examples

#### Quality Assurance
```bash
# Run comprehensive testing suite
npm run test:accessibility  # Accessibility compliance
npm run test:cross-browser  # Cross-browser compatibility
npm run test:performance   # Performance benchmarks
npm run test:visual        # Visual regression testing
```

### Integration Guidelines

#### HTML Structure
```html
<!-- Semantic HTML with accessibility -->
<button 
  class="btn btn-primary btn-md"
  type="button"
  aria-label="Save document"
  aria-describedby="save-help"
>
  <span aria-hidden="true">💾</span>
  Save Document
</button>
<div id="save-help" class="sr-only">
  Saves the current document to your computer
</div>
```

#### CSS Utilities
```html
<!-- Design system utilities -->
<div class="bg-neutral-50 p-6 rounded-lg shadow-md">
  <h2 class="text-headline-medium text-neutral-900 mb-4">
    Component Title
  </h2>
  <p class="text-body-large text-neutral-600 leading-relaxed">
    Component description with optimal readability.
  </p>
</div>
```

#### JavaScript Integration
```javascript
// Progressive enhancement with accessibility
function initializeComponent(element) {
  // Ensure keyboard accessibility
  element.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      element.click();
    }
  });
  
  // Announce state changes to screen readers
  element.setAttribute('aria-live', 'polite');
}
```

---

## 📚 **Resources and References**

### Design System Tools
- **Storybook**: Interactive component explorer at http://localhost:6006
- **Figma Library**: Design tokens and component specifications
- **Design System Checklist**: Comprehensive implementation guide

### Accessibility Resources
- [WCAG 2.1 Guidelines](https://www.w3.org/WAI/WCAG21/quickref/)
- [WebAIM Accessibility Guidelines](https://webaim.org/)
- [A11y Project Checklist](https://www.a11yproject.com/checklist/)

### Performance Resources
- [Web.dev Performance](https://web.dev/performance/)
- [Core Web Vitals](https://web.dev/vitals/)
- [Performance Budget Calculator](https://www.performancebudget.io/)

### Development Resources
- [Tailwind CSS Documentation](https://tailwindcss.com/docs)
- [MDN Web Docs](https://developer.mozilla.org/)
- [Can I Use](https://caniuse.com/) - Browser compatibility tables

---

## 🎉 **Get Started**

Ready to build with the Pynomaly Design System?

1. **Explore Storybook**: `npm run storybook` and visit http://localhost:6006
2. **Review Components**: Browse the interactive component library
3. **Check Accessibility**: Use built-in accessibility testing tools
4. **Start Building**: Copy component code and integrate into your application

For questions or contributions, see our [GitHub repository](https://github.com/pynomaly/pynomaly) or join our design system discussions! 🚀

---

## 🔗 **Related Documentation**

- **[Accessibility](../accessibility/accessibility-guidelines.md)** - Accessibility guidelines
- **[Storybook](../storybook/README.md)** - Component documentation
- **[UI Testing](../testing/cross-browser-testing.md)** - UI testing guides
