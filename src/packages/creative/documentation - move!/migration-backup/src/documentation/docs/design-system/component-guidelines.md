# Pynomaly Component Guidelines

🍞 **Breadcrumb:** 🏠 [Home](../index.md) > 📁 Design-System

---


## Overview

This document provides comprehensive guidelines for developing, maintaining, and using components within the Pynomaly design system. These guidelines ensure consistency, accessibility, and quality across all UI components.

## Core Principles

### 1. Accessibility First
Every component must meet WCAG 2.1 AA standards from the initial design phase.

**Requirements:**
- **Semantic HTML**: Use proper HTML elements for their intended purpose
- **ARIA Support**: Implement appropriate ARIA labels, roles, and properties
- **Keyboard Navigation**: Full keyboard accessibility for all interactive elements
- **Color Contrast**: Minimum 4.5:1 contrast ratio for normal text, 3:1 for large text
- **Screen Reader Support**: Test with NVDA, JAWS, VoiceOver, and TalkBack
- **Focus Management**: Clear focus indicators and logical tab order

### 2. Mobile-First Responsive Design
Design and build for mobile devices first, then enhance for larger screens.

**Requirements:**
- **Touch Targets**: Minimum 44×44px for all interactive elements
- **Responsive Breakpoints**: Mobile (320px+), Tablet (768px+), Desktop (1024px+)
- **Flexible Layouts**: Use relative units and flexible grid systems
- **Progressive Enhancement**: Core functionality available on all devices
- **Performance**: Optimize for slower mobile connections

### 3. Performance Optimization
Build lightweight, fast-loading components with minimal impact on Core Web Vitals.

**Requirements:**
- **Bundle Size**: Minimize JavaScript and CSS footprint
- **Lazy Loading**: Load components only when needed
- **Critical Path**: Prioritize above-the-fold content
- **Caching**: Implement appropriate caching strategies
- **Core Web Vitals**: LCP <2.5s, FID <100ms, CLS <0.1

### 4. Consistency and Predictability
Maintain visual and functional consistency across all components.

**Requirements:**
- **Design Tokens**: Use established design tokens for all styling
- **Naming Conventions**: Follow consistent naming patterns
- **Interaction Patterns**: Use familiar interaction models
- **Visual Hierarchy**: Maintain consistent visual relationships
- **Error Handling**: Implement standardized error states

## Component Architecture

### Base Component Structure

Every component should follow this structure:

```html
<!-- Component wrapper with semantic meaning -->
<div class="component-name"
     role="[appropriate-role]"
     aria-label="[descriptive-label]"
     data-component="component-name">

  <!-- Component header (if applicable) -->
  <header class="component-name__header">
    <h2 class="component-name__title">Component Title</h2>
  </header>

  <!-- Component body/content -->
  <div class="component-name__body">
    <!-- Main component content -->
  </div>

  <!-- Component footer (if applicable) -->
  <footer class="component-name__footer">
    <!-- Action buttons, metadata, etc. -->
  </footer>
</div>
```

### CSS Architecture

Use BEM (Block Element Modifier) methodology for CSS class naming:

```css
/* Block */
.component-name {
  /* Base styles */
}

/* Element */
.component-name__element {
  /* Element styles */
}

/* Modifier */
.component-name--modifier {
  /* Modifier styles */
}

/* State */
.component-name.is-active {
  /* State styles */
}
```

### Design Token Usage

Always use design tokens instead of hardcoded values:

```css
.button {
  /* Good - Uses design tokens */
  background-color: var(--color-primary-500);
  padding: var(--spacing-btn-padding-y-base) var(--spacing-btn-padding-x-base);
  border-radius: var(--border-radius-md);
  transition: var(--transition-colors);
}

.button {
  /* Bad - Hardcoded values */
  background-color: #3b82f6;
  padding: 8px 16px;
  border-radius: 6px;
  transition: all 0.2s ease;
}
```

## Component Categories

### 1. Foundation Components

**Purpose**: Basic building blocks and design primitives

**Examples**: Colors, Typography, Spacing, Icons, Layouts

**Guidelines:**
- Minimal functionality, maximum reusability
- No business logic or complex interactions
- Highly customizable through design tokens
- Excellent documentation with visual examples

### 2. Basic UI Components

**Purpose**: Standard interface elements for user interaction

**Examples**: Buttons, Form Controls, Cards, Badges, Alerts

**Guidelines:**
- Single responsibility principle
- Consistent API across similar components
- Support for all necessary states (default, hover, focus, disabled, loading)
- Comprehensive prop validation and error handling

### 3. Complex Components

**Purpose**: Sophisticated interface elements with internal state and logic

**Examples**: Data Tables, Charts, Modals, Dropdowns, Date Pickers

**Guidelines:**
- Well-defined public API
- Internal state management
- Event handling and callbacks
- Performance optimization for large datasets
- Comprehensive testing coverage

### 4. Layout Components

**Purpose**: Structural elements for organizing content and other components

**Examples**: Grid Systems, Containers, Panels, Navigation

**Guidelines:**
- Flexible and responsive design
- Minimal visual styling (rely on content components)
- Semantic HTML structure
- Support for nested layouts

### 5. Domain-Specific Components

**Purpose**: Specialized components for anomaly detection workflows

**Examples**: Anomaly Charts, Detection Status, Algorithm Selector, Dataset Uploader

**Guidelines:**
- Deep integration with Pynomaly data models
- Optimized for anomaly detection use cases
- Comprehensive error handling for data issues
- Real-time update capabilities

## Component Development Process

### 1. Planning Phase

Before coding, complete these steps:

**Design Review:**
- [ ] UI/UX design approved
- [ ] Accessibility considerations documented
- [ ] Responsive behavior defined
- [ ] Interactive states specified

**Technical Planning:**
- [ ] Component API defined
- [ ] Dependencies identified
- [ ] Performance requirements set
- [ ] Testing strategy planned

### 2. Implementation Phase

Follow this development sequence:

**1. HTML Structure**
```html
<!-- Start with semantic HTML -->
<button type="button" class="btn" aria-describedby="btn-help">
  <span class="btn__icon" aria-hidden="true">
    <!-- Icon content -->
  </span>
  <span class="btn__text">
    Button Text
  </span>
</button>
<div id="btn-help" class="sr-only">
  Additional context for screen readers
</div>
```

**2. CSS Styling**
```css
.btn {
  /* Base styles using design tokens */
  display: inline-flex;
  align-items: center;
  justify-content: center;
  padding: var(--spacing-btn-padding-y-base) var(--spacing-btn-padding-x-base);
  background-color: var(--color-primary-500);
  color: var(--color-text-inverse);
  border: 1px solid var(--color-primary-500);
  border-radius: var(--border-radius-md);
  font-family: var(--font-family-sans);
  font-size: var(--font-size-base);
  font-weight: var(--font-weight-medium);
  line-height: var(--line-height-none);
  text-decoration: none;
  cursor: pointer;
  transition: var(--transition-colors);
}

.btn:hover:not(:disabled) {
  background-color: var(--color-primary-600);
  border-color: var(--color-primary-600);
}

.btn:focus-visible {
  outline: none;
  box-shadow: var(--shadow-focus);
}

.btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
```

**3. JavaScript Behavior**
```javascript
class Button {
  constructor(element) {
    this.element = element;
    this.init();
  }

  init() {
    this.bindEvents();
  }

  bindEvents() {
    this.element.addEventListener('click', this.handleClick.bind(this));
    this.element.addEventListener('keydown', this.handleKeyDown.bind(this));
  }

  handleClick(event) {
    if (this.element.disabled) {
      event.preventDefault();
      return;
    }
    // Handle click logic
  }

  handleKeyDown(event) {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      this.handleClick(event);
    }
  }
}
```

### 3. Testing Phase

Implement comprehensive testing:

**Unit Tests:**
```javascript
describe('Button Component', () => {
  it('should render with correct attributes', () => {
    const button = createButton({ text: 'Test' });
    expect(button.getAttribute('type')).toBe('button');
    expect(button.textContent).toBe('Test');
  });

  it('should handle click events', () => {
    const onClick = jest.fn();
    const button = createButton({ onClick });
    button.click();
    expect(onClick).toHaveBeenCalled();
  });
});
```

**Accessibility Tests:**
```javascript
describe('Button Accessibility', () => {
  it('should meet WCAG standards', async () => {
    const button = createButton({ text: 'Test' });
    const results = await axe(button);
    expect(results).toHaveNoViolations();
  });

  it('should be keyboard accessible', () => {
    const button = createButton({ text: 'Test' });
    button.focus();
    expect(document.activeElement).toBe(button);
  });
});
```

**Visual Regression Tests:**
```javascript
describe('Button Visual Tests', () => {
  it('should match visual snapshot', async () => {
    const button = createButton({ text: 'Test' });
    await expect(button).toMatchSnapshot();
  });

  it('should handle different states', async () => {
    const states = ['default', 'hover', 'focus', 'disabled'];
    for (const state of states) {
      const button = createButton({ state });
      await expect(button).toMatchSnapshot(`button-${state}`);
    }
  });
});
```

### 4. Documentation Phase

Create comprehensive documentation:

**Storybook Story:**
```javascript
export default {
  title: 'Components/Button',
  component: Button,
  tags: ['autodocs'],
  argTypes: {
    variant: {
      control: { type: 'select' },
      options: ['primary', 'secondary', 'success', 'warning', 'danger'],
    },
    size: {
      control: { type: 'select' },
      options: ['sm', 'base', 'lg'],
    },
  },
};

export const Primary = {
  args: {
    text: 'Primary Button',
    variant: 'primary',
  },
};
```

**API Documentation:**
```markdown
## Button

### Props

| Prop | Type | Default | Description |
|------|------|---------|-------------|
| `variant` | `string` | `'primary'` | Visual style variant |
| `size` | `string` | `'base'` | Button size |
| `disabled` | `boolean` | `false` | Whether button is disabled |
| `loading` | `boolean` | `false` | Whether button shows loading state |

### Accessibility

- Uses semantic `<button>` element
- Supports keyboard navigation (Enter/Space)
- Includes proper ARIA attributes
- Meets WCAG 2.1 AA contrast requirements
```

## Component States

### Standard States

All interactive components must support these states:

**1. Default State**
- Normal appearance and behavior
- Default styling and interactions

**2. Hover State**
- Visual feedback on mouse hover
- Should not rely solely on color changes
- Smooth transitions between states

**3. Focus State**
- Clear focus indicator for keyboard navigation
- High contrast outline or shadow
- Meets WCAG focus visibility requirements

**4. Active State**
- Visual feedback during interaction
- Pressed/clicked appearance
- Brief transition back to default

**5. Disabled State**
- Reduced opacity or grayed out appearance
- No interactive behavior
- Proper ARIA attributes (`aria-disabled="true"`)

**6. Loading State**
- Loading indicator (spinner, skeleton, etc.)
- Disabled interactions during loading
- Clear loading progress when possible

**7. Error State**
- Clear error indication
- Descriptive error messages
- Actionable recovery options

### Anomaly-Specific States

For anomaly detection components:

**1. Normal State**
- Indicates normal/expected data
- Green color coding
- Calm visual treatment

**2. Anomaly State**
- Indicates detected anomalies
- Red color coding with proper contrast
- Alert visual treatment

**3. Warning State**
- Indicates potential issues
- Yellow/orange color coding
- Cautionary visual treatment

**4. Unknown State**
- Indicates insufficient data
- Gray color coding
- Neutral visual treatment

## Responsive Design Guidelines

### Breakpoint Strategy

Use these breakpoints for responsive design:

```css
/* Mobile First */
.component {
  /* Mobile styles (320px+) */
}

@media (min-width: 768px) {
  .component {
    /* Tablet styles */
  }
}

@media (min-width: 1024px) {
  .component {
    /* Desktop styles */
  }
}

@media (min-width: 1280px) {
  .component {
    /* Large desktop styles */
  }
}
```

### Touch Target Guidelines

- **Minimum size**: 44×44px for all touch targets
- **Spacing**: Minimum 8px between adjacent touch targets
- **Visual feedback**: Clear hover/active states
- **Error tolerance**: Forgiving interaction areas

### Content Strategy

- **Progressive disclosure**: Show essential information first
- **Truncation**: Graceful text truncation with tooltips
- **Stacking**: Logical content stacking on smaller screens
- **Priority**: Most important content remains visible

## Error Handling

### Error States

**1. Validation Errors**
```html
<div class="form-field">
  <label for="email" class="form-label">Email</label>
  <input
    type="email"
    id="email"
    class="form-input form-input--error"
    aria-invalid="true"
    aria-describedby="email-error"
  >
  <div id="email-error" class="form-error" role="alert">
    Please enter a valid email address
  </div>
</div>
```

**2. Loading Errors**
```html
<div class="component-error" role="alert">
  <h3 class="component-error__title">Failed to Load Data</h3>
  <p class="component-error__message">
    Unable to fetch anomaly detection results. Please check your connection and try again.
  </p>
  <button type="button" class="btn btn--secondary" onclick="retry()">
    Try Again
  </button>
</div>
```

**3. Empty States**
```html
<div class="empty-state">
  <div class="empty-state__icon" aria-hidden="true">
    <!-- Empty state illustration -->
  </div>
  <h3 class="empty-state__title">No Data Available</h3>
  <p class="empty-state__description">
    Upload a dataset to start detecting anomalies.
  </p>
  <button type="button" class="btn btn--primary">
    Upload Dataset
  </button>
</div>
```

### Error Messages

**Guidelines for Error Messages:**
- **Clear and specific**: Explain exactly what went wrong
- **Actionable**: Provide steps to resolve the issue
- **Polite tone**: Avoid technical jargon and blame
- **Accessible**: Use appropriate ARIA roles and live regions

**Examples:**

```javascript
// Good error messages
const errorMessages = {
  validation: {
    required: 'This field is required.',
    email: 'Please enter a valid email address.',
    minLength: 'Password must be at least 8 characters long.',
  },
  network: {
    timeout: 'Request timed out. Please check your connection and try again.',
    offline: 'You appear to be offline. Please check your connection.',
    server: 'Server error occurred. Please try again in a few minutes.',
  },
  data: {
    notFound: 'The requested data could not be found.',
    corrupt: 'The data file appears to be corrupted. Please upload a valid file.',
    tooLarge: 'File is too large. Maximum size is 100MB.',
  }
};
```

## Performance Guidelines

### Code Splitting

Split components for optimal loading:

```javascript
// Dynamic imports for large components
const LazyChart = lazy(() => import('./AnomalyChart'));
const LazyDataTable = lazy(() => import('./DataTable'));

// Use with Suspense
function Dashboard() {
  return (
    <div>
      <Suspense fallback={<ChartSkeleton />}>
        <LazyChart data={data} />
      </Suspense>
    </div>
  );
}
```

### Memory Management

- **Event listeners**: Always remove event listeners on component unmount
- **Timers**: Clear intervals and timeouts
- **Subscriptions**: Unsubscribe from data streams
- **Large datasets**: Implement virtualization for large lists

### Bundle Optimization

- **Tree shaking**: Export only used functions
- **Minification**: Minimize CSS and JavaScript
- **Compression**: Use gzip/brotli compression
- **Caching**: Implement proper cache headers

## Quality Checklist

Before shipping any component, verify:

### Functionality
- [ ] All features work as specified
- [ ] Edge cases handled appropriately
- [ ] Error states implemented
- [ ] Loading states implemented
- [ ] Empty states implemented

### Accessibility
- [ ] WCAG 2.1 AA compliant
- [ ] Keyboard navigation works
- [ ] Screen reader compatible
- [ ] Color contrast verified
- [ ] Focus management correct
- [ ] ARIA attributes proper

### Performance
- [ ] Bundle size minimized
- [ ] Core Web Vitals optimized
- [ ] Memory leaks prevented
- [ ] Animations smooth (60fps)
- [ ] Large datasets handled efficiently

### Design
- [ ] Design tokens used consistently
- [ ] Responsive across all breakpoints
- [ ] Visual hierarchy clear
- [ ] Brand guidelines followed
- [ ] Animation timing appropriate

### Testing
- [ ] Unit tests pass (>90% coverage)
- [ ] Integration tests pass
- [ ] Accessibility tests pass
- [ ] Visual regression tests pass
- [ ] Cross-browser testing complete

### Documentation
- [ ] Storybook story created
- [ ] API documentation complete
- [ ] Usage examples provided
- [ ] Accessibility notes included
- [ ] Migration guide (if needed)

## Maintenance Guidelines

### Regular Reviews

**Monthly:**
- Accessibility audit
- Performance analysis
- Browser compatibility check
- Dependency updates

**Quarterly:**
- Design system alignment
- Usage analytics review
- User feedback analysis
- API consistency review

### Version Management

Follow semantic versioning for component changes:

- **Major (1.0.0)**: Breaking API changes
- **Minor (0.1.0)**: New features, backwards compatible
- **Patch (0.0.1)**: Bug fixes, internal improvements

### Deprecation Process

When deprecating components:

1. **Announce**: Document deprecation with timeline
2. **Alternative**: Provide migration path to new component
3. **Support**: Maintain for at least 2 minor versions
4. **Remove**: Remove after sufficient notice period

This comprehensive component guideline ensures consistent, accessible, and high-quality components across the Pynomaly design system.
