# Pynomaly Storybook Component Explorer

🍞 **Breadcrumb:** 🏠 [Home](../index.md) > 📁 Storybook

---


## 🎨 **Overview**

The Pynomaly Storybook provides comprehensive, interactive documentation for our design system and UI component library. Built with accessibility-first principles and production-ready components.

## 🚀 **Getting Started**

### Prerequisites
- Node.js 18+ and npm 9+
- Python 3.11+ with Pynomaly installed

### Installation and Setup

```bash
# Install dependencies
npm install

# Start Storybook development server
npm run storybook

# Start with CSS watching (recommended for development)
npm run dev:storybook

# Build static Storybook for deployment
npm run build-storybook

# Serve built Storybook
npm run storybook:serve
```

### Access Points

- **Development**: http://localhost:6006
- **Static Build**: http://localhost:6007 (after build)
- **Production**: https://pynomaly.github.io/storybook (deployed)

## 📋 **Component Library Structure**

### Foundation
- **Colors** - WCAG 2.1 AA compliant color system
- **Typography** - Accessible text hierarchy and scales
- **Spacing** - 8pt grid system for consistent layouts
- **Icons** - Comprehensive iconography library

### Core Components
- **Buttons** - Primary, secondary, and specialized variants
- **Forms** - Input fields with validation and accessibility
- **Navigation** - Main navigation and breadcrumb patterns
- **Feedback** - Alerts, notifications, and status indicators

### Data Visualization
- **Charts** - Anomaly detection visualization components
- **Tables** - Data grid and table patterns
- **Dashboards** - Analytics and monitoring layouts
- **Filters** - Data filtering and search interfaces

### Layout Components
- **Cards** - Content containers and information displays
- **Modals** - Dialog and overlay patterns
- **Panels** - Collapsible and sidebar layouts
- **Grids** - Responsive layout systems

## ♿ **Accessibility Features**

### WCAG 2.1 AA Compliance
- **Color Contrast**: Minimum 4.5:1 ratio for normal text
- **Touch Targets**: 44x44px minimum with adequate spacing
- **Keyboard Navigation**: Full keyboard accessibility
- **Screen Reader Support**: Semantic HTML and ARIA labels
- **Focus Management**: Clear focus indicators and logical order

### Testing Tools
- **Automated Accessibility Audit**: Built-in axe-core integration
- **Manual Testing Guidelines**: Step-by-step accessibility checks
- **Cross-browser Compatibility**: Chrome, Firefox, Safari, Edge support
- **Responsive Testing**: Mobile-first design validation

### Accessibility Addon Features
- Real-time WCAG violation detection
- Color contrast analysis
- Keyboard navigation testing
- Screen reader simulation
- Focus order validation

## 📱 **Responsive Design**

### Viewport Testing
- **Mobile**: 375px (iPhone SE)
- **Tablet**: 768px (iPad)
- **Desktop**: 1280px (Standard)
- **Wide**: 1920px (Large monitors)

### Mobile-First Approach
- Progressive enhancement strategy
- Touch-friendly interface design
- Optimized performance on mobile devices
- Accessible across all device types

## 🎛️ **Interactive Controls**

### Component Testing
- **Property Editor**: Real-time component customization
- **State Testing**: Toggle different component states
- **Variant Explorer**: Test all available component variants
- **Content Variation**: Test with different text lengths and content

### Development Tools
- **Code Examples**: Copy-paste ready HTML/CSS
- **API Documentation**: Complete component property reference
- **Design Tokens**: Access to color, spacing, and typography values
- **Usage Guidelines**: Best practices and implementation notes

## 🔍 **Visual Testing**

### Visual Regression Testing
```bash
# Run visual tests with Percy
npm run test:visual:storybook

# Compare visual changes
# Results available in Percy dashboard
```

### Cross-Browser Testing
```bash
# Test across multiple browsers
npm run test:cross-browser

# Specific browser testing
npx playwright test --project=chromium
npx playwright test --project=firefox
npx playwright test --project=webkit
```

## 📊 **Performance Monitoring**

### Bundle Analysis
```bash
# Analyze Storybook bundle size
npm run analyze-bundle

# Performance audit
npm run lighthouse

# Core Web Vitals monitoring
npm run test:performance
```

### Performance Features
- **Optimized Loading**: Code splitting and lazy loading
- **Minimal Dependencies**: Lightweight component library
- **Efficient Rendering**: Optimized for smooth interactions
- **Core Web Vitals**: LCP, FID, CLS optimization

## 🛠️ **Development Workflow**

### Adding New Components

1. **Create Component Story**
   ```javascript
   // stories/ComponentName.stories.js
   export default {
     title: 'Components/ComponentName',
     argTypes: {
       // Define component properties
     }
   };
   ```

2. **Implement Component Template**
   ```javascript
   const ComponentTemplate = ({ prop1, prop2 }) => {
     return `<div class="component">${content}</div>`;
   };
   ```

3. **Add Accessibility Documentation**
   ```javascript
   Component.parameters = {
     docs: {
       description: {
         story: 'Accessibility considerations and usage guidelines'
       }
     }
   };
   ```

4. **Test Component**
   ```bash
   # View in Storybook
   npm run storybook
   
   # Run accessibility tests
   npm run accessibility-audit
   ```

### Story Organization
- **Introduction** - Getting started and overview
- **Foundation** - Design tokens and principles
- **Components** - Interactive component library
- **Patterns** - Complex UI patterns and layouts
- **Guidelines** - Accessibility and best practices

### Code Standards
- **TypeScript Support**: Full type safety for component props
- **ESLint Integration**: Code quality enforcement
- **Prettier Formatting**: Consistent code formatting
- **Accessibility Linting**: Automated accessibility checks

## 📚 **Documentation Standards**

### Story Documentation
Each component story includes:

- **Overview**: Component purpose and use cases
- **API Documentation**: Complete property reference
- **Accessibility Guidelines**: WCAG compliance notes
- **Code Examples**: Ready-to-use implementation
- **Design Tokens**: Related color, spacing, typography
- **Best Practices**: Usage recommendations and warnings

### MDX Documentation
```mdx
import { Meta, Story } from '@storybook/addon-docs';

<Meta title="Documentation/ComponentName" />

# Component Name

Description of the component and its purpose.

## Usage

<Story id="components-componentname--default" />
```

## 🔧 **Configuration**

### Storybook Configuration
- **Main Config**: `.storybook/main.js` - Core Storybook setup
- **Preview Config**: `.storybook/preview.js` - Global decorators and parameters
- **Manager Config**: `.storybook/manager.js` - UI customization

### Addon Configuration
- **Accessibility**: Real-time WCAG validation
- **Viewport**: Responsive design testing
- **Controls**: Interactive property editing
- **Docs**: Automated documentation generation
- **Backgrounds**: Design system color testing

### Theme Customization
- **Pynomaly Brand Colors**: Custom theme matching platform design
- **Typography**: Inter and JetBrains Mono font integration
- **Component Styling**: Tailwind CSS integration
- **Dark Mode Support**: Theme switching capabilities

## 🧪 **Testing Integration**

### Automated Testing
```bash
# Component interaction testing
npm run test:interactions

# Accessibility compliance testing
npm run test:accessibility

# Visual regression testing
npm run test:visual

# Performance testing
npm run test:performance
```

### Manual Testing Checklist
- [ ] Keyboard navigation works correctly
- [ ] Screen reader announces content properly
- [ ] Color contrast meets WCAG standards
- [ ] Touch targets are adequately sized
- [ ] Component responds to all prop changes
- [ ] Error states are handled gracefully

## 📈 **Continuous Integration**

### GitHub Actions Integration
```yaml
# .github/workflows/storybook.yml
name: Storybook Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Install dependencies
        run: npm ci
      - name: Build Storybook
        run: npm run build-storybook
      - name: Run visual tests
        run: npm run test:visual:storybook
```

### Deployment Pipeline
- **Build Validation**: Ensure Storybook builds successfully
- **Accessibility Testing**: Automated WCAG compliance checks
- **Visual Regression**: Compare component appearances
- **Performance Monitoring**: Track bundle size and loading times

## 🎯 **Best Practices**

### Component Development
- **Accessibility First**: Design with WCAG 2.1 AA compliance
- **Mobile Responsive**: Test across all viewport sizes
- **Performance Conscious**: Optimize for Core Web Vitals
- **Documentation Complete**: Include comprehensive usage guides

### Story Writing
- **Clear Examples**: Provide realistic usage scenarios
- **Interactive Controls**: Enable property exploration
- **Accessibility Notes**: Document compliance considerations
- **Error States**: Include failure and edge cases

### Maintenance
- **Regular Audits**: Monthly accessibility and performance reviews
- **Dependency Updates**: Keep Storybook and addons current
- **Browser Testing**: Validate across supported browsers
- **User Feedback**: Incorporate developer and designer input

## 📞 **Support and Resources**

### Getting Help
- **Documentation**: Comprehensive guides in `/docs/storybook/`
- **GitHub Issues**: Report bugs and request features
- **Team Chat**: Design system discussion channel
- **Office Hours**: Weekly design system sessions

### External Resources
- [Storybook Documentation](https://storybook.js.org/docs)
- [WCAG 2.1 Guidelines](https://www.w3.org/WAI/WCAG21/quickref/)
- [Tailwind CSS Documentation](https://tailwindcss.com/docs)
- [Web Accessibility Guidelines](https://webaim.org/)

---

## 🎉 **Get Started**

Ready to explore the Pynomaly Design System?

```bash
npm install && npm run storybook
```

Visit http://localhost:6006 and start building accessible, beautiful interfaces! 🚀

---

## 🔗 **Related Documentation**

- **[User Guides](../user-guides/README.md)** - Feature documentation
- **[Getting Started](../getting-started/README.md)** - Installation and setup
- **[Examples](../examples/README.md)** - Real-world examples
