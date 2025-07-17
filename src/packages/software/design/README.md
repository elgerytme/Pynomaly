# Design

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Design System](https://img.shields.io/badge/design-system-purple.svg)](https://designsystem.pynomaly.com/)
[![UI Components](https://img.shields.io/badge/UI-components-blue.svg)](https://components.pynomaly.com/)

## Overview

Design system, UI components, and visual identity for the Pynomaly platform.

**Architecture Layer**: Presentation Design Layer  
**Package Type**: Design System & UI Components  
**Status**: Production Ready

## Purpose

This package provides the complete design system that ensures visual consistency, usability, and accessibility across all Pynomaly interfaces. It includes UI components, design tokens, styling guidelines, and brand assets that create a cohesive user experience.

### Key Features

- **Design System**: Complete design tokens, patterns, and guidelines
- **UI Components**: Reusable React/Vue components with consistent styling
- **Visual Identity**: Brand colors, typography, iconography, and logos
- **Accessibility**: WCAG 2.1 AA compliant components and patterns
- **Responsive Design**: Mobile-first responsive design system
- **Dark Mode**: Complete dark mode theme support
- **Internationalization**: Multi-language design patterns

### Use Cases

- Building consistent user interfaces across all Pynomaly applications
- Creating new components following established design patterns
- Maintaining visual consistency in documentation and presentations
- Ensuring accessibility compliance across all interfaces
- Implementing responsive designs for different screen sizes

## Design System Architecture

### Component Structure

```
src/packages/software/design/
├── design/                    # Main package source
│   ├── tokens/               # Design tokens
│   │   ├── colors/          # Color palettes and schemes
│   │   ├── typography/      # Font families, sizes, weights
│   │   ├── spacing/         # Spacing scale and layout
│   │   ├── shadows/         # Box shadows and elevation
│   │   └── animations/      # Animation timing and easing
│   ├── components/          # UI components
│   │   ├── atoms/           # Basic building blocks
│   │   │   ├── Button/      # Button component
│   │   │   ├── Input/       # Input field component
│   │   │   ├── Icon/        # Icon component
│   │   │   └── Badge/       # Badge component
│   │   ├── molecules/       # Component combinations
│   │   │   ├── SearchBox/   # Search input with button
│   │   │   ├── Card/        # Card component
│   │   │   ├── Toast/       # Toast notifications
│   │   │   └── Modal/       # Modal dialogs
│   │   ├── organisms/       # Complex components
│   │   │   ├── Header/      # Application header
│   │   │   ├── Sidebar/     # Navigation sidebar
│   │   │   ├── DataTable/   # Data table with controls
│   │   │   └── Dashboard/   # Dashboard layouts
│   │   └── templates/       # Page templates
│   │       ├── LoginPage/   # Login page template
│   │       ├── DashboardPage/ # Dashboard page template
│   │       └── SettingsPage/ # Settings page template
│   ├── themes/              # Theme configurations
│   │   ├── light/           # Light theme
│   │   ├── dark/            # Dark theme
│   │   ├── high_contrast/   # High contrast theme
│   │   └── custom/          # Custom theme support
│   ├── assets/              # Design assets
│   │   ├── icons/           # Icon library (SVG)
│   │   ├── images/          # Brand images and illustrations
│   │   ├── fonts/           # Font files
│   │   └── logos/           # Logo variations
│   ├── patterns/            # Design patterns
│   │   ├── layouts/         # Layout patterns
│   │   ├── navigation/      # Navigation patterns
│   │   ├── forms/           # Form patterns
│   │   └── data_visualization/ # Chart and graph patterns
│   └── utilities/           # Design utilities
│       ├── css/             # CSS utilities and helpers
│       ├── scss/            # SCSS mixins and functions
│       └── js/              # JavaScript design utilities
├── storybook/               # Storybook component documentation
├── tests/                   # Component tests
├── docs/                    # Design documentation
└── examples/               # Usage examples
```

### Dependencies

- **Internal Dependencies**: None (design foundation layer)
- **External Dependencies**: React, Vue, Styled Components, Tailwind CSS
- **Optional Dependencies**: Storybook, Figma API, Adobe XD API

### Design Principles

1. **Consistency**: Uniform visual language across all interfaces
2. **Accessibility**: WCAG 2.1 AA compliance for all components
3. **Scalability**: Design system that grows with the platform
4. **Usability**: User-centered design with clear interaction patterns
5. **Performance**: Optimized components with minimal bundle impact
6. **Maintainability**: Well-structured, documented, and testable components

## Installation

### Prerequisites

- Python 3.11 or higher
- Node.js 16+ (for frontend components)
- npm or yarn package manager
- Basic understanding of design systems

### Package Installation

```bash
# Install design system (Python components)
cd src/packages/software/design
pip install -e .

# Install frontend dependencies
npm install

# Install with all design tools
pip install pynomaly-design[storybook,figma,adobe-xd]
```

### Frontend Installation

```bash
# Install React components
npm install @pynomaly/design-system

# Install Vue components
npm install @pynomaly/design-system-vue

# Install CSS framework
npm install @pynomaly/design-tokens
```

### Pynomaly Installation

```bash
# Install entire Pynomaly platform with this package
cd /path/to/pynomaly
pip install -e ".[design]"
```

## Usage

### Quick Start

```python
from pynomaly.design.tokens import DesignTokens
from pynomaly.design.components import ComponentLibrary
from pynomaly.design.themes import ThemeManager

# Load design tokens
tokens = DesignTokens()
colors = tokens.colors
typography = tokens.typography
spacing = tokens.spacing

# Create component library
library = ComponentLibrary(theme="light")
button = library.create_button(
    text="Detect Anomalies",
    variant="primary",
    size="medium"
)

# Apply theme
theme_manager = ThemeManager()
theme_manager.apply_theme("dark")
```

### Basic Examples

#### Example 1: Using Design Tokens

```python
from pynomaly.design.tokens import Colors, Typography, Spacing

# Access design tokens
colors = Colors()
primary_color = colors.primary.blue_500
secondary_color = colors.secondary.gray_600
success_color = colors.semantic.success

# Typography tokens
typography = Typography()
heading_font = typography.families.heading
body_font = typography.families.body
large_text = typography.sizes.large

# Spacing tokens
spacing = Spacing()
small_padding = spacing.padding.small
medium_margin = spacing.margin.medium
large_gap = spacing.gap.large
```

#### Example 2: Creating Components

```python
from pynomaly.design.components import Button, Card, DataTable

# Create button component
button = Button(
    text="Run Detection",
    variant="primary",
    size="large",
    icon="play",
    disabled=False,
    loading=False
)

# Create card component
card = Card(
    title="Anomaly Detection Results",
    content="Found 15 anomalies in the dataset",
    actions=[button],
    elevated=True
)

# Create data table
table = DataTable(
    columns=["Timestamp", "Value", "Anomaly Score", "Status"],
    data=anomaly_data,
    sortable=True,
    filterable=True,
    pagination=True
)
```

### Frontend Usage

#### React Components

```jsx
import React from 'react';
import { Button, Card, DataTable } from '@pynomaly/design-system';
import { ThemeProvider } from '@pynomaly/design-system/themes';

function AnomalyDashboard() {
  return (
    <ThemeProvider theme="light">
      <div className="dashboard">
        <Card title="Detection Status">
          <Button 
            variant="primary" 
            size="large"
            onClick={handleDetection}
          >
            Start Detection
          </Button>
        </Card>
        <DataTable 
          data={anomalies}
          columns={columns}
          sortable
          filterable
        />
      </div>
    </ThemeProvider>
  );
}
```

#### Vue Components

```vue
<template>
  <div class="dashboard">
    <PCard title="Detection Status">
      <PButton 
        variant="primary" 
        size="large"
        @click="handleDetection"
      >
        Start Detection
      </PButton>
    </PCard>
    <PDataTable 
      :data="anomalies"
      :columns="columns"
      sortable
      filterable
    />
  </div>
</template>

<script>
import { PButton, PCard, PDataTable } from '@pynomaly/design-system-vue';

export default {
  components: {
    PButton,
    PCard,
    PDataTable
  },
  // Component logic
};
</script>
```

### Advanced Usage

#### Custom Theme Creation

```python
from pynomaly.design.themes import ThemeBuilder
from pynomaly.design.tokens import ColorPalette

# Create custom theme
theme_builder = ThemeBuilder()
custom_theme = theme_builder \
    .with_color_palette(ColorPalette.enterprise) \
    .with_typography_scale("comfortable") \
    .with_spacing_scale("compact") \
    .with_border_radius("rounded") \
    .with_shadows("elevated") \
    .build()

# Apply custom theme
theme_manager = ThemeManager()
theme_manager.register_theme("enterprise", custom_theme)
theme_manager.apply_theme("enterprise")
```

#### Component Customization

```python
from pynomaly.design.components import ComponentFactory
from pynomaly.design.patterns import FormPattern

# Create custom component
factory = ComponentFactory()
custom_button = factory.create_component(
    type="button",
    styles={
        "background": "gradient(primary, secondary)",
        "border": "none",
        "padding": "12px 24px",
        "border_radius": "8px"
    },
    interactions={
        "hover": "scale(1.05)",
        "active": "scale(0.98)",
        "focus": "ring(primary, 2px)"
    }
)

# Apply form pattern
form_pattern = FormPattern()
login_form = form_pattern.create_form([
    {"type": "input", "name": "username", "label": "Username"},
    {"type": "password", "name": "password", "label": "Password"},
    {"type": "submit", "text": "Login", "variant": "primary"}
])
```

## API Reference

### Core Classes

#### Design Tokens

- **`Colors`**: Color palette and semantic color definitions
- **`Typography`**: Font families, sizes, weights, and line heights
- **`Spacing`**: Margin, padding, and gap spacing scales
- **`Shadows`**: Box shadow and elevation definitions
- **`Animations`**: Animation timing and easing functions

#### Components

- **`Button`**: Button component with variants and states
- **`Input`**: Input field component with validation
- **`Card`**: Card component with header, content, and actions
- **`Modal`**: Modal dialog component
- **`DataTable`**: Data table with sorting and filtering
- **`Toast`**: Toast notification component

#### Themes

- **`ThemeManager`**: Theme switching and management
- **`ThemeBuilder`**: Custom theme creation
- **`ColorPalette`**: Pre-defined color palettes
- **`TypographyScale`**: Typography scale definitions

### Key Functions

```python
# Design token functions
from pynomaly.design.tokens import (
    get_color_token,
    get_typography_token,
    get_spacing_token,
    get_shadow_token
)

# Component creation functions
from pynomaly.design.components import (
    create_button,
    create_input,
    create_card,
    create_modal,
    create_data_table
)

# Theme management functions
from pynomaly.design.themes import (
    apply_theme,
    create_custom_theme,
    get_theme_tokens,
    switch_theme
)
```

## Development

### Storybook Development

```bash
# Start Storybook development server
npm run storybook

# Build Storybook
npm run build-storybook

# Run visual regression tests
npm run test:visual
```

### Testing

```bash
# Run component tests
npm test

# Run accessibility tests
npm run test:a11y

# Run Python tests
pytest tests/

# Run with coverage
pytest --cov=design --cov-report=html
```

### Code Quality

```bash
# Format code
ruff format design/
prettier --write "**/*.{js,jsx,ts,tsx,css,scss}"

# Type checking
mypy design/
npm run type-check

# Lint
eslint src/
stylelint "**/*.{css,scss}"
```

## Design Guidelines

### Color Usage

```python
# Primary colors for main actions
primary_blue = colors.primary.blue_500
primary_hover = colors.primary.blue_600

# Secondary colors for supporting actions
secondary_gray = colors.secondary.gray_500
secondary_hover = colors.secondary.gray_600

# Semantic colors for status
success_green = colors.semantic.success
warning_yellow = colors.semantic.warning
error_red = colors.semantic.error
info_blue = colors.semantic.info
```

### Typography Hierarchy

```python
# Heading levels
h1 = typography.heading.h1  # 32px, bold
h2 = typography.heading.h2  # 24px, semibold
h3 = typography.heading.h3  # 20px, semibold
h4 = typography.heading.h4  # 16px, semibold

# Body text
body_large = typography.body.large    # 16px, regular
body_medium = typography.body.medium  # 14px, regular
body_small = typography.body.small    # 12px, regular

# Special text
code = typography.code.regular        # monospace, 14px
caption = typography.caption.regular  # 12px, medium
```

### Spacing Scale

```python
# Spacing values (based on 4px grid)
xs = spacing.xs    # 4px
sm = spacing.sm    # 8px
md = spacing.md    # 16px
lg = spacing.lg    # 24px
xl = spacing.xl    # 32px
xxl = spacing.xxl  # 48px
```

## Accessibility

### WCAG Compliance

- **Color Contrast**: Minimum 4.5:1 ratio for normal text, 3:1 for large text
- **Keyboard Navigation**: Full keyboard accessibility for all interactive elements
- **Screen Reader**: Proper ARIA labels and semantic HTML
- **Focus Management**: Visible focus indicators and logical tab order

### Testing Tools

```bash
# Run accessibility tests
npm run test:a11y

# Test with screen reader
npm run test:screen-reader

# Color contrast validation
npm run test:contrast
```

## Integration

### Figma Integration

```python
from pynomaly.design.integrations import FigmaSync

# Sync tokens from Figma
figma_sync = FigmaSync(api_token="your-token")
figma_sync.sync_tokens("design-system-file-key")

# Export components to Figma
figma_sync.export_components(library.components)
```

### Design System Documentation

```bash
# Generate design system documentation
npm run docs:generate

# Deploy documentation
npm run docs:deploy
```

## Troubleshooting

### Common Issues

**Issue**: Components not rendering with correct styles
**Solution**: Check theme provider is wrapping components and theme is loaded

**Issue**: Icons not displaying
**Solution**: Ensure icon library is imported and SVG assets are accessible

**Issue**: Accessibility warnings
**Solution**: Review ARIA labels and semantic HTML structure

### Debug Mode

```python
from pynomaly.design.config import enable_debug_mode

# Enable debug mode for design system
enable_debug_mode(
    component_debugging=True,
    style_debugging=True,
    accessibility_debugging=True
)
```

## Contributing

1. **Fork and Clone**: Fork the repository and clone your fork
2. **Branch**: Create a feature branch (`git checkout -b feature/design-enhancement`)
3. **Design**: Follow design system principles and accessibility guidelines
4. **Test**: Add comprehensive tests including visual regression and accessibility tests
5. **Document**: Update Storybook stories and design documentation
6. **Commit**: Use conventional commit messages
7. **Pull Request**: Submit a PR with clear description and design rationale

### Adding New Components

Follow the component template:

```python
from pynomaly.design.components.base import BaseComponent

class NewComponent(BaseComponent):
    def __init__(self, **props):
        super().__init__(**props)
        self.validate_props()
    
    def render(self) -> str:
        # Implement component rendering
        pass
    
    def get_accessibility_props(self) -> dict:
        # Implement accessibility properties
        pass
```

## Support

- **Documentation**: [Design System Docs](docs/)
- **Storybook**: [Component Library](storybook/)
- **Design Assets**: [Figma Design System](https://figma.com/pynomaly-design)
- **Issues**: [GitHub Issues](../../../issues)
- **Discussions**: [GitHub Discussions](../../../discussions)

## License

MIT License. See [LICENSE](../../../LICENSE) file for details.

---

**Part of the [Pynomaly](../../../) monorepo** - Advanced platform design system