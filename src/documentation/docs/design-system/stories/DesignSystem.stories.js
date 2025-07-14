/**
 * Design System Stories
 * Interactive documentation for Pynomaly design system foundations
 */

export default {
  title: 'Design System/Overview',
  tags: ['autodocs'],
  parameters: {
    docs: {
      description: {
        component: 'Pynomaly Design System foundations including colors, typography, spacing, and component guidelines. All design tokens follow accessibility standards and support both light and dark themes.',
      },
    },
  },
};

// Color Palette Component
const createColorPalette = () => {
  const container = document.createElement('div');
  container.className = 'space-y-8';

  const title = document.createElement('h2');
  title.className = 'text-2xl font-bold text-gray-900 mb-6';
  title.textContent = 'Color System';
  container.appendChild(title);

  const colorPalettes = {
    'Primary': {
      description: 'Main brand colors for primary actions and emphasis',
      colors: {
        'primary-50': '#eff6ff',
        'primary-100': '#dbeafe',
        'primary-200': '#bfdbfe',
        'primary-300': '#93c5fd',
        'primary-400': '#60a5fa',
        'primary-500': '#3b82f6',
        'primary-600': '#2563eb',
        'primary-700': '#1d4ed8',
        'primary-800': '#1e40af',
        'primary-900': '#1e3a8a',
      }
    },
    'Success': {
      description: 'Colors for positive states, confirmations, and success messages',
      colors: {
        'success-50': '#ecfdf5',
        'success-100': '#d1fae5',
        'success-200': '#a7f3d0',
        'success-300': '#6ee7b7',
        'success-400': '#34d399',
        'success-500': '#10b981',
        'success-600': '#059669',
        'success-700': '#047857',
        'success-800': '#065f46',
        'success-900': '#064e3b',
      }
    },
    'Warning': {
      description: 'Colors for caution states and warning messages',
      colors: {
        'warning-50': '#fffbeb',
        'warning-100': '#fef3c7',
        'warning-200': '#fde68a',
        'warning-300': '#fcd34d',
        'warning-400': '#fbbf24',
        'warning-500': '#f59e0b',
        'warning-600': '#d97706',
        'warning-700': '#b45309',
        'warning-800': '#92400e',
        'warning-900': '#78350f',
      }
    },
    'Danger': {
      description: 'Colors for error states, destructive actions, and anomaly alerts',
      colors: {
        'danger-50': '#fef2f2',
        'danger-100': '#fee2e2',
        'danger-200': '#fecaca',
        'danger-300': '#fca5a5',
        'danger-400': '#f87171',
        'danger-500': '#ef4444',
        'danger-600': '#dc2626',
        'danger-700': '#b91c1c',
        'danger-800': '#991b1b',
        'danger-900': '#7f1d1d',
      }
    },
    'Neutral': {
      description: 'Grayscale colors for text, borders, backgrounds, and secondary elements',
      colors: {
        'gray-50': '#f9fafb',
        'gray-100': '#f3f4f6',
        'gray-200': '#e5e7eb',
        'gray-300': '#d1d5db',
        'gray-400': '#9ca3af',
        'gray-500': '#6b7280',
        'gray-600': '#4b5563',
        'gray-700': '#374151',
        'gray-800': '#1f2937',
        'gray-900': '#111827',
      }
    }
  };

  Object.entries(colorPalettes).forEach(([paletteName, palette]) => {
    const section = document.createElement('div');
    section.className = 'space-y-4';

    const sectionTitle = document.createElement('h3');
    sectionTitle.className = 'text-lg font-semibold text-gray-900';
    sectionTitle.textContent = paletteName;
    section.appendChild(sectionTitle);

    const description = document.createElement('p');
    description.className = 'text-sm text-gray-600';
    description.textContent = palette.description;
    section.appendChild(description);

    const colorGrid = document.createElement('div');
    colorGrid.className = 'grid grid-cols-5 gap-2';

    Object.entries(palette.colors).forEach(([colorName, colorValue]) => {
      const colorSwatch = document.createElement('div');
      colorSwatch.className = 'space-y-2';

      const swatch = document.createElement('div');
      swatch.className = 'h-16 rounded-lg border border-gray-200 shadow-sm flex items-center justify-center relative';
      swatch.style.backgroundColor = colorValue;

      // Add contrast check
      const isLight = isLightColor(colorValue);
      swatch.style.color = isLight ? '#000000' : '#ffffff';

      // Accessible color information
      swatch.setAttribute('title', `${colorName}: ${colorValue}`);
      swatch.setAttribute('aria-label', `Color swatch for ${colorName}, hex value ${colorValue}`);

      const swatchInfo = document.createElement('div');
      swatchInfo.className = 'text-center';

      const name = document.createElement('div');
      name.className = 'text-xs font-medium';
      name.textContent = colorName.split('-')[1] || colorName;
      swatchInfo.appendChild(name);

      const value = document.createElement('div');
      value.className = 'text-xs opacity-80';
      value.textContent = colorValue.toUpperCase();
      swatchInfo.appendChild(value);

      swatch.appendChild(swatchInfo);
      colorSwatch.appendChild(swatch);

      const label = document.createElement('div');
      label.className = 'text-xs text-center text-gray-600 font-mono';
      label.textContent = colorName;
      colorSwatch.appendChild(label);

      colorGrid.appendChild(colorSwatch);
    });

    section.appendChild(colorGrid);
    container.appendChild(section);
  });

  return container;
};

// Helper function to determine if a color is light
const isLightColor = (color) => {
  const hex = color.replace('#', '');
  const r = parseInt(hex.substr(0, 2), 16);
  const g = parseInt(hex.substr(2, 2), 16);
  const b = parseInt(hex.substr(4, 2), 16);
  const brightness = (r * 299 + g * 587 + b * 114) / 1000;
  return brightness > 155;
};

// Typography System
const createTypographySystem = () => {
  const container = document.createElement('div');
  container.className = 'space-y-8';

  const title = document.createElement('h2');
  title.className = 'text-2xl font-bold text-gray-900 mb-6';
  title.textContent = 'Typography';
  container.appendChild(title);

  const typeScale = [
    { name: 'Display Large', class: 'text-6xl font-bold', usage: 'Hero headlines, marketing' },
    { name: 'Display Medium', class: 'text-5xl font-bold', usage: 'Page headers, major sections' },
    { name: 'Display Small', class: 'text-4xl font-bold', usage: 'Section headers' },
    { name: 'Heading 1', class: 'text-3xl font-bold', usage: 'Main page titles' },
    { name: 'Heading 2', class: 'text-2xl font-semibold', usage: 'Major section headers' },
    { name: 'Heading 3', class: 'text-xl font-semibold', usage: 'Subsection headers' },
    { name: 'Heading 4', class: 'text-lg font-semibold', usage: 'Minor section headers' },
    { name: 'Body Large', class: 'text-lg font-normal', usage: 'Large body text, introductions' },
    { name: 'Body Regular', class: 'text-base font-normal', usage: 'Standard body text' },
    { name: 'Body Small', class: 'text-sm font-normal', usage: 'Supporting text, captions' },
    { name: 'Caption', class: 'text-xs font-normal', usage: 'Labels, metadata' },
    { name: 'Code', class: 'text-sm font-mono', usage: 'Code snippets, technical text' },
  ];

  typeScale.forEach(type => {
    const typeExample = document.createElement('div');
    typeExample.className = 'border-b border-gray-100 pb-4 mb-4';

    const sample = document.createElement('div');
    sample.className = `${type.class} text-gray-900 mb-2`;
    sample.textContent = 'The quick brown fox jumps over the lazy dog';
    typeExample.appendChild(sample);

    const info = document.createElement('div');
    info.className = 'flex justify-between items-center text-sm text-gray-600';

    const details = document.createElement('div');
    details.innerHTML = `<span class="font-medium">${type.name}</span> • ${type.usage}`;

    const classes = document.createElement('code');
    classes.className = 'bg-gray-100 px-2 py-1 rounded text-xs font-mono';
    classes.textContent = type.class;

    info.appendChild(details);
    info.appendChild(classes);
    typeExample.appendChild(info);

    container.appendChild(typeExample);
  });

  return container;
};

// Spacing System
const createSpacingSystem = () => {
  const container = document.createElement('div');
  container.className = 'space-y-8';

  const title = document.createElement('h2');
  title.className = 'text-2xl font-bold text-gray-900 mb-6';
  title.textContent = 'Spacing Scale';
  container.appendChild(title);

  const spacingScale = [
    { name: 'xs', value: '0.25rem', px: '4px', class: 'p-1' },
    { name: 'sm', value: '0.5rem', px: '8px', class: 'p-2' },
    { name: 'base', value: '1rem', px: '16px', class: 'p-4' },
    { name: 'md', value: '1.5rem', px: '24px', class: 'p-6' },
    { name: 'lg', value: '2rem', px: '32px', class: 'p-8' },
    { name: 'xl', value: '3rem', px: '48px', class: 'p-12' },
    { name: '2xl', value: '4rem', px: '64px', class: 'p-16' },
    { name: '3xl', value: '6rem', px: '96px', class: 'p-24' },
  ];

  spacingScale.forEach(space => {
    const spacingExample = document.createElement('div');
    spacingExample.className = 'flex items-center gap-4 py-3';

    const visual = document.createElement('div');
    visual.className = 'bg-blue-100 border border-blue-300';
    visual.style.cssText = `
      width: ${space.value};
      height: ${space.value};
      min-width: ${space.value};
    `;

    const info = document.createElement('div');
    info.className = 'flex-1';
    info.innerHTML = `
      <div class="font-medium text-gray-900">${space.name}</div>
      <div class="text-sm text-gray-600">${space.value} (${space.px})</div>
      <code class="text-xs bg-gray-100 px-2 py-1 rounded font-mono">${space.class}</code>
    `;

    spacingExample.appendChild(visual);
    spacingExample.appendChild(info);
    container.appendChild(spacingExample);
  });

  return container;
};

// Component Guidelines
const createComponentGuidelines = () => {
  const container = document.createElement('div');
  container.className = 'space-y-8';

  const title = document.createElement('h2');
  title.className = 'text-2xl font-bold text-gray-900 mb-6';
  title.textContent = 'Component Guidelines';
  container.appendChild(title);

  const guidelines = [
    {
      title: 'Accessibility First',
      description: 'All components must meet WCAG 2.1 AA standards',
      rules: [
        'Include proper ARIA labels and roles',
        'Ensure keyboard navigation support',
        'Maintain 4.5:1 color contrast minimum',
        'Provide alternative text for visual elements',
        'Support screen readers and assistive technologies'
      ]
    },
    {
      title: 'Responsive Design',
      description: 'Components should work across all device sizes',
      rules: [
        'Mobile-first approach (320px and up)',
        'Touch-friendly targets (44px minimum)',
        'Flexible layouts that adapt to screen size',
        'Readable text at all zoom levels (up to 200%)',
        'Progressive enhancement for advanced features'
      ]
    },
    {
      title: 'Consistency',
      description: 'Maintain visual and functional consistency',
      rules: [
        'Use design tokens for colors, spacing, and typography',
        'Follow established patterns for similar components',
        'Consistent naming conventions for classes and props',
        'Standardized interaction patterns',
        'Unified error handling and feedback'
      ]
    },
    {
      title: 'Performance',
      description: 'Optimize for fast loading and smooth interactions',
      rules: [
        'Minimize bundle size impact',
        'Use CSS-in-JS efficiently',
        'Implement proper loading states',
        'Avoid layout shifts',
        'Optimize for Core Web Vitals'
      ]
    }
  ];

  guidelines.forEach(guideline => {
    const section = document.createElement('div');
    section.className = 'bg-gray-50 border border-gray-200 rounded-lg p-6';

    const header = document.createElement('div');
    header.className = 'mb-4';

    const guidelineTitle = document.createElement('h3');
    guidelineTitle.className = 'text-lg font-semibold text-gray-900';
    guidelineTitle.textContent = guideline.title;

    const guidelineDesc = document.createElement('p');
    guidelineDesc.className = 'text-sm text-gray-600 mt-1';
    guidelineDesc.textContent = guideline.description;

    header.appendChild(guidelineTitle);
    header.appendChild(guidelineDesc);
    section.appendChild(header);

    const rulesList = document.createElement('ul');
    rulesList.className = 'space-y-2';

    guideline.rules.forEach(rule => {
      const listItem = document.createElement('li');
      listItem.className = 'flex items-start gap-2 text-sm text-gray-700';
      listItem.innerHTML = `
        <svg class="w-4 h-4 text-green-500 mt-0.5 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
          <path fill-rule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clip-rule="evenodd"></path>
        </svg>
        <span>${rule}</span>
      `;
      rulesList.appendChild(listItem);
    });

    section.appendChild(rulesList);
    container.appendChild(section);
  });

  return container;
};

// Stories
export const ColorPalette = {
  render: createColorPalette,
  parameters: {
    docs: {
      description: {
        story: 'Complete color palette for the Pynomaly design system. All colors meet WCAG AA contrast requirements when used appropriately.',
      },
    },
  },
};

export const Typography = {
  render: createTypographySystem,
  parameters: {
    docs: {
      description: {
        story: 'Typography scale and usage guidelines. Uses system fonts for optimal performance and accessibility.',
      },
    },
  },
};

export const Spacing = {
  render: createSpacingSystem,
  parameters: {
    docs: {
      description: {
        story: 'Consistent spacing scale used throughout the design system for margins, padding, and layout.',
      },
    },
  },
};

export const Guidelines = {
  render: createComponentGuidelines,
  parameters: {
    docs: {
      description: {
        story: 'Core principles and guidelines for creating consistent, accessible components in the Pynomaly design system.',
      },
    },
  },
};

export const DesignTokens = {
  render: () => {
    const container = document.createElement('div');
    container.className = 'space-y-8';

    const title = document.createElement('h2');
    title.className = 'text-2xl font-bold text-gray-900 mb-6';
    title.textContent = 'Design Tokens';
    container.appendChild(title);

    const description = document.createElement('div');
    description.className = 'bg-blue-50 border border-blue-200 rounded-lg p-6 mb-8';
    description.innerHTML = `
      <h3 class="text-lg font-semibold text-blue-900 mb-2">CSS Custom Properties</h3>
      <p class="text-blue-800 mb-4">The Pynomaly design system uses CSS custom properties (variables) for consistent theming and easy customization.</p>
      <code class="block bg-blue-100 p-4 rounded text-sm font-mono text-blue-900">
:root {<br>
  /* Primary Colors */<br>
  --color-primary-500: #3b82f6;<br>
  --color-primary-600: #2563eb;<br>
  <br>
  /* Typography */<br>
  --font-family-sans: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;<br>
  --font-size-base: 1rem;<br>
  --line-height-base: 1.5;<br>
  <br>
  /* Spacing */<br>
  --spacing-4: 1rem;<br>
  --spacing-6: 1.5rem;<br>
  <br>
  /* Shadows */<br>
  --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);<br>
  --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1);<br>
}
      </code>
    `;
    container.appendChild(description);

    return container;
  },
  parameters: {
    docs: {
      description: {
        story: 'Design tokens implementation using CSS custom properties for consistent theming.',
      },
    },
  },
};
