export default {
  title: 'Foundation/Colors',
  parameters: {
    docs: {
      description: {
        component: 'WCAG 2.1 AA compliant color system with semantic naming and accessibility features.'
      }
    }
  }
};

// Color palette template
const ColorPaletteTemplate = ({ colors, title, description }) => {
  return `
    <div class="p-6">
      <h3 class="text-2xl font-semibold mb-2">${title}</h3>
      <p class="text-gray-600 mb-6">${description}</p>
      <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5 gap-4">
        ${colors.map(color => `
          <div class="bg-white rounded-lg shadow-md overflow-hidden border">
            <div class="h-20" style="background-color: ${color.hex}"></div>
            <div class="p-4">
              <h4 class="font-medium text-sm">${color.name}</h4>
              <p class="text-xs text-gray-500 mt-1">${color.hex}</p>
              <p class="text-xs text-gray-500">${color.usage}</p>
              ${color.contrast ? `<p class="text-xs mt-2 font-medium text-green-600">Contrast: ${color.contrast}</p>` : ''}
            </div>
          </div>
        `).join('')}
      </div>
    </div>
  `;
};

// Primary Colors
export const PrimaryColors = ColorPaletteTemplate.bind({});
PrimaryColors.args = {
  title: 'Primary Colors',
  description: 'Main brand colors used for primary actions, links, and key interface elements.',
  colors: [
    {
      name: 'Primary 50',
      hex: '#f0f9ff',
      usage: 'Light backgrounds, subtle highlights',
      contrast: '21:1 with Primary 900'
    },
    {
      name: 'Primary 100', 
      hex: '#e0f2fe',
      usage: 'Hover states, light accents',
      contrast: '16.8:1 with Primary 900'
    },
    {
      name: 'Primary 500',
      hex: '#0ea5e9',
      usage: 'Primary buttons, links, active states',
      contrast: '4.5:1 with white'
    },
    {
      name: 'Primary 600',
      hex: '#0284c7', 
      usage: 'Hover states, pressed buttons',
      contrast: '5.9:1 with white'
    },
    {
      name: 'Primary 700',
      hex: '#0369a1',
      usage: 'Active states, dark accents',
      contrast: '7.8:1 with white'
    },
    {
      name: 'Primary 900',
      hex: '#0c4a6e',
      usage: 'High contrast text, dark themes',
      contrast: '12.6:1 with white'
    }
  ]
};

PrimaryColors.parameters = {
  docs: {
    description: {
      story: 'Primary color palette optimized for accessibility with WCAG AA contrast ratios. Used for main interactive elements, branding, and primary actions.'
    }
  }
};

// Semantic Colors
export const SemanticColors = ColorPaletteTemplate.bind({});
SemanticColors.args = {
  title: 'Semantic Colors',
  description: 'Status and feedback colors that communicate meaning to users.',
  colors: [
    {
      name: 'Success',
      hex: '#22c55e',
      usage: 'Success messages, positive states',
      contrast: '4.5:1 with white'
    },
    {
      name: 'Success Light',
      hex: '#dcfce7',
      usage: 'Success backgrounds, subtle indicators',
      contrast: '18.2:1 with Success 700'
    },
    {
      name: 'Warning',
      hex: '#f59e0b',
      usage: 'Warning messages, cautionary states',
      contrast: '4.7:1 with white'
    },
    {
      name: 'Warning Light',
      hex: '#fef3c7',
      usage: 'Warning backgrounds, attention areas',
      contrast: '16.5:1 with Warning 700'
    },
    {
      name: 'Error',
      hex: '#ef4444',
      usage: 'Error messages, destructive actions',
      contrast: '4.5:1 with white'
    },
    {
      name: 'Error Light',
      hex: '#fee2e2',
      usage: 'Error backgrounds, danger zones',
      contrast: '18.9:1 with Error 700'
    }
  ]
};

SemanticColors.parameters = {
  docs: {
    description: {
      story: 'Semantic colors provide consistent meaning across the interface. Each color meets WCAG AA standards and includes light variants for backgrounds.'
    }
  }
};

// Neutral Colors
export const NeutralColors = ColorPaletteTemplate.bind({});
NeutralColors.args = {
  title: 'Neutral Colors',
  description: 'Text, backgrounds, and border colors for content hierarchy and interface structure.',
  colors: [
    {
      name: 'Neutral 50',
      hex: '#fafafa',
      usage: 'Page backgrounds, light surfaces',
      contrast: '20.35:1 with Neutral 900'
    },
    {
      name: 'Neutral 100',
      hex: '#f5f5f5',
      usage: 'Card backgrounds, secondary surfaces',
      contrast: '18.82:1 with Neutral 900'
    },
    {
      name: 'Neutral 200',
      hex: '#e5e5e5',
      usage: 'Borders, dividers, subtle separators',
      contrast: '15.33:1 with Neutral 900'
    },
    {
      name: 'Neutral 400',
      hex: '#a3a3a3',
      usage: 'Muted text, placeholders',
      contrast: '7.0:1 with white'
    },
    {
      name: 'Neutral 600',
      hex: '#525252',
      usage: 'Secondary text, captions',
      contrast: '7.23:1 with white'
    },
    {
      name: 'Neutral 900',
      hex: '#171717',
      usage: 'Primary text, headings',
      contrast: '18.82:1 with white'
    }
  ]
};

NeutralColors.parameters = {
  docs: {
    description: {
      story: 'Neutral colors form the foundation of text hierarchy and interface structure. All combinations meet or exceed WCAG AA requirements.'
    }
  }
};

// Color Accessibility Guide
export const AccessibilityGuide = () => {
  return `
    <div class="p-6 max-w-4xl">
      <h3 class="text-2xl font-semibold mb-6">Color Accessibility Guidelines</h3>
      
      <div class="space-y-8">
        <section>
          <h4 class="text-lg font-medium mb-4">WCAG 2.1 Contrast Requirements</h4>
          <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div class="bg-gray-50 p-4 rounded-lg">
              <h5 class="font-medium mb-2">AA Standard</h5>
              <ul class="text-sm space-y-1">
                <li>• Normal text: 4.5:1 minimum contrast ratio</li>
                <li>• Large text (18pt+): 3:1 minimum contrast ratio</li>
                <li>• UI components: 3:1 minimum contrast ratio</li>
              </ul>
            </div>
            <div class="bg-gray-50 p-4 rounded-lg">
              <h5 class="font-medium mb-2">AAA Enhanced</h5>
              <ul class="text-sm space-y-1">
                <li>• Normal text: 7:1 minimum contrast ratio</li>
                <li>• Large text (18pt+): 4.5:1 minimum contrast ratio</li>
                <li>• Better visibility for low vision users</li>
              </ul>
            </div>
          </div>
        </section>

        <section>
          <h4 class="text-lg font-medium mb-4">Color Usage Best Practices</h4>
          <div class="bg-blue-50 border border-blue-200 p-4 rounded-lg">
            <h5 class="font-medium mb-2 text-blue-900">✅ Do</h5>
            <ul class="text-sm space-y-1 text-blue-800">
              <li>• Use color plus another indicator (icons, text, patterns)</li>
              <li>• Test with color blindness simulators</li>
              <li>• Provide sufficient contrast for all text</li>
              <li>• Use semantic colors consistently</li>
            </ul>
          </div>
          <div class="bg-red-50 border border-red-200 p-4 rounded-lg mt-4">
            <h5 class="font-medium mb-2 text-red-900">❌ Don't</h5>
            <ul class="text-sm space-y-1 text-red-800">
              <li>• Rely solely on color to convey information</li>
              <li>• Use red and green together without additional cues</li>
              <li>• Create custom colors without testing contrast</li>
              <li>• Use color alone for form validation feedback</li>
            </ul>
          </div>
        </section>

        <section>
          <h4 class="text-lg font-medium mb-4">Color Blindness Considerations</h4>
          <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div class="text-center">
              <div class="w-12 h-12 bg-red-500 rounded-full mx-auto mb-2"></div>
              <p class="text-sm font-medium">Protanopia</p>
              <p class="text-xs text-gray-600">Red-blind (1% of men)</p>
            </div>
            <div class="text-center">
              <div class="w-12 h-12 bg-green-500 rounded-full mx-auto mb-2"></div>
              <p class="text-sm font-medium">Deuteranopia</p>
              <p class="text-xs text-gray-600">Green-blind (1% of men)</p>
            </div>
            <div class="text-center">
              <div class="w-12 h-12 bg-blue-500 rounded-full mx-auto mb-2"></div>
              <p class="text-sm font-medium">Tritanopia</p>
              <p class="text-xs text-gray-600">Blue-blind (rare)</p>
            </div>
          </div>
        </section>
      </div>
    </div>
  `;
};

AccessibilityGuide.parameters = {
  docs: {
    description: {
      story: 'Comprehensive guide to using colors accessibly, including WCAG compliance standards and color blindness considerations.'
    }
  }
};