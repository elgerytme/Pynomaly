import '../src/pynomaly/presentation/web/static/css/design-system.css';

/** @type { import('@storybook/html').Preview } */
const preview = {
  parameters: {
    actions: { argTypesRegex: '^on[A-Z].*' },
    controls: {
      matchers: {
        color: /(background|color)$/i,
        date: /Date$/,
      },
    },
    docs: {
      theme: {
        base: 'light',
        brandTitle: 'Pynomaly Design System',
        brandUrl: 'https://pynomaly.io',
        brandImage: '/static/images/pynomaly-logo.svg',
        brandTarget: '_self',

        colorPrimary: '#0ea5e9',
        colorSecondary: '#22c55e',

        // UI
        appBg: '#f8fafc',
        appContentBg: '#ffffff',
        appBorderColor: '#e2e8f0',
        appBorderRadius: 8,

        // Typography
        fontBase: '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
        fontCode: '"JetBrains Mono", Consolas, "Liberation Mono", Menlo, Courier, monospace',

        // Text colors
        textColor: '#1e293b',
        textInverseColor: '#ffffff',

        // Toolbar default and active colors
        barTextColor: '#64748b',
        barSelectedColor: '#0ea5e9',
        barBg: '#ffffff',

        // Form colors
        inputBg: '#ffffff',
        inputBorder: '#d1d5db',
        inputTextColor: '#1f2937',
        inputBorderRadius: 6,
      }
    },
    backgrounds: {
      default: 'light',
      values: [
        {
          name: 'light',
          value: '#ffffff'
        },
        {
          name: 'gray-50',
          value: '#f8fafc'
        },
        {
          name: 'gray-100',
          value: '#f1f5f9'
        },
        {
          name: 'dark',
          value: '#0f172a'
        },
        {
          name: 'primary',
          value: '#0ea5e9'
        }
      ]
    },
    viewport: {
      viewports: {
        mobile: {
          name: 'Mobile',
          styles: {
            width: '375px',
            height: '667px',
          },
        },
        tablet: {
          name: 'Tablet',
          styles: {
            width: '768px',
            height: '1024px',
          },
        },
        desktop: {
          name: 'Desktop',
          styles: {
            width: '1200px',
            height: '800px',
          },
        },
        desktopHD: {
          name: 'Desktop HD',
          styles: {
            width: '1920px',
            height: '1080px',
          },
        },
      },
    },
    a11y: {
      config: {
        rules: [
          {
            id: 'color-contrast',
            enabled: true,
          },
          {
            id: 'focus-management',
            enabled: true,
          },
          {
            id: 'keyboard-navigation',
            enabled: true,
          },
        ],
      },
      options: {
        checks: { 'color-contrast': { options: { noScroll: true } } },
        restoreScroll: true,
      },
    },
    measure: {
      results: {
        margin: {
          color: '#ff5722',
        },
        padding: {
          color: '#8bc34a',
        },
      },
    },
    outline: {
      color: '#0ea5e9',
      width: '2px',
    },
  },

  argTypes: {
    // Global argTypes for consistent component props
    size: {
      control: { type: 'select' },
      options: ['xs', 'sm', 'base', 'lg', 'xl'],
      description: 'Size variant of the component'
    },
    variant: {
      control: { type: 'select' },
      options: ['primary', 'secondary', 'success', 'warning', 'danger', 'info'],
      description: 'Visual variant of the component'
    },
    disabled: {
      control: { type: 'boolean' },
      description: 'Whether the component is disabled'
    },
    loading: {
      control: { type: 'boolean' },
      description: 'Whether the component is in a loading state'
    },
    theme: {
      control: { type: 'select' },
      options: ['light', 'dark'],
      description: 'Theme variant'
    }
  },

  globalTypes: {
    theme: {
      description: 'Global theme for components',
      defaultValue: 'light',
      toolbar: {
        title: 'Theme',
        icon: 'paintbrush',
        items: [
          { value: 'light', title: 'Light' },
          { value: 'dark', title: 'Dark' }
        ],
        dynamicTitle: true,
      },
    },
    locale: {
      description: 'Internationalization locale',
      defaultValue: 'en',
      toolbar: {
        icon: 'globe',
        items: [
          { value: 'en', title: 'English' },
          { value: 'es', title: 'Español' },
          { value: 'fr', title: 'Français' },
          { value: 'de', title: 'Deutsch' },
        ],
        showName: true,
      },
    },
    density: {
      description: 'Component density',
      defaultValue: 'normal',
      toolbar: {
        title: 'Density',
        icon: 'component',
        items: [
          { value: 'compact', title: 'Compact' },
          { value: 'normal', title: 'Normal' },
          { value: 'comfortable', title: 'Comfortable' }
        ],
      },
    },
  },

  decorators: [
    (story, context) => {
      const theme = context.globals.theme || 'light';
      const density = context.globals.density || 'normal';

      // Apply theme class to the story wrapper
      const wrapper = document.createElement('div');
      wrapper.className = `storybook-wrapper theme-${theme} density-${density}`;
      wrapper.style.cssText = `
        min-height: 100vh;
        padding: 1rem;
        background-color: ${theme === 'dark' ? '#0f172a' : '#ffffff'};
        color: ${theme === 'dark' ? '#f1f5f9' : '#1e293b'};
        font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        transition: background-color 0.2s ease, color 0.2s ease;
      `;

      // Apply density styles
      if (density === 'compact') {
        wrapper.style.fontSize = '0.875rem';
        wrapper.style.lineHeight = '1.25rem';
      } else if (density === 'comfortable') {
        wrapper.style.fontSize = '1.125rem';
        wrapper.style.lineHeight = '1.75rem';
      }

      wrapper.appendChild(story());
      return wrapper;
    },
  ],

  tags: ['autodocs'],
};

export default preview;
