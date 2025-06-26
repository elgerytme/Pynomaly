import { INITIAL_VIEWPORTS } from '@storybook/addon-viewport';

// Import design system CSS
import '../src/pynomaly/presentation/web/static/css/design-system.css';
import '../src/pynomaly/presentation/web/static/css/tailwind.css';

export const parameters = {
  actions: { argTypesRegex: '^on[A-Z].*' },
  
  controls: {
    matchers: {
      color: /(background|color)$/i,
      date: /Date$/,
    },
    expanded: true,
    sort: 'requiredFirst'
  },

  docs: {
    extractComponentDescription: (component, { notes }) => {
      if (notes) {
        return typeof notes === 'string' ? notes : notes.markdown || notes.text;
      }
      return null;
    },
    source: {
      state: 'open',
      type: 'dynamic'
    }
  },

  // Accessibility testing configuration
  a11y: {
    element: '#storybook-root',
    config: {
      rules: [
        {
          id: 'color-contrast',
          reviewOnFail: true
        },
        {
          id: 'focus-order-semantics',
          reviewOnFail: true
        },
        {
          id: 'keyboard-navigation',
          reviewOnFail: true
        }
      ]
    },
    options: {
      checks: { 'color-contrast': { options: { noScroll: true } } },
      restoreScroll: true
    }
  },

  // Viewport configuration for responsive testing
  viewport: {
    viewports: {
      ...INITIAL_VIEWPORTS,
      pynomaly_mobile: {
        name: 'Pynomaly Mobile',
        styles: {
          width: '375px',
          height: '667px'
        }
      },
      pynomaly_tablet: {
        name: 'Pynomaly Tablet',
        styles: {
          width: '768px',
          height: '1024px'
        }
      },
      pynomaly_desktop: {
        name: 'Pynomaly Desktop',
        styles: {
          width: '1280px',
          height: '800px'
        }
      },
      pynomaly_wide: {
        name: 'Pynomaly Wide',
        styles: {
          width: '1920px',
          height: '1080px'
        }
      }
    },
    defaultViewport: 'pynomaly_desktop'
  },

  // Background configuration for design system testing
  backgrounds: {
    default: 'light',
    values: [
      {
        name: 'light',
        value: '#fafafa'
      },
      {
        name: 'dark',
        value: '#171717'
      },
      {
        name: 'primary',
        value: '#0ea5e9'
      },
      {
        name: 'surface',
        value: '#f5f5f5'
      }
    ]
  },

  // Layout configuration
  layout: 'centered',

  // Custom toolbar options
  toolbar: {
    title: { hidden: false },
    zoom: { hidden: false },
    eject: { hidden: false },
    copy: { hidden: false },
    fullscreen: { hidden: false }
  }
};

// Global decorators
export const decorators = [
  (Story, context) => {
    // Add design system context
    const wrapper = document.createElement('div');
    wrapper.className = 'pynomaly-storybook-wrapper';
    wrapper.style.fontFamily = 'Inter, system-ui, sans-serif';
    wrapper.style.fontSize = '14px';
    wrapper.style.lineHeight = '1.5';
    wrapper.style.color = '#171717';
    
    // Add accessibility landmarks for better screen reader support
    wrapper.setAttribute('role', 'main');
    wrapper.setAttribute('aria-label', 'Component demonstration');
    
    const story = Story();
    if (typeof story === 'string') {
      wrapper.innerHTML = story;
    } else {
      wrapper.appendChild(story);
    }
    
    return wrapper;
  },
  
  (Story, context) => {
    // Theme decorator for consistent theming
    const { theme = 'light' } = context.globals;
    
    document.documentElement.setAttribute('data-theme', theme);
    document.documentElement.className = `theme-${theme}`;
    
    return Story();
  }
];

// Global types for addon controls
export const globalTypes = {
  theme: {
    name: 'Theme',
    description: 'Global theme for components',
    defaultValue: 'light',
    toolbar: {
      icon: 'paintbrush',
      items: [
        { value: 'light', title: 'Light Theme' },
        { value: 'dark', title: 'Dark Theme' },
        { value: 'high-contrast', title: 'High Contrast' }
      ],
      showName: true
    }
  },
  locale: {
    name: 'Locale',
    description: 'Internationalization locale',
    defaultValue: 'en',
    toolbar: {
      icon: 'globe',
      items: [
        { value: 'en', title: 'English' },
        { value: 'es', title: 'Español' },
        { value: 'fr', title: 'Français' }
      ],
      showName: true
    }
  }
};