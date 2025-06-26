import { addons } from '@storybook/addons';
import { themes } from '@storybook/theming';

// Custom Pynomaly theme
const pynomaly_theme = {
  ...themes.light,
  
  // Color palette
  colorPrimary: '#0ea5e9',
  colorSecondary: '#78716c',
  
  // UI colors
  appBg: '#fafafa',
  appContentBg: '#ffffff',
  appBorderColor: '#e5e5e5',
  appBorderRadius: 6,
  
  // Typography
  fontBase: '"Inter", "Helvetica Neue", Helvetica, Arial, sans-serif',
  fontCode: '"JetBrains Mono", "Fira Code", Consolas, monospace',
  
  // Text colors
  textColor: '#171717',
  textInverseColor: '#ffffff',
  textMutedColor: '#737373',
  
  // Toolbar colors
  barTextColor: '#525252',
  barSelectedColor: '#0ea5e9',
  barBg: '#ffffff',
  
  // Form colors
  inputBg: '#ffffff',
  inputBorder: '#d4d4d4',
  inputTextColor: '#171717',
  inputBorderRadius: 4,
  
  // Brand
  brandTitle: 'Pynomaly Design System',
  brandUrl: 'https://github.com/pynomaly/pynomaly',
  brandImage: '/static/icons/icon-192x192.png',
  brandTarget: '_self',
};

addons.setConfig({
  theme: pynomaly_theme,
  
  // Panel configuration
  panelPosition: 'bottom',
  selectedPanel: 'controls',
  
  // Sidebar configuration
  sidebar: {
    showRoots: true,
    collapsedRoots: ['other']
  },
  
  // Toolbar configuration
  toolbar: {
    title: { hidden: false },
    zoom: { hidden: false },
    eject: { hidden: false },
    copy: { hidden: false },
    fullscreen: { hidden: false },
    'storybook/background': { hidden: false },
    'storybook/viewport': { hidden: false },
    'storybook/toolbars': { hidden: false }
  },
  
  // Enable keyboard shortcuts
  enableShortcuts: true,
  
  // Show addon panel by default
  showPanel: true,
  
  // Initial active tab
  initialActive: 'sidebar'
});