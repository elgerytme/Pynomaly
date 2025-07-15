/**
 * Storybook Manager Configuration
 * Customizes the Storybook UI and manager interface
 */

import { addons } from '@storybook/manager-api';
import pynomalyTheme from './theme';

addons.setConfig({
  theme: pynomalyTheme,

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
