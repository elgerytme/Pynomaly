/**
 * Pynomaly Storybook Theme
 * Custom theming for the Storybook interface
 */

import { create } from '@storybook/theming/create';

export default create({
  base: 'light',

  // Brand information
  brandTitle: 'Pynomaly Design System',
  brandUrl: 'https://github.com/pynomaly/pynomaly',
  brandImage: '/static/images/pynomaly-logo.svg',
  brandTarget: '_self',

  // Color palette
  colorPrimary: '#0ea5e9',      // Primary blue
  colorSecondary: '#78716c',    // Warm gray

  // UI colors
  appBg: '#fafafa',            // Light background
  appContentBg: '#ffffff',     // Content background
  appBorderColor: '#e5e5e5',   // Border color
  appBorderRadius: 8,          // Border radius

  // Typography
  fontBase: '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
  fontCode: '"JetBrains Mono", "Fira Code", "SF Mono", Consolas, "Liberation Mono", Menlo, Courier, monospace',

  // Text colors
  textColor: '#171717',        // Primary text
  textInverseColor: '#ffffff', // Inverse text
  textMutedColor: '#737373',   // Muted text

  // Toolbar default and active colors
  barTextColor: '#525252',     // Toolbar text
  barSelectedColor: '#0ea5e9', // Selected toolbar item
  barBg: '#ffffff',            // Toolbar background

  // Form colors
  inputBg: '#ffffff',
  inputBorder: '#d4d4d4',
  inputTextColor: '#171717',
  inputBorderRadius: 6,

  // Button colors
  buttonBg: '#0ea5e9',
  buttonBorder: '#0ea5e9',

  // Boolean colors
  booleanBg: '#f3f4f6',
  booleanSelectedBg: '#0ea5e9',
});
