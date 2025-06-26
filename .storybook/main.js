const path = require('path');

module.exports = {
  stories: [
    '../src/pynomaly/presentation/web/**/*.stories.@(js|jsx|ts|tsx|mdx)',
    '../docs/components/**/*.stories.@(js|jsx|ts|tsx|mdx)',
    '../stories/**/*.stories.@(js|jsx|ts|tsx|mdx)'
  ],
  
  addons: [
    '@storybook/addon-essentials',
    '@storybook/addon-a11y',
    '@storybook/addon-docs',
    '@storybook/addon-controls',
    '@storybook/addon-viewport',
    '@storybook/addon-backgrounds',
    '@storybook/addon-toolbars',
    '@storybook/addon-measure',
    '@storybook/addon-outline',
    '@storybook/addon-interactions'
  ],

  framework: {
    name: '@storybook/html-vite',
    options: {
      vite: {
        resolve: {
          alias: {
            '@': path.resolve(__dirname, '../src/pynomaly/presentation/web'),
            '@components': path.resolve(__dirname, '../src/pynomaly/presentation/web/components'),
            '@static': path.resolve(__dirname, '../src/pynomaly/presentation/web/static')
          }
        }
      }
    }
  },

  core: {
    disableTelemetry: true
  },

  docs: {
    autodocs: 'tag',
    defaultName: 'Documentation'
  },

  typescript: {
    check: false,
    reactDocgen: 'react-docgen-typescript',
    reactDocgenTypescriptOptions: {
      shouldExtractLiteralValuesFromEnum: true,
      propFilter: (prop) => (prop.parent ? !/node_modules/.test(prop.parent.fileName) : true),
    },
  },

  features: {
    interactionsDebugger: true,
    buildStoriesJson: true
  },

  staticDirs: [
    '../src/pynomaly/presentation/web/static',
    { from: '../src/pynomaly/presentation/web/assets', to: '/assets' }
  ]
};