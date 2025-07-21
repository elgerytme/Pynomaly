/** @type { import('@storybook/html-vite').StorybookConfig } */
const config = {
  stories: [
    '../src/anomaly_detection/presentation/web/**/*.stories.@(js|jsx|ts|tsx|mdx)',
    '../docs/ui/**/*.stories.@(js|jsx|ts|tsx|mdx)',
    '../stories/**/*.stories.@(js|jsx|ts|tsx|mdx)'
  ],

  addons: [
    '@storybook/addon-essentials',
    '@storybook/addon-a11y',
    '@storybook/addon-docs',
    '@storybook/addon-controls',
    '@storybook/addon-viewport',
    '@storybook/addon-backgrounds',
    '@storybook/addon-measure',
    '@storybook/addon-outline',
    '@storybook/addon-interactions',
    '@storybook/addon-toolbars'
  ],

  framework: {
    name: '@storybook/html-vite',
    options: {}
  },

  features: {
    buildStoriesJson: true,
    storyStoreV7: true
  },

  docs: {
    autodocs: 'tag',
    defaultName: 'Documentation'
  },

  staticDirs: [
    '../src/anomaly_detection/presentation/web/static',
    '../docs/ui/assets'
  ],

  viteFinal: async (config) => {
    // Customize Vite config for Storybook
    config.resolve = config.resolve || {};
    config.resolve.alias = {
      ...config.resolve.alias,
      '@': '/src/anomaly_detection/presentation/web/static',
      '@components': '/src/anomaly_detection/presentation/web/static/js/src/components',
      '@utils': '/src/anomaly_detection/presentation/web/static/js/src/utils',
      '@styles': '/src/anomaly_detection/presentation/web/static/css'
    };

    // Add PostCSS support for Tailwind
    const { default: tailwindcss } = await import('tailwindcss');
    const { default: autoprefixer } = await import('autoprefixer');
    config.css = config.css || {};
    config.css.postcss = {
      plugins: [
        tailwindcss,
        autoprefixer
      ]
    };

    return config;
  },

  env: (config) => ({
    ...config,
    STORYBOOK_THEME: 'anomaly_detection'
  }),

  typescript: {
    check: false,
    reactDocgen: 'react-docgen-typescript',
    reactDocgenTypescriptOptions: {
      shouldExtractLiteralValuesFromEnum: true,
      propFilter: (prop) => (prop.parent ? !/node_modules/.test(prop.parent.fileName) : true),
    },
  },
}

export default config;
