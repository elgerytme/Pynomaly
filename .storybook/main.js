/** @type { import('@storybook/html-vite').StorybookConfig } */
const config = {
  stories: [
    '../src/pynomaly/presentation/web/**/*.stories.@(js|jsx|ts|tsx|mdx)',
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
    '../src/pynomaly/presentation/web/static',
    '../docs/ui/assets'
  ],
  
  viteFinal: async (config) => {
    // Customize Vite config for Storybook
    config.resolve = config.resolve || {};
    config.resolve.alias = {
      ...config.resolve.alias,
      '@': '/src/pynomaly/presentation/web/static',
      '@components': '/src/pynomaly/presentation/web/static/js/src/components',
      '@utils': '/src/pynomaly/presentation/web/static/js/src/utils',
      '@styles': '/src/pynomaly/presentation/web/static/css'
    };
    
    // Add PostCSS support for Tailwind
    config.css = config.css || {};
    config.css.postcss = {
      plugins: [
        require('tailwindcss'),
        require('autoprefixer')
      ]
    };
    
    return config;
  },
  
  env: (config) => ({
    ...config,
    STORYBOOK_THEME: 'pynomaly'
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