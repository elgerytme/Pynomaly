/**
 * Webpack Configuration for anomaly_detection Web UI Performance Optimization
 * Implements code splitting, tree shaking, and advanced optimization strategies
 */

const path = require('path');
const webpack = require('webpack');
const TerserPlugin = require('terser-webpack-plugin');
const { BundleAnalyzerPlugin } = require('webpack-bundle-analyzer');
const CompressionPlugin = require('compression-webpack-plugin');
const WorkboxPlugin = require('workbox-webpack-plugin');

const isProduction = process.env.NODE_ENV === 'production';
const isAnalyze = process.env.ANALYZE === 'true';

module.exports = {
  mode: isProduction ? 'production' : 'development',

  entry: {
    // Core application entry points
    main: './src/anomaly_detection/presentation/web/static/js/main.js',

    // Vendor libraries (will be split automatically)
    vendor: [
      'd3',
      'echarts',
      'plotly.js-dist-min',
      'htmx.org'
    ],

    // Chart components (code splitting)
    charts: './src/anomaly_detection/presentation/web/static/js/charts/index.js',

    // Dashboard components (code splitting)
    dashboard: './src/anomaly_detection/presentation/web/static/js/components/dashboard-index.js',

    // Utilities and shared modules
    utils: './src/anomaly_detection/presentation/web/static/js/utils/index.js'
  },

  output: {
    path: path.resolve(__dirname, 'src/anomaly_detection/presentation/web/static/js/dist'),
    filename: isProduction ? '[name].[contenthash:8].js' : '[name].js',
    chunkFilename: isProduction ? '[name].[contenthash:8].chunk.js' : '[name].chunk.js',
    clean: true,
    publicPath: '/static/js/dist/',

    // Enable dynamic imports for code splitting
    chunkLoadingGlobal: 'anomaly_detection_chunks'
  },

  resolve: {
    extensions: ['.js', '.mjs'],
    alias: {
      '@': path.resolve(__dirname, 'src/anomaly_detection/presentation/web/static/js'),
      '@charts': path.resolve(__dirname, 'src/anomaly_detection/presentation/web/static/js/charts'),
      '@components': path.resolve(__dirname, 'src/anomaly_detection/presentation/web/static/js/components'),
      '@utils': path.resolve(__dirname, 'src/anomaly_detection/presentation/web/static/js/utils'),
      '@state': path.resolve(__dirname, 'src/anomaly_detection/presentation/web/static/js/state')
    }
  },

  module: {
    rules: [
      {
        test: /\.m?js$/,
        exclude: /node_modules/,
        use: {
          loader: 'babel-loader',
          options: {
            presets: [
              [
                '@babel/preset-env',
                {
                  targets: {
                    browsers: ['> 1%', 'last 2 versions', 'not ie <= 11']
                  },
                  modules: false, // Let webpack handle modules
                  useBuiltIns: 'usage',
                  corejs: 3
                }
              ]
            ],
            plugins: [
              '@babel/plugin-syntax-dynamic-import',
              '@babel/plugin-proposal-class-properties'
            ]
          }
        }
      },
      {
        test: /\.css$/,
        use: ['style-loader', 'css-loader', 'postcss-loader']
      },
      {
        test: /\.(png|jpe?g|gif|svg|woff2?|eot|ttf|otf)$/,
        type: 'asset',
        parser: {
          dataUrlCondition: {
            maxSize: 8 * 1024 // 8kb - inline smaller assets
          }
        },
        generator: {
          filename: 'assets/[hash][ext][query]'
        }
      }
    ]
  },

  optimization: {
    minimize: isProduction,
    minimizer: [
      new TerserPlugin({
        terserOptions: {
          compress: {
            drop_console: isProduction,
            drop_debugger: isProduction,
            pure_funcs: isProduction ? ['console.log', 'console.info'] : []
          },
          mangle: {
            safari10: true
          },
          format: {
            comments: false
          }
        },
        extractComments: false
      })
    ],

    // Advanced code splitting
    splitChunks: {
      chunks: 'all',
      minSize: 20000,
      maxSize: 244000,
      cacheGroups: {
        // Vendor libraries
        vendor: {
          test: /[\\/]node_modules[\\/]/,
          name: 'vendor',
          chunks: 'all',
          priority: 20,
          reuseExistingChunk: true
        },

        // D3 and visualization libraries
        visualization: {
          test: /[\\/]node_modules[\\/](d3|echarts|plotly\.js)[\\/]/,
          name: 'visualization',
          chunks: 'all',
          priority: 30,
          reuseExistingChunk: true
        },

        // HTMX and UI libraries
        ui: {
          test: /[\\/]node_modules[\\/](htmx\.org|alpinejs)[\\/]/,
          name: 'ui',
          chunks: 'all',
          priority: 25,
          reuseExistingChunk: true
        },

        // Shared components
        common: {
          name: 'common',
          minChunks: 2,
          priority: 10,
          chunks: 'all',
          reuseExistingChunk: true
        },

        // Large chart components (lazy loaded)
        charts: {
          test: /[\\/]charts[\\/]/,
          name: 'charts',
          chunks: 'async',
          priority: 15,
          minSize: 30000
        },

        // Dashboard components (lazy loaded)
        dashboard: {
          test: /[\\/]components[\\/](dashboard|advanced)/,
          name: 'dashboard',
          chunks: 'async',
          priority: 15,
          minSize: 20000
        }
      }
    },

    // Runtime chunk for webpack bootstrap code
    runtimeChunk: {
      name: 'runtime'
    },

    // Tree shaking
    usedExports: true,
    sideEffects: false
  },

  plugins: [
    // Define environment variables
    new webpack.DefinePlugin({
      'process.env.NODE_ENV': JSON.stringify(process.env.NODE_ENV || 'development'),
      'process.env.VERSION': JSON.stringify(require('./package.json').version || '1.0.0'),
      '__DEVELOPMENT__': !isProduction,
      '__PRODUCTION__': isProduction
    }),

    // Provide global libraries
    new webpack.ProvidePlugin({
      htmx: 'htmx.org'
    }),

    // Generate service worker with Workbox
    ...(isProduction ? [
      new WorkboxPlugin.GenerateSW({
        clientsClaim: true,
        skipWaiting: true,
        runtimeCaching: [
          {
            urlPattern: /^https:\/\/fonts\.googleapis\.com/,
            handler: 'StaleWhileRevalidate',
            options: {
              cacheName: 'google-fonts-stylesheets'
            }
          },
          {
            urlPattern: /^https:\/\/fonts\.gstatic\.com/,
            handler: 'CacheFirst',
            options: {
              cacheName: 'google-fonts-webfonts',
              expiration: {
                maxEntries: 30,
                maxAgeSeconds: 60 * 60 * 24 * 365 // 1 year
              }
            }
          },
          {
            urlPattern: /\/api\/.*\.(js|css|png|jpg|jpeg|svg|woff2)$/,
            handler: 'CacheFirst',
            options: {
              cacheName: 'static-resources',
              expiration: {
                maxEntries: 100,
                maxAgeSeconds: 60 * 60 * 24 * 30 // 30 days
              }
            }
          },
          {
            urlPattern: /\/api\/(datasets|detectors|health)$/,
            handler: 'StaleWhileRevalidate',
            options: {
              cacheName: 'api-cache',
              expiration: {
                maxEntries: 50,
                maxAgeSeconds: 60 * 60 * 24 // 1 day
              }
            }
          }
        ]
      })
    ] : []),

    // Compression for production builds
    ...(isProduction ? [
      new CompressionPlugin({
        algorithm: 'gzip',
        test: /\.(js|css|html|svg)$/,
        threshold: 10240,
        minRatio: 0.8
      }),
      new CompressionPlugin({
        algorithm: 'brotliCompress',
        test: /\.(js|css|html|svg)$/,
        threshold: 10240,
        minRatio: 0.8,
        filename: '[path][base].br'
      })
    ] : []),

    // Bundle analyzer for development
    ...(isAnalyze ? [
      new BundleAnalyzerPlugin({
        analyzerMode: 'server',
        openAnalyzer: true,
        analyzerHost: 'localhost',
        analyzerPort: 8888
      })
    ] : [])
  ],

  // Development server configuration
  devServer: {
    static: {
      directory: path.join(__dirname, 'src/anomaly_detection/presentation/web/static')
    },
    compress: true,
    port: 9000,
    hot: true,
    liveReload: true,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true
      }
    }
  },

  // Performance budgets
  performance: {
    hints: isProduction ? 'warning' : false,
    maxEntrypointSize: 500000, // 500kb
    maxAssetSize: 300000, // 300kb
    assetFilter: function(assetFilename) {
      return !assetFilename.endsWith('.map');
    }
  },

  // Source maps
  devtool: isProduction ? 'source-map' : 'eval-cheap-module-source-map',

  // Advanced optimizations
  experiments: {
    topLevelAwait: true
  },

  // Cache configuration for faster rebuilds
  cache: {
    type: 'filesystem',
    cacheDirectory: path.resolve(__dirname, '.webpack-cache'),
    buildDependencies: {
      config: [__filename]
    }
  },

  stats: {
    colors: true,
    modules: false,
    chunks: false,
    chunkModules: false,
    assets: false,
    children: false
  }
};
