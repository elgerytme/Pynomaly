import { defineConfig } from 'vite';
import { VitePWA } from 'vite-plugin-pwa';
import { resolve } from 'path';
import { compressionPlugin } from 'vite-plugin-compression';

export default defineConfig({
  root: 'src/pynomaly/presentation/web',
  base: '/static/',
  
  build: {
    outDir: 'static/dist',
    emptyOutDir: true,
    sourcemap: true,
    minify: 'terser',
    terserOptions: {
      compress: {
        drop_console: true,
        drop_debugger: true,
      },
    },
    rollupOptions: {
      input: {
        main: resolve(__dirname, 'src/pynomaly/presentation/web/static/js/src/main.js'),
        'auth-service': resolve(__dirname, 'src/pynomaly/presentation/web/static/js/src/services/auth-service.js'),
        'pwa-service': resolve(__dirname, 'src/pynomaly/presentation/web/static/js/src/services/pwa-service.js'),
        'offline-dashboard': resolve(__dirname, 'src/pynomaly/presentation/web/static/js/src/components/offline-dashboard.js'),
        'sync-manager': resolve(__dirname, 'src/pynomaly/presentation/web/static/js/src/utils/sync-manager.js'),
      },
      output: {
        entryFileNames: '[name].[hash].js',
        chunkFileNames: '[name].[hash].js',
        assetFileNames: '[name].[hash].[ext]',
        manualChunks: {
          vendor: ['htmx.org', 'd3', 'echarts'],
          utils: ['fuse.js', 'sortablejs'],
          auth: ['alpinejs']
        }
      }
    },
    // Enable tree shaking
    treeshake: {
      moduleSideEffects: false
    },
    // Optimize chunk size
    chunkSizeWarningLimit: 1000,
    // Enable CSS code splitting
    cssCodeSplit: true,
    // Asset inlining threshold
    assetsInlineLimit: 4096,
  },

  css: {
    postcss: {
      plugins: [
        require('tailwindcss'),
        require('autoprefixer'),
        require('cssnano')({
          preset: 'default'
        })
      ]
    }
  },

  plugins: [
    // PWA Plugin
    VitePWA({
      registerType: 'autoUpdate',
      workbox: {
        globPatterns: ['**/*.{js,css,html,ico,png,svg,woff2}'],
        runtimeCaching: [
          {
            urlPattern: /^https:\/\/fonts\.googleapis\.com\//,
            handler: 'CacheFirst',
            options: {
              cacheName: 'google-fonts-stylesheets',
              expiration: {
                maxEntries: 10,
                maxAgeSeconds: 60 * 60 * 24 * 365 // 1 year
              }
            }
          },
          {
            urlPattern: /^https:\/\/fonts\.gstatic\.com\//,
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
            urlPattern: /^\/api\/(?!auth|detection|analysis|train|predict|upload|real-time)/,
            handler: 'StaleWhileRevalidate',
            options: {
              cacheName: 'api-cache',
              expiration: {
                maxEntries: 50,
                maxAgeSeconds: 60 * 60 * 24 // 24 hours
              }
            }
          },
          {
            urlPattern: /^\/api\/(detection|analysis|train|predict|upload|real-time)/,
            handler: 'NetworkFirst',
            options: {
              cacheName: 'api-critical',
              expiration: {
                maxEntries: 20,
                maxAgeSeconds: 60 * 60 // 1 hour
              }
            }
          }
        ]
      },
      manifest: {
        name: 'Pynomaly - Anomaly Detection Platform',
        short_name: 'Pynomaly',
        description: 'Advanced anomaly detection and machine learning platform',
        theme_color: '#0ea5e9',
        background_color: '#ffffff',
        display: 'standalone',
        orientation: 'portrait-primary',
        scope: '/',
        start_url: '/',
        icons: [
          {
            src: '/static/images/pynomaly-icon-192.png',
            sizes: '192x192',
            type: 'image/png'
          },
          {
            src: '/static/images/pynomaly-icon-512.png',
            sizes: '512x512',
            type: 'image/png'
          },
          {
            src: '/static/images/pynomaly-icon-maskable.png',
            sizes: '512x512',
            type: 'image/png',
            purpose: 'maskable'
          }
        ]
      }
    }),

    // Compression plugins
    compressionPlugin({
      algorithm: 'gzip',
      ext: '.gz',
      threshold: 1024,
      deleteOriginFile: false,
      filter: /\.(js|css|html|svg|json|woff2?)$/i
    }),
    
    compressionPlugin({
      algorithm: 'brotliCompress',
      ext: '.br',
      threshold: 1024,
      deleteOriginFile: false,
      filter: /\.(js|css|html|svg|json|woff2?)$/i
    })
  ],

  server: {
    port: 3000,
    host: '0.0.0.0',
    cors: true,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        secure: false
      }
    }
  },

  preview: {
    port: 3001,
    host: '0.0.0.0'
  },

  optimizeDeps: {
    include: ['htmx.org', 'd3', 'echarts', 'fuse.js', 'sortablejs', 'alpinejs']
  },

  define: {
    __APP_VERSION__: JSON.stringify(process.env.npm_package_version),
    __BUILD_DATE__: JSON.stringify(new Date().toISOString()),
    __PROD__: JSON.stringify(process.env.NODE_ENV === 'production')
  }
});
