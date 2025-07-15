import { defineConfig } from 'vitest/config'
import { resolve } from 'path'

export default defineConfig({
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./tests/ui/setup.ts'],
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html'],
      exclude: [
        'node_modules/',
        'tests/',
        '**/*.d.ts',
        '**/*.config.*',
        '**/dist/**',
        '**/build/**'
      ],
      thresholds: {
        global: {
          branches: 80,
          functions: 80,
          lines: 80,
          statements: 80
        }
      }
    },
    testTimeout: 10000,
    hookTimeout: 10000,
    pool: 'threads',
    poolOptions: {
      threads: {
        singleThread: true
      }
    }
  },
  resolve: {
    alias: {
      '@': resolve(__dirname, '../../src'),
      '@ui': resolve(__dirname, '../../src/pynomaly/presentation/web/static'),
      '@components': resolve(__dirname, '../../src/pynomaly/presentation/web/static/js/components'),
      '@utils': resolve(__dirname, '../../src/pynomaly/presentation/web/static/js/utils'),
      '@tests': resolve(__dirname, './')
    }
  },
  define: {
    'import.meta.vitest': 'undefined'
  }
})
