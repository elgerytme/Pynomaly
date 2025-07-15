/**
 * Memory Leak Testing Configuration for Pynomaly
 * Specialized configuration for memory monitoring and leak detection
 */

import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './tests/ui/memory-leak-testing',
  timeout: 120000, // Memory tests may take longer

  expect: {
    timeout: 15000, // Allow more time for memory operations
  },

  retries: process.env.CI ? 1 : 0, // Limited retries for memory tests
  workers: 1, // Run memory tests sequentially to avoid interference

  reporter: [
    ['html', { outputFolder: 'test_reports/memory-leak-report' }],
    ['json', { outputFile: 'test_reports/memory-leak-results.json' }],
    ['junit', { outputFile: 'test_reports/memory-leak-junit.xml' }],
  ],

  use: {
    baseURL: process.env.BASE_URL || 'http://localhost:8000',
    trace: 'retain-on-failure',
    video: 'retain-on-failure',
    screenshot: 'only-on-failure',

    // Memory leak testing specific settings
    launchOptions: {
      args: [
        '--enable-precise-memory-info', // Enable performance.memory API
        '--js-flags="--expose-gc"', // Expose garbage collection
        '--max-old-space-size=4096', // Limit memory to detect leaks faster
        '--no-sandbox',
        '--disable-dev-shm-usage',
      ]
    }
  },

  // Memory testing projects
  projects: [
    {
      name: 'memory-leak-chrome',
      use: {
        browserName: 'chromium',
        viewport: { width: 1280, height: 720 },
        launchOptions: {
          args: [
            '--enable-precise-memory-info',
            '--js-flags="--expose-gc"',
            '--max-old-space-size=2048',
            '--no-sandbox',
          ]
        }
      },
      testMatch: '**/memory-leak-*.spec.ts',
    },

    {
      name: 'performance-monitoring',
      use: {
        browserName: 'chromium',
        viewport: { width: 1280, height: 720 },
        launchOptions: {
          args: [
            '--enable-precise-memory-info',
            '--js-flags="--expose-gc"',
            '--enable-web-platform-features',
            '--no-sandbox',
          ]
        }
      },
      testMatch: '**/performance-monitoring.spec.ts',
    }
  ],

  globalSetup: require.resolve('./memory-global-setup.ts'),
  globalTeardown: require.resolve('./memory-global-teardown.ts'),
});
