/**
 * Playwright Configuration for Pynomaly UI Testing
 * Production-ready setup with comprehensive browser testing, visual regression, and CI/CD integration
 */

import { defineConfig, devices } from '@playwright/test';

/**
 * Read environment variables from .env files if they exist
 */
require('dotenv').config();

/**
 * See https://playwright.dev/docs/test-configuration
 */
export default defineConfig({
  testDir: './tests/ui',
  
  /* Run tests in files in parallel */
  fullyParallel: true,
  
  /* Fail the build on CI if you accidentally left test.only in the source code */
  forbidOnly: !!process.env.CI,
  
  /* Retry on CI only */
  retries: process.env.CI ? 2 : 0,
  
  /* Opt out of parallel tests on CI */
  workers: process.env.CI ? 1 : undefined,
  
  /* Reporter to use. See https://playwright.dev/docs/test-reporters */
  reporter: [
    ['html', { outputFolder: 'test_reports/playwright-report' }],
    ['json', { outputFile: 'test_reports/test-results.json' }],
    ['junit', { outputFile: 'test_reports/junit.xml' }],
    process.env.CI ? ['github'] : ['list']
  ],
  
  /* Shared settings for all the projects below */
  use: {
    /* Base URL to use in actions like `await page.goto('/')` */
    baseURL: process.env.PYNOMALY_BASE_URL || 'http://localhost:8000',
    
    /* Collect trace when retrying the failed test */
    trace: 'on-first-retry',
    
    /* Record video only on failure */
    video: 'retain-on-failure',
    
    /* Take screenshot on failure */
    screenshot: 'only-on-failure',
    
    /* Maximum time each action can take */
    actionTimeout: 10000,
    
    /* Maximum time each test can run */
    timeout: 30000,
  },

  /* Configure projects for major browsers */
  projects: [
    {
      name: 'chromium',
      use: { 
        ...devices['Desktop Chrome'],
        // Enable additional chrome features for testing
        launchOptions: {
          args: ['--disable-web-security', '--allow-running-insecure-content']
        }
      },
    },

    {
      name: 'firefox',
      use: { ...devices['Desktop Firefox'] },
    },

    {
      name: 'webkit',
      use: { ...devices['Desktop Safari'] },
    },

    /* Test against mobile viewports */
    {
      name: 'Mobile Chrome',
      use: { ...devices['Pixel 5'] },
    },
    {
      name: 'Mobile Safari',
      use: { ...devices['iPhone 12'] },
    },

    /* Test against branded browsers */
    {
      name: 'Microsoft Edge',
      use: { ...devices['Desktop Edge'], channel: 'msedge' },
    },
    {
      name: 'Google Chrome',
      use: { ...devices['Desktop Chrome'], channel: 'chrome' },
    },

    /* Tablet testing */
    {
      name: 'iPad',
      use: { ...devices['iPad Pro'] },
    },

    /* High-DPI testing */
    {
      name: 'High-DPI',
      use: {
        ...devices['Desktop Chrome'],
        viewport: { width: 1920, height: 1080 },
        deviceScaleFactor: 2
      }
    },

    /* Accessibility testing setup */
    {
      name: 'accessibility',
      use: {
        ...devices['Desktop Chrome'],
        // Add axe-core for accessibility testing
        actionTimeout: 15000
      }
    }
  ],

  /* Run your local dev server before starting the tests */
  webServer: process.env.CI ? undefined : {
    command: 'python -m uvicorn src.pynomaly.presentation.api.app:app --host 0.0.0.0 --port 8000',
    url: 'http://127.0.0.1:8000/api/health',
    reuseExistingServer: !process.env.CI,
    timeout: 30000,
    env: {
      PYNOMALY_ENVIRONMENT: 'testing',
      PYNOMALY_LOG_LEVEL: 'WARNING',
      PYNOMALY_AUTH_ENABLED: 'false',
      PYNOMALY_DOCS_ENABLED: 'true',
      PYNOMALY_CORS_ENABLED: 'true'
    }
  },

  /* Global setup and teardown */
  globalSetup: './tests/ui/global-setup.ts',
  globalTeardown: './tests/ui/global-teardown.ts',

  /* Expect configuration */
  expect: {
    /* Maximum time expect() should wait for the condition to be met */
    timeout: 10000,
    
    /* Threshold for visual comparisons */
    toHaveScreenshot: { 
      threshold: 0.2,
      animations: 'disabled'
    },
    toMatchSnapshot: { 
      threshold: 0.2 
    }
  },

  /* Test metadata and tags */
  metadata: {
    'test-type': 'UI Automation',
    'framework': 'Playwright',
    'application': 'Pynomaly',
    'environment': process.env.PYNOMALY_ENVIRONMENT || 'testing'
  }
});