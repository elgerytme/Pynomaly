import { defineConfig, devices } from '@playwright/test';

/**
 * Comprehensive Playwright Configuration for Cross-Browser Testing
 * Supports multiple browsers, devices, and testing scenarios
 */
export default defineConfig({
  // Test directory
  testDir: './tests/ui',
  
  // Global test timeout
  timeout: 30 * 1000,
  
  // Test expectations timeout
  expect: {
    timeout: 5000,
  },

  // Fail the build on CI if you accidentally left test.only in the source code
  forbidOnly: !!process.env.CI,
  
  // Retry on CI only
  retries: process.env.CI ? 2 : 0,
  
  // Opt out of parallel tests on CI
  workers: process.env.CI ? 1 : undefined,
  
  // Reporter configuration
  reporter: [
    ['html', { outputFolder: 'test_reports/playwright-report' }],
    ['json', { outputFile: 'test_reports/playwright-results.json' }],
    ['junit', { outputFile: 'test_reports/playwright-junit.xml' }],
    ['line'],
    // Custom reporter for cross-browser analysis
    ['./tests/ui/reporters/cross-browser-reporter.ts']
  ],

  // Global test configuration
  use: {
    // Base URL for tests
    baseURL: process.env.BASE_URL || 'http://localhost:8000',
    
    // Collect trace when retrying the failed test
    trace: 'on-first-retry',
    
    // Record video on failure
    video: 'retain-on-failure',
    
    // Take screenshot on failure
    screenshot: 'only-on-failure',
    
    // Browser context options
    locale: 'en-US',
    timezoneId: 'America/New_York',
    
    // Viewport for desktop tests (will be overridden by device-specific configs)
    viewport: { width: 1280, height: 720 },
    
    // Ignore HTTPS errors
    ignoreHTTPSErrors: true,
  },

  // Project configurations for different browsers and devices
  projects: [
    // === DESKTOP BROWSERS ===
    {
      name: 'Desktop Chrome',
      use: { 
        ...devices['Desktop Chrome'],
        viewport: { width: 1920, height: 1080 },
        contextOptions: {
          permissions: ['clipboard-read', 'clipboard-write', 'notifications']
        }
      },
      testMatch: [
        '**/test_*.spec.ts',
        '**/cross-browser/*.spec.ts',
        '**/responsive/*.spec.ts'
      ]
    },

    {
      name: 'Desktop Firefox',
      use: { 
        ...devices['Desktop Firefox'],
        viewport: { width: 1920, height: 1080 },
        contextOptions: {
          permissions: ['clipboard-read', 'clipboard-write', 'notifications']
        }
      },
      testMatch: [
        '**/test_*.spec.ts',
        '**/cross-browser/*.spec.ts',
        '**/responsive/*.spec.ts'
      ]
    },

    {
      name: 'Desktop Safari',
      use: { 
        ...devices['Desktop Safari'],
        viewport: { width: 1920, height: 1080 }
      },
      testMatch: [
        '**/test_*.spec.ts',
        '**/cross-browser/*.spec.ts',
        '**/responsive/*.spec.ts'
      ]
    },

    {
      name: 'Desktop Edge',
      use: { 
        ...devices['Desktop Edge'],
        viewport: { width: 1920, height: 1080 },
        contextOptions: {
          permissions: ['clipboard-read', 'clipboard-write', 'notifications']
        }
      },
      testMatch: [
        '**/test_*.spec.ts',
        '**/cross-browser/*.spec.ts',
        '**/responsive/*.spec.ts'
      ]
    },

    // === MOBILE DEVICES ===
    {
      name: 'Mobile Chrome',
      use: { 
        ...devices['Pixel 5'],
        contextOptions: {
          permissions: ['geolocation', 'notifications']
        }
      },
      testMatch: [
        '**/mobile/*.spec.ts',
        '**/responsive/*.spec.ts',
        '**/touch/*.spec.ts'
      ]
    },

    {
      name: 'Mobile Safari',
      use: { 
        ...devices['iPhone 12'],
        contextOptions: {
          permissions: ['geolocation', 'notifications']
        }
      },
      testMatch: [
        '**/mobile/*.spec.ts',
        '**/responsive/*.spec.ts',
        '**/touch/*.spec.ts'
      ]
    },

    {
      name: 'Mobile Firefox',
      use: { 
        ...devices['Pixel 5'],
        channel: 'firefox',
        contextOptions: {
          permissions: ['geolocation', 'notifications']
        }
      },
      testMatch: [
        '**/mobile/*.spec.ts',
        '**/responsive/*.spec.ts'
      ]
    },

    // === TABLET DEVICES ===
    {
      name: 'iPad',
      use: { 
        ...devices['iPad Pro'],
        contextOptions: {
          permissions: ['geolocation', 'notifications']
        }
      },
      testMatch: [
        '**/tablet/*.spec.ts',
        '**/responsive/*.spec.ts',
        '**/touch/*.spec.ts'
      ]
    },

    {
      name: 'Android Tablet',
      use: { 
        ...devices['Galaxy Tab S4'],
        contextOptions: {
          permissions: ['geolocation', 'notifications']
        }
      },
      testMatch: [
        '**/tablet/*.spec.ts',
        '**/responsive/*.spec.ts',
        '**/touch/*.spec.ts'
      ]
    },

    // === HIGH DPI DISPLAYS ===
    {
      name: 'Desktop Chrome HiDPI',
      use: { 
        ...devices['Desktop Chrome HiDPI'],
        viewport: { width: 1920, height: 1080 }
      },
      testMatch: [
        '**/responsive/*.spec.ts',
        '**/visual-regression/*.spec.ts'
      ]
    },

    // === ACCESSIBILITY TESTING ===
    {
      name: 'Accessibility Chrome',
      use: { 
        ...devices['Desktop Chrome'],
        viewport: { width: 1280, height: 720 },
        contextOptions: {
          forcedColors: 'active', // Test high contrast mode
          reducedMotion: 'reduce'  // Test reduced motion preference
        }
      },
      testMatch: [
        '**/accessibility/*.spec.ts'
      ]
    },

    // === PERFORMANCE TESTING ===
    {
      name: 'Performance Chrome',
      use: { 
        ...devices['Desktop Chrome'],
        viewport: { width: 1920, height: 1080 },
        contextOptions: {
          permissions: ['performance-observer-entry']
        }
      },
      testMatch: [
        '**/performance/*.spec.ts'
      ]
    },

    // === SLOW NETWORK SIMULATION ===
    {
      name: 'Slow Network Chrome',
      use: { 
        ...devices['Desktop Chrome'],
        viewport: { width: 1280, height: 720 },
        contextOptions: {
          offline: false,
          // Simulate slow 3G connection
          extraHTTPHeaders: {
            'Connection': 'slow-3g'
          }
        }
      },
      testMatch: [
        '**/network/*.spec.ts',
        '**/offline/*.spec.ts'
      ]
    },

    // === OLDER BROWSER VERSIONS ===
    {
      name: 'Legacy Chrome',
      use: { 
        channel: 'chrome',
        viewport: { width: 1280, height: 720 },
        // Simulate older Chrome capabilities
        contextOptions: {
          javascriptEnabled: true,
          acceptDownloads: false
        }
      },
      testMatch: [
        '**/legacy/*.spec.ts',
        '**/progressive-enhancement/*.spec.ts'
      ]
    }
  ],

  // Web server configuration for local testing
  webServer: {
    command: 'npm run dev',
    url: 'http://localhost:8000',
    reuseExistingServer: !process.env.CI,
    timeout: 120 * 1000
  },

  // Global setup and teardown
  globalSetup: require.resolve('./tests/ui/global-setup.ts'),
  globalTeardown: require.resolve('./tests/ui/global-teardown.ts'),

  // Test metadata
  metadata: {
    'test-type': 'cross-browser-e2e',
    'framework': 'playwright',
    'ci': process.env.CI || 'false'
  }
});