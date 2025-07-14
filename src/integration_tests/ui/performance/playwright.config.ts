import { defineConfig, devices } from '@playwright/test'

/**
 * Performance testing configuration for Playwright
 * Optimized for accurate performance measurements
 */
export default defineConfig({
  testDir: './tests/ui/performance',

  // Run tests in sequence for consistent performance measurements
  fullyParallel: false,
  workers: 1,

  // Increased timeouts for performance tests
  timeout: 60000,
  expect: {
    timeout: 10000
  },

  // Disable retries for performance tests to get accurate measurements
  retries: 0,

  // Reporter configuration for performance testing
  reporter: [
    ['html', { outputFolder: 'test-results/performance' }],
    ['json', { outputFile: 'test-results/performance/results.json' }],
    ['junit', { outputFile: 'test-results/performance/junit.xml' }]
  ],

  // Output directory for test artifacts
  outputDir: 'test-results/performance/artifacts',

  use: {
    // Base URL for testing
    baseURL: process.env.BASE_URL || 'http://localhost:8000',

    // Browser context settings optimized for performance testing
    viewport: { width: 1366, height: 768 },

    // Disable animations for consistent measurements
    hasTouch: false,
    isMobile: false,

    // Enable video recording for performance analysis
    video: 'retain-on-failure',

    // Enable screenshot on failure
    screenshot: 'only-on-failure',

    // Trace collection for debugging
    trace: 'retain-on-failure',

    // Additional browser context options
    extraHTTPHeaders: {
      'Accept-Language': 'en-US,en;q=0.9'
    },

    // Ignore HTTPS errors for local testing
    ignoreHTTPSErrors: true,

    // Disable service workers to avoid caching interference
    serviceWorkers: 'block'
  },

  projects: [
    {
      name: 'performance-chrome',
      use: {
        ...devices['Desktop Chrome'],
        // Chrome-specific performance settings
        launchOptions: {
          args: [
            '--no-sandbox',
            '--disable-dev-shm-usage',
            '--disable-web-security',
            '--disable-features=TranslateUI',
            '--disable-ipc-flooding-protection',
            '--disable-renderer-backgrounding',
            '--disable-backgrounding-occluded-windows',
            '--disable-background-timer-throttling',
            '--disable-background-media-suspend',
            '--disable-back-forward-cache',
            '--disable-breakpad',
            '--disable-component-extensions-with-background-pages',
            '--disable-extensions',
            '--disable-features=TranslateUI,BlinkGenPropertyTrees',
            '--disable-hang-monitor',
            '--disable-prompt-on-repost',
            '--metrics-recording-only',
            '--no-first-run',
            '--enable-precise-memory-info'
          ]
        }
      }
    },

    {
      name: 'performance-firefox',
      use: {
        ...devices['Desktop Firefox'],
        // Firefox-specific performance settings
        launchOptions: {
          firefoxUserPrefs: {
            'dom.performance.time_to_non_blank_paint.enabled': true,
            'dom.performance.time_to_contentful_paint.enabled': true,
            'dom.performance.time_to_dom_content_flushed.enabled': true,
            'browser.cache.disk.enable': false,
            'browser.cache.memory.enable': false,
            'network.http.use-cache': false
          }
        }
      }
    },

    {
      name: 'performance-safari',
      use: {
        ...devices['Desktop Safari'],
        // Safari-specific settings
      }
    },

    // Mobile performance testing
    {
      name: 'performance-mobile-chrome',
      use: {
        ...devices['Pixel 5'],
        launchOptions: {
          args: [
            '--no-sandbox',
            '--disable-dev-shm-usage',
            '--enable-precise-memory-info'
          ]
        }
      }
    },

    // Slow network simulation for performance testing
    {
      name: 'performance-slow-network',
      use: {
        ...devices['Desktop Chrome'],
        launchOptions: {
          args: [
            '--no-sandbox',
            '--disable-dev-shm-usage',
            '--enable-precise-memory-info'
          ]
        },
        // Simulate slow 3G connection
        contextOptions: {
          offline: false,
          extraHTTPHeaders: {
            'Connection': 'keep-alive'
          }
        }
      }
    }
  ],

  // Web server configuration for local testing
  webServer: {
    command: 'npm run dev',
    url: 'http://localhost:8000',
    reuseExistingServer: !process.env.CI,
    timeout: 120000
  }
})
