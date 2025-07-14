/**
 * Lighthouse CI Configuration for Pynomaly
 * Performance monitoring and Core Web Vitals tracking
 */

module.exports = {
  ci: {
    // Collection settings
    collect: {
      // URLs to test
      url: [
        'http://localhost:8000/',
        'http://localhost:8000/detection',
        'http://localhost:8000/detectors',
        'http://localhost:8000/datasets',
        'http://localhost:8000/visualizations'
      ],

      // Number of runs per URL
      numberOfRuns: 3,

      // Chrome settings
      settings: {
        chromeFlags: [
          '--no-sandbox',
          '--disable-dev-shm-usage',
          '--disable-gpu',
          '--disable-extensions'
        ],

        // Lighthouse configuration
        preset: 'desktop',

        // Custom throttling for CI
        throttling: {
          rttMs: 40,
          throughputKbps: 10240,
          cpuSlowdownMultiplier: 1
        },

        // Extended configuration for Pynomaly
        onlyCategories: ['performance', 'accessibility', 'best-practices', 'pwa'],
        skipAudits: ['uses-http2'], // Skip HTTP/2 audit for local testing
      }
    },

    // Assertion settings for performance budgets
    assert: {
      // Performance budgets
      assertions: {
        // Core Web Vitals
        'categories:performance': ['error', { minScore: 0.8 }],
        'categories:accessibility': ['error', { minScore: 0.9 }],
        'categories:best-practices': ['error', { minScore: 0.8 }],
        'categories:seo': ['error', { minScore: 0.8 }],
        'categories:pwa': ['warn', { minScore: 0.6 }],

        // Specific metrics
        'metrics:first-contentful-paint': ['error', { maxNumericValue: 2000 }],
        'metrics:largest-contentful-paint': ['error', { maxNumericValue: 2500 }],
        'metrics:cumulative-layout-shift': ['error', { maxNumericValue: 0.1 }],
        'metrics:total-blocking-time': ['error', { maxNumericValue: 300 }],
        'metrics:speed-index': ['error', { maxNumericValue: 3000 }],
        'metrics:interactive': ['error', { maxNumericValue: 3000 }],

        // Resource efficiency
        'metrics:total-byte-weight': ['warn', { maxNumericValue: 1000000 }], // 1MB
        'metrics:unused-css-rules': ['warn', { maxNumericValue: 50000 }],   // 50KB
        'metrics:unused-javascript': ['warn', { maxNumericValue: 100000 }], // 100KB

        // Security and best practices
        'audits:uses-https': 'error',
        'audits:is-on-https': 'error',
        'audits:redirects-http': 'error',
        'audits:uses-long-cache-ttl': 'warn',
        'audits:efficient-animated-content': 'warn',

        // Accessibility
        'audits:color-contrast': 'error',
        'audits:focus-traps': 'error',
        'audits:focusable-controls': 'error',
        'audits:heading-order': 'error',
        'audits:label': 'error',
        'audits:link-name': 'error'
      }
    },

    // Upload settings
    upload: {
      // GitHub target for status checks
      target: 'temporary-public-storage',

      // Server settings (if using LHCI server)
      // serverBaseUrl: 'https://your-lhci-server.com',
      // token: process.env.LHCI_TOKEN
    },

    // Server settings for temporary storage
    server: {
      port: 9001,
      storage: {
        storageMethod: 'filesystem',
        storagePath: '.lighthouseci'
      }
    },

    // Wizard settings
    wizard: {
      // GitHub repository configuration
      github: {
        token: process.env.LHCI_GITHUB_APP_TOKEN
      }
    }
  }
};
