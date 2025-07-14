module.exports = {
  ci: {
    collect: {
      url: [
        'http://localhost:8000',
        'http://localhost:8000/dashboard',
        'http://localhost:8000/datasets',
        'http://localhost:8000/detectors',
        'http://localhost:8000/results',
        'http://localhost:8000/monitor'
      ],
      numberOfRuns: 3,
      settings: {
        chromeFlags: '--no-sandbox --disable-dev-shm-usage',
        preset: 'desktop',
        throttling: {
          rttMs: 40,
          throughputKbps: 10240,
          cpuSlowdownMultiplier: 1,
          requestLatencyMs: 0,
          downloadThroughputKbps: 0,
          uploadThroughputKbps: 0
        },
        emulatedFormFactor: 'desktop',
        locale: 'en-US'
      }
    },
    upload: {
      target: 'filesystem',
      outputDir: './test_reports/lighthouse'
    },
    assert: {
      assertions: {
        'categories:performance': ['error', { minScore: 0.85 }],
        'categories:accessibility': ['error', { minScore: 0.95 }],
        'categories:best-practices': ['error', { minScore: 0.90 }],
        'categories:seo': ['error', { minScore: 0.85 }],
        'categories:pwa': ['error', { minScore: 0.80 }],

        // Core Web Vitals
        'first-contentful-paint': ['error', { maxNumericValue: 2000 }],
        'largest-contentful-paint': ['error', { maxNumericValue: 2500 }],
        'first-input-delay': ['error', { maxNumericValue: 100 }],
        'cumulative-layout-shift': ['error', { maxNumericValue: 0.1 }],
        'speed-index': ['error', { maxNumericValue: 3000 }],
        'total-blocking-time': ['error', { maxNumericValue: 300 }],

        // Resource Optimization
        'unused-css-rules': ['warn', { maxLength: 5 }],
        'unused-javascript': ['warn', { maxLength: 5 }],
        'unminified-css': ['error', { maxLength: 0 }],
        'unminified-javascript': ['error', { maxLength: 0 }],
        'render-blocking-resources': ['warn', { maxLength: 3 }],

        // Network Optimization
        'uses-text-compression': 'error',
        'uses-rel-preconnect': 'warn',
        'uses-rel-preload': 'warn',
        'efficient-animated-content': 'warn',
        'modern-image-formats': 'warn',
        'uses-optimized-images': 'warn',
        'uses-responsive-images': 'warn',

        // JavaScript Performance
        'bootup-time': ['warn', { maxNumericValue: 4000 }],
        'mainthread-work-breakdown': ['warn', { maxNumericValue: 4000 }],
        'dom-size': ['warn', { maxNumericValue: 800 }],

        // Accessibility
        'color-contrast': 'error',
        'heading-order': 'error',
        'link-name': 'error',
        'button-name': 'error',
        'image-alt': 'error',
        'aria-valid-attr': 'error',
        'aria-required-attr': 'error',

        // PWA
        'installable-manifest': 'error',
        'service-worker': 'error',
        'works-offline': 'warn',
        'viewport': 'error',
        'themed-omnibox': 'warn',
        'maskable-icon': 'warn',

        // Security & Best Practices
        'is-on-https': 'error',
        'redirects-http': 'error',
        'uses-http2': 'warn',
        'no-vulnerable-libraries': 'error',
        'csp-xss': 'warn'
      }
    },
    server: {
      port: 9001,
      storage: './test_reports/lighthouse/.lighthouseci'
    }
  }
};
