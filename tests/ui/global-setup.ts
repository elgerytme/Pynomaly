import { chromium, FullConfig } from '@playwright/test';

/**
 * Global setup for cross-browser testing
 * Prepares test environment and shared resources
 */
async function globalSetup(config: FullConfig) {
  console.log('🔧 Setting up cross-browser testing environment...');

  // Ensure test reports directory exists
  const fs = require('fs');
  const path = require('path');

  const reportDirs = [
    'test_reports/playwright-report',
    'test_reports/cross-browser',
    'test_reports/visual-regression',
    'test_reports/device-compatibility',
    'test_reports/performance'
  ];

  for (const dir of reportDirs) {
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
    }
  }

  // Check if the web server is running
  const baseURL = config.webServer?.url || 'http://localhost:8000';
  console.log(`🌐 Checking web server at ${baseURL}...`);

  try {
    const browser = await chromium.launch();
    const page = await browser.newPage();
    await page.goto(baseURL, { timeout: 30000 });

    // Check if the page loads correctly
    const title = await page.title();
    console.log(`✅ Web server is ready - Page title: "${title}"`);

    // Verify critical elements exist
    const hasNavigation = await page.locator('nav').count() > 0;
    const hasMain = await page.locator('main').count() > 0;

    if (!hasNavigation || !hasMain) {
      console.warn('⚠️  Warning: Page structure may be incomplete');
    }

    await browser.close();
  } catch (error) {
    console.error('❌ Failed to connect to web server:', error.message);
    throw new Error(`Web server is not accessible at ${baseURL}`);
  }

  // Initialize browser capabilities detection
  await initializeBrowserCapabilities();

  // Setup cross-browser test data
  await setupCrossBrowserTestData();

  console.log('✅ Global setup completed successfully');
}

/**
 * Initialize browser capabilities detection
 */
async function initializeBrowserCapabilities() {
  console.log('🔍 Initializing browser capabilities detection...');

  const capabilities = {
    timestamp: new Date().toISOString(),
    browsers: {}
  };

  // Test basic capabilities across browsers
  const browsers = [
    { name: 'chromium', engine: chromium },
    // Firefox and WebKit would be added here if available
  ];

  for (const { name, engine } of browsers) {
    try {
      const browser = await engine.launch();
      const context = await browser.newContext();
      const page = await context.newPage();

      // Test basic JavaScript capabilities
      const jsCapabilities = await page.evaluate(() => {
        return {
          es6: typeof Symbol !== 'undefined',
          es2017: typeof Promise.resolve().finally === 'function',
          es2018: typeof Object.fromEntries === 'function',
          webgl: !!document.createElement('canvas').getContext('webgl'),
          webgl2: !!document.createElement('canvas').getContext('webgl2'),
          serviceWorker: 'serviceWorker' in navigator,
          pushManager: 'PushManager' in window,
          indexedDB: 'indexedDB' in window,
          localStorage: 'localStorage' in window,
          sessionStorage: 'sessionStorage' in window,
          geolocation: 'geolocation' in navigator,
          deviceMotion: 'DeviceMotionEvent' in window,
          touchEvents: 'ontouchstart' in window,
          pointerEvents: 'PointerEvent' in window,
          intersectionObserver: 'IntersectionObserver' in window,
          mutationObserver: 'MutationObserver' in window,
          customElements: 'customElements' in window,
          shadowDOM: 'attachShadow' in Element.prototype,
          webComponents: 'customElements' in window && 'attachShadow' in Element.prototype
        };
      });

      capabilities.browsers[name] = {
        available: true,
        capabilities: jsCapabilities,
        userAgent: await page.evaluate(() => navigator.userAgent)
      };

      await browser.close();
      console.log(`  ✅ ${name} capabilities detected`);
    } catch (error) {
      capabilities.browsers[name] = {
        available: false,
        error: error.message
      };
      console.log(`  ❌ ${name} not available: ${error.message}`);
    }
  }

  // Save capabilities for test reference
  const fs = require('fs');
  const capabilitiesPath = 'test_reports/cross-browser/capabilities.json';
  fs.writeFileSync(capabilitiesPath, JSON.stringify(capabilities, null, 2));
}

/**
 * Setup cross-browser test data
 */
async function setupCrossBrowserTestData() {
  console.log('📄 Setting up cross-browser test data...');

  const testData = {
    viewport: {
      desktop: { width: 1920, height: 1080 },
      tablet: { width: 768, height: 1024 },
      mobile: { width: 375, height: 667 }
    },
    breakpoints: {
      sm: 640,
      md: 768,
      lg: 1024,
      xl: 1280,
      '2xl': 1536
    },
    testUrls: [
      '/',
      '/dashboard',
      '/datasets',
      '/detectors',
      '/results',
      '/monitor'
    ],
    sampleData: {
      csvData: generateSampleCSVData(),
      anomalyData: generateSampleAnomalyData(),
      userPreferences: generateSampleUserPreferences()
    }
  };

  const fs = require('fs');
  const testDataPath = 'test_reports/cross-browser/test-data.json';
  fs.writeFileSync(testDataPath, JSON.stringify(testData, null, 2));
}

/**
 * Generate sample CSV data for testing
 */
function generateSampleCSVData(): string {
  const headers = ['timestamp', 'value', 'category'];
  const rows = [];

  for (let i = 0; i < 100; i++) {
    const timestamp = new Date(Date.now() - (i * 60000)).toISOString();
    const value = Math.random() * 100 + (Math.random() > 0.9 ? 50 : 0); // Occasional anomalies
    const category = ['normal', 'warning', 'critical'][Math.floor(Math.random() * 3)];
    rows.push([timestamp, value.toFixed(2), category]);
  }

  return [headers, ...rows].map(row => row.join(',')).join('\n');
}

/**
 * Generate sample anomaly detection data
 */
function generateSampleAnomalyData() {
  return {
    datasets: [
      {
        id: 'sample-1',
        name: 'Sample Dataset 1',
        size: 1000,
        anomalies: 12,
        confidence: 0.85
      },
      {
        id: 'sample-2',
        name: 'Sample Dataset 2',
        size: 5000,
        anomalies: 45,
        confidence: 0.92
      }
    ],
    algorithms: [
      { name: 'Isolation Forest', accuracy: 0.89, speed: 'fast' },
      { name: 'One-Class SVM', accuracy: 0.85, speed: 'medium' },
      { name: 'Local Outlier Factor', accuracy: 0.87, speed: 'slow' }
    ],
    results: {
      precision: 0.86,
      recall: 0.84,
      f1Score: 0.85,
      auc: 0.91
    }
  };
}

/**
 * Generate sample user preferences
 */
function generateSampleUserPreferences() {
  return {
    theme: 'light',
    language: 'en',
    notifications: true,
    autoRefresh: true,
    defaultAlgorithm: 'isolation-forest',
    chartType: 'line',
    dataRetention: 30,
    accessibility: {
      highContrast: false,
      reducedMotion: false,
      screenReader: false
    }
  };
}

export default globalSetup;
