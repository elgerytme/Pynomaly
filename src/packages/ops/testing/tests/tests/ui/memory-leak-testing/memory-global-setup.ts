/**
 * Memory Leak Testing Global Setup
 * Configures browser environment for memory monitoring and leak detection
 */

import { chromium, FullConfig } from '@playwright/test';

async function globalSetup(config: FullConfig) {
  console.log('🔧 Setting up memory leak testing environment...');

  const baseURL = process.env.BASE_URL || 'http://localhost:8000';

  // Launch browser with memory monitoring flags
  const browser = await chromium.launch({
    args: [
      '--enable-precise-memory-info',
      '--js-flags="--expose-gc"',
      '--max-old-space-size=2048',
      '--no-sandbox',
      '--disable-dev-shm-usage',
    ]
  });

  const page = await browser.newPage();

  try {
    // 1. Check if HTTP server is running
    console.log('🌐 Checking HTTP server availability...');
    const response = await page.goto(`${baseURL}/api/v1/health`);
    if (!response || !response.ok()) {
      throw new Error(`HTTP server not available at ${baseURL}`);
    }
    console.log('✅ HTTP server is running');

    // 2. Setup memory monitoring environment
    console.log('🧠 Configuring memory monitoring...');
    await configureMemoryMonitoring(page, baseURL);
    console.log('✅ Memory monitoring configured');

    // 3. Clear any existing data that might affect tests
    console.log('🧹 Clearing existing test data...');
    await clearTestData(page, baseURL);
    console.log('✅ Test data cleared');

    // 4. Set up performance monitoring
    console.log('📊 Setting up performance monitoring...');
    await setupPerformanceMonitoring(page, baseURL);
    console.log('✅ Performance monitoring setup complete');

  } catch (error) {
    console.error('❌ Memory leak testing setup failed:', error);
    throw error;
  } finally {
    await browser.close();
  }

  console.log('🎉 Memory leak testing environment setup complete!');
}

async function configureMemoryMonitoring(page: any, baseURL: string) {
  // Enable memory monitoring features
  const config = {
    memory: {
      enable_monitoring: true,
      gc_threshold: 50, // MB
      monitoring_interval: 5000, // 5 seconds
      alert_threshold: 100 // MB
    },
    performance: {
      enable_profiling: true,
      track_long_tasks: true,
      track_layout_shifts: true,
      track_first_contentful_paint: true
    },
    testing: {
      mock_data_enabled: true,
      performance_mode: true,
      debug_memory_usage: true
    }
  };

  try {
    await page.evaluate(async ({ baseURL, config }) => {
      const response = await fetch(`${baseURL}/api/v1/config/memory`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config)
      });
      return response.ok;
    }, { baseURL, config });
  } catch (error) {
    console.warn('Warning: Could not configure memory monitoring:', error);
    // Continue with defaults
  }
}

async function clearTestData(page: any, baseURL: string) {
  // Clear any test artifacts that might interfere with memory testing
  const testDataTypes = [
    'memory_test_detectors',
    'memory_test_datasets',
    'memory_test_sessions',
    'temp_visualizations'
  ];

  for (const dataType of testDataTypes) {
    try {
      await page.evaluate(async ({ baseURL, dataType }) => {
        const response = await fetch(`${baseURL}/api/v1/test/cleanup/${dataType}`, {
          method: 'DELETE'
        });
        return response.ok || response.status === 404;
      }, { baseURL, dataType });
    } catch (error) {
      console.warn(`Warning: Could not clear ${dataType}:`, error);
    }
  }

  // Clear browser cache and storage
  try {
    await page.context().clearCookies();
    await page.evaluate(() => {
      if (typeof Storage !== 'undefined') {
        localStorage.clear();
        sessionStorage.clear();
      }
    });
  } catch (error) {
    console.warn('Warning: Could not clear browser storage:', error);
  }
}

async function setupPerformanceMonitoring(page: any, baseURL: string) {
  // Set up performance monitoring endpoints
  const performanceConfig = {
    endpoints: {
      memory_stats: '/api/v1/monitoring/memory',
      performance_stats: '/api/v1/monitoring/performance',
      gc_stats: '/api/v1/monitoring/gc'
    },
    thresholds: {
      memory_leak_mb: 10,
      memory_growth_rate: 0.1, // 10% growth per minute
      dom_node_limit: 10000,
      event_listener_limit: 1000
    },
    reporting: {
      auto_capture_heap_snapshot: true,
      generate_timeline: true,
      track_allocations: true
    }
  };

  try {
    await page.evaluate(async ({ baseURL, performanceConfig }) => {
      const response = await fetch(`${baseURL}/api/v1/config/performance`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(performanceConfig)
      });
      return response.ok;
    }, { baseURL, performanceConfig });
  } catch (error) {
    console.warn('Warning: Could not setup performance monitoring:', error);
  }

  // Initialize client-side performance observers
  await page.addInitScript(() => {
    // Set up performance observers for memory leak detection
    if ('PerformanceObserver' in window) {
      // Monitor long tasks
      const longTaskObserver = new PerformanceObserver((list) => {
        list.getEntries().forEach((entry) => {
          if (entry.duration > 50) {
            console.warn('Long task detected:', {
              duration: entry.duration,
              startTime: entry.startTime,
              name: entry.name
            });
          }
        });
      });

      try {
        longTaskObserver.observe({ entryTypes: ['longtask'] });
      } catch (e) {
        console.warn('Long task observer not supported');
      }

      // Monitor layout shifts
      const layoutShiftObserver = new PerformanceObserver((list) => {
        list.getEntries().forEach((entry) => {
          if (entry.value > 0.1) {
            console.warn('Significant layout shift detected:', {
              value: entry.value,
              startTime: entry.startTime
            });
          }
        });
      });

      try {
        layoutShiftObserver.observe({ entryTypes: ['layout-shift'] });
      } catch (e) {
        console.warn('Layout shift observer not supported');
      }
    }

    // Global memory monitoring
    window.memoryMonitoringEnabled = true;
    window.memoryBaseline = null;

    // Set up periodic memory checks
    if ((performance as any).memory) {
      window.memoryBaseline = (performance as any).memory.usedJSHeapSize;

      setInterval(() => {
        const currentMemory = (performance as any).memory.usedJSHeapSize;
        const growth = currentMemory - window.memoryBaseline;
        const growthMB = growth / (1024 * 1024);

        if (growthMB > 50) { // 50MB growth threshold
          console.warn('Significant memory growth detected:', {
            baseline: `${(window.memoryBaseline / 1024 / 1024).toFixed(2)}MB`,
            current: `${(currentMemory / 1024 / 1024).toFixed(2)}MB`,
            growth: `${growthMB.toFixed(2)}MB`
          });
        }
      }, 30000); // Check every 30 seconds
    }

    // Override console methods to capture memory-related warnings
    window.memoryWarnings = [];
    window.memoryErrors = [];

    const originalWarn = console.warn;
    const originalError = console.error;

    console.warn = (...args) => {
      const message = args.join(' ');
      if (message.toLowerCase().includes('memory') ||
          message.toLowerCase().includes('leak') ||
          message.toLowerCase().includes('heap')) {
        window.memoryWarnings.push({
          message,
          timestamp: Date.now(),
          stack: new Error().stack
        });
      }
      originalWarn.apply(console, args);
    };

    console.error = (...args) => {
      const message = args.join(' ');
      if (message.toLowerCase().includes('memory') ||
          message.toLowerCase().includes('leak') ||
          message.toLowerCase().includes('heap')) {
        window.memoryErrors.push({
          message,
          timestamp: Date.now(),
          stack: new Error().stack
        });
      }
      originalError.apply(console, args);
    };
  });
}

export default globalSetup;
