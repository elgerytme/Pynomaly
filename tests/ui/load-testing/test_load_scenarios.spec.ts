/**
 * Comprehensive Load Testing Scenarios for Pynomaly Web UI
 *
 * This test suite implements various load testing scenarios to ensure
 * the web interface can handle multiple concurrent users and high traffic.
 */

import { test, expect, chromium, Browser, BrowserContext, Page } from '@playwright/test';

interface LoadTestConfig {
  concurrentUsers: number;
  testDuration: number; // milliseconds
  rampUpTime: number; // milliseconds
  actions: LoadTestAction[];
}

interface LoadTestAction {
  name: string;
  weight: number; // probability weight
  execute: (page: Page) => Promise<void>;
}

interface LoadTestResults {
  totalRequests: number;
  successfulRequests: number;
  failedRequests: number;
  averageResponseTime: number;
  maxResponseTime: number;
  minResponseTime: number;
  requestsPerSecond: number;
  errors: LoadTestError[];
}

interface LoadTestError {
  timestamp: number;
  action: string;
  error: string;
  userIndex: number;
}

class LoadTestRunner {
  private config: LoadTestConfig;
  private results: LoadTestResults;
  private startTime: number;
  private browsers: Browser[] = [];
  private contexts: BrowserContext[] = [];
  private pages: Page[] = [];

  constructor(config: LoadTestConfig) {
    this.config = config;
    this.results = {
      totalRequests: 0,
      successfulRequests: 0,
      failedRequests: 0,
      averageResponseTime: 0,
      maxResponseTime: 0,
      minResponseTime: Infinity,
      requestsPerSecond: 0,
      errors: []
    };
    this.startTime = Date.now();
  }

  async setup(): Promise<void> {
    // Create browsers and contexts for concurrent users
    for (let i = 0; i < this.config.concurrentUsers; i++) {
      const browser = await chromium.launch({
        headless: true,
        args: [
          '--no-sandbox',
          '--disable-dev-shm-usage',
          '--disable-background-timer-throttling',
          '--disable-backgrounding-occluded-windows',
          '--disable-renderer-backgrounding'
        ]
      });

      const context = await browser.newContext({
        viewport: { width: 1280, height: 720 },
        userAgent: `LoadTestUser-${i}`,
        extraHTTPHeaders: {
          'X-Load-Test-User': `${i}`,
          'X-Load-Test-Session': `${this.startTime}`
        }
      });

      const page = await context.newPage();

      // Add request/response monitoring
      page.on('request', (request) => {
        this.results.totalRequests++;
      });

      page.on('response', (response) => {
        const timing = response.request().timing();
        if (timing) {
          const responseTime = timing.responseEnd - timing.requestStart;
          this.updateResponseTimeStats(responseTime);
        }

        if (response.status() >= 400) {
          this.results.failedRequests++;
          this.results.errors.push({
            timestamp: Date.now(),
            action: 'HTTP_ERROR',
            error: `${response.status()} ${response.statusText()} - ${response.url()}`,
            userIndex: parseInt(page.context().browser()?.contexts().indexOf(page.context()).toString() || '0')
          });
        } else {
          this.results.successfulRequests++;
        }
      });

      page.on('pageerror', (error) => {
        this.results.errors.push({
          timestamp: Date.now(),
          action: 'PAGE_ERROR',
          error: error.message,
          userIndex: parseInt(page.context().browser()?.contexts().indexOf(page.context()).toString() || '0')
        });
      });

      this.browsers.push(browser);
      this.contexts.push(context);
      this.pages.push(page);
    }
  }

  private updateResponseTimeStats(responseTime: number): void {
    this.results.maxResponseTime = Math.max(this.results.maxResponseTime, responseTime);
    this.results.minResponseTime = Math.min(this.results.minResponseTime, responseTime);

    // Calculate rolling average
    const currentAvg = this.results.averageResponseTime;
    const totalSuccessful = this.results.successfulRequests;
    this.results.averageResponseTime = (currentAvg * (totalSuccessful - 1) + responseTime) / totalSuccessful;
  }

  async runLoadTest(): Promise<LoadTestResults> {
    await this.setup();

    const testPromises: Promise<void>[] = [];

    // Start users with ramp-up
    for (let i = 0; i < this.config.concurrentUsers; i++) {
      const delayMs = (this.config.rampUpTime / this.config.concurrentUsers) * i;

      testPromises.push(
        this.runUserSession(this.pages[i], i, delayMs)
      );
    }

    // Wait for all user sessions to complete
    await Promise.allSettled(testPromises);

    // Calculate final metrics
    const totalTime = Date.now() - this.startTime;
    this.results.requestsPerSecond = (this.results.totalRequests / totalTime) * 1000;

    await this.cleanup();
    return this.results;
  }

  private async runUserSession(page: Page, userIndex: number, delayMs: number): Promise<void> {
    try {
      // Ramp-up delay
      if (delayMs > 0) {
        await new Promise(resolve => setTimeout(resolve, delayMs));
      }

      const endTime = this.startTime + this.config.testDuration;

      while (Date.now() < endTime) {
        const action = this.selectRandomAction();

        try {
          const actionStartTime = Date.now();
          await action.execute(page);
          const actionDuration = Date.now() - actionStartTime;

          // Add realistic think time between actions
          const thinkTime = Math.random() * 2000 + 1000; // 1-3 seconds
          await page.waitForTimeout(thinkTime);

        } catch (error) {
          this.results.errors.push({
            timestamp: Date.now(),
            action: action.name,
            error: error instanceof Error ? error.message : String(error),
            userIndex
          });
        }
      }
    } catch (error) {
      this.results.errors.push({
        timestamp: Date.now(),
        action: 'USER_SESSION',
        error: error instanceof Error ? error.message : String(error),
        userIndex
      });
    }
  }

  private selectRandomAction(): LoadTestAction {
    const totalWeight = this.config.actions.reduce((sum, action) => sum + action.weight, 0);
    let randomWeight = Math.random() * totalWeight;

    for (const action of this.config.actions) {
      randomWeight -= action.weight;
      if (randomWeight <= 0) {
        return action;
      }
    }

    return this.config.actions[0]; // fallback
  }

  private async cleanup(): Promise<void> {
    for (const context of this.contexts) {
      await context.close();
    }
    for (const browser of this.browsers) {
      await browser.close();
    }
  }
}

// Define load testing scenarios
const dashboardNavigationActions: LoadTestAction[] = [
  {
    name: 'visit_dashboard',
    weight: 30,
    execute: async (page: Page) => {
      await page.goto('/dashboard', { waitUntil: 'networkidle', timeout: 10000 });
      await page.waitForSelector('[data-testid="dashboard-content"]', { timeout: 5000 });
    }
  },
  {
    name: 'navigate_to_detectors',
    weight: 20,
    execute: async (page: Page) => {
      await page.click('[data-testid="detectors-tab"]');
      await page.waitForSelector('[data-testid="detectors-list"]', { timeout: 5000 });
    }
  },
  {
    name: 'navigate_to_datasets',
    weight: 20,
    execute: async (page: Page) => {
      await page.click('[data-testid="datasets-tab"]');
      await page.waitForSelector('[data-testid="datasets-list"]', { timeout: 5000 });
    }
  },
  {
    name: 'navigate_to_visualizations',
    weight: 15,
    execute: async (page: Page) => {
      await page.click('[data-testid="visualizations-tab"]');
      await page.waitForSelector('[data-testid="chart-container"]', { timeout: 5000 });
    }
  },
  {
    name: 'refresh_dashboard',
    weight: 10,
    execute: async (page: Page) => {
      if (await page.isVisible('[data-testid="refresh-button"]')) {
        await page.click('[data-testid="refresh-button"]');
        await page.waitForTimeout(2000);
      }
    }
  },
  {
    name: 'toggle_realtime',
    weight: 5,
    execute: async (page: Page) => {
      if (await page.isVisible('[data-testid="realtime-toggle"]')) {
        await page.click('[data-testid="realtime-toggle"]');
        await page.waitForTimeout(1000);
      }
    }
  }
];

const dataInteractionActions: LoadTestAction[] = [
  {
    name: 'create_detector',
    weight: 25,
    execute: async (page: Page) => {
      await page.goto('/detectors', { waitUntil: 'networkidle' });
      if (await page.isVisible('[data-testid="create-detector"]')) {
        await page.click('[data-testid="create-detector"]');
        await page.waitForSelector('[data-testid="detector-form"]', { timeout: 3000 });
        await page.fill('input[name="name"]', `LoadTest-${Date.now()}`);
        await page.selectOption('select[name="algorithm"]', 'IsolationForest');
        // Don't actually submit to avoid creating test data
      }
    }
  },
  {
    name: 'upload_dataset',
    weight: 20,
    execute: async (page: Page) => {
      await page.goto('/datasets', { waitUntil: 'networkidle' });
      if (await page.isVisible('[data-testid="upload-dataset"]')) {
        await page.click('[data-testid="upload-dataset"]');
        await page.waitForSelector('[data-testid="upload-form"]', { timeout: 3000 });
        // Simulate form interaction without actual upload
        await page.fill('input[name="dataset-name"]', `TestDataset-${Date.now()}`);
      }
    }
  },
  {
    name: 'view_detection_results',
    weight: 30,
    execute: async (page: Page) => {
      await page.goto('/detection', { waitUntil: 'networkidle' });
      if (await page.isVisible('[data-testid="results-list"]')) {
        const firstResult = await page.locator('[data-testid="result-item"]').first();
        if (await firstResult.isVisible()) {
          await firstResult.click();
          await page.waitForTimeout(2000);
        }
      }
    }
  },
  {
    name: 'filter_and_search',
    weight: 15,
    execute: async (page: Page) => {
      if (await page.isVisible('[data-testid="search-input"]')) {
        await page.fill('[data-testid="search-input"]', 'test');
        await page.waitForTimeout(1000);
        await page.fill('[data-testid="search-input"]', '');
      }
    }
  },
  {
    name: 'sort_data',
    weight: 10,
    execute: async (page: Page) => {
      if (await page.isVisible('[data-testid="sort-button"]')) {
        await page.click('[data-testid="sort-button"]');
        await page.waitForTimeout(500);
      }
    }
  }
];

test.describe('Load Testing Scenarios', () => {
  test('Light load - Normal traffic simulation', async () => {
    const config: LoadTestConfig = {
      concurrentUsers: 5,
      testDuration: 30000, // 30 seconds
      rampUpTime: 5000,    // 5 seconds
      actions: dashboardNavigationActions
    };

    const runner = new LoadTestRunner(config);
    const results = await runner.runLoadTest();

    console.log('Light Load Test Results:', {
      totalRequests: results.totalRequests,
      successRate: `${((results.successfulRequests / results.totalRequests) * 100).toFixed(2)}%`,
      avgResponseTime: `${results.averageResponseTime.toFixed(2)}ms`,
      requestsPerSecond: results.requestsPerSecond.toFixed(2),
      errors: results.errors.length
    });

    // Assertions for light load
    expect(results.successfulRequests / results.totalRequests).toBeGreaterThan(0.95); // 95% success rate
    expect(results.averageResponseTime).toBeLessThan(2000); // Under 2 seconds
    expect(results.errors.length).toBeLessThan(5); // Minimal errors
  });

  test('Medium load - Peak usage simulation', async () => {
    const config: LoadTestConfig = {
      concurrentUsers: 15,
      testDuration: 60000, // 1 minute
      rampUpTime: 10000,   // 10 seconds
      actions: [...dashboardNavigationActions, ...dataInteractionActions]
    };

    const runner = new LoadTestRunner(config);
    const results = await runner.runLoadTest();

    console.log('Medium Load Test Results:', {
      totalRequests: results.totalRequests,
      successRate: `${((results.successfulRequests / results.totalRequests) * 100).toFixed(2)}%`,
      avgResponseTime: `${results.averageResponseTime.toFixed(2)}ms`,
      requestsPerSecond: results.requestsPerSecond.toFixed(2),
      errors: results.errors.length
    });

    // Assertions for medium load
    expect(results.successfulRequests / results.totalRequests).toBeGreaterThan(0.90); // 90% success rate
    expect(results.averageResponseTime).toBeLessThan(5000); // Under 5 seconds
    expect(results.requestsPerSecond).toBeGreaterThan(1); // At least 1 RPS
  });

  test('Heavy load - Stress testing', async () => {
    const config: LoadTestConfig = {
      concurrentUsers: 25,
      testDuration: 90000, // 1.5 minutes
      rampUpTime: 15000,   // 15 seconds
      actions: dataInteractionActions
    };

    const runner = new LoadTestRunner(config);
    const results = await runner.runLoadTest();

    console.log('Heavy Load Test Results:', {
      totalRequests: results.totalRequests,
      successRate: `${((results.successfulRequests / results.totalRequests) * 100).toFixed(2)}%`,
      avgResponseTime: `${results.averageResponseTime.toFixed(2)}ms`,
      maxResponseTime: `${results.maxResponseTime.toFixed(2)}ms`,
      requestsPerSecond: results.requestsPerSecond.toFixed(2),
      errors: results.errors.length
    });

    // Assertions for heavy load (more lenient)
    expect(results.successfulRequests / results.totalRequests).toBeGreaterThan(0.80); // 80% success rate
    expect(results.averageResponseTime).toBeLessThan(10000); // Under 10 seconds
    expect(results.maxResponseTime).toBeLessThan(30000); // Under 30 seconds max
  });

  test('WebSocket load testing', async () => {
    const websocketActions: LoadTestAction[] = [
      {
        name: 'start_realtime_monitoring',
        weight: 40,
        execute: async (page: Page) => {
          await page.goto('/dashboard', { waitUntil: 'networkidle' });
          if (await page.isVisible('[data-testid="start-realtime"]')) {
            await page.click('[data-testid="start-realtime"]');
            await page.waitForTimeout(3000); // Let WebSocket establish
          }
        }
      },
      {
        name: 'websocket_message_handling',
        weight: 30,
        execute: async (page: Page) => {
          await page.evaluate(() => {
            // Simulate WebSocket message load
            if ((window as any).anomalyWebSocket) {
              const ws = (window as any).anomalyWebSocket;
              if (ws.readyState === WebSocket.OPEN) {
                for (let i = 0; i < 5; i++) {
                  ws.send(JSON.stringify({
                    type: 'load_test',
                    timestamp: Date.now(),
                    data: { value: Math.random() }
                  }));
                }
              }
            }
          });
          await page.waitForTimeout(1000);
        }
      },
      {
        name: 'stop_realtime_monitoring',
        weight: 30,
        execute: async (page: Page) => {
          if (await page.isVisible('[data-testid="stop-realtime"]')) {
            await page.click('[data-testid="stop-realtime"]');
            await page.waitForTimeout(1000);
          }
        }
      }
    ];

    const config: LoadTestConfig = {
      concurrentUsers: 10,
      testDuration: 45000, // 45 seconds
      rampUpTime: 5000,    // 5 seconds
      actions: websocketActions
    };

    const runner = new LoadTestRunner(config);
    const results = await runner.runLoadTest();

    console.log('WebSocket Load Test Results:', {
      totalRequests: results.totalRequests,
      successRate: `${((results.successfulRequests / results.totalRequests) * 100).toFixed(2)}%`,
      avgResponseTime: `${results.averageResponseTime.toFixed(2)}ms`,
      errors: results.errors.length
    });

    // WebSocket-specific assertions
    expect(results.successfulRequests / results.totalRequests).toBeGreaterThan(0.85);
    expect(results.errors.filter(e => e.action.includes('websocket')).length).toBeLessThan(3);
  });

  test('Mobile load testing', async () => {
    const mobileConfig: LoadTestConfig = {
      concurrentUsers: 8,
      testDuration: 30000,
      rampUpTime: 4000,
      actions: dashboardNavigationActions.filter(action =>
        !action.name.includes('keyboard') // Remove desktop-specific actions
      )
    };

    // Override to use mobile viewport
    const runner = new class extends LoadTestRunner {
      async setup(): Promise<void> {
        for (let i = 0; i < this.config.concurrentUsers; i++) {
          const browser = await chromium.launch({ headless: true });
          const context = await browser.newContext({
            ...chromium.devices['iPhone 12'],
            userAgent: `MobileLoadTestUser-${i}`
          });

          const page = await context.newPage();
          // Add monitoring as in parent class
          // ... (monitoring setup code)

          this.browsers.push(browser);
          this.contexts.push(context);
          this.pages.push(page);
        }
      }
    }(mobileConfig);

    const results = await runner.runLoadTest();

    console.log('Mobile Load Test Results:', {
      totalRequests: results.totalRequests,
      successRate: `${((results.successfulRequests / results.totalRequests) * 100).toFixed(2)}%`,
      avgResponseTime: `${results.averageResponseTime.toFixed(2)}ms`
    });

    // Mobile-specific assertions (slightly more lenient due to device simulation)
    expect(results.successfulRequests / results.totalRequests).toBeGreaterThan(0.88);
    expect(results.averageResponseTime).toBeLessThan(6000);
  });
});
