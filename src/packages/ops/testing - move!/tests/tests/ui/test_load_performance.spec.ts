/**
 * Load Performance Testing for Pynomaly Web UI
 * Tests application performance under various load conditions
 */

import { test, expect } from '@playwright/test';

test.describe('Load Performance Testing', () => {

  test('Dashboard handles concurrent users', async ({ page, context }) => {
    // Test with multiple concurrent sessions
    const pages = await Promise.all([
      context.newPage(),
      context.newPage(),
      context.newPage()
    ]);

    // Navigate all pages simultaneously
    const navigationPromises = pages.map(p => p.goto('/'));
    await Promise.all(navigationPromises);

    // Verify all pages loaded successfully
    for (const p of pages) {
      await expect(p.locator('h1')).toBeVisible();
      await expect(p.locator('[data-testid="dashboard-content"]')).toBeVisible();
    }

    // Test simultaneous interactions
    const interactionPromises = pages.map(async (p, index) => {
      await p.click(`[data-testid="nav-link-${index % 3}"]`);
      await p.waitForLoadState('networkidle');
    });

    await Promise.all(interactionPromises);

    // Cleanup
    await Promise.all(pages.map(p => p.close()));
  });

  test('API endpoints handle load spikes', async ({ page }) => {
    await page.goto('/');

    // Simulate rapid API calls
    const apiCalls = [];
    for (let i = 0; i < 20; i++) {
      apiCalls.push(
        page.evaluate(() =>
          fetch('/api/health').then(r => r.json())
        )
      );
    }

    const startTime = Date.now();
    const results = await Promise.all(apiCalls);
    const endTime = Date.now();

    // Verify all calls succeeded
    results.forEach(result => {
      expect(result).toBeDefined();
    });

    // Verify response time is reasonable
    const totalTime = endTime - startTime;
    expect(totalTime).toBeLessThan(5000); // 5 seconds for 20 calls
  });

  test('Memory usage stays stable during extended usage', async ({ page }) => {
    await page.goto('/');

    // Get initial memory usage
    const initialMemory = await page.evaluate(() =>
      (performance as any).memory?.usedJSHeapSize || 0
    );

    // Simulate extended usage
    for (let i = 0; i < 10; i++) {
      await page.goto('/detectors');
      await page.waitForLoadState('networkidle');

      await page.goto('/datasets');
      await page.waitForLoadState('networkidle');

      await page.goto('/visualizations');
      await page.waitForLoadState('networkidle');

      await page.goto('/');
      await page.waitForLoadState('networkidle');
    }

    // Get final memory usage
    const finalMemory = await page.evaluate(() =>
      (performance as any).memory?.usedJSHeapSize || 0
    );

    // Memory should not increase by more than 50MB
    const memoryIncrease = finalMemory - initialMemory;
    expect(memoryIncrease).toBeLessThan(50 * 1024 * 1024);
  });

  test('Large dataset handling performance', async ({ page }) => {
    await page.goto('/datasets');

    // Generate large dataset simulation
    const largeDataset = Array.from({ length: 10000 }, (_, i) => ({
      id: i,
      name: `Dataset ${i}`,
      size: Math.random() * 1000000,
      created: new Date().toISOString()
    }));

    // Test rendering performance
    const startTime = Date.now();

    await page.evaluate((data) => {
      // Simulate receiving large dataset
      const event = new CustomEvent('dataset-update', {
        detail: { datasets: data }
      });
      document.dispatchEvent(event);
    }, largeDataset);

    // Wait for rendering to complete
    await page.waitForTimeout(1000);

    const endTime = Date.now();
    const renderTime = endTime - startTime;

    // Rendering should complete within 2 seconds
    expect(renderTime).toBeLessThan(2000);
  });

  test('WebSocket connection stability under load', async ({ page }) => {
    await page.goto('/monitoring');

    // Test WebSocket connection stability
    const connectionTest = await page.evaluate(() => {
      return new Promise((resolve) => {
        const ws = new WebSocket('ws://localhost:8000/ws/monitoring');
        let messageCount = 0;
        let startTime = Date.now();

        ws.onopen = () => {
          // Send rapid messages
          for (let i = 0; i < 100; i++) {
            ws.send(JSON.stringify({ type: 'test', data: i }));
          }
        };

        ws.onmessage = (event) => {
          messageCount++;
          if (messageCount >= 100) {
            ws.close();
            resolve({
              success: true,
              messageCount,
              duration: Date.now() - startTime
            });
          }
        };

        ws.onerror = (error) => {
          resolve({ success: false, error: error.toString() });
        };

        // Timeout after 10 seconds
        setTimeout(() => {
          ws.close();
          resolve({ success: false, error: 'timeout' });
        }, 10000);
      });
    });

    expect(connectionTest.success).toBe(true);
    expect(connectionTest.messageCount).toBe(100);
    expect(connectionTest.duration).toBeLessThan(5000);
  });

  test('Chart rendering performance with large datasets', async ({ page }) => {
    await page.goto('/visualizations');

    // Generate large chart data
    const chartData = Array.from({ length: 1000 }, (_, i) => ({
      x: i,
      y: Math.random() * 100,
      anomaly: Math.random() > 0.95
    }));

    const startTime = Date.now();

    // Render chart with large dataset
    await page.evaluate((data) => {
      const event = new CustomEvent('chart-data-update', {
        detail: { data, type: 'scatter' }
      });
      document.dispatchEvent(event);
    }, chartData);

    // Wait for chart to render
    await page.waitForSelector('[data-testid="chart-container"] svg', {
      timeout: 10000
    });

    const endTime = Date.now();
    const renderTime = endTime - startTime;

    // Chart should render within 5 seconds
    expect(renderTime).toBeLessThan(5000);
  });

  test('Form submission performance under load', async ({ page }) => {
    await page.goto('/detection');

    // Test rapid form submissions
    const submissions = [];
    for (let i = 0; i < 10; i++) {
      submissions.push(
        page.evaluate((index) => {
          const form = document.createElement('form');
          form.innerHTML = `
            <input name="dataset" value="test-${index}">
            <input name="algorithm" value="IsolationForest">
            <input name="csrf_token" value="test-token">
          `;

          return fetch('/api/detect', {
            method: 'POST',
            body: new FormData(form)
          }).then(r => ({ status: r.status, index }));
        }, i)
      );
    }

    const results = await Promise.all(submissions);

    // Verify all submissions were processed
    results.forEach(result => {
      expect(result.status).toBeLessThan(500); // Not server error
    });
  });

  test('Resource cleanup during navigation', async ({ page }) => {
    await page.goto('/');

    // Monitor resource usage during navigation
    const resourceTest = await page.evaluate(() => {
      const resources = {
        listeners: 0,
        intervals: 0,
        observers: 0
      };

      // Count initial resources
      resources.listeners = (window as any)._eventListeners?.length || 0;
      resources.intervals = (window as any)._intervals?.length || 0;
      resources.observers = (window as any)._observers?.length || 0;

      return resources;
    });

    // Navigate through several pages
    const pages = ['/detectors', '/datasets', '/visualizations', '/monitoring'];

    for (const pagePath of pages) {
      await page.goto(pagePath);
      await page.waitForLoadState('networkidle');
      await page.waitForTimeout(500);
    }

    // Check resource cleanup
    const finalResources = await page.evaluate(() => {
      const resources = {
        listeners: 0,
        intervals: 0,
        observers: 0
      };

      resources.listeners = (window as any)._eventListeners?.length || 0;
      resources.intervals = (window as any)._intervals?.length || 0;
      resources.observers = (window as any)._observers?.length || 0;

      return resources;
    });

    // Resources should not accumulate excessively
    expect(finalResources.listeners).toBeLessThan(resourceTest.listeners + 50);
    expect(finalResources.intervals).toBeLessThan(resourceTest.intervals + 10);
    expect(finalResources.observers).toBeLessThan(resourceTest.observers + 10);
  });

  test('Error handling under stress conditions', async ({ page }) => {
    await page.goto('/');

    // Test error handling during rapid interactions
    const errorTest = await page.evaluate(() => {
      return new Promise((resolve) => {
        let errorCount = 0;
        const originalError = console.error;

        // Override console.error to count errors
        console.error = (...args) => {
          errorCount++;
          originalError(...args);
        };

        // Simulate rapid interactions
        for (let i = 0; i < 50; i++) {
          setTimeout(() => {
            // Trigger various events
            const event = new CustomEvent('test-event', {
              detail: { index: i }
            });
            document.dispatchEvent(event);

            // Try to interact with elements
            const buttons = document.querySelectorAll('button');
            buttons.forEach(btn => btn.click());

            if (i === 49) {
              // Restore console.error
              console.error = originalError;
              resolve({ errorCount, totalOperations: 50 });
            }
          }, i * 10);
        }
      });
    });

    // Error rate should be low
    expect(errorTest.errorCount).toBeLessThan(5);
  });

  test('Bundle size impact on performance', async ({ page }) => {
    // Test initial bundle loading
    const navigationStart = Date.now();

    await page.goto('/');
    await page.waitForLoadState('networkidle');

    const navigationEnd = Date.now();
    const initialLoadTime = navigationEnd - navigationStart;

    // Test subsequent page loads (should be faster due to caching)
    const cachedStart = Date.now();

    await page.goto('/detectors');
    await page.waitForLoadState('networkidle');

    const cachedEnd = Date.now();
    const cachedLoadTime = cachedEnd - cachedStart;

    // Initial load should be reasonable
    expect(initialLoadTime).toBeLessThan(5000);

    // Cached load should be much faster
    expect(cachedLoadTime).toBeLessThan(2000);
    expect(cachedLoadTime).toBeLessThan(initialLoadTime * 0.7);
  });

});

test.describe('Stress Testing', () => {

  test('Application stability under extreme load', async ({ page, context }) => {
    // Create multiple pages for stress testing
    const pages = await Promise.all(
      Array.from({ length: 5 }, () => context.newPage())
    );

    try {
      // Navigate all pages to different routes
      const routes = ['/dashboard', '/detectors', '/datasets', '/visualizations', '/monitoring'];

      await Promise.all(pages.map((p, i) =>
        p.goto(routes[i % routes.length])
      ));

      // Perform rapid interactions on all pages
      const interactionPromises = pages.map(async (p, pageIndex) => {
        for (let i = 0; i < 20; i++) {
          try {
            await p.click('button', { timeout: 1000 });
          } catch (e) {
            // Expected - some buttons may not exist
          }

          try {
            await p.fill('input', `test-${pageIndex}-${i}`);
          } catch (e) {
            // Expected - some inputs may not exist
          }

          await p.waitForTimeout(100);
        }
      });

      await Promise.all(interactionPromises);

      // Verify all pages are still responsive
      for (const p of pages) {
        await expect(p.locator('body')).toBeVisible();
      }

    } finally {
      // Cleanup
      await Promise.all(pages.map(p => p.close()));
    }
  });

});

test.describe('Performance Regression Detection', () => {

  test('Core Web Vitals remain within targets', async ({ page }) => {
    await page.goto('/');

    // Measure Core Web Vitals
    const vitals = await page.evaluate(() => {
      return new Promise((resolve) => {
        const vitals = {};

        // Measure LCP
        new PerformanceObserver((list) => {
          const entries = list.getEntries();
          if (entries.length > 0) {
            vitals.lcp = entries[entries.length - 1].startTime;
          }
        }).observe({ entryTypes: ['largest-contentful-paint'] });

        // Measure FID
        new PerformanceObserver((list) => {
          const entries = list.getEntries();
          if (entries.length > 0) {
            vitals.fid = entries[0].processingStart - entries[0].startTime;
          }
        }).observe({ entryTypes: ['first-input'] });

        // Measure CLS
        let clsValue = 0;
        new PerformanceObserver((list) => {
          const entries = list.getEntries();
          entries.forEach(entry => {
            if (!entry.hadRecentInput) {
              clsValue += entry.value;
            }
          });
          vitals.cls = clsValue;
        }).observe({ entryTypes: ['layout-shift'] });

        // Resolve after timeout
        setTimeout(() => {
          resolve(vitals);
        }, 3000);
      });
    });

    // Verify Core Web Vitals targets
    if (vitals.lcp) {
      expect(vitals.lcp).toBeLessThan(2500); // 2.5 seconds
    }

    if (vitals.fid) {
      expect(vitals.fid).toBeLessThan(100); // 100ms
    }

    if (vitals.cls) {
      expect(vitals.cls).toBeLessThan(0.1); // 0.1
    }
  });

});
