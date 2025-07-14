/**
 * Memory Leak Detection and Performance Profiling Tests
 *
 * This test suite implements comprehensive memory leak detection
 * for the Pynomaly web UI to ensure stable performance over time.
 */

import { test, expect, Page, BrowserContext } from '@playwright/test';

interface MemoryMetrics {
  usedJSHeapSize: number;
  totalJSHeapSize: number;
  jsHeapSizeLimit: number;
  timestamp: number;
}

interface PerformanceEntry {
  name: string;
  duration: number;
  startTime: number;
  entryType: string;
}

class MemoryProfiler {
  private measurements: MemoryMetrics[] = [];
  private performanceEntries: PerformanceEntry[] = [];

  async measureMemory(page: Page): Promise<MemoryMetrics> {
    const metrics = await page.evaluate(() => {
      // Force garbage collection if available (Chrome with --enable-precise-memory-info)
      if (window.performance && (window.performance as any).measureUserAgentSpecificMemory) {
        (window.performance as any).measureUserAgentSpecificMemory();
      }

      if ((window.performance as any).memory) {
        const memory = (window.performance as any).memory;
        return {
          usedJSHeapSize: memory.usedJSHeapSize,
          totalJSHeapSize: memory.totalJSHeapSize,
          jsHeapSizeLimit: memory.jsHeapSizeLimit,
          timestamp: Date.now()
        };
      }

      // Fallback for browsers without memory API
      return {
        usedJSHeapSize: 0,
        totalJSHeapSize: 0,
        jsHeapSizeLimit: 0,
        timestamp: Date.now()
      };
    });

    this.measurements.push(metrics);
    return metrics;
  }

  async collectPerformanceEntries(page: Page): Promise<PerformanceEntry[]> {
    const entries = await page.evaluate(() => {
      return performance.getEntries().map(entry => ({
        name: entry.name,
        duration: entry.duration,
        startTime: entry.startTime,
        entryType: entry.entryType
      }));
    });

    this.performanceEntries.push(...entries);
    return entries;
  }

  getMemoryTrend(): { increasing: boolean; growthRate: number } {
    if (this.measurements.length < 3) {
      return { increasing: false, growthRate: 0 };
    }

    const recent = this.measurements.slice(-5);
    const initial = recent[0].usedJSHeapSize;
    const final = recent[recent.length - 1].usedJSHeapSize;

    const growthRate = (final - initial) / initial;
    return {
      increasing: growthRate > 0.1, // 10% growth threshold
      growthRate
    };
  }

  getMemoryStats() {
    if (this.measurements.length === 0) return null;

    const sizes = this.measurements.map(m => m.usedJSHeapSize);
    const min = Math.min(...sizes);
    const max = Math.max(...sizes);
    const avg = sizes.reduce((a, b) => a + b, 0) / sizes.length;

    return { min, max, avg, samples: sizes.length };
  }

  exportReport(): string {
    const stats = this.getMemoryStats();
    const trend = this.getMemoryTrend();

    return JSON.stringify({
      memoryStats: stats,
      memoryTrend: trend,
      measurements: this.measurements,
      performanceEntries: this.performanceEntries.slice(-100) // Last 100 entries
    }, null, 2);
  }
}

test.describe('Memory Leak Detection', () => {
  let profiler: MemoryProfiler;

  test.beforeEach(async ({ page }) => {
    profiler = new MemoryProfiler();

    // Enable memory monitoring in Chrome
    await page.addInitScript(() => {
      // Expose gc function for manual garbage collection
      if ('gc' in window) {
        (window as any).forceGC = () => {
          (window as any).gc();
        };
      }
    });
  });

  test('Dashboard memory usage remains stable during normal interaction', async ({ page }) => {
    await page.goto('/dashboard');
    await page.waitForLoadState('networkidle');

    // Initial memory measurement
    await profiler.measureMemory(page);

    // Simulate normal user interactions over time
    for (let cycle = 0; cycle < 10; cycle++) {
      // Navigate through different sections
      await page.click('[data-testid="detectors-tab"]');
      await page.waitForTimeout(1000);

      // Interact with forms and charts
      if (await page.isVisible('[data-testid="detector-form"]')) {
        await page.fill('input[name="name"]', `Test Detector ${cycle}`);
        await page.selectOption('select[name="algorithm"]', 'IsolationForest');
      }

      // Navigate back
      await page.click('[data-testid="dashboard-tab"]');
      await page.waitForTimeout(1000);

      // Force garbage collection if available
      await page.evaluate(() => {
        if ('forceGC' in window) {
          (window as any).forceGC();
        }
      });

      // Measure memory after interaction
      const metrics = await profiler.measureMemory(page);
      console.log(`Cycle ${cycle + 1}: Memory usage: ${(metrics.usedJSHeapSize / 1024 / 1024).toFixed(2)} MB`);

      // Small delay between cycles
      await page.waitForTimeout(500);
    }

    // Analyze memory trend
    const trend = profiler.getMemoryTrend();
    const stats = profiler.getMemoryStats();

    console.log('Memory Analysis:', {
      trend,
      stats: stats ? {
        ...stats,
        minMB: (stats.min / 1024 / 1024).toFixed(2),
        maxMB: (stats.max / 1024 / 1024).toFixed(2),
        avgMB: (stats.avg / 1024 / 1024).toFixed(2)
      } : null
    });

    // Assert memory growth is within acceptable limits
    expect(trend.increasing).toBe(false);
    if (stats) {
      const memoryGrowth = (stats.max - stats.min) / stats.min;
      expect(memoryGrowth).toBeLessThan(0.5); // Less than 50% growth
    }
  });

  test('WebSocket connections do not cause memory leaks', async ({ page }) => {
    await page.goto('/dashboard');
    await page.waitForLoadState('networkidle');

    // Initial measurement
    await profiler.measureMemory(page);

    // Simulate WebSocket stress test
    for (let i = 0; i < 20; i++) {
      // Start real-time monitoring
      await page.click('[data-testid="start-realtime"]');
      await page.waitForTimeout(2000);

      // Generate some activity to trigger WebSocket messages
      await page.evaluate(() => {
        // Simulate WebSocket message handling
        if (window.WebSocket && (window as any).anomalyWebSocket) {
          const ws = (window as any).anomalyWebSocket;
          // Send test messages to trigger internal processing
          for (let j = 0; j < 10; j++) {
            if (ws.readyState === WebSocket.OPEN) {
              ws.send(JSON.stringify({
                type: 'test_message',
                data: { iteration: j, timestamp: Date.now() }
              }));
            }
          }
        }
      });

      await page.waitForTimeout(1000);

      // Stop monitoring
      await page.click('[data-testid="stop-realtime"]');
      await page.waitForTimeout(1000);

      // Force garbage collection
      await page.evaluate(() => {
        if ('forceGC' in window) {
          (window as any).forceGC();
        }
      });

      // Measure memory
      const metrics = await profiler.measureMemory(page);
      console.log(`WebSocket cycle ${i + 1}: Memory: ${(metrics.usedJSHeapSize / 1024 / 1024).toFixed(2)} MB`);
    }

    // Verify no significant memory leak
    const trend = profiler.getMemoryTrend();
    expect(trend.growthRate).toBeLessThan(0.3); // Less than 30% growth
  });

  test('Chart rendering and updates do not accumulate memory', async ({ page }) => {
    await page.goto('/visualizations');
    await page.waitForLoadState('networkidle');

    await profiler.measureMemory(page);

    // Stress test chart rendering
    for (let i = 0; i < 15; i++) {
      // Switch between different chart types
      const chartTypes = ['line', 'scatter', 'heatmap', 'histogram'];
      const chartType = chartTypes[i % chartTypes.length];

      if (await page.isVisible(`[data-testid="chart-type-${chartType}"]`)) {
        await page.click(`[data-testid="chart-type-${chartType}"]`);
        await page.waitForTimeout(2000); // Wait for chart to render
      }

      // Update chart data
      if (await page.isVisible('[data-testid="refresh-chart"]')) {
        await page.click('[data-testid="refresh-chart"]');
        await page.waitForTimeout(1500);
      }

      // Force cleanup
      await page.evaluate(() => {
        // Trigger any cleanup events
        window.dispatchEvent(new Event('cleanup'));
        if ('forceGC' in window) {
          (window as any).forceGC();
        }
      });

      await profiler.measureMemory(page);
      console.log(`Chart cycle ${i + 1}: Memory state recorded`);
    }

    const trend = profiler.getMemoryTrend();
    expect(trend.growthRate).toBeLessThan(0.4); // Chart memory should stabilize
  });

  test('Long-running detection session memory stability', async ({ page }) => {
    await page.goto('/detection');
    await page.waitForLoadState('networkidle');

    await profiler.measureMemory(page);

    // Simulate a long-running detection session
    const totalDuration = 60000; // 1 minute
    const intervalMs = 5000; // 5 seconds
    const cycles = totalDuration / intervalMs;

    for (let i = 0; i < cycles; i++) {
      // Start a detection if available
      if (await page.isVisible('[data-testid="start-detection"]')) {
        await page.click('[data-testid="start-detection"]');
        await page.waitForTimeout(2000);
      }

      // Simulate periodic data updates
      await page.evaluate(() => {
        // Trigger data refresh
        if ((window as any).updateDetectionData) {
          (window as any).updateDetectionData({
            timestamp: Date.now(),
            anomalies: Array.from({ length: 10 }, (_, i) => ({
              id: i,
              score: Math.random(),
              features: Array.from({ length: 5 }, () => Math.random())
            }))
          });
        }
      });

      await page.waitForTimeout(intervalMs);

      // Periodic memory measurement
      if (i % 3 === 0) { // Every 3 cycles
        await profiler.measureMemory(page);
      }
    }

    // Final memory check
    await page.evaluate(() => {
      if ('forceGC' in window) {
        (window as any).forceGC();
      }
    });

    await profiler.measureMemory(page);

    const trend = profiler.getMemoryTrend();
    const stats = profiler.getMemoryStats();

    // Long-running sessions should not exceed memory limits
    expect(trend.growthRate).toBeLessThan(0.6);
    if (stats) {
      expect(stats.max / 1024 / 1024).toBeLessThan(200); // Less than 200MB
    }
  });

  test('Event listener cleanup prevents memory leaks', async ({ page }) => {
    await page.goto('/dashboard');
    await page.waitForLoadState('networkidle');

    await profiler.measureMemory(page);

    // Test event listener lifecycle
    for (let i = 0; i < 25; i++) {
      // Add dynamic content that creates event listeners
      await page.evaluate((iteration) => {
        // Simulate adding many event listeners
        const div = document.createElement('div');
        div.id = `dynamic-content-${iteration}`;
        div.innerHTML = `<button onclick="console.log('Click ${iteration}')">Test ${iteration}</button>`;

        // Add various event listeners
        div.addEventListener('click', () => console.log(`Click handler ${iteration}`));
        div.addEventListener('mouseover', () => console.log(`Mouseover ${iteration}`));
        div.addEventListener('scroll', () => console.log(`Scroll ${iteration}`));

        document.body.appendChild(div);

        // Remove old content to test cleanup
        const oldElement = document.getElementById(`dynamic-content-${iteration - 10}`);
        if (oldElement) {
          oldElement.remove();
        }
      }, i);

      await page.waitForTimeout(100);

      // Periodic memory check
      if (i % 5 === 0) {
        await profiler.measureMemory(page);
      }
    }

    // Force garbage collection
    await page.evaluate(() => {
      if ('forceGC' in window) {
        (window as any).forceGC();
      }
    });

    await profiler.measureMemory(page);

    const trend = profiler.getMemoryTrend();
    expect(trend.growthRate).toBeLessThan(0.5); // Event listeners should be cleaned up
  });

  test.afterEach(async ({ page }, testInfo) => {
    // Export memory profile report
    const report = profiler.exportReport();

    // Save report as test artifact
    const reportPath = `memory-report-${testInfo.title.replace(/[^a-zA-Z0-9]/g, '-')}.json`;
    await testInfo.attach(reportPath, {
      body: report,
      contentType: 'application/json'
    });

    // Log summary
    const stats = profiler.getMemoryStats();
    if (stats) {
      console.log(`Memory Profile Summary for "${testInfo.title}":`, {
        samples: stats.samples,
        minMB: (stats.min / 1024 / 1024).toFixed(2),
        maxMB: (stats.max / 1024 / 1024).toFixed(2),
        avgMB: (stats.avg / 1024 / 1024).toFixed(2)
      });
    }
  });
});

test.describe('Performance Monitoring Integration', () => {
  test('Resource usage tracking across user flows', async ({ page }) => {
    const profiler = new MemoryProfiler();

    // Complete user workflow
    const workflow = [
      '/dashboard',
      '/datasets',
      '/detectors',
      '/detection',
      '/visualizations',
      '/dashboard' // Return to start
    ];

    for (const route of workflow) {
      await page.goto(route);
      await page.waitForLoadState('networkidle');

      // Collect performance entries
      await profiler.collectPerformanceEntries(page);

      // Measure memory at each step
      const metrics = await profiler.measureMemory(page);
      console.log(`Route ${route}: Memory: ${(metrics.usedJSHeapSize / 1024 / 1024).toFixed(2)} MB`);

      await page.waitForTimeout(1000);
    }

    // Verify overall resource health
    const trend = profiler.getMemoryTrend();
    expect(trend.growthRate).toBeLessThan(1.0); // Less than 100% growth for full workflow
  });
});
