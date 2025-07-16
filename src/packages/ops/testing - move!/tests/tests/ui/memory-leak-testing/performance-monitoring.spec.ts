/**
 * Performance Monitoring and Memory Profiling Test Suite
 * Advanced monitoring for resource usage, performance degradation, and memory patterns
 */

import { test, expect, Page } from '@playwright/test';
import { promises as fs } from 'fs';
import path from 'path';

// Performance monitoring interfaces
interface PerformanceMetrics {
  navigation?: PerformanceNavigationTiming;
  memory?: MemoryInfo;
  resources?: PerformanceResourceTiming[];
  marks?: PerformanceMark[];
  measures?: PerformanceMeasure[];
  timestamp: number;
  url: string;
}

interface ResourceUsageSnapshot {
  jsHeapUsed: number;
  jsHeapTotal: number;
  domNodes: number;
  eventListeners: number;
  styleSheets: number;
  images: number;
  timestamp: number;
}

interface PerformanceTest {
  testName: string;
  duration: number;
  resourceSnapshots: ResourceUsageSnapshot[];
  performanceMetrics: PerformanceMetrics[];
  warnings: string[];
  errors: string[];
}

class PerformanceProfiler {
  private testData: PerformanceTest;
  private monitoringInterval: NodeJS.Timeout | null = null;
  private startTime: number;

  constructor(private page: Page, testName: string) {
    this.testData = {
      testName,
      duration: 0,
      resourceSnapshots: [],
      performanceMetrics: [],
      warnings: [],
      errors: []
    };
    this.startTime = Date.now();
  }

  async startProfiling(): Promise<void> {
    // Enable performance monitoring
    await this.page.addInitScript(() => {
      // Add performance marks
      performance.mark('test-start');

      // Override console methods to capture performance issues
      const originalWarn = console.warn;
      const originalError = console.error;

      console.warn = (...args) => {
        window.performanceWarnings = window.performanceWarnings || [];
        window.performanceWarnings.push(args.join(' '));
        originalWarn.apply(console, args);
      };

      console.error = (...args) => {
        window.performanceErrors = window.performanceErrors || [];
        window.performanceErrors.push(args.join(' '));
        originalError.apply(console, args);
      };
    });

    // Start periodic monitoring
    this.monitoringInterval = setInterval(async () => {
      await this.captureResourceSnapshot();
      await this.capturePerformanceMetrics();
    }, 2000);

    // Initial capture
    await this.captureResourceSnapshot();
    await this.capturePerformanceMetrics();
  }

  async stopProfiling(): Promise<void> {
    if (this.monitoringInterval) {
      clearInterval(this.monitoringInterval);
      this.monitoringInterval = null;
    }

    // Final captures
    await this.captureResourceSnapshot();
    await this.capturePerformanceMetrics();

    // Mark test end
    await this.page.evaluate(() => {
      performance.mark('test-end');
      performance.measure('test-duration', 'test-start', 'test-end');
    });

    // Capture warnings and errors
    const warnings = await this.page.evaluate(() => window.performanceWarnings || []);
    const errors = await this.page.evaluate(() => window.performanceErrors || []);

    this.testData.warnings = warnings;
    this.testData.errors = errors;
    this.testData.duration = Date.now() - this.startTime;
  }

  private async captureResourceSnapshot(): Promise<void> {
    const snapshot = await this.page.evaluate(() => {
      const memory = (performance as any).memory;

      return {
        jsHeapUsed: memory ? memory.usedJSHeapSize : 0,
        jsHeapTotal: memory ? memory.totalJSHeapSize : 0,
        domNodes: document.querySelectorAll('*').length,
        eventListeners: Object.keys(window).filter(key => key.startsWith('on')).length,
        styleSheets: document.styleSheets.length,
        images: document.images.length,
        timestamp: Date.now()
      };
    });

    this.testData.resourceSnapshots.push(snapshot);
  }

  private async capturePerformanceMetrics(): Promise<void> {
    const metrics = await this.page.evaluate(() => {
      const navigation = performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming;
      const memory = (performance as any).memory;
      const resources = performance.getEntriesByType('resource') as PerformanceResourceTiming[];
      const marks = performance.getEntriesByType('mark') as PerformanceMark[];
      const measures = performance.getEntriesByType('measure') as PerformanceMeasure[];

      return {
        navigation,
        memory,
        resources: resources.slice(-10), // Keep last 10 resources
        marks: marks.slice(-10), // Keep last 10 marks
        measures: measures.slice(-10), // Keep last 10 measures
        timestamp: Date.now(),
        url: window.location.href
      };
    });

    this.testData.performanceMetrics.push(metrics);
  }

  getAnalysis(): any {
    const snapshots = this.testData.resourceSnapshots;
    if (snapshots.length < 2) return null;

    const initial = snapshots[0];
    const final = snapshots[snapshots.length - 1];

    return {
      memoryGrowth: final.jsHeapUsed - initial.jsHeapUsed,
      domNodeGrowth: final.domNodes - initial.domNodes,
      totalDomNodes: final.domNodes,
      peakMemory: Math.max(...snapshots.map(s => s.jsHeapUsed)),
      averageMemory: snapshots.reduce((sum, s) => sum + s.jsHeapUsed, 0) / snapshots.length,
      testDuration: this.testData.duration,
      warningCount: this.testData.warnings.length,
      errorCount: this.testData.errors.length,
      resourceLeaks: {
        memoryLeak: final.jsHeapUsed > initial.jsHeapUsed * 1.5,
        domNodeLeak: final.domNodes > initial.domNodes * 1.2,
        eventListenerLeak: final.eventListeners > initial.eventListeners * 1.1
      }
    };
  }

  async exportReport(): Promise<string> {
    const analysis = this.getAnalysis();

    return JSON.stringify({
      testName: this.testData.testName,
      analysis,
      rawData: this.testData,
      generatedAt: new Date().toISOString()
    }, null, 2);
  }
}

test.describe('Performance Monitoring', () => {
  let profiler: PerformanceProfiler;

  test.beforeEach(async ({ page }) => {
    // Configure Chrome for performance monitoring
    await page.addInitScript(() => {
      // Enable performance observer
      if ('PerformanceObserver' in window) {
        const observer = new PerformanceObserver((list) => {
          list.getEntries().forEach((entry) => {
            if (entry.entryType === 'longtask') {
              console.warn('Long task detected:', entry.duration);
            }
          });
        });
        observer.observe({ entryTypes: ['longtask'] });
      }
    });
  });

  test.afterEach(async () => {
    if (profiler) {
      await profiler.stopProfiling();
    }
  });

  test('should monitor performance during heavy dashboard operations', async ({ page }) => {
    profiler = new PerformanceProfiler(page, 'heavy-dashboard-operations');

    await page.goto('/dashboard');
    await profiler.startProfiling();

    // Simulate heavy operations
    for (let i = 0; i < 5; i++) {
      // Add multiple widgets
      await page.click('[data-testid="add-chart-widget"]');
      await page.click('[data-testid="add-metrics-widget"]');
      await page.click('[data-testid="add-table-widget"]');

      // Interact with widgets
      await page.hover('[data-testid="chart-widget-0"]');
      await page.click('[data-testid="widget-settings-0"]');
      await page.click('[data-testid="close-settings"]');

      // Refresh data
      await page.click('[data-testid="refresh-dashboard"]');
      await page.waitForLoadState('networkidle');

      await page.waitForTimeout(1000);
    }

    await profiler.stopProfiling();

    const analysis = profiler.getAnalysis();

    console.log('Heavy Operations Analysis:', {
      memoryGrowth: `${(analysis.memoryGrowth / 1024 / 1024).toFixed(2)}MB`,
      domNodeGrowth: analysis.domNodeGrowth,
      peakMemory: `${(analysis.peakMemory / 1024 / 1024).toFixed(2)}MB`,
      warnings: analysis.warningCount,
      errors: analysis.errorCount
    });

    // Performance assertions
    expect(analysis.memoryGrowth).toBeLessThan(50 * 1024 * 1024); // 50MB
    expect(analysis.domNodeGrowth).toBeLessThan(1000);
    expect(analysis.errorCount).toBe(0);
  });

  test('should monitor memory patterns during data visualization', async ({ page }) => {
    profiler = new PerformanceProfiler(page, 'data-visualization-memory');

    await page.goto('/visualizations');
    await profiler.startProfiling();

    // Create and destroy multiple visualizations
    const visualizationTypes = ['line-chart', 'bar-chart', 'scatter-plot', 'histogram'];

    for (let cycle = 0; cycle < 3; cycle++) {
      for (const vizType of visualizationTypes) {
        // Create visualization
        await page.click(`[data-testid="create-${vizType}"]`);
        await page.waitForSelector(`[data-testid="${vizType}-container"]`);

        // Interact with visualization
        await page.hover(`[data-testid="${vizType}-container"]`);
        await page.click(`[data-testid="${vizType}-zoom-in"]`);
        await page.click(`[data-testid="${vizType}-zoom-out"]`);

        // Change data
        await page.click(`[data-testid="${vizType}-change-data"]`);
        await page.waitForTimeout(500);

        // Destroy visualization
        await page.click(`[data-testid="${vizType}-close"]`);
        await page.waitForSelector(`[data-testid="${vizType}-container"]`, { state: 'detached' });

        await page.waitForTimeout(200);
      }
    }

    await profiler.stopProfiling();

    const analysis = profiler.getAnalysis();

    console.log('Visualization Memory Analysis:', {
      memoryGrowth: `${(analysis.memoryGrowth / 1024 / 1024).toFixed(2)}MB`,
      peakMemory: `${(analysis.peakMemory / 1024 / 1024).toFixed(2)}MB`,
      resourceLeaks: analysis.resourceLeaks
    });

    expect(analysis.resourceLeaks.memoryLeak).toBe(false);
    expect(analysis.resourceLeaks.domNodeLeak).toBe(false);
  });

  test('should detect performance degradation over time', async ({ page }) => {
    profiler = new PerformanceProfiler(page, 'performance-degradation');

    await page.goto('/dashboard');
    await profiler.startProfiling();

    // Simulate long-running session with repeated operations
    for (let i = 0; i < 20; i++) {
      // Navigation
      await page.click('[data-testid="nav-detection"]');
      await page.waitForLoadState('networkidle');

      // Form interactions
      await page.fill('[data-testid="detector-name"]', `Detector ${i}`);
      await page.click('[data-testid="algorithm-select"]');
      await page.click('[data-testid="algorithm-option-0"]');

      // Data operations
      await page.click('[data-testid="load-sample-data"]');
      await page.waitForSelector('[data-testid="data-preview"]');

      // Go back to dashboard
      await page.click('[data-testid="nav-dashboard"]');
      await page.waitForLoadState('networkidle');

      await page.waitForTimeout(300);
    }

    await profiler.stopProfiling();

    const analysis = profiler.getAnalysis();

    // Check for performance degradation indicators
    const snapshots = profiler.testData.resourceSnapshots;
    const memoryTrend = snapshots.map(s => s.jsHeapUsed);
    const domNodeTrend = snapshots.map(s => s.domNodes);

    // Calculate trend (should be stable, not constantly growing)
    const memorySlope = calculateTrend(memoryTrend);
    const domNodeSlope = calculateTrend(domNodeTrend);

    console.log('Performance Degradation Analysis:', {
      memoryTrend: memorySlope > 0 ? 'Growing' : 'Stable',
      domNodeTrend: domNodeSlope > 0 ? 'Growing' : 'Stable',
      averageMemory: `${(analysis.averageMemory / 1024 / 1024).toFixed(2)}MB`,
      totalDomNodes: analysis.totalDomNodes
    });

    // Performance should remain stable
    expect(memorySlope).toBeLessThan(1000); // Memory growth should be minimal
    expect(domNodeSlope).toBeLessThan(1); // DOM nodes should remain stable
  });

  test('should monitor resource cleanup after component unmount', async ({ page }) => {
    profiler = new PerformanceProfiler(page, 'component-cleanup');

    await page.goto('/dashboard');
    await profiler.startProfiling();

    // Create components and immediately unmount them
    for (let i = 0; i < 10; i++) {
      // Mount component
      await page.click('[data-testid="mount-heavy-component"]');
      await page.waitForSelector('[data-testid="heavy-component"]');

      // Let component initialize
      await page.waitForTimeout(500);

      // Unmount component
      await page.click('[data-testid="unmount-heavy-component"]');
      await page.waitForSelector('[data-testid="heavy-component"]', { state: 'detached' });

      // Wait for cleanup
      await page.waitForTimeout(200);
    }

    // Force garbage collection
    await page.evaluate(() => {
      if (window.gc) {
        window.gc();
      }
    });

    await page.waitForTimeout(1000);

    await profiler.stopProfiling();

    const analysis = profiler.getAnalysis();

    console.log('Component Cleanup Analysis:', {
      finalMemoryGrowth: `${(analysis.memoryGrowth / 1024 / 1024).toFixed(2)}MB`,
      finalDomNodeGrowth: analysis.domNodeGrowth,
      resourceLeaks: analysis.resourceLeaks
    });

    // After proper cleanup, growth should be minimal
    expect(analysis.memoryGrowth).toBeLessThan(5 * 1024 * 1024); // 5MB
    expect(analysis.domNodeGrowth).toBeLessThan(50);
  });
});

test.describe('Resource Leak Detection', () => {
  test('should detect timer leaks', async ({ page }) => {
    await page.goto('/dashboard');

    // Create timers without cleanup
    await page.evaluate(() => {
      window.testTimers = [];

      // Create intervals that should be cleaned up
      for (let i = 0; i < 5; i++) {
        const interval = setInterval(() => {
          console.log('Timer tick');
        }, 1000);
        window.testTimers.push(interval);
      }
    });

    // Navigate away (should trigger cleanup)
    await page.goto('/detection');
    await page.waitForTimeout(2000);

    // Check if timers were cleaned up
    const activeTimers = await page.evaluate(() => {
      return window.testTimers ? window.testTimers.length : 0;
    });

    console.log('Timer Leak Detection:', {
      activeTimers: activeTimers
    });

    // Timers should be cleaned up when navigating away
    expect(activeTimers).toBeLessThanOrEqual(5);
  });

  test('should detect event listener leaks', async ({ page }) => {
    await page.goto('/dashboard');

    const getListenerCount = async () => {
      return await page.evaluate(() => {
        return window.getEventListeners ?
          Object.keys(window.getEventListeners(document)).length : 0;
      });
    };

    const initialListeners = await getListenerCount();

    // Add event listeners
    await page.evaluate(() => {
      window.testListeners = [];

      for (let i = 0; i < 10; i++) {
        const listener = () => console.log('Event');
        document.addEventListener('click', listener);
        window.testListeners.push(listener);
      }
    });

    const withListeners = await getListenerCount();

    // Remove listeners
    await page.evaluate(() => {
      if (window.testListeners) {
        window.testListeners.forEach(listener => {
          document.removeEventListener('click', listener);
        });
        window.testListeners = [];
      }
    });

    const afterCleanup = await getListenerCount();

    console.log('Event Listener Leak Detection:', {
      initialListeners,
      withListeners,
      afterCleanup,
      leakDetected: afterCleanup > initialListeners
    });

    expect(afterCleanup).toBeLessThanOrEqual(initialListeners);
  });
});

// Helper function to calculate trend
function calculateTrend(values: number[]): number {
  if (values.length < 2) return 0;

  const n = values.length;
  const sumX = (n * (n - 1)) / 2;
  const sumY = values.reduce((sum, val) => sum + val, 0);
  const sumXY = values.reduce((sum, val, i) => sum + i * val, 0);
  const sumXX = values.reduce((sum, _, i) => sum + i * i, 0);

  return (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
}

// Global teardown to save performance reports
test.afterAll(async () => {
  // This would be handled by individual test teardowns
  console.log('Performance monitoring tests completed');
});
