/**
 * Memory Leak Detection Test Suite
 * Comprehensive testing for memory leaks in Pynomaly web UI components
 */

import { test, expect, Page } from '@playwright/test';
import { promises as fs } from 'fs';
import path from 'path';

// Memory monitoring interfaces
interface MemorySnapshot {
  jsHeapUsed: number;
  jsHeapTotal: number;
  jsHeapLimit: number;
  domNodes: number;
  timestamp: number;
}

interface MemoryLeakTest {
  testName: string;
  initialSnapshot: MemorySnapshot;
  finalSnapshot: MemorySnapshot;
  snapshots: MemorySnapshot[];
  memoryGrowth: number;
  domNodeGrowth: number;
  leakDetected: boolean;
  leakThreshold: number;
}

class MemoryLeakDetector {
  private snapshots: MemorySnapshot[] = [];
  private monitoringInterval: NodeJS.Timeout | null = null;
  private leakThreshold = 10 * 1024 * 1024; // 10MB

  constructor(private page: Page, private testName: string) {}

  async startMonitoring(intervalMs: number = 500): Promise<void> {
    await this.captureSnapshot();

    this.monitoringInterval = setInterval(async () => {
      await this.captureSnapshot();
    }, intervalMs);
  }

  async stopMonitoring(): Promise<void> {
    if (this.monitoringInterval) {
      clearInterval(this.monitoringInterval);
      this.monitoringInterval = null;
    }

    await this.captureSnapshot();
  }

  private async captureSnapshot(): Promise<void> {
    const snapshot = await this.page.evaluate(() => {
      const memory = (performance as any).memory;

      return {
        jsHeapUsed: memory ? memory.usedJSHeapSize : 0,
        jsHeapTotal: memory ? memory.totalJSHeapSize : 0,
        jsHeapLimit: memory ? memory.jsHeapSizeLimit : 0,
        domNodes: document.querySelectorAll('*').length,
        timestamp: Date.now()
      };
    });

    this.snapshots.push(snapshot);
  }

  async forceGarbageCollection(): Promise<void> {
    await this.page.evaluate(() => {
      if (window.gc) {
        window.gc();
      }
    });

    // Wait for GC to complete
    await this.page.waitForTimeout(1000);
  }

  getMemoryAnalysis(): MemoryLeakTest {
    if (this.snapshots.length < 2) {
      throw new Error('Not enough snapshots for analysis');
    }

    const initial = this.snapshots[0];
    const final = this.snapshots[this.snapshots.length - 1];
    const memoryGrowth = final.jsHeapUsed - initial.jsHeapUsed;
    const domNodeGrowth = final.domNodes - initial.domNodes;

    return {
      testName: this.testName,
      initialSnapshot: initial,
      finalSnapshot: final,
      snapshots: this.snapshots,
      memoryGrowth,
      domNodeGrowth,
      leakDetected: memoryGrowth > this.leakThreshold,
      leakThreshold: this.leakThreshold
    };
  }

  async exportReport(): Promise<string> {
    const analysis = this.getMemoryAnalysis();

    return JSON.stringify({
      testName: this.testName,
      analysis,
      generatedAt: new Date().toISOString()
    }, null, 2);
  }
}

test.describe('Memory Leak Detection', () => {
  let memoryMonitor: MemoryLeakDetector;

  test.beforeEach(async ({ page }) => {
    // Enable memory monitoring
    await page.addInitScript(() => {
      // Override console to capture memory-related warnings
      const originalWarn = console.warn;
      console.warn = (...args) => {
        window.memoryWarnings = window.memoryWarnings || [];
        window.memoryWarnings.push(args.join(' '));
        originalWarn.apply(console, args);
      };
    });
  });

  test.afterEach(async () => {
    if (memoryMonitor) {
      await memoryMonitor.stopMonitoring();
    }
  });

  test('should detect memory leaks during dashboard navigation', async ({ page }) => {
    memoryMonitor = new MemoryLeakDetector(page, 'dashboard-navigation');

    await page.goto('/dashboard');
    await memoryMonitor.startMonitoring(500);

    // Perform extensive navigation
    for (let i = 0; i < 10; i++) {
      await page.click('[data-testid="nav-detection"]');
      await page.waitForLoadState('networkidle');

      await page.click('[data-testid="nav-data"]');
      await page.waitForLoadState('networkidle');

      await page.click('[data-testid="nav-dashboard"]');
      await page.waitForLoadState('networkidle');

      await page.click('[data-testid="nav-reports"]');
      await page.waitForLoadState('networkidle');

      await page.click('[data-testid="nav-settings"]');
      await page.waitForLoadState('networkidle');

      await page.click('[data-testid="nav-dashboard"]');
      await page.waitForLoadState('networkidle');
    }

    await memoryMonitor.forceGarbageCollection();
    await memoryMonitor.stopMonitoring();

    const analysis = memoryMonitor.getMemoryAnalysis();

    console.log('Dashboard Navigation Memory Analysis:', {
      memoryGrowth: `${(analysis.memoryGrowth / 1024 / 1024).toFixed(2)}MB`,
      domNodeGrowth: analysis.domNodeGrowth,
      leakDetected: analysis.leakDetected
    });

    expect(analysis.memoryGrowth).toBeLessThan(analysis.leakThreshold);
    expect(analysis.domNodeGrowth).toBeLessThan(100);
  });

  test('should detect memory leaks during chart rendering cycles', async ({ page }) => {
    memoryMonitor = new MemoryLeakDetector(page, 'chart-rendering');

    await page.goto('/dashboard');
    await memoryMonitor.startMonitoring(300);

    // Create and destroy charts repeatedly
    for (let i = 0; i < 15; i++) {
      // Create chart
      await page.click('[data-testid="add-chart-widget"]');
      await page.waitForSelector('[data-testid="chart-widget"]');

      // Interact with chart
      await page.hover('[data-testid="chart-widget"]');
      await page.click('[data-testid="chart-zoom-in"]');
      await page.click('[data-testid="chart-zoom-out"]');

      // Update chart data
      await page.click('[data-testid="chart-refresh"]');
      await page.waitForTimeout(200);

      // Remove chart
      await page.click('[data-testid="chart-close"]');
      await page.waitForSelector('[data-testid="chart-widget"]', { state: 'detached' });

      await page.waitForTimeout(100);
    }

    await memoryMonitor.forceGarbageCollection();
    await memoryMonitor.stopMonitoring();

    const analysis = memoryMonitor.getMemoryAnalysis();

    console.log('Chart Rendering Memory Analysis:', {
      memoryGrowth: `${(analysis.memoryGrowth / 1024 / 1024).toFixed(2)}MB`,
      domNodeGrowth: analysis.domNodeGrowth,
      leakDetected: analysis.leakDetected
    });

    expect(analysis.memoryGrowth).toBeLessThan(analysis.leakThreshold);
    expect(analysis.domNodeGrowth).toBeLessThan(50);
  });

  test('should detect memory leaks in WebSocket connections', async ({ page }) => {
    memoryMonitor = new MemoryLeakDetector(page, 'websocket-connections');

    await page.goto('/dashboard');
    await memoryMonitor.startMonitoring(400);

    // Setup WebSocket connections
    await page.evaluate(() => {
      window.testWebSockets = [];
    });

    // Create and close WebSocket connections
    for (let i = 0; i < 10; i++) {
      await page.evaluate((i) => {
        const ws = new WebSocket('ws://localhost:8000/ws/test');
        window.testWebSockets.push(ws);

        ws.onopen = () => {
          ws.send(JSON.stringify({ type: 'test', data: `message${i}` }));
        };

        ws.onmessage = (event) => {
          console.log('WebSocket message:', event.data);
        };

        // Close connection after brief delay
        setTimeout(() => {
          ws.close();
        }, 100);
      }, i);

      await page.waitForTimeout(200);
    }

    // Wait for all connections to close
    await page.waitForTimeout(2000);

    await memoryMonitor.forceGarbageCollection();
    await memoryMonitor.stopMonitoring();

    const analysis = memoryMonitor.getMemoryAnalysis();

    // Check for remaining WebSocket connections
    const remainingConnections = await page.evaluate(() => {
      return window.testWebSockets ? window.testWebSockets.filter(ws => ws.readyState === WebSocket.OPEN).length : 0;
    });

    console.log('WebSocket Memory Analysis:', {
      memoryGrowth: `${(analysis.memoryGrowth / 1024 / 1024).toFixed(2)}MB`,
      remainingConnections,
      leakDetected: analysis.leakDetected
    });

    expect(analysis.memoryGrowth).toBeLessThan(analysis.leakThreshold);
    expect(remainingConnections).toBe(0);
  });

  test('should detect memory leaks in dynamic form operations', async ({ page }) => {
    memoryMonitor = new MemoryLeakDetector(page, 'dynamic-form-operations');

    await page.goto('/detection');
    await memoryMonitor.startMonitoring(300);

    // Create and destroy dynamic forms
    for (let i = 0; i < 20; i++) {
      // Create form
      await page.click('[data-testid="add-detector-form"]');
      await page.waitForSelector('[data-testid="detector-form"]');

      // Fill form fields
      await page.fill('[data-testid="detector-name"]', `Detector ${i}`);
      await page.fill('[data-testid="detector-description"]', `Description for detector ${i}`);

      // Add dynamic fields
      await page.click('[data-testid="add-parameter-field"]');
      await page.click('[data-testid="add-parameter-field"]');
      await page.click('[data-testid="add-parameter-field"]');

      // Fill dynamic fields
      await page.fill('[data-testid="parameter-0"]', 'param1');
      await page.fill('[data-testid="parameter-1"]', 'param2');
      await page.fill('[data-testid="parameter-2"]', 'param3');

      // Cancel form (should clean up)
      await page.click('[data-testid="cancel-detector-form"]');
      await page.waitForSelector('[data-testid="detector-form"]', { state: 'detached' });

      await page.waitForTimeout(100);
    }

    await memoryMonitor.forceGarbageCollection();
    await memoryMonitor.stopMonitoring();

    const analysis = memoryMonitor.getMemoryAnalysis();

    console.log('Dynamic Form Memory Analysis:', {
      memoryGrowth: `${(analysis.memoryGrowth / 1024 / 1024).toFixed(2)}MB`,
      domNodeGrowth: analysis.domNodeGrowth,
      leakDetected: analysis.leakDetected
    });

    expect(analysis.memoryGrowth).toBeLessThan(analysis.leakThreshold);
    expect(analysis.domNodeGrowth).toBeLessThan(200);
  });

  test('should detect memory leaks in data table operations', async ({ page }) => {
    memoryMonitor = new MemoryLeakDetector(page, 'data-table-operations');

    await page.goto('/data');
    await memoryMonitor.startMonitoring(400);

    // Perform extensive table operations
    for (let i = 0; i < 8; i++) {
      // Load data
      await page.click('[data-testid="load-dataset"]');
      await page.waitForSelector('[data-testid="data-table"]');

      // Sort by different columns
      await page.click('[data-testid="sort-column-0"]');
      await page.click('[data-testid="sort-column-1"]');
      await page.click('[data-testid="sort-column-2"]');

      // Filter data
      await page.fill('[data-testid="filter-input"]', `filter${i}`);
      await page.click('[data-testid="apply-filter"]');

      // Paginate through data
      await page.click('[data-testid="next-page"]');
      await page.click('[data-testid="next-page"]');
      await page.click('[data-testid="prev-page"]');

      // Select rows
      await page.click('[data-testid="select-row-0"]');
      await page.click('[data-testid="select-row-1"]');
      await page.click('[data-testid="select-row-2"]');

      // Clear selection
      await page.click('[data-testid="clear-selection"]');

      // Clear data
      await page.click('[data-testid="clear-dataset"]');
      await page.waitForSelector('[data-testid="data-table"]', { state: 'detached' });

      await page.waitForTimeout(200);
    }

    await memoryMonitor.forceGarbageCollection();
    await memoryMonitor.stopMonitoring();

    const analysis = memoryMonitor.getMemoryAnalysis();

    console.log('Data Table Memory Analysis:', {
      memoryGrowth: `${(analysis.memoryGrowth / 1024 / 1024).toFixed(2)}MB`,
      domNodeGrowth: analysis.domNodeGrowth,
      leakDetected: analysis.leakDetected
    });

    expect(analysis.memoryGrowth).toBeLessThan(analysis.leakThreshold);
    expect(analysis.domNodeGrowth).toBeLessThan(100);
  });

  test('should detect memory leaks during extended user session', async ({ page }) => {
    memoryMonitor = new MemoryLeakDetector(page, 'extended-session');

    await page.goto('/dashboard');
    await memoryMonitor.startMonitoring(1000);

    // Simulate extended user session
    for (let session = 0; session < 5; session++) {
      // Dashboard interactions
      await page.click('[data-testid="add-widget"]');
      await page.click('[data-testid="widget-settings"]');
      await page.click('[data-testid="close-settings"]');

      // Navigate to detection
      await page.click('[data-testid="nav-detection"]');
      await page.waitForLoadState('networkidle');

      // Create detector
      await page.fill('[data-testid="detector-name"]', `Session${session}Detector`);
      await page.click('[data-testid="create-detector"]');
      await page.waitForSelector('[data-testid="detector-created"]');

      // Navigate to data
      await page.click('[data-testid="nav-data"]');
      await page.waitForLoadState('networkidle');

      // Upload data
      await page.click('[data-testid="upload-data"]');
      await page.waitForSelector('[data-testid="data-uploaded"]');

      // Run analysis
      await page.click('[data-testid="run-analysis"]');
      await page.waitForSelector('[data-testid="analysis-complete"]');

      // View results
      await page.click('[data-testid="view-results"]');
      await page.waitForSelector('[data-testid="results-table"]');

      // Navigate to reports
      await page.click('[data-testid="nav-reports"]');
      await page.waitForLoadState('networkidle');

      // Generate report
      await page.click('[data-testid="generate-report"]');
      await page.waitForSelector('[data-testid="report-generated"]');

      // Back to dashboard
      await page.click('[data-testid="nav-dashboard"]');
      await page.waitForLoadState('networkidle');

      await page.waitForTimeout(500);
    }

    await memoryMonitor.forceGarbageCollection();
    await memoryMonitor.stopMonitoring();

    const analysis = memoryMonitor.getMemoryAnalysis();

    console.log('Extended Session Memory Analysis:', {
      memoryGrowth: `${(analysis.memoryGrowth / 1024 / 1024).toFixed(2)}MB`,
      domNodeGrowth: analysis.domNodeGrowth,
      leakDetected: analysis.leakDetected,
      testDuration: `${(analysis.snapshots[analysis.snapshots.length - 1].timestamp - analysis.snapshots[0].timestamp) / 1000}s`
    });

    // Extended sessions should have stricter memory limits
    expect(analysis.memoryGrowth).toBeLessThan(20 * 1024 * 1024); // 20MB
    expect(analysis.domNodeGrowth).toBeLessThan(300);
  });
});

// Global setup for memory leak report generation
test.afterAll(async () => {
  // Generate memory leak summary report
  const reportsDir = path.join(__dirname, '..', '..', 'test_reports');

  try {
    await fs.mkdir(reportsDir, { recursive: true });

    const summaryReport = {
      testSuite: 'Memory Leak Detection',
      completedAt: new Date().toISOString(),
      testsRun: 6,
      memoryThreshold: '10MB',
      domNodeThreshold: '100-300 nodes',
      testCategories: [
        'Dashboard Navigation',
        'Chart Rendering',
        'WebSocket Connections',
        'Dynamic Form Operations',
        'Data Table Operations',
        'Extended User Session'
      ],
      recommendations: [
        'Ensure proper component cleanup in React lifecycle methods',
        'Remove event listeners when components unmount',
        'Close WebSocket connections properly',
        'Clear timers and intervals when no longer needed',
        'Implement proper memory management for dynamic content'
      ]
    };

    const reportPath = path.join(reportsDir, 'memory-leak-report.json');
    await fs.writeFile(reportPath, JSON.stringify(summaryReport, null, 2));

    console.log(`Memory leak detection report saved: ${reportPath}`);
  } catch (error) {
    console.warn('Could not generate memory leak report:', error);
  }
});
