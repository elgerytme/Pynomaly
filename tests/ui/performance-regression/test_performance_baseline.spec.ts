/**
 * Performance Regression Detection System
 * 
 * This test suite establishes performance baselines and detects regressions
 * across different builds and deployments of the Pynomaly web UI.
 */

import { test, expect, Page } from '@playwright/test';
import { readFileSync, writeFileSync, existsSync, mkdirSync } from 'fs';
import { join } from 'path';

interface PerformanceBaseline {
  timestamp: string;
  buildId: string;
  branch: string;
  commit: string;
  metrics: {
    [testName: string]: PerformanceMetrics;
  };
}

interface PerformanceMetrics {
  // Core Web Vitals
  firstContentfulPaint: number;
  largestContentfulPaint: number;
  firstInputDelay: number;
  cumulativeLayoutShift: number;
  
  // Additional Performance Metrics
  timeToInteractive: number;
  totalBlockingTime: number;
  speedIndex: number;
  
  // Resource Metrics
  totalResourceCount: number;
  totalResourceSize: number;
  javascriptSize: number;
  cssSize: number;
  imageSize: number;
  
  // Custom App Metrics
  dashboardLoadTime: number;
  chartRenderTime: number;
  websocketConnectionTime: number;
  memoryUsage: number;
  
  // User Interaction Metrics
  buttonClickResponseTime: number;
  formSubmissionTime: number;
  searchResponseTime: number;
  navigationTime: number;
}

interface RegressionAlert {
  metric: string;
  current: number;
  baseline: number;
  regression: number; // percentage
  severity: 'low' | 'medium' | 'high' | 'critical';
  threshold: number;
}

class PerformanceRegressionDetector {
  private baselinePath: string;
  private currentBaseline: PerformanceBaseline | null = null;
  private regressionThresholds: { [key: string]: number } = {
    // Core Web Vitals thresholds (percentage regression)
    firstContentfulPaint: 20,
    largestContentfulPaint: 25,
    firstInputDelay: 30,
    cumulativeLayoutShift: 15,
    
    // Performance thresholds
    timeToInteractive: 30,
    totalBlockingTime: 25,
    speedIndex: 20,
    
    // Resource thresholds
    totalResourceSize: 15,
    javascriptSize: 20,
    cssSize: 25,
    
    // App-specific thresholds
    dashboardLoadTime: 25,
    chartRenderTime: 30,
    websocketConnectionTime: 40,
    memoryUsage: 35,
    
    // Interaction thresholds
    buttonClickResponseTime: 50,
    formSubmissionTime: 30,
    searchResponseTime: 25,
    navigationTime: 20
  };

  constructor() {
    this.baselinePath = join(process.cwd(), 'performance-baselines');
    this.ensureBaselineDirectory();
    this.loadCurrentBaseline();
  }

  private ensureBaselineDirectory(): void {
    if (!existsSync(this.baselinePath)) {
      mkdirSync(this.baselinePath, { recursive: true });
    }
  }

  private loadCurrentBaseline(): void {
    const baselineFile = join(this.baselinePath, 'current-baseline.json');
    if (existsSync(baselineFile)) {
      try {
        const content = readFileSync(baselineFile, 'utf-8');
        this.currentBaseline = JSON.parse(content);
      } catch (error) {
        console.warn('Failed to load performance baseline:', error);
      }
    }
  }

  async measurePerformance(page: Page, testName: string): Promise<PerformanceMetrics> {
    // Collect Core Web Vitals and performance metrics
    const metrics = await page.evaluate(() => {
      return new Promise<any>((resolve) => {
        const observer = new PerformanceObserver((list) => {
          const entries = list.getEntries();
          const webVitalsMetrics: any = {};
          
          entries.forEach((entry) => {
            switch (entry.entryType) {
              case 'paint':
                if (entry.name === 'first-contentful-paint') {
                  webVitalsMetrics.firstContentfulPaint = entry.startTime;
                }
                break;
              case 'largest-contentful-paint':
                webVitalsMetrics.largestContentfulPaint = entry.startTime;
                break;
              case 'first-input':
                webVitalsMetrics.firstInputDelay = entry.processingStart - entry.startTime;
                break;
              case 'layout-shift':
                if (!webVitalsMetrics.cumulativeLayoutShift) {
                  webVitalsMetrics.cumulativeLayoutShift = 0;
                }
                webVitalsMetrics.cumulativeLayoutShift += entry.value;
                break;
            }
          });
          
          resolve(webVitalsMetrics);
        });
        
        // Observe all performance entry types
        try {
          observer.observe({ entryTypes: ['paint', 'largest-contentful-paint', 'first-input', 'layout-shift'] });
        } catch (e) {
          // Fallback for browsers that don't support all entry types
          resolve({});
        }
        
        // Timeout after 10 seconds
        setTimeout(() => resolve({}), 10000);
      });
    });

    // Get additional performance timing data
    const performanceTiming = await page.evaluate(() => {
      if (!performance || !performance.timing) return {};
      
      const timing = performance.timing;
      const navigation = performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming;
      
      return {
        timeToInteractive: navigation?.domInteractive - navigation?.navigationStart || 0,
        totalBlockingTime: 0, // This would need more complex calculation
        speedIndex: 0, // This would need lighthouse integration
        
        // Resource metrics
        resourceMetrics: performance.getEntriesByType('resource').reduce((acc: any, resource: any) => {
          acc.totalCount = (acc.totalCount || 0) + 1;
          acc.totalSize = (acc.totalSize || 0) + (resource.transferSize || 0);
          
          if (resource.name.includes('.js')) {
            acc.javascriptSize = (acc.javascriptSize || 0) + (resource.transferSize || 0);
          } else if (resource.name.includes('.css')) {
            acc.cssSize = (acc.cssSize || 0) + (resource.transferSize || 0);
          } else if (resource.name.match(/\.(jpg|jpeg|png|gif|webp|svg)$/)) {
            acc.imageSize = (acc.imageSize || 0) + (resource.transferSize || 0);
          }
          
          return acc;
        }, {})
      };
    });

    // Get memory usage if available
    const memoryUsage = await page.evaluate(() => {
      if ((window.performance as any).memory) {
        return (window.performance as any).memory.usedJSHeapSize;
      }
      return 0;
    });

    // Measure app-specific metrics
    const appMetrics = await this.measureAppSpecificMetrics(page);

    // Combine all metrics
    const combinedMetrics: PerformanceMetrics = {
      // Core Web Vitals
      firstContentfulPaint: metrics.firstContentfulPaint || 0,
      largestContentfulPaint: metrics.largestContentfulPaint || 0,
      firstInputDelay: metrics.firstInputDelay || 0,
      cumulativeLayoutShift: metrics.cumulativeLayoutShift || 0,
      
      // Performance metrics
      timeToInteractive: performanceTiming.timeToInteractive || 0,
      totalBlockingTime: performanceTiming.totalBlockingTime || 0,
      speedIndex: performanceTiming.speedIndex || 0,
      
      // Resource metrics
      totalResourceCount: performanceTiming.resourceMetrics?.totalCount || 0,
      totalResourceSize: performanceTiming.resourceMetrics?.totalSize || 0,
      javascriptSize: performanceTiming.resourceMetrics?.javascriptSize || 0,
      cssSize: performanceTiming.resourceMetrics?.cssSize || 0,
      imageSize: performanceTiming.resourceMetrics?.imageSize || 0,
      
      // App-specific metrics
      dashboardLoadTime: appMetrics.dashboardLoadTime,
      chartRenderTime: appMetrics.chartRenderTime,
      websocketConnectionTime: appMetrics.websocketConnectionTime,
      memoryUsage,
      
      // Interaction metrics
      buttonClickResponseTime: appMetrics.buttonClickResponseTime,
      formSubmissionTime: appMetrics.formSubmissionTime,
      searchResponseTime: appMetrics.searchResponseTime,
      navigationTime: appMetrics.navigationTime
    };

    return combinedMetrics;
  }

  private async measureAppSpecificMetrics(page: Page): Promise<Partial<PerformanceMetrics>> {
    const metrics: Partial<PerformanceMetrics> = {};

    try {
      // Measure dashboard load time
      const dashboardStart = Date.now();
      await page.goto('/dashboard', { waitUntil: 'networkidle' });
      await page.waitForSelector('[data-testid="dashboard-content"]', { timeout: 10000 });
      metrics.dashboardLoadTime = Date.now() - dashboardStart;

      // Measure chart render time
      const chartStart = Date.now();
      if (await page.isVisible('[data-testid="chart-container"]')) {
        await page.waitForSelector('[data-testid="chart-canvas"], [data-testid="chart-svg"]', { timeout: 5000 });
        metrics.chartRenderTime = Date.now() - chartStart;
      } else {
        metrics.chartRenderTime = 0;
      }

      // Measure WebSocket connection time
      const wsStart = Date.now();
      if (await page.isVisible('[data-testid="start-realtime"]')) {
        await page.click('[data-testid="start-realtime"]');
        await page.waitForFunction(() => {
          return (window as any).anomalyWebSocket && 
                 (window as any).anomalyWebSocket.readyState === WebSocket.OPEN;
        }, { timeout: 5000 });
        metrics.websocketConnectionTime = Date.now() - wsStart;
      } else {
        metrics.websocketConnectionTime = 0;
      }

      // Measure button click response time
      const buttonStart = Date.now();
      if (await page.isVisible('[data-testid="refresh-button"]')) {
        await page.click('[data-testid="refresh-button"]');
        await page.waitForTimeout(100); // Wait for immediate response
        metrics.buttonClickResponseTime = Date.now() - buttonStart;
      } else {
        metrics.buttonClickResponseTime = 0;
      }

      // Measure search response time
      const searchStart = Date.now();
      if (await page.isVisible('[data-testid="search-input"]')) {
        await page.fill('[data-testid="search-input"]', 'test');
        await page.waitForTimeout(500); // Wait for search results
        metrics.searchResponseTime = Date.now() - searchStart;
        await page.fill('[data-testid="search-input"]', ''); // Clear search
      } else {
        metrics.searchResponseTime = 0;
      }

      // Measure navigation time
      const navStart = Date.now();
      if (await page.isVisible('[data-testid="detectors-tab"]')) {
        await page.click('[data-testid="detectors-tab"]');
        await page.waitForSelector('[data-testid="detectors-list"]', { timeout: 5000 });
        metrics.navigationTime = Date.now() - navStart;
      } else {
        metrics.navigationTime = 0;
      }

      // Measure form submission time (without actually submitting)
      const formStart = Date.now();
      if (await page.isVisible('[data-testid="detector-form"]')) {
        await page.fill('input[name="name"]', 'PerformanceTest');
        await page.selectOption('select[name="algorithm"]', 'IsolationForest');
        metrics.formSubmissionTime = Date.now() - formStart;
      } else {
        metrics.formSubmissionTime = 0;
      }

    } catch (error) {
      console.warn('Error measuring app-specific metrics:', error);
    }

    return metrics;
  }

  detectRegressions(currentMetrics: PerformanceMetrics, testName: string): RegressionAlert[] {
    if (!this.currentBaseline || !this.currentBaseline.metrics[testName]) {
      console.log(`No baseline found for test: ${testName}. Creating new baseline.`);
      return [];
    }

    const baselineMetrics = this.currentBaseline.metrics[testName];
    const regressions: RegressionAlert[] = [];

    for (const [metricName, currentValue] of Object.entries(currentMetrics)) {
      const baselineValue = (baselineMetrics as any)[metricName];
      const threshold = this.regressionThresholds[metricName];

      if (baselineValue && threshold && currentValue > 0) {
        const regression = ((currentValue - baselineValue) / baselineValue) * 100;
        
        if (regression > threshold) {
          const severity = this.calculateSeverity(regression, threshold);
          regressions.push({
            metric: metricName,
            current: currentValue,
            baseline: baselineValue,
            regression,
            severity,
            threshold
          });
        }
      }
    }

    return regressions;
  }

  private calculateSeverity(regression: number, threshold: number): 'low' | 'medium' | 'high' | 'critical' {
    if (regression > threshold * 3) return 'critical';
    if (regression > threshold * 2) return 'high';
    if (regression > threshold * 1.5) return 'medium';
    return 'low';
  }

  updateBaseline(metrics: PerformanceMetrics, testName: string): void {
    const buildId = process.env.BUILD_ID || 'local';
    const branch = process.env.BRANCH_NAME || 'main';
    const commit = process.env.COMMIT_SHA || 'unknown';

    if (!this.currentBaseline) {
      this.currentBaseline = {
        timestamp: new Date().toISOString(),
        buildId,
        branch,
        commit,
        metrics: {}
      };
    }

    this.currentBaseline.metrics[testName] = metrics;
    this.currentBaseline.timestamp = new Date().toISOString();
    this.currentBaseline.buildId = buildId;
    this.currentBaseline.branch = branch;
    this.currentBaseline.commit = commit;

    // Save baseline
    const baselineFile = join(this.baselinePath, 'current-baseline.json');
    writeFileSync(baselineFile, JSON.stringify(this.currentBaseline, null, 2));

    // Also save historical baseline
    const historicalFile = join(this.baselinePath, `baseline-${buildId}-${Date.now()}.json`);
    writeFileSync(historicalFile, JSON.stringify(this.currentBaseline, null, 2));
  }

  generateRegressionReport(regressions: RegressionAlert[]): string {
    if (regressions.length === 0) {
      return '✅ No performance regressions detected';
    }

    const critical = regressions.filter(r => r.severity === 'critical');
    const high = regressions.filter(r => r.severity === 'high');
    const medium = regressions.filter(r => r.severity === 'medium');
    const low = regressions.filter(r => r.severity === 'low');

    let report = '🚨 Performance Regressions Detected:\n\n';

    if (critical.length > 0) {
      report += '🔴 Critical Regressions:\n';
      critical.forEach(r => {
        report += `  - ${r.metric}: ${r.regression.toFixed(1)}% slower (${r.current.toFixed(1)}ms vs ${r.baseline.toFixed(1)}ms)\n`;
      });
      report += '\n';
    }

    if (high.length > 0) {
      report += '🟠 High Priority Regressions:\n';
      high.forEach(r => {
        report += `  - ${r.metric}: ${r.regression.toFixed(1)}% slower (${r.current.toFixed(1)}ms vs ${r.baseline.toFixed(1)}ms)\n`;
      });
      report += '\n';
    }

    if (medium.length > 0) {
      report += '🟡 Medium Priority Regressions:\n';
      medium.forEach(r => {
        report += `  - ${r.metric}: ${r.regression.toFixed(1)}% slower\n`;
      });
      report += '\n';
    }

    if (low.length > 0) {
      report += '🟢 Low Priority Regressions:\n';
      low.forEach(r => {
        report += `  - ${r.metric}: ${r.regression.toFixed(1)}% slower\n`;
      });
    }

    return report;
  }
}

test.describe('Performance Regression Detection', () => {
  let detector: PerformanceRegressionDetector;

  test.beforeEach(() => {
    detector = new PerformanceRegressionDetector();
  });

  test('Dashboard performance baseline', async ({ page }) => {
    const testName = 'dashboard_performance';
    
    // Navigate to dashboard and measure performance
    await page.goto('/dashboard', { waitUntil: 'networkidle' });
    const metrics = await detector.measurePerformance(page, testName);
    
    // Detect regressions
    const regressions = detector.detectRegressions(metrics, testName);
    
    // Log results
    console.log(`Performance metrics for ${testName}:`, {
      firstContentfulPaint: `${metrics.firstContentfulPaint.toFixed(1)}ms`,
      largestContentfulPaint: `${metrics.largestContentfulPaint.toFixed(1)}ms`,
      dashboardLoadTime: `${metrics.dashboardLoadTime.toFixed(1)}ms`,
      memoryUsage: `${(metrics.memoryUsage / 1024 / 1024).toFixed(2)}MB`
    });
    
    if (regressions.length > 0) {
      const report = detector.generateRegressionReport(regressions);
      console.log(report);
      
      // Fail test for critical regressions
      const criticalRegressions = regressions.filter(r => r.severity === 'critical');
      if (criticalRegressions.length > 0) {
        throw new Error(`Critical performance regression detected: ${criticalRegressions.map(r => r.metric).join(', ')}`);
      }
    }
    
    // Update baseline if this is a baseline run
    if (process.env.UPDATE_BASELINE === 'true') {
      detector.updateBaseline(metrics, testName);
      console.log('✅ Performance baseline updated');
    }

    // Basic performance assertions
    expect(metrics.firstContentfulPaint).toBeLessThan(3000);
    expect(metrics.dashboardLoadTime).toBeLessThan(5000);
    expect(metrics.memoryUsage).toBeLessThan(100 * 1024 * 1024); // 100MB
  });

  test('Chart rendering performance', async ({ page }) => {
    const testName = 'chart_rendering_performance';
    
    await page.goto('/visualizations', { waitUntil: 'networkidle' });
    const metrics = await detector.measurePerformance(page, testName);
    
    const regressions = detector.detectRegressions(metrics, testName);
    
    console.log(`Chart rendering metrics:`, {
      chartRenderTime: `${metrics.chartRenderTime.toFixed(1)}ms`,
      totalResourceSize: `${(metrics.totalResourceSize / 1024).toFixed(1)}KB`,
      javascriptSize: `${(metrics.javascriptSize / 1024).toFixed(1)}KB`
    });

    if (regressions.length > 0) {
      console.log(detector.generateRegressionReport(regressions));
    }

    if (process.env.UPDATE_BASELINE === 'true') {
      detector.updateBaseline(metrics, testName);
    }

    expect(metrics.chartRenderTime).toBeLessThan(2000);
  });

  test('Form interaction performance', async ({ page }) => {
    const testName = 'form_interaction_performance';
    
    await page.goto('/detectors', { waitUntil: 'networkidle' });
    const metrics = await detector.measurePerformance(page, testName);
    
    const regressions = detector.detectRegressions(metrics, testName);
    
    console.log(`Form interaction metrics:`, {
      buttonClickResponseTime: `${metrics.buttonClickResponseTime.toFixed(1)}ms`,
      formSubmissionTime: `${metrics.formSubmissionTime.toFixed(1)}ms`,
      searchResponseTime: `${metrics.searchResponseTime.toFixed(1)}ms`
    });

    if (regressions.length > 0) {
      console.log(detector.generateRegressionReport(regressions));
    }

    if (process.env.UPDATE_BASELINE === 'true') {
      detector.updateBaseline(metrics, testName);
    }

    expect(metrics.buttonClickResponseTime).toBeLessThan(200);
    expect(metrics.searchResponseTime).toBeLessThan(1000);
  });

  test('WebSocket performance', async ({ page }) => {
    const testName = 'websocket_performance';
    
    await page.goto('/dashboard', { waitUntil: 'networkidle' });
    const metrics = await detector.measurePerformance(page, testName);
    
    const regressions = detector.detectRegressions(metrics, testName);
    
    console.log(`WebSocket metrics:`, {
      websocketConnectionTime: `${metrics.websocketConnectionTime.toFixed(1)}ms`,
      navigationTime: `${metrics.navigationTime.toFixed(1)}ms`
    });

    if (regressions.length > 0) {
      console.log(detector.generateRegressionReport(regressions));
    }

    if (process.env.UPDATE_BASELINE === 'true') {
      detector.updateBaseline(metrics, testName);
    }

    expect(metrics.websocketConnectionTime).toBeLessThan(3000);
  });
});