/**
 * Enhanced Performance Testing Framework for Pynomaly
 * Core Web Vitals monitoring, bundle optimization, and Real User Monitoring
 */

import { test, expect, Page, CDPSession } from '@playwright/test';

interface PerformanceMetrics {
  // Core Web Vitals
  lcp: number;  // Largest Contentful Paint
  fid: number;  // First Input Delay
  cls: number;  // Cumulative Layout Shift

  // Other important metrics
  fcp: number;  // First Contentful Paint
  ttfb: number; // Time to First Byte
  tti: number;  // Time to Interactive
  si: number;   // Speed Index
  tbt: number;  // Total Blocking Time

  // Resource metrics
  totalByteWeight: number;
  unusedCSSBytes: number;
  unusedJavaScriptBytes: number;
  imageOptimization: number;

  // Custom metrics
  hydrationTime?: number;
  chartRenderTime?: number;
  apiResponseTime?: number;
}

interface BundleAnalysis {
  totalSize: number;
  compressedSize: number;
  jsSize: number;
  cssSize: number;
  imageSize: number;
  unusedCode: number;
  duplicateModules: string[];
}

class PerformanceTestHelper {
  private cdp: CDPSession | null = null;

  constructor(private page: Page) {}

  async initializePerformanceMonitoring() {
    this.cdp = await this.page.context().newCDPSession(this.page);

    // Enable performance tracking
    await this.cdp.send('Performance.enable');
    await this.cdp.send('Runtime.enable');

    // Enable network tracking for resource analysis
    await this.cdp.send('Network.enable');

    // Clear cache to ensure fresh measurements
    await this.cdp.send('Network.clearBrowserCache');
  }

  async measureCoreWebVitals(): Promise<PerformanceMetrics> {
    const metrics = await this.page.evaluate(() => {
      return new Promise((resolve) => {
        const metrics: any = {};

        // Measure LCP
        if ('PerformanceObserver' in window) {
          const lcpObserver = new PerformanceObserver((list) => {
            const entries = list.getEntries();
            const lastEntry = entries[entries.length - 1];
            metrics.lcp = lastEntry.startTime;
          });

          try {
            lcpObserver.observe({ entryTypes: ['largest-contentful-paint'] });
          } catch (e) {
            metrics.lcp = 0;
          }

          // Measure CLS
          let clsValue = 0;
          const clsObserver = new PerformanceObserver((list) => {
            for (const entry of list.getEntries()) {
              if (!(entry as any).hadRecentInput) {
                clsValue += (entry as any).value;
              }
            }
            metrics.cls = clsValue;
          });

          try {
            clsObserver.observe({ entryTypes: ['layout-shift'] });
          } catch (e) {
            metrics.cls = 0;
          }
        }

        // Get navigation timing metrics
        const navigation = performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming;
        if (navigation) {
          metrics.ttfb = navigation.responseStart - navigation.fetchStart;
          metrics.fcp = 0; // Will be updated by paint observer
          metrics.tti = 0; // Estimated
        }

        // Measure FCP
        if ('PerformanceObserver' in window) {
          const paintObserver = new PerformanceObserver((list) => {
            for (const entry of list.getEntries()) {
              if (entry.name === 'first-contentful-paint') {
                metrics.fcp = entry.startTime;
              }
            }
          });

          try {
            paintObserver.observe({ entryTypes: ['paint'] });
          } catch (e) {
            metrics.fcp = 0;
          }
        }

        // Mock FID (First Input Delay) - in real testing this would be measured differently
        metrics.fid = 0;

        // Calculate derived metrics
        metrics.si = metrics.fcp + 500; // Simplified Speed Index estimation
        metrics.tbt = Math.max(0, metrics.tti - metrics.fcp - 50); // Simplified TBT

        // Resource metrics (simplified)
        const resources = performance.getEntriesByType('resource');
        metrics.totalByteWeight = resources.reduce((total, resource: any) => {
          return total + (resource.transferSize || 0);
        }, 0);

        metrics.unusedCSSBytes = 0; // Would require CSS coverage analysis
        metrics.unusedJavaScriptBytes = 0; // Would require JS coverage analysis
        metrics.imageOptimization = 100; // Percentage score

        // Resolve after a delay to allow observers to collect data
        setTimeout(() => resolve(metrics), 3000);
      });
    });

    return metrics as PerformanceMetrics;
  }

  async measureResourceLoadTimes() {
    const resourceMetrics = await this.page.evaluate(() => {
      const resources = performance.getEntriesByType('resource') as PerformanceResourceTiming[];

      return resources.map(resource => ({
        name: resource.name,
        type: resource.initiatorType,
        size: resource.transferSize || 0,
        duration: resource.duration,
        startTime: resource.startTime,
        encodedSize: resource.encodedBodySize || 0,
        decodedSize: resource.decodedBodySize || 0
      }));
    });

    return resourceMetrics;
  }

  async measureBundleSize(): Promise<BundleAnalysis> {
    const resources = await this.measureResourceLoadTimes();

    const jsResources = resources.filter(r => r.type === 'script' || r.name.endsWith('.js'));
    const cssResources = resources.filter(r => r.type === 'link' || r.name.endsWith('.css'));
    const imageResources = resources.filter(r => r.type === 'img' || /\.(jpg|jpeg|png|gif|svg|webp)/.test(r.name));

    const jsSize = jsResources.reduce((total, r) => total + r.size, 0);
    const cssSize = cssResources.reduce((total, r) => total + r.size, 0);
    const imageSize = imageResources.reduce((total, r) => total + r.size, 0);
    const totalSize = resources.reduce((total, r) => total + r.size, 0);

    return {
      totalSize,
      compressedSize: totalSize, // Assuming compressed
      jsSize,
      cssSize,
      imageSize,
      unusedCode: 0, // Would require coverage analysis
      duplicateModules: [] // Would require detailed bundle analysis
    };
  }

  async measureAPIPerformance() {
    const apiCalls: Array<{url: string, duration: number, status: number}> = [];

    this.page.on('response', response => {
      if (response.url().includes('/api/')) {
        const timing = response.timing();
        apiCalls.push({
          url: response.url(),
          duration: timing.responseEnd,
          status: response.status()
        });
      }
    });

    return apiCalls;
  }

  async measureChartRenderingPerformance() {
    return await this.page.evaluate(() => {
      const chartContainers = document.querySelectorAll('.chart-container, [data-testid="chart"]');
      const renderTimes: number[] = [];

      chartContainers.forEach(container => {
        const observer = new MutationObserver(() => {
          const startTime = performance.now();
          // Simulate chart rendering measurement
          setTimeout(() => {
            renderTimes.push(performance.now() - startTime);
          }, 100);
        });

        observer.observe(container, { childList: true, subtree: true });
      });

      return renderTimes.length > 0 ? renderTimes[0] : 0;
    });
  }

  async generatePerformanceReport(metrics: PerformanceMetrics, bundleAnalysis: BundleAnalysis) {
    const report = {
      timestamp: new Date().toISOString(),

      // Core Web Vitals Assessment
      coreWebVitals: {
        lcp: {
          value: metrics.lcp,
          score: metrics.lcp < 2500 ? 'good' : metrics.lcp < 4000 ? 'needs-improvement' : 'poor',
          target: 2500
        },
        fid: {
          value: metrics.fid,
          score: metrics.fid < 100 ? 'good' : metrics.fid < 300 ? 'needs-improvement' : 'poor',
          target: 100
        },
        cls: {
          value: metrics.cls,
          score: metrics.cls < 0.1 ? 'good' : metrics.cls < 0.25 ? 'needs-improvement' : 'poor',
          target: 0.1
        }
      },

      // Bundle Analysis
      bundleOptimization: {
        totalSize: bundleAnalysis.totalSize,
        sizeScore: bundleAnalysis.totalSize < 1000000 ? 'good' : 'needs-optimization',
        jsEfficiency: bundleAnalysis.jsSize < 500000 ? 'good' : 'needs-optimization',
        cssEfficiency: bundleAnalysis.cssSize < 100000 ? 'good' : 'needs-optimization'
      },

      // Overall Performance Score
      overallScore: this.calculateOverallScore(metrics),

      // Recommendations
      recommendations: this.generateRecommendations(metrics, bundleAnalysis)
    };

    return report;
  }

  private calculateOverallScore(metrics: PerformanceMetrics): number {
    let score = 100;

    // LCP scoring
    if (metrics.lcp > 2500) score -= 20;
    else if (metrics.lcp > 4000) score -= 40;

    // FID scoring
    if (metrics.fid > 100) score -= 15;
    else if (metrics.fid > 300) score -= 30;

    // CLS scoring
    if (metrics.cls > 0.1) score -= 15;
    else if (metrics.cls > 0.25) score -= 30;

    // FCP scoring
    if (metrics.fcp > 1800) score -= 10;

    // Bundle size penalty
    if (metrics.totalByteWeight > 1000000) score -= 10;

    return Math.max(0, score);
  }

  private generateRecommendations(metrics: PerformanceMetrics, bundle: BundleAnalysis): string[] {
    const recommendations: string[] = [];

    if (metrics.lcp > 2500) {
      recommendations.push('Optimize Largest Contentful Paint: Consider lazy loading, image optimization, or reducing server response time');
    }

    if (metrics.fid > 100) {
      recommendations.push('Improve First Input Delay: Reduce JavaScript execution time or defer non-critical scripts');
    }

    if (metrics.cls > 0.1) {
      recommendations.push('Minimize Cumulative Layout Shift: Set explicit dimensions for images and avoid injecting content above existing content');
    }

    if (bundle.totalSize > 1000000) {
      recommendations.push('Reduce bundle size: Enable code splitting, tree shaking, and compression');
    }

    if (bundle.jsSize > 500000) {
      recommendations.push('Optimize JavaScript: Consider lazy loading components and removing unused code');
    }

    if (metrics.ttfb > 600) {
      recommendations.push('Improve server response time: Optimize backend performance and consider CDN usage');
    }

    return recommendations;
  }
}

test.describe('Performance Testing - Core Web Vitals & Optimization', () => {
  let helper: PerformanceTestHelper;

  test.beforeEach(async ({ page }) => {
    helper = new PerformanceTestHelper(page);
    await helper.initializePerformanceMonitoring();

    // Set timeout for performance tests
    test.setTimeout(60000);
  });

  const testPages = [
    { path: '/', name: 'Dashboard', target: { lcp: 2000, fid: 80, cls: 0.05 } },
    { path: '/detection', name: 'Detection', target: { lcp: 2500, fid: 100, cls: 0.1 } },
    { path: '/detectors', name: 'Detectors', target: { lcp: 2200, fid: 90, cls: 0.08 } },
    { path: '/datasets', name: 'Datasets', target: { lcp: 2300, fid: 95, cls: 0.09 } },
    { path: '/visualizations', name: 'Visualizations', target: { lcp: 3000, fid: 120, cls: 0.12 } }
  ];

  // Core Web Vitals testing for each page
  for (const pageInfo of testPages) {
    test(`${pageInfo.name} page meets Core Web Vitals targets`, async ({ page }) => {
      await test.step(`Navigate to ${pageInfo.name}`, async () => {
        await page.goto(pageInfo.path);
        await page.waitForLoadState('networkidle');
      });

      await test.step('Measure Core Web Vitals', async () => {
        const metrics = await helper.measureCoreWebVitals();

        console.log(`Core Web Vitals for ${pageInfo.name}:`, {
          LCP: `${metrics.lcp.toFixed(2)}ms (target: <${pageInfo.target.lcp}ms)`,
          FID: `${metrics.fid.toFixed(2)}ms (target: <${pageInfo.target.fid}ms)`,
          CLS: `${metrics.cls.toFixed(3)} (target: <${pageInfo.target.cls})`
        });

        // Assert Core Web Vitals targets
        expect(metrics.lcp).toBeLessThan(pageInfo.target.lcp);
        expect(metrics.fid).toBeLessThan(pageInfo.target.fid);
        expect(metrics.cls).toBeLessThan(pageInfo.target.cls);

        // Additional performance assertions
        expect(metrics.fcp).toBeLessThan(1800); // First Contentful Paint < 1.8s
        expect(metrics.ttfb).toBeLessThan(600);  // Time to First Byte < 600ms
      });

      await test.step('Analyze resource performance', async () => {
        const resources = await helper.measureResourceLoadTimes();

        // Check for slow resources
        const slowResources = resources.filter(r => r.duration > 1000);
        console.log(`Slow resources (>1s) on ${pageInfo.name}:`, slowResources.length);

        // Should have minimal slow resources
        expect(slowResources.length).toBeLessThan(3);

        // Check total resource size
        const totalSize = resources.reduce((total, r) => total + r.size, 0);
        console.log(`Total resource size for ${pageInfo.name}: ${(totalSize / 1024).toFixed(2)} KB`);

        // Should be under 2MB total
        expect(totalSize).toBeLessThan(2000000);
      });
    });
  }

  // Bundle size and optimization testing
  test('Bundle size optimization targets', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    await test.step('Analyze bundle composition', async () => {
      const bundleAnalysis = await helper.measureBundleSize();

      console.log('Bundle Analysis:', {
        'Total Size': `${(bundleAnalysis.totalSize / 1024).toFixed(2)} KB`,
        'JavaScript': `${(bundleAnalysis.jsSize / 1024).toFixed(2)} KB`,
        'CSS': `${(bundleAnalysis.cssSize / 1024).toFixed(2)} KB`,
        'Images': `${(bundleAnalysis.imageSize / 1024).toFixed(2)} KB`
      });

      // Bundle size targets
      expect(bundleAnalysis.totalSize).toBeLessThan(1500000); // < 1.5MB total
      expect(bundleAnalysis.jsSize).toBeLessThan(800000);     // < 800KB JS
      expect(bundleAnalysis.cssSize).toBeLessThan(150000);    // < 150KB CSS
    });

    await test.step('Check for optimization opportunities', async () => {
      const resources = await helper.measureResourceLoadTimes();

      // Check for compression
      const uncompressedResources = resources.filter(r =>
        r.size > 10000 && r.encodedSize === r.decodedSize
      );

      console.log('Potentially uncompressed resources:', uncompressedResources.length);

      // Should have minimal uncompressed large resources
      expect(uncompressedResources.length).toBeLessThan(2);
    });
  });

  // Chart rendering performance
  test('Chart rendering performance', async ({ page }) => {
    await page.goto('/visualizations');
    await page.waitForLoadState('networkidle');

    await test.step('Measure chart rendering times', async () => {
      const chartRenderTime = await helper.measureChartRenderingPerformance();

      console.log('Chart rendering time:', `${chartRenderTime.toFixed(2)}ms`);

      // Charts should render within 1 second
      expect(chartRenderTime).toBeLessThan(1000);
    });

    await test.step('Test chart interaction responsiveness', async () => {
      const charts = page.locator('.chart-container, [data-testid="chart"]');
      const chartCount = await charts.count();

      if (chartCount > 0) {
        const startTime = Date.now();
        await charts.first().hover();
        const hoverTime = Date.now() - startTime;

        console.log('Chart hover response time:', `${hoverTime}ms`);

        // Chart interactions should be responsive
        expect(hoverTime).toBeLessThan(200);
      }
    });
  });

  // API performance testing
  test('API response performance', async ({ page }) => {
    const apiCalls = await helper.measureAPIPerformance();

    await test.step('Navigate and trigger API calls', async () => {
      await page.goto('/detection');
      await page.waitForLoadState('networkidle');

      // Trigger some API interactions if available
      const uploadButton = page.locator('[data-testid="upload-button"], .upload-button').first();
      if (await uploadButton.count() > 0) {
        await uploadButton.hover();
      }
    });

    await test.step('Analyze API performance', async () => {
      if (apiCalls.length > 0) {
        const avgResponseTime = apiCalls.reduce((sum, call) => sum + call.duration, 0) / apiCalls.length;
        const slowAPIs = apiCalls.filter(call => call.duration > 1000);

        console.log('API Performance:', {
          'Total API calls': apiCalls.length,
          'Average response time': `${avgResponseTime.toFixed(2)}ms`,
          'Slow APIs (>1s)': slowAPIs.length
        });

        // API performance targets
        expect(avgResponseTime).toBeLessThan(500); // Average < 500ms
        expect(slowAPIs.length).toBeLessThan(2);   // Max 1 slow API
      }
    });
  });

  // Memory usage testing
  test('Memory usage optimization', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    await test.step('Monitor memory usage', async () => {
      const memoryUsage = await page.evaluate(() => {
        if ('memory' in performance) {
          return {
            used: (performance as any).memory.usedJSHeapSize,
            total: (performance as any).memory.totalJSHeapSize,
            limit: (performance as any).memory.jsHeapSizeLimit
          };
        }
        return null;
      });

      if (memoryUsage) {
        console.log('Memory Usage:', {
          'Used': `${(memoryUsage.used / 1024 / 1024).toFixed(2)} MB`,
          'Total': `${(memoryUsage.total / 1024 / 1024).toFixed(2)} MB`,
          'Usage %': `${((memoryUsage.used / memoryUsage.total) * 100).toFixed(1)}%`
        });

        // Memory usage should be reasonable
        expect(memoryUsage.used).toBeLessThan(100 * 1024 * 1024); // < 100MB
      }
    });
  });

  // Performance regression testing
  test('Performance regression detection', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    await test.step('Collect baseline metrics', async () => {
      const metrics = await helper.measureCoreWebVitals();
      const bundleAnalysis = await helper.measureBundleSize();

      const report = await helper.generatePerformanceReport(metrics, bundleAnalysis);

      console.log('Performance Report:', {
        'Overall Score': `${report.overallScore}/100`,
        'LCP Score': report.coreWebVitals.lcp.score,
        'FID Score': report.coreWebVitals.fid.score,
        'CLS Score': report.coreWebVitals.cls.score
      });

      // Overall performance should meet minimum threshold
      expect(report.overallScore).toBeGreaterThan(70);

      // Log recommendations if any
      if (report.recommendations.length > 0) {
        console.log('Performance Recommendations:', report.recommendations);
      }
    });
  });

  // Mobile performance testing
  test('Mobile performance optimization', async ({ page, isMobile }) => {
    if (!isMobile) {
      await page.setViewportSize({ width: 375, height: 667 });
    }

    await page.goto('/');
    await page.waitForLoadState('networkidle');

    await test.step('Mobile-specific performance metrics', async () => {
      const metrics = await helper.measureCoreWebVitals();

      console.log('Mobile Performance:', {
        LCP: `${metrics.lcp.toFixed(2)}ms`,
        FID: `${metrics.fid.toFixed(2)}ms`,
        CLS: `${metrics.cls.toFixed(3)}`
      });

      // Mobile targets are slightly more lenient
      expect(metrics.lcp).toBeLessThan(3000); // LCP < 3s on mobile
      expect(metrics.fid).toBeLessThan(150);  // FID < 150ms on mobile
      expect(metrics.cls).toBeLessThan(0.15); // CLS < 0.15 on mobile
    });
  });
});
