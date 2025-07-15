import { test, expect } from '@playwright/test'
import {
  collectPerformanceMetrics,
  expectPerformance,
  waitForVisualStability,
  monitorNetworkPerformance,
  analyzeBundleSize
} from './utils'

// Performance thresholds based on Core Web Vitals guidelines
const PERFORMANCE_THRESHOLDS = {
  // Core Web Vitals
  FCP: 1800,    // First Contentful Paint - Good: 1.8s
  LCP: 2500,    // Largest Contentful Paint - Good: 2.5s
  FID: 100,     // First Input Delay - Good: 100ms
  CLS: 0.1,     // Cumulative Layout Shift - Good: 0.1

  // Additional metrics
  TTI: 3800,    // Time to Interactive - Good: 3.8s
  TBT: 200,     // Total Blocking Time - Good: 200ms
  SI: 3400,     // Speed Index - Good: 3.4s

  // Page load metrics
  DOMContentLoaded: 2000,
  FullPageLoad: 4000,

  // Resource metrics
  MaxResources: 50,
  MaxBundleSize: 1024 * 1024, // 1MB

  // Memory metrics
  MaxMemoryUsage: 50 // 50MB
}

const TEST_USER = {
  username: 'perf_test_user',
  password: 'PerfTest123!',
  email: 'perf@test.com'
}

test.describe('Core Web Vitals Performance Tests', () => {
  // Helper function to login
  const loginUser = async (page: any) => {
    await page.goto('/login')
    await page.fill('input[name="username"]', TEST_USER.username)
    await page.fill('input[name="password"]', TEST_USER.password)
    await page.click('button[type="submit"]')
    await page.waitForURL(/.*dashboard|.*\/$/)
  }

  test('should meet Core Web Vitals thresholds on homepage', async ({ page }) => {
    await page.goto('/')
    await waitForVisualStability(page)

    const metrics = await collectPerformanceMetrics(page)

    expectPerformance(metrics)
      .expectFCPUnder(PERFORMANCE_THRESHOLDS.FCP)
      .expectLCPUnder(PERFORMANCE_THRESHOLDS.LCP)
      .expectCLSUnder(PERFORMANCE_THRESHOLDS.CLS)
      .expectTTIUnder(PERFORMANCE_THRESHOLDS.TTI)
      .expectPageLoadUnder(PERFORMANCE_THRESHOLDS.FullPageLoad)
      .expectResourceCountUnder(PERFORMANCE_THRESHOLDS.MaxResources)
      .logMetrics('Homepage')
  })

  test('should meet Core Web Vitals thresholds on dashboard', async ({ page }) => {
    await loginUser(page)
    await page.goto('/dashboard')
    await waitForVisualStability(page)

    // Wait for dashboard-specific elements
    await page.waitForSelector('.metrics-grid', { timeout: 10000 })
    await page.waitForSelector('.chart-container', { timeout: 10000 })

    const metrics = await collectPerformanceMetrics(page)

    expectPerformance(metrics)
      .expectFCPUnder(PERFORMANCE_THRESHOLDS.FCP)
      .expectLCPUnder(PERFORMANCE_THRESHOLDS.LCP)
      .expectCLSUnder(PERFORMANCE_THRESHOLDS.CLS)
      .expectTTIUnder(PERFORMANCE_THRESHOLDS.TTI)
      .expectPageLoadUnder(PERFORMANCE_THRESHOLDS.FullPageLoad)
      .expectMemoryUsageUnder(PERFORMANCE_THRESHOLDS.MaxMemoryUsage)
      .logMetrics('Dashboard')
  })

  test('should meet Core Web Vitals thresholds on datasets page', async ({ page }) => {
    await loginUser(page)
    await page.goto('/datasets')
    await waitForVisualStability(page)

    // Wait for datasets table
    await page.waitForSelector('.datasets-container', { timeout: 10000 })

    const metrics = await collectPerformanceMetrics(page)

    expectPerformance(metrics)
      .expectFCPUnder(PERFORMANCE_THRESHOLDS.FCP)
      .expectLCPUnder(PERFORMANCE_THRESHOLDS.LCP)
      .expectCLSUnder(PERFORMANCE_THRESHOLDS.CLS)
      .expectTTIUnder(PERFORMANCE_THRESHOLDS.TTI)
      .logMetrics('Datasets Page')
  })

  test('should meet Core Web Vitals thresholds on detectors page', async ({ page }) => {
    await loginUser(page)
    await page.goto('/detectors')
    await waitForVisualStability(page)

    const metrics = await collectPerformanceMetrics(page)

    expectPerformance(metrics)
      .expectFCPUnder(PERFORMANCE_THRESHOLDS.FCP)
      .expectLCPUnder(PERFORMANCE_THRESHOLDS.LCP)
      .expectCLSUnder(PERFORMANCE_THRESHOLDS.CLS)
      .expectTTIUnder(PERFORMANCE_THRESHOLDS.TTI)
      .logMetrics('Detectors Page')
  })

  test('should meet Core Web Vitals thresholds on detection page', async ({ page }) => {
    await loginUser(page)
    await page.goto('/detection')
    await waitForVisualStability(page)

    const metrics = await collectPerformanceMetrics(page)

    expectPerformance(metrics)
      .expectFCPUnder(PERFORMANCE_THRESHOLDS.FCP)
      .expectLCPUnder(PERFORMANCE_THRESHOLDS.LCP)
      .expectCLSUnder(PERFORMANCE_THRESHOLDS.CLS)
      .expectTTIUnder(PERFORMANCE_THRESHOLDS.TTI)
      .logMetrics('Detection Page')
  })

  test('should meet Core Web Vitals thresholds on security dashboard', async ({ page }) => {
    await loginUser(page)
    await page.goto('/security-dashboard')
    await waitForVisualStability(page)

    // Wait for security dashboard elements
    await page.waitForSelector('.security-dashboard', { timeout: 10000 })
    await page.waitForSelector('.security-metrics', { timeout: 10000 })

    const metrics = await collectPerformanceMetrics(page)

    expectPerformance(metrics)
      .expectFCPUnder(PERFORMANCE_THRESHOLDS.FCP)
      .expectLCPUnder(PERFORMANCE_THRESHOLDS.LCP)
      .expectCLSUnder(PERFORMANCE_THRESHOLDS.CLS)
      .expectTTIUnder(PERFORMANCE_THRESHOLDS.TTI)
      .logMetrics('Security Dashboard')
  })

  test('should have acceptable bundle sizes', async ({ page }) => {
    const bundleAnalysis = await analyzeBundleSize(page)

    await page.goto('/')
    await waitForVisualStability(page)

    // Wait for all resources to load
    await page.waitForTimeout(2000)

    expect(bundleAnalysis.javascript).toBeLessThan(PERFORMANCE_THRESHOLDS.MaxBundleSize)
    expect(bundleAnalysis.css).toBeLessThan(256 * 1024) // 256KB for CSS
    expect(bundleAnalysis.total).toBeLessThan(PERFORMANCE_THRESHOLDS.MaxBundleSize * 1.5)

    console.log('Bundle size analysis:', {
      javascript: `${(bundleAnalysis.javascript / 1024).toFixed(2)}KB`,
      css: `${(bundleAnalysis.css / 1024).toFixed(2)}KB`,
      images: `${(bundleAnalysis.images / 1024).toFixed(2)}KB`,
      fonts: `${(bundleAnalysis.fonts / 1024).toFixed(2)}KB`,
      total: `${(bundleAnalysis.total / 1024).toFixed(2)}KB`,
      resourceCount: bundleAnalysis.resources.length
    })
  })

  test('should have efficient network performance', async ({ page }) => {
    const networkMonitor = await monitorNetworkPerformance(page)

    await page.goto('/dashboard')
    await waitForVisualStability(page)

    // Allow time for all network requests to complete
    await page.waitForTimeout(3000)

    expect(networkMonitor.requestCount).toBeGreaterThan(0)
    expect(networkMonitor.averageResponseTime).toBeLessThan(1000) // 1 second average
    expect(networkMonitor.slowestRequest.duration).toBeLessThan(3000) // 3 second max

    console.log('Network performance:', {
      requestCount: networkMonitor.requestCount,
      averageResponseTime: `${networkMonitor.averageResponseTime.toFixed(2)}ms`,
      slowestRequest: `${networkMonitor.slowestRequest.duration}ms (${networkMonitor.slowestRequest.url})`,
      fastestRequest: `${networkMonitor.fastestRequest.duration}ms (${networkMonitor.fastestRequest.url})`,
      totalSize: `${(networkMonitor.totalSize / 1024).toFixed(2)}KB`
    })
  })

  test('should maintain performance during navigation', async ({ page }) => {
    await loginUser(page)

    const pages = [
      { path: '/dashboard', name: 'Dashboard' },
      { path: '/datasets', name: 'Datasets' },
      { path: '/detectors', name: 'Detectors' },
      { path: '/detection', name: 'Detection' }
    ]

    const navigationMetrics: any[] = []

    for (const pageConfig of pages) {
      const startTime = Date.now()

      await page.goto(pageConfig.path)
      await waitForVisualStability(page)

      const navigationTime = Date.now() - startTime
      const metrics = await collectPerformanceMetrics(page)

      navigationMetrics.push({
        page: pageConfig.name,
        navigationTime,
        metrics
      })

      // Each page should load within reasonable time
      expect(navigationTime).toBeLessThan(5000)
      expect(metrics.fcp).toBeLessThan(PERFORMANCE_THRESHOLDS.FCP)
      expect(metrics.lcp).toBeLessThan(PERFORMANCE_THRESHOLDS.LCP)
    }

    console.log('Navigation performance:')
    navigationMetrics.forEach(metric => {
      console.log(`${metric.page}: ${metric.navigationTime}ms navigation, ${metric.metrics.fcp.toFixed(2)}ms FCP, ${metric.metrics.lcp.toFixed(2)}ms LCP`)
    })
  })

  test('should handle chart rendering performance efficiently', async ({ page }) => {
    await loginUser(page)
    await page.goto('/dashboard')

    // Wait for page to load
    await waitForVisualStability(page)

    // Measure chart rendering time
    const chartRenderStart = Date.now()

    // Wait for charts to render
    await page.waitForSelector('canvas, .chart', { timeout: 15000 })

    // Wait for all charts to be fully rendered
    await page.waitForFunction(() => {
      const canvases = document.querySelectorAll('canvas')
      return Array.from(canvases).every(canvas => {
        return canvas.width > 0 && canvas.height > 0
      })
    }, { timeout: 15000 })

    const chartRenderTime = Date.now() - chartRenderStart

    expect(chartRenderTime).toBeLessThan(5000) // Charts should render within 5 seconds

    // Test chart interaction performance
    const chartElements = await page.locator('canvas').count()
    expect(chartElements).toBeGreaterThan(0)

    // Test hover performance on first chart
    if (chartElements > 0) {
      const firstChart = page.locator('canvas').first()

      const hoverStart = Date.now()
      await firstChart.hover()
      await page.waitForTimeout(100) // Wait for any hover effects
      const hoverTime = Date.now() - hoverStart

      expect(hoverTime).toBeLessThan(100) // Hover should be immediate
    }

    console.log('Chart performance:', {
      renderTime: `${chartRenderTime}ms`,
      chartCount: chartElements
    })
  })

  test('should handle form interactions efficiently', async ({ page }) => {
    await loginUser(page)
    await page.goto('/detectors')

    // Open create detector form
    await page.click('.create-detector-btn')
    await page.waitForSelector('.detector-form, #detector-modal')

    // Measure form interaction performance
    const formInteractionStart = Date.now()

    // Fill form fields
    await page.fill('input[name="name"]', 'Performance Test Detector')
    await page.selectOption('select[name="algorithm"]', 'IsolationForest')
    await page.fill('textarea[name="description"]', 'A detector for performance testing')

    const formInteractionTime = Date.now() - formInteractionStart

    expect(formInteractionTime).toBeLessThan(2000) // Form interactions should be fast

    // Test form validation performance
    await page.locator('input[name="name"]').clear()

    const validationStart = Date.now()
    await page.click('.submit-detector, button[type="submit"]')

    // Wait for validation error to appear
    await page.waitForSelector('.validation-error, .error-message', { timeout: 1000 })

    const validationTime = Date.now() - validationStart

    expect(validationTime).toBeLessThan(500) // Validation should be immediate

    console.log('Form performance:', {
      interactionTime: `${formInteractionTime}ms`,
      validationTime: `${validationTime}ms`
    })
  })

  test('should maintain performance with large datasets', async ({ page }) => {
    await loginUser(page)

    // Mock large dataset response
    await page.route('**/api/v1/datasets**', route => {
      const largeDataset = {
        datasets: Array.from({ length: 100 }, (_, i) => ({
          id: i + 1,
          name: `Large Dataset ${i + 1}`,
          size: Math.floor(Math.random() * 1000000),
          created_at: new Date().toISOString(),
          description: `This is a large dataset for performance testing with ID ${i + 1}`,
          status: 'ready'
        }))
      }

      // Add realistic delay
      setTimeout(() => {
        route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify(largeDataset)
        })
      }, 300)
    })

    const loadStart = Date.now()
    await page.goto('/datasets')

    // Wait for table to load
    await page.waitForSelector('.data-table tbody tr', { timeout: 15000 })

    const loadTime = Date.now() - loadStart

    // Collect performance metrics
    const metrics = await collectPerformanceMetrics(page)

    expect(loadTime).toBeLessThan(8000) // Large dataset should load within 8 seconds
    expectPerformance(metrics)
      .expectLCPUnder(PERFORMANCE_THRESHOLDS.LCP * 1.5) // Allow 50% more time for large datasets
      .expectMemoryUsageUnder(PERFORMANCE_THRESHOLDS.MaxMemoryUsage * 1.2) // Allow 20% more memory

    // Test table interactions with large dataset
    const sortStart = Date.now()
    const sortableHeader = page.locator('th[data-sortable="true"], th.sortable').first()

    if (await sortableHeader.count() > 0) {
      await sortableHeader.click()
      await page.waitForTimeout(500)

      const sortTime = Date.now() - sortStart
      expect(sortTime).toBeLessThan(1000) // Sorting should be fast even with large datasets
    }

    console.log('Large dataset performance:', {
      loadTime: `${loadTime}ms`,
      datasetCount: 100,
      fcp: `${metrics.fcp.toFixed(2)}ms`,
      lcp: `${metrics.lcp.toFixed(2)}ms`
    })
  })
})
