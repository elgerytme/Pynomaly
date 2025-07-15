import { test, expect, Page, BrowserContext } from '@playwright/test'

// Performance testing configuration
const BASE_URL = process.env.BASE_URL || 'http://localhost:8000'

// Performance thresholds (in milliseconds)
const PERFORMANCE_THRESHOLDS = {
  firstContentfulPaint: 1500,
  largestContentfulPaint: 2500,
  firstInputDelay: 100,
  cumulativeLayoutShift: 0.1,
  timeToInteractive: 3000,
  totalBlockingTime: 300,
  speedIndex: 2000
}

// Test user for authenticated performance testing
const TEST_USER = {
  username: 'perf_test_user',
  password: 'PerfTest123!',
  email: 'perf@test.com'
}

// Page configurations for performance testing
const PAGES_TO_TEST = [
  { path: '/', name: 'homepage', requiresAuth: false, expectedElements: ['.hero', '.features'] },
  { path: '/login', name: 'login-page', requiresAuth: false, expectedElements: ['form#login-form'] },
  { path: '/dashboard', name: 'dashboard', requiresAuth: true, expectedElements: ['.metrics-grid', '.chart-container'] },
  { path: '/datasets', name: 'datasets-list', requiresAuth: true, expectedElements: ['.datasets-container', '.data-table'] },
  { path: '/detectors', name: 'detectors-list', requiresAuth: true, expectedElements: ['.detectors-container'] },
  { path: '/detection', name: 'detection-page', requiresAuth: true, expectedElements: ['.detection-container'] },
  { path: '/security-dashboard', name: 'security-dashboard', requiresAuth: true, expectedElements: ['.security-dashboard', '.security-metrics'] }
]

test.describe('Web UI Performance Testing', () => {
  let page: Page
  let context: BrowserContext

  test.beforeEach(async ({ page: testPage, context: testContext }) => {
    page = testPage
    context = testContext

    // Enable performance metrics collection
    await page.addInitScript(() => {
      // Performance observer for Core Web Vitals
      window.performanceMetrics = {
        fcp: 0,
        lcp: 0,
        fid: 0,
        cls: 0,
        tti: 0,
        tbt: 0
      }

      // Observe paint timings
      const paintObserver = new PerformanceObserver((list) => {
        for (const entry of list.getEntries()) {
          if (entry.name === 'first-contentful-paint') {
            window.performanceMetrics.fcp = entry.startTime
          }
        }
      })
      paintObserver.observe({ entryTypes: ['paint'] })

      // Observe LCP
      const lcpObserver = new PerformanceObserver((list) => {
        const entries = list.getEntries()
        const lastEntry = entries[entries.length - 1]
        window.performanceMetrics.lcp = lastEntry.startTime
      })
      lcpObserver.observe({ entryTypes: ['largest-contentful-paint'] })

      // Observe CLS
      let clsValue = 0
      const clsObserver = new PerformanceObserver((list) => {
        for (const entry of list.getEntries()) {
          if (!entry.hadRecentInput) {
            clsValue += entry.value
          }
        }
        window.performanceMetrics.cls = clsValue
      })
      clsObserver.observe({ entryTypes: ['layout-shift'] })

      // Observe First Input Delay
      const fidObserver = new PerformanceObserver((list) => {
        for (const entry of list.getEntries()) {
          window.performanceMetrics.fid = entry.processingStart - entry.startTime
        }
      })
      fidObserver.observe({ entryTypes: ['first-input'] })

      // Memory usage tracking
      window.memoryMetrics = {
        initialHeapSize: 0,
        peakHeapSize: 0,
        currentHeapSize: 0
      }

      if ('memory' in performance) {
        window.memoryMetrics.initialHeapSize = (performance as any).memory.usedJSHeapSize

        setInterval(() => {
          const currentHeap = (performance as any).memory.usedJSHeapSize
          window.memoryMetrics.currentHeapSize = currentHeap
          if (currentHeap > window.memoryMetrics.peakHeapSize) {
            window.memoryMetrics.peakHeapSize = currentHeap
          }
        }, 1000)
      }
    })
  })

  // Helper function to login
  const loginUser = async (page: Page) => {
    await page.goto(`${BASE_URL}/login`)
    await page.fill('input[name="username"]', TEST_USER.username)
    await page.fill('input[name="password"]', TEST_USER.password)
    await page.click('button[type="submit"]')
    await page.waitForURL(/.*dashboard|.*\/$/)
  }

  // Helper function to collect performance metrics
  const collectPerformanceMetrics = async (page: Page) => {
    // Wait for page to be fully loaded
    await page.waitForLoadState('networkidle')

    // Get Core Web Vitals
    const metrics = await page.evaluate(() => {
      return {
        ...window.performanceMetrics,
        navigationTiming: performance.getEntriesByType('navigation')[0],
        resourceTiming: performance.getEntriesByType('resource').length,
        memory: window.memoryMetrics
      }
    })

    // Calculate additional metrics
    const navigation = metrics.navigationTiming as PerformanceNavigationTiming
    if (navigation) {
      metrics.timeToInteractive = navigation.domInteractive - navigation.navigationStart
      metrics.domContentLoaded = navigation.domContentLoadedEventEnd - navigation.navigationStart
      metrics.fullPageLoad = navigation.loadEventEnd - navigation.navigationStart
    }

    return metrics
  }

  test.describe('Core Web Vitals Testing', () => {
    for (const pageConfig of PAGES_TO_TEST) {
      test(`should meet Core Web Vitals thresholds for ${pageConfig.name}`, async () => {
        if (pageConfig.requiresAuth) {
          await loginUser(page)
        }

        // Navigate to page and measure performance
        const navigationPromise = page.goto(`${BASE_URL}${pageConfig.path}`)
        await navigationPromise

        // Wait for expected elements to ensure page is functional
        for (const selector of pageConfig.expectedElements) {
          await page.waitForSelector(selector, { timeout: 10000 })
        }

        // Collect performance metrics
        const metrics = await collectPerformanceMetrics(page)

        // Verify Core Web Vitals thresholds
        expect(metrics.fcp).toBeLessThan(PERFORMANCE_THRESHOLDS.firstContentfulPaint)
        expect(metrics.lcp).toBeLessThan(PERFORMANCE_THRESHOLDS.largestContentfulPaint)
        expect(metrics.cls).toBeLessThan(PERFORMANCE_THRESHOLDS.cumulativeLayoutShift)

        // Verify loading performance
        if (metrics.timeToInteractive) {
          expect(metrics.timeToInteractive).toBeLessThan(PERFORMANCE_THRESHOLDS.timeToInteractive)
        }

        console.log(`Performance metrics for ${pageConfig.name}:`, {
          FCP: `${metrics.fcp.toFixed(2)}ms`,
          LCP: `${metrics.lcp.toFixed(2)}ms`,
          CLS: metrics.cls.toFixed(3),
          TTI: `${metrics.timeToInteractive?.toFixed(2)}ms`,
          Resources: metrics.resourceTiming
        })
      })
    }
  })

  test.describe('Page Load Performance', () => {
    test('should load dashboard efficiently with heavy data', async () => {
      await loginUser(page)

      // Measure dashboard load with metrics
      const startTime = Date.now()
      await page.goto(`${BASE_URL}/dashboard`)

      // Wait for all critical elements
      await page.waitForSelector('.metrics-grid')
      await page.waitForSelector('.chart-container')

      // Wait for charts to render
      await page.waitForSelector('canvas, .chart', { timeout: 15000 })

      const endTime = Date.now()
      const totalLoadTime = endTime - startTime

      // Collect detailed metrics
      const metrics = await collectPerformanceMetrics(page)

      // Performance assertions
      expect(totalLoadTime).toBeLessThan(5000) // Total load should be under 5 seconds
      expect(metrics.fullPageLoad).toBeLessThan(4000) // Page load should be under 4 seconds
      expect(metrics.resourceTiming).toBeLessThan(50) // Reasonable number of resources

      console.log('Dashboard load performance:', {
        totalLoadTime: `${totalLoadTime}ms`,
        fullPageLoad: `${metrics.fullPageLoad}ms`,
        resourceCount: metrics.resourceTiming
      })
    })

    test('should handle large dataset loading efficiently', async () => {
      await loginUser(page)
      await page.goto(`${BASE_URL}/datasets`)

      // Simulate loading large dataset
      await page.route('**/api/v1/datasets**', route => {
        // Add artificial delay to simulate large dataset
        setTimeout(() => {
          route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify({
              datasets: Array.from({ length: 100 }, (_, i) => ({
                id: i + 1,
                name: `Dataset ${i + 1}`,
                size: Math.floor(Math.random() * 1000000),
                created_at: new Date().toISOString()
              }))
            })
          })
        }, 500)
      })

      const startTime = Date.now()
      await page.reload()

      // Wait for table to load
      await page.waitForSelector('.data-table tbody tr', { timeout: 10000 })

      const endTime = Date.now()
      const loadTime = endTime - startTime

      // Check that virtual scrolling or pagination is working
      const visibleRows = await page.locator('.data-table tbody tr').count()
      expect(visibleRows).toBeLessThanOrEqual(50) // Should not render all 100 rows at once

      expect(loadTime).toBeLessThan(3000) // Should load within 3 seconds

      console.log('Large dataset load performance:', {
        loadTime: `${loadTime}ms`,
        visibleRows: visibleRows
      })
    })

    test('should handle concurrent API requests efficiently', async () => {
      await loginUser(page)

      // Track API requests
      const apiRequests: any[] = []
      page.on('request', request => {
        if (request.url().includes('/api/')) {
          apiRequests.push({
            url: request.url(),
            startTime: Date.now()
          })
        }
      })

      page.on('response', response => {
        if (response.url().includes('/api/')) {
          const request = apiRequests.find(req => req.url === response.url())
          if (request) {
            request.duration = Date.now() - request.startTime
            request.status = response.status()
          }
        }
      })

      // Navigate to page that makes multiple API calls
      await page.goto(`${BASE_URL}/dashboard`)
      await page.waitForLoadState('networkidle')

      // Analyze API performance
      const completedRequests = apiRequests.filter(req => req.duration)
      const averageRequestTime = completedRequests.reduce((sum, req) => sum + req.duration, 0) / completedRequests.length
      const maxRequestTime = Math.max(...completedRequests.map(req => req.duration))

      expect(averageRequestTime).toBeLessThan(1000) // Average API response under 1 second
      expect(maxRequestTime).toBeLessThan(3000) // No single request over 3 seconds
      expect(completedRequests.length).toBeGreaterThan(0) // Ensure we captured requests

      console.log('API performance:', {
        totalRequests: completedRequests.length,
        averageTime: `${averageRequestTime.toFixed(2)}ms`,
        maxTime: `${maxRequestTime}ms`
      })
    })
  })

  test.describe('Memory Usage Testing', () => {
    test('should maintain reasonable memory usage', async () => {
      await loginUser(page)

      // Navigate through multiple pages to test memory usage
      const pages = ['/dashboard', '/datasets', '/detectors', '/detection']

      for (const pagePath of pages) {
        await page.goto(`${BASE_URL}${pagePath}`)
        await page.waitForLoadState('networkidle')

        const memoryUsage = await page.evaluate(() => {
          if ('memory' in performance) {
            return {
              used: (performance as any).memory.usedJSHeapSize,
              total: (performance as any).memory.totalJSHeapSize,
              limit: (performance as any).memory.jsHeapSizeLimit
            }
          }
          return null
        })

        if (memoryUsage) {
          const memoryUsageMB = memoryUsage.used / (1024 * 1024)
          expect(memoryUsageMB).toBeLessThan(50) // Should use less than 50MB

          console.log(`Memory usage on ${pagePath}:`, {
            used: `${memoryUsageMB.toFixed(2)}MB`,
            total: `${(memoryUsage.total / (1024 * 1024)).toFixed(2)}MB`
          })
        }
      }
    })

    test('should handle memory cleanup on page navigation', async () => {
      await loginUser(page)

      // Get initial memory
      await page.goto(`${BASE_URL}/dashboard`)
      await page.waitForLoadState('networkidle')

      const initialMemory = await page.evaluate(() => {
        return 'memory' in performance ? (performance as any).memory.usedJSHeapSize : 0
      })

      // Navigate through heavy pages multiple times
      for (let i = 0; i < 3; i++) {
        await page.goto(`${BASE_URL}/datasets`)
        await page.waitForLoadState('networkidle')

        await page.goto(`${BASE_URL}/detectors`)
        await page.waitForLoadState('networkidle')

        await page.goto(`${BASE_URL}/dashboard`)
        await page.waitForLoadState('networkidle')
      }

      // Force garbage collection if available
      await page.evaluate(() => {
        if ('gc' in window) {
          (window as any).gc()
        }
      })

      const finalMemory = await page.evaluate(() => {
        return 'memory' in performance ? (performance as any).memory.usedJSHeapSize : 0
      })

      if (initialMemory > 0 && finalMemory > 0) {
        const memoryIncrease = (finalMemory - initialMemory) / initialMemory
        expect(memoryIncrease).toBeLessThan(0.5) // Memory should not increase by more than 50%

        console.log('Memory cleanup test:', {
          initial: `${(initialMemory / (1024 * 1024)).toFixed(2)}MB`,
          final: `${(finalMemory / (1024 * 1024)).toFixed(2)}MB`,
          increase: `${(memoryIncrease * 100).toFixed(2)}%`
        })
      }
    })
  })

  test.describe('Interactive Performance', () => {
    test('should respond quickly to user interactions', async () => {
      await loginUser(page)
      await page.goto(`${BASE_URL}/dashboard`)

      // Test button click responsiveness
      const buttons = page.locator('button, .btn')
      const buttonCount = await buttons.count()

      for (let i = 0; i < Math.min(buttonCount, 5); i++) {
        const button = buttons.nth(i)

        if (await button.isVisible() && await button.isEnabled()) {
          const startTime = Date.now()
          await button.click()

          // Wait for any visual feedback or action to complete
          await page.waitForTimeout(100)

          const responseTime = Date.now() - startTime
          expect(responseTime).toBeLessThan(100) // Should respond within 100ms
        }
      }
    })

    test('should handle form interactions efficiently', async () => {
      await loginUser(page)
      await page.goto(`${BASE_URL}/detectors`)

      // Open create detector form
      await page.click('.create-detector-btn')
      await page.waitForSelector('.detector-form, #detector-modal')

      // Test form field responsiveness
      const nameInput = page.locator('input[name="name"]')

      const startTime = Date.now()

      // Type in input field
      await nameInput.fill('Performance Test Detector')

      // Select algorithm
      await page.selectOption('select[name="algorithm"]', 'IsolationForest')

      // Fill description
      await page.fill('textarea[name="description"]', 'A detector created for performance testing')

      const formFillTime = Date.now() - startTime

      expect(formFillTime).toBeLessThan(2000) // Form should be responsive

      // Test form validation responsiveness
      await nameInput.clear()
      const validationStartTime = Date.now()

      await page.click('.submit-detector')
      await page.waitForSelector('.validation-error, .error-message', { timeout: 1000 })

      const validationTime = Date.now() - validationStartTime
      expect(validationTime).toBeLessThan(500) // Validation should be immediate
    })

    test('should handle table sorting and filtering efficiently', async () => {
      await loginUser(page)
      await page.goto(`${BASE_URL}/datasets`)

      // Test table sorting
      const sortHeader = page.locator('th[data-sortable="true"], th.sortable').first()

      if (await sortHeader.count() > 0) {
        const sortStartTime = Date.now()
        await sortHeader.click()

        // Wait for sort to complete
        await page.waitForTimeout(500)

        const sortTime = Date.now() - sortStartTime
        expect(sortTime).toBeLessThan(1000) // Sorting should be fast
      }

      // Test search/filter
      const searchInput = page.locator('input[name="search"], input[type="search"]')

      if (await searchInput.count() > 0) {
        const searchStartTime = Date.now()
        await searchInput.fill('test')

        // Wait for search results
        await page.waitForTimeout(500)

        const searchTime = Date.now() - searchStartTime
        expect(searchTime).toBeLessThan(1000) // Search should be responsive
      }
    })
  })

  test.describe('Chart and Visualization Performance', () => {
    test('should render charts efficiently', async () => {
      await loginUser(page)
      await page.goto(`${BASE_URL}/dashboard`)

      // Wait for charts to start rendering
      await page.waitForSelector('canvas, .chart', { timeout: 10000 })

      const chartRenderStart = Date.now()

      // Wait for all charts to be fully rendered
      await page.waitForFunction(() => {
        const canvases = document.querySelectorAll('canvas')
        return Array.from(canvases).every(canvas => {
          const ctx = canvas.getContext('2d')
          return ctx && canvas.width > 0 && canvas.height > 0
        })
      }, { timeout: 15000 })

      const chartRenderTime = Date.now() - chartRenderStart

      expect(chartRenderTime).toBeLessThan(5000) // Charts should render within 5 seconds

      // Test chart interaction performance
      const firstChart = page.locator('canvas').first()

      if (await firstChart.count() > 0) {
        const interactionStart = Date.now()

        // Hover over chart
        await firstChart.hover()

        // Wait for any tooltip or interaction feedback
        await page.waitForTimeout(100)

        const interactionTime = Date.now() - interactionStart
        expect(interactionTime).toBeLessThan(200) // Chart interactions should be responsive
      }

      console.log('Chart performance:', {
        renderTime: `${chartRenderTime}ms`,
        chartCount: await page.locator('canvas').count()
      })
    })

    test('should handle large dataset visualization efficiently', async () => {
      await loginUser(page)
      await page.goto(`${BASE_URL}/detection`)

      // Mock large dataset for visualization
      await page.route('**/api/v1/detection/results', route => {
        const largeDataset = {
          anomalies: Array.from({ length: 1000 }, (_, i) => ({
            id: i,
            score: Math.random(),
            timestamp: new Date(Date.now() - i * 60000).toISOString()
          })),
          visualization_data: {
            scatter_plot: {
              x: Array.from({ length: 1000 }, () => Math.random() * 100),
              y: Array.from({ length: 1000 }, () => Math.random() * 100)
            }
          }
        }

        route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify(largeDataset)
        })
      })

      // Select detector and dataset
      await page.selectOption('select[name="detector"]', { index: 0 })
      await page.selectOption('select[name="dataset"]', { index: 0 })

      const detectionStart = Date.now()
      await page.click('.run-detection')

      // Wait for visualization to render
      await page.waitForSelector('.results-visualization', { timeout: 30000 })
      await page.waitForSelector('canvas', { timeout: 15000 })

      const detectionTime = Date.now() - detectionStart

      expect(detectionTime).toBeLessThan(15000) // Large dataset visualization should complete within 15 seconds

      console.log('Large dataset visualization performance:', {
        detectionTime: `${detectionTime}ms`,
        dataPoints: 1000
      })
    })
  })

  test.describe('Bundle Size and Asset Performance', () => {
    test('should have reasonable bundle sizes', async () => {
      const resourceSizes: any[] = []

      // Track resource loading
      page.on('response', async response => {
        if (response.url().includes('.js') || response.url().includes('.css')) {
          try {
            const buffer = await response.body()
            resourceSizes.push({
              url: response.url(),
              size: buffer.length,
              type: response.url().includes('.js') ? 'javascript' : 'css'
            })
          } catch (error) {
            // Ignore errors for resources we can't access
          }
        }
      })

      await page.goto(BASE_URL)
      await page.waitForLoadState('networkidle')

      // Analyze bundle sizes
      const jsSize = resourceSizes
        .filter(r => r.type === 'javascript')
        .reduce((sum, r) => sum + r.size, 0)

      const cssSize = resourceSizes
        .filter(r => r.type === 'css')
        .reduce((sum, r) => sum + r.size, 0)

      const totalSize = jsSize + cssSize

      // Bundle size thresholds
      expect(jsSize).toBeLessThan(1024 * 1024) // JS should be under 1MB
      expect(cssSize).toBeLessThan(256 * 1024) // CSS should be under 256KB
      expect(totalSize).toBeLessThan(1280 * 1024) // Total should be under 1.25MB

      console.log('Bundle size analysis:', {
        javascript: `${(jsSize / 1024).toFixed(2)}KB`,
        css: `${(cssSize / 1024).toFixed(2)}KB`,
        total: `${(totalSize / 1024).toFixed(2)}KB`,
        resourceCount: resourceSizes.length
      })
    })

    test('should cache static assets effectively', async () => {
      // First load
      const firstLoadStart = Date.now()
      await page.goto(BASE_URL)
      await page.waitForLoadState('networkidle')
      const firstLoadTime = Date.now() - firstLoadStart

      // Second load (should be cached)
      const secondLoadStart = Date.now()
      await page.reload()
      await page.waitForLoadState('networkidle')
      const secondLoadTime = Date.now() - secondLoadStart

      // Cached load should be significantly faster
      expect(secondLoadTime).toBeLessThan(firstLoadTime * 0.8) // At least 20% faster

      console.log('Caching performance:', {
        firstLoad: `${firstLoadTime}ms`,
        secondLoad: `${secondLoadTime}ms`,
        improvement: `${((firstLoadTime - secondLoadTime) / firstLoadTime * 100).toFixed(2)}%`
      })
    })
  })
})
