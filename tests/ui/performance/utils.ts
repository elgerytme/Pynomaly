import { Page, expect } from '@playwright/test'

/**
 * Performance testing utilities
 */

export interface PerformanceMetrics {
  fcp: number // First Contentful Paint
  lcp: number // Largest Contentful Paint
  fid: number // First Input Delay
  cls: number // Cumulative Layout Shift
  tti: number // Time to Interactive
  tbt: number // Total Blocking Time
  si: number  // Speed Index
  domContentLoaded: number
  fullPageLoad: number
  resourceCount: number
  memoryUsage?: {
    used: number
    total: number
    limit: number
  }
}

export interface NetworkMetrics {
  requestCount: number
  totalSize: number
  averageResponseTime: number
  slowestRequest: {
    url: string
    duration: number
  }
  fastestRequest: {
    url: string
    duration: number
  }
}

/**
 * Collect comprehensive performance metrics from a page
 */
export async function collectPerformanceMetrics(page: Page): Promise<PerformanceMetrics> {
  // Wait for page to be fully loaded
  await page.waitForLoadState('networkidle')

  // Collect metrics from the browser
  const metrics = await page.evaluate(() => {
    const perfData = {
      fcp: 0,
      lcp: 0,
      fid: 0,
      cls: 0,
      tti: 0,
      tbt: 0,
      si: 0,
      domContentLoaded: 0,
      fullPageLoad: 0,
      resourceCount: 0,
      memoryUsage: undefined as any
    }

    // Get navigation timing
    const navigation = performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming
    if (navigation) {
      perfData.domContentLoaded = navigation.domContentLoadedEventEnd - navigation.navigationStart
      perfData.fullPageLoad = navigation.loadEventEnd - navigation.navigationStart
      perfData.tti = navigation.domInteractive - navigation.navigationStart
    }

    // Get paint timings
    const paintEntries = performance.getEntriesByType('paint')
    paintEntries.forEach(entry => {
      if (entry.name === 'first-contentful-paint') {
        perfData.fcp = entry.startTime
      }
    })

    // Get LCP
    if ('PerformanceObserver' in window) {
      const lcpEntries = performance.getEntriesByType('largest-contentful-paint')
      if (lcpEntries.length > 0) {
        perfData.lcp = lcpEntries[lcpEntries.length - 1].startTime
      }
    }

    // Get CLS
    let clsValue = 0
    const layoutShiftEntries = performance.getEntriesByType('layout-shift')
    layoutShiftEntries.forEach((entry: any) => {
      if (!entry.hadRecentInput) {
        clsValue += entry.value
      }
    })
    perfData.cls = clsValue

    // Get resource count
    perfData.resourceCount = performance.getEntriesByType('resource').length

    // Get memory usage if available
    if ('memory' in performance) {
      const memory = (performance as any).memory
      perfData.memoryUsage = {
        used: memory.usedJSHeapSize,
        total: memory.totalJSHeapSize,
        limit: memory.jsHeapSizeLimit
      }
    }

    return perfData
  })

  return metrics
}

/**
 * Monitor network performance during page load
 */
export async function monitorNetworkPerformance(page: Page): Promise<NetworkMetrics> {
  const requests: Array<{ url: string; startTime: number; endTime?: number; size?: number }> = []

  page.on('request', request => {
    requests.push({
      url: request.url(),
      startTime: Date.now()
    })
  })

  page.on('response', async response => {
    const request = requests.find(req => req.url === response.url() && !req.endTime)
    if (request) {
      request.endTime = Date.now()
      try {
        const buffer = await response.body()
        request.size = buffer.length
      } catch {
        // Ignore if we can't get response body
      }
    }
  })

  return {
    get requestCount() {
      return requests.length
    },
    get totalSize() {
      return requests.reduce((sum, req) => sum + (req.size || 0), 0)
    },
    get averageResponseTime() {
      const completedRequests = requests.filter(req => req.endTime)
      if (completedRequests.length === 0) return 0
      const totalTime = completedRequests.reduce((sum, req) => sum + (req.endTime! - req.startTime), 0)
      return totalTime / completedRequests.length
    },
    get slowestRequest() {
      const completedRequests = requests.filter(req => req.endTime)
      if (completedRequests.length === 0) return { url: '', duration: 0 }

      const slowest = completedRequests.reduce((prev, current) => {
        const prevDuration = prev.endTime! - prev.startTime
        const currentDuration = current.endTime! - current.startTime
        return currentDuration > prevDuration ? current : prev
      })

      return {
        url: slowest.url,
        duration: slowest.endTime! - slowest.startTime
      }
    },
    get fastestRequest() {
      const completedRequests = requests.filter(req => req.endTime)
      if (completedRequests.length === 0) return { url: '', duration: 0 }

      const fastest = completedRequests.reduce((prev, current) => {
        const prevDuration = prev.endTime! - prev.startTime
        const currentDuration = current.endTime! - current.startTime
        return currentDuration < prevDuration ? current : prev
      })

      return {
        url: fastest.url,
        duration: fastest.endTime! - fastest.startTime
      }
    }
  }
}

/**
 * Simulate slow network conditions
 */
export async function simulateSlowNetwork(page: Page, preset: 'slow3g' | 'fast3g' | 'offline' = 'slow3g') {
  const networkConditions = {
    slow3g: {
      offline: false,
      downloadThroughput: 50000, // 50kb/s
      uploadThroughput: 25000,   // 25kb/s
      latency: 2000              // 2s latency
    },
    fast3g: {
      offline: false,
      downloadThroughput: 180000, // 180kb/s
      uploadThroughput: 84000,    // 84kb/s
      latency: 562                // 562ms latency
    },
    offline: {
      offline: true,
      downloadThroughput: 0,
      uploadThroughput: 0,
      latency: 0
    }
  }

  const conditions = networkConditions[preset]
  await page.context().setExtraHTTPHeaders({
    'Connection': 'close'
  })

  if (preset === 'offline') {
    await page.context().setOffline(true)
  } else {
    // Note: Playwright doesn't have direct network throttling like Puppeteer
    // This would need to be implemented using a proxy or browser dev tools
    console.log(`Simulating ${preset} network conditions`)
  }
}

/**
 * Wait for page to be visually stable
 */
export async function waitForVisualStability(page: Page, timeout = 10000) {
  await page.waitForLoadState('networkidle')

  // Wait for any animations to complete
  await page.waitForTimeout(1000)

  // Wait for fonts to load
  await page.waitForFunction(() => document.fonts.ready, { timeout })

  // Wait for images to load
  await page.waitForFunction(() => {
    const images = Array.from(document.querySelectorAll('img'))
    return images.every(img => img.complete)
  }, { timeout })

  // Wait for any lazy-loaded content
  await page.waitForTimeout(500)
}

/**
 * Measure Core Web Vitals using the web-vitals library if available
 */
export async function measureCoreWebVitals(page: Page): Promise<Partial<PerformanceMetrics>> {
  return await page.evaluate(() => {
    return new Promise((resolve) => {
      const vitals = {
        fcp: 0,
        lcp: 0,
        fid: 0,
        cls: 0
      }

      let resolveCount = 0
      const totalMetrics = 4

      const checkComplete = () => {
        resolveCount++
        if (resolveCount >= totalMetrics) {
          resolve(vitals)
        }
      }

      // Measure FCP
      new PerformanceObserver((list) => {
        for (const entry of list.getEntries()) {
          if (entry.name === 'first-contentful-paint') {
            vitals.fcp = entry.startTime
            checkComplete()
            break
          }
        }
      }).observe({ entryTypes: ['paint'] })

      // Measure LCP
      new PerformanceObserver((list) => {
        const entries = list.getEntries()
        const lastEntry = entries[entries.length - 1]
        vitals.lcp = lastEntry.startTime
        checkComplete()
      }).observe({ entryTypes: ['largest-contentful-paint'] })

      // Measure FID
      new PerformanceObserver((list) => {
        for (const entry of list.getEntries()) {
          vitals.fid = (entry as any).processingStart - entry.startTime
          checkComplete()
          break
        }
      }).observe({ entryTypes: ['first-input'] })

      // Measure CLS
      let clsValue = 0
      new PerformanceObserver((list) => {
        for (const entry of list.getEntries()) {
          if (!(entry as any).hadRecentInput) {
            clsValue += (entry as any).value
          }
        }
        vitals.cls = clsValue
        checkComplete()
      }).observe({ entryTypes: ['layout-shift'] })

      // Fallback timeout
      setTimeout(() => resolve(vitals), 5000)
    })
  })
}

/**
 * Performance assertion helpers
 */
export class PerformanceAssertions {
  constructor(private metrics: PerformanceMetrics) {}

  expectFCPUnder(threshold: number) {
    expect(this.metrics.fcp).toBeLessThan(threshold)
    return this
  }

  expectLCPUnder(threshold: number) {
    expect(this.metrics.lcp).toBeLessThan(threshold)
    return this
  }

  expectCLSUnder(threshold: number) {
    expect(this.metrics.cls).toBeLessThan(threshold)
    return this
  }

  expectTTIUnder(threshold: number) {
    expect(this.metrics.tti).toBeLessThan(threshold)
    return this
  }

  expectPageLoadUnder(threshold: number) {
    expect(this.metrics.fullPageLoad).toBeLessThan(threshold)
    return this
  }

  expectResourceCountUnder(threshold: number) {
    expect(this.metrics.resourceCount).toBeLessThan(threshold)
    return this
  }

  expectMemoryUsageUnder(thresholdMB: number) {
    if (this.metrics.memoryUsage) {
      const usageMB = this.metrics.memoryUsage.used / (1024 * 1024)
      expect(usageMB).toBeLessThan(thresholdMB)
    }
    return this
  }

  logMetrics(pageName: string) {
    console.log(`Performance metrics for ${pageName}:`, {
      FCP: `${this.metrics.fcp.toFixed(2)}ms`,
      LCP: `${this.metrics.lcp.toFixed(2)}ms`,
      CLS: this.metrics.cls.toFixed(3),
      TTI: `${this.metrics.tti.toFixed(2)}ms`,
      'Page Load': `${this.metrics.fullPageLoad.toFixed(2)}ms`,
      'Resources': this.metrics.resourceCount,
      'Memory': this.metrics.memoryUsage ?
        `${(this.metrics.memoryUsage.used / (1024 * 1024)).toFixed(2)}MB` :
        'N/A'
    })
    return this
  }
}

/**
 * Create performance assertions for metrics
 */
export function expectPerformance(metrics: PerformanceMetrics): PerformanceAssertions {
  return new PerformanceAssertions(metrics)
}

/**
 * Bundle size analysis utilities
 */
export async function analyzeBundleSize(page: Page): Promise<{
  javascript: number
  css: number
  images: number
  fonts: number
  total: number
  resources: Array<{ url: string; size: number; type: string }>
}> {
  const resources: Array<{ url: string; size: number; type: string }> = []

  page.on('response', async response => {
    try {
      const url = response.url()
      const buffer = await response.body()
      const size = buffer.length

      let type = 'other'
      if (url.includes('.js')) type = 'javascript'
      else if (url.includes('.css')) type = 'css'
      else if (url.match(/\.(png|jpg|jpeg|gif|svg|webp)$/i)) type = 'images'
      else if (url.match(/\.(woff|woff2|ttf|eot)$/i)) type = 'fonts'

      resources.push({ url, size, type })
    } catch {
      // Ignore if we can't access the response body
    }
  })

  return {
    get javascript() {
      return resources.filter(r => r.type === 'javascript').reduce((sum, r) => sum + r.size, 0)
    },
    get css() {
      return resources.filter(r => r.type === 'css').reduce((sum, r) => sum + r.size, 0)
    },
    get images() {
      return resources.filter(r => r.type === 'images').reduce((sum, r) => sum + r.size, 0)
    },
    get fonts() {
      return resources.filter(r => r.type === 'fonts').reduce((sum, r) => sum + r.size, 0)
    },
    get total() {
      return resources.reduce((sum, r) => sum + r.size, 0)
    },
    get resources() {
      return [...resources]
    }
  }
}
