import { test, expect, Page } from '@playwright/test'

// Visual regression testing configuration
const BASE_URL = process.env.BASE_URL || 'http://localhost:8000'
const VIEWPORTS = [
  { width: 1920, height: 1080, name: 'desktop-large' },
  { width: 1366, height: 768, name: 'desktop-medium' },
  { width: 1024, height: 768, name: 'tablet-landscape' },
  { width: 768, height: 1024, name: 'tablet-portrait' },
  { width: 375, height: 667, name: 'mobile-small' },
  { width: 414, height: 896, name: 'mobile-large' }
]

const TEST_USER = {
  username: 'visual_test_user',
  password: 'VisualTest123!',
  email: 'visual@test.com'
}

// Pages and components to test
const PAGES_TO_TEST = [
  { path: '/', name: 'homepage', requiresAuth: false },
  { path: '/login', name: 'login-page', requiresAuth: false },
  { path: '/dashboard', name: 'dashboard', requiresAuth: true },
  { path: '/datasets', name: 'datasets-list', requiresAuth: true },
  { path: '/detectors', name: 'detectors-list', requiresAuth: true },
  { path: '/detection', name: 'detection-page', requiresAuth: true },
  { path: '/security-dashboard', name: 'security-dashboard', requiresAuth: true },
  { path: '/experiments', name: 'experiments-page', requiresAuth: true }
]

const COMPONENTS_TO_TEST = [
  { selector: '.metrics-grid', name: 'metrics-cards' },
  { selector: '.chart-container', name: 'chart-components' },
  { selector: '.data-table', name: 'data-tables' },
  { selector: '.security-alerts', name: 'security-alerts' },
  { selector: '.navigation-menu', name: 'navigation-menu' },
  { selector: '.modal-dialog', name: 'modal-dialogs' },
  { selector: '.form-container', name: 'form-components' },
  { selector: '.button-group', name: 'button-groups' }
]

test.describe('Visual Regression Testing', () => {
  let page: Page

  test.beforeEach(async ({ page: testPage }) => {
    page = testPage

    // Set consistent test conditions
    await page.addInitScript(() => {
      // Disable animations for consistent screenshots
      const style = document.createElement('style')
      style.textContent = `
        *, *::before, *::after {
          animation-duration: 0s !important;
          animation-delay: 0s !important;
          transition-duration: 0s !important;
          transition-delay: 0s !important;
        }
      `
      document.head.appendChild(style)

      // Mock Date for consistent timestamps
      const mockDate = new Date('2024-01-15T10:30:00Z')
      const OriginalDate = Date
      global.Date = class extends Date {
        constructor(...args: any[]) {
          if (args.length === 0) {
            super(mockDate)
          } else {
            super(...args)
          }
        }
        static now() {
          return mockDate.getTime()
        }
      } as any
    })

    // Wait for fonts to load
    await page.waitForLoadState('networkidle')
    await page.waitForFunction(() => document.fonts.ready)
  })

  // Helper function to login
  const loginUser = async (page: Page) => {
    await page.goto(`${BASE_URL}/login`)
    await page.fill('input[name="username"]', TEST_USER.username)
    await page.fill('input[name="password"]', TEST_USER.password)
    await page.click('button[type="submit"]')
    await page.waitForURL(/.*dashboard|.*\/$/)
  }

  // Helper function to wait for page stability
  const waitForPageStability = async (page: Page) => {
    // Wait for network idle
    await page.waitForLoadState('networkidle')

    // Wait for any charts or dynamic content to render
    await page.waitForTimeout(2000)

    // Wait for any lazy-loaded images
    await page.waitForFunction(() => {
      const images = Array.from(document.querySelectorAll('img'))
      return images.every(img => img.complete)
    })
  }

  // Test full page screenshots across different viewports
  for (const viewport of VIEWPORTS) {
    test.describe(`Viewport: ${viewport.name} (${viewport.width}x${viewport.height})`, () => {
      test.beforeEach(async ({ page }) => {
        await page.setViewportSize({ width: viewport.width, height: viewport.height })
      })

      for (const pageConfig of PAGES_TO_TEST) {
        test(`should match visual snapshot for ${pageConfig.name}`, async () => {
          if (pageConfig.requiresAuth) {
            await loginUser(page)
          }

          await page.goto(`${BASE_URL}${pageConfig.path}`)
          await waitForPageStability(page)

          // Hide dynamic content that changes between runs
          await page.addStyleTag({
            content: `
              .timestamp, .last-updated, .current-time,
              .real-time-indicator, .connection-status,
              .loading-spinner, .progress-bar {
                visibility: hidden !important;
              }
            `
          })

          // Take full page screenshot
          await expect(page).toHaveScreenshot(`${pageConfig.name}-${viewport.name}.png`, {
            fullPage: true,
            animations: 'disabled',
            mask: [
              // Mask elements that contain dynamic data
              page.locator('.timestamp'),
              page.locator('.last-updated'),
              page.locator('.real-time-data')
            ]
          })
        })
      }
    })
  }

  test.describe('Component Visual Testing', () => {
    test.beforeEach(async () => {
      await page.setViewportSize({ width: 1366, height: 768 })
      await loginUser(page)
    })

    for (const component of COMPONENTS_TO_TEST) {
      test(`should match visual snapshot for ${component.name}`, async () => {
        // Navigate to a page that contains this component
        let testPage = '/dashboard'

        if (component.name.includes('security')) {
          testPage = '/security-dashboard'
        } else if (component.name.includes('data-table')) {
          testPage = '/datasets'
        } else if (component.name.includes('form')) {
          testPage = '/detectors'
          await page.goto(`${BASE_URL}${testPage}`)
          await page.click('.create-detector-btn')
        }

        if (testPage !== '/detectors') {
          await page.goto(`${BASE_URL}${testPage}`)
        }

        await waitForPageStability(page)

        // Wait for component to be visible
        await page.waitForSelector(component.selector, { timeout: 10000 })

        // Take component screenshot
        const componentElement = page.locator(component.selector).first()
        await expect(componentElement).toHaveScreenshot(`${component.name}-component.png`, {
          animations: 'disabled'
        })
      })
    }
  })

  test.describe('Interactive State Testing', () => {
    test.beforeEach(async () => {
      await page.setViewportSize({ width: 1366, height: 768 })
      await loginUser(page)
    })

    test('should capture hover states', async () => {
      await page.goto(`${BASE_URL}/dashboard`)
      await waitForPageStability(page)

      // Test button hover states
      const buttons = page.locator('button, .btn')
      const buttonCount = await buttons.count()

      for (let i = 0; i < Math.min(buttonCount, 5); i++) {
        const button = buttons.nth(i)
        await button.hover()
        await page.waitForTimeout(500)

        await expect(button).toHaveScreenshot(`button-hover-${i}.png`)
      }
    })

    test('should capture form validation states', async () => {
      await page.goto(`${BASE_URL}/detectors`)
      await page.click('.create-detector-btn')

      // Submit empty form to trigger validation
      await page.click('.submit-detector')
      await page.waitForTimeout(1000)

      // Capture form with validation errors
      const form = page.locator('.detector-form, form')
      await expect(form).toHaveScreenshot('form-validation-errors.png')

      // Fill form partially and capture valid state
      await page.fill('input[name="name"]', 'Test Detector')
      await page.selectOption('select[name="algorithm"]', 'IsolationForest')

      await expect(form).toHaveScreenshot('form-partially-filled.png')
    })

    test('should capture modal states', async () => {
      await page.goto(`${BASE_URL}/datasets`)

      // Open upload modal
      await page.click('.upload-dataset-btn')
      await page.waitForSelector('.modal, .upload-form', { timeout: 5000 })

      const modal = page.locator('.modal, .upload-form')
      await expect(modal).toHaveScreenshot('upload-modal-open.png')

      // Fill modal form
      await page.fill('input[name="name"]', 'Test Dataset')
      await page.fill('textarea[name="description"]', 'A test dataset for visual regression testing')

      await expect(modal).toHaveScreenshot('upload-modal-filled.png')
    })

    test('should capture loading states', async () => {
      await page.goto(`${BASE_URL}/dashboard`)

      // Simulate slow network to capture loading states
      await page.route('**/api/**', route => {
        setTimeout(() => route.continue(), 2000)
      })

      // Trigger data refresh
      await page.click('.refresh-button, button:has-text("Refresh")')

      // Capture loading state
      await page.waitForSelector('.loading, .spinner', { timeout: 3000 })
      await expect(page.locator('.dashboard-container')).toHaveScreenshot('dashboard-loading.png')
    })
  })

  test.describe('Error State Testing', () => {
    test.beforeEach(async () => {
      await page.setViewportSize({ width: 1366, height: 768 })
      await loginUser(page)
    })

    test('should capture error message displays', async () => {
      // Mock API error responses
      await page.route('**/api/**', route => {
        route.fulfill({
          status: 500,
          contentType: 'application/json',
          body: JSON.stringify({ error: 'Internal server error' })
        })
      })

      await page.goto(`${BASE_URL}/dashboard`)
      await page.waitForTimeout(3000)

      // Capture error state
      await expect(page.locator('.error-message, .alert-error')).toHaveScreenshot('api-error-message.png')
    })

    test('should capture 404 page', async () => {
      await page.goto(`${BASE_URL}/nonexistent-page`)
      await waitForPageStability(page)

      await expect(page).toHaveScreenshot('404-page.png', {
        fullPage: true
      })
    })

    test('should capture network offline state', async () => {
      await page.goto(`${BASE_URL}/dashboard`)
      await waitForPageStability(page)

      // Simulate offline
      await page.context().setOffline(true)
      await page.waitForTimeout(2000)

      // Try to refresh data
      await page.click('.refresh-button', { timeout: 5000 }).catch(() => {})
      await page.waitForTimeout(2000)

      // Capture offline state
      await expect(page).toHaveScreenshot('offline-state.png')
    })
  })

  test.describe('Theme and Dark Mode Testing', () => {
    test.beforeEach(async () => {
      await page.setViewportSize({ width: 1366, height: 768 })
      await loginUser(page)
    })

    test('should capture dark mode if available', async () => {
      await page.goto(`${BASE_URL}/dashboard`)

      // Check if dark mode toggle exists
      const darkModeToggle = page.locator('.dark-mode-toggle, .theme-toggle')

      if (await darkModeToggle.count() > 0) {
        // Switch to dark mode
        await darkModeToggle.click()
        await page.waitForTimeout(1000)

        await expect(page).toHaveScreenshot('dashboard-dark-mode.png', {
          fullPage: true
        })

        // Test security dashboard in dark mode
        await page.goto(`${BASE_URL}/security-dashboard`)
        await waitForPageStability(page)

        await expect(page).toHaveScreenshot('security-dashboard-dark-mode.png', {
          fullPage: true
        })
      }
    })

    test('should capture high contrast mode', async () => {
      await page.goto(`${BASE_URL}/dashboard`)

      // Simulate high contrast mode
      await page.addStyleTag({
        content: `
          @media (prefers-contrast: high) {
            * {
              filter: contrast(150%) !important;
            }
          }
        `
      })

      await page.emulateMedia({ colorScheme: 'dark', reducedMotion: 'reduce' })
      await page.waitForTimeout(1000)

      await expect(page).toHaveScreenshot('dashboard-high-contrast.png', {
        fullPage: true
      })
    })
  })

  test.describe('Chart and Visualization Testing', () => {
    test.beforeEach(async () => {
      await page.setViewportSize({ width: 1366, height: 768 })
      await loginUser(page)
    })

    test('should capture chart rendering', async () => {
      await page.goto(`${BASE_URL}/dashboard`)

      // Wait for charts to fully render
      await page.waitForSelector('canvas, .chart', { timeout: 10000 })
      await page.waitForTimeout(3000)

      // Capture individual charts
      const charts = page.locator('canvas, .chart')
      const chartCount = await charts.count()

      for (let i = 0; i < chartCount; i++) {
        const chart = charts.nth(i)
        if (await chart.isVisible()) {
          await expect(chart).toHaveScreenshot(`chart-${i}.png`)
        }
      }
    })

    test('should capture visualization states', async () => {
      await page.goto(`${BASE_URL}/detection`)

      // Mock detection results with visualization data
      await page.route('**/api/v1/detection/results', route => {
        route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({
            anomalies: [
              { id: 1, score: 0.95, timestamp: '2024-01-15T10:30:00Z' },
              { id: 2, score: 0.87, timestamp: '2024-01-15T10:31:00Z' },
              { id: 3, score: 0.92, timestamp: '2024-01-15T10:32:00Z' }
            ],
            visualization_data: {
              scatter_plot: { x: [1, 2, 3], y: [0.95, 0.87, 0.92] },
              time_series: { timestamps: [], values: [] }
            }
          })
        })
      })

      // Trigger detection visualization
      await page.selectOption('select[name="detector"]', { index: 0 })
      await page.selectOption('select[name="dataset"]', { index: 0 })
      await page.click('.run-detection')

      await page.waitForSelector('.results-visualization', { timeout: 15000 })
      await page.waitForTimeout(3000)

      await expect(page.locator('.results-visualization')).toHaveScreenshot('detection-results-visualization.png')
    })
  })

  test.describe('Responsive Design Testing', () => {
    for (const viewport of VIEWPORTS) {
      test(`should display responsive layout correctly on ${viewport.name}`, async () => {
        await page.setViewportSize({ width: viewport.width, height: viewport.height })
        await loginUser(page)

        await page.goto(`${BASE_URL}/dashboard`)
        await waitForPageStability(page)

        // Test navigation menu responsiveness
        const navMenu = page.locator('.navigation-menu, .navbar')
        if (await navMenu.count() > 0) {
          await expect(navMenu).toHaveScreenshot(`navigation-${viewport.name}.png`)
        }

        // Test metrics grid responsiveness
        const metricsGrid = page.locator('.metrics-grid')
        if (await metricsGrid.count() > 0) {
          await expect(metricsGrid).toHaveScreenshot(`metrics-grid-${viewport.name}.png`)
        }

        // Test table responsiveness on mobile
        if (viewport.width < 768) {
          await page.goto(`${BASE_URL}/datasets`)
          await waitForPageStability(page)

          const dataTable = page.locator('.data-table, table')
          if (await dataTable.count() > 0) {
            await expect(dataTable).toHaveScreenshot(`data-table-mobile-${viewport.name}.png`)
          }
        }
      })
    }
  })
})
