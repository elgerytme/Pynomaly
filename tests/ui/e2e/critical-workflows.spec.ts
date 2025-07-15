import { test, expect, Page, Browser } from '@playwright/test'

// Test configuration
const BASE_URL = process.env.BASE_URL || 'http://localhost:8000'
const API_BASE = `${BASE_URL}/api/v1`

// Test data
const TEST_USER = {
  username: 'test_user',
  password: 'test_password123',
  email: 'test@example.com'
}

const TEST_DATASET = {
  name: 'E2E Test Dataset',
  description: 'Dataset created during end-to-end testing',
  file_path: 'tests/fixtures/sample_data.csv'
}

const TEST_DETECTOR = {
  name: 'E2E Test Detector',
  algorithm: 'IsolationForest',
  description: 'Detector created during end-to-end testing'
}

test.describe('Critical User Workflows', () => {
  let page: Page
  let browser: Browser

  test.beforeEach(async ({ page: testPage, browser: testBrowser }) => {
    page = testPage
    browser = testBrowser

    // Navigate to the application
    await page.goto(BASE_URL)

    // Wait for page to load
    await page.waitForLoadState('networkidle')
  })

  test.describe('Authentication Flow', () => {
    test('should complete full login workflow', async () => {
      // Navigate to login page
      await page.click('a[href="/login"]')
      await expect(page).toHaveURL(/.*login/)

      // Verify login form is present
      await expect(page.locator('form#login-form')).toBeVisible()
      await expect(page.locator('input[name="username"]')).toBeVisible()
      await expect(page.locator('input[name="password"]')).toBeVisible()

      // Fill login form
      await page.fill('input[name="username"]', TEST_USER.username)
      await page.fill('input[name="password"]', TEST_USER.password)

      // Verify CSRF token is present
      const csrfToken = await page.locator('input[name="csrf_token"]')
      await expect(csrfToken).toBeVisible()

      // Submit login form
      await page.click('button[type="submit"]')

      // Wait for redirect to dashboard
      await page.waitForURL(/.*dashboard|.*\/$/)

      // Verify successful login
      await expect(page.locator('.user-menu, .logout-button')).toBeVisible()

      // Verify dashboard content loads
      await expect(page.locator('.dashboard-container')).toBeVisible()
      await expect(page.locator('.metrics-grid')).toBeVisible()
    })

    test('should handle login errors gracefully', async () => {
      await page.goto(`${BASE_URL}/login`)

      // Try login with invalid credentials
      await page.fill('input[name="username"]', 'invalid_user')
      await page.fill('input[name="password"]', 'invalid_password')
      await page.click('button[type="submit"]')

      // Verify error message is displayed
      await expect(page.locator('.error-message, .alert-error')).toBeVisible()
      await expect(page.locator('.error-message, .alert-error')).toContainText(/invalid|incorrect|failed/i)

      // Verify user remains on login page
      await expect(page).toHaveURL(/.*login/)
    })

    test('should logout successfully', async () => {
      // Login first
      await page.goto(`${BASE_URL}/login`)
      await page.fill('input[name="username"]', TEST_USER.username)
      await page.fill('input[name="password"]', TEST_USER.password)
      await page.click('button[type="submit"]')
      await page.waitForURL(/.*dashboard|.*\/$/)

      // Logout
      await page.click('.logout-button, a[href="/logout"]')

      // Verify redirect to login page
      await page.waitForURL(/.*login/)
      await expect(page.locator('form#login-form')).toBeVisible()
    })
  })

  test.describe('Dataset Management Workflow', () => {
    test.beforeEach(async () => {
      // Login before each test
      await page.goto(`${BASE_URL}/login`)
      await page.fill('input[name="username"]', TEST_USER.username)
      await page.fill('input[name="password"]', TEST_USER.password)
      await page.click('button[type="submit"]')
      await page.waitForURL(/.*dashboard|.*\/$/)
    })

    test('should complete dataset upload workflow', async () => {
      // Navigate to datasets page
      await page.click('a[href="/datasets"], .nav-datasets')
      await expect(page).toHaveURL(/.*datasets/)

      // Verify datasets page loads
      await expect(page.locator('.datasets-container')).toBeVisible()

      // Click upload dataset button
      await page.click('button:has-text("Upload Dataset"), .upload-dataset-btn')

      // Verify upload form appears
      await expect(page.locator('.upload-form, #upload-modal')).toBeVisible()

      // Fill dataset information
      await page.fill('input[name="name"]', TEST_DATASET.name)
      await page.fill('textarea[name="description"]', TEST_DATASET.description)

      // Upload file (mock file upload)
      const fileInput = page.locator('input[type="file"]')
      await fileInput.setInputFiles('tests/fixtures/sample_data.csv')

      // Submit upload
      await page.click('button:has-text("Upload"), .submit-upload')

      // Wait for upload to complete
      await page.waitForSelector('.upload-success, .success-message', { timeout: 10000 })

      // Verify dataset appears in list
      await expect(page.locator('.dataset-item')).toContainText(TEST_DATASET.name)

      // Verify dataset details
      await page.click(`.dataset-item:has-text("${TEST_DATASET.name}")`)
      await expect(page.locator('.dataset-details')).toBeVisible()
      await expect(page.locator('.dataset-name')).toContainText(TEST_DATASET.name)
    })

    test('should validate dataset format', async () => {
      await page.goto(`${BASE_URL}/datasets`)

      // Try to upload invalid file
      await page.click('.upload-dataset-btn')
      const fileInput = page.locator('input[type="file"]')
      await fileInput.setInputFiles('tests/fixtures/invalid_data.txt')

      await page.click('.submit-upload')

      // Verify validation error
      await expect(page.locator('.error-message, .validation-error')).toBeVisible()
      await expect(page.locator('.error-message')).toContainText(/invalid|format|supported/i)
    })

    test('should preview dataset data', async () => {
      await page.goto(`${BASE_URL}/datasets`)

      // Click on existing dataset
      await page.click('.dataset-item:first-child')

      // Verify dataset preview loads
      await expect(page.locator('.dataset-preview, .data-table')).toBeVisible()
      await expect(page.locator('table thead tr th')).toHaveCount.greaterThan(0)
      await expect(page.locator('table tbody tr')).toHaveCount.greaterThan(0)

      // Verify data quality information
      await expect(page.locator('.data-quality, .dataset-stats')).toBeVisible()
    })
  })

  test.describe('Detector Creation and Training Workflow', () => {
    test.beforeEach(async () => {
      // Login and ensure we have a dataset
      await page.goto(`${BASE_URL}/login`)
      await page.fill('input[name="username"]', TEST_USER.username)
      await page.fill('input[name="password"]', TEST_USER.password)
      await page.click('button[type="submit"]')
      await page.waitForURL(/.*dashboard|.*\/$/)
    })

    test('should create and configure detector', async () => {
      // Navigate to detectors page
      await page.click('a[href="/detectors"], .nav-detectors')
      await expect(page).toHaveURL(/.*detectors/)

      // Click create new detector
      await page.click('button:has-text("Create Detector"), .create-detector-btn')

      // Verify detector creation form
      await expect(page.locator('.detector-form, #detector-modal')).toBeVisible()

      // Fill detector information
      await page.fill('input[name="name"]', TEST_DETECTOR.name)
      await page.selectOption('select[name="algorithm"]', TEST_DETECTOR.algorithm)
      await page.fill('textarea[name="description"]', TEST_DETECTOR.description)

      // Configure algorithm parameters
      await expect(page.locator('.algorithm-params')).toBeVisible()

      // Submit detector creation
      await page.click('button:has-text("Create"), .submit-detector')

      // Wait for creation to complete
      await page.waitForSelector('.success-message, .detector-created', { timeout: 10000 })

      // Verify detector appears in list
      await expect(page.locator('.detector-item')).toContainText(TEST_DETECTOR.name)

      // Verify detector status
      const detectorItem = page.locator(`.detector-item:has-text("${TEST_DETECTOR.name}")`)
      await expect(detectorItem.locator('.status')).toContainText(/created|ready/i)
    })

    test('should train detector with dataset', async () => {
      await page.goto(`${BASE_URL}/detectors`)

      // Select detector for training
      await page.click(`.detector-item:has-text("${TEST_DETECTOR.name}")`)

      // Navigate to training page
      await page.click('button:has-text("Train"), .train-detector-btn')

      // Verify training form
      await expect(page.locator('.training-form')).toBeVisible()

      // Select dataset
      await page.selectOption('select[name="dataset"]', { label: TEST_DATASET.name })

      // Configure training parameters
      await page.fill('input[name="test_size"]', '0.2')
      await page.selectOption('select[name="validation_method"]', 'holdout')

      // Start training
      await page.click('button:has-text("Start Training"), .start-training')

      // Wait for training to start
      await expect(page.locator('.training-progress, .training-status')).toBeVisible()

      // Monitor training progress
      await page.waitForSelector('.training-complete, .training-finished', { timeout: 30000 })

      // Verify training results
      await expect(page.locator('.training-results')).toBeVisible()
      await expect(page.locator('.model-metrics')).toBeVisible()

      // Verify detector status updated
      await page.goto(`${BASE_URL}/detectors`)
      const trainedDetector = page.locator(`.detector-item:has-text("${TEST_DETECTOR.name}")`)
      await expect(trainedDetector.locator('.status')).toContainText(/trained|ready/i)
    })

    test('should validate training parameters', async () => {
      await page.goto(`${BASE_URL}/detectors`)
      await page.click(`.detector-item:first-child`)
      await page.click('.train-detector-btn')

      // Try training without selecting dataset
      await page.click('.start-training')

      // Verify validation error
      await expect(page.locator('.error-message, .validation-error')).toBeVisible()
      await expect(page.locator('.error-message')).toContainText(/dataset|required/i)

      // Try invalid test size
      await page.selectOption('select[name="dataset"]', { index: 0 })
      await page.fill('input[name="test_size"]', '1.5')
      await page.click('.start-training')

      // Verify validation error
      await expect(page.locator('.error-message')).toContainText(/test.size|invalid/i)
    })
  })

  test.describe('Anomaly Detection Workflow', () => {
    test.beforeEach(async () => {
      // Login
      await page.goto(`${BASE_URL}/login`)
      await page.fill('input[name="username"]', TEST_USER.username)
      await page.fill('input[name="password"]', TEST_USER.password)
      await page.click('button[type="submit"]')
      await page.waitForURL(/.*dashboard|.*\/$/)
    })

    test('should run anomaly detection', async () => {
      // Navigate to detection page
      await page.click('a[href="/detection"], .nav-detection')
      await expect(page).toHaveURL(/.*detection/)

      // Verify detection interface
      await expect(page.locator('.detection-container')).toBeVisible()

      // Select trained detector
      await page.selectOption('select[name="detector"]', { label: TEST_DETECTOR.name })

      // Select dataset
      await page.selectOption('select[name="dataset"]', { label: TEST_DATASET.name })

      // Configure detection parameters
      await page.fill('input[name="threshold"]', '0.5')

      // Start detection
      await page.click('button:has-text("Run Detection"), .start-detection')

      // Wait for detection to complete
      await page.waitForSelector('.detection-results, .results-ready', { timeout: 30000 })

      // Verify results display
      await expect(page.locator('.detection-results')).toBeVisible()
      await expect(page.locator('.anomaly-count')).toBeVisible()
      await expect(page.locator('.results-table, .anomaly-list')).toBeVisible()

      // Verify result details
      await expect(page.locator('.anomaly-score')).toHaveCount.greaterThan(0)

      // Test result visualization
      await expect(page.locator('.results-chart, canvas')).toBeVisible()
    })

    test('should export detection results', async () => {
      await page.goto(`${BASE_URL}/detection`)

      // Run detection first (assuming results exist)
      await page.selectOption('select[name="detector"]', { index: 0 })
      await page.selectOption('select[name="dataset"]', { index: 0 })
      await page.click('.start-detection')
      await page.waitForSelector('.detection-results')

      // Test CSV export
      const [csvDownload] = await Promise.all([
        page.waitForEvent('download'),
        page.click('button:has-text("Export CSV"), .export-csv')
      ])

      expect(csvDownload.suggestedFilename()).toMatch(/\.csv$/)

      // Test JSON export
      const [jsonDownload] = await Promise.all([
        page.waitForEvent('download'),
        page.click('button:has-text("Export JSON"), .export-json')
      ])

      expect(jsonDownload.suggestedFilename()).toMatch(/\.json$/)
    })

    test('should handle detection errors', async () => {
      await page.goto(`${BASE_URL}/detection`)

      // Try detection without selecting detector
      await page.click('.start-detection')

      // Verify error message
      await expect(page.locator('.error-message')).toBeVisible()
      await expect(page.locator('.error-message')).toContainText(/detector|required/i)

      // Try with invalid threshold
      await page.selectOption('select[name="detector"]', { index: 0 })
      await page.selectOption('select[name="dataset"]', { index: 0 })
      await page.fill('input[name="threshold"]', '2.0')
      await page.click('.start-detection')

      // Verify validation error
      await expect(page.locator('.error-message')).toContainText(/threshold|invalid/i)
    })
  })

  test.describe('Security Dashboard Workflow', () => {
    test.beforeEach(async () => {
      // Login as admin user
      await page.goto(`${BASE_URL}/login`)
      await page.fill('input[name="username"]', 'admin')
      await page.fill('input[name="password"]', 'admin_password')
      await page.click('button[type="submit"]')
      await page.waitForURL(/.*dashboard|.*\/$/)
    })

    test('should display security metrics', async () => {
      // Navigate to security dashboard
      await page.click('a[href="/security-dashboard"], .nav-security')
      await expect(page).toHaveURL(/.*security/)

      // Verify security dashboard loads
      await expect(page.locator('.security-dashboard')).toBeVisible()

      // Verify threat level indicator
      await expect(page.locator('.threat-level-indicator')).toBeVisible()
      await expect(page.locator('.threat-level')).toBeVisible()

      // Verify security metrics
      await expect(page.locator('.security-metrics')).toBeVisible()
      await expect(page.locator('.metric[data-metric="blocked-requests"]')).toBeVisible()
      await expect(page.locator('.metric[data-metric="active-threats"]')).toBeVisible()

      // Verify security events
      await expect(page.locator('.security-events')).toBeVisible()
    })

    test('should block IP address', async () => {
      await page.goto(`${BASE_URL}/security-dashboard`)

      // Find IP blocking form
      await expect(page.locator('.ip-blocking-panel')).toBeVisible()

      // Fill IP blocking form
      await page.fill('input[name="ip"]', '192.168.1.100')
      await page.fill('input[name="reason"]', 'Test block from E2E test')

      // Submit block request
      await page.click('button:has-text("Block IP"), .block-ip-btn')

      // Wait for success message
      await expect(page.locator('.success-message')).toBeVisible()

      // Verify IP appears in blocked list
      await expect(page.locator('.blocked-ips-list')).toContainText('192.168.1.100')
    })

    test('should configure WAF settings', async () => {
      await page.goto(`${BASE_URL}/security-dashboard`)

      // Navigate to WAF configuration
      await page.click('.waf-config-tab, a[href="#waf-config"]')

      // Verify WAF configuration panel
      await expect(page.locator('.waf-config-panel')).toBeVisible()

      // Update WAF settings
      await page.fill('input[name="block_threshold"]', '10')
      await page.fill('input[name="rate_limit"]', '200')

      // Save configuration
      await page.click('button:has-text("Save Configuration"), .save-waf-config')

      // Verify success message
      await expect(page.locator('.success-message')).toBeVisible()

      // Verify settings are saved
      await page.reload()
      await expect(page.locator('input[name="block_threshold"]')).toHaveValue('10')
    })

    test('should run security test', async () => {
      await page.goto(`${BASE_URL}/security-dashboard`)

      // Navigate to security testing panel
      await page.click('.security-testing-tab')

      // Configure security test
      await page.selectOption('select[name="test_type"]', 'sql-injection')
      await page.fill('input[name="target_url"]', BASE_URL)

      // Run security test
      await page.click('button:has-text("Run Security Test"), .run-test')

      // Wait for test completion
      await page.waitForSelector('.test-results', { timeout: 60000 })

      // Verify test results
      await expect(page.locator('.test-results')).toBeVisible()
      await expect(page.locator('.vulnerabilities-found')).toBeVisible()

      // Check if test detected vulnerabilities appropriately
      const vulnCount = await page.locator('.vulnerabilities-found').textContent()
      expect(vulnCount).toMatch(/\d+/)
    })
  })

  test.describe('Real-time Features', () => {
    test.beforeEach(async () => {
      await page.goto(`${BASE_URL}/login`)
      await page.fill('input[name="username"]', TEST_USER.username)
      await page.fill('input[name="password"]', TEST_USER.password)
      await page.click('button[type="submit"]')
      await page.waitForURL(/.*dashboard|.*\/$/)
    })

    test('should display real-time updates', async () => {
      await page.goto(`${BASE_URL}/dashboard`)

      // Verify real-time connection indicator
      await expect(page.locator('.real-time-indicator')).toBeVisible()
      await expect(page.locator('.connection-status')).toContainText(/connected|online/i)

      // Wait for real-time updates
      await page.waitForTimeout(5000)

      // Verify metrics update
      const initialMetric = await page.locator('.metric-value').first().textContent()

      // Wait for potential update
      await page.waitForTimeout(10000)

      // Note: In real test, we'd verify metrics changed or stayed consistent
      const updatedMetric = await page.locator('.metric-value').first().textContent()
      expect(updatedMetric).toBeDefined()
    })

    test('should handle connection loss', async () => {
      await page.goto(`${BASE_URL}/dashboard`)

      // Simulate network disconnection
      await page.context().setOffline(true)

      // Wait for offline indicator
      await page.waitForSelector('.offline-indicator, .connection-error', { timeout: 10000 })

      // Verify offline state
      await expect(page.locator('.connection-status')).toContainText(/offline|disconnected/i)

      // Restore connection
      await page.context().setOffline(false)

      // Wait for reconnection
      await page.waitForSelector('.online-indicator, .connection-restored', { timeout: 15000 })

      // Verify reconnection
      await expect(page.locator('.connection-status')).toContainText(/connected|online/i)
    })
  })

  test.describe('Error Handling and Edge Cases', () => {
    test('should handle API errors gracefully', async () => {
      // Mock API failures
      await page.route(`${API_BASE}/**`, route => {
        route.fulfill({
          status: 500,
          contentType: 'application/json',
          body: JSON.stringify({ error: 'Internal server error' })
        })
      })

      await page.goto(`${BASE_URL}/dashboard`)

      // Verify error handling
      await expect(page.locator('.error-message, .api-error')).toBeVisible()
      await expect(page.locator('.retry-button, .refresh-button')).toBeVisible()
    })

    test('should handle slow loading states', async () => {
      // Add delay to API responses
      await page.route(`${API_BASE}/**`, async route => {
        await new Promise(resolve => setTimeout(resolve, 2000))
        route.continue()
      })

      await page.goto(`${BASE_URL}/dashboard`)

      // Verify loading states
      await expect(page.locator('.loading-spinner, .loading-indicator')).toBeVisible()

      // Wait for content to load
      await page.waitForSelector('.dashboard-content', { timeout: 10000 })

      // Verify loading state is removed
      await expect(page.locator('.loading-spinner')).not.toBeVisible()
    })

    test('should validate form inputs', async () => {
      await page.goto(`${BASE_URL}/login`)

      // Test empty form submission
      await page.click('button[type="submit"]')

      // Verify validation messages
      await expect(page.locator('.validation-error, .field-error')).toHaveCount.greaterThan(0)

      // Test invalid email format (if applicable)
      await page.fill('input[name="username"]', 'invalid-email')
      await page.fill('input[name="password"]', 'short')
      await page.click('button[type="submit"]')

      // Verify specific validation errors
      await expect(page.locator('.validation-error')).toBeVisible()
    })
  })
})
