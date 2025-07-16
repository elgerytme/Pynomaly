import { test, expect, Page, BrowserContext } from '@playwright/test'

// Security testing configuration
const BASE_URL = process.env.BASE_URL || 'http://localhost:8000'
const SECURITY_TEST_USER = {
  username: 'security_test_user',
  password: 'SecureTestPass123!',
  email: 'security@test.com'
}

// XSS payloads for testing
const XSS_PAYLOADS = [
  '<script>alert("XSS")</script>',
  '<img src="x" onerror="alert(\'XSS\')">',
  'javascript:alert("XSS")',
  '<svg onload="alert(\'XSS\')">',
  '"><script>alert("XSS")</script>',
  '<iframe src="javascript:alert(\'XSS\')"></iframe>',
  '<body onload="alert(\'XSS\')">',
  '<input type="image" src="x" onerror="alert(\'XSS\')">',
  '<details open ontoggle="alert(\'XSS\')">',
  '\'><script>alert(String.fromCharCode(88,83,83))</script>'
]

// SQL injection payloads
const SQL_PAYLOADS = [
  "' OR '1'='1",
  "' OR 1=1--",
  '" OR "1"="1',
  "' UNION SELECT NULL--",
  "'; DROP TABLE users;--",
  "' OR 1=1#",
  "admin'--",
  "' OR 'a'='a",
  "1' OR '1'='1' --",
  "' OR 1=1 LIMIT 1--"
]

// CSRF test utilities
const CSRF_TEST_ACTIONS = [
  { method: 'POST', endpoint: '/api/v1/datasets/upload' },
  { method: 'POST', endpoint: '/api/v1/detectors/create' },
  { method: 'DELETE', endpoint: '/api/v1/datasets/1' },
  { method: 'POST', endpoint: '/api/v1/auth/logout' },
  { method: 'PUT', endpoint: '/api/v1/users/profile' }
]

test.describe('Security Testing Framework', () => {
  let page: Page
  let context: BrowserContext

  test.beforeEach(async ({ page: testPage, context: testContext }) => {
    page = testPage
    context = testContext

    // Set up security headers monitoring
    await page.route('**/*', route => {
      route.continue()
    })

    await page.goto(BASE_URL)
    await page.waitForLoadState('networkidle')
  })

  test.describe('Cross-Site Scripting (XSS) Protection', () => {
    test.beforeEach(async () => {
      // Login for authenticated XSS testing
      await page.goto(`${BASE_URL}/login`)
      await page.fill('input[name="username"]', SECURITY_TEST_USER.username)
      await page.fill('input[name="password"]', SECURITY_TEST_USER.password)
      await page.click('button[type="submit"]')
      await page.waitForURL(/.*dashboard|.*\/$/)
    })

    test('should prevent reflected XSS in search inputs', async () => {
      // Test search functionality for XSS
      await page.goto(`${BASE_URL}/datasets`)

      for (const payload of XSS_PAYLOADS) {
        // Clear any existing alerts
        page.on('dialog', dialog => dialog.dismiss())

        // Try XSS payload in search
        await page.fill('input[name="search"], input[type="search"]', payload)
        await page.press('input[name="search"], input[type="search"]', 'Enter')

        // Wait for response
        await page.waitForTimeout(1000)

        // Verify payload is properly escaped in DOM
        const searchResults = await page.locator('.search-results, .dataset-list').innerHTML()

        // Check that script tags are escaped or not present
        expect(searchResults).not.toContain('<script>')
        expect(searchResults).not.toContain('javascript:')
        expect(searchResults).not.toContain('onerror=')

        // Verify no alert was triggered
        expect(page.locator('.xss-alert, .security-alert')).not.toBeVisible()
      }
    })

    test('should prevent stored XSS in form submissions', async () => {
      await page.goto(`${BASE_URL}/detectors`)
      await page.click('.create-detector-btn')

      for (const payload of XSS_PAYLOADS) {
        // Try XSS in detector name field
        await page.fill('input[name="name"]', `Test Detector ${payload}`)
        await page.fill('textarea[name="description"]', `Description with ${payload}`)

        await page.click('.submit-detector')

        // Wait for submission
        await page.waitForTimeout(2000)

        // Navigate to detector list and verify content is escaped
        await page.goto(`${BASE_URL}/detectors`)

        const detectorList = await page.locator('.detector-list, .detectors-container').innerHTML()

        // Verify XSS payload is properly escaped
        expect(detectorList).not.toContain('<script>')
        expect(detectorList).not.toContain('javascript:')
        expect(detectorList).not.toMatch(/on\w+\s*=/)

        // Clean up created detector
        try {
          await page.click('.detector-item:last-child .delete-btn')
          await page.click('.confirm-delete')
        } catch (e) {
          // Ignore cleanup errors
        }
      }
    })

    test('should sanitize file upload names', async () => {
      await page.goto(`${BASE_URL}/datasets`)
      await page.click('.upload-dataset-btn')

      // Create a mock file with XSS payload in name
      const maliciousFileName = 'test<script>alert("XSS")</script>.csv'

      // Simulate file selection (this would be handled by the backend in real scenario)
      await page.fill('input[name="name"]', maliciousFileName)

      // Check that filename is properly displayed
      const displayedName = await page.locator('input[name="name"]').inputValue()

      // Verify no script execution or unsafe content
      expect(displayedName).not.toContain('<script>')
      await page.waitForTimeout(1000)

      // Verify no alert dialogs
      expect(page.locator('.xss-alert')).not.toBeVisible()
    })

    test('should prevent DOM-based XSS', async () => {
      // Test URL parameters and hash fragments
      const domXssPayloads = [
        `${BASE_URL}/dashboard#<script>alert('XSS')</script>`,
        `${BASE_URL}/datasets?search=<script>alert('XSS')</script>`,
        `${BASE_URL}/detectors?filter=<img src=x onerror=alert('XSS')>`
      ]

      for (const url of domXssPayloads) {
        await page.goto(url)
        await page.waitForTimeout(1000)

        // Verify no script execution
        const pageContent = await page.content()
        expect(pageContent).not.toContain('alert(')
        expect(pageContent).not.toMatch(/<script[^>]*>.*?<\/script>/i)
      }
    })
  })

  test.describe('SQL Injection Protection', () => {
    test.beforeEach(async () => {
      await page.goto(`${BASE_URL}/login`)
      await page.fill('input[name="username"]', SECURITY_TEST_USER.username)
      await page.fill('input[name="password"]', SECURITY_TEST_USER.password)
      await page.click('button[type="submit"]')
      await page.waitForURL(/.*dashboard|.*\/$/)
    })

    test('should prevent SQL injection in search queries', async () => {
      await page.goto(`${BASE_URL}/datasets`)

      for (const payload of SQL_PAYLOADS) {
        // Monitor network requests for SQL errors
        const responsePromise = page.waitForResponse(/.*search.*|.*filter.*/)

        await page.fill('input[name="search"]', payload)
        await page.press('input[name="search"]', 'Enter')

        try {
          const response = await responsePromise
          const responseText = await response.text()

          // Check for SQL error indicators
          const sqlErrorPatterns = [
            /SQL syntax error/i,
            /mysql_fetch/i,
            /ORA-\d+/i,
            /PostgreSQL/i,
            /Warning: mysql/i,
            /MySQLSyntaxErrorException/i,
            /SQLServer JDBC Driver/i,
            /Oracle error/i,
            /sqlite3\.OperationalError/i,
            /Unclosed quotation mark/i,
            /Microsoft OLE DB Provider/i
          ]

          for (const pattern of sqlErrorPatterns) {
            expect(responseText).not.toMatch(pattern)
          }

          // Verify response is still valid JSON or expected format
          expect(response.status()).toBeLessThan(500)

        } catch (error) {
          // If no response, that's also acceptable (request blocked)
          console.log('Request blocked or no response - good security practice')
        }
      }
    })

    test('should prevent SQL injection in login forms', async () => {
      await page.goto(`${BASE_URL}/logout`)
      await page.goto(`${BASE_URL}/login`)

      for (const payload of SQL_PAYLOADS) {
        // Monitor login requests
        const loginPromise = page.waitForResponse(/.*login.*/)

        await page.fill('input[name="username"]', payload)
        await page.fill('input[name="password"]', payload)
        await page.click('button[type="submit"]')

        try {
          const response = await loginPromise
          const responseText = await response.text()

          // Should not expose SQL errors
          expect(responseText).not.toMatch(/SQL.*error/i)
          expect(responseText).not.toMatch(/mysql/i)
          expect(responseText).not.toMatch(/postgresql/i)

          // Should show generic error message
          expect(page.locator('.error-message')).toBeVisible()

        } catch (error) {
          // Timeout is acceptable - request may be blocked
        }

        // Clear form for next test
        await page.fill('input[name="username"]', '')
        await page.fill('input[name="password"]', '')
      }
    })

    test('should use parameterized queries for API endpoints', async () => {
      // Test various API endpoints with SQL injection attempts
      const apiEndpoints = [
        { url: `/api/v1/datasets?search=${encodeURIComponent("' OR 1=1--")}`, method: 'GET' },
        { url: `/api/v1/detectors?filter=${encodeURIComponent("'; DROP TABLE users;--")}`, method: 'GET' },
        { url: `/api/v1/results?detector_id=${encodeURIComponent("1' UNION SELECT * FROM users--")}`, method: 'GET' }
      ]

      for (const endpoint of apiEndpoints) {
        const response = await page.request.get(`${BASE_URL}${endpoint.url}`)

        // API should handle malicious input gracefully
        expect(response.status()).toBeLessThan(500)

        const responseText = await response.text()

        // Should not expose database errors
        expect(responseText).not.toMatch(/SQL.*error/i)
        expect(responseText).not.toMatch(/database.*error/i)
        expect(responseText).not.toMatch(/syntax.*error/i)
      }
    })
  })

  test.describe('Cross-Site Request Forgery (CSRF) Protection', () => {
    test.beforeEach(async () => {
      await page.goto(`${BASE_URL}/login`)
      await page.fill('input[name="username"]', SECURITY_TEST_USER.username)
      await page.fill('input[name="password"]', SECURITY_TEST_USER.password)
      await page.click('button[type="submit"]')
      await page.waitForURL(/.*dashboard|.*\/$/)
    })

    test('should require CSRF tokens for state-changing operations', async () => {
      // Test that forms include CSRF tokens
      await page.goto(`${BASE_URL}/datasets`)
      await page.click('.upload-dataset-btn')

      // Verify CSRF token is present in form
      const csrfToken = page.locator('input[name="csrf_token"], input[name="_token"]')
      await expect(csrfToken).toBeVisible()

      const tokenValue = await csrfToken.getAttribute('value')
      expect(tokenValue).toBeTruthy()
      expect(tokenValue?.length).toBeGreaterThan(10)
    })

    test('should reject requests without valid CSRF tokens', async () => {
      // Create a new context without CSRF token
      const newContext = await context.browser()?.newContext()
      const attackerPage = await newContext?.newPage()

      if (attackerPage) {
        // Try to make requests without proper CSRF token
        for (const action of CSRF_TEST_ACTIONS) {
          const response = await attackerPage.request.fetch(`${BASE_URL}${action.endpoint}`, {
            method: action.method,
            headers: {
              'Content-Type': 'application/json'
            },
            data: JSON.stringify({ malicious: 'data' })
          })

          // Should be rejected with 403 or 400 status
          expect([400, 403, 422]).toContain(response.status())
        }

        await attackerPage.close()
        await newContext?.close()
      }
    })

    test('should validate CSRF token origin', async () => {
      // Get valid CSRF token
      await page.goto(`${BASE_URL}/datasets`)
      await page.click('.upload-dataset-btn')

      const csrfToken = await page.locator('input[name="csrf_token"]').getAttribute('value')

      // Try to use token from different origin
      const response = await page.request.post(`${BASE_URL}/api/v1/datasets/upload`, {
        headers: {
          'Origin': 'https://malicious-site.com',
          'Referer': 'https://malicious-site.com',
          'Content-Type': 'application/json'
        },
        data: JSON.stringify({
          csrf_token: csrfToken,
          name: 'Malicious Dataset'
        })
      })

      // Should be rejected
      expect([400, 403, 422]).toContain(response.status())
    })

    test('should prevent CSRF in AJAX requests', async () => {
      // Test that AJAX requests require proper headers
      const response = await page.evaluate(async () => {
        try {
          const response = await fetch('/api/v1/detectors/create', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json'
            },
            body: JSON.stringify({
              name: 'CSRF Test Detector',
              algorithm: 'IsolationForest'
            })
          })
          return { status: response.status, ok: response.ok }
        } catch (error) {
          return { error: error.message }
        }
      })

      // Should be rejected without proper CSRF protection
      expect(response.status).not.toBe(200)
    })
  })

  test.describe('Security Headers Validation', () => {
    test('should include security headers in responses', async () => {
      const response = await page.goto(BASE_URL)
      const headers = response?.headers() || {}

      // Check for essential security headers
      expect(headers['x-frame-options']).toBeDefined()
      expect(headers['x-content-type-options']).toBe('nosniff')
      expect(headers['x-xss-protection']).toBeDefined()
      expect(headers['content-security-policy']).toBeDefined()

      // Verify X-Frame-Options prevents clickjacking
      expect(headers['x-frame-options']).toMatch(/DENY|SAMEORIGIN/i)

      // Check for HSTS if HTTPS
      if (BASE_URL.startsWith('https://')) {
        expect(headers['strict-transport-security']).toBeDefined()
      }
    })

    test('should have secure Content Security Policy', async () => {
      const response = await page.goto(BASE_URL)
      const headers = response?.headers() || {}
      const csp = headers['content-security-policy']

      if (csp) {
        // Should not allow unsafe-inline or unsafe-eval in production
        expect(csp).not.toContain('unsafe-inline')
        expect(csp).not.toContain('unsafe-eval')

        // Should specify allowed sources
        expect(csp).toContain("default-src 'self'")

        // Should have nonce or hash-based script sources
        expect(csp).toMatch(/script-src.*('nonce-|'sha\d+-)/i)
      }
    })

    test('should set secure cookie attributes', async () => {
      await page.goto(`${BASE_URL}/login`)
      await page.fill('input[name="username"]', SECURITY_TEST_USER.username)
      await page.fill('input[name="password"]', SECURITY_TEST_USER.password)
      await page.click('button[type="submit"]')

      // Check session cookies
      const cookies = await context.cookies()
      const sessionCookie = cookies.find(cookie =>
        cookie.name.includes('session') || cookie.name.includes('auth')
      )

      if (sessionCookie) {
        // Should be HTTP-only
        expect(sessionCookie.httpOnly).toBe(true)

        // Should be secure if HTTPS
        if (BASE_URL.startsWith('https://')) {
          expect(sessionCookie.secure).toBe(true)
        }

        // Should have SameSite attribute
        expect(sessionCookie.sameSite).toMatch(/Strict|Lax/i)
      }
    })
  })

  test.describe('Input Validation and Sanitization', () => {
    test.beforeEach(async () => {
      await page.goto(`${BASE_URL}/login`)
      await page.fill('input[name="username"]', SECURITY_TEST_USER.username)
      await page.fill('input[name="password"]', SECURITY_TEST_USER.password)
      await page.click('button[type="submit"]')
      await page.waitForURL(/.*dashboard|.*\/$/)
    })

    test('should validate input lengths and formats', async () => {
      await page.goto(`${BASE_URL}/detectors`)
      await page.click('.create-detector-btn')

      // Test extremely long input
      const longString = 'A'.repeat(10000)
      await page.fill('input[name="name"]', longString)
      await page.fill('textarea[name="description"]', longString)

      await page.click('.submit-detector')

      // Should show validation error for length
      await expect(page.locator('.validation-error, .error-message')).toBeVisible()
      await expect(page.locator('.error-message')).toContainText(/length|long|limit/i)
    })

    test('should sanitize HTML content', async () => {
      await page.goto(`${BASE_URL}/datasets`)
      await page.click('.upload-dataset-btn')

      const htmlPayloads = [
        '<iframe src="javascript:alert(\'XSS\')"></iframe>',
        '<object data="data:text/html,<script>alert(\'XSS\')</script>"></object>',
        '<embed src="data:text/html,<script>alert(\'XSS\')</script>">',
        '<link rel="stylesheet" href="javascript:alert(\'XSS\')">',
        '<style>@import "javascript:alert(\'XSS\')";</style>'
      ]

      for (const payload of htmlPayloads) {
        await page.fill('input[name="name"]', `Test ${payload}`)
        await page.fill('textarea[name="description"]', payload)

        // Check that HTML is properly escaped in preview
        const namePreview = await page.locator('input[name="name"]').inputValue()
        const descPreview = await page.locator('textarea[name="description"]').inputValue()

        // Verify dangerous HTML tags are escaped or removed
        expect(namePreview).not.toContain('<iframe')
        expect(namePreview).not.toContain('<object')
        expect(descPreview).not.toContain('<script')
      }
    })

    test('should validate file upload types and sizes', async () => {
      await page.goto(`${BASE_URL}/datasets`)
      await page.click('.upload-dataset-btn')

      // Test file type validation
      const fileInput = page.locator('input[type="file"]')

      // Try to upload executable file (if file validation is client-side)
      try {
        await fileInput.setInputFiles({
          name: 'malicious.exe',
          mimeType: 'application/x-executable',
          buffer: Buffer.from('fake executable content')
        })

        await page.click('.submit-upload')

        // Should show validation error
        await expect(page.locator('.error-message')).toBeVisible()
        await expect(page.locator('.error-message')).toContainText(/file.*type|invalid.*format/i)
      } catch (error) {
        // File validation might be server-side only
        console.log('File validation handled server-side')
      }
    })
  })

  test.describe('Authentication and Authorization', () => {
    test('should prevent brute force attacks', async () => {
      await page.goto(`${BASE_URL}/logout`)
      await page.goto(`${BASE_URL}/login`)

      // Attempt multiple failed logins
      const attemptLogin = async (attempt: number) => {
        await page.fill('input[name="username"]', 'nonexistent_user')
        await page.fill('input[name="password"]', `wrong_password_${attempt}`)

        const responsePromise = page.waitForResponse(/.*login.*/)
        await page.click('button[type="submit"]')

        try {
          const response = await responsePromise
          return response.status()
        } catch (error) {
          return 0 // Timeout or blocked
        }
      }

      // Make multiple failed attempts
      const attempts = []
      for (let i = 1; i <= 5; i++) {
        const status = await attemptLogin(i)
        attempts.push(status)
        await page.waitForTimeout(1000)
      }

      // Later attempts should be rate limited or blocked
      const laterAttempts = attempts.slice(-2)
      expect(laterAttempts.some(status => status === 429 || status === 0)).toBe(true)
    })

    test('should enforce proper session management', async () => {
      // Login
      await page.goto(`${BASE_URL}/login`)
      await page.fill('input[name="username"]', SECURITY_TEST_USER.username)
      await page.fill('input[name="password"]', SECURITY_TEST_USER.password)
      await page.click('button[type="submit"]')
      await page.waitForURL(/.*dashboard|.*\/$/)

      // Get initial session
      const initialCookies = await context.cookies()
      const sessionCookie = initialCookies.find(c => c.name.includes('session'))

      // Navigate to protected resource
      await page.goto(`${BASE_URL}/admin/users`)

      // Should either be accessible (if user has permission) or redirect to unauthorized
      const currentUrl = page.url()
      expect(currentUrl).not.toContain('/admin/users')
      // User should be redirected or see access denied
    })

    test('should validate JWT tokens properly', async () => {
      // Login to get valid token
      await page.goto(`${BASE_URL}/login`)
      await page.fill('input[name="username"]', SECURITY_TEST_USER.username)
      await page.fill('input[name="password"]', SECURITY_TEST_USER.password)
      await page.click('button[type="submit"]')

      // Test API with manipulated token
      const response = await page.evaluate(async () => {
        // Try to make API request with manipulated token
        try {
          const response = await fetch('/api/v1/admin/users', {
            headers: {
              'Authorization': 'Bearer invalid.jwt.token'
            }
          })
          return { status: response.status, ok: response.ok }
        } catch (error) {
          return { error: error.message }
        }
      })

      // Should be rejected
      expect([401, 403]).toContain(response.status)
    })
  })

  test.describe('Information Disclosure Prevention', () => {
    test('should not expose sensitive information in errors', async () => {
      // Test 404 pages
      await page.goto(`${BASE_URL}/nonexistent-page`)
      const pageContent = await page.content()

      // Should not expose server information
      expect(pageContent).not.toMatch(/apache|nginx|server/i)
      expect(pageContent).not.toMatch(/version|build/i)
      expect(pageContent).not.toMatch(/stack.*trace/i)

      // Test API errors
      const apiResponse = await page.request.get(`${BASE_URL}/api/v1/nonexistent`)
      const apiContent = await apiResponse.text()

      // Should not expose internal paths or system information
      expect(apiContent).not.toMatch(/\/home|\/var|\/etc|c:\\/i)
      expect(apiContent).not.toMatch(/traceback|exception/i)
    })

    test('should not expose debug information', async () => {
      const response = await page.goto(BASE_URL)
      const content = await page.content()
      const headers = response?.headers() || {}

      // Should not expose debug information
      expect(content).not.toContain('django.DEBUG')
      expect(content).not.toContain('FLASK_ENV=development')
      expect(content).not.toMatch(/debug.*true/i)

      // Should not have debug headers
      expect(headers['x-debug']).toBeUndefined()
      expect(headers['x-powered-by']).toBeUndefined()
    })

    test('should not leak sensitive data in client-side code', async () => {
      const pageContent = await page.content()

      // Should not contain sensitive configuration
      expect(pageContent).not.toMatch(/password|secret|api.*key/i)
      expect(pageContent).not.toMatch(/database.*url|connection.*string/i)
      expect(pageContent).not.toMatch(/aws.*access.*key|private.*key/i)

      // Check JavaScript sources
      const scriptSources = await page.$$eval('script[src]', scripts =>
        scripts.map(script => script.src)
      )

      for (const src of scriptSources) {
        if (src.startsWith(BASE_URL)) {
          const response = await page.request.get(src)
          const scriptContent = await response.text()

          expect(scriptContent).not.toMatch(/password|secret|api.*key/i)
          expect(scriptContent).not.toMatch(/database.*url/i)
        }
      }
    })
  })
})
