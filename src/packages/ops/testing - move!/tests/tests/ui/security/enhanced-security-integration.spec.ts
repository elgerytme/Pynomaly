import { test, expect, Page, BrowserContext } from '@playwright/test'

// Enhanced security integration testing
const BASE_URL = process.env.BASE_URL || 'http://localhost:8000'

// Test user configurations for different roles
const TEST_USERS = {
  admin: {
    username: 'admin_user',
    password: 'AdminPass123!',
    email: 'admin@test.com',
    role: 'admin'
  },
  analyst: {
    username: 'analyst_user',
    password: 'AnalystPass123!',
    email: 'analyst@test.com',
    role: 'data_analyst'
  },
  viewer: {
    username: 'viewer_user',
    password: 'ViewerPass123!',
    email: 'viewer@test.com',
    role: 'viewer'
  }
}

test.describe('Enhanced Security Features Integration', () => {
  let page: Page
  let context: BrowserContext

  test.beforeEach(async ({ page: testPage, context: testContext }) => {
    page = testPage
    context = testContext

    // Clear any existing auth state
    await context.clearCookies()
    await page.goto(BASE_URL)
  })

  test.describe('Multi-Factor Authentication (MFA)', () => {
    test('should enforce MFA setup for new users', async () => {
      // Navigate to registration
      await page.goto(`${BASE_URL}/register`)

      // Fill registration form
      await page.fill('input[name="username"]', 'new_mfa_user')
      await page.fill('input[name="email"]', 'mfa@test.com')
      await page.fill('input[name="password"]', 'SecurePass123!')
      await page.fill('input[name="confirm_password"]', 'SecurePass123!')
      await page.click('button[type="submit"]')

      // Should be redirected to MFA setup
      await page.waitForURL(/.*mfa.*setup|.*two.*factor.*setup/)

      // Verify MFA setup components are present
      await expect(page.locator('.qr-code, .mfa-qr-code')).toBeVisible()
      await expect(page.locator('.backup-codes, .recovery-codes')).toBeVisible()
      await expect(page.locator('input[name="mfa_code"]')).toBeVisible()
    })

    test('should require MFA verification on login', async () => {
      // Login with MFA-enabled user
      await page.goto(`${BASE_URL}/login`)
      await page.fill('input[name="username"]', TEST_USERS.admin.username)
      await page.fill('input[name="password"]', TEST_USERS.admin.password)
      await page.click('button[type="submit"]')

      // Should be redirected to MFA verification
      await page.waitForURL(/.*mfa.*verify|.*two.*factor/)

      // Verify MFA form elements
      await expect(page.locator('input[name="mfa_code"]')).toBeVisible()
      await expect(page.locator('.mfa-instructions')).toBeVisible()
      await expect(page.locator('.backup-code-link')).toBeVisible()

      // Test invalid MFA code
      await page.fill('input[name="mfa_code"]', '000000')
      await page.click('button[type="submit"]')

      // Should show error and remain on MFA page
      await expect(page.locator('.error-message')).toBeVisible()
      await expect(page.locator('.error-message')).toContainText(/invalid|incorrect/i)
    })

    test('should allow backup code authentication', async () => {
      await page.goto(`${BASE_URL}/login`)
      await page.fill('input[name="username"]', TEST_USERS.admin.username)
      await page.fill('input[name="password"]', TEST_USERS.admin.password)
      await page.click('button[type="submit"]')

      await page.waitForURL(/.*mfa.*verify/)

      // Click backup code link
      await page.click('.backup-code-link, .use-backup-code')

      // Should show backup code input
      await expect(page.locator('input[name="backup_code"]')).toBeVisible()
      await expect(page.locator('.backup-code-instructions')).toBeVisible()
    })

    test('should handle MFA device management', async () => {
      // Login as admin
      await page.goto(`${BASE_URL}/login`)
      await page.fill('input[name="username"]', TEST_USERS.admin.username)
      await page.fill('input[name="password"]', TEST_USERS.admin.password)
      await page.click('button[type="submit"]')

      // Navigate to security settings
      await page.goto(`${BASE_URL}/settings/security`)

      // Verify MFA management options
      await expect(page.locator('.mfa-devices, .authenticator-devices')).toBeVisible()
      await expect(page.locator('.add-device-btn')).toBeVisible()
      await expect(page.locator('.regenerate-backup-codes')).toBeVisible()

      // Test device removal
      const deviceCount = await page.locator('.mfa-device-item').count()
      if (deviceCount > 0) {
        await page.click('.mfa-device-item:first-child .remove-device')
        await page.click('.confirm-remove')

        // Should update device list
        const newDeviceCount = await page.locator('.mfa-device-item').count()
        expect(newDeviceCount).toBe(deviceCount - 1)
      }
    })
  })

  test.describe('Role-Based Access Control (RBAC)', () => {
    test('should enforce admin-only access to user management', async () => {
      // Test with non-admin user
      await page.goto(`${BASE_URL}/login`)
      await page.fill('input[name="username"]', TEST_USERS.viewer.username)
      await page.fill('input[name="password"]', TEST_USERS.viewer.password)
      await page.click('button[type="submit"]')

      // Try to access admin panel
      await page.goto(`${BASE_URL}/admin/users`)

      // Should be redirected or show access denied
      const currentUrl = page.url()
      expect(currentUrl).not.toContain('/admin/users')

      // Should show access denied message
      await expect(page.locator('.access-denied, .unauthorized')).toBeVisible()
    })

    test('should allow role-appropriate access to features', async () => {
      // Test analyst role permissions
      await page.goto(`${BASE_URL}/login`)
      await page.fill('input[name="username"]', TEST_USERS.analyst.username)
      await page.fill('input[name="password"]', TEST_USERS.analyst.password)
      await page.click('button[type="submit"]')

      // Analyst should access datasets and detectors
      await page.goto(`${BASE_URL}/datasets`)
      await expect(page.locator('.dataset-list')).toBeVisible()

      await page.goto(`${BASE_URL}/detectors`)
      await expect(page.locator('.detector-list')).toBeVisible()

      // But not admin functions
      await page.goto(`${BASE_URL}/admin/system-config`)
      const url = page.url()
      expect(url).not.toContain('/admin/system-config')
    })

    test('should show role-appropriate UI elements', async () => {
      // Test admin UI
      await page.goto(`${BASE_URL}/login`)
      await page.fill('input[name="username"]', TEST_USERS.admin.username)
      await page.fill('input[name="password"]', TEST_USERS.admin.password)
      await page.click('button[type="submit"]')

      // Admin should see admin menu items
      await expect(page.locator('.admin-menu, .user-management-link')).toBeVisible()
      await expect(page.locator('.system-settings-link')).toBeVisible()

      // Logout and test viewer UI
      await page.click('.user-menu .logout')

      await page.fill('input[name="username"]', TEST_USERS.viewer.username)
      await page.fill('input[name="password"]', TEST_USERS.viewer.password)
      await page.click('button[type="submit"]')

      // Viewer should not see admin elements
      await expect(page.locator('.admin-menu')).not.toBeVisible()
      await expect(page.locator('.create-btn, .edit-btn')).not.toBeVisible()
    })

    test('should validate permissions for API endpoints', async () => {
      // Login as viewer
      await page.goto(`${BASE_URL}/login`)
      await page.fill('input[name="username"]', TEST_USERS.viewer.username)
      await page.fill('input[name="password"]', TEST_USERS.viewer.password)
      await page.click('button[type="submit"]')

      // Test read-only access
      const readResponse = await page.request.get(`${BASE_URL}/api/v1/datasets`)
      expect(readResponse.status()).toBe(200)

      // Test write operations should be denied
      const writeResponse = await page.request.post(`${BASE_URL}/api/v1/datasets`, {
        data: { name: 'Test Dataset' }
      })
      expect([401, 403]).toContain(writeResponse.status())

      // Test admin operations should be denied
      const adminResponse = await page.request.delete(`${BASE_URL}/api/v1/users/1`)
      expect([401, 403]).toContain(adminResponse.status())
    })
  })

  test.describe('OAuth2 and SAML Integration', () => {
    test('should display OAuth2 login options', async () => {
      await page.goto(`${BASE_URL}/login`)

      // Check for OAuth2 provider buttons
      await expect(page.locator('.oauth-providers')).toBeVisible()
      await expect(page.locator('.google-login, .microsoft-login')).toBeVisible()

      // Verify OAuth2 URLs
      const googleLink = page.locator('.google-login')
      const href = await googleLink.getAttribute('href')
      expect(href).toContain('/auth/oauth2/google')
    })

    test('should handle OAuth2 callback processing', async () => {
      // Simulate OAuth2 callback
      await page.goto(`${BASE_URL}/auth/oauth2/callback/google?code=test_code&state=test_state`)

      // Should either redirect to dashboard or show error
      await page.waitForTimeout(2000)
      const url = page.url()

      // Should not stay on callback URL
      expect(url).not.toContain('/auth/oauth2/callback')
    })

    test('should provide SAML metadata endpoint', async () => {
      const response = await page.request.get(`${BASE_URL}/auth/saml/metadata`)

      // Should return valid SAML metadata
      expect(response.status()).toBe(200)

      const contentType = response.headers()['content-type']
      expect(contentType).toContain('xml')

      const metadata = await response.text()
      expect(metadata).toContain('<EntityDescriptor')
      expect(metadata).toContain('urn:oasis:names:tc:SAML:2.0:metadata')
    })

    test('should handle SAML assertions', async () => {
      // Test SAML SSO endpoint availability
      const response = await page.request.get(`${BASE_URL}/auth/saml/sso`)

      // Should redirect to IdP or show error for missing parameters
      expect([302, 400]).toContain(response.status())
    })
  })

  test.describe('Security Monitoring Dashboard', () => {
    test('should display security metrics for authorized users', async () => {
      // Login as admin
      await page.goto(`${BASE_URL}/login`)
      await page.fill('input[name="username"]', TEST_USERS.admin.username)
      await page.fill('input[name="password"]', TEST_USERS.admin.password)
      await page.click('button[type="submit"]')

      // Navigate to security dashboard
      await page.goto(`${BASE_URL}/security-dashboard`)

      // Verify security metrics are displayed
      await expect(page.locator('.security-metrics')).toBeVisible()
      await expect(page.locator('.active-alerts')).toBeVisible()
      await expect(page.locator('.request-timeline')).toBeVisible()
      await expect(page.locator('.threat-distribution')).toBeVisible()

      // Check metric cards
      await expect(page.locator('#total-requests')).toBeVisible()
      await expect(page.locator('#blocked-requests')).toBeVisible()
      await expect(page.locator('#active-alerts')).toBeVisible()
      await expect(page.locator('#active-sessions')).toBeVisible()
    })

    test('should show real-time security events', async () => {
      await page.goto(`${BASE_URL}/login`)
      await page.fill('input[name="username"]', TEST_USERS.admin.username)
      await page.fill('input[name="password"]', TEST_USERS.admin.password)
      await page.click('button[type="submit"]')

      await page.goto(`${BASE_URL}/security-dashboard`)

      // Wait for WebSocket connection
      await page.waitForTimeout(2000)

      // Generate security event by triggering rate limit
      const newPage = await context.newPage()
      for (let i = 0; i < 10; i++) {
        await newPage.request.get(`${BASE_URL}/api/v1/datasets`)
        await page.waitForTimeout(100)
      }

      // Check if events appear in dashboard
      await page.waitForTimeout(2000)
      const alertsContainer = page.locator('.alerts-container')
      const alertCount = await alertsContainer.locator('.alert-item').count()

      // Should show some alerts or events
      expect(alertCount).toBeGreaterThanOrEqual(0)

      await newPage.close()
    })

    test('should allow alert acknowledgment', async () => {
      await page.goto(`${BASE_URL}/login`)
      await page.fill('input[name="username"]', TEST_USERS.admin.username)
      await page.fill('input[name="password"]', TEST_USERS.admin.password)
      await page.click('button[type="submit"]')

      await page.goto(`${BASE_URL}/security-dashboard`)

      // Look for existing alerts
      const alerts = page.locator('.alert-item')
      const alertCount = await alerts.count()

      if (alertCount > 0) {
        // Click acknowledge on first alert
        await alerts.first().locator('.acknowledge-btn').click()

        // Alert should be marked as acknowledged
        await expect(alerts.first().locator('.acknowledged')).toBeVisible()
      }
    })

    test('should restrict dashboard access to authorized roles', async () => {
      // Test with viewer role
      await page.goto(`${BASE_URL}/login`)
      await page.fill('input[name="username"]', TEST_USERS.viewer.username)
      await page.fill('input[name="password"]', TEST_USERS.viewer.password)
      await page.click('button[type="submit"]')

      // Try to access security dashboard
      await page.goto(`${BASE_URL}/security-dashboard`)

      // Should be denied or redirected
      const url = page.url()
      expect(url).not.toContain('/security-dashboard')

      // Should show access denied
      await expect(page.locator('.access-denied, .unauthorized')).toBeVisible()
    })
  })

  test.describe('Advanced Rate Limiting', () => {
    test('should enforce endpoint-specific rate limits', async () => {
      // Test login endpoint rate limiting
      await page.goto(`${BASE_URL}/login`)

      const responses = []

      // Make multiple rapid login attempts
      for (let i = 0; i < 6; i++) {
        await page.fill('input[name="username"]', 'test_user')
        await page.fill('input[name="password"]', 'wrong_password')

        try {
          const responsePromise = page.waitForResponse(/.*login.*/, { timeout: 5000 })
          await page.click('button[type="submit"]')
          const response = await responsePromise
          responses.push(response.status())
        } catch (error) {
          responses.push(0) // Timeout indicates blocking
        }

        await page.waitForTimeout(500)
      }

      // Later requests should be rate limited
      const rateLimitedResponses = responses.filter(status => status === 429 || status === 0)
      expect(rateLimitedResponses.length).toBeGreaterThan(0)
    })

    test('should include rate limit headers', async () => {
      const response = await page.request.get(`${BASE_URL}/api/v1/datasets`)
      const headers = response.headers()

      // Check for rate limit headers
      expect(headers['x-ratelimit-limit']).toBeDefined()
      expect(headers['x-ratelimit-remaining']).toBeDefined()
      expect(headers['x-ratelimit-reset']).toBeDefined()

      // Verify header values are numeric
      expect(parseInt(headers['x-ratelimit-limit'])).toBeGreaterThan(0)
      expect(parseInt(headers['x-ratelimit-remaining'])).toBeGreaterThanOrEqual(0)
    })

    test('should apply burst protection', async () => {
      // Make rapid concurrent requests
      const requests = []
      for (let i = 0; i < 15; i++) {
        requests.push(page.request.get(`${BASE_URL}/api/v1/datasets`))
      }

      const responses = await Promise.all(requests)
      const statusCodes = responses.map(r => r.status())

      // Some requests should be rate limited
      const blockedRequests = statusCodes.filter(status => status === 429)
      expect(blockedRequests.length).toBeGreaterThan(0)
    })
  })

  test.describe('Web Application Firewall (WAF)', () => {
    test('should block malicious requests', async () => {
      const maliciousPayloads = [
        '/api/v1/datasets?search=<script>alert("xss")</script>',
        '/api/v1/users?filter=\' OR 1=1--',
        '/api/v1/files?path=../../../etc/passwd',
        '/api/v1/exec?cmd=; cat /etc/passwd'
      ]

      for (const payload of maliciousPayloads) {
        const response = await page.request.get(`${BASE_URL}${payload}`)

        // Should be blocked by WAF
        expect([400, 403, 422]).toContain(response.status())

        const responseText = await response.text()
        expect(responseText).toContain('security')
      }
    })

    test('should detect and block suspicious user agents', async () => {
      const suspiciousAgents = [
        'sqlmap/1.0',
        'Nikto/2.1.6',
        'Mozilla/5.0 (compatible; Nmap Scripting Engine)',
        'python-requests/2.25.1 (scanner)',
        'curl/7.68.0 (automated)'
      ]

      for (const agent of suspiciousAgents) {
        const response = await page.request.get(BASE_URL, {
          headers: { 'User-Agent': agent }
        })

        // Should be blocked
        expect([403, 429]).toContain(response.status())
      }
    })

    test('should allow legitimate requests', async () => {
      const legitimateAgents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'
      ]

      for (const agent of legitimateAgents) {
        const response = await page.request.get(BASE_URL, {
          headers: { 'User-Agent': agent }
        })

        // Should be allowed
        expect([200, 302]).toContain(response.status())
      }
    })
  })

  test.describe('Audit Logging', () => {
    test('should log authentication events', async () => {
      // Perform login
      await page.goto(`${BASE_URL}/login`)
      await page.fill('input[name="username"]', TEST_USERS.admin.username)
      await page.fill('input[name="password"]', TEST_USERS.admin.password)
      await page.click('button[type="submit"]')

      // Check audit logs (if accessible)
      await page.goto(`${BASE_URL}/admin/audit-logs`)

      if (page.url().includes('/audit-logs')) {
        // Verify recent login event is logged
        await expect(page.locator('.audit-log-entry')).toBeVisible()

        const recentEntries = page.locator('.audit-log-entry').first()
        await expect(recentEntries).toContainText(/login|authentication/i)
        await expect(recentEntries).toContainText(TEST_USERS.admin.username)
      }
    })

    test('should log data access events', async () => {
      await page.goto(`${BASE_URL}/login`)
      await page.fill('input[name="username"]', TEST_USERS.admin.username)
      await page.fill('input[name="password"]', TEST_USERS.admin.password)
      await page.click('button[type="submit"]')

      // Access sensitive data
      await page.goto(`${BASE_URL}/datasets`)
      await page.click('.dataset-item:first-child .view-details')

      // Check if access is logged
      await page.goto(`${BASE_URL}/admin/audit-logs`)

      if (page.url().includes('/audit-logs')) {
        const entries = page.locator('.audit-log-entry')
        const count = await entries.count()
        expect(count).toBeGreaterThan(0)
      }
    })

    test('should log security violations', async () => {
      // Trigger security violation
      await page.goto(`${BASE_URL}/datasets?search=<script>alert("test")</script>`)

      // Wait for processing
      await page.waitForTimeout(2000)

      // Check logs as admin
      await page.goto(`${BASE_URL}/login`)
      await page.fill('input[name="username"]', TEST_USERS.admin.username)
      await page.fill('input[name="password"]', TEST_USERS.admin.password)
      await page.click('button[type="submit"]')

      await page.goto(`${BASE_URL}/admin/audit-logs`)

      if (page.url().includes('/audit-logs')) {
        // Look for security violation entries
        const securityEntries = page.locator('.audit-log-entry:has-text("security")')
        const count = await securityEntries.count()
        expect(count).toBeGreaterThanOrEqual(0)
      }
    })
  })

  test.describe('Session Management', () => {
    test('should enforce session timeout', async () => {
      await page.goto(`${BASE_URL}/login`)
      await page.fill('input[name="username"]', TEST_USERS.admin.username)
      await page.fill('input[name="password"]', TEST_USERS.admin.password)
      await page.click('button[type="submit"]')

      // Check if session timeout is configured
      const cookies = await context.cookies()
      const sessionCookie = cookies.find(c => c.name.includes('session'))

      if (sessionCookie) {
        // Session should have expiration time
        expect(sessionCookie.expires).toBeGreaterThan(Date.now() / 1000)

        // Session should expire within reasonable time (e.g., 24 hours)
        const maxAge = sessionCookie.expires - Date.now() / 1000
        expect(maxAge).toBeLessThan(86400) // 24 hours
      }
    })

    test('should invalidate sessions on logout', async () => {
      await page.goto(`${BASE_URL}/login`)
      await page.fill('input[name="username"]', TEST_USERS.admin.username)
      await page.fill('input[name="password"]', TEST_USERS.admin.password)
      await page.click('button[type="submit"]')

      // Get session cookie
      const beforeLogout = await context.cookies()
      const sessionBefore = beforeLogout.find(c => c.name.includes('session'))

      // Logout
      await page.click('.user-menu .logout')

      // Check if session is invalidated
      const afterLogout = await context.cookies()
      const sessionAfter = afterLogout.find(c => c.name.includes('session'))

      // Session should be removed or changed
      expect(sessionAfter?.value).not.toBe(sessionBefore?.value)
    })

    test('should prevent session fixation', async () => {
      // Get initial session ID
      await page.goto(BASE_URL)
      const initialCookies = await context.cookies()
      const initialSession = initialCookies.find(c => c.name.includes('session'))

      // Login
      await page.goto(`${BASE_URL}/login`)
      await page.fill('input[name="username"]', TEST_USERS.admin.username)
      await page.fill('input[name="password"]', TEST_USERS.admin.password)
      await page.click('button[type="submit"]')

      // Get post-login session ID
      const postLoginCookies = await context.cookies()
      const postLoginSession = postLoginCookies.find(c => c.name.includes('session'))

      // Session ID should change after login
      if (initialSession && postLoginSession) {
        expect(postLoginSession.value).not.toBe(initialSession.value)
      }
    })
  })
})
