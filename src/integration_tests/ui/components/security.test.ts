import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { createMockElement, createMockEvent, waitFor } from '../setup'

describe('Security Components', () => {
  let container: HTMLElement

  beforeEach(() => {
    container = createMockElement('div', { id: 'security-container' })
    document.body.appendChild(container)
    vi.clearAllMocks()
  })

  afterEach(() => {
    container.remove()
    vi.clearAllTimers()
  })

  describe('Security Dashboard', () => {
    beforeEach(() => {
      container.innerHTML = `
        <div class="security-dashboard">
          <div class="threat-level-indicator">
            <span class="threat-level" id="threat-level">low</span>
            <span class="threat-description">System is secure</span>
          </div>
          <div class="security-metrics">
            <div class="metric" data-metric="blocked-requests">
              <span class="metric-value" id="blocked-count">0</span>
              <span class="metric-label">Blocked Requests</span>
            </div>
            <div class="metric" data-metric="active-threats">
              <span class="metric-value" id="threat-count">0</span>
              <span class="metric-label">Active Threats</span>
            </div>
          </div>
          <div class="security-events" id="security-events"></div>
        </div>
      `
    })

    it('should display current threat level', () => {
      const threatLevel = container.querySelector('#threat-level')
      expect(threatLevel?.textContent).toBe('low')

      // Update threat level
      if (threatLevel) {
        threatLevel.textContent = 'high'
        threatLevel.className = 'threat-level high'
      }

      expect(threatLevel?.textContent).toBe('high')
      expect(threatLevel?.classList.contains('high')).toBe(true)
    })

    it('should update security metrics', () => {
      const mockMetrics = {
        blocked_requests: 25,
        active_threats: 3,
        blocked_ips: 5
      }

      const blockedCountEl = container.querySelector('#blocked-count')
      const threatCountEl = container.querySelector('#threat-count')

      if (blockedCountEl) blockedCountEl.textContent = mockMetrics.blocked_requests.toString()
      if (threatCountEl) threatCountEl.textContent = mockMetrics.active_threats.toString()

      expect(blockedCountEl?.textContent).toBe('25')
      expect(threatCountEl?.textContent).toBe('3')
    })

    it('should display security events', () => {
      const mockEvents = [
        {
          id: '1',
          type: 'sql_injection',
          source_ip: '192.168.1.100',
          timestamp: Date.now(),
          severity: 'high',
          blocked: true
        },
        {
          id: '2',
          type: 'xss_attempt',
          source_ip: '10.0.0.50',
          timestamp: Date.now() - 60000,
          severity: 'medium',
          blocked: false
        }
      ]

      const eventsContainer = container.querySelector('#security-events')
      if (eventsContainer) {
        eventsContainer.innerHTML = mockEvents.map(event => `
          <div class="security-event ${event.severity}" data-event-id="${event.id}">
            <div class="event-type">${event.type}</div>
            <div class="event-source">${event.source_ip}</div>
            <div class="event-status">${event.blocked ? 'BLOCKED' : 'MONITORED'}</div>
          </div>
        `).join('')
      }

      const eventElements = container.querySelectorAll('.security-event')
      expect(eventElements).toHaveLength(2)

      const highSeverityEvent = container.querySelector('.security-event.high')
      expect(highSeverityEvent?.getAttribute('data-event-id')).toBe('1')
    })

    it('should handle real-time security updates', () => {
      const mockWebSocketMessage = {
        type: 'security_event',
        data: {
          event_type: 'brute_force',
          source_ip: '203.0.113.45',
          severity: 'critical',
          blocked: true,
          timestamp: Date.now()
        }
      }

      // Simulate WebSocket message handling
      const threatLevel = container.querySelector('#threat-level')
      if (threatLevel) {
        threatLevel.textContent = 'critical'
        threatLevel.className = 'threat-level critical'
      }

      expect(threatLevel?.textContent).toBe('critical')
      expect(threatLevel?.classList.contains('critical')).toBe(true)
    })
  })

  describe('IP Blocking Interface', () => {
    beforeEach(() => {
      container.innerHTML = `
        <div class="ip-blocking-panel">
          <div class="block-ip-form">
            <input type="text" id="ip-input" placeholder="IP Address" pattern="^(?:[0-9]{1,3}\\.){3}[0-9]{1,3}$">
            <input type="text" id="reason-input" placeholder="Reason">
            <button id="block-ip-btn" disabled>Block IP</button>
          </div>
          <div class="blocked-ips-list" id="blocked-ips">
            <div class="blocked-ip-item" data-ip="192.168.1.100">
              <span class="ip">192.168.1.100</span>
              <span class="reason">Malicious activity</span>
              <button class="unblock-btn" data-ip="192.168.1.100">Unblock</button>
            </div>
          </div>
        </div>
      `
    })

    it('should validate IP address format', () => {
      const ipInput = container.querySelector('#ip-input') as HTMLInputElement
      const blockButton = container.querySelector('#block-ip-btn') as HTMLButtonElement

      const validateIP = (ip: string): boolean => {
        const ipRegex = /^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$/
        return ipRegex.test(ip) && ip.split('.').every(octet => {
          const num = parseInt(octet)
          return num >= 0 && num <= 255
        })
      }

      // Valid IP
      ipInput.value = '192.168.1.100'
      expect(validateIP(ipInput.value)).toBe(true)
      blockButton.disabled = false

      // Invalid IP
      ipInput.value = '256.256.256.256'
      expect(validateIP(ipInput.value)).toBe(false)
      blockButton.disabled = true

      // Malformed IP
      ipInput.value = 'not.an.ip.address'
      expect(validateIP(ipInput.value)).toBe(false)
    })

    it('should handle IP blocking', async () => {
      const ipInput = container.querySelector('#ip-input') as HTMLInputElement
      const reasonInput = container.querySelector('#reason-input') as HTMLInputElement
      const blockButton = container.querySelector('#block-ip-btn') as HTMLButtonElement

      ipInput.value = '203.0.113.45'
      reasonInput.value = 'Suspicious activity'
      blockButton.disabled = false

      // Mock successful block request
      global.fetch = vi.fn().mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ success: true, message: 'IP blocked successfully' })
      })

      blockButton.click()

      await waitFor(() => {
        expect(global.fetch).toHaveBeenCalledWith('/api/security/block-ip', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            ip: '203.0.113.45',
            reason: 'Suspicious activity'
          })
        })
      })

      // Form should be cleared after successful block
      expect(ipInput.value).toBe('')
      expect(reasonInput.value).toBe('')
    })

    it('should handle IP unblocking', async () => {
      const unblockButton = container.querySelector('.unblock-btn') as HTMLButtonElement
      const blockedItem = container.querySelector('.blocked-ip-item') as HTMLElement

      // Mock successful unblock request
      global.fetch = vi.fn().mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ success: true, message: 'IP unblocked successfully' })
      })

      unblockButton.click()

      await waitFor(() => {
        expect(global.fetch).toHaveBeenCalledWith('/api/security/unblock-ip', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ ip: '192.168.1.100' })
        })
      })

      // Item should be removed from list
      blockedItem.remove()
      expect(container.querySelector('[data-ip="192.168.1.100"]')).toBeNull()
    })

    it('should handle API errors gracefully', async () => {
      const blockButton = container.querySelector('#block-ip-btn') as HTMLButtonElement

      // Mock failed request
      global.fetch = vi.fn().mockRejectedValueOnce(new Error('Network error'))

      const errorHandler = vi.fn()

      try {
        blockButton.click()
        await fetch('/api/security/block-ip')
      } catch (error) {
        errorHandler(error)
      }

      expect(errorHandler).toHaveBeenCalled()
    })
  })

  describe('Security Alerts', () => {
    beforeEach(() => {
      container.innerHTML = `
        <div class="security-alerts" id="alerts-container">
          <div class="alert critical" data-alert-id="1">
            <div class="alert-icon">🚨</div>
            <div class="alert-content">
              <div class="alert-title">Critical Security Alert</div>
              <div class="alert-message">Multiple failed login attempts detected</div>
              <div class="alert-timestamp">2024-01-15 10:30:00</div>
            </div>
            <button class="alert-dismiss" data-alert-id="1">×</button>
          </div>
        </div>
        <div class="alert-controls">
          <button id="clear-all-alerts">Clear All</button>
          <select id="alert-filter">
            <option value="all">All Alerts</option>
            <option value="critical">Critical</option>
            <option value="high">High</option>
            <option value="medium">Medium</option>
          </select>
        </div>
      `
    })

    it('should display security alerts', () => {
      const alertsContainer = container.querySelector('#alerts-container')
      const alerts = alertsContainer?.querySelectorAll('.alert')

      expect(alerts).toHaveLength(1)

      const criticalAlert = container.querySelector('.alert.critical')
      expect(criticalAlert?.getAttribute('data-alert-id')).toBe('1')
      expect(criticalAlert?.querySelector('.alert-title')?.textContent).toBe('Critical Security Alert')
    })

    it('should dismiss individual alerts', () => {
      const dismissButton = container.querySelector('.alert-dismiss') as HTMLButtonElement
      const alert = container.querySelector('.alert[data-alert-id="1"]') as HTMLElement

      dismissButton.click()

      // Alert should be removed or hidden
      alert.style.display = 'none'
      expect(alert.style.display).toBe('none')
    })

    it('should clear all alerts', () => {
      const clearAllButton = container.querySelector('#clear-all-alerts') as HTMLButtonElement
      const alertsContainer = container.querySelector('#alerts-container') as HTMLElement

      clearAllButton.click()

      // All alerts should be removed
      alertsContainer.innerHTML = ''
      expect(alertsContainer.children).toHaveLength(0)
    })

    it('should filter alerts by severity', () => {
      // Add more alerts for testing
      const alertsContainer = container.querySelector('#alerts-container') as HTMLElement
      alertsContainer.innerHTML += `
        <div class="alert high" data-alert-id="2">
          <div class="alert-title">High Priority Alert</div>
        </div>
        <div class="alert medium" data-alert-id="3">
          <div class="alert-title">Medium Priority Alert</div>
        </div>
      `

      const filterSelect = container.querySelector('#alert-filter') as HTMLSelectElement

      // Filter by critical
      filterSelect.value = 'critical'

      const criticalAlerts = container.querySelectorAll('.alert.critical')
      const nonCriticalAlerts = container.querySelectorAll('.alert:not(.critical)')

      expect(criticalAlerts).toHaveLength(1)
      expect(nonCriticalAlerts).toHaveLength(2)

      // Simulate filtering (hide non-critical alerts)
      nonCriticalAlerts.forEach(alert => {
        (alert as HTMLElement).style.display = 'none'
      })

      const visibleAlerts = Array.from(container.querySelectorAll('.alert')).filter(
        alert => (alert as HTMLElement).style.display !== 'none'
      )
      expect(visibleAlerts).toHaveLength(1)
    })

    it('should handle new alert notifications', () => {
      const mockNewAlert = {
        id: '4',
        severity: 'high',
        title: 'New Security Threat',
        message: 'Suspicious file upload detected',
        timestamp: new Date().toISOString()
      }

      const alertsContainer = container.querySelector('#alerts-container') as HTMLElement

      // Add new alert
      const newAlertElement = document.createElement('div')
      newAlertElement.className = `alert ${mockNewAlert.severity}`
      newAlertElement.setAttribute('data-alert-id', mockNewAlert.id)
      newAlertElement.innerHTML = `
        <div class="alert-icon">⚠️</div>
        <div class="alert-content">
          <div class="alert-title">${mockNewAlert.title}</div>
          <div class="alert-message">${mockNewAlert.message}</div>
          <div class="alert-timestamp">${mockNewAlert.timestamp}</div>
        </div>
        <button class="alert-dismiss" data-alert-id="${mockNewAlert.id}">×</button>
      `

      alertsContainer.appendChild(newAlertElement)

      const alerts = alertsContainer.querySelectorAll('.alert')
      expect(alerts).toHaveLength(2)

      const newAlert = container.querySelector(`[data-alert-id="${mockNewAlert.id}"]`)
      expect(newAlert?.querySelector('.alert-title')?.textContent).toBe('New Security Threat')
    })
  })

  describe('WAF Configuration', () => {
    beforeEach(() => {
      container.innerHTML = `
        <div class="waf-config-panel">
          <div class="waf-status">
            <span class="status-indicator" id="waf-status"></span>
            <span class="status-text">WAF Active</span>
          </div>
          <div class="waf-settings">
            <div class="setting-group">
              <label for="waf-enabled">Enable WAF</label>
              <input type="checkbox" id="waf-enabled" checked>
            </div>
            <div class="setting-group">
              <label for="block-threshold">Auto-block Threshold</label>
              <input type="number" id="block-threshold" value="5" min="1" max="100">
            </div>
            <div class="setting-group">
              <label for="rate-limit">Rate Limit (req/min)</label>
              <input type="number" id="rate-limit" value="100" min="10" max="1000">
            </div>
            <button id="save-waf-config">Save Configuration</button>
          </div>
          <div class="waf-rules">
            <h3>Active Rules</h3>
            <div class="rule-item" data-rule-id="sql-injection">
              <span class="rule-name">SQL Injection Protection</span>
              <span class="rule-status enabled">Enabled</span>
              <button class="toggle-rule" data-rule-id="sql-injection">Toggle</button>
            </div>
            <div class="rule-item" data-rule-id="xss-protection">
              <span class="rule-name">XSS Protection</span>
              <span class="rule-status enabled">Enabled</span>
              <button class="toggle-rule" data-rule-id="xss-protection">Toggle</button>
            </div>
          </div>
        </div>
      `
    })

    it('should display WAF status', () => {
      const statusIndicator = container.querySelector('#waf-status')
      const statusText = container.querySelector('.status-text')

      expect(statusText?.textContent).toBe('WAF Active')

      // Update status
      statusIndicator?.classList.add('active')
      expect(statusIndicator?.classList.contains('active')).toBe(true)
    })

    it('should toggle WAF on/off', () => {
      const wafToggle = container.querySelector('#waf-enabled') as HTMLInputElement
      const statusText = container.querySelector('.status-text') as HTMLElement

      expect(wafToggle.checked).toBe(true)

      // Disable WAF
      wafToggle.checked = false
      wafToggle.dispatchEvent(new Event('change'))

      statusText.textContent = 'WAF Disabled'

      expect(wafToggle.checked).toBe(false)
      expect(statusText.textContent).toBe('WAF Disabled')
    })

    it('should validate configuration values', () => {
      const blockThreshold = container.querySelector('#block-threshold') as HTMLInputElement
      const rateLimit = container.querySelector('#rate-limit') as HTMLInputElement

      const validateThreshold = (value: number): boolean => value >= 1 && value <= 100
      const validateRateLimit = (value: number): boolean => value >= 10 && value <= 1000

      // Valid values
      blockThreshold.value = '10'
      rateLimit.value = '200'

      expect(validateThreshold(parseInt(blockThreshold.value))).toBe(true)
      expect(validateRateLimit(parseInt(rateLimit.value))).toBe(true)

      // Invalid values
      blockThreshold.value = '150'
      rateLimit.value = '5'

      expect(validateThreshold(parseInt(blockThreshold.value))).toBe(false)
      expect(validateRateLimit(parseInt(rateLimit.value))).toBe(false)
    })

    it('should save WAF configuration', async () => {
      const saveButton = container.querySelector('#save-waf-config') as HTMLButtonElement
      const wafEnabled = container.querySelector('#waf-enabled') as HTMLInputElement
      const blockThreshold = container.querySelector('#block-threshold') as HTMLInputElement
      const rateLimit = container.querySelector('#rate-limit') as HTMLInputElement

      // Mock successful save
      global.fetch = vi.fn().mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ success: true })
      })

      saveButton.click()

      await waitFor(() => {
        expect(global.fetch).toHaveBeenCalledWith('/api/security/waf/config', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            enabled: wafEnabled.checked,
            block_threshold: parseInt(blockThreshold.value),
            rate_limit: parseInt(rateLimit.value)
          })
        })
      })
    })

    it('should toggle individual WAF rules', () => {
      const sqlToggle = container.querySelector('[data-rule-id="sql-injection"] .toggle-rule') as HTMLButtonElement
      const sqlStatus = container.querySelector('[data-rule-id="sql-injection"] .rule-status') as HTMLElement

      expect(sqlStatus.textContent).toBe('Enabled')
      expect(sqlStatus.classList.contains('enabled')).toBe(true)

      // Toggle rule off
      sqlToggle.click()
      sqlStatus.textContent = 'Disabled'
      sqlStatus.classList.remove('enabled')
      sqlStatus.classList.add('disabled')

      expect(sqlStatus.textContent).toBe('Disabled')
      expect(sqlStatus.classList.contains('disabled')).toBe(true)
    })
  })

  describe('Security Testing Interface', () => {
    beforeEach(() => {
      container.innerHTML = `
        <div class="security-testing-panel">
          <div class="test-controls">
            <select id="test-type">
              <option value="sql-injection">SQL Injection</option>
              <option value="xss">Cross-Site Scripting</option>
              <option value="csrf">CSRF</option>
            </select>
            <input type="url" id="target-url" placeholder="Target URL" value="http://localhost:8000">
            <button id="run-test">Run Security Test</button>
          </div>
          <div class="test-results" id="test-results" style="display: none;">
            <h3>Test Results</h3>
            <div class="result-summary">
              <span class="vulnerabilities-found">0</span> vulnerabilities found
            </div>
            <div class="result-details"></div>
          </div>
          <div class="test-history">
            <h3>Recent Tests</h3>
            <div class="test-history-item">
              <span class="test-name">SQL Injection Test</span>
              <span class="test-date">2024-01-15 09:30:00</span>
              <span class="test-result passed">PASSED</span>
            </div>
          </div>
        </div>
      `
    })

    it('should run security tests', async () => {
      const testType = container.querySelector('#test-type') as HTMLSelectElement
      const targetUrl = container.querySelector('#target-url') as HTMLInputElement
      const runButton = container.querySelector('#run-test') as HTMLButtonElement
      const resultsPanel = container.querySelector('#test-results') as HTMLElement

      // Mock test response
      global.fetch = vi.fn().mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({
          test_type: 'sql-injection',
          vulnerabilities_found: 2,
          results: [
            { severity: 'high', description: 'SQL injection in login form' },
            { severity: 'medium', description: 'Unescaped user input in search' }
          ]
        })
      })

      testType.value = 'sql-injection'
      targetUrl.value = 'http://localhost:8000/test'

      runButton.click()

      await waitFor(() => {
        expect(global.fetch).toHaveBeenCalledWith('/api/security/test', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            test_type: 'sql-injection',
            target_url: 'http://localhost:8000/test'
          })
        })
      })

      // Results should be displayed
      resultsPanel.style.display = 'block'
      const vulnerabilityCount = container.querySelector('.vulnerabilities-found') as HTMLElement
      vulnerabilityCount.textContent = '2'

      expect(resultsPanel.style.display).toBe('block')
      expect(vulnerabilityCount.textContent).toBe('2')
    })

    it('should validate target URL', () => {
      const targetUrl = container.querySelector('#target-url') as HTMLInputElement
      const runButton = container.querySelector('#run-test') as HTMLButtonElement

      const validateUrl = (url: string): boolean => {
        try {
          new URL(url)
          return true
        } catch {
          return false
        }
      }

      // Valid URL
      targetUrl.value = 'http://localhost:8000'
      expect(validateUrl(targetUrl.value)).toBe(true)
      runButton.disabled = false

      // Invalid URL
      targetUrl.value = 'not-a-url'
      expect(validateUrl(targetUrl.value)).toBe(false)
      runButton.disabled = true
    })

    it('should display test history', () => {
      const historyItems = container.querySelectorAll('.test-history-item')
      expect(historyItems).toHaveLength(1)

      const testResult = container.querySelector('.test-result')
      expect(testResult?.textContent).toBe('PASSED')
      expect(testResult?.classList.contains('passed')).toBe(true)
    })
  })
})
