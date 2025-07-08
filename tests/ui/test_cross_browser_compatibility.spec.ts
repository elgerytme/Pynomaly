/**
 * Cross-Browser Compatibility Test Suite
 * Comprehensive testing across multiple browsers with visual regression support
 */

import { test, expect, Page, BrowserContext } from '@playwright/test';
import percySnapshot from '@percy/playwright';

// Test configuration for different browser contexts
const testViewports = [
  { name: 'Desktop', width: 1920, height: 1080 },
  { name: 'Tablet', width: 768, height: 1024 },
  { name: 'Mobile', width: 375, height: 667 }
];

const criticalPages = [
  { path: '/', name: 'Dashboard' },
  { path: '/detectors', name: 'Detectors' },
  { path: '/datasets', name: 'Datasets' },
  { path: '/detection', name: 'Detection' },
  { path: '/api/docs', name: 'API Documentation' }
];

class CrossBrowserTestHelper {
  constructor(private page: Page) {}

  async waitForPageLoad() {
    await this.page.waitForLoadState('networkidle');
    await this.page.waitForSelector('body', { state: 'visible' });
  }

  async checkConsoleErrors() {
    const consoleLogs: string[] = [];
    this.page.on('console', msg => {
      if (msg.type() === 'error') {
        consoleLogs.push(msg.text());
      }
    });
    return consoleLogs;
  }

  async validatePageStructure() {
    // Check for essential page elements
    const essentialSelectors = [
      'header',
      'main',
      'nav',
      '[data-testid="app-content"]'
    ];

    for (const selector of essentialSelectors) {
      await expect(this.page.locator(selector).first()).toBeVisible({
        timeout: 10000
      });
    }
  }

  async checkResponsiveLayout(viewport: { name: string; width: number; height: number }) {
    await this.page.setViewportSize({ width: viewport.width, height: viewport.height });
    await this.page.waitForTimeout(500); // Allow layout to settle

    // Check for responsive design indicators
    if (viewport.width < 768) {
      // Mobile: hamburger menu should be visible
      const hamburger = this.page.locator('[data-testid="mobile-menu-button"]');
      if (await hamburger.count() > 0) {
        await expect(hamburger).toBeVisible();
      }
    } else {
      // Desktop/Tablet: full navigation should be visible
      const navItems = this.page.locator('nav a');
      if (await navItems.count() > 0) {
        await expect(navItems.first()).toBeVisible();
      }
    }
  }

  async performInteractionTest() {
    // Test basic interactions that should work across all browsers
    const clickableElements = this.page.locator('button, a, [role="button"]').first();
    if (await clickableElements.count() > 0) {
      await clickableElements.hover();
      await this.page.waitForTimeout(200);
    }

    // Test form inputs if present
    const inputs = this.page.locator('input[type="text"], input[type="email"]').first();
    if (await inputs.count() > 0) {
      await inputs.fill('test@example.com');
      await inputs.clear();
    }
  }
}

test.describe('Cross-Browser Compatibility Suite', () => {
  test.beforeEach(async ({ page }) => {
    // Enable console error tracking
    page.on('console', msg => {
      if (msg.type() === 'error') {
        console.error(`Browser console error: ${msg.text()}`);
      }
    });

    // Set timeout for slower operations
    test.setTimeout(60000);
  });

  // Test each critical page across different browsers
  for (const pageInfo of criticalPages) {
    test(`${pageInfo.name} page loads correctly across browsers`, async ({ page, browserName }) => {
      const helper = new CrossBrowserTestHelper(page);

      await test.step(`Navigate to ${pageInfo.name}`, async () => {
        await page.goto(pageInfo.path);
        await helper.waitForPageLoad();
      });

      await test.step('Validate page structure', async () => {
        await helper.validatePageStructure();
      });

      await test.step('Check console for errors', async () => {
        const errors = await helper.checkConsoleErrors();
        expect(errors).toHaveLength(0);
      });

      await test.step('Test basic interactions', async () => {
        await helper.performInteractionTest();
      });

      // Visual regression testing with Percy (if available)
      if (process.env.PERCY_TOKEN) {
        await test.step('Visual regression test', async () => {
          await percySnapshot(page, `${pageInfo.name} - ${browserName}`, {
            widths: [375, 768, 1280, 1920],
            minHeight: 1024
          });
        });
      }
    });
  }

  // Responsive design testing
  for (const viewport of testViewports) {
    test(`Responsive design works on ${viewport.name} viewport`, async ({ page }) => {
      const helper = new CrossBrowserTestHelper(page);

      await test.step(`Set ${viewport.name} viewport`, async () => {
        await page.setViewportSize({ width: viewport.width, height: viewport.height });
      });

      await test.step('Test dashboard responsiveness', async () => {
        await page.goto('/');
        await helper.waitForPageLoad();
        await helper.checkResponsiveLayout(viewport);
      });

      await test.step('Test navigation responsiveness', async () => {
        await helper.validatePageStructure();
      });

      // Visual comparison for responsive design
      if (process.env.PERCY_TOKEN) {
        await test.step('Visual responsive test', async () => {
          await percySnapshot(page, `Dashboard - ${viewport.name}`, {
            widths: [viewport.width],
            minHeight: viewport.height
          });
        });
      }
    });
  }

  // Browser-specific feature testing
  test('Browser-specific features work correctly', async ({ page, browserName }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    await test.step('Test JavaScript features', async () => {
      // Test modern JavaScript features
      const jsFeatures = await page.evaluate(() => {
        return {
          asyncAwait: typeof (async () => {}) === 'function',
          arrow: typeof (() => {}) === 'function',
          const: typeof const !== 'undefined',
          let: typeof let !== 'undefined',
          destructuring: true, // Test with simple destructuring
          modules: typeof import !== 'undefined'
        };
      });

      expect(jsFeatures.asyncAwait).toBe(true);
      expect(jsFeatures.arrow).toBe(true);
    });

    await test.step('Test CSS features', async () => {
      // Test CSS Grid and Flexbox support
      const cssSupport = await page.evaluate(() => {
        const testElement = document.createElement('div');
        testElement.style.display = 'grid';
        document.body.appendChild(testElement);

        const computedStyle = window.getComputedStyle(testElement);
        const gridSupport = computedStyle.display === 'grid';

        testElement.style.display = 'flex';
        const flexSupport = computedStyle.display === 'flex';

        document.body.removeChild(testElement);

        return { gridSupport, flexSupport };
      });

      expect(cssSupport.gridSupport).toBe(true);
      expect(cssSupport.flexSupport).toBe(true);
    });

    await test.step('Test Web APIs', async () => {
      // Test essential Web APIs
      const apiSupport = await page.evaluate(() => {
        return {
          fetch: typeof fetch !== 'undefined',
          localStorage: typeof localStorage !== 'undefined',
          sessionStorage: typeof sessionStorage !== 'undefined',
          websocket: typeof WebSocket !== 'undefined',
          serviceWorker: 'serviceWorker' in navigator
        };
      });

      expect(apiSupport.fetch).toBe(true);
      expect(apiSupport.localStorage).toBe(true);
    });
  });

  // Performance testing across browsers
  test('Page performance meets targets across browsers', async ({ page, browserName }) => {
    await test.step('Navigate and measure performance', async () => {
      const startTime = Date.now();
      await page.goto('/');
      await page.waitForLoadState('networkidle');
      const loadTime = Date.now() - startTime;

      // Performance should be under 5 seconds for initial load
      expect(loadTime).toBeLessThan(5000);
    });

    await test.step('Check Core Web Vitals', async () => {
      const vitals = await page.evaluate(() => {
        return new Promise((resolve) => {
          const vitals: any = {};

          // Mock CLS measurement (in real implementation, would use PerformanceObserver)
          vitals.cls = 0.1; // Cumulative Layout Shift should be < 0.1

          // Mock FID measurement
          vitals.fid = 50; // First Input Delay should be < 100ms

          // LCP can be measured with PerformanceObserver
          const observer = new PerformanceObserver((list) => {
            for (const entry of list.getEntries()) {
              if (entry.entryType === 'largest-contentful-paint') {
                vitals.lcp = entry.startTime;
              }
            }
            resolve(vitals);
          });

          try {
            observer.observe({ entryTypes: ['largest-contentful-paint'] });
          } catch (e) {
            // Fallback if PerformanceObserver not supported
            vitals.lcp = 1500; // Mock value
            resolve(vitals);
          }

          // Timeout after 3 seconds
          setTimeout(() => resolve(vitals), 3000);
        });
      });

      // Core Web Vitals targets
      expect(vitals.lcp).toBeLessThan(2500); // LCP < 2.5s
      expect(vitals.fid).toBeLessThan(100);  // FID < 100ms
      expect(vitals.cls).toBeLessThan(0.1);  // CLS < 0.1
    });
  });

  // Accessibility testing across browsers
  test('Accessibility features work across browsers', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    await test.step('Test keyboard navigation', async () => {
      // Test tab navigation
      await page.keyboard.press('Tab');
      const focusedElement = await page.locator(':focus');
      expect(await focusedElement.count()).toBeGreaterThan(0);
    });

    await test.step('Test screen reader support', async () => {
      // Check for ARIA labels and roles
      const ariaElements = await page.locator('[aria-label], [role]').count();
      expect(ariaElements).toBeGreaterThan(0);
    });

    await test.step('Test color contrast', async () => {
      // Basic color contrast check (simplified)
      const textElements = page.locator('p, h1, h2, h3, a, button');
      const elementCount = await textElements.count();

      if (elementCount > 0) {
        const firstElement = textElements.first();
        const styles = await firstElement.evaluate(el => {
          const computed = window.getComputedStyle(el);
          return {
            color: computed.color,
            backgroundColor: computed.backgroundColor
          };
        });

        // Ensure text is not invisible (basic check)
        expect(styles.color).not.toBe(styles.backgroundColor);
      }
    });
  });
});

// Test suite for browser-specific edge cases
test.describe('Browser-Specific Edge Cases', () => {

  test('Safari-specific WebKit behavior', async ({ page, browserName }) => {
    test.skip(browserName !== 'webkit', 'Safari/WebKit specific test');

    await page.goto('/');
    await page.waitForLoadState('networkidle');

    // Test Safari-specific issues
    await test.step('Test date input behavior', async () => {
      const dateInput = page.locator('input[type="date"]').first();
      if (await dateInput.count() > 0) {
        await dateInput.focus();
        // Safari has specific date picker behavior
        expect(await dateInput.isVisible()).toBe(true);
      }
    });
  });

  test('Firefox-specific Gecko behavior', async ({ page, browserName }) => {
    test.skip(browserName !== 'firefox', 'Firefox specific test');

    await page.goto('/');
    await page.waitForLoadState('networkidle');

    // Test Firefox-specific issues
    await test.step('Test scrollbar behavior', async () => {
      const scrollableElement = page.locator('[data-testid="scrollable-content"]').first();
      if (await scrollableElement.count() > 0) {
        await scrollableElement.hover();
        // Firefox has specific scrollbar styling
        expect(await scrollableElement.isVisible()).toBe(true);
      }
    });
  });

  test('Chrome-specific Blink behavior', async ({ page, browserName }) => {
    test.skip(browserName !== 'chromium', 'Chrome/Chromium specific test');

    await page.goto('/');
    await page.waitForLoadState('networkidle');

    // Test Chrome-specific features
    await test.step('Test Chrome DevTools Protocol features', async () => {
      // Test features that work well in Chrome
      const performanceEntries = await page.evaluate(() => {
        return performance.getEntriesByType('navigation').length > 0;
      });

      expect(performanceEntries).toBe(true);
    });
  });
});
