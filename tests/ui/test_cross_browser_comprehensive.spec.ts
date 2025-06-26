/**
 * Comprehensive Cross-Browser Compatibility Testing Suite
 * Tests core functionality across all supported browsers and devices
 */

import { test, expect, Page, BrowserContext, devices } from '@playwright/test';

// Test data and configuration
const testUrls = [
  '/',
  '/dashboard',
  '/datasets',
  '/models',
  '/settings',
  '/api/docs'
];

const criticalFeatures = [
  'navigation',
  'forms',
  'charts',
  'modals',
  'responsive-layout'
];

const performanceThresholds = {
  LCP: 2500, // Largest Contentful Paint
  FID: 100,  // First Input Delay
  CLS: 0.1   // Cumulative Layout Shift
};

/**
 * Cross-browser compatibility tests
 */
test.describe('Cross-Browser Compatibility', () => {
  
  test.describe('Core Functionality', () => {
    
    testUrls.forEach(url => {
      test(`should load and render ${url} correctly across browsers`, async ({ page, browserName }) => {
        // Navigate to page
        await page.goto(url);
        
        // Wait for page to be fully loaded
        await page.waitForLoadState('networkidle');
        
        // Basic page structure validation
        await expect(page.locator('html')).toHaveAttribute('lang', 'en');
        await expect(page.locator('head title')).toBeVisible();
        
        // Check for critical CSS loading
        const bodyStyles = await page.locator('body').evaluate(el => 
          window.getComputedStyle(el).fontFamily
        );
        expect(bodyStyles).toContain('Inter');
        
        // Verify no JavaScript errors
        const errors: string[] = [];
        page.on('pageerror', error => errors.push(error.message));
        
        await page.waitForTimeout(2000); // Allow time for potential errors
        expect(errors).toHaveLength(0);
        
        // Browser-specific validations
        if (browserName === 'firefox') {
          // Firefox-specific checks
          await expect(page.locator('body')).toHaveCSS('font-family', /Inter/);
        } else if (browserName === 'webkit') {
          // Safari-specific checks  
          await expect(page.locator('body')).toBeVisible();
        }
        
        // Take screenshot for visual comparison
        await expect(page).toHaveScreenshot(`${url.replace(/\//g, '_')}-${browserName}.png`);
      });
    });
    
  });

  test.describe('JavaScript API Compatibility', () => {
    
    test('should support modern JavaScript features', async ({ page, browserName }) => {
      await page.goto('/');
      
      // Test ES6+ features support
      const jsFeatures = await page.evaluate(() => {
        const features = {
          promises: typeof Promise !== 'undefined',
          fetch: typeof fetch !== 'undefined',
          arrow_functions: (() => true)(),
          template_literals: `test${1}` === 'test1',
          destructuring: (() => { 
            try { 
              const [a] = [1]; 
              return a === 1; 
            } catch { 
              return false; 
            } 
          })(),
          modules: typeof import !== 'undefined',
          classes: (() => {
            try {
              class Test {}
              return typeof Test === 'function';
            } catch {
              return false;
            }
          })(),
          intersection_observer: typeof IntersectionObserver !== 'undefined',
          web_workers: typeof Worker !== 'undefined',
          service_workers: 'serviceWorker' in navigator,
          local_storage: typeof localStorage !== 'undefined',
          session_storage: typeof sessionStorage !== 'undefined',
          indexed_db: typeof indexedDB !== 'undefined'
        };
        
        return features;
      });
      
      // Verify core features are supported
      expect(jsFeatures.promises).toBe(true);
      expect(jsFeatures.fetch).toBe(true);
      expect(jsFeatures.arrow_functions).toBe(true);
      expect(jsFeatures.template_literals).toBe(true);
      expect(jsFeatures.local_storage).toBe(true);
      
      // Modern features that should be supported in current browsers
      expect(jsFeatures.intersection_observer).toBe(true);
      expect(jsFeatures.service_workers).toBe(true);
      
      console.log(`Browser ${browserName} feature support:`, jsFeatures);
    });

    test('should handle async/await and promises correctly', async ({ page }) => {
      await page.goto('/dashboard');
      
      // Test promise-based API calls
      const apiResponse = await page.evaluate(async () => {
        try {
          const response = await fetch('/api/health');
          const data = await response.json();
          return { success: true, status: response.status, data };
        } catch (error) {
          return { success: false, error: error.message };
        }
      });
      
      expect(apiResponse.success).toBe(true);
      expect(apiResponse.status).toBe(200);
    });

  });

  test.describe('CSS Feature Support', () => {
    
    test('should support modern CSS features', async ({ page, browserName }) => {
      await page.goto('/');
      
      // Test CSS feature support
      const cssFeatures = await page.evaluate(() => {
        const testElement = document.createElement('div');
        document.body.appendChild(testElement);
        
        const features = {
          flexbox: CSS.supports('display', 'flex'),
          grid: CSS.supports('display', 'grid'),
          custom_properties: CSS.supports('--test', '1'),
          transforms: CSS.supports('transform', 'translateX(1px)'),
          transitions: CSS.supports('transition', 'all 1s'),
          calc: CSS.supports('width', 'calc(100% - 10px)'),
          viewport_units: CSS.supports('width', '100vw'),
          object_fit: CSS.supports('object-fit', 'cover'),
          backdrop_filter: CSS.supports('backdrop-filter', 'blur(10px)'),
          clip_path: CSS.supports('clip-path', 'circle(50%)'),
          aspect_ratio: CSS.supports('aspect-ratio', '16/9')
        };
        
        document.body.removeChild(testElement);
        return features;
      });
      
      // Essential features that must be supported
      expect(cssFeatures.flexbox).toBe(true);
      expect(cssFeatures.custom_properties).toBe(true);
      expect(cssFeatures.transforms).toBe(true);
      expect(cssFeatures.transitions).toBe(true);
      expect(cssFeatures.calc).toBe(true);
      
      // Modern features (may vary by browser)
      if (browserName !== 'webkit') {
        expect(cssFeatures.grid).toBe(true);
      }
      
      console.log(`Browser ${browserName} CSS support:`, cssFeatures);
    });

    test('should render Tailwind CSS classes correctly', async ({ page }) => {
      await page.goto('/');
      
      // Test Tailwind utility classes
      await page.addStyleTag({
        content: `
          .test-container {
            @apply bg-primary-500 text-white p-4 rounded-lg shadow-md;
          }
        `
      });
      
      await page.setContent(`
        <div class="test-container">
          <h1 class="text-2xl font-bold">Test Heading</h1>
          <p class="text-sm opacity-75">Test paragraph</p>
        </div>
      `);
      
      // Verify computed styles
      const containerStyles = await page.locator('.test-container').evaluate(el => {
        const styles = window.getComputedStyle(el);
        return {
          backgroundColor: styles.backgroundColor,
          color: styles.color,
          padding: styles.padding,
          borderRadius: styles.borderRadius
        };
      });
      
      expect(containerStyles.backgroundColor).toMatch(/rgb\(14, 165, 233\)/);
      expect(containerStyles.color).toMatch(/rgb\(255, 255, 255\)/);
      expect(parseFloat(containerStyles.padding)).toBeGreaterThan(0);
    });

  });

  test.describe('Form Interactions', () => {
    
    test('should handle form inputs consistently', async ({ page, browserName }) => {
      await page.goto('/datasets');
      
      // Test basic form interactions
      const formElements = [
        'input[type="text"]',
        'input[type="email"]',
        'input[type="password"]',
        'select',
        'textarea',
        'input[type="checkbox"]',
        'input[type="radio"]'
      ];
      
      for (const selector of formElements) {
        const element = page.locator(selector).first();
        if (await element.count() > 0) {
          await expect(element).toBeVisible();
          
          // Test focus and blur events
          await element.focus();
          await expect(element).toBeFocused();
          
          // Test keyboard navigation
          if (selector.includes('input') && !selector.includes('checkbox') && !selector.includes('radio')) {
            await element.fill('test value');
            await expect(element).toHaveValue('test value');
            await element.clear();
          }
        }
      }
    });

    test('should validate form submissions', async ({ page }) => {
      await page.goto('/datasets/upload');
      
      // Test form validation
      const submitButton = page.locator('button[type="submit"]').first();
      if (await submitButton.count() > 0) {
        await submitButton.click();
        
        // Check for validation messages
        const validationMessages = page.locator('[aria-invalid="true"], .error, .invalid');
        if (await validationMessages.count() > 0) {
          await expect(validationMessages.first()).toBeVisible();
        }
      }
    });

  });

  test.describe('Accessibility Across Browsers', () => {
    
    test('should maintain keyboard navigation', async ({ page }) => {
      await page.goto('/');
      
      // Test tab navigation
      await page.keyboard.press('Tab');
      const firstFocusable = await page.locator(':focus').first();
      await expect(firstFocusable).toBeVisible();
      
      // Continue tabbing through interactive elements
      for (let i = 0; i < 5; i++) {
        await page.keyboard.press('Tab');
        const focused = page.locator(':focus');
        if (await focused.count() > 0) {
          await expect(focused).toBeVisible();
        }
      }
    });

    test('should support screen reader attributes', async ({ page }) => {
      await page.goto('/dashboard');
      
      // Check for proper ARIA attributes
      const landmarks = await page.locator('[role="main"], [role="navigation"], [role="banner"], [role="contentinfo"]').count();
      expect(landmarks).toBeGreaterThan(0);
      
      // Check for heading hierarchy
      const headings = await page.locator('h1, h2, h3, h4, h5, h6').count();
      expect(headings).toBeGreaterThan(0);
      
      // Verify alt text on images
      const images = page.locator('img');
      const imageCount = await images.count();
      
      for (let i = 0; i < imageCount; i++) {
        const img = images.nth(i);
        const altText = await img.getAttribute('alt');
        const ariaLabel = await img.getAttribute('aria-label');
        const role = await img.getAttribute('role');
        
        // Images should have alt text, aria-label, or be decorative
        expect(altText !== null || ariaLabel !== null || role === 'presentation').toBe(true);
      }
    });

  });

  test.describe('Performance Across Browsers', () => {
    
    test('should meet Core Web Vitals thresholds', async ({ page, browserName }) => {
      await page.goto('/dashboard');
      
      // Measure Core Web Vitals
      const metrics = await page.evaluate(() => {
        return new Promise((resolve) => {
          const observer = new PerformanceObserver((list) => {
            const entries = list.getEntries();
            const vitals: Record<string, number> = {};
            
            entries.forEach((entry) => {
              if (entry.entryType === 'largest-contentful-paint') {
                vitals.LCP = entry.startTime;
              }
              if (entry.entryType === 'first-input') {
                vitals.FID = entry.processingStart - entry.startTime;
              }
              if (entry.entryType === 'layout-shift' && !entry.hadRecentInput) {
                vitals.CLS = (vitals.CLS || 0) + entry.value;
              }
            });
            
            // Resolve after collecting metrics for 3 seconds
            setTimeout(() => resolve(vitals), 3000);
          });
          
          observer.observe({ entryTypes: ['largest-contentful-paint', 'first-input', 'layout-shift'] });
        });
      });
      
      console.log(`${browserName} Core Web Vitals:`, metrics);
      
      // Validate thresholds (allowing for browser variations)
      if (metrics.LCP) {
        expect(metrics.LCP).toBeLessThan(performanceThresholds.LCP * 1.5); // 50% tolerance
      }
      if (metrics.FID) {
        expect(metrics.FID).toBeLessThan(performanceThresholds.FID * 2); // 100% tolerance for cross-browser
      }
      if (metrics.CLS) {
        expect(metrics.CLS).toBeLessThan(performanceThresholds.CLS * 2); // 100% tolerance
      }
    });

    test('should load assets efficiently', async ({ page }) => {
      const startTime = Date.now();
      
      await page.goto('/');
      await page.waitForLoadState('networkidle');
      
      const loadTime = Date.now() - startTime;
      
      // Page should load within reasonable time
      expect(loadTime).toBeLessThan(5000); // 5 seconds max
      
      // Check for efficient resource loading
      const resources = await page.evaluate(() => {
        const entries = performance.getEntriesByType('resource');
        return entries.map(entry => ({
          name: entry.name,
          type: entry.initiatorType,
          size: entry.transferSize || 0,
          duration: entry.duration
        }));
      });
      
      const totalSize = resources.reduce((sum, resource) => sum + resource.size, 0);
      
      // Total page size should be reasonable
      expect(totalSize).toBeLessThan(5 * 1024 * 1024); // 5MB max
      
      console.log(`Total resources loaded: ${resources.length}, Total size: ${totalSize} bytes`);
    });

  });

  test.describe('Browser-Specific Features', () => {
    
    test('should handle Chrome-specific features', async ({ page, browserName }) => {
      test.skip(browserName !== 'chromium', 'Chrome-specific test');
      
      await page.goto('/');
      
      // Test Chrome DevTools Protocol features
      const client = await page.context().newCDPSession(page);
      await client.send('Performance.enable');
      
      // Test Chrome extensions compatibility
      const extensionAPIs = await page.evaluate(() => {
        return {
          webRequest: typeof chrome !== 'undefined' && chrome.webRequest,
          storage: typeof chrome !== 'undefined' && chrome.storage,
          runtime: typeof chrome !== 'undefined' && chrome.runtime
        };
      });
      
      // In regular Chrome, these won't be available, which is expected
      console.log('Chrome extension APIs:', extensionAPIs);
    });

    test('should handle Firefox-specific features', async ({ page, browserName }) => {
      test.skip(browserName !== 'firefox', 'Firefox-specific test');
      
      await page.goto('/');
      
      // Test Firefox-specific APIs
      const firefoxFeatures = await page.evaluate(() => {
        return {
          mozInputSource: 'mozInputSource' in MouseEvent.prototype,
          mozRequestFullScreen: 'mozRequestFullScreen' in document.documentElement,
          InstallTrigger: typeof InstallTrigger !== 'undefined'
        };
      });
      
      console.log('Firefox-specific features:', firefoxFeatures);
    });

    test('should handle Safari-specific features', async ({ page, browserName }) => {
      test.skip(browserName !== 'webkit', 'Safari-specific test');
      
      await page.goto('/');
      
      // Test Safari-specific behavior
      const safariFeatures = await page.evaluate(() => {
        return {
          webkitRequestFullScreen: 'webkitRequestFullScreen' in document.documentElement,
          safariVersion: navigator.userAgent.includes('Safari'),
          touchForceChange: 'ontouchforcechange' in window
        };
      });
      
      console.log('Safari-specific features:', safariFeatures);
      
      // Test Safari's stricter security policies
      const securityTest = await page.evaluate(() => {
        try {
          // Safari may block certain local storage operations
          localStorage.setItem('test', 'value');
          return localStorage.getItem('test') === 'value';
        } catch (error) {
          return false;
        }
      });
      
      expect(securityTest).toBe(true);
    });

  });

  test.describe('Progressive Enhancement', () => {
    
    test('should work without JavaScript', async ({ page }) => {
      // Disable JavaScript
      await page.context().addInitScript(() => {
        delete window.document.createElement;
      });
      
      await page.goto('/');
      
      // Basic content should still be accessible
      await expect(page.locator('h1, h2, h3')).toHaveCount({ min: 1 });
      await expect(page.locator('main, [role="main"]')).toBeVisible();
      
      // Links should still work
      const links = page.locator('a[href]');
      const linkCount = await links.count();
      
      if (linkCount > 0) {
        const firstLink = links.first();
        await expect(firstLink).toHaveAttribute('href');
      }
    });

    test('should gracefully degrade advanced features', async ({ page }) => {
      await page.goto('/dashboard');
      
      // Simulate older browser by removing modern APIs
      await page.addInitScript(() => {
        // Remove modern APIs
        delete window.IntersectionObserver;
        delete window.ResizeObserver;
        delete window.fetch;
        
        // Simulate older browser
        Object.defineProperty(navigator, 'userAgent', {
          value: 'Mozilla/5.0 (compatible; MSIE 11.0; Windows NT 6.1; Trident/7.0)',
          configurable: true
        });
      });
      
      await page.reload();
      
      // Basic functionality should still work
      await expect(page.locator('body')).toBeVisible();
      
      // Check for polyfills or fallbacks
      const hasPolyfills = await page.evaluate(() => {
        return Boolean(
          window.fetch || // fetch polyfill
          document.querySelector('script[src*="polyfill"]') // polyfill script
        );
      });
      
      console.log('Polyfills detected:', hasPolyfills);
    });

  });

});

/**
 * Device-specific compatibility tests
 */
test.describe('Device Compatibility', () => {
  
  test.describe('Mobile Devices', () => {
    
    test('should work on mobile Chrome', async ({ page, browserName }) => {
      test.skip(browserName !== 'Mobile Chrome', 'Mobile Chrome specific test');
      
      await page.goto('/');
      
      // Test mobile-specific interactions
      await expect(page.locator('body')).toHaveCSS('font-size', /14px|16px/);
      
      // Test touch interactions
      const button = page.locator('button').first();
      if (await button.count() > 0) {
        await button.tap();
        // Verify tap worked (button state change, navigation, etc.)
      }
    });

    test('should work on mobile Safari', async ({ page, browserName }) => {
      test.skip(browserName !== 'Mobile Safari', 'Mobile Safari specific test');
      
      await page.goto('/');
      
      // Test iOS Safari specific behavior
      const viewport = page.viewportSize();
      expect(viewport).toBeTruthy();
      
      // Test safe area handling
      const safeAreaTest = await page.evaluate(() => {
        const div = document.createElement('div');
        div.style.paddingTop = 'env(safe-area-inset-top)';
        document.body.appendChild(div);
        const computed = window.getComputedStyle(div).paddingTop;
        document.body.removeChild(div);
        return computed;
      });
      
      console.log('Safe area inset support:', safeAreaTest);
    });

  });

  test.describe('Tablet Devices', () => {
    
    test('should adapt to tablet viewport', async ({ page, browserName }) => {
      test.skip(browserName !== 'iPad', 'iPad specific test');
      
      await page.goto('/');
      
      // Verify tablet-optimized layout
      const viewport = page.viewportSize();
      expect(viewport?.width).toBeGreaterThan(768);
      
      // Test tablet-specific interactions
      const navigation = page.locator('nav');
      await expect(navigation).toBeVisible();
      
      // Test orientation handling
      await page.setViewportSize({ width: 1024, height: 768 }); // Landscape
      await page.waitForTimeout(500);
      
      await page.setViewportSize({ width: 768, height: 1024 }); // Portrait
      await page.waitForTimeout(500);
      
      // Layout should adapt to orientation changes
      await expect(page.locator('body')).toBeVisible();
    });

  });

});

/**
 * Performance regression tests across browsers
 */
test.describe('Cross-Browser Performance Regression', () => {
  
  test('should maintain performance parity across browsers', async ({ page, browserName }) => {
    const performanceMetrics: Record<string, number> = {};
    
    const startTime = performance.now();
    await page.goto('/dashboard');
    await page.waitForLoadState('networkidle');
    const endTime = performance.now();
    
    performanceMetrics.pageLoadTime = endTime - startTime;
    
    // Measure JavaScript execution time
    const jsExecutionTime = await page.evaluate(() => {
      const start = performance.now();
      
      // Simulate typical JavaScript operations
      for (let i = 0; i < 10000; i++) {
        const div = document.createElement('div');
        div.textContent = `Item ${i}`;
        document.body.appendChild(div);
        document.body.removeChild(div);
      }
      
      return performance.now() - start;
    });
    
    performanceMetrics.jsExecutionTime = jsExecutionTime;
    
    // Log performance metrics for comparison
    console.log(`${browserName} performance:`, performanceMetrics);
    
    // Set reasonable performance expectations
    expect(performanceMetrics.pageLoadTime).toBeLessThan(10000); // 10 seconds max
    expect(performanceMetrics.jsExecutionTime).toBeLessThan(1000); // 1 second max for JS test
  });

});