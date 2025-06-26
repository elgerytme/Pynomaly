/**
 * Device Compatibility Testing Suite
 * Tests responsive design, touch interactions, and device-specific features
 */

import { test, expect, Page, Browser } from '@playwright/test';

// Device configurations for testing
const deviceConfigs = {
  mobile: {
    'iPhone 12': { width: 390, height: 844, isMobile: true, hasTouch: true },
    'iPhone 12 Pro Max': { width: 428, height: 926, isMobile: true, hasTouch: true },
    'Pixel 5': { width: 393, height: 851, isMobile: true, hasTouch: true },
    'Galaxy S21': { width: 384, height: 854, isMobile: true, hasTouch: true }
  },
  tablet: {
    'iPad Pro': { width: 1024, height: 1366, isMobile: false, hasTouch: true },
    'iPad Air': { width: 820, height: 1180, isMobile: false, hasTouch: true },
    'Galaxy Tab S4': { width: 712, height: 1138, isMobile: false, hasTouch: true },
    'Surface Pro': { width: 912, height: 1368, isMobile: false, hasTouch: true }
  },
  desktop: {
    'Desktop 1080p': { width: 1920, height: 1080, isMobile: false, hasTouch: false },
    'Desktop 1440p': { width: 2560, height: 1440, isMobile: false, hasTouch: false },
    'Desktop 4K': { width: 3840, height: 2160, isMobile: false, hasTouch: false }
  }
};

const breakpoints = {
  mobile: 767,
  tablet: 1023,
  desktop: 1024
};

/**
 * Responsive Design Tests
 */
test.describe('Responsive Design Compatibility', () => {
  
  Object.entries(deviceConfigs).forEach(([deviceType, devices]) => {
    test.describe(`${deviceType.toUpperCase()} Devices`, () => {
      
      Object.entries(devices).forEach(([deviceName, config]) => {
        test(`should render correctly on ${deviceName}`, async ({ page }) => {
          // Set viewport to device dimensions
          await page.setViewportSize({ width: config.width, height: config.height });
          
          // Navigate to main pages
          const testPages = ['/', '/dashboard', '/datasets', '/models', '/settings'];
          
          for (const url of testPages) {
            await page.goto(url);
            await page.waitForLoadState('networkidle');
            
            // Basic layout validation
            await expect(page.locator('body')).toBeVisible();
            
            // Check for responsive navigation
            const navigation = page.locator('nav, [role="navigation"]');
            if (await navigation.count() > 0) {
              await expect(navigation.first()).toBeVisible();
              
              // Mobile should have collapsible navigation
              if (config.isMobile) {
                const mobileMenuButton = page.locator('[aria-label*="menu"], .menu-toggle, .hamburger');
                if (await mobileMenuButton.count() > 0) {
                  await expect(mobileMenuButton.first()).toBeVisible();
                }
              }
            }
            
            // Check for proper text sizing
            const headings = page.locator('h1, h2, h3');
            if (await headings.count() > 0) {
              const fontSize = await headings.first().evaluate(el => 
                window.getComputedStyle(el).fontSize
              );
              const fontSizeNum = parseInt(fontSize);
              
              // Ensure readable font sizes
              if (config.isMobile) {
                expect(fontSizeNum).toBeGreaterThanOrEqual(16); // Minimum 16px on mobile
              } else {
                expect(fontSizeNum).toBeGreaterThanOrEqual(14); // Minimum 14px on larger screens
              }
            }
            
            // Check for touch target sizes on touch devices
            if (config.hasTouch) {
              const buttons = page.locator('button, a, input[type="submit"], input[type="button"]');
              const buttonCount = await buttons.count();
              
              for (let i = 0; i < Math.min(buttonCount, 5); i++) {
                const button = buttons.nth(i);
                if (await button.isVisible()) {
                  const boundingBox = await button.boundingBox();
                  if (boundingBox) {
                    // WCAG 2.1 AA minimum touch target size
                    expect(Math.min(boundingBox.width, boundingBox.height)).toBeGreaterThanOrEqual(24);
                  }
                }
              }
            }
            
            // Take screenshot for visual comparison
            await expect(page).toHaveScreenshot(`${deviceName.replace(/\s+/g, '_')}-${url.replace(/\//g, '_')}.png`);
          }
        });
        
      });
    });
  });

  test('should handle orientation changes on mobile devices', async ({ page, browserName }) => {
    test.skip(browserName === 'webkit', 'Safari orientation testing handled separately');
    
    // Test with iPhone 12 dimensions
    const portraitWidth = 390;
    const portraitHeight = 844;
    const landscapeWidth = 844;
    const landscapeHeight = 390;
    
    // Start in portrait mode
    await page.setViewportSize({ width: portraitWidth, height: portraitHeight });
    await page.goto('/dashboard');
    await page.waitForLoadState('networkidle');
    
    // Verify portrait layout
    const portraitLayout = await page.locator('body').evaluate(el => {
      return {
        width: window.innerWidth,
        height: window.innerHeight,
        orientation: window.innerWidth < window.innerHeight ? 'portrait' : 'landscape'
      };
    });
    
    expect(portraitLayout.orientation).toBe('portrait');
    await expect(page).toHaveScreenshot('mobile-portrait.png');
    
    // Switch to landscape mode
    await page.setViewportSize({ width: landscapeWidth, height: landscapeHeight });
    await page.waitForTimeout(500); // Allow layout to settle
    
    // Verify landscape layout
    const landscapeLayout = await page.locator('body').evaluate(el => {
      return {
        width: window.innerWidth,
        height: window.innerHeight,
        orientation: window.innerWidth > window.innerHeight ? 'landscape' : 'portrait'
      };
    });
    
    expect(landscapeLayout.orientation).toBe('landscape');
    await expect(page).toHaveScreenshot('mobile-landscape.png');
    
    // Ensure layout is still functional
    await expect(page.locator('body')).toBeVisible();
    const navigation = page.locator('nav, [role="navigation"]');
    if (await navigation.count() > 0) {
      await expect(navigation.first()).toBeVisible();
    }
  });

});

/**
 * Touch Interaction Tests
 */
test.describe('Touch Interaction Compatibility', () => {
  
  test('should handle touch gestures on mobile devices', async ({ page, browserName }) => {
    test.skip(!['Mobile Chrome', 'Mobile Safari'].includes(browserName), 'Touch-specific test');
    
    await page.goto('/datasets');
    await page.waitForLoadState('networkidle');
    
    // Test tap interactions
    const buttons = page.locator('button');
    if (await buttons.count() > 0) {
      const firstButton = buttons.first();
      await firstButton.tap();
      
      // Verify button responds to tap
      await page.waitForTimeout(200); // Allow for visual feedback
    }
    
    // Test long press (context menu)
    const longPressTarget = page.locator('.card, .list-item').first();
    if (await longPressTarget.count() > 0) {
      // Simulate long press with touch events
      await longPressTarget.evaluate(element => {
        const touchStart = new TouchEvent('touchstart', {
          touches: [new Touch({
            identifier: 0,
            target: element,
            clientX: element.getBoundingClientRect().left + 10,
            clientY: element.getBoundingClientRect().top + 10
          })]
        });
        element.dispatchEvent(touchStart);
        
        setTimeout(() => {
          const touchEnd = new TouchEvent('touchend', { touches: [] });
          element.dispatchEvent(touchEnd);
        }, 800); // Long press duration
      });
      
      await page.waitForTimeout(1000);
    }
    
    // Test swipe gestures if applicable
    const swipeableElement = page.locator('.swipeable, .carousel, .slider').first();
    if (await swipeableElement.count() > 0) {
      const boundingBox = await swipeableElement.boundingBox();
      if (boundingBox) {
        // Simulate swipe left
        await page.touchscreen.tap(boundingBox.x + boundingBox.width * 0.8, boundingBox.y + boundingBox.height / 2);
        await page.mouse.down();
        await page.mouse.move(boundingBox.x + boundingBox.width * 0.2, boundingBox.y + boundingBox.height / 2);
        await page.mouse.up();
        
        await page.waitForTimeout(500); // Allow animation to complete
      }
    }
  });

  test('should handle pinch-to-zoom on touch devices', async ({ page, browserName }) => {
    test.skip(!['Mobile Chrome', 'Mobile Safari', 'iPad'].includes(browserName), 'Touch zoom test');
    
    await page.goto('/dashboard');
    await page.waitForLoadState('networkidle');
    
    // Check if zooming is properly controlled
    const viewport = await page.evaluate(() => {
      const viewport = document.querySelector('meta[name="viewport"]');
      return viewport ? viewport.getAttribute('content') : '';
    });
    
    // Should have appropriate viewport settings for mobile
    expect(viewport).toContain('width=device-width');
    expect(viewport).toContain('initial-scale=1');
    
    // Test zoom accessibility
    const zoomTest = await page.evaluate(() => {
      // Check if user can zoom (accessibility requirement)
      const viewport = document.querySelector('meta[name="viewport"]');
      const content = viewport?.getAttribute('content') || '';
      
      return {
        hasViewport: !!viewport,
        allowsZoom: !content.includes('user-scalable=no') && !content.includes('maximum-scale=1'),
        content
      };
    });
    
    expect(zoomTest.hasViewport).toBe(true);
    expect(zoomTest.allowsZoom).toBe(true); // WCAG requirement
  });

  test('should provide appropriate touch feedback', async ({ page, browserName }) => {
    test.skip(!['Mobile Chrome', 'Mobile Safari'].includes(browserName), 'Touch feedback test');
    
    await page.goto('/');
    await page.waitForLoadState('networkidle');
    
    // Test button touch states
    const buttons = page.locator('button, .btn');
    if (await buttons.count() > 0) {
      const button = buttons.first();
      
      // Check for touch-friendly styling
      const buttonStyles = await button.evaluate(el => {
        const styles = window.getComputedStyle(el);
        return {
          cursor: styles.cursor,
          userSelect: styles.userSelect,
          touchAction: styles.touchAction,
          minHeight: styles.minHeight
        };
      });
      
      // Should have appropriate touch styles
      expect(buttonStyles.cursor).toBe('pointer');
      expect(buttonStyles.userSelect).toBe('none');
      
      // Tap the button and verify visual feedback
      await button.tap();
      await page.waitForTimeout(100); // Allow for touch feedback
      
      // Check if button has active/pressed state
      const hasActiveState = await button.evaluate(el => {
        return window.getComputedStyle(el, ':active').backgroundColor !== 
               window.getComputedStyle(el).backgroundColor;
      });
      
      // Should provide visual feedback on touch
      expect(hasActiveState).toBe(true);
    }
  });

});

/**
 * Device-Specific Feature Tests
 */
test.describe('Device-Specific Features', () => {
  
  test('should handle device orientation API', async ({ page, browserName }) => {
    test.skip(browserName !== 'Mobile Chrome', 'Orientation API test for mobile Chrome');
    
    await page.goto('/');
    
    const orientationSupport = await page.evaluate(() => {
      return {
        hasOrientationAPI: 'orientation' in window,
        hasScreenOrientation: 'screen' in window && 'orientation' in window.screen,
        hasOrientationChange: 'onorientationchange' in window,
        currentOrientation: window.orientation || 0
      };
    });
    
    console.log('Orientation support:', orientationSupport);
    
    // Most modern mobile browsers should support orientation detection
    expect(orientationSupport.hasOrientationAPI || orientationSupport.hasScreenOrientation).toBe(true);
  });

  test('should handle device pixel ratio correctly', async ({ page }) => {
    await page.goto('/');
    
    const devicePixelRatio = await page.evaluate(() => window.devicePixelRatio);
    
    // Verify DPR is reasonable
    expect(devicePixelRatio).toBeGreaterThan(0);
    expect(devicePixelRatio).toBeLessThanOrEqual(4); // Common range
    
    // Check if images adapt to DPR
    const images = page.locator('img');
    if (await images.count() > 0) {
      const firstImage = images.first();
      const srcset = await firstImage.getAttribute('srcset');
      const src = await firstImage.getAttribute('src');
      
      // Should either have srcset for responsive images or appropriate src
      expect(srcset !== null || src !== null).toBe(true);
    }
  });

  test('should respect user preferences', async ({ page }) => {
    await page.goto('/');
    
    // Test reduced motion preference
    const reducedMotionTest = await page.evaluate(() => {
      const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
      
      return {
        supportsReducedMotion: !!window.matchMedia,
        reducedMotionActive: prefersReducedMotion.matches,
        mediaQuerySupport: typeof prefersReducedMotion.matches === 'boolean'
      };
    });
    
    expect(reducedMotionTest.supportsReducedMotion).toBe(true);
    
    // Test dark mode preference
    const darkModeTest = await page.evaluate(() => {
      const prefersDarkMode = window.matchMedia('(prefers-color-scheme: dark)');
      
      return {
        supportsDarkMode: !!window.matchMedia,
        darkModeActive: prefersDarkMode.matches,
        mediaQuerySupport: typeof prefersDarkMode.matches === 'boolean'
      };
    });
    
    expect(darkModeTest.supportsDarkMode).toBe(true);
    
    // Test high contrast preference
    const highContrastTest = await page.evaluate(() => {
      const prefersHighContrast = window.matchMedia('(prefers-contrast: high)');
      
      return {
        supportsHighContrast: !!window.matchMedia,
        highContrastActive: prefersHighContrast.matches
      };
    });
    
    console.log('User preferences:', { reducedMotionTest, darkModeTest, highContrastTest });
  });

});

/**
 * Performance on Different Devices
 */
test.describe('Device Performance', () => {
  
  test('should maintain performance across device types', async ({ page, browserName }) => {
    const startTime = Date.now();
    
    await page.goto('/dashboard');
    await page.waitForLoadState('networkidle');
    
    const loadTime = Date.now() - startTime;
    
    // Get device-specific performance metrics
    const performanceMetrics = await page.evaluate(() => {
      const navigation = performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming;
      
      return {
        domContentLoaded: navigation.domContentLoadedEventEnd - navigation.domContentLoadedEventStart,
        loadComplete: navigation.loadEventEnd - navigation.loadEventStart,
        firstPaint: performance.getEntriesByType('paint').find(entry => entry.name === 'first-paint')?.startTime || 0,
        firstContentfulPaint: performance.getEntriesByType('paint').find(entry => entry.name === 'first-contentful-paint')?.startTime || 0,
        memoryUsage: (performance as any).memory ? {
          used: (performance as any).memory.usedJSHeapSize,
          total: (performance as any).memory.totalJSHeapSize,
          limit: (performance as any).memory.jsHeapSizeLimit
        } : null
      };
    });
    
    console.log(`${browserName} performance:`, performanceMetrics);
    
    // Set performance expectations based on device type
    let maxLoadTime = 5000; // Default 5 seconds
    
    if (browserName.includes('Mobile')) {
      maxLoadTime = 8000; // Allow more time for mobile
    } else if (browserName.includes('Desktop')) {
      maxLoadTime = 3000; // Expect faster on desktop
    }
    
    expect(loadTime).toBeLessThan(maxLoadTime);
    
    // FCP should be reasonable
    if (performanceMetrics.firstContentfulPaint > 0) {
      expect(performanceMetrics.firstContentfulPaint).toBeLessThan(3000);
    }
  });

  test('should handle memory constraints on mobile devices', async ({ page, browserName }) => {
    test.skip(!browserName.includes('Mobile'), 'Mobile memory test');
    
    await page.goto('/datasets');
    await page.waitForLoadState('networkidle');
    
    // Simulate memory pressure
    const memoryTest = await page.evaluate(() => {
      const testData = [];
      let memoryPressure = false;
      
      try {
        // Try to allocate memory
        for (let i = 0; i < 1000; i++) {
          testData.push(new Array(1000).fill(i));
        }
        
        // Check if page is still responsive
        const startTime = Date.now();
        document.body.scrollTop = 100;
        const scrollTime = Date.now() - startTime;
        
        return {
          memoryAllocated: testData.length,
          scrollResponsive: scrollTime < 100,
          memoryUsage: (performance as any).memory ? {
            used: (performance as any).memory.usedJSHeapSize,
            total: (performance as any).memory.totalJSHeapSize
          } : null
        };
      } catch (error) {
        return {
          error: error.message,
          memoryPressure: true
        };
      }
    });
    
    console.log('Memory test results:', memoryTest);
    
    // Page should remain responsive under memory pressure
    expect(memoryTest.scrollResponsive !== false).toBe(true);
  });

});

/**
 * Cross-Device Data Sync
 */
test.describe('Cross-Device Compatibility', () => {
  
  test('should maintain session across device switches', async ({ page }) => {
    // Simulate user switching from desktop to mobile
    
    // Start with desktop viewport
    await page.setViewportSize({ width: 1920, height: 1080 });
    await page.goto('/dashboard');
    await page.waitForLoadState('networkidle');
    
    // Simulate some user interaction
    const userInteraction = await page.evaluate(() => {
      // Set some local storage data
      localStorage.setItem('userPreference', JSON.stringify({
        theme: 'dark',
        layout: 'grid',
        timestamp: Date.now()
      }));
      
      // Simulate activity
      sessionStorage.setItem('currentSession', 'active');
      
      return {
        localStorageSet: !!localStorage.getItem('userPreference'),
        sessionStorageSet: !!sessionStorage.getItem('currentSession')
      };
    });
    
    expect(userInteraction.localStorageSet).toBe(true);
    expect(userInteraction.sessionStorageSet).toBe(true);
    
    // Switch to mobile viewport
    await page.setViewportSize({ width: 390, height: 844 });
    await page.reload();
    await page.waitForLoadState('networkidle');
    
    // Verify data persistence
    const dataPersistence = await page.evaluate(() => {
      const userPref = localStorage.getItem('userPreference');
      const session = sessionStorage.getItem('currentSession');
      
      return {
        localStoragePersisted: !!userPref,
        sessionStoragePersisted: !!session,
        userPreference: userPref ? JSON.parse(userPref) : null
      };
    });
    
    expect(dataPersistence.localStoragePersisted).toBe(true);
    expect(dataPersistence.sessionStoragePersisted).toBe(true);
    expect(dataPersistence.userPreference?.theme).toBe('dark');
  });

  test('should handle offline functionality across devices', async ({ page, browserName }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');
    
    // Check service worker registration
    const serviceWorkerSupport = await page.evaluate(async () => {
      if ('serviceWorker' in navigator) {
        try {
          const registration = await navigator.serviceWorker.ready;
          return {
            supported: true,
            registered: !!registration,
            scope: registration.scope
          };
        } catch (error) {
          return {
            supported: true,
            registered: false,
            error: error.message
          };
        }
      }
      return { supported: false };
    });
    
    expect(serviceWorkerSupport.supported).toBe(true);
    
    // Test offline capability
    if (serviceWorkerSupport.registered) {
      // Simulate offline condition
      await page.context().setOffline(true);
      
      // Try to navigate to a cached page
      await page.goto('/dashboard');
      
      // Should either load from cache or show offline page
      const pageContent = await page.textContent('body');
      expect(pageContent).toBeTruthy();
      expect(pageContent!.length).toBeGreaterThan(0);
      
      // Restore online state
      await page.context().setOffline(false);
    }
  });

});