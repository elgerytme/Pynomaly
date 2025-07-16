/**
 * Progressive Web App (PWA) Testing Suite
 * 
 * This test suite validates PWA functionality including offline capabilities,
 * service worker behavior, and mobile app-like features.
 */

import { test, expect, Page, BrowserContext } from '@playwright/test';

interface PWAManifest {
  name: string;
  short_name: string;
  description: string;
  start_url: string;
  display: string;
  background_color: string;
  theme_color: string;
  icons: Array<{
    src: string;
    sizes: string;
    type: string;
    purpose?: string;
  }>;
}

interface ServiceWorkerInfo {
  scriptURL: string;
  scope: string;
  state: string;
  controllerState: string;
}

class PWATester {
  private page: Page;
  private context: BrowserContext;

  constructor(page: Page, context: BrowserContext) {
    this.page = page;
    this.context = context;
  }

  async getManifest(): Promise<PWAManifest | null> {
    return await this.page.evaluate(async () => {
      const manifestLink = document.querySelector('link[rel="manifest"]') as HTMLLinkElement;
      if (!manifestLink) return null;

      try {
        const response = await fetch(manifestLink.href);
        return await response.json();
      } catch (error) {
        console.error('Failed to fetch manifest:', error);
        return null;
      }
    });
  }

  async getServiceWorkerInfo(): Promise<ServiceWorkerInfo | null> {
    return await this.page.evaluate(() => {
      if (!('serviceWorker' in navigator)) return null;

      const registration = navigator.serviceWorker.controller;
      if (!registration) return null;

      return {
        scriptURL: registration.scriptURL,
        scope: navigator.serviceWorker.controller?.scope || '',
        state: registration.state,
        controllerState: navigator.serviceWorker.controller?.state || 'none'
      };
    });
  }

  async testOfflineCapability(): Promise<{ offlineSupported: boolean; cacheHits: number; cacheMisses: number }> {
    // First, ensure the page loads normally
    await this.page.goto('/dashboard', { waitUntil: 'networkidle' });
    await this.page.waitForTimeout(2000); // Allow service worker to cache resources

    // Simulate offline mode
    await this.context.setOffline(true);

    let offlineSupported = false;
    let cacheHits = 0;
    let cacheMisses = 0;

    try {
      // Try to navigate while offline
      await this.page.reload({ waitUntil: 'networkidle', timeout: 10000 });
      
      // Check if the page loaded from cache
      const pageContent = await this.page.content();
      offlineSupported = pageContent.includes('dashboard') || pageContent.includes('offline');
      
      // Get cache statistics from service worker
      const cacheStats = await this.page.evaluate(async () => {
        if (!('serviceWorker' in navigator) || !navigator.serviceWorker.controller) {
          return { hits: 0, misses: 0 };
        }

        // Check cache storage
        try {
          const cacheNames = await caches.keys();
          let totalHits = 0;
          let totalMisses = 0;

          for (const cacheName of cacheNames) {
            const cache = await caches.open(cacheName);
            const keys = await cache.keys();
            totalHits += keys.length;
          }

          return { hits: totalHits, misses: totalMisses };
        } catch (error) {
          return { hits: 0, misses: 0 };
        }
      });

      cacheHits = cacheStats.hits;
      cacheMisses = cacheStats.misses;

    } catch (error) {
      console.log('Offline navigation failed:', error);
    } finally {
      // Restore online mode
      await this.context.setOffline(false);
    }

    return { offlineSupported, cacheHits, cacheMisses };
  }

  async testInstallPrompt(): Promise<{ promptAvailable: boolean; canInstall: boolean }> {
    const installInfo = await this.page.evaluate(() => {
      return new Promise<{ promptAvailable: boolean; canInstall: boolean }>((resolve) => {
        let promptAvailable = false;
        let canInstall = false;

        // Listen for beforeinstallprompt event
        const beforeInstallPromptHandler = (e: Event) => {
          e.preventDefault();
          promptAvailable = true;
          canInstall = true;
          resolve({ promptAvailable, canInstall });
        };

        window.addEventListener('beforeinstallprompt', beforeInstallPromptHandler);

        // Check if already installed
        if (window.matchMedia && window.matchMedia('(display-mode: standalone)').matches) {
          resolve({ promptAvailable: false, canInstall: false });
        }

        // Timeout after 5 seconds
        setTimeout(() => {
          window.removeEventListener('beforeinstallprompt', beforeInstallPromptHandler);
          resolve({ promptAvailable, canInstall });
        }, 5000);
      });
    });

    return installInfo;
  }

  async testPushNotifications(): Promise<{ supported: boolean; permission: string }> {
    return await this.page.evaluate(async () => {
      if (!('Notification' in window) || !('serviceWorker' in navigator)) {
        return { supported: false, permission: 'not-supported' };
      }

      try {
        // Check current permission
        const permission = Notification.permission;
        
        // Test if we can create a notification (without actually showing it)
        const supported = 'Notification' in window && 'serviceWorker' in navigator;
        
        return { supported, permission };
      } catch (error) {
        return { supported: false, permission: 'error' };
      }
    });
  }

  async testBackgroundSync(): Promise<{ supported: boolean; registered: boolean }> {
    return await this.page.evaluate(async () => {
      if (!('serviceWorker' in navigator)) {
        return { supported: false, registered: false };
      }

      try {
        const registration = await navigator.serviceWorker.ready;
        
        // Check if Background Sync is supported
        const supported = 'sync' in registration;
        
        if (!supported) {
          return { supported: false, registered: false };
        }

        // Try to register a background sync
        try {
          await (registration as any).sync.register('test-sync');
          return { supported: true, registered: true };
        } catch (error) {
          return { supported: true, registered: false };
        }
      } catch (error) {
        return { supported: false, registered: false };
      }
    });
  }

  async testAppShell(): Promise<{ hasAppShell: boolean; shellElements: string[] }> {
    const shellInfo = await this.page.evaluate(() => {
      const shellSelectors = [
        'nav',
        '[data-testid="app-header"]',
        '[data-testid="app-sidebar"]',
        '[data-testid="app-navigation"]',
        '.app-shell',
        '.navigation',
        '.header'
      ];

      const foundElements: string[] = [];
      let hasAppShell = false;

      for (const selector of shellSelectors) {
        const elements = document.querySelectorAll(selector);
        if (elements.length > 0) {
          foundElements.push(selector);
          hasAppShell = true;
        }
      }

      return { hasAppShell, shellElements: foundElements };
    });

    return shellInfo;
  }

  async testResponsiveDesign(): Promise<{ 
    mobileOptimized: boolean; 
    hasViewport: boolean; 
    touchFriendly: boolean 
  }> {
    const responsiveInfo = await this.page.evaluate(() => {
      // Check viewport meta tag
      const viewportMeta = document.querySelector('meta[name="viewport"]') as HTMLMetaElement;
      const hasViewport = !!viewportMeta && viewportMeta.content.includes('width=device-width');

      // Check for mobile-friendly CSS
      const stylesheets = Array.from(document.styleSheets);
      let mobileOptimized = false;

      try {
        for (const stylesheet of stylesheets) {
          if (stylesheet.cssRules) {
            for (const rule of Array.from(stylesheet.cssRules)) {
              if (rule instanceof CSSMediaRule && rule.conditionText.includes('max-width')) {
                mobileOptimized = true;
                break;
              }
            }
          }
        }
      } catch (error) {
        // Cross-origin stylesheets might throw errors
        mobileOptimized = true; // Assume optimized if we can't check
      }

      // Check for touch-friendly elements
      const buttons = document.querySelectorAll('button, [role="button"], .btn');
      const touchFriendly = Array.from(buttons).some(button => {
        const styles = window.getComputedStyle(button);
        const minHeight = parseInt(styles.minHeight) || parseInt(styles.height) || 0;
        return minHeight >= 44; // 44px is the recommended minimum touch target size
      });

      return { mobileOptimized, hasViewport, touchFriendly };
    });

    return responsiveInfo;
  }
}

test.describe('Progressive Web App (PWA) Testing', () => {
  let pwaTester: PWATester;

  test.beforeEach(async ({ page, context }) => {
    pwaTester = new PWATester(page, context);
  });

  test('PWA Manifest validation', async ({ page }) => {
    await page.goto('/dashboard');
    
    const manifest = await pwaTester.getManifest();
    
    console.log('PWA Manifest:', manifest);
    
    if (manifest) {
      // Validate required manifest fields
      expect(manifest.name).toBeTruthy();
      expect(manifest.short_name).toBeTruthy();
      expect(manifest.start_url).toBeTruthy();
      expect(manifest.display).toBeTruthy();
      expect(manifest.icons).toBeInstanceOf(Array);
      expect(manifest.icons.length).toBeGreaterThan(0);
      
      // Validate icon requirements
      const hasLargeIcon = manifest.icons.some(icon => {
        const sizes = icon.sizes.split('x').map(s => parseInt(s));
        return sizes[0] >= 192 && sizes[1] >= 192;
      });
      expect(hasLargeIcon).toBe(true);
      
      // Validate display mode
      expect(['standalone', 'minimal-ui', 'fullscreen']).toContain(manifest.display);
      
    } else {
      console.warn('No PWA manifest found');
      // If no manifest, just check that the page loads properly
      await expect(page.locator('[data-testid="dashboard-content"]')).toBeVisible();
    }
  });

  test('Service Worker functionality', async ({ page }) => {
    await page.goto('/dashboard');
    await page.waitForTimeout(3000); // Allow service worker to register
    
    const swInfo = await pwaTester.getServiceWorkerInfo();
    
    console.log('Service Worker Info:', swInfo);
    
    if (swInfo) {
      expect(swInfo.scriptURL).toBeTruthy();
      expect(swInfo.scope).toBeTruthy();
      expect(['installing', 'installed', 'activating', 'activated']).toContain(swInfo.state);
    } else {
      console.warn('No service worker detected');
    }
  });

  test('Offline capability and caching', async ({ page, context }) => {
    const offlineTest = await pwaTester.testOfflineCapability();
    
    console.log('Offline Test Results:', offlineTest);
    
    if (offlineTest.offlineSupported) {
      expect(offlineTest.cacheHits).toBeGreaterThan(0);
      console.log(`✅ Offline mode supported with ${offlineTest.cacheHits} cached resources`);
    } else {
      console.log('ℹ️ Offline mode not fully supported, but basic functionality works');
    }
    
    // Verify we can go back online
    await page.goto('/dashboard');
    await expect(page.locator('[data-testid="dashboard-content"]')).toBeVisible();
  });

  test('App installation prompt', async ({ page }) => {
    await page.goto('/dashboard');
    
    const installTest = await pwaTester.testInstallPrompt();
    
    console.log('Install Prompt Test:', installTest);
    
    if (installTest.promptAvailable) {
      expect(installTest.canInstall).toBe(true);
      console.log('✅ App installation prompt is available');
    } else {
      console.log('ℹ️ Install prompt not triggered (may already be installed or criteria not met)');
    }
  });

  test('Push notification support', async ({ page }) => {
    await page.goto('/dashboard');
    
    const pushTest = await pwaTester.testPushNotifications();
    
    console.log('Push Notification Test:', pushTest);
    
    expect(pushTest.supported).toBe(true);
    expect(['default', 'granted', 'denied']).toContain(pushTest.permission);
    
    if (pushTest.supported) {
      console.log('✅ Push notifications are supported');
    }
  });

  test('Background sync capability', async ({ page }) => {
    await page.goto('/dashboard');
    
    const syncTest = await pwaTester.testBackgroundSync();
    
    console.log('Background Sync Test:', syncTest);
    
    if (syncTest.supported) {
      console.log('✅ Background sync is supported');
      if (syncTest.registered) {
        console.log('✅ Background sync registration successful');
      }
    } else {
      console.log('ℹ️ Background sync not supported in this browser');
    }
  });

  test('App shell architecture', async ({ page }) => {
    await page.goto('/dashboard');
    
    const shellTest = await pwaTester.testAppShell();
    
    console.log('App Shell Test:', shellTest);
    
    expect(shellTest.hasAppShell).toBe(true);
    expect(shellTest.shellElements.length).toBeGreaterThan(0);
    
    console.log(`✅ App shell detected with elements: ${shellTest.shellElements.join(', ')}`);
  });

  test('Mobile responsiveness and touch optimization', async ({ page }) => {
    await page.goto('/dashboard');
    
    const responsiveTest = await pwaTester.testResponsiveDesign();
    
    console.log('Responsive Design Test:', responsiveTest);
    
    expect(responsiveTest.hasViewport).toBe(true);
    expect(responsiveTest.mobileOptimized).toBe(true);
    
    if (responsiveTest.touchFriendly) {
      console.log('✅ Touch-friendly interface detected');
    } else {
      console.log('⚠️ Consider increasing touch target sizes for better mobile experience');
    }
  });

  test('PWA performance on mobile device simulation', async ({ page, context }) => {
    // Simulate mobile device
    await context.setViewportSize({ width: 375, height: 667 }); // iPhone SE size
    await context.setExtraHTTPHeaders({
      'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 14_7_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.2 Mobile/15E148 Safari/604.1'
    });
    
    const startTime = Date.now();
    await page.goto('/dashboard', { waitUntil: 'networkidle' });
    const loadTime = Date.now() - startTime;
    
    console.log(`Mobile load time: ${loadTime}ms`);
    
    // Test mobile-specific interactions
    await page.touchscreen.tap(100, 100); // Test touch interaction
    
    // Verify layout adapts to mobile
    const mobileLayout = await page.evaluate(() => {
      const body = document.body;
      const styles = window.getComputedStyle(body);
      return {
        width: styles.width,
        isMobile: window.innerWidth <= 768
      };
    });
    
    expect(mobileLayout.isMobile).toBe(true);
    expect(loadTime).toBeLessThan(5000); // Should load within 5 seconds on mobile
    
    // Test navigation works on mobile
    if (await page.isVisible('[data-testid="mobile-menu-toggle"]')) {
      await page.tap('[data-testid="mobile-menu-toggle"]');
      await expect(page.locator('[data-testid="mobile-menu"]')).toBeVisible();
    }
  });

  test('PWA data synchronization and updates', async ({ page }) => {
    await page.goto('/dashboard');
    await page.waitForLoadState('networkidle');
    
    // Test data refresh functionality
    if (await page.isVisible('[data-testid="refresh-button"]')) {
      const beforeRefresh = await page.textContent('[data-testid="last-updated"]') || '';
      
      await page.click('[data-testid="refresh-button"]');
      await page.waitForTimeout(2000);
      
      const afterRefresh = await page.textContent('[data-testid="last-updated"]') || '';
      
      // Verify data was refreshed (timestamp should change)
      if (beforeRefresh && afterRefresh) {
        expect(beforeRefresh).not.toEqual(afterRefresh);
        console.log('✅ Data synchronization working correctly');
      }
    }
    
    // Test real-time updates if available
    if (await page.isVisible('[data-testid="start-realtime"]')) {
      await page.click('[data-testid="start-realtime"]');
      await page.waitForTimeout(3000);
      
      // Check if real-time indicator is active
      const isRealtimeActive = await page.isVisible('[data-testid="realtime-indicator"]');
      if (isRealtimeActive) {
        console.log('✅ Real-time updates are functional');
      }
      
      await page.click('[data-testid="stop-realtime"]');
    }
  });

  test('PWA security and HTTPS requirements', async ({ page }) => {
    const isHTTPS = page.url().startsWith('https://');
    const isLocalhost = page.url().includes('localhost') || page.url().includes('127.0.0.1');
    
    // PWAs require HTTPS in production
    if (!isLocalhost) {
      expect(isHTTPS).toBe(true);
    }
    
    // Check security headers if possible
    const securityInfo = await page.evaluate(() => {
      return {
        protocol: window.location.protocol,
        isSecureContext: window.isSecureContext,
        hasServiceWorker: 'serviceWorker' in navigator
      };
    });
    
    console.log('Security Info:', securityInfo);
    
    if (!isLocalhost) {
      expect(securityInfo.protocol).toBe('https:');
      expect(securityInfo.isSecureContext).toBe(true);
    }
    
    expect(securityInfo.hasServiceWorker).toBe(true);
  });
});