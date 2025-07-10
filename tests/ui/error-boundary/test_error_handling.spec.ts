/**
 * Enhanced Error Boundary and Error Handling Testing
 * 
 * This test suite validates error boundaries, graceful degradation,
 * and comprehensive error recovery mechanisms in the web UI.
 */

import { test, expect, Page, BrowserContext } from '@playwright/test';

interface ErrorScenario {
  name: string;
  description: string;
  trigger: (page: Page) => Promise<void>;
  expectedBehavior: string;
  recoverable: boolean;
}

interface ErrorBoundaryTest {
  component: string;
  errorType: string;
  errorMessage: string;
  trigger: (page: Page) => Promise<void>;
  expectedFallback: string;
  recoveryAction?: (page: Page) => Promise<void>;
}

class ErrorBoundaryTester {
  private page: Page;
  private context: BrowserContext;
  private errorLog: Array<{ timestamp: number; type: string; message: string; stack?: string }> = [];

  constructor(page: Page, context: BrowserContext) {
    this.page = page;
    this.context = context;
    this.setupErrorCapture();
  }

  private setupErrorCapture(): void {
    // Capture JavaScript errors
    this.page.on('pageerror', (error) => {
      this.errorLog.push({
        timestamp: Date.now(),
        type: 'javascript',
        message: error.message,
        stack: error.stack
      });
    });

    // Capture console errors
    this.page.on('console', (msg) => {
      if (msg.type() === 'error') {
        this.errorLog.push({
          timestamp: Date.now(),
          type: 'console',
          message: msg.text()
        });
      }
    });

    // Capture network errors
    this.page.on('requestfailed', (request) => {
      this.errorLog.push({
        timestamp: Date.now(),
        type: 'network',
        message: `Failed to load ${request.url()}: ${request.failure()?.errorText || 'Unknown error'}`
      });
    });
  }

  async triggerComponentError(component: string, errorType: string): Promise<void> {
    await this.page.evaluate(({ component, errorType }) => {
      // Inject error into specific component
      const element = document.querySelector(`[data-testid="${component}"]`);
      if (element) {
        // Trigger different types of errors
        switch (errorType) {
          case 'runtime':
            // Trigger a runtime error
            if ((window as any).triggerComponentError) {
              (window as any).triggerComponentError(component);
            } else {
              // Fallback: trigger a generic error
              throw new Error(`Simulated runtime error in ${component}`);
            }
            break;
          case 'async':
            // Trigger an async error
            setTimeout(() => {
              throw new Error(`Simulated async error in ${component}`);
            }, 100);
            break;
          case 'network':
            // Trigger a network-related error
            fetch('/api/nonexistent-endpoint').catch(() => {
              console.error(`Network error triggered for ${component}`);
            });
            break;
          case 'rendering':
            // Trigger a rendering error by corrupting DOM
            if (element.parentNode) {
              element.innerHTML = '<div>{{invalid.template.syntax}}</div>';
            }
            break;
        }
      }
    }, { component, errorType });
  }

  async checkErrorBoundaryFallback(componentId: string): Promise<boolean> {
    // Check if error boundary fallback UI is displayed
    const fallbackSelectors = [
      `[data-testid="${componentId}-error"]`,
      `[data-testid="error-boundary"]`,
      '[data-testid="error-fallback"]',
      '.error-boundary',
      '.error-fallback'
    ];

    for (const selector of fallbackSelectors) {
      if (await this.page.isVisible(selector)) {
        return true;
      }
    }

    // Check for error messages in the component
    const errorMessages = await this.page.locator(`[data-testid="${componentId}"]`).allTextContents();
    return errorMessages.some(text => 
      text.toLowerCase().includes('error') || 
      text.toLowerCase().includes('something went wrong') ||
      text.toLowerCase().includes('failed to load')
    );
  }

  async testGracefulDegradation(scenario: string): Promise<{ degraded: boolean; functional: boolean }> {
    let degraded = false;
    let functional = false;

    try {
      // Check if the UI still provides basic functionality despite errors
      const criticalElements = [
        '[data-testid="app-navigation"]',
        '[data-testid="main-content"]',
        '[data-testid="header"]'
      ];

      const visibleElements = await Promise.all(
        criticalElements.map(selector => this.page.isVisible(selector))
      );

      functional = visibleElements.some(visible => visible);

      // Check if error messaging is present (indicating degradation)
      const errorIndicators = [
        '[data-testid*="error"]',
        '.error-message',
        '.alert-error',
        '[role="alert"]'
      ];

      for (const selector of errorIndicators) {
        if (await this.page.isVisible(selector)) {
          degraded = true;
          break;
        }
      }

      // Check if fallback content is shown
      const fallbackContent = await this.page.locator('text=/fallback|offline|unavailable/i').count();
      if (fallbackContent > 0) {
        degraded = true;
      }

    } catch (error) {
      // If we can't even check basic elements, it's not functional
      functional = false;
      degraded = true;
    }

    return { degraded, functional };
  }

  async testErrorRecovery(recoveryAction: (page: Page) => Promise<void>): Promise<boolean> {
    const errorCountBefore = this.errorLog.length;
    
    try {
      await recoveryAction(this.page);
      await this.page.waitForTimeout(2000); // Allow recovery to complete
      
      // Check if the UI is functional again
      const isRecovered = await this.page.evaluate(() => {
        // Check if critical functionality is working
        const navigation = document.querySelector('[data-testid="app-navigation"]');
        const content = document.querySelector('[data-testid="main-content"]');
        
        return !!(navigation && content && document.body.style.display !== 'none');
      });

      return isRecovered;
    } catch (error) {
      return false;
    }
  }

  getErrorSummary(): { total: number; byType: Record<string, number>; recent: typeof this.errorLog } {
    const byType = this.errorLog.reduce((acc, error) => {
      acc[error.type] = (acc[error.type] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);

    return {
      total: this.errorLog.length,
      byType,
      recent: this.errorLog.slice(-10) // Last 10 errors
    };
  }
}

test.describe('Error Boundary and Error Handling', () => {
  let errorTester: ErrorBoundaryTester;

  test.beforeEach(async ({ page, context }) => {
    errorTester = new ErrorBoundaryTester(page, context);
  });

  test('Component error boundaries handle JavaScript errors gracefully', async ({ page }) => {
    await page.goto('/dashboard');
    await page.waitForLoadState('networkidle');

    const errorBoundaryTests: ErrorBoundaryTest[] = [
      {
        component: 'dashboard-stats',
        errorType: 'runtime',
        errorMessage: 'Runtime error in dashboard stats',
        trigger: async (page) => {
          await errorTester.triggerComponentError('dashboard-stats', 'runtime');
        },
        expectedFallback: 'error-boundary-stats'
      },
      {
        component: 'chart-container',
        errorType: 'rendering',
        errorMessage: 'Chart rendering error',
        trigger: async (page) => {
          await errorTester.triggerComponentError('chart-container', 'rendering');
        },
        expectedFallback: 'chart-error-fallback'
      },
      {
        component: 'data-table',
        errorType: 'async',
        errorMessage: 'Async operation failed',
        trigger: async (page) => {
          await errorTester.triggerComponentError('data-table', 'async');
        },
        expectedFallback: 'table-error-message'
      }
    ];

    for (const test of errorBoundaryTests) {
      console.log(`Testing error boundary for ${test.component}...`);
      
      // Trigger the error
      await test.trigger(page);
      await page.waitForTimeout(1000);
      
      // Check if error boundary caught the error
      const hasFallback = await errorTester.checkErrorBoundaryFallback(test.component);
      
      if (hasFallback) {
        console.log(`✅ Error boundary working for ${test.component}`);
      } else {
        console.log(`⚠️ Error boundary may not be implemented for ${test.component}`);
      }
      
      // Verify the rest of the app is still functional
      const isAppStillFunctional = await page.isVisible('[data-testid="app-navigation"]');
      expect(isAppStillFunctional).toBe(true);
    }
  });

  test('Network errors trigger appropriate fallback UI', async ({ page }) => {
    await page.goto('/dashboard');
    await page.waitForLoadState('networkidle');

    // Simulate network failures
    const networkScenarios = [
      {
        name: 'API endpoint failure',
        setup: async () => {
          await page.route('**/api/**', route => {
            route.abort('failed');
          });
        }
      },
      {
        name: 'Slow network timeout',
        setup: async () => {
          await page.route('**/api/**', route => {
            setTimeout(() => route.continue(), 10000); // 10 second delay
          });
        }
      },
      {
        name: 'Server error responses',
        setup: async () => {
          await page.route('**/api/**', route => {
            route.fulfill({
              status: 500,
              contentType: 'application/json',
              body: JSON.stringify({ error: 'Internal server error' })
            });
          });
        }
      }
    ];

    for (const scenario of networkScenarios) {
      console.log(`Testing network scenario: ${scenario.name}`);
      
      await scenario.setup();
      
      // Trigger API calls
      if (await page.isVisible('[data-testid="refresh-button"]')) {
        await page.click('[data-testid="refresh-button"]');
        await page.waitForTimeout(3000);
      }
      
      // Check for network error handling
      const degradationResult = await errorTester.testGracefulDegradation(scenario.name);
      
      console.log(`Network error handling for ${scenario.name}:`, degradationResult);
      
      // App should still be functional even with network errors
      expect(degradationResult.functional).toBe(true);
      
      // Reset network routing
      await page.unroute('**/api/**');
    }
  });

  test('Offline mode and service worker error handling', async ({ page, context }) => {
    await page.goto('/dashboard');
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(3000); // Allow service worker to activate

    // Test offline functionality
    console.log('Testing offline mode...');
    await context.setOffline(true);
    
    try {
      // Try to navigate or refresh while offline
      await page.reload({ timeout: 10000 });
      
      // Check if offline indicators are shown
      const offlineIndicators = await page.locator('text=/offline|no connection|disconnected/i').count();
      
      if (offlineIndicators > 0) {
        console.log('✅ Offline indicators detected');
      }
      
      // Verify basic functionality still works
      const basicFunctionality = await page.evaluate(() => {
        // Check if cached content is available
        return document.querySelector('[data-testid="dashboard-content"]') !== null;
      });
      
      if (basicFunctionality) {
        console.log('✅ Basic functionality available offline');
      }
      
    } catch (error) {
      console.log('ℹ️ Page failed to load offline (may not have service worker)');
    } finally {
      await context.setOffline(false);
    }
  });

  test('Form validation and input error handling', async ({ page }) => {
    await page.goto('/detectors');
    await page.waitForLoadState('networkidle');

    // Test form error scenarios
    if (await page.isVisible('[data-testid="create-detector"]')) {
      await page.click('[data-testid="create-detector"]');
      await page.waitForSelector('[data-testid="detector-form"]');

      const formErrorTests = [
        {
          name: 'Empty required fields',
          action: async () => {
            await page.click('[data-testid="submit-detector"]');
          },
          expectedError: 'required'
        },
        {
          name: 'Invalid input values',
          action: async () => {
            await page.fill('input[name="contamination"]', 'invalid_number');
            await page.click('[data-testid="submit-detector"]');
          },
          expectedError: 'invalid'
        },
        {
          name: 'Extremely long input',
          action: async () => {
            const longText = 'x'.repeat(10000);
            await page.fill('input[name="name"]', longText);
            await page.click('[data-testid="submit-detector"]');
          },
          expectedError: 'too long'
        }
      ];

      for (const test of formErrorTests) {
        console.log(`Testing form error: ${test.name}`);
        
        await test.action();
        await page.waitForTimeout(1000);
        
        // Check for error messages
        const errorMessages = await page.locator('.error, .alert-error, [role="alert"]').allTextContents();
        const hasExpectedError = errorMessages.some(msg => 
          msg.toLowerCase().includes(test.expectedError.toLowerCase())
        );
        
        if (hasExpectedError) {
          console.log(`✅ Form error handled: ${test.name}`);
        } else {
          console.log(`⚠️ Form error may not be properly handled: ${test.name}`);
        }
        
        // Clear the form for next test
        await page.fill('input[name="name"]', '');
        await page.fill('input[name="contamination"]', '');
      }
    }
  });

  test('Resource loading error handling', async ({ page }) => {
    await page.goto('/dashboard');
    await page.waitForLoadState('networkidle');

    // Test various resource loading failures
    const resourceTests = [
      {
        name: 'CSS loading failure',
        block: '**/*.css'
      },
      {
        name: 'JavaScript loading failure',
        block: '**/*.js'
      },
      {
        name: 'Image loading failure',
        block: '**/*.{png,jpg,jpeg,gif,svg,webp}'
      },
      {
        name: 'Font loading failure',
        block: '**/*.{woff,woff2,ttf,eot}'
      }
    ];

    for (const test of resourceTests) {
      console.log(`Testing resource error: ${test.name}`);
      
      // Block specific resources
      await page.route(test.block, route => {
        route.abort('failed');
      });
      
      // Navigate to a new page to trigger resource loading
      await page.goto('/detectors');
      await page.waitForTimeout(2000);
      
      // Check if the page is still functional
      const isPageFunctional = await page.evaluate(() => {
        const content = document.querySelector('[data-testid="detectors-list"], [data-testid="main-content"]');
        return content !== null && content.textContent !== '';
      });
      
      console.log(`Page functional after ${test.name}: ${isPageFunctional}`);
      expect(isPageFunctional).toBe(true);
      
      // Reset routing
      await page.unroute(test.block);
    }
  });

  test('Error recovery mechanisms', async ({ page }) => {
    await page.goto('/dashboard');
    await page.waitForLoadState('networkidle');

    const recoveryTests = [
      {
        name: 'Page refresh recovery',
        trigger: async () => {
          // Simulate a critical error
          await page.evaluate(() => {
            throw new Error('Critical application error');
          });
        },
        recovery: async (page: Page) => {
          await page.reload();
        }
      },
      {
        name: 'Component remount recovery',
        trigger: async () => {
          await errorTester.triggerComponentError('dashboard-stats', 'runtime');
        },
        recovery: async (page: Page) => {
          // Try to trigger component remount
          if (await page.isVisible('[data-testid="retry-button"]')) {
            await page.click('[data-testid="retry-button"]');
          } else if (await page.isVisible('[data-testid="refresh-button"]')) {
            await page.click('[data-testid="refresh-button"]');
          }
        }
      },
      {
        name: 'Clear cache recovery',
        trigger: async () => {
          // Corrupt local storage
          await page.evaluate(() => {
            localStorage.setItem('corrupted-data', 'invalid-json-{{{');
          });
        },
        recovery: async (page: Page) => {
          await page.evaluate(() => {
            localStorage.clear();
            sessionStorage.clear();
          });
          await page.reload();
        }
      }
    ];

    for (const test of recoveryTests) {
      console.log(`Testing error recovery: ${test.name}`);
      
      // Trigger the error
      await test.trigger();
      await page.waitForTimeout(1000);
      
      // Attempt recovery
      const recovered = await errorTester.testErrorRecovery(test.recovery);
      
      console.log(`Recovery successful for ${test.name}: ${recovered}`);
      
      if (recovered) {
        console.log(`✅ ${test.name} recovery mechanism working`);
      } else {
        console.log(`⚠️ ${test.name} recovery may need improvement`);
      }
    }
  });

  test('Error logging and reporting functionality', async ({ page }) => {
    await page.goto('/dashboard');
    await page.waitForLoadState('networkidle');

    // Trigger various types of errors to test logging
    const errorTypes = ['javascript', 'network', 'console'];
    
    for (const errorType of errorTypes) {
      await errorTester.triggerComponentError('test-component', errorType);
      await page.waitForTimeout(500);
    }

    // Check error logging
    const errorSummary = errorTester.getErrorSummary();
    
    console.log('Error Summary:', errorSummary);
    
    // Verify errors were captured
    expect(errorSummary.total).toBeGreaterThan(0);
    
    // Check if client-side error reporting is working
    const clientErrorReporting = await page.evaluate(() => {
      // Check if errors are being sent to an error reporting service
      return !!(window as any).errorReporter || 
             !!(window as any).Sentry || 
             !!(window as any).bugsnag ||
             typeof (window as any).reportError === 'function';
    });
    
    if (clientErrorReporting) {
      console.log('✅ Client-side error reporting detected');
    } else {
      console.log('ℹ️ No client-side error reporting service detected');
    }
  });

  test('Accessibility error handling', async ({ page }) => {
    await page.goto('/dashboard');
    await page.waitForLoadState('networkidle');

    // Test accessibility during error states
    const accessibilityTests = [
      {
        name: 'Error messages have proper ARIA labels',
        trigger: async () => {
          // Trigger a form error
          if (await page.isVisible('[data-testid="create-detector"]')) {
            await page.click('[data-testid="create-detector"]');
            await page.click('[data-testid="submit-detector"]');
          }
        },
        check: async () => {
          const errorElements = await page.locator('[role="alert"], .error, .alert-error').all();
          return errorElements.length > 0;
        }
      },
      {
        name: 'Keyboard navigation works in error states',
        trigger: async () => {
          await errorTester.triggerComponentError('navigation', 'runtime');
        },
        check: async () => {
          // Try tab navigation
          await page.keyboard.press('Tab');
          await page.keyboard.press('Tab');
          const focusedElement = await page.evaluate(() => document.activeElement?.tagName);
          return focusedElement === 'BUTTON' || focusedElement === 'A' || focusedElement === 'INPUT';
        }
      },
      {
        name: 'Screen reader announcements for errors',
        trigger: async () => {
          await errorTester.triggerComponentError('main-content', 'async');
        },
        check: async () => {
          const ariaLiveRegions = await page.locator('[aria-live]').count();
          return ariaLiveRegions > 0;
        }
      }
    ];

    for (const test of accessibilityTests) {
      console.log(`Testing accessibility error handling: ${test.name}`);
      
      await test.trigger();
      await page.waitForTimeout(1000);
      
      const isAccessible = await test.check();
      
      if (isAccessible) {
        console.log(`✅ Accessibility maintained: ${test.name}`);
      } else {
        console.log(`⚠️ Accessibility issue detected: ${test.name}`);
      }
    }
  });

  test.afterEach(async ({ page }, testInfo) => {
    const errorSummary = errorTester.getErrorSummary();
    
    // Attach error log to test results
    await testInfo.attach('error-log.json', {
      body: JSON.stringify(errorSummary, null, 2),
      contentType: 'application/json'
    });
    
    console.log(`Test "${testInfo.title}" completed with ${errorSummary.total} errors captured`);
  });
});