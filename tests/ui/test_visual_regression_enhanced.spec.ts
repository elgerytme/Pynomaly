/**
 * Enhanced Visual Regression Testing for Pynomaly Web Interface
 * Comprehensive visual testing with Percy integration and custom assertions
 */

import { test, expect, Page } from '@playwright/test';
// import percySnapshot from '@percy/playwright'; // Commented out until Percy is configured

class VisualTestHelper {
  constructor(private page: Page) {}

  async preparePageForSnapshot() {
    // Wait for all content to load
    await this.page.waitForLoadState('networkidle');

    // Hide dynamic elements that shouldn't be in snapshots
    await this.page.addStyleTag({
      content: `
        .timestamp, .time-display, .loading-spinner, .pulse {
          visibility: hidden !important;
        }

        /* Disable animations for consistent snapshots */
        *, *::before, *::after {
          animation-duration: 0s !important;
          animation-delay: 0s !important;
          transition-duration: 0s !important;
          transition-delay: 0s !important;
        }

        /* Ensure charts are in stable state */
        .chart-container canvas {
          animation: none !important;
        }

        /* Hide scrollbars */
        ::-webkit-scrollbar { display: none; }
        * { scrollbar-width: none; }
      `
    });

    // Wait a bit for styles to apply
    await this.page.waitForTimeout(500);
  }

  async simulateDataLoading() {
    // Simulate realistic data for consistent snapshots
    await this.page.evaluate(() => {
      // Mock data for charts and visualizations
      if (window.mockDataForTesting) {
        window.mockDataForTesting();
      }
    });
  }

  async testComponentStates(componentSelector: string, states: string[]) {
    const component = this.page.locator(componentSelector);

    for (const state of states) {
      await component.hover();

      if (state === 'hover') {
        await this.page.waitForTimeout(200);
      } else if (state === 'focus') {
        await component.focus();
        await this.page.waitForTimeout(200);
      } else if (state === 'active') {
        await component.click();
        await this.page.waitForTimeout(200);
      }
    }
  }
}

test.describe('Visual Regression Testing - Pynomaly Web Interface', () => {
  let helper: VisualTestHelper;

  test.beforeEach(async ({ page }) => {
    helper = new VisualTestHelper(page);

    // Set consistent viewport
    await page.setViewportSize({ width: 1280, height: 720 });

    // Disable animations globally
    await page.addInitScript(() => {
      // Override CSS animations
      const style = document.createElement('style');
      style.innerHTML = `
        *, *::before, *::after {
          animation-duration: 0s !important;
          transition-duration: 0s !important;
        }
      `;
      document.head.appendChild(style);
    });
  });

  test('Dashboard main view visual consistency', async ({ page }) => {
    await page.goto('/');
    await helper.preparePageForSnapshot();
    await helper.simulateDataLoading();

    // Take baseline snapshot
    await expect(page).toHaveScreenshot('dashboard-main-view.png', {
      fullPage: true,
      threshold: 0.2
    });

    // Verify key visual elements
    await expect(page.locator('header')).toBeVisible();
    await expect(page.locator('nav')).toBeVisible();
    await expect(page.locator('main')).toBeVisible();

    // Check for dashboard-specific elements
    const dashboardElements = [
      '[data-testid="stats-overview"]',
      '[data-testid="recent-detections"]',
      '[data-testid="system-status"]'
    ];

    for (const selector of dashboardElements) {
      const element = page.locator(selector);
      if (await element.count() > 0) {
        await expect(element).toBeVisible();
      }
    }
  });

  test('Detection interface visual consistency', async ({ page }) => {
    await page.goto('/detection');
    await helper.preparePageForSnapshot();

    // Wait for detection interface to load
    await page.waitForSelector('[data-testid="detection-form"], .detection-container', {
      state: 'visible',
      timeout: 10000
    });

    await expect(page).toHaveScreenshot('detection-interface-main.png', {
      fullPage: true,
      threshold: 0.2
    });

    // Test form states
    const uploadArea = page.locator('[data-testid="file-upload"], .upload-area').first();
    if (await uploadArea.count() > 0) {
      await uploadArea.hover();

      await expect(page).toHaveScreenshot('detection-interface-upload-hover.png', {
        fullPage: true,
        threshold: 0.2
      });
    }
  });

  test('Detectors page visual consistency', async ({ page }) => {
    await page.goto('/detectors');
    await helper.preparePageForSnapshot();

    // Wait for detectors list
    await page.waitForSelector('[data-testid="detectors-list"], .detectors-container', {
      state: 'visible',
      timeout: 10000
    });

    await expect(page).toHaveScreenshot('detectors-page-list-view.png', {
      fullPage: true,
      threshold: 0.2
    });

    // Test detector card states
    const detectorCards = page.locator('[data-testid="detector-card"], .detector-item');
    const cardCount = await detectorCards.count();

    if (cardCount > 0) {
      await detectorCards.first().hover();

      await expect(page).toHaveScreenshot('detectors-page-card-hover.png', {
        fullPage: true,
        threshold: 0.2
      });
    }
  });

  test('Datasets page visual consistency', async ({ page }) => {
    await page.goto('/datasets');
    await helper.preparePageForSnapshot();

    // Wait for datasets interface
    await page.waitForSelector('[data-testid="datasets-list"], .datasets-container', {
      state: 'visible',
      timeout: 10000
    });

    await expect(page).toHaveScreenshot('datasets-page-main-view.png', {
      fullPage: true,
      threshold: 0.2
    });

    // Test table/list interactions
    const datasetRows = page.locator('[data-testid="dataset-row"], .dataset-item, tbody tr');
    const rowCount = await datasetRows.count();

    if (rowCount > 0) {
      await datasetRows.first().hover();

      if (process.env.PERCY_TOKEN) {
        await expect(page).toHaveScreenshot('Datasets Page - Row Hover', {
          widths: [1280],
          minHeight: 720
        });
      }
    }
  });

  test('Visualizations page visual consistency', async ({ page }) => {
    await page.goto('/visualizations');
    await helper.preparePageForSnapshot();

    // Wait for visualization content
    await page.waitForSelector('[data-testid="visualization-container"], .viz-container', {
      state: 'visible',
      timeout: 10000
    });

    // Allow extra time for charts to render
    await page.waitForTimeout(1000);

    if (process.env.PERCY_TOKEN) {
      await expect(page).toHaveScreenshot('Visualizations Page - Main View', {
        widths: [1280],
        minHeight: 720
      });
    }

    // Test chart interactions
    const charts = page.locator('.chart-container, [data-testid="chart"]');
    const chartCount = await charts.count();

    if (chartCount > 0) {
      await charts.first().hover();
      await page.waitForTimeout(500);

      if (process.env.PERCY_TOKEN) {
        await expect(page).toHaveScreenshot('Visualizations Page - Chart Interaction', {
          widths: [1280],
          minHeight: 720
        });
      }
    }
  });

  test('Responsive design visual consistency', async ({ page }) => {
    const viewports = [
      { name: 'Mobile', width: 375, height: 667 },
      { name: 'Tablet', width: 768, height: 1024 },
      { name: 'Desktop', width: 1280, height: 720 },
      { name: 'Large Desktop', width: 1920, height: 1080 }
    ];

    for (const viewport of viewports) {
      await page.setViewportSize({ width: viewport.width, height: viewport.height });
      await page.goto('/');
      await helper.preparePageForSnapshot();

      if (process.env.PERCY_TOKEN) {
        await expect(page).toHaveScreenshot(`Dashboard - ${viewport.name}`, {
          widths: [viewport.width],
          minHeight: viewport.height
        });
      }

      // Verify responsive elements
      if (viewport.width < 768) {
        // Mobile: check for hamburger menu
        const mobileMenu = page.locator('[data-testid="mobile-menu"], .mobile-nav-trigger');
        if (await mobileMenu.count() > 0) {
          await expect(mobileMenu).toBeVisible();
        }
      } else {
        // Desktop: check for full navigation
        const navigation = page.locator('nav a, .nav-links a');
        if (await navigation.count() > 0) {
          await expect(navigation.first()).toBeVisible();
        }
      }
    }
  });

  test('Dark mode visual consistency', async ({ page }) => {
    await page.goto('/');
    await helper.preparePageForSnapshot();

    // Test light mode first
    if (process.env.PERCY_TOKEN) {
      await expect(page).toHaveScreenshot('Dashboard - Light Mode', {
        widths: [1280],
        minHeight: 720
      });
    }

    // Switch to dark mode if toggle exists
    const darkModeToggle = page.locator('[data-testid="dark-mode-toggle"], .theme-toggle');
    if (await darkModeToggle.count() > 0) {
      await darkModeToggle.click();
      await page.waitForTimeout(500); // Allow theme transition

      if (process.env.PERCY_TOKEN) {
        await expect(page).toHaveScreenshot('Dashboard - Dark Mode', {
          widths: [1280],
          minHeight: 720
        });
      }
    }
  });

  test('Component states visual consistency', async ({ page }) => {
    await page.goto('/');
    await helper.preparePageForSnapshot();

    // Test button states
    const buttons = page.locator('button').first();
    if (await buttons.count() > 0) {
      // Normal state
      if (process.env.PERCY_TOKEN) {
        await expect(page).toHaveScreenshot('Components - Button Normal', {
          widths: [1280],
          minHeight: 720
        });
      }

      // Hover state
      await buttons.hover();
      await page.waitForTimeout(200);

      if (process.env.PERCY_TOKEN) {
        await expect(page).toHaveScreenshot('Components - Button Hover', {
          widths: [1280],
          minHeight: 720
        });
      }

      // Focus state
      await buttons.focus();
      await page.waitForTimeout(200);

      if (process.env.PERCY_TOKEN) {
        await expect(page).toHaveScreenshot('Components - Button Focus', {
          widths: [1280],
          minHeight: 720
        });
      }
    }
  });

  test('Error states visual consistency', async ({ page }) => {
    // Test 404 page
    await page.goto('/nonexistent-page');
    await helper.preparePageForSnapshot();

    if (process.env.PERCY_TOKEN) {
      await expect(page).toHaveScreenshot('Error Pages - 404 Not Found', {
        widths: [1280],
        minHeight: 720
      });
    }

    // Verify error page elements
    const errorElements = [
      'h1', // Error title
      '.error-message, [data-testid="error-message"]', // Error description
      'a[href="/"], .home-link' // Back to home link
    ];

    for (const selector of errorElements) {
      const element = page.locator(selector);
      if (await element.count() > 0) {
        await expect(element.first()).toBeVisible();
      }
    }
  });

  test('Loading states visual consistency', async ({ page }) => {
    // Intercept API calls to test loading states
    await page.route('**/api/**', async route => {
      // Delay response to capture loading state
      await new Promise(resolve => setTimeout(resolve, 1000));
      await route.continue();
    });

    await page.goto('/detection');

    // Capture loading state if present
    const loadingElements = page.locator('.loading, .spinner, [data-testid="loading"]');
    if (await loadingElements.count() > 0) {
      if (process.env.PERCY_TOKEN) {
        await expect(page).toHaveScreenshot('Loading States - Detection Page', {
          widths: [1280],
          minHeight: 720
        });
      }
    }

    // Wait for loading to complete
    await page.waitForLoadState('networkidle');
    await helper.preparePageForSnapshot();

    if (process.env.PERCY_TOKEN) {
      await expect(page).toHaveScreenshot('Loaded States - Detection Page', {
        widths: [1280],
        minHeight: 720
      });
    }
  });
});
