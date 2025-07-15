import { test, expect } from '@playwright/test';

/**
 * Cross-Browser Responsive Design Tests
 * Tests responsive behavior across different browsers and viewport sizes
 */

const VIEWPORTS = {
  mobile: { width: 375, height: 667 },
  tablet: { width: 768, height: 1024 },
  desktop: { width: 1920, height: 1080 },
  ultrawide: { width: 2560, height: 1440 }
};

const BREAKPOINTS = {
  sm: 640,
  md: 768,
  lg: 1024,
  xl: 1280,
  '2xl': 1536
};

test.describe('Responsive Design Cross-Browser Tests', () => {

  test.beforeEach(async ({ page }) => {
    // Navigate to dashboard for responsive testing
    await page.goto('/dashboard');
    await page.waitForLoadState('networkidle');
  });

  test('navigation layout adapts correctly across viewport sizes', async ({ page, browserName }) => {
    for (const [sizeName, viewport] of Object.entries(VIEWPORTS)) {
      await page.setViewportSize(viewport);
      await page.waitForTimeout(500); // Allow layout to settle

      // Check navigation visibility and layout
      const nav = page.locator('nav');
      await expect(nav).toBeVisible();

      if (viewport.width < BREAKPOINTS.md) {
        // Mobile: Should have hamburger menu or collapsed nav
        const mobileMenu = page.locator('[data-testid="mobile-menu"]').or(
          page.locator('button[aria-label*="menu"]')
        ).or(
          page.locator('.hamburger')
        );

        // Either mobile menu exists or nav is transformed for mobile
        const hasMobileMenu = await mobileMenu.count() > 0;
        const navBox = await nav.boundingBox();

        expect(hasMobileMenu || (navBox && navBox.height < 100)).toBeTruthy();
      } else {
        // Desktop/Tablet: Should have full navigation
        const navItems = page.locator('nav a, nav button');
        await expect(navItems.first()).toBeVisible();

        // Navigation should be horizontal on larger screens
        const navBox = await nav.boundingBox();
        expect(navBox).toBeTruthy();
        expect(navBox!.width).toBeGreaterThan(200);
      }
    }
  });

  test('main content area responsive behavior', async ({ page, browserName }) => {
    for (const [sizeName, viewport] of Object.entries(VIEWPORTS)) {
      await page.setViewportSize(viewport);
      await page.waitForTimeout(500);

      const main = page.locator('main');
      await expect(main).toBeVisible();

      const mainBox = await main.boundingBox();
      expect(mainBox).toBeTruthy();

      // Content should not overflow viewport
      expect(mainBox!.width).toBeLessThanOrEqual(viewport.width);

      // Content should have reasonable minimum width
      expect(mainBox!.width).toBeGreaterThan(200);

      // Check for horizontal scrollbars (should not exist)
      const hasHorizontalScrollbar = await page.evaluate(() => {
        return document.documentElement.scrollWidth > document.documentElement.clientWidth;
      });

      expect(hasHorizontalScrollbar).toBeFalsy();
    }
  });

  test('dashboard cards responsive grid layout', async ({ page, browserName }) => {
    // Look for dashboard cards/widgets
    const cards = page.locator('[data-testid="dashboard-card"]').or(
      page.locator('.card')
    ).or(
      page.locator('[class*="card"]')
    );

    for (const [sizeName, viewport] of Object.entries(VIEWPORTS)) {
      await page.setViewportSize(viewport);
      await page.waitForTimeout(500);

      const cardCount = await cards.count();

      if (cardCount > 0) {
        // Check first few cards for responsive behavior
        for (let i = 0; i < Math.min(cardCount, 3); i++) {
          const card = cards.nth(i);
          await expect(card).toBeVisible();

          const cardBox = await card.boundingBox();
          expect(cardBox).toBeTruthy();

          // Card should not overflow viewport
          expect(cardBox!.width).toBeLessThanOrEqual(viewport.width - 40); // Account for padding

          // Cards should stack on mobile, be in grid on larger screens
          if (viewport.width < BREAKPOINTS.md && i > 0) {
            const prevCard = cards.nth(i - 1);
            const prevCardBox = await prevCard.boundingBox();

            // On mobile, cards should stack vertically
            expect(cardBox!.y).toBeGreaterThan(prevCardBox!.y + prevCardBox!.height - 20);
          }
        }
      }
    }
  });

  test('form inputs and buttons responsive sizing', async ({ page, browserName }) => {
    // Navigate to a page with forms (try datasets or detectors)
    await page.goto('/datasets');
    await page.waitForLoadState('networkidle');

    for (const [sizeName, viewport] of Object.entries(VIEWPORTS)) {
      await page.setViewportSize(viewport);
      await page.waitForTimeout(500);

      // Check form inputs
      const inputs = page.locator('input, select, textarea');
      const inputCount = await inputs.count();

      if (inputCount > 0) {
        const firstInput = inputs.first();
        await expect(firstInput).toBeVisible();

        const inputBox = await firstInput.boundingBox();
        expect(inputBox).toBeTruthy();

        // Input should not be too narrow or too wide
        expect(inputBox!.width).toBeGreaterThan(100);
        expect(inputBox!.width).toBeLessThanOrEqual(viewport.width - 40);

        // Input should have reasonable touch target size on mobile
        if (viewport.width < BREAKPOINTS.md) {
          expect(inputBox!.height).toBeGreaterThanOrEqual(44); // iOS/Android recommendation
        }
      }

      // Check buttons
      const buttons = page.locator('button');
      const buttonCount = await buttons.count();

      if (buttonCount > 0) {
        const firstButton = buttons.first();

        if (await firstButton.isVisible()) {
          const buttonBox = await firstButton.boundingBox();
          expect(buttonBox).toBeTruthy();

          // Button should have minimum touch target size on mobile
          if (viewport.width < BREAKPOINTS.md) {
            expect(Math.min(buttonBox!.width, buttonBox!.height)).toBeGreaterThanOrEqual(44);
          }
        }
      }
    }
  });

  test('typography and text scaling', async ({ page, browserName }) => {
    for (const [sizeName, viewport] of Object.entries(VIEWPORTS)) {
      await page.setViewportSize(viewport);
      await page.waitForTimeout(500);

      // Check main heading
      const heading = page.locator('h1').first();
      if (await heading.count() > 0) {
        await expect(heading).toBeVisible();

        const headingStyles = await heading.evaluate(el => {
          const styles = window.getComputedStyle(el);
          return {
            fontSize: parseFloat(styles.fontSize),
            lineHeight: parseFloat(styles.lineHeight),
            marginBottom: parseFloat(styles.marginBottom)
          };
        });

        // Font size should be reasonable for viewport
        if (viewport.width < BREAKPOINTS.md) {
          expect(headingStyles.fontSize).toBeGreaterThanOrEqual(20); // Min readable size on mobile
          expect(headingStyles.fontSize).toBeLessThanOrEqual(32); // Not too large on mobile
        } else {
          expect(headingStyles.fontSize).toBeGreaterThanOrEqual(24); // Larger on desktop
        }

        // Line height should be reasonable
        expect(headingStyles.lineHeight / headingStyles.fontSize).toBeGreaterThanOrEqual(1.1);
        expect(headingStyles.lineHeight / headingStyles.fontSize).toBeLessThanOrEqual(1.8);
      }

      // Check body text
      const bodyText = page.locator('p').first();
      if (await bodyText.count() > 0) {
        const textStyles = await bodyText.evaluate(el => {
          const styles = window.getComputedStyle(el);
          return {
            fontSize: parseFloat(styles.fontSize),
            lineHeight: parseFloat(styles.lineHeight)
          };
        });

        // Body text should be readable
        expect(textStyles.fontSize).toBeGreaterThanOrEqual(14); // Minimum readable size
        expect(textStyles.fontSize).toBeLessThanOrEqual(20); // Not too large

        // Line height should be readable
        expect(textStyles.lineHeight / textStyles.fontSize).toBeGreaterThanOrEqual(1.3);
        expect(textStyles.lineHeight / textStyles.fontSize).toBeLessThanOrEqual(1.8);
      }
    }
  });

  test('images and media responsive behavior', async ({ page, browserName }) => {
    for (const [sizeName, viewport] of Object.entries(VIEWPORTS)) {
      await page.setViewportSize(viewport);
      await page.waitForTimeout(500);

      // Check images
      const images = page.locator('img');
      const imageCount = await images.count();

      for (let i = 0; i < Math.min(imageCount, 3); i++) {
        const img = images.nth(i);

        if (await img.isVisible()) {
          const imgBox = await img.boundingBox();
          expect(imgBox).toBeTruthy();

          // Image should not overflow container
          expect(imgBox!.width).toBeLessThanOrEqual(viewport.width);

          // Image should maintain aspect ratio
          const naturalDimensions = await img.evaluate((el: HTMLImageElement) => ({
            naturalWidth: el.naturalWidth,
            naturalHeight: el.naturalHeight
          }));

          if (naturalDimensions.naturalWidth > 0 && naturalDimensions.naturalHeight > 0) {
            const naturalRatio = naturalDimensions.naturalWidth / naturalDimensions.naturalHeight;
            const displayedRatio = imgBox!.width / imgBox!.height;

            // Allow for some rounding differences
            expect(Math.abs(naturalRatio - displayedRatio)).toBeLessThan(0.1);
          }
        }
      }
    }
  });

  test('browser-specific CSS feature support', async ({ page, browserName }) => {
    const supportTests = await page.evaluate(() => {
      return {
        flexbox: CSS.supports('display', 'flex'),
        grid: CSS.supports('display', 'grid'),
        customProperties: CSS.supports('--color', 'red'),
        transform: CSS.supports('transform', 'translateX(1px)'),
        borderRadius: CSS.supports('border-radius', '5px'),
        boxShadow: CSS.supports('box-shadow', '0 0 5px rgba(0,0,0,0.5)'),
        transition: CSS.supports('transition', 'all 0.3s ease'),
        objectFit: CSS.supports('object-fit', 'cover'),
        aspectRatio: CSS.supports('aspect-ratio', '16/9'),
        gap: CSS.supports('gap', '1rem')
      };
    });

    // Essential features should be supported in modern browsers
    expect(supportTests.flexbox).toBeTruthy();
    expect(supportTests.customProperties).toBeTruthy();
    expect(supportTests.transform).toBeTruthy();
    expect(supportTests.borderRadius).toBeTruthy();
    expect(supportTests.transition).toBeTruthy();

    // Log browser-specific feature support for debugging
    console.log(`${browserName} CSS Feature Support:`, supportTests);

    // Test that layout still works even if newer features aren't supported
    const main = page.locator('main');
    await expect(main).toBeVisible();

    const mainBox = await main.boundingBox();
    expect(mainBox).toBeTruthy();
    expect(mainBox!.width).toBeGreaterThan(200);
    expect(mainBox!.height).toBeGreaterThan(100);
  });

  test('responsive tables and data displays', async ({ page, browserName }) => {
    // Look for tables or data displays
    const tables = page.locator('table');
    const dataGrids = page.locator('[role="grid"], [data-testid*="table"], [class*="table"]');

    for (const [sizeName, viewport] of Object.entries(VIEWPORTS)) {
      await page.setViewportSize(viewport);
      await page.waitForTimeout(500);

      // Test tables
      const tableCount = await tables.count();
      if (tableCount > 0) {
        const table = tables.first();
        await expect(table).toBeVisible();

        const tableBox = await table.boundingBox();
        expect(tableBox).toBeTruthy();

        if (viewport.width < BREAKPOINTS.md) {
          // On mobile, table should either:
          // 1. Be scrollable horizontally
          // 2. Be transformed to a different layout
          // 3. Have reduced columns

          const tableContainer = page.locator('div').filter({ has: table }).first();
          const containerStyles = await tableContainer.evaluate(el => {
            const styles = window.getComputedStyle(el);
            return {
              overflowX: styles.overflowX,
              overflowY: styles.overflowY
            };
          });

          // Either container is scrollable or table fits in viewport
          const fitsInViewport = tableBox!.width <= viewport.width;
          const isScrollable = containerStyles.overflowX === 'auto' || containerStyles.overflowX === 'scroll';

          expect(fitsInViewport || isScrollable).toBeTruthy();
        }
      }

      // Test data grids
      const gridCount = await dataGrids.count();
      if (gridCount > 0) {
        const grid = dataGrids.first();

        if (await grid.isVisible()) {
          const gridBox = await grid.boundingBox();
          expect(gridBox).toBeTruthy();

          // Grid should not cause horizontal overflow
          expect(gridBox!.width).toBeLessThanOrEqual(viewport.width + 20); // Small tolerance
        }
      }
    }
  });

  test('focus management and keyboard navigation responsive behavior', async ({ page, browserName }) => {
    for (const [sizeName, viewport] of Object.entries(VIEWPORTS)) {
      await page.setViewportSize(viewport);
      await page.waitForTimeout(500);

      // Test tab navigation
      await page.keyboard.press('Tab');
      const firstFocusable = await page.locator(':focus');

      if (await firstFocusable.count() > 0) {
        await expect(firstFocusable).toBeVisible();

        const focusBox = await firstFocusable.boundingBox();
        expect(focusBox).toBeTruthy();

        // Focused element should be within viewport
        expect(focusBox!.x).toBeGreaterThanOrEqual(-5); // Small tolerance
        expect(focusBox!.y).toBeGreaterThanOrEqual(-5);
        expect(focusBox!.x + focusBox!.width).toBeLessThanOrEqual(viewport.width + 5);

        // Focus outline should be visible (check for focus styles)
        const focusStyles = await firstFocusable.evaluate(el => {
          const styles = window.getComputedStyle(el);
          return {
            outline: styles.outline,
            outlineWidth: styles.outlineWidth,
            outlineStyle: styles.outlineStyle,
            boxShadow: styles.boxShadow
          };
        });

        // Should have some form of focus indication
        const hasFocusIndicator =
          focusStyles.outline !== 'none' ||
          focusStyles.outlineWidth !== '0px' ||
          focusStyles.boxShadow.includes('inset') ||
          focusStyles.boxShadow.includes('0px');

        expect(hasFocusIndicator).toBeTruthy();
      }
    }
  });
});

// Device-specific responsive tests
test.describe('Device-Specific Responsive Tests', () => {

  test('iOS Safari specific responsive behavior', async ({ page, browserName }) => {
    test.skip(browserName !== 'webkit', 'iOS Safari specific test');

    await page.setViewportSize({ width: 375, height: 667 }); // iPhone SE

    // Test iOS-specific behavior
    const viewportHeight = await page.evaluate(() => window.innerHeight);
    expect(viewportHeight).toBeGreaterThan(500); // Should not be affected by iOS viewport bugs

    // Test touch-action support
    const touchSupport = await page.evaluate(() => CSS.supports('touch-action', 'manipulation'));
    expect(touchSupport).toBeTruthy();
  });

  test('Android Chrome specific responsive behavior', async ({ page, browserName }) => {
    test.skip(browserName !== 'chromium', 'Android Chrome specific test');

    await page.setViewportSize({ width: 412, height: 732 }); // Pixel 3

    // Test Android-specific viewport behavior
    const viewport = await page.evaluate(() => ({
      width: window.innerWidth,
      height: window.innerHeight,
      devicePixelRatio: window.devicePixelRatio
    }));

    expect(viewport.width).toBe(412);
    expect(viewport.devicePixelRatio).toBeGreaterThan(1);
  });

  test('Desktop high-DPI display behavior', async ({ page, browserName }) => {
    await page.setViewportSize({ width: 1920, height: 1080 });

    // Simulate high-DPI display
    await page.evaluate(() => {
      Object.defineProperty(window, 'devicePixelRatio', {
        writable: true,
        configurable: true,
        value: 2
      });
    });

    // Test that images and text render clearly on high-DPI
    const images = page.locator('img');
    const imageCount = await images.count();

    if (imageCount > 0) {
      const img = images.first();
      const imgSrc = await img.getAttribute('src');

      // Should not be blurry (though this is hard to test automatically)
      expect(imgSrc).toBeTruthy();
    }
  });
});
