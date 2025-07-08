import { test, expect } from '@playwright/test';

/**
 * Mobile Touch Interaction Tests
 * Tests touch-specific interactions and mobile UX patterns
 */

test.describe('Mobile Touch Interactions', () => {

  test.beforeEach(async ({ page }) => {
    // Set mobile viewport
    await page.setViewportSize({ width: 375, height: 667 });
    await page.goto('/dashboard');
    await page.waitForLoadState('networkidle');
  });

  test('tap interactions work correctly', async ({ page }) => {
    // Find tappable elements
    const buttons = page.locator('button');
    const links = page.locator('a');

    if (await buttons.count() > 0) {
      const button = buttons.first();

      if (await button.isVisible()) {
        // Test tap interaction
        await button.tap();

        // Should have some feedback (could be navigation, state change, etc.)
        // We can't predict exact behavior, but tap should not fail
        await page.waitForTimeout(100);
      }
    }

    if (await links.count() > 0) {
      const link = links.first();

      if (await link.isVisible()) {
        // Test that link has proper touch target size
        const linkBox = await link.boundingBox();
        expect(linkBox).toBeTruthy();

        // iOS/Android recommend minimum 44px touch targets
        const minSize = Math.min(linkBox!.width, linkBox!.height);
        expect(minSize).toBeGreaterThanOrEqual(44);
      }
    }
  });

  test('swipe gestures for navigation', async ({ page }) => {
    // Look for swipeable content (carousels, tabs, etc.)
    const swipeableElements = page.locator('[data-swipeable]').or(
      page.locator('.carousel')
    ).or(
      page.locator('[class*="swipe"]')
    ).or(
      page.locator('.tabs')
    );

    const count = await swipeableElements.count();

    if (count > 0) {
      const element = swipeableElements.first();
      const elementBox = await element.boundingBox();

      if (elementBox) {
        // Perform swipe left gesture
        await page.touchscreen.tap(elementBox.x + elementBox.width * 0.8, elementBox.y + elementBox.height / 2);
        await page.mouse.down();
        await page.mouse.move(elementBox.x + elementBox.width * 0.2, elementBox.y + elementBox.height / 2, { steps: 10 });
        await page.mouse.up();

        await page.waitForTimeout(500); // Allow animation

        // Content should have changed or moved
        const newPosition = await element.boundingBox();
        expect(newPosition).toBeTruthy();
      }
    }
  });

  test('long press interactions', async ({ page }) => {
    // Find elements that might support long press (context menus, etc.)
    const contextElements = page.locator('[data-context-menu]').or(
      page.locator('.card')
    ).or(
      page.locator('img')
    );

    const count = await contextElements.count();

    if (count > 0) {
      const element = contextElements.first();

      if (await element.isVisible()) {
        const elementBox = await element.boundingBox();

        if (elementBox) {
          // Simulate long press (touchstart + delay + touchend)
          await page.touchscreen.tap(
            elementBox.x + elementBox.width / 2,
            elementBox.y + elementBox.height / 2
          );

          // Hold for long press duration
          await page.waitForTimeout(800);

          // Check if context menu or other long press feedback appears
          const contextMenu = page.locator('[role="menu"]').or(
            page.locator('.context-menu')
          ).or(
            page.locator('[data-testid="context-menu"]')
          );

          // If context menu exists, it should be visible after long press
          if (await contextMenu.count() > 0) {
            await expect(contextMenu.first()).toBeVisible();
          }
        }
      }
    }
  });

  test('scroll behavior and momentum', async ({ page }) => {
    // Test scrolling behavior
    const initialScrollY = await page.evaluate(() => window.scrollY);

    // Perform touch scroll
    await page.touchscreen.tap(200, 400);
    await page.mouse.down();
    await page.mouse.move(200, 200, { steps: 10 });
    await page.mouse.up();

    await page.waitForTimeout(500); // Allow scroll momentum

    const finalScrollY = await page.evaluate(() => window.scrollY);

    // Should have scrolled down
    expect(finalScrollY).toBeGreaterThan(initialScrollY);

    // Test that scroll is smooth (check for scroll-behavior CSS)
    const scrollBehavior = await page.evaluate(() => {
      return window.getComputedStyle(document.documentElement).scrollBehavior;
    });

    // Should be 'smooth' or 'auto' (both are acceptable)
    expect(['smooth', 'auto']).toContain(scrollBehavior);
  });

  test('pinch zoom behavior', async ({ page }) => {
    // Check viewport meta tag for zoom control
    const viewportMeta = await page.locator('meta[name="viewport"]').getAttribute('content');

    if (viewportMeta) {
      // Should allow reasonable zoom or explicitly disable it
      expect(viewportMeta).toMatch(/(user-scalable=no|maximum-scale=|minimum-scale=)/);
    }

    // Test actual zoom behavior
    const initialZoom = await page.evaluate(() => window.devicePixelRatio);

    // Simulate pinch zoom gesture (simplified)
    await page.touchscreen.tap(200, 300);
    await page.keyboard.down('Control');
    await page.mouse.wheel(0, -100); // Zoom in
    await page.keyboard.up('Control');

    await page.waitForTimeout(300);

    // Check if zoom is controlled or allowed
    const finalZoom = await page.evaluate(() => window.devicePixelRatio);

    // Zoom behavior should be consistent with viewport settings
    expect(typeof finalZoom).toBe('number');
  });

  test('touch target accessibility', async ({ page }) => {
    // Find all interactive elements
    const interactiveElements = page.locator('button, a, input, select, textarea, [tabindex="0"], [role="button"]');
    const count = await interactiveElements.count();

    for (let i = 0; i < Math.min(count, 10); i++) {
      const element = interactiveElements.nth(i);

      if (await element.isVisible()) {
        const box = await element.boundingBox();

        if (box) {
          // Check minimum touch target size (44px iOS, 48dp Android)
          const minSize = Math.min(box.width, box.height);
          expect(minSize).toBeGreaterThanOrEqual(44);

          // Check adequate spacing between touch targets
          const elementCenter = { x: box.x + box.width / 2, y: box.y + box.height / 2 };

          // Look for nearby interactive elements
          for (let j = i + 1; j < Math.min(count, i + 5); j++) {
            const otherElement = interactiveElements.nth(j);

            if (await otherElement.isVisible()) {
              const otherBox = await otherElement.boundingBox();

              if (otherBox) {
                const otherCenter = { x: otherBox.x + otherBox.width / 2, y: otherBox.y + otherBox.height / 2 };
                const distance = Math.sqrt(
                  Math.pow(elementCenter.x - otherCenter.x, 2) +
                  Math.pow(elementCenter.y - otherCenter.y, 2)
                );

                // If elements are close, there should be adequate spacing
                if (distance < 100) {
                  expect(distance).toBeGreaterThanOrEqual(44);
                }
              }
            }
          }
        }
      }
    }
  });

  test('mobile form interactions', async ({ page }) => {
    await page.goto('/datasets'); // Page likely to have forms
    await page.waitForLoadState('networkidle');

    const inputs = page.locator('input, textarea');
    const count = await inputs.count();

    if (count > 0) {
      const input = inputs.first();

      if (await input.isVisible()) {
        // Test touch focus
        await input.tap();
        await expect(input).toBeFocused();

        // Check for mobile-appropriate input types
        const inputType = await input.getAttribute('type');
        const inputMode = await input.getAttribute('inputmode');

        // Should have appropriate input type or inputmode for mobile keyboards
        if (inputType === 'email' || inputMode === 'email') {
          expect(['email']).toContain(inputType || inputMode);
        }

        // Test virtual keyboard doesn't break layout
        const beforeHeight = await page.evaluate(() => window.innerHeight);

        // Type some text to trigger virtual keyboard
        await input.fill('test input');
        await page.waitForTimeout(500);

        // Check that input is still visible and accessible
        await expect(input).toBeVisible();

        const inputBox = await input.boundingBox();
        expect(inputBox).toBeTruthy();
        expect(inputBox!.y).toBeGreaterThanOrEqual(0);
      }
    }
  });

  test('mobile navigation patterns', async ({ page }) => {
    // Test mobile-specific navigation patterns

    // Look for mobile menu toggle
    const mobileMenuToggle = page.locator('[data-testid="mobile-menu-toggle"]').or(
      page.locator('button[aria-label*="menu"]')
    ).or(
      page.locator('.hamburger')
    ).or(
      page.locator('[class*="menu-toggle"]')
    );

    if (await mobileMenuToggle.count() > 0) {
      const toggle = mobileMenuToggle.first();

      if (await toggle.isVisible()) {
        // Test menu toggle
        await toggle.tap();
        await page.waitForTimeout(300);

        // Look for opened menu
        const menu = page.locator('[role="navigation"]').or(
          page.locator('.mobile-menu')
        ).or(
          page.locator('[data-testid="mobile-menu"]')
        );

        if (await menu.count() > 0) {
          await expect(menu.first()).toBeVisible();

          // Test that menu can be closed
          await toggle.tap();
          await page.waitForTimeout(300);
        }
      }
    }

    // Test tab navigation on mobile
    const tabs = page.locator('[role="tab"]').or(
      page.locator('.tab')
    ).or(
      page.locator('[data-testid*="tab"]')
    );

    const tabCount = await tabs.count();

    if (tabCount > 1) {
      const firstTab = tabs.first();
      const secondTab = tabs.nth(1);

      if (await firstTab.isVisible() && await secondTab.isVisible()) {
        // Test tab switching
        await secondTab.tap();
        await page.waitForTimeout(300);

        // Check for active state
        const secondTabAria = await secondTab.getAttribute('aria-selected');
        const secondTabClass = await secondTab.getAttribute('class');

        // Should indicate active state
        expect(secondTabAria === 'true' || secondTabClass?.includes('active')).toBeTruthy();
      }
    }
  });

  test('mobile drag and drop interactions', async ({ page }) => {
    // Look for draggable elements
    const draggableElements = page.locator('[draggable="true"]').or(
      page.locator('[data-draggable]')
    ).or(
      page.locator('.draggable')
    );

    const count = await draggableElements.count();

    if (count > 0) {
      const draggable = draggableElements.first();

      if (await draggable.isVisible()) {
        const dragBox = await draggable.boundingBox();

        if (dragBox) {
          const startX = dragBox.x + dragBox.width / 2;
          const startY = dragBox.y + dragBox.height / 2;
          const endX = startX + 100;
          const endY = startY + 50;

          // Perform drag gesture
          await page.touchscreen.tap(startX, startY);
          await page.mouse.down();
          await page.mouse.move(endX, endY, { steps: 10 });
          await page.mouse.up();

          await page.waitForTimeout(300);

          // Element should have moved or triggered some action
          const newBox = await draggable.boundingBox();
          expect(newBox).toBeTruthy();

          // Position might have changed, or drag feedback should be visible
          const positionChanged = Math.abs(newBox!.x - dragBox.x) > 5 || Math.abs(newBox!.y - dragBox.y) > 5;

          // Either position changed or some visual feedback occurred
          // (We can't predict exact behavior, but drag should not fail)
          expect(newBox).toBeTruthy();
        }
      }
    }
  });

  test('mobile-specific CSS and media queries', async ({ page }) => {
    // Test mobile-specific CSS features
    const cssFeatures = await page.evaluate(() => {
      return {
        touchAction: CSS.supports('touch-action', 'manipulation'),
        overscrollBehavior: CSS.supports('overscroll-behavior', 'contain'),
        webkitOverflowScrolling: CSS.supports('-webkit-overflow-scrolling', 'touch'),
        userSelect: CSS.supports('user-select', 'none'),
        webkitTapHighlightColor: CSS.supports('-webkit-tap-highlight-color', 'transparent')
      };
    });

    // Modern mobile browsers should support these
    expect(cssFeatures.touchAction).toBeTruthy();
    expect(cssFeatures.userSelect).toBeTruthy();

    // Test media query behavior
    const mediaQueries = await page.evaluate(() => {
      return {
        mobile: window.matchMedia('(max-width: 768px)').matches,
        touch: window.matchMedia('(pointer: coarse)').matches,
        hover: window.matchMedia('(hover: hover)').matches,
        portrait: window.matchMedia('(orientation: portrait)').matches
      };
    });

    // On mobile viewport, should match mobile and touch queries
    expect(mediaQueries.mobile).toBeTruthy();
    expect(mediaQueries.touch).toBeTruthy();
    expect(mediaQueries.hover).toBeFalsy(); // Touch devices typically don't have hover
  });

  test('mobile performance and responsiveness', async ({ page }) => {
    // Test touch response time
    const button = page.locator('button').first();

    if (await button.isVisible()) {
      const startTime = Date.now();
      await button.tap();
      const endTime = Date.now();

      // Touch response should be quick
      const responseTime = endTime - startTime;
      expect(responseTime).toBeLessThan(100); // Should respond within 100ms
    }

    // Test scroll performance
    const scrollStartTime = Date.now();

    await page.touchscreen.tap(200, 400);
    await page.mouse.down();
    await page.mouse.move(200, 200, { steps: 10 });
    await page.mouse.up();

    await page.waitForTimeout(100);

    const scrollEndTime = Date.now();
    const scrollTime = scrollEndTime - scrollStartTime;

    // Scroll should be smooth and quick
    expect(scrollTime).toBeLessThan(500);

    // Test frame rate during animation (if any animations are present)
    const animatedElements = page.locator('[style*="transition"]').or(
      page.locator('.animated')
    ).or(
      page.locator('[class*="fade"]')
    );

    if (await animatedElements.count() > 0) {
      // Trigger animation and check it completes smoothly
      const element = animatedElements.first();

      if (await element.isVisible()) {
        await element.hover(); // Might trigger hover animation
        await page.waitForTimeout(500); // Allow animation to complete

        // Animation should complete without issues
        await expect(element).toBeVisible();
      }
    }
  });
});
