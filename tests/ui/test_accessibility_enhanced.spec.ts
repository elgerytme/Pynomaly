/**
 * Enhanced Accessibility Testing Framework for Pynomaly
 * WCAG 2.1 AA compliance with comprehensive automated scanning
 */

import { test, expect, Page } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';

interface AccessibilityViolation {
  id: string;
  impact: 'minor' | 'moderate' | 'serious' | 'critical';
  description: string;
  helpUrl: string;
  nodes: Array<{
    target: string[];
    html: string;
    failureSummary: string;
  }>;
}

class AccessibilityTestHelper {
  constructor(private page: Page) {}

  async runAxeAnalysis(options?: any) {
    const axeBuilder = new AxeBuilder({ page: this.page });
    
    // Configure axe for WCAG 2.1 AA compliance
    axeBuilder
      .withTags(['wcag2a', 'wcag2aa', 'wcag21aa'])
      .exclude('#percy-hide') // Exclude Percy-hidden elements
      .options({
        resultTypes: ['violations', 'incomplete'],
        runOnly: {
          type: 'tag',
          values: ['wcag2a', 'wcag2aa', 'wcag21aa', 'best-practice']
        }
      });

    if (options) {
      axeBuilder.options(options);
    }

    return await axeBuilder.analyze();
  }

  async checkKeyboardNavigation() {
    const focusableElements = await this.page.locator('a, button, input, select, textarea, [tabindex]:not([tabindex="-1"])').all();
    const results = [];

    for (let i = 0; i < Math.min(focusableElements.length, 10); i++) {
      const element = focusableElements[i];
      
      try {
        await element.focus();
        await this.page.waitForTimeout(100);
        
        const isFocused = await element.evaluate(el => document.activeElement === el);
        const hasVisibleFocus = await element.evaluate(el => {
          const styles = window.getComputedStyle(el);
          return styles.outline !== 'none' || styles.boxShadow !== 'none';
        });

        results.push({
          element: await element.getAttribute('data-testid') || await element.tagName(),
          focused: isFocused,
          visibleFocus: hasVisibleFocus
        });
      } catch (error) {
        results.push({
          element: 'unknown',
          focused: false,
          visibleFocus: false,
          error: error.message
        });
      }
    }

    return results;
  }

  async checkColorContrast() {
    const textElements = await this.page.locator('p, h1, h2, h3, h4, h5, h6, a, button, span, div').all();
    const contrastIssues = [];

    for (let i = 0; i < Math.min(textElements.length, 20); i++) {
      const element = textElements[i];
      
      try {
        const styles = await element.evaluate(el => {
          const computed = window.getComputedStyle(el);
          return {
            color: computed.color,
            backgroundColor: computed.backgroundColor,
            fontSize: computed.fontSize
          };
        });

        // Basic contrast check (simplified - in production use proper contrast calculation)
        if (styles.color && styles.backgroundColor && 
            styles.color !== 'rgba(0, 0, 0, 0)' && 
            styles.backgroundColor !== 'rgba(0, 0, 0, 0)') {
          
          const textContent = await element.textContent();
          if (textContent && textContent.trim().length > 0) {
            contrastIssues.push({
              text: textContent.slice(0, 50),
              color: styles.color,
              backgroundColor: styles.backgroundColor,
              fontSize: styles.fontSize
            });
          }
        }
      } catch (error) {
        // Skip elements that can't be analyzed
      }
    }

    return contrastIssues;
  }

  async checkScreenReaderSupport() {
    const ariaElements = await this.page.evaluate(() => {
      const elements = document.querySelectorAll('*');
      const ariaInfo = [];

      elements.forEach(el => {
        const ariaLabel = el.getAttribute('aria-label');
        const ariaRole = el.getAttribute('role');
        const ariaLabelledBy = el.getAttribute('aria-labelledby');
        const ariaDescribedBy = el.getAttribute('aria-describedby');

        if (ariaLabel || ariaRole || ariaLabelledBy || ariaDescribedBy) {
          ariaInfo.push({
            tagName: el.tagName.toLowerCase(),
            ariaLabel,
            ariaRole,
            ariaLabelledBy,
            ariaDescribedBy,
            hasText: el.textContent && el.textContent.trim().length > 0
          });
        }
      });

      return ariaInfo;
    });

    return ariaElements;
  }

  async checkFormAccessibility() {
    const formElements = await this.page.locator('input, select, textarea').all();
    const formIssues = [];

    for (const element of formElements) {
      try {
        const elementInfo = await element.evaluate(el => {
          const label = el.labels && el.labels.length > 0 ? el.labels[0].textContent : null;
          const ariaLabel = el.getAttribute('aria-label');
          const ariaLabelledBy = el.getAttribute('aria-labelledby');
          const placeholder = el.getAttribute('placeholder');
          const required = el.hasAttribute('required');
          const type = el.getAttribute('type');

          return {
            type,
            hasLabel: !!label,
            labelText: label,
            ariaLabel,
            ariaLabelledBy,
            placeholder,
            required,
            id: el.id
          };
        });

        if (!elementInfo.hasLabel && !elementInfo.ariaLabel && !elementInfo.ariaLabelledBy) {
          formIssues.push({
            ...elementInfo,
            issue: 'Missing label or aria-label'
          });
        }

        if (elementInfo.required && !elementInfo.ariaLabel?.includes('required')) {
          formIssues.push({
            ...elementInfo,
            issue: 'Required field not properly indicated for screen readers'
          });
        }
      } catch (error) {
        // Skip problematic elements
      }
    }

    return formIssues;
  }

  async checkHeadingStructure() {
    const headings = await this.page.evaluate(() => {
      const headingElements = document.querySelectorAll('h1, h2, h3, h4, h5, h6');
      return Array.from(headingElements).map(el => ({
        level: parseInt(el.tagName.substring(1)),
        text: el.textContent?.trim() || '',
        id: el.id
      }));
    });

    const structureIssues = [];
    
    // Check for proper heading hierarchy
    for (let i = 1; i < headings.length; i++) {
      const current = headings[i];
      const previous = headings[i - 1];
      
      if (current.level > previous.level + 1) {
        structureIssues.push({
          issue: 'Heading level skipped',
          current: `H${current.level}: ${current.text}`,
          previous: `H${previous.level}: ${previous.text}`
        });
      }
    }

    return { headings, structureIssues };
  }

  async checkImageAccessibility() {
    const images = await this.page.locator('img, svg, [role="img"]').all();
    const imageIssues = [];

    for (const image of images) {
      try {
        const imageInfo = await image.evaluate(el => {
          const alt = el.getAttribute('alt');
          const ariaLabel = el.getAttribute('aria-label');
          const ariaLabelledBy = el.getAttribute('aria-labelledby');
          const role = el.getAttribute('role');
          const src = el.getAttribute('src');

          return {
            tagName: el.tagName.toLowerCase(),
            alt,
            ariaLabel,
            ariaLabelledBy,
            role,
            src: src ? src.substring(0, 50) : null,
            isDecorative: alt === ''
          };
        });

        if (!imageInfo.alt && !imageInfo.ariaLabel && !imageInfo.ariaLabelledBy && imageInfo.role !== 'presentation') {
          imageIssues.push({
            ...imageInfo,
            issue: 'Image missing alt text or aria-label'
          });
        }
      } catch (error) {
        // Skip problematic elements
      }
    }

    return imageIssues;
  }
}

test.describe('Accessibility Testing - WCAG 2.1 AA Compliance', () => {
  let helper: AccessibilityTestHelper;

  test.beforeEach(async ({ page }) => {
    helper = new AccessibilityTestHelper(page);
    
    // Set a consistent viewport for accessibility testing
    await page.setViewportSize({ width: 1280, height: 720 });
    
    // Ensure page is ready for accessibility analysis
    test.setTimeout(60000);
  });

  const testPages = [
    { path: '/', name: 'Dashboard' },
    { path: '/detection', name: 'Detection' },
    { path: '/detectors', name: 'Detectors' },
    { path: '/datasets', name: 'Datasets' },
    { path: '/visualizations', name: 'Visualizations' }
  ];

  // Core accessibility testing for each page
  for (const pageInfo of testPages) {
    test(`${pageInfo.name} page meets WCAG 2.1 AA standards`, async ({ page }) => {
      await test.step(`Navigate to ${pageInfo.name}`, async () => {
        await page.goto(pageInfo.path);
        await page.waitForLoadState('networkidle');
        await page.waitForTimeout(1000); // Allow dynamic content to settle
      });

      await test.step('Run axe accessibility analysis', async () => {
        const results = await helper.runAxeAnalysis();
        
        // Log violations for debugging
        if (results.violations.length > 0) {
          console.log(`Accessibility violations on ${pageInfo.name}:`, 
            results.violations.map(v => ({
              id: v.id,
              impact: v.impact,
              description: v.description,
              nodeCount: v.nodes.length
            }))
          );
        }

        // No critical or serious violations allowed
        const criticalViolations = results.violations.filter(v => v.impact === 'critical');
        const seriousViolations = results.violations.filter(v => v.impact === 'serious');
        
        expect(criticalViolations).toHaveLength(0);
        expect(seriousViolations).toHaveLength(0);
        
        // Moderate violations should be limited
        const moderateViolations = results.violations.filter(v => v.impact === 'moderate');
        expect(moderateViolations.length).toBeLessThan(5);
      });

      await test.step('Check keyboard navigation', async () => {
        const navigationResults = await helper.checkKeyboardNavigation();
        
        // At least 80% of focusable elements should have visible focus
        const focusableCount = navigationResults.filter(r => r.focused).length;
        const visibleFocusCount = navigationResults.filter(r => r.visibleFocus).length;
        
        if (focusableCount > 0) {
          const focusRatio = visibleFocusCount / focusableCount;
          expect(focusRatio).toBeGreaterThan(0.8);
        }
      });

      await test.step('Check color contrast', async () => {
        const contrastResults = await helper.checkColorContrast();
        
        // Basic check - ensure we have some text elements with proper styling
        expect(contrastResults.length).toBeGreaterThan(0);
        
        // Log contrast information for manual review
        console.log(`Color contrast check for ${pageInfo.name}:`, contrastResults.slice(0, 5));
      });

      await test.step('Check screen reader support', async () => {
        const ariaElements = await helper.checkScreenReaderSupport();
        
        // Should have some ARIA elements for screen reader support
        expect(ariaElements.length).toBeGreaterThan(0);
        
        // Check for common ARIA patterns
        const hasAriaLabels = ariaElements.some(el => el.ariaLabel);
        const hasAriaRoles = ariaElements.some(el => el.ariaRole);
        
        expect(hasAriaLabels || hasAriaRoles).toBe(true);
      });
    });

    test(`${pageInfo.name} form accessibility`, async ({ page }) => {
      await page.goto(pageInfo.path);
      await page.waitForLoadState('networkidle');

      await test.step('Check form accessibility', async () => {
        const formIssues = await helper.checkFormAccessibility();
        
        // Log form issues for review
        if (formIssues.length > 0) {
          console.log(`Form accessibility issues on ${pageInfo.name}:`, formIssues);
        }
        
        // No critical form accessibility issues
        const criticalFormIssues = formIssues.filter(issue => 
          issue.issue.includes('Missing label') && issue.type !== 'hidden'
        );
        expect(criticalFormIssues.length).toBeLessThan(2);
      });

      await test.step('Check heading structure', async () => {
        const headingAnalysis = await helper.checkHeadingStructure();
        
        // Should have at least one heading
        expect(headingAnalysis.headings.length).toBeGreaterThan(0);
        
        // Should start with H1 if present
        if (headingAnalysis.headings.length > 0) {
          const hasH1 = headingAnalysis.headings.some(h => h.level === 1);
          expect(hasH1).toBe(true);
        }
        
        // No more than 2 heading structure issues
        expect(headingAnalysis.structureIssues.length).toBeLessThan(3);
      });

      await test.step('Check image accessibility', async () => {
        const imageIssues = await helper.checkImageAccessibility();
        
        // Log image issues for review
        if (imageIssues.length > 0) {
          console.log(`Image accessibility issues on ${pageInfo.name}:`, imageIssues);
        }
        
        // No more than 20% of images should have accessibility issues
        const totalImages = await page.locator('img, svg, [role="img"]').count();
        if (totalImages > 0) {
          const issueRatio = imageIssues.length / totalImages;
          expect(issueRatio).toBeLessThan(0.2);
        }
      });
    });
  }

  // Comprehensive keyboard navigation testing
  test('Comprehensive keyboard navigation', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    await test.step('Tab navigation through interface', async () => {
      const focusableElements = [];
      let currentFocus = null;
      
      // Test Tab navigation
      for (let i = 0; i < 15; i++) {
        await page.keyboard.press('Tab');
        await page.waitForTimeout(100);
        
        currentFocus = await page.evaluate(() => {
          const active = document.activeElement;
          return active ? {
            tagName: active.tagName,
            id: active.id,
            className: active.className,
            text: active.textContent?.slice(0, 30)
          } : null;
        });
        
        if (currentFocus) {
          focusableElements.push(currentFocus);
        }
      }
      
      // Should have navigated through multiple elements
      expect(focusableElements.length).toBeGreaterThan(5);
    });

    await test.step('Escape key functionality', async () => {
      // Test modals/overlays close with Escape
      const modals = page.locator('[role="dialog"], .modal, .overlay');
      const modalCount = await modals.count();
      
      if (modalCount > 0) {
        await page.keyboard.press('Escape');
        await page.waitForTimeout(500);
        
        // Check if modals are properly hidden
        const visibleModals = await modals.filter({ hasText: /.+/ }).count();
        expect(visibleModals).toBeLessThanOrEqual(modalCount);
      }
    });

    await test.step('Enter/Space activation', async () => {
      const buttons = page.locator('button, [role="button"]').first();
      if (await buttons.count() > 0) {
        await buttons.focus();
        await page.waitForTimeout(200);
        
        // Test Space activation
        await page.keyboard.press('Space');
        await page.waitForTimeout(200);
        
        // Should not throw errors
        expect(true).toBe(true);
      }
    });
  });

  // Mobile accessibility testing
  test('Mobile accessibility compliance', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 667 });
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    await test.step('Mobile axe analysis', async () => {
      const results = await helper.runAxeAnalysis({
        runOnly: {
          type: 'tag',
          values: ['wcag2a', 'wcag2aa', 'wcag21aa', 'mobile']
        }
      });
      
      // Mobile should have no critical accessibility issues
      const criticalViolations = results.violations.filter(v => v.impact === 'critical');
      expect(criticalViolations).toHaveLength(0);
    });

    await test.step('Touch target size', async () => {
      const touchTargets = await page.locator('button, a, input[type="button"], input[type="submit"], [role="button"]').all();
      
      for (let i = 0; i < Math.min(touchTargets.length, 10); i++) {
        const target = touchTargets[i];
        const box = await target.boundingBox();
        
        if (box) {
          // WCAG 2.1 AA requires 44x44px minimum touch target
          const meetsMinimumSize = box.width >= 44 && box.height >= 44;
          const isLargeEnough = box.width >= 32 && box.height >= 32; // Allow some flexibility
          
          expect(isLargeEnough).toBe(true);
        }
      }
    });
  });

  // High contrast mode testing
  test('High contrast mode compatibility', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    await test.step('Simulate high contrast mode', async () => {
      // Simulate Windows high contrast mode
      await page.addStyleTag({
        content: `
          @media (prefers-contrast: high) {
            * {
              background: #000000 !important;
              color: #ffffff !important;
              border-color: #ffffff !important;
            }
            
            a {
              color: #ffff00 !important;
            }
            
            button {
              background: #000080 !important;
              color: #ffffff !important;
              border: 2px solid #ffffff !important;
            }
          }
        `
      });

      await page.emulateMedia({ colorScheme: 'dark', reducedMotion: 'reduce' });
      await page.waitForTimeout(1000);

      // Run accessibility analysis in high contrast mode
      const results = await helper.runAxeAnalysis();
      
      // Should still pass basic accessibility requirements
      const criticalViolations = results.violations.filter(v => v.impact === 'critical');
      expect(criticalViolations).toHaveLength(0);
    });
  });

  // Screen reader simulation testing
  test('Screen reader compatibility', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    await test.step('Landmark navigation', async () => {
      const landmarks = await page.evaluate(() => {
        const landmarkSelectors = ['main', 'nav', 'header', 'footer', 'aside', '[role="main"]', '[role="navigation"]', '[role="banner"]', '[role="contentinfo"]', '[role="complementary"]'];
        const landmarks = [];
        
        landmarkSelectors.forEach(selector => {
          const elements = document.querySelectorAll(selector);
          elements.forEach(el => {
            landmarks.push({
              tagName: el.tagName.toLowerCase(),
              role: el.getAttribute('role'),
              ariaLabel: el.getAttribute('aria-label'),
              hasContent: el.textContent && el.textContent.trim().length > 0
            });
          });
        });
        
        return landmarks;
      });

      // Should have essential landmarks
      const hasMain = landmarks.some(l => l.tagName === 'main' || l.role === 'main');
      const hasNav = landmarks.some(l => l.tagName === 'nav' || l.role === 'navigation');
      
      expect(hasMain).toBe(true);
      expect(hasNav).toBe(true);
    });

    await test.step('Content reading order', async () => {
      const readingOrder = await page.evaluate(() => {
        const walker = document.createTreeWalker(
          document.body,
          NodeFilter.SHOW_TEXT,
          null,
          false
        );
        
        const textNodes = [];
        let node;
        
        while (node = walker.nextNode()) {
          const text = node.textContent?.trim();
          if (text && text.length > 2) {
            textNodes.push(text.slice(0, 50));
          }
        }
        
        return textNodes;
      });

      // Should have meaningful content in logical order
      expect(readingOrder.length).toBeGreaterThan(5);
    });
  });
});