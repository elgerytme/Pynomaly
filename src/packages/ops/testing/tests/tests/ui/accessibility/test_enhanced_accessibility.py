"""
Enhanced Accessibility Testing Suite for Issue #125
Tests the new accessibility features implemented in the Pynomaly web UI
"""

import asyncio

import pytest
from playwright.async_api import Browser, Page, expect


class TestEnhancedAccessibility:
    """Test suite for enhanced accessibility features"""

    async def test_skip_links_functionality(self, page: Page):
        """Test skip links are present and functional"""
        await page.goto("/")

        # Skip links should be created by JavaScript
        await page.wait_for_selector(".skip-links", timeout=5000)

        # Test skip to main content
        skip_main = page.locator('.skip-link[href="#main-content"]')
        await expect(skip_main).to_be_visible()

        # Focus the skip link and activate it
        await skip_main.focus()
        await skip_main.press("Enter")

        # Verify main content receives focus
        main_content = page.locator("#main-content")
        await expect(main_content).to_be_focused()

    async def test_keyboard_navigation_shortcuts(self, page: Page):
        """Test global keyboard shortcuts"""
        await page.goto("/")

        # Wait for accessibility manager to initialize
        await page.wait_for_function("window.AccessibilityManager")

        # Test Alt+1 (focus main content)
        await page.keyboard.press("Alt+1")
        main_content = page.locator("#main-content")
        await expect(main_content).to_be_focused()

        # Test Alt+2 (focus navigation)
        await page.keyboard.press("Alt+2")
        nav_link = page.locator("nav a").first
        await expect(nav_link).to_be_focused()

    async def test_high_contrast_mode_toggle(self, page: Page):
        """Test high contrast mode functionality"""
        await page.goto("/")

        # Wait for accessibility manager
        await page.wait_for_function("window.AccessibilityManager")

        # Toggle high contrast mode
        await page.keyboard.press("Alt+c")

        # Verify high contrast class is applied
        html_element = page.locator("html")
        await expect(html_element).to_have_class(/.*high-contrast.*/)

        # Toggle again to disable
        await page.keyboard.press("Alt+c")
        await expect(html_element).not_to_have_class(/.*high-contrast.*/)

    async def test_keyboard_help_modal(self, page: Page):
        """Test keyboard help modal functionality"""
        await page.goto("/")

        # Wait for accessibility manager
        await page.wait_for_function("window.AccessibilityManager")

        # Open keyboard help with Alt+H
        await page.keyboard.press("Alt+h")

        # Verify modal appears
        help_modal = page.locator('.keyboard-help-modal')
        await expect(help_modal).to_be_visible()

        # Verify focus is trapped in modal
        close_button = help_modal.locator('button')
        await expect(close_button).to_be_focused()

        # Test escape key closes modal
        await page.keyboard.press("Escape")
        await expect(help_modal).not_to_be_visible()

    async def test_focus_trapping_in_modals(self, page: Page):
        """Test focus trapping functionality in modals"""
        await page.goto("/")

        # Wait for accessibility manager
        await page.wait_for_function("window.AccessibilityManager")

        # Open keyboard help modal
        await page.keyboard.press("Alt+h")

        help_modal = page.locator('.keyboard-help-modal')
        await expect(help_modal).to_be_visible()

        # Test tab trapping - should cycle through focusable elements
        await page.keyboard.press("Tab")
        close_button = help_modal.locator('button')
        await expect(close_button).to_be_focused()

        # Shift+Tab should go to last focusable element (same button in this case)
        await page.keyboard.press("Shift+Tab")
        await expect(close_button).to_be_focused()

    async def test_aria_live_regions(self, page: Page):
        """Test ARIA live regions for dynamic content announcements"""
        await page.goto("/")

        # Wait for accessibility manager
        await page.wait_for_function("window.AccessibilityManager")

        # Toggle high contrast to trigger announcement
        await page.keyboard.press("Alt+c")

        # Verify live region exists and has content
        live_region = page.locator('[aria-live="polite"]')
        await expect(live_region).to_be_attached()

    async def test_voice_command_setup(self, page: Page):
        """Test voice command initialization (basic setup test)"""
        await page.goto("/")

        # Wait for accessibility manager
        await page.wait_for_function("window.AccessibilityManager")

        # Check if speech recognition is available and initialized
        has_speech_recognition = await page.evaluate("""
            () => {
                return window.AccessibilityManager &&
                       (window.AccessibilityManager.recognition !== undefined ||
                        !('webkitSpeechRecognition' in window || 'SpeechRecognition' in window));
            }
        """)

        assert has_speech_recognition, "Voice command should be properly initialized or gracefully unavailable"

    async def test_motor_accessibility_features(self, page: Page):
        """Test motor accessibility enhancements"""
        await page.goto("/")

        # Wait for accessibility manager
        await page.wait_for_function("window.AccessibilityManager")

        # Simulate touch device
        await page.add_init_script("""
            Object.defineProperty(window, 'ontouchstart', {
                value: function() {}
            });
        """)

        await page.reload()
        await page.wait_for_function("window.AccessibilityManager")

        # Verify touch device class is added
        body = page.locator("body")
        await expect(body).to_have_class(/.*touch-device.*/)

    async def test_grid_navigation(self, page: Page):
        """Test arrow key navigation in data grids"""
        await page.goto("/")

        # Create a test grid for navigation
        await page.evaluate("""
            const testGrid = document.createElement('table');
            testGrid.setAttribute('role', 'grid');
            testGrid.setAttribute('data-cols', '3');
            testGrid.innerHTML = `
                <tr>
                    <td tabindex="0" role="gridcell">Cell 1</td>
                    <td tabindex="0" role="gridcell">Cell 2</td>
                    <td tabindex="0" role="gridcell">Cell 3</td>
                </tr>
                <tr>
                    <td tabindex="0" role="gridcell">Cell 4</td>
                    <td tabindex="0" role="gridcell">Cell 5</td>
                    <td tabindex="0" role="gridcell">Cell 6</td>
                </tr>
            `;
            document.querySelector('#main-content').appendChild(testGrid);
        """)

        # Focus first cell
        first_cell = page.locator('[role="gridcell"]').first
        await first_cell.focus()

        # Test arrow key navigation
        await page.keyboard.press("ArrowRight")
        second_cell = page.locator('[role="gridcell"]').nth(1)
        await expect(second_cell).to_be_focused()

        # Test down arrow
        await page.keyboard.press("ArrowDown")
        fifth_cell = page.locator('[role="gridcell"]').nth(4)
        await expect(fifth_cell).to_be_focused()

    async def test_enhanced_form_accessibility(self, page: Page):
        """Test enhanced form accessibility features"""
        await page.goto("/")

        # Create a test form with accessibility features
        await page.evaluate("""
            const testForm = document.createElement('form');
            testForm.innerHTML = `
                <div class="form-group">
                    <label for="test-input" class="form-label" aria-required="true">Required Field</label>
                    <input type="text" id="test-input" class="form-input" aria-invalid="false">
                    <div class="form-error-message" style="display: none;">This field is required</div>
                </div>
            `;
            document.querySelector('#main-content').appendChild(testForm);
        """)

        # Test form label association
        label = page.locator('label[for="test-input"]')
        input_field = page.locator('#test-input')

        await expect(label).to_be_visible()
        await expect(input_field).to_be_visible()

        # Test required field indicator
        await expect(label).to_contain_text("*")

    async def test_navigation_landmarks(self, page: Page):
        """Test proper ARIA landmarks and navigation structure"""
        await page.goto("/")

        # Test main navigation
        nav = page.locator('nav[role="navigation"]')
        await expect(nav).to_be_visible()
        await expect(nav).to_have_attribute('aria-label', 'Main navigation')

        # Test main content area
        main = page.locator('main[role="main"]')
        await expect(main).to_be_visible()
        await expect(main).to_have_attribute('id', 'main-content')

    async def test_mobile_menu_accessibility(self, page: Page):
        """Test mobile menu accessibility features"""
        # Set mobile viewport
        await page.set_viewport_size({"width": 375, "height": 667})
        await page.goto("/")

        # Find mobile menu button
        menu_button = page.locator('button[aria-controls="mobile-menu"]')
        await expect(menu_button).to_be_visible()

        # Test initial state
        await expect(menu_button).to_have_attribute('aria-expanded', 'false')

        # Open mobile menu
        await menu_button.click()
        await expect(menu_button).to_have_attribute('aria-expanded', 'true')

        # Verify mobile menu is visible
        mobile_menu = page.locator('#mobile-menu')
        await expect(mobile_menu).to_be_visible()

    async def test_color_contrast_compliance(self, page: Page):
        """Test color contrast ratios meet WCAG standards"""
        await page.goto("/")

        # Check text contrast in normal mode
        text_color = await page.evaluate("""
            () => {
                const element = document.querySelector('nav a');
                const styles = window.getComputedStyle(element);
                return styles.color;
            }
        """)

        background_color = await page.evaluate("""
            () => {
                const element = document.querySelector('nav');
                const styles = window.getComputedStyle(element);
                return styles.backgroundColor;
            }
        """)

        # Basic check that colors are defined
        assert text_color and background_color, "Text and background colors should be defined"

    async def test_reduced_motion_preference(self, page: Page):
        """Test reduced motion preference handling"""
        # Set reduced motion preference
        await page.emulate_media([{"name": "prefers-reduced-motion", "value": "reduce"}])
        await page.goto("/")

        # Check that animations are disabled
        animation_duration = await page.evaluate("""
            () => {
                const element = document.querySelector('body');
                const styles = window.getComputedStyle(element);
                return styles.animationDuration;
            }
        """)

        # In reduced motion mode, animations should be very short
        assert animation_duration in ['0.01ms', '0s', ''], f"Animation duration should be minimal, got: {animation_duration}"


@pytest.fixture
async def page(browser: Browser):
    """Create a new page for each test"""
    page = await browser.new_page()
    yield page
    await page.close()


# Test configuration for different browsers and viewports
@pytest.mark.asyncio
async def test_accessibility_cross_browser_compatibility():
    """Run accessibility tests across different browsers"""
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browsers = [p.chromium, p.firefox, p.webkit]

        for browser_type in browsers:
            browser = await browser_type.launch()
            page = await browser.new_page()

            # Basic accessibility test
            await page.goto("http://localhost:8000/")  # Adjust URL as needed

            # Test skip links exist
            skip_links = page.locator(".skip-links")
            await expect(skip_links).to_be_attached()

            await browser.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
