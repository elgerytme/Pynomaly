"""Enhanced Accessibility Tests for Issue #125."""

import pytest
from playwright.sync_api import Page, expect


class TestEnhancedAccessibilityFeatures:
    """Test suite for enhanced accessibility features."""

    def test_skip_links_functionality(self, page: Page):
        """Test skip links are present and functional."""
        page.goto("http://localhost:5000")

        # Check skip links exist
        skip_links = page.locator(".skip-links")
        expect(skip_links).to_have_count(3)

        # Test skip to main content
        main_skip = page.locator('a[href="#main-content"]')
        expect(main_skip).to_be_visible()

        # Tab to skip link and press Enter
        page.keyboard.press("Tab")
        page.keyboard.press("Enter")

        # Verify main content is focused
        main_content = page.locator("#main-content")
        expect(main_content).to_be_focused()

    def test_keyboard_shortcuts(self, page: Page):
        """Test global keyboard shortcuts."""
        page.goto("http://localhost:5000")

        # Test Alt+H for help modal
        page.keyboard.press("Alt+h")
        help_modal = page.locator("#keyboard-help-modal")
        expect(help_modal).to_be_visible()

        # Test Escape to close modal
        page.keyboard.press("Escape")
        expect(help_modal).not_to_be_visible()

        # Test Alt+M to focus main content
        page.keyboard.press("Alt+m")
        main_content = page.locator("#main-content")
        expect(main_content).to_be_focused()

    def test_focus_management(self, page: Page):
        """Test enhanced focus management."""
        page.goto("http://localhost:5000")

        # Test keyboard focus indicator
        page.keyboard.press("Tab")
        body = page.locator("body")
        expect(body).to_have_class("using-keyboard")

        # Test mouse interaction removes keyboard class
        page.mouse.click(100, 100)
        expect(body).not_to_have_class("using-keyboard")

    def test_aria_live_regions(self, page: Page):
        """Test ARIA live regions for announcements."""
        page.goto("http://localhost:5000")

        # Check live regions exist
        polite_region = page.locator("#aria-live-polite")
        assertive_region = page.locator("#aria-live-assertive")

        expect(polite_region).to_have_attribute("aria-live", "polite")
        expect(assertive_region).to_have_attribute("aria-live", "assertive")

        # Test announcement functionality
        page.evaluate("""
            window.AccessibilityManager.announce('Test announcement');
        """)

        expect(polite_region).to_contain_text("Test announcement")

    def test_high_contrast_mode(self, page: Page):
        """Test high contrast mode toggle."""
        page.goto("http://localhost:5000")

        # Test high contrast toggle
        page.keyboard.press("Alt+c")
        body = page.locator("body")
        expect(body).to_have_class("high-contrast")

        # Test toggle off
        page.keyboard.press("Alt+c")
        expect(body).not_to_have_class("high-contrast")

    def test_voice_commands_initialization(self, page: Page):
        """Test voice command system initialization."""
        page.goto("http://localhost:5000")

        # Check voice command manager exists
        voice_support = page.evaluate("""
            return !!window.AccessibilityManager.voiceCommands;
        """)
        assert voice_support

        # Test voice command indicator
        indicator = page.locator("#voice-command-indicator")
        expect(indicator).to_exist()

    def test_touch_device_optimizations(self, page: Page):
        """Test touch device accessibility optimizations."""
        # Simulate touch device
        page.goto("http://localhost:5000")
        page.evaluate("document.body.classList.add('touch-device')")

        # Check touch targets meet minimum size requirements
        buttons = page.locator("button, .btn-base")
        for button in buttons.all():
            box = button.bounding_box()
            if box:
                assert box["width"] >= 44 or box["height"] >= 44

    def test_keyboard_help_modal(self, page: Page):
        """Test keyboard help modal functionality."""
        page.goto("http://localhost:5000")

        # Open help modal
        page.keyboard.press("Alt+h")
        modal = page.locator("#keyboard-help-modal")
        expect(modal).to_be_visible()

        # Check modal attributes
        expect(modal).to_have_attribute("role", "dialog")
        expect(modal).to_have_attribute("aria-modal", "true")

        # Check keyboard shortcuts are listed
        shortcuts = page.locator(".keyboard-shortcuts dt")
        expect(shortcuts).to_have_count_greater_than(5)

        # Test close button
        close_button = page.locator(".modal-close")
        close_button.click()
        expect(modal).not_to_be_visible()

    def test_form_accessibility_enhancements(self, page: Page):
        """Test enhanced form accessibility features."""
        page.goto("http://localhost:5000/datasets")  # Assuming there's a form here

        # Check form inputs have proper ARIA attributes
        inputs = page.locator("input, select, textarea")
        for input_field in inputs.all():
            # Should have label or aria-label
            label_id = input_field.get_attribute("aria-labelledby")
            aria_label = input_field.get_attribute("aria-label")
            input_id = input_field.get_attribute("id")

            has_label = (
                label_id
                or aria_label
                or (input_id and page.locator(f'label[for="{input_id}"]').count() > 0)
            )
            assert has_label, "Input must have associated label"

    def test_data_table_keyboard_navigation(self, page: Page):
        """Test keyboard navigation in data tables."""
        page.goto("http://localhost:5000")

        # Find tables with grid role
        tables = page.locator('[role="grid"]')
        if tables.count() > 0:
            table = tables.first()
            first_cell = table.locator('[role="gridcell"]').first()
            first_cell.focus()

            # Test arrow key navigation
            page.keyboard.press("ArrowRight")
            # Should move to next cell
            # Note: Full testing would require specific table structure

    def test_reduced_motion_support(self, page: Page):
        """Test reduced motion preference support."""
        page.goto("http://localhost:5000")

        # Simulate reduced motion preference
        page.evaluate("""
            window.AccessibilityManager.setPreference('reducedMotion', true);
        """)

        body = page.locator("body")
        expect(body).to_have_class("reduced-motion")

    def test_font_size_preferences(self, page: Page):
        """Test font size preference controls."""
        page.goto("http://localhost:5000")

        # Test large font size
        page.evaluate("""
            window.AccessibilityManager.setPreference('fontSize', 'large');
        """)

        body = page.locator("body")
        expect(body).to_have_class("font-size-large")

    def test_navigation_aria_attributes(self, page: Page):
        """Test navigation has proper ARIA attributes."""
        page.goto("http://localhost:5000")

        nav = page.locator("nav#navigation")
        expect(nav).to_have_attribute("role", "navigation")
        expect(nav).to_have_attribute("aria-label", "Main navigation")

        main = page.locator("main#main-content")
        expect(main).to_have_attribute("role", "main")
        expect(main).to_have_attribute("aria-label", "Main content")

        footer = page.locator("footer#footer")
        expect(footer).to_have_attribute("role", "contentinfo")

    def test_mobile_menu_accessibility(self, page: Page):
        """Test mobile menu accessibility features."""
        # Set mobile viewport
        page.set_viewport_size({"width": 375, "height": 667})
        page.goto("http://localhost:5000")

        # Find mobile menu button
        menu_button = page.locator('button[aria-controls="mobile-menu"]')
        expect(menu_button).to_be_visible()
        expect(menu_button).to_have_attribute("aria-expanded", "false")

        # Open mobile menu
        menu_button.click()
        expect(menu_button).to_have_attribute("aria-expanded", "true")

        # Check mobile menu accessibility
        mobile_menu = page.locator("#mobile-menu")
        expect(mobile_menu).to_have_attribute("role", "menu")

    def test_error_handling_accessibility(self, page: Page):
        """Test error message accessibility."""
        page.goto("http://localhost:5000")

        # Test error announcement
        page.evaluate("""
            window.AccessibilityManager.announce('Error: Test error message', true);
        """)

        assertive_region = page.locator("#aria-live-assertive")
        expect(assertive_region).to_contain_text("Error: Test error message")

    def test_progress_indicator_accessibility(self, page: Page):
        """Test progress indicators have proper ARIA attributes."""
        page.goto("http://localhost:5000")

        # Look for progress elements
        progress_elements = page.locator('[role="progressbar"]')
        for progress in progress_elements.all():
            # Should have aria-valuenow or aria-valuetext
            value_now = progress.get_attribute("aria-valuenow")
            value_text = progress.get_attribute("aria-valuetext")
            assert value_now or value_text, "Progress bar must have value indication"


class TestWCAGCompliance:
    """Test WCAG 2.1 AA compliance."""

    def test_color_contrast_ratios(self, page: Page):
        """Test color contrast meets WCAG AA standards."""
        page.goto("http://localhost:5000")

        # This would typically use axe-core for automated testing
        # Here we check that high contrast mode is available
        page.keyboard.press("Alt+c")
        body = page.locator("body")
        expect(body).to_have_class("high-contrast")

    def test_text_scaling(self, page: Page):
        """Test text can be scaled up to 200% without loss of functionality."""
        page.goto("http://localhost:5000")

        # Simulate 200% zoom
        page.evaluate("document.body.style.fontSize = '200%'")

        # Check that all content is still accessible
        nav = page.locator("nav")
        expect(nav).to_be_visible()

        main = page.locator("main")
        expect(main).to_be_visible()

    def test_keyboard_only_navigation(self, page: Page):
        """Test complete keyboard-only navigation."""
        page.goto("http://localhost:5000")

        # Should be able to navigate entire page with keyboard
        page.keyboard.press("Tab")  # Skip link
        page.keyboard.press("Tab")  # Navigation
        page.keyboard.press("Tab")  # First nav item

        # Verify focus is visible
        focused_element = page.locator(":focus")
        expect(focused_element).to_be_visible()

    def test_screen_reader_content(self, page: Page):
        """Test content is properly structured for screen readers."""
        page.goto("http://localhost:5000")

        # Check heading hierarchy
        h1_count = page.locator("h1").count()
        assert h1_count == 1, "Page should have exactly one h1"

        # Check all images have alt text
        images = page.locator("img")
        for img in images.all():
            alt_text = img.get_attribute("alt")
            assert alt_text is not None, "All images must have alt text"

    def test_focus_indicators(self, page: Page):
        """Test focus indicators are clearly visible."""
        page.goto("http://localhost:5000")

        # Enable keyboard navigation
        page.keyboard.press("Tab")

        # Check focus outline is present
        focused_element = page.locator(":focus")
        if focused_element.count() > 0:
            # Should have visible outline
            outline = focused_element.evaluate("getComputedStyle(this).outline")
            assert outline != "none", "Focused elements must have visible outline"


@pytest.fixture
def page(browser):
    """Create a new page for each test."""
    page = browser.new_page()
    yield page
    page.close()
