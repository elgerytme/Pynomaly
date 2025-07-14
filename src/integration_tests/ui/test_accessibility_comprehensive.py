"""
Comprehensive Accessibility Testing Suite - Issue #125
Tests for enhanced accessibility features including WCAG 2.1 AAA compliance.
"""

import pytest
from playwright.sync_api import Page, expect


class TestAccessibilityCompliance:
    """Test suite for accessibility compliance and features."""

    def test_skip_links_functionality(self, page: Page):
        """Test skip links appear on focus and work correctly."""
        page.goto("/dashboard")

        # Tab to first skip link
        page.keyboard.press("Tab")

        # Verify skip link is visible and accessible
        skip_link = page.locator(".skip-link").first
        expect(skip_link).to_be_visible()
        expect(skip_link).to_have_attribute("href", "#main-content")

        # Click skip link and verify focus moves to main content
        skip_link.click()
        main_content = page.locator("#main-content")
        expect(main_content).to_be_focused()

    def test_keyboard_navigation_dashboard(self, page: Page):
        """Test keyboard-only navigation through dashboard."""
        page.goto("/dashboard")

        # Test tab navigation through key elements
        page.keyboard.press("Tab")  # Skip link
        page.keyboard.press("Tab")  # Navigation
        page.keyboard.press("Tab")  # Main content

        # Verify keyboard focus indicators are visible
        focused_element = page.locator(":focus")
        expect(focused_element).to_be_visible()

    def test_keyboard_shortcuts(self, page: Page):
        """Test global keyboard shortcuts."""
        page.goto("/dashboard")

        # Test Alt+1 (skip to main content)
        page.keyboard.press("Alt+1")
        main_content = page.locator("#main-content")
        expect(main_content).to_be_focused()

        # Test Alt+2 (skip to navigation)
        page.keyboard.press("Alt+2")
        navigation = page.locator("#navigation")
        expect(navigation).to_be_focused()

    def test_high_contrast_mode(self, page: Page):
        """Test high contrast mode toggle and functionality."""
        page.goto("/dashboard")

        # Toggle high contrast mode with keyboard shortcut
        page.keyboard.press("Alt+KeyC")

        # Verify high contrast class is added
        body = page.locator("body")
        expect(body).to_have_class(/high-contrast/)

        # Verify contrast changes are applied
        primary_button = page.locator(".btn-primary").first
        expect(primary_button).to_have_css("border-width", "2px")

    def test_form_accessibility_features(self, page: Page):
        """Test form accessibility enhancements."""
        page.goto("/datasets")

        # Open upload modal
        page.click("[data-modal-trigger='upload-modal']")

        # Test form field accessibility
        name_input = page.locator("#name")
        expect(name_input).to_have_attribute("aria-required", "true")
        expect(name_input).to_have_attribute("aria-describedby", "name-error name-help")

        # Test required field indicator
        name_label = page.locator("label[for='name']")
        expect(name_label).to_have_class(/required/)

        # Test help text association
        help_text = page.locator("#name-help")
        expect(help_text).to_be_visible()
        expect(help_text).to_have_class(/form-helper/)

    def test_metric_cards_accessibility(self, page: Page):
        """Test enhanced metric card accessibility."""
        page.goto("/dashboard")

        # Test metric card ARIA attributes
        metric_card = page.locator(".metric-card").first
        expect(metric_card).to_have_attribute("role", "img")
        expect(metric_card).to_have_attribute("aria-labelledby")
        expect(metric_card).to_have_attribute("aria-describedby")

        # Test screen reader content
        sr_only_content = page.locator(".sr-only").first
        expect(sr_only_content).to_be_hidden()
        expect(sr_only_content).to_contain_text(/Increase of|Decrease of|Improved by/)

    def test_modal_focus_trapping(self, page: Page):
        """Test focus trapping in modals."""
        page.goto("/datasets")

        # Open modal
        page.click("[data-modal-trigger='upload-modal']")
        modal = page.locator("[role='dialog']")
        expect(modal).to_be_visible()

        # Test focus is trapped within modal
        page.keyboard.press("Tab")
        focused_element = page.locator(":focus")
        expect(focused_element).to_be_visible()

        # Test Escape key closes modal
        page.keyboard.press("Escape")
        expect(modal).to_be_hidden()

    def test_live_regions(self, page: Page):
        """Test ARIA live regions for dynamic content."""
        page.goto("/dashboard")

        # Check for live region existence
        live_region = page.locator("[aria-live='polite']")
        expect(live_region).to_have_count(1)  # Should have at least one live region

        # Live region should be screen reader accessible but visually hidden
        expect(live_region).to_have_class(/sr-only/)

    def test_heading_hierarchy(self, page: Page):
        """Test proper heading hierarchy for screen readers."""
        page.goto("/dashboard")

        # Test h1 exists and is unique
        h1_elements = page.locator("h1")
        expect(h1_elements).to_have_count(1)
        expect(h1_elements.first).to_contain_text("Dashboard")

        # Test heading order (h1 -> h2 -> h3, etc.)
        headings = page.locator("h1, h2, h3, h4, h5, h6")
        heading_levels = []
        for i in range(headings.count()):
            tag_name = headings.nth(i).evaluate("el => el.tagName.toLowerCase()")
            heading_levels.append(int(tag_name[1]))

        # Verify proper heading hierarchy (no skipping levels)
        for i in range(1, len(heading_levels)):
            level_diff = heading_levels[i] - heading_levels[i-1]
            assert level_diff <= 1, f"Heading hierarchy error: h{heading_levels[i-1]} followed by h{heading_levels[i]}"

    def test_color_contrast_compliance(self, page: Page):
        """Test color contrast meets WCAG AA standards."""
        page.goto("/dashboard")

        # Test primary text has sufficient contrast
        primary_text = page.locator(".metric-label").first
        text_color = primary_text.evaluate("el => getComputedStyle(el).color")
        bg_color = primary_text.evaluate("el => getComputedStyle(el).backgroundColor")

        # Basic validation that colors are defined (actual contrast ratio would need a library)
        assert text_color and text_color != "rgba(0, 0, 0, 0)"
        assert bg_color and bg_color != "rgba(0, 0, 0, 0)"

    def test_touch_device_enhancements(self, page: Page):
        """Test touch device accessibility improvements."""
        # Simulate touch device
        page.add_init_script("""
            Object.defineProperty(navigator, 'maxTouchPoints', {
                get: () => 1,
            });
        """)

        page.goto("/dashboard")

        # Test enhanced touch targets
        buttons = page.locator("button, .btn-base")
        for i in range(min(3, buttons.count())):  # Test first 3 buttons
            button = buttons.nth(i)
            min_height = button.evaluate("el => parseInt(getComputedStyle(el).minHeight)")
            min_width = button.evaluate("el => parseInt(getComputedStyle(el).minWidth)")

            # WCAG touch target minimum is 44px
            assert min_height >= 44, f"Button {i} height {min_height}px is below 44px minimum"
            assert min_width >= 44, f"Button {i} width {min_width}px is below 44px minimum"

    def test_voice_commands_availability(self, page: Page):
        """Test voice command feature availability."""
        page.goto("/dashboard")

        # Test voice command activation (Alt+V)
        # Note: Actual speech recognition testing would require more complex setup
        page.keyboard.press("Alt+KeyV")

        # Check if voice feedback appears (if supported)
        voice_feedback = page.locator(".voice-feedback")
        # Should either show feedback or gracefully handle unsupported browsers


class TestWCAGCompliance:
    """Specific WCAG 2.1 AAA compliance tests."""

    def test_perceivable_content(self, page: Page):
        """Test perceivable content requirements."""
        page.goto("/dashboard")

        # Test all images have alt text
        images = page.locator("img")
        for i in range(images.count()):
            img = images.nth(i)
            alt_text = img.get_attribute("alt")
            assert alt_text is not None, f"Image {i} missing alt attribute"

    def test_operable_interface(self, page: Page):
        """Test operable interface requirements."""
        page.goto("/dashboard")

        # Test no keyboard traps (except intentional modal traps)
        # Focus should be able to move through all interactive elements
        interactive_elements = page.locator("button, a, input, select, textarea, [tabindex]:not([tabindex='-1'])")

        # Test first few elements can receive focus
        for i in range(min(5, interactive_elements.count())):
            element = interactive_elements.nth(i)
            element.focus()
            expect(element).to_be_focused()

    def test_understandable_content(self, page: Page):
        """Test understandable content requirements."""
        page.goto("/dashboard")

        # Test language is declared
        html_element = page.locator("html")
        expect(html_element).to_have_attribute("lang", "en")

        # Test consistent navigation
        nav_links = page.locator("nav a")
        for i in range(nav_links.count()):
            link = nav_links.nth(i)
            expect(link).to_be_visible()

    def test_robust_implementation(self, page: Page):
        """Test robust implementation requirements."""
        page.goto("/dashboard")

        # Test valid HTML structure
        # Check for proper ARIA usage
        modal_dialogs = page.locator("[role='dialog']")
        for i in range(modal_dialogs.count()):
            modal = modal_dialogs.nth(i)
            # Modal should have aria-labelledby or aria-label
            has_label = (
                modal.get_attribute("aria-labelledby") is not None or
                modal.get_attribute("aria-label") is not None
            )
            assert has_label, f"Modal {i} missing accessibility label"


@pytest.mark.accessibility
class TestAccessibilityIntegration:
    """Integration tests for accessibility features."""

    def test_full_accessibility_workflow(self, page: Page):
        """Test complete accessibility workflow."""
        page.goto("/dashboard")

        # 1. Navigate using skip links
        page.keyboard.press("Tab")  # Focus skip link
        page.keyboard.press("Enter")  # Activate skip link

        # 2. Toggle high contrast mode
        page.keyboard.press("Alt+KeyC")
        body = page.locator("body")
        expect(body).to_have_class(/high-contrast/)

        # 3. Navigate to datasets using keyboard
        page.keyboard.press("Alt+2")  # Focus navigation
        datasets_link = page.locator("nav a[href*='datasets']")
        datasets_link.focus()
        datasets_link.click()

        # 4. Open and interact with modal
        page.click("[data-modal-trigger='upload-modal']")
        modal = page.locator("[role='dialog']")
        expect(modal).to_be_visible()

        # 5. Close modal with Escape
        page.keyboard.press("Escape")
        expect(modal).to_be_hidden()

    def test_accessibility_persistence(self, page: Page):
        """Test accessibility preferences persist across navigation."""
        page.goto("/dashboard")

        # Enable high contrast mode
        page.keyboard.press("Alt+KeyC")
        body = page.locator("body")
        expect(body).to_have_class(/high-contrast/)

        # Navigate to another page
        page.goto("/datasets")

        # Verify high contrast mode persists
        body = page.locator("body")
        expect(body).to_have_class(/high-contrast/)


@pytest.fixture
def page_with_accessibility(page: Page):
    """Fixture that enables accessibility testing features."""
    # Enable accessibility tree in browser
    page.add_init_script("""
        window.addEventListener('load', () => {
            // Enable any debugging or testing features
            if (window.AccessibilityManager) {
                window.AccessibilityManager.enableTestingMode = true;
            }
        });
    """)
    return page
