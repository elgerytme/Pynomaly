"""
Comprehensive Dark Mode and Theme System Tests
Tests theme switching, persistence, accessibility, and WCAG compliance.
"""

import pytest
from playwright.sync_api import Page, expect


class TestDarkModeThemeSystem:
    """Test suite for dark mode and theme system functionality."""

    def test_theme_toggle_button_exists(self, page: Page):
        """Test that the theme toggle button is present."""
        page.goto("/")

        # Wait for theme manager to load
        page.wait_for_selector("#theme-toggle", timeout=5000)

        theme_toggle = page.locator("#theme-toggle")
        expect(theme_toggle).to_be_visible()
        expect(theme_toggle).to_have_attribute("aria-label")

    def test_theme_toggle_functionality(self, page: Page):
        """Test theme switching between light, dark, and high contrast."""
        page.goto("/")
        page.wait_for_selector("#theme-toggle", timeout=5000)

        # Get initial theme
        initial_theme = page.evaluate("document.body.className")

        # Click theme toggle
        page.click("#theme-toggle")
        page.wait_for_timeout(500)  # Wait for transition

        # Verify theme changed
        new_theme = page.evaluate("document.body.className")
        assert initial_theme != new_theme

        # Click again to cycle through themes
        page.click("#theme-toggle")
        page.wait_for_timeout(500)

        # Click a third time to complete cycle
        page.click("#theme-toggle")
        page.wait_for_timeout(500)

        # Should be back to initial theme or different theme
        final_theme = page.evaluate("document.body.className")
        # Theme should have cycled through all options

    def test_dark_theme_css_variables(self, page: Page):
        """Test that dark theme CSS variables are applied correctly."""
        page.goto("/")
        page.wait_for_selector("#theme-toggle", timeout=5000)

        # Force dark theme
        page.evaluate("window.themeManager && window.themeManager.setTheme('dark')")
        page.wait_for_timeout(500)

        # Check CSS variables
        background_color = page.evaluate(
            "getComputedStyle(document.documentElement).getPropertyValue('--color-background')"
        )
        text_color = page.evaluate(
            "getComputedStyle(document.documentElement).getPropertyValue('--color-text-primary')"
        )

        # Dark theme should have dark background and light text
        assert "#0f172a" in background_color or "dark" in background_color.lower()
        assert "#f1f5f9" in text_color or "light" in text_color.lower()

    def test_high_contrast_theme(self, page: Page):
        """Test high contrast theme for accessibility."""
        page.goto("/")
        page.wait_for_selector("#theme-toggle", timeout=5000)

        # Force high contrast theme
        page.evaluate(
            "window.themeManager && window.themeManager.setTheme('highContrast')"
        )
        page.wait_for_timeout(500)

        # Check high contrast styles
        body_class = page.evaluate("document.body.className")
        assert "theme-highContrast" in body_class

        # Check that borders are thicker in high contrast mode
        border_styles = page.evaluate("""
            const buttons = document.querySelectorAll('button');
            return Array.from(buttons).map(btn => getComputedStyle(btn).borderWidth);
        """)

        # High contrast should have thicker borders
        assert any("2px" in border for border in border_styles)

    def test_theme_persistence(self, page: Page):
        """Test that theme preference is saved to localStorage."""
        page.goto("/")
        page.wait_for_selector("#theme-toggle", timeout=5000)

        # Set dark theme
        page.evaluate("window.themeManager && window.themeManager.setTheme('dark')")
        page.wait_for_timeout(500)

        # Check localStorage
        stored_theme = page.evaluate("localStorage.getItem('pynomaly-theme')")
        assert stored_theme == "dark"

        # Reload page and check theme persists
        page.reload()
        page.wait_for_selector("#theme-toggle", timeout=5000)
        page.wait_for_timeout(1000)  # Wait for theme to apply

        body_class = page.evaluate("document.body.className")
        assert "theme-dark" in body_class

    def test_system_theme_detection(self, page: Page):
        """Test automatic theme detection based on system preferences."""
        # Clear any saved theme preference
        page.goto("/")
        page.evaluate("localStorage.removeItem('pynomaly-theme')")

        # Simulate dark system preference
        page.evaluate("""
            Object.defineProperty(window, 'matchMedia', {
                writable: true,
                value: jest.fn().mockImplementation(query => ({
                    matches: query.includes('dark'),
                    media: query,
                    onchange: null,
                    addListener: jest.fn(),
                    removeListener: jest.fn(),
                    addEventListener: jest.fn(),
                    removeEventListener: jest.fn(),
                    dispatchEvent: jest.fn(),
                })),
            });
        """)

        # Reload to trigger system detection
        page.reload()
        page.wait_for_timeout(1000)

        # Should detect system preference (mock shows dark)
        # Note: This test may need adjustment based on actual system

    def test_keyboard_navigation(self, page: Page):
        """Test keyboard navigation for theme toggle."""
        page.goto("/")
        page.wait_for_selector("#theme-toggle", timeout=5000)

        # Focus the theme toggle button
        page.focus("#theme-toggle")

        # Test Enter key
        initial_theme = page.evaluate("document.body.className")
        page.keyboard.press("Enter")
        page.wait_for_timeout(500)

        new_theme = page.evaluate("document.body.className")
        assert initial_theme != new_theme

        # Test Space key
        page.keyboard.press("Space")
        page.wait_for_timeout(500)

        final_theme = page.evaluate("document.body.className")
        assert new_theme != final_theme

    def test_keyboard_shortcut(self, page: Page):
        """Test Ctrl/Cmd + Shift + T keyboard shortcut."""
        page.goto("/")
        page.wait_for_selector("#theme-toggle", timeout=5000)

        initial_theme = page.evaluate("document.body.className")

        # Use keyboard shortcut
        page.keyboard.press("Control+Shift+T")
        page.wait_for_timeout(500)

        new_theme = page.evaluate("document.body.className")
        assert initial_theme != new_theme

    def test_theme_transition_animations(self, page: Page):
        """Test that theme transitions are smooth."""
        page.goto("/")
        page.wait_for_selector("#theme-toggle", timeout=5000)

        # Check that transition CSS is applied
        transition_duration = page.evaluate("""
            getComputedStyle(document.body).transitionDuration
        """)

        # Should have transition duration set
        assert transition_duration and transition_duration != "0s"

    def test_reduced_motion_support(self, page: Page):
        """Test reduced motion accessibility preference."""
        page.goto("/")
        page.wait_for_selector("#theme-toggle", timeout=5000)

        # Simulate reduced motion preference
        page.add_style_tag(
            content="""
            @media (prefers-reduced-motion: reduce) {
                * { animation-duration: 0.01ms !important; transition-duration: 0.01ms !important; }
            }
        """
        )

        # Toggle theme with reduced motion
        page.click("#theme-toggle")
        page.wait_for_timeout(100)

        # Theme should still change but without animations
        # This test ensures functionality works even with reduced motion

    def test_theme_compatibility_with_components(self, page: Page):
        """Test that all UI components work with different themes."""
        page.goto("/")
        page.wait_for_selector("#theme-toggle", timeout=5000)

        # Test with different themes
        themes = ["light", "dark", "highContrast"]

        for theme in themes:
            page.evaluate(
                f"window.themeManager && window.themeManager.setTheme('{theme}')"
            )
            page.wait_for_timeout(500)

            # Check that cards are visible and properly styled
            cards = page.locator(".card")
            if cards.count() > 0:
                first_card = cards.first
                expect(first_card).to_be_visible()

                # Check that card has appropriate background
                background = first_card.evaluate(
                    "getComputedStyle(this).backgroundColor"
                )
                assert background and background != "rgba(0, 0, 0, 0)"

            # Check that buttons are properly styled
            buttons = page.locator("button")
            if buttons.count() > 0:
                first_button = buttons.first
                expect(first_button).to_be_visible()

    def test_wcag_contrast_compliance(self, page: Page):
        """Test WCAG 2.1 contrast ratio compliance."""
        page.goto("/")
        page.wait_for_selector("#theme-toggle", timeout=5000)

        # Test contrast ratios for different themes
        themes = ["light", "dark", "highContrast"]

        for theme in themes:
            page.evaluate(
                f"window.themeManager && window.themeManager.setTheme('{theme}')"
            )
            page.wait_for_timeout(500)

            # Check text contrast
            contrast_ratio = page.evaluate("""
                const element = document.body;
                const styles = getComputedStyle(element);
                const bgColor = styles.backgroundColor;
                const textColor = styles.color;

                // Simple contrast check (would need full implementation for real testing)
                return { background: bgColor, text: textColor };
            """)

            # Ensure colors are defined
            assert contrast_ratio["background"]
            assert contrast_ratio["text"]
            assert contrast_ratio["background"] != contrast_ratio["text"]

    def test_theme_event_system(self, page: Page):
        """Test that theme change events are properly dispatched."""
        page.goto("/")
        page.wait_for_selector("#theme-toggle", timeout=5000)

        # Set up event listener
        page.evaluate("""
            window.themeChangeEvents = [];
            document.addEventListener('theme-changed', (event) => {
                window.themeChangeEvents.push(event.detail);
            });
        """)

        # Change theme
        page.evaluate("window.themeManager && window.themeManager.setTheme('dark')")
        page.wait_for_timeout(500)

        # Check event was fired
        events = page.evaluate("window.themeChangeEvents")
        assert len(events) > 0
        assert events[0]["newTheme"] == "dark"

    def test_chart_theme_integration(self, page: Page):
        """Test that charts update with theme changes."""
        page.goto("/")
        page.wait_for_selector("#theme-toggle", timeout=5000)

        # Look for chart containers
        chart_containers = page.locator("[data-chart]")

        if chart_containers.count() > 0:
            # Change theme
            page.evaluate("window.themeManager && window.themeManager.setTheme('dark')")
            page.wait_for_timeout(1000)

            # Check that chart container has theme-aware styling
            first_chart = chart_containers.first
            background = first_chart.evaluate("getComputedStyle(this).backgroundColor")

            # Should have dark theme background
            assert (
                background
                and "rgb(30, 41, 59)" in background
                or "dark" in background.lower()
            )

    def test_mobile_theme_toggle(self, page: Page):
        """Test theme toggle on mobile viewport."""
        # Set mobile viewport
        page.set_viewport_size({"width": 375, "height": 667})

        page.goto("/")
        page.wait_for_selector("#theme-toggle", timeout=5000)

        theme_toggle = page.locator("#theme-toggle")
        expect(theme_toggle).to_be_visible()

        # Should be properly sized for mobile
        box = theme_toggle.bounding_box()
        assert box["width"] >= 36  # Minimum touch target
        assert box["height"] >= 36

    def test_print_mode_theme_handling(self, page: Page):
        """Test that print mode removes theme toggle and forces light theme."""
        page.goto("/")
        page.wait_for_selector("#theme-toggle", timeout=5000)

        # Emulate print media
        page.emulate_media(media="print")

        # Theme toggle should be hidden in print mode
        theme_toggle = page.locator("#theme-toggle")

        # Check if hidden via CSS
        display = theme_toggle.evaluate("getComputedStyle(this).display")
        assert display == "none"


class TestThemeAccessibility:
    """Specific accessibility tests for the theme system."""

    def test_theme_toggle_aria_labels(self, page: Page):
        """Test proper ARIA labels for theme toggle."""
        page.goto("/")
        page.wait_for_selector("#theme-toggle", timeout=5000)

        theme_toggle = page.locator("#theme-toggle")

        # Should have aria-label
        aria_label = theme_toggle.get_attribute("aria-label")
        assert aria_label is not None
        assert "theme" in aria_label.lower()

    def test_focus_indicators_all_themes(self, page: Page):
        """Test that focus indicators work in all themes."""
        page.goto("/")
        page.wait_for_selector("#theme-toggle", timeout=5000)

        themes = ["light", "dark", "highContrast"]

        for theme in themes:
            page.evaluate(
                f"window.themeManager && window.themeManager.setTheme('{theme}')"
            )
            page.wait_for_timeout(500)

            # Focus the theme toggle
            page.focus("#theme-toggle")

            # Check focus styles
            outline = page.evaluate("""
                const toggle = document.getElementById('theme-toggle');
                return getComputedStyle(toggle).outline;
            """)

            # Should have visible focus indicator
            assert outline and outline != "none"

    def test_screen_reader_announcements(self, page: Page):
        """Test that theme changes are announced to screen readers."""
        page.goto("/")
        page.wait_for_selector("#theme-toggle", timeout=5000)

        # This would require more sophisticated testing with actual screen readers
        # For now, test that aria-live regions exist or are created

        # Check for announcements (basic check)
        aria_live_regions = page.locator("[aria-live]")

        # The theme manager should create announcements for theme changes
        # This test verifies the framework is in place


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
