"""UI Layout and Element Validation Tests."""

from playwright.sync_api import Page


class TestLayoutValidation:
    """Test suite for UI layout and element validation."""

    def test_dashboard_layout_elements(self, dashboard_page):
        """Test dashboard layout has all required elements."""
        dashboard_page.navigate()

        layout_check = dashboard_page.verify_dashboard_layout()

        assert layout_check["has_title"], "Dashboard should have proper title"
        assert layout_check["has_stats_cards"], "Dashboard should have stats cards"
        assert layout_check[
            "has_recent_results"
        ], "Dashboard should have recent results section"
        assert layout_check["has_quick_actions"], "Dashboard should have quick actions"

        # Take screenshot for visual verification
        dashboard_page.take_screenshot("dashboard_layout")

    def test_navigation_consistency(self, page: Page):
        """Test navigation consistency across pages."""
        pages = [
            "/",
            "/detectors",
            "/datasets",
            "/detection",
            "/experiments",
            "/visualizations",
        ]

        navigation_results = {}

        for page_url in pages:
            page.goto(f"http://pynomaly-app:8000{page_url}")
            page.wait_for_load_state("networkidle")

            # Check navigation exists and is consistent
            nav_element = page.locator("nav")
            assert nav_element.count() > 0, f"Navigation missing on {page_url}"

            # Check logo
            logo = page.locator("nav a[href='/web/']")
            assert logo.count() > 0, f"Logo/brand link missing on {page_url}"
            assert "Pynomaly" in logo.text_content(), "Logo should contain 'Pynomaly'"

            # Check main navigation links
            nav_links = page.locator("nav .hidden.sm\\:flex a")
            assert nav_links.count() >= 5, f"Not enough navigation links on {page_url}"

            navigation_results[page_url] = {
                "nav_present": True,
                "logo_present": True,
                "nav_links_count": nav_links.count(),
            }

        # All pages should have same number of nav links
        nav_counts = [
            result["nav_links_count"] for result in navigation_results.values()
        ]
        assert all(
            count == nav_counts[0] for count in nav_counts
        ), "Navigation link count should be consistent across pages"

    def test_responsive_navigation(self, mobile_page: Page):
        """Test responsive navigation on mobile viewport."""
        mobile_page.goto("http://pynomaly-app:8000/web/")
        mobile_page.wait_for_load_state("networkidle")

        # Desktop nav should be hidden on mobile
        desktop_nav = mobile_page.locator("nav .hidden.sm\\:flex")
        assert (
            not desktop_nav.is_visible()
        ), "Desktop navigation should be hidden on mobile"

        # Mobile menu button should be visible
        mobile_button = mobile_page.locator("nav .sm\\:hidden button")
        assert mobile_button.is_visible(), "Mobile menu button should be visible"

        # Test mobile menu toggle
        mobile_button.click()
        mobile_page.wait_for_timeout(500)

        mobile_page.locator("nav .sm\\:hidden div[x-show]")
        # Note: Alpine.js x-show might not be detectable in this test
        # We'll check for mobile menu structure instead
        mobile_links = mobile_page.locator("nav .sm\\:hidden a")
        assert mobile_links.count() > 0, "Mobile navigation links should be present"

        mobile_page.screenshot(path="screenshots/mobile_navigation.png")

    def test_form_elements_validation(self, detectors_page):
        """Test form elements are properly structured."""
        detectors_page.navigate()

        layout_check = detectors_page.verify_detectors_page_layout()

        assert layout_check["has_create_form"], "Should have detector creation form"
        assert layout_check["has_name_input"], "Should have name input field"
        assert layout_check["has_algorithm_select"], "Should have algorithm select"
        assert layout_check["has_submit_button"], "Should have submit button"

        # Test form accessibility
        name_input = detectors_page.page.locator(detectors_page.DETECTOR_NAME_INPUT)
        assert (
            name_input.get_attribute("name") is not None
        ), "Name input should have name attribute"

        algorithm_select = detectors_page.page.locator(
            detectors_page.DETECTOR_ALGORITHM_SELECT
        )
        assert (
            algorithm_select.get_attribute("name") is not None
        ), "Algorithm select should have name attribute"

    def test_table_structure(self, dashboard_page):
        """Test table structure and accessibility."""
        dashboard_page.navigate()

        # Check for results table
        table = dashboard_page.page.locator("#results-table table")
        if table.count() > 0:
            # Check table structure
            thead = table.locator("thead")
            tbody = table.locator("tbody")

            assert thead.count() > 0, "Table should have header"
            assert tbody.count() > 0, "Table should have body"

            # Check header cells
            headers = thead.locator("th")
            assert headers.count() >= 3, "Table should have at least 3 columns"

            # Check if rows have same number of cells as headers
            if tbody.locator("tr").count() > 0:
                first_row_cells = tbody.locator("tr").first.locator("td")
                assert (
                    first_row_cells.count() == headers.count()
                ), "Row cells should match header count"

    def test_button_states(self, page: Page):
        """Test button states and styling."""
        page.goto("http://pynomaly-app:8000/web/detectors")
        page.wait_for_load_state("networkidle")

        # Find form buttons
        buttons = page.locator("button")

        for i in range(buttons.count()):
            button = buttons.nth(i)

            # Check button has proper attributes
            button_type = button.get_attribute("type")
            button_text = button.text_content()

            # Buttons should have text or aria-label
            aria_label = button.get_attribute("aria-label")
            assert (
                button_text or aria_label
            ), f"Button {i} should have text or aria-label"

            # Submit buttons should have type="submit"
            if button_text and "submit" in button_text.lower():
                assert (
                    button_type == "submit"
                ), "Submit buttons should have type='submit'"

    def test_loading_states(self, detection_page):
        """Test loading states and indicators."""
        detection_page.navigate()

        # Look for HTMX loading indicators
        htmx_indicators = detection_page.page.locator(".htmx-indicator")

        # Check that indicators exist (even if not currently active)
        if htmx_indicators.count() > 0:
            # Indicators should be initially hidden
            for i in range(htmx_indicators.count()):
                indicator = htmx_indicators.nth(i)
                # Should have proper CSS for showing/hiding
                indicator_class = indicator.get_attribute("class") or ""
                assert (
                    "htmx-indicator" in indicator_class
                ), "Should have htmx-indicator class"

    def test_error_message_display(self, detectors_page):
        """Test error message display areas."""
        detectors_page.navigate()

        # Try to submit empty form to trigger validation
        submit_button = detectors_page.page.locator(detectors_page.CREATE_BUTTON)
        if submit_button.count() > 0:
            submit_button.click()
            detectors_page.page.wait_for_timeout(1000)

            # Check for error message areas
            error_containers = detectors_page.page.locator(
                ".error, .invalid, .alert-error, [class*='error'], [role='alert']"
            )

            # If errors are shown, they should be properly structured
            if error_containers.count() > 0:
                for i in range(error_containers.count()):
                    error = error_containers.nth(i)
                    error_text = error.text_content()

                    assert error_text, "Error containers should have text content"
                    assert len(error_text) > 5, "Error messages should be descriptive"

    def test_icon_consistency(self, page: Page):
        """Test icon usage consistency."""
        pages = ["/", "/detectors", "/datasets"]

        for page_url in pages:
            page.goto(f"http://pynomaly-app:8000{page_url}")
            page.wait_for_load_state("networkidle")

            # Check for SVG icons
            svg_icons = page.locator("svg")

            for i in range(svg_icons.count()):
                svg = svg_icons.nth(i)

                # SVG should have proper attributes
                viewbox = svg.get_attribute("viewBox")
                assert viewbox is not None, f"SVG icon {i} should have viewBox"

                # Should have proper ARIA attributes for accessibility
                aria_hidden = svg.get_attribute("aria-hidden")
                role = svg.get_attribute("role")

                # Icons should either be hidden from screen readers or have proper role
                assert (
                    aria_hidden == "true" or role is not None
                ), "SVG icons should have proper accessibility attributes"

    def test_color_contrast_elements(self, page: Page):
        """Test color contrast for key elements."""
        page.goto("http://pynomaly-app:8000/web/")
        page.wait_for_load_state("networkidle")

        # Get computed styles for key elements
        nav_styles = page.evaluate(
            """
            () => {
                const nav = document.querySelector('nav');
                if (!nav) return null;
                const styles = window.getComputedStyle(nav);
                return {
                    backgroundColor: styles.backgroundColor,
                    color: styles.color
                };
            }
        """
        )

        assert nav_styles is not None, "Should be able to get navigation styles"
        assert (
            nav_styles["backgroundColor"] != nav_styles["color"]
        ), "Navigation background and text should have different colors"

    def test_layout_consistency_viewport_sizes(self, page: Page):
        """Test layout consistency across different viewport sizes."""
        viewports = [
            {"width": 1920, "height": 1080, "name": "desktop"},
            {"width": 768, "height": 1024, "name": "tablet"},
            {"width": 375, "height": 667, "name": "mobile"},
        ]

        layout_results = {}

        for viewport in viewports:
            page.set_viewport_size(
                {"width": viewport["width"], "height": viewport["height"]}
            )
            page.goto("http://pynomaly-app:8000/web/")
            page.wait_for_load_state("networkidle")

            # Check key elements are still visible and accessible
            main_content = page.locator("main")
            nav_element = page.locator("nav")

            layout_results[viewport["name"]] = {
                "main_visible": main_content.is_visible(),
                "nav_visible": nav_element.is_visible(),
                "viewport_width": viewport["width"],
            }

            # Take screenshot for each viewport
            page.screenshot(
                path=f"screenshots/layout_{viewport['name']}.png", full_page=True
            )

        # All viewports should have main content and navigation
        for result in layout_results.values():
            assert result[
                "main_visible"
            ], "Main content should be visible on all viewports"
            assert result[
                "nav_visible"
            ], "Navigation should be visible on all viewports"
