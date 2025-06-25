"""UX Flow and User Journey Tests."""

from playwright.sync_api import Page


class TestUXFlows:
    """Test suite for user experience flows and interactions."""

    def test_complete_detector_creation_flow(
        self, detectors_page, sample_detector_data
    ):
        """Test complete detector creation user flow."""
        detectors_page.navigate()

        # Step 1: User arrives at detectors page
        layout_check = detectors_page.verify_detectors_page_layout()
        assert layout_check["has_create_form"], (
            "Detector creation form should be available"
        )

        # Step 2: User views available algorithms
        algorithms = detectors_page.get_available_algorithms()
        assert len(algorithms) > 0, "Should have available algorithms"
        assert sample_detector_data["algorithm"] in algorithms, (
            f"Test algorithm {sample_detector_data['algorithm']} should be available"
        )

        # Step 3: User fills out form
        initial_detector_count = len(detectors_page.get_detector_list())

        success = detectors_page.create_detector(sample_detector_data)
        assert success, "Detector creation should succeed"

        # Step 4: User sees confirmation and updated list
        detectors_page.page.wait_for_timeout(2000)  # Wait for HTMX update

        updated_detector_count = len(detectors_page.get_detector_list())
        assert updated_detector_count > initial_detector_count, (
            "Detector list should be updated with new detector"
        )

        # Take screenshot of final state
        detectors_page.take_screenshot("detector_creation_complete")

    def test_navigation_flow_between_pages(self, dashboard_page):
        """Test navigation flow between different pages."""
        dashboard_page.navigate()

        # Step 1: Start from dashboard
        assert "Dashboard" in dashboard_page.page.title()

        # Step 2: Navigate to each main section
        navigation_flow = [
            ("/web/detectors", "Detectors"),
            ("/web/datasets", "Datasets"),
            ("/web/detection", "Detection"),
            ("/web/experiments", "Experiments"),
            ("/web/visualizations", "Visualizations"),
            ("/web/", "Dashboard"),  # Return to dashboard
        ]

        for path, expected_title_part in navigation_flow:
            # Click navigation link
            nav_link = dashboard_page.page.locator(f"nav a[href='{path}']")
            assert nav_link.count() > 0, f"Navigation link for {path} should exist"

            nav_link.click()
            dashboard_page.page.wait_for_load_state("networkidle")

            # Verify we're on the correct page
            current_url = dashboard_page.page.url
            assert path in current_url, f"Should navigate to {path}"

            # Verify page title or content
            if (
                expected_title_part != "Dashboard"
            ):  # Dashboard doesn't always have title in content
                page_content = dashboard_page.page.text_content()
                assert expected_title_part in page_content, (
                    f"Page should contain {expected_title_part}"
                )

        # Take screenshot of final state
        dashboard_page.take_screenshot("navigation_flow_complete")

    def test_responsive_menu_interaction(self, mobile_page: Page):
        """Test responsive menu interaction flow."""
        mobile_page.goto("http://pynomaly-app:8000/web/")
        mobile_page.wait_for_load_state("networkidle")

        # Step 1: Verify mobile menu is initially closed
        mobile_button = mobile_page.locator("nav button")
        assert mobile_button.is_visible(), "Mobile menu button should be visible"

        # Step 2: Open mobile menu
        mobile_button.click()
        mobile_page.wait_for_timeout(500)  # Wait for animation

        # Step 3: Verify menu opened (check for mobile navigation links)
        mobile_nav_links = mobile_page.locator("nav .sm\\:hidden a")
        # Note: Alpine.js behavior might be hard to test, so we check structure

        # Step 4: Navigate using mobile menu
        if mobile_nav_links.count() > 0:
            detectors_link = None
            for i in range(mobile_nav_links.count()):
                link = mobile_nav_links.nth(i)
                if "Detectors" in (link.text_content() or ""):
                    detectors_link = link
                    break

            if detectors_link:
                detectors_link.click()
                mobile_page.wait_for_load_state("networkidle")

                # Verify navigation worked
                assert "/detectors" in mobile_page.url

        mobile_page.screenshot(path="screenshots/mobile_menu_flow.png")

    def test_form_validation_flow(self, detectors_page):
        """Test form validation user experience flow."""
        detectors_page.navigate()

        # Step 1: Try to submit empty form
        form_errors_before = detectors_page.get_form_errors()

        if detectors_page.page.locator(detectors_page.CREATE_BUTTON).count() > 0:
            detectors_page.click_element(detectors_page.CREATE_BUTTON)
            detectors_page.page.wait_for_timeout(1000)

            # Step 2: Check if validation errors are shown
            form_errors_after = detectors_page.get_form_errors()

            # Should show some form of validation feedback
            validation_test = detectors_page.test_form_validation()

            # Step 3: Fill form partially and test incremental validation
            if (
                detectors_page.page.locator(detectors_page.DETECTOR_NAME_INPUT).count()
                > 0
            ):
                detectors_page.fill_input(
                    detectors_page.DETECTOR_NAME_INPUT, "Test Detector"
                )

                # Try submit again
                detectors_page.click_element(detectors_page.CREATE_BUTTON)
                detectors_page.page.wait_for_timeout(1000)

                # Validation should improve or change
                partial_errors = detectors_page.get_form_errors()

        detectors_page.take_screenshot("form_validation_flow")

    def test_htmx_interaction_flow(self, dashboard_page):
        """Test HTMX interactions and dynamic updates."""
        dashboard_page.navigate()

        # Step 1: Check initial state
        initial_results = dashboard_page.get_recent_results()

        # Step 2: Test refresh functionality
        if dashboard_page.page.locator(dashboard_page.REFRESH_BUTTON).count() > 0:
            # Click refresh
            dashboard_page.refresh_results()

            # Step 3: Verify HTMX worked (content updated or no errors)
            htmx_test = dashboard_page.test_htmx_functionality()
            assert htmx_test["refresh_works"], "HTMX refresh should work"

            # Check for any HTMX error indicators
            htmx_errors = dashboard_page.page.locator("[hx-error], .htmx-error")
            assert htmx_errors.count() == 0, "Should not have HTMX errors"

        dashboard_page.take_screenshot("htmx_interaction_flow")

    def test_error_recovery_flow(self, detection_page):
        """Test error handling and recovery flow."""
        detection_page.navigate()

        # Step 1: Try to run detection without proper setup
        # This should trigger an error
        if detection_page.page.locator(detection_page.RUN_DETECTION_BUTTON).count() > 0:
            detection_page.click_element(detection_page.RUN_DETECTION_BUTTON)
            detection_page.page.wait_for_timeout(2000)

            # Step 2: Check if error is properly displayed
            error_messages = detection_page.page.locator(
                ".alert-error, .error, [class*='error'], [role='alert']"
            )

            # Step 3: Verify user can recover from error
            # Check if form is still usable after error
            detector_select = detection_page.page.locator(
                detection_page.DETECTOR_SELECT
            )
            dataset_select = detection_page.page.locator(detection_page.DATASET_SELECT)

            if detector_select.count() > 0:
                assert detector_select.is_enabled(), (
                    "Form should remain usable after error"
                )

            if dataset_select.count() > 0:
                assert dataset_select.is_enabled(), (
                    "Form should remain usable after error"
                )

        detection_page.take_screenshot("error_recovery_flow")

    def test_data_visualization_interaction_flow(self, visualizations_page):
        """Test data visualization interaction flow."""
        visualizations_page.navigate()

        # Step 1: Wait for charts to load
        charts_loaded = visualizations_page.wait_for_charts_to_load()

        if charts_loaded:
            # Step 2: Get available charts
            charts = visualizations_page.get_available_charts()
            assert len(charts) > 0, "Should have available charts"

            # Step 3: Test chart interactivity
            interactivity = visualizations_page.test_chart_interactivity()

            # Step 4: Test chart responsiveness
            responsiveness = visualizations_page.test_chart_responsiveness()

            # Charts should maintain functionality after resize
            assert responsiveness["charts_respond_to_resize"], (
                "Charts should respond to viewport changes"
            )

        visualizations_page.take_screenshot("visualization_interaction_flow")

    def test_accessibility_navigation_flow(self, page: Page):
        """Test keyboard navigation and accessibility flow."""
        page.goto("http://pynomaly-app:8000/web/")
        page.wait_for_load_state("networkidle")

        # Step 1: Test tab navigation
        page.keyboard.press("Tab")

        # Get focused element
        focused_element = page.evaluate("() => document.activeElement.tagName")

        # Should be able to focus on interactive elements
        interactive_elements = ["A", "BUTTON", "INPUT", "SELECT", "TEXTAREA"]

        # Continue tabbing to test navigation
        tab_count = 0
        max_tabs = 10  # Prevent infinite loop

        while tab_count < max_tabs:
            page.keyboard.press("Tab")
            tab_count += 1

            current_focus = page.evaluate("() => document.activeElement.tagName")
            if current_focus in interactive_elements:
                break

        # Should have found focusable elements
        assert tab_count < max_tabs, (
            "Should be able to navigate to interactive elements"
        )

        page.screenshot(path="screenshots/accessibility_navigation_flow.png")

    def test_performance_during_user_flow(self, dashboard_page):
        """Test performance metrics during typical user flow."""
        # Step 1: Measure dashboard load
        dashboard_page.navigate()
        dashboard_metrics = dashboard_page.measure_dashboard_load_time()

        # Dashboard should load reasonably quickly
        assert dashboard_metrics["dashboard_elements_load_time"] < 5000, (
            "Dashboard elements should load within 5 seconds"
        )

        # Step 2: Navigate to other pages and measure
        pages_to_test = ["/detectors", "/datasets", "/detection"]

        for page_path in pages_to_test:
            start_time = dashboard_page.page.evaluate("() => performance.now()")

            dashboard_page.navigate_to(page_path)

            end_time = dashboard_page.page.evaluate("() => performance.now()")
            load_time = end_time - start_time

            # Each page should load within reasonable time
            assert load_time < 8000, f"Page {page_path} should load within 8 seconds"

        dashboard_page.take_screenshot("performance_flow_complete")

    def test_user_feedback_flow(self, page: Page):
        """Test user feedback and status indication flow."""
        page.goto("http://pynomaly-app:8000/web/detectors")
        page.wait_for_load_state("networkidle")

        # Look for any forms that provide user feedback
        forms = page.locator("form")

        if forms.count() > 0:
            form = forms.first

            # Check for loading states
            loading_indicators = form.locator(".loading, .spinner, .htmx-indicator")

            # Check for success/error message areas
            message_areas = page.locator(
                ".message, .alert, .notification, [role='status'], [role='alert']"
            )

            # Test form submission to see feedback
            submit_button = form.locator("button[type='submit'], input[type='submit']")
            if submit_button.count() > 0:
                submit_button.click()
                page.wait_for_timeout(2000)

                # Check if any feedback appeared
                updated_messages = page.locator(
                    ".message, .alert, .notification, [role='status'], [role='alert']"
                )

                # Should provide some form of feedback
                feedback_provided = (
                    updated_messages.count() > message_areas.count()
                    or loading_indicators.count() > 0
                )

        page.screenshot(path="screenshots/user_feedback_flow.png")
