"""Comprehensive UI automation tests for Pynomaly web application."""

import json
import time
from pathlib import Path

import pytest
from playwright.sync_api import expect


class TestWebAppAutomation:
    """Main UI automation test suite for Pynomaly web application."""

    def test_dashboard_loads(self, dashboard_page):
        """Test that the dashboard loads correctly."""
        dashboard_page.navigate()

        # Take initial screenshot
        dashboard_page.take_screenshot("dashboard_initial_load")

        # Verify page title
        expect(dashboard_page.page).to_have_title("Pynomaly")

        # Verify main navigation elements
        assert dashboard_page.is_navigation_visible()

        # Verify logo is present
        logo = dashboard_page.page.locator('a:has-text("🔍 Pynomaly")')
        expect(logo).to_be_visible()

        # Take screenshot after verification
        dashboard_page.take_screenshot("dashboard_loaded")

    def test_navigation_menu(self, dashboard_page):
        """Test navigation menu functionality."""
        dashboard_page.navigate()

        # Test navigation links
        nav_links = [
            ("Dashboard", "/web/"),
            ("Detectors", "/web/detectors"),
            ("Datasets", "/web/datasets"),
            ("Detection", "/web/detection"),
            ("Experiments", "/web/experiments"),
            ("Visualizations", "/web/visualizations"),
            ("Exports", "/web/exports"),
        ]

        for link_text, expected_url in nav_links:
            # Click navigation link
            nav_link = dashboard_page.page.locator(f'nav a:has-text("{link_text}")')
            if nav_link.count() > 0:
                nav_link.click()

                # Wait for navigation
                dashboard_page.page.wait_for_load_state("networkidle")

                # Verify URL
                current_url = dashboard_page.page.url
                assert (
                    expected_url in current_url
                ), f"Expected {expected_url} in {current_url}"

                # Take screenshot of each page
                dashboard_page.take_screenshot(f"nav_{link_text.lower()}")

    def test_responsive_design(self, browser):
        """Test responsive design across different viewport sizes."""
        viewports = [
            {"width": 1920, "height": 1080, "name": "desktop"},
            {"width": 768, "height": 1024, "name": "tablet"},
            {"width": 375, "height": 667, "name": "mobile"},
        ]

        for viewport in viewports:
            # Create context with specific viewport
            context = browser.new_context(
                viewport={"width": viewport["width"], "height": viewport["height"]}
            )
            page = context.new_page()

            try:
                # Navigate to dashboard
                page.goto("http://localhost:8000/web/")
                page.wait_for_load_state("networkidle")

                # Take screenshot for this viewport
                screenshot_path = (
                    Path("screenshots") / f"responsive_{viewport['name']}_dashboard.png"
                )
                page.screenshot(path=str(screenshot_path), full_page=True)

                # Test mobile menu if on mobile viewport
                if viewport["width"] <= 768:
                    mobile_menu_btn = page.locator("[x-data] button")
                    if mobile_menu_btn.count() > 0:
                        mobile_menu_btn.click()
                        page.screenshot(
                            path=f"screenshots/responsive_{viewport['name']}_menu_open.png"
                        )

            finally:
                page.close()
                context.close()

    def test_detectors_page_functionality(self, detectors_page):
        """Test detectors page functionality."""
        detectors_page.navigate()

        # Take initial screenshot
        detectors_page.take_screenshot("detectors_page_initial")

        # Verify page loads
        expect(detectors_page.page).to_have_url("**/detectors")

        # Check for detector list or empty state
        detector_table = detectors_page.page.locator(
            'table, .empty-state, [data-testid="detector-list"]'
        )
        expect(detector_table).to_be_visible()

        # Test create detector button if present
        create_btn = detectors_page.page.locator(
            'button:has-text("Create"), a:has-text("Create")'
        )
        if create_btn.count() > 0:
            create_btn.click()
            detectors_page.take_screenshot("detectors_create_form")

        detectors_page.take_screenshot("detectors_page_final")

    def test_datasets_page_functionality(self, datasets_page):
        """Test datasets page functionality."""
        datasets_page.navigate()

        # Take initial screenshot
        datasets_page.take_screenshot("datasets_page_initial")

        # Verify page loads
        expect(datasets_page.page).to_have_url("**/datasets")

        # Check for dataset list or empty state
        dataset_content = datasets_page.page.locator(
            'table, .empty-state, [data-testid="dataset-list"]'
        )
        expect(dataset_content).to_be_visible()

        # Test upload functionality if present
        upload_btn = datasets_page.page.locator(
            'button:has-text("Upload"), input[type="file"]'
        )
        if upload_btn.count() > 0:
            datasets_page.take_screenshot("datasets_upload_available")

        datasets_page.take_screenshot("datasets_page_final")

    def test_detection_workflow(self, detection_page):
        """Test detection workflow page."""
        detection_page.navigate()

        # Take initial screenshot
        detection_page.take_screenshot("detection_page_initial")

        # Verify page loads
        expect(detection_page.page).to_have_url("**/detection")

        # Check for detection form or workflow
        detection_form = detection_page.page.locator(
            'form, .detection-form, [data-testid="detection-form"]'
        )
        if detection_form.count() > 0:
            detection_page.take_screenshot("detection_form_available")

        # Check for recent results
        results_section = detection_page.page.locator(
            '.results, [data-testid="results"]'
        )
        if results_section.count() > 0:
            detection_page.take_screenshot("detection_results_section")

        detection_page.take_screenshot("detection_page_final")

    def test_visualizations_page(self, visualizations_page):
        """Test visualizations page functionality."""
        visualizations_page.navigate()

        # Take initial screenshot
        visualizations_page.take_screenshot("visualizations_page_initial")

        # Verify page loads
        expect(visualizations_page.page).to_have_url("**/visualizations")

        # Wait for any charts to load
        visualizations_page.page.wait_for_timeout(2000)

        # Check for chart containers
        chart_containers = visualizations_page.page.locator(
            '[id*="chart"], .chart, canvas, svg'
        )
        if chart_containers.count() > 0:
            visualizations_page.take_screenshot("visualizations_charts_loaded")

        visualizations_page.take_screenshot("visualizations_page_final")

    def test_progressive_web_app_features(self, page):
        """Test PWA features."""
        page.goto("http://localhost:8000/web/")

        # Check for service worker registration
        page.evaluate(
            """
            navigator.serviceWorker.getRegistrations().then(registrations => registrations.length > 0)
        """
        )

        # Check for manifest
        manifest_link = page.locator('link[rel="manifest"]')
        if manifest_link.count() > 0:
            manifest_href = manifest_link.get_attribute("href")
            print(f"Manifest found: {manifest_href}")

        # Check for app icons
        app_icons = page.locator('link[rel*="icon"], link[rel="apple-touch-icon"]')
        icon_count = app_icons.count()
        print(f"Found {icon_count} app icons")

        # Take screenshot
        page.screenshot(path="screenshots/pwa_features_test.png", full_page=True)

    def test_error_handling(self, page):
        """Test error handling and 404 pages."""
        # Test 404 page
        page.goto("http://localhost:8000/web/nonexistent-page")
        page.wait_for_load_state("networkidle")

        # Take screenshot of error page
        page.screenshot(path="screenshots/error_404_page.png", full_page=True)

        # Check if error page contains helpful information
        error_content = page.locator('h1, .error, [data-testid="error"]')
        if error_content.count() > 0:
            print("Error page properly displayed")

    def test_form_interactions(self, page):
        """Test form interactions and HTMX functionality."""
        page.goto("http://localhost:8000/web/detectors")
        page.wait_for_load_state("networkidle")

        # Look for any forms
        forms = page.locator("form")
        form_count = forms.count()

        if form_count > 0:
            page.screenshot(path="screenshots/forms_available.png", full_page=True)

            # Test form interactions
            for i in range(min(form_count, 3)):  # Test up to 3 forms
                form = forms.nth(i)

                # Take screenshot before interaction
                page.screenshot(path=f"screenshots/form_{i}_before.png", full_page=True)

                # Try to interact with form inputs
                inputs = form.locator("input, select, textarea")
                input_count = inputs.count()

                for j in range(min(input_count, 2)):  # Test up to 2 inputs per form
                    input_elem = inputs.nth(j)
                    input_type = input_elem.get_attribute("type") or "text"

                    if input_type in ["text", "email", "password"]:
                        input_elem.fill("test_value")
                    elif input_type == "checkbox":
                        input_elem.check()

                # Take screenshot after interaction
                page.screenshot(path=f"screenshots/form_{i}_after.png", full_page=True)

    def test_accessibility_basics(self, page):
        """Test basic accessibility features."""
        page.goto("http://localhost:8000/web/")
        page.wait_for_load_state("networkidle")

        # Check for proper heading structure
        headings = page.locator("h1, h2, h3, h4, h5, h6")
        heading_count = headings.count()
        print(f"Found {heading_count} headings")

        # Check for alt text on images
        images = page.locator("img")
        for i in range(images.count()):
            img = images.nth(i)
            alt_text = img.get_attribute("alt")
            if not alt_text:
                print(f"Image {i} missing alt text")

        # Check for form labels
        inputs = page.locator("input")
        for i in range(inputs.count()):
            input_elem = inputs.nth(i)
            input_id = input_elem.get_attribute("id")
            if input_id:
                label = page.locator(f'label[for="{input_id}"]')
                if label.count() == 0:
                    print(f"Input {input_id} missing label")

        # Take accessibility screenshot
        page.screenshot(path="screenshots/accessibility_test.png", full_page=True)

    def test_performance_metrics(self, page):
        """Test basic performance metrics."""
        start_time = time.time()

        page.goto("http://localhost:8000/web/")
        page.wait_for_load_state("networkidle")

        load_time = time.time() - start_time

        # Get performance metrics
        metrics = page.evaluate(
            """
            () => {
                const perfData = performance.getEntriesByType('navigation')[0];
                return {
                    loadTime: perfData.loadEventEnd - perfData.loadEventStart,
                    domContentLoaded: perfData.domContentLoadedEventEnd - perfData.domContentLoadedEventStart,
                    firstPaint: performance.getEntriesByType('paint').find(p => p.name === 'first-paint')?.startTime || 0,
                    firstContentfulPaint: performance.getEntriesByType('paint').find(p => p.name === 'first-contentful-paint')?.startTime || 0
                };
            }
        """
        )

        print(f"Page load time: {load_time:.2f}s")
        print(f"Performance metrics: {metrics}")

        # Take screenshot with metrics
        page.screenshot(path="screenshots/performance_test.png", full_page=True)

        # Assert reasonable load times
        assert load_time < 10, f"Page load time {load_time:.2f}s is too slow"

    def test_javascript_console_errors(self, page):
        """Test for JavaScript console errors."""
        console_messages = []

        def handle_console(msg):
            console_messages.append(
                {"type": msg.type, "text": msg.text, "location": msg.location}
            )

        page.on("console", handle_console)

        # Visit main pages and collect console messages
        pages_to_test = [
            "/web/",
            "/web/detectors",
            "/web/datasets",
            "/web/detection",
            "/web/visualizations",
        ]

        for test_page in pages_to_test:
            try:
                page.goto(f"http://localhost:8000{test_page}")
                page.wait_for_load_state("networkidle")
                page.wait_for_timeout(1000)  # Wait for any delayed JS execution
            except Exception as e:
                print(f"Error loading {test_page}: {e}")

        # Filter and report significant errors
        errors = [msg for msg in console_messages if msg["type"] == "error"]
        warnings = [msg for msg in console_messages if msg["type"] == "warning"]

        print(f"Found {len(errors)} console errors and {len(warnings)} warnings")

        for error in errors:
            print(f"Console Error: {error['text']}")

        # Save console log
        with open("screenshots/console_log.json", "w") as f:
            json.dump(console_messages, f, indent=2)

        # Take final screenshot
        page.screenshot(path="screenshots/console_test_final.png", full_page=True)


@pytest.mark.smoke
class TestSmokeTests:
    """Smoke tests for critical functionality."""

    def test_app_loads(self, page):
        """Basic smoke test - app loads without crashing."""
        page.goto("http://localhost:8000/web/")
        expect(page).to_have_title("Pynomaly")
        page.screenshot(path="screenshots/smoke_test_app_loads.png")

    def test_navigation_works(self, page):
        """Basic smoke test - navigation works."""
        page.goto("http://localhost:8000/web/")

        # Test at least one navigation link
        detectors_link = page.locator('nav a:has-text("Detectors")')
        if detectors_link.count() > 0:
            detectors_link.click()
            page.wait_for_load_state("networkidle")
            expect(page).to_have_url("**/detectors")

        page.screenshot(path="screenshots/smoke_test_navigation.png")


@pytest.mark.responsive
class TestResponsiveDesign:
    """Tests for responsive design across different screen sizes."""

    def test_mobile_layout(self, mobile_page):
        """Test mobile responsive layout."""
        mobile_page.goto("http://localhost:8000/web/")
        mobile_page.wait_for_load_state("networkidle")

        # Test mobile navigation
        mobile_menu = mobile_page.locator("[x-data] button, .mobile-menu-toggle")
        if mobile_menu.count() > 0:
            mobile_page.screenshot(path="screenshots/mobile_menu_closed.png")
            mobile_menu.click()
            mobile_page.screenshot(path="screenshots/mobile_menu_open.png")

        mobile_page.screenshot(path="screenshots/mobile_layout.png", full_page=True)

    def test_tablet_layout(self, tablet_page):
        """Test tablet responsive layout."""
        tablet_page.goto("http://localhost:8000/web/")
        tablet_page.wait_for_load_state("networkidle")
        tablet_page.screenshot(path="screenshots/tablet_layout.png", full_page=True)
