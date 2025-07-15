"""Enhanced dashboard page object with robust UI interaction patterns."""

from .base_page import BasePage, retry_on_failure


class DashboardPage(BasePage):
    """Enhanced dashboard page object with comprehensive interaction methods."""

    def __init__(self, page, base_url: str = "http://localhost:8000"):
        super().__init__(page, base_url)

        # Selectors
        self.logo_selector = 'a:has-text("🔍 Pynomaly")'
        self.navigation_selector = "nav.main-navigation"
        self.dashboard_title_selector = "h1"
        self.metrics_cards_selector = ".metric-card"
        self.chart_container_selector = ".chart-container"
        self.recent_detections_selector = ".recent-detections"
        self.quick_actions_selector = ".quick-actions"

        # Navigation links mapping
        self.nav_links = {
            "Dashboard": "/web/",
            "Detectors": "/web/detectors",
            "Datasets": "/web/datasets",
            "Detection": "/web/detection",
            "Experiments": "/web/experiments",
            "Visualizations": "/web/visualizations",
            "Exports": "/web/exports",
        }

    def navigate(self):
        """Navigate to dashboard with enhanced loading verification."""
        super().navigate("/web/")
        self.wait_for_dashboard_load()

    @retry_on_failure(max_retries=3, delay=1.0)
    def wait_for_dashboard_load(self):
        """Wait for dashboard to fully load with all components."""
        # Wait for basic page structure
        self.wait_for_element_visible(self.logo_selector)
        self.wait_for_element_visible(self.navigation_selector)

        # Wait for dashboard-specific content
        self.wait_for_element_visible(self.dashboard_title_selector)

        # Wait for dynamic content to load
        self.wait_for_charts_load()
        self.wait_for_metrics_load()

        # Wait for any HTMX content
        self.wait_for_htmx_settle()

    @retry_on_failure(max_retries=3, delay=0.5)
    def wait_for_charts_load(self):
        """Wait for chart components to load."""
        try:
            # Wait for chart containers to be present
            self.wait_for_element_visible(
                self.chart_container_selector, timeout=self.short_timeout
            )

            # Wait for charts to be rendered (D3.js/ECharts)
            self.page.wait_for_function(
                """() => {
                    const charts = document.querySelectorAll('.chart-container');
                    return Array.from(charts).every(chart =>
                        chart.children.length > 0 || chart.textContent.includes('No data')
                    );
                }""",
                timeout=self.default_timeout,
            )
        except:
            # Charts might not be present on all dashboard views
            pass

    @retry_on_failure(max_retries=3, delay=0.5)
    def wait_for_metrics_load(self):
        """Wait for metrics cards to load with data."""
        try:
            # Wait for metrics cards to be present
            self.wait_for_element_visible(
                self.metrics_cards_selector, timeout=self.short_timeout
            )

            # Wait for metrics to have actual values (not loading state)
            self.page.wait_for_function(
                """() => {
                    const metrics = document.querySelectorAll('.metric-card .metric-value');
                    return Array.from(metrics).every(metric =>
                        metric.textContent.trim() !== '' &&
                        !metric.textContent.includes('Loading') &&
                        !metric.textContent.includes('...')
                    );
                }""",
                timeout=self.default_timeout,
            )
        except:
            # Metrics might not be present on all dashboard views
            pass

    def is_navigation_visible(self) -> bool:
        """Check if main navigation is visible."""
        return self.is_element_visible(self.navigation_selector)

    @retry_on_failure(max_retries=3, delay=0.5)
    def click_nav_link(self, link_text: str):
        """Click navigation link with verification."""
        if link_text not in self.nav_links:
            raise ValueError(f"Unknown navigation link: {link_text}")

        # Click the navigation link
        nav_selector = f"{self.navigation_selector} a:has-text('{link_text}')"
        self.click_element(nav_selector)

        # Wait for URL change
        expected_url = f"{self.base_url}{self.nav_links[link_text]}"
        self.wait_for_url_change(expected_url)

        # Wait for new page to load
        self.wait_for_page_load()

    def get_page_title(self) -> str:
        """Get the current page title."""
        return self.page.title()

    def get_dashboard_title(self) -> str:
        """Get the dashboard main title."""
        return self.get_element_text(self.dashboard_title_selector)

    def get_metrics_data(self) -> list[dict[str, str]]:
        """Extract metrics data from dashboard cards."""
        metrics = []

        try:
            cards = self.page.locator(self.metrics_cards_selector).all()

            for card in cards:
                title_element = card.locator(".metric-title")
                value_element = card.locator(".metric-value")

                if title_element.count() > 0 and value_element.count() > 0:
                    metrics.append(
                        {
                            "title": title_element.text_content().strip(),
                            "value": value_element.text_content().strip(),
                        }
                    )
        except:
            # Metrics might not be available
            pass

        return metrics

    def get_recent_detections_count(self) -> int:
        """Get count of recent detections displayed."""
        try:
            detections = self.page.locator(
                f"{self.recent_detections_selector} .detection-item"
            )
            return detections.count()
        except:
            return 0

    @retry_on_failure(max_retries=3, delay=0.5)
    def click_quick_action(self, action_text: str):
        """Click a quick action button."""
        action_selector = (
            f"{self.quick_actions_selector} button:has-text('{action_text}')"
        )
        self.click_element(action_selector)

    def verify_dashboard_components(self) -> dict[str, bool]:
        """Verify presence of all expected dashboard components."""
        components = {
            "logo": self.is_element_visible(self.logo_selector),
            "navigation": self.is_element_visible(self.navigation_selector),
            "title": self.is_element_visible(self.dashboard_title_selector),
            "metrics": self.is_element_visible(self.metrics_cards_selector),
            "charts": self.is_element_visible(self.chart_container_selector),
            "recent_detections": self.is_element_visible(
                self.recent_detections_selector
            ),
            "quick_actions": self.is_element_visible(self.quick_actions_selector),
        }

        return components

    def take_dashboard_screenshot(self, name: str = "dashboard"):
        """Take screenshot of full dashboard."""
        return self.take_screenshot(f"{name}_full", full_page=True)

    def wait_for_data_refresh(self, timeout: int | None = None):
        """Wait for dashboard data to refresh."""
        timeout = timeout or self.default_timeout

        # Look for refresh indicators
        self.wait_for_loading_complete(".loading-indicator", timeout)

        # Wait for HTMX to settle after refresh
        self.wait_for_htmx_settle(timeout)

        # Wait for metrics and charts to reload
        self.wait_for_metrics_load()
        self.wait_for_charts_load()

    def verify_responsive_layout(self, viewport_sizes: list[tuple] = None):
        """Verify dashboard layout across different viewport sizes."""
        if viewport_sizes is None:
            viewport_sizes = [
                (1920, 1080),  # Desktop
                (1366, 768),  # Laptop
                (768, 1024),  # Tablet
                (375, 667),  # Mobile
            ]

        results = {}

        for width, height in viewport_sizes:
            self.page.set_viewport_size({"width": width, "height": height})
            self.wait_for_page_load()

            # Check if navigation is still functional
            nav_visible = self.is_element_visible(self.navigation_selector)

            # Check if main content is visible
            content_visible = self.is_element_visible(self.dashboard_title_selector)

            # Take screenshot for manual verification
            self.take_screenshot(f"dashboard_responsive_{width}x{height}")

            results[f"{width}x{height}"] = {
                "navigation_visible": nav_visible,
                "content_visible": content_visible,
                "layout_ok": nav_visible and content_visible,
            }

        return results

    def verify_accessibility(self) -> dict[str, bool]:
        """Verify basic accessibility features."""
        accessibility_checks = {}

        # Check for page title
        accessibility_checks["has_title"] = len(self.page.title()) > 0

        # Check for main heading
        h1_elements = self.page.locator("h1")
        accessibility_checks["has_main_heading"] = h1_elements.count() > 0

        # Check for alt text on images
        images = self.page.locator("img")
        images_with_alt = self.page.locator("img[alt]")
        if images.count() > 0:
            accessibility_checks["images_have_alt"] = (
                images_with_alt.count() == images.count()
            )
        else:
            accessibility_checks["images_have_alt"] = True

        # Check for proper link text (not just "click here")
        generic_links = self.page.locator(
            "a:has-text('click here'), a:has-text('here'), a:has-text('more')"
        )
        accessibility_checks["no_generic_links"] = generic_links.count() == 0

        # Check for form labels if forms exist
        inputs = self.page.locator("input:not([type='hidden'])")
        if inputs.count() > 0:
            labeled_inputs = self.page.locator(
                "input[aria-label], input[aria-labelledby], label input"
            )
            accessibility_checks["inputs_have_labels"] = (
                labeled_inputs.count() >= inputs.count() * 0.8
            )
        else:
            accessibility_checks["inputs_have_labels"] = True

        return accessibility_checks

    def simulate_user_interaction_flow(self):
        """Simulate common user interaction patterns."""
        interactions = []

        # Navigate through main sections
        for link_text in ["Detectors", "Datasets", "Detection"]:
            try:
                self.click_nav_link(link_text)
                interactions.append(f"✓ Successfully navigated to {link_text}")

                # Return to dashboard
                self.click_nav_link("Dashboard")
                interactions.append(
                    f"✓ Successfully returned to Dashboard from {link_text}"
                )

            except Exception as e:
                interactions.append(f"✗ Failed to navigate to {link_text}: {str(e)}")

        return interactions
