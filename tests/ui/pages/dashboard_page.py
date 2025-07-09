"""
Enhanced dashboard page object with comprehensive wait strategies and retry mechanisms.
"""

from playwright.sync_api import Page, expect
from .base_page import BasePage
from ..helpers import retry_on_failure


class DashboardPage(BasePage):
    """Enhanced dashboard page object with robust wait strategies."""

    def __init__(self, page: Page, base_url: str = "http://localhost:8000"):
        super().__init__(page, base_url)
        
        # Page elements
        self.logo_selector = "img[alt*='Pynomaly'], .logo"
        self.nav_selector = "nav, .navigation"
        self.title_selector = "h1, .page-title"
        self.metrics_selector = ".metrics, .dashboard-metrics"
        self.charts_selector = ".charts, .dashboard-charts"
        self.recent_detections_selector = ".recent-detections, .recent-results"
        self.quick_actions_selector = ".quick-actions, .actions"
        
        # Navigation links mapping
        self.nav_links = {
            "Dashboard": "/",
            "Datasets": "/datasets",
            "Detectors": "/detectors",
            "Detection": "/detection",
            "Visualizations": "/visualizations",
        }

    @retry_on_failure(max_retries=3, delay=1.0)
    def navigate(self, path: str = "/"):
        """Navigate to dashboard with enhanced waiting."""
        super().navigate(path)
        self.wait_for_dashboard_load()

    def wait_for_dashboard_load(self):
        """Wait for dashboard to be fully loaded."""
        # Wait for essential elements
        self.wait_for_element_visible(self.logo_selector)
        self.wait_for_element_visible(self.nav_selector)
        self.wait_for_element_visible(self.title_selector)
        
        # Wait for HTMX to settle
        self.htmx_wait.wait_for_htmx_settle()
        
        # Wait for any dynamic content loading
        self.htmx_wait.wait_for_htmx_indicator_hidden()

    def is_navigation_visible(self) -> bool:
        """Check if main navigation is visible."""
        return self.page.locator(self.nav_selector).is_visible()

    @retry_on_failure(max_retries=3, delay=0.5)
    def click_nav_link(self, link_text: str):
        """Click navigation link with retry logic."""
        nav_link = self.page.locator(f"nav a:has-text('{link_text}')")
        expect(nav_link).to_be_visible(timeout=self.default_timeout)
        
        nav_link.click()
        
        # Wait for navigation to complete
        self.htmx_wait.wait_for_htmx_settle()
        
        # Wait for page load
        self.page.wait_for_load_state("networkidle", timeout=self.default_timeout)

    def wait_for_url_change(self, expected_url: str):
        """Wait for URL to change to expected value."""
        expect(self.page).to_have_url(expected_url, timeout=self.default_timeout)

    def wait_for_page_load(self):
        """Wait for page to be fully loaded."""
        self.page.wait_for_load_state("networkidle", timeout=self.default_timeout)
        self.htmx_wait.wait_for_htmx_settle()

    def verify_dashboard_components(self) -> dict:
        """Verify dashboard components are present."""
        components = {
            "logo": self.page.locator(self.logo_selector).is_visible(),
            "navigation": self.page.locator(self.nav_selector).is_visible(),
            "title": self.page.locator(self.title_selector).is_visible(),
            "metrics": self.page.locator(self.metrics_selector).is_visible(),
            "charts": self.page.locator(self.charts_selector).is_visible(),
            "recent_detections": self.page.locator(self.recent_detections_selector).is_visible(),
            "quick_actions": self.page.locator(self.quick_actions_selector).is_visible(),
        }
        return components

    def verify_responsive_layout(self) -> dict:
        """Test responsive layout across different viewport sizes."""
        viewports = {
            "mobile": {"width": 375, "height": 667},
            "tablet": {"width": 768, "height": 1024},
            "desktop": {"width": 1920, "height": 1080},
        }
        
        results = {}
        
        for size_name, viewport in viewports.items():
            # Set viewport
            self.page.set_viewport_size(viewport)
            
            # Wait for layout to adjust
            self.page.wait_for_timeout(1000)
            
            # Check if layout is working
            navigation_visible = self.page.locator(self.nav_selector).is_visible()
            content_visible = self.page.locator(self.title_selector).is_visible()
            
            results[size_name] = {
                "layout_ok": navigation_visible and content_visible,
                "navigation_visible": navigation_visible,
                "content_visible": content_visible,
            }
        
        # Reset to default viewport
        self.page.set_viewport_size({"width": 1920, "height": 1080})
        
        return results

    def verify_accessibility(self) -> dict:
        """Verify basic accessibility requirements."""
        accessibility_results = {
            "has_title": bool(self.page.title()),
            "has_main_heading": self.page.locator("h1").is_visible(),
            "has_lang_attribute": bool(self.page.locator("html[lang]").count()),
            "has_viewport_meta": bool(self.page.locator('meta[name="viewport"]').count()),
            "images_have_alt": True,
            "buttons_have_labels": True,
            "inputs_have_labels": True,
        }
        
        # Check images have alt text
        images = self.page.locator("img")
        for i in range(images.count()):
            img = images.nth(i)
            if not img.get_attribute("alt"):
                accessibility_results["images_have_alt"] = False
                break
        
        # Check buttons have labels
        buttons = self.page.locator("button")
        for i in range(buttons.count()):
            btn = buttons.nth(i)
            text = btn.text_content()
            aria_label = btn.get_attribute("aria-label")
            if not (text and text.strip()) and not aria_label:
                accessibility_results["buttons_have_labels"] = False
                break
        
        # Check inputs have labels
        inputs = self.page.locator("input, select, textarea")
        for i in range(inputs.count()):
            input_elem = inputs.nth(i)
            input_id = input_elem.get_attribute("id")
            placeholder = input_elem.get_attribute("placeholder")
            
            has_label = False
            if input_id:
                label = self.page.locator(f'label[for="{input_id}"]')
                has_label = label.count() > 0
            
            if not has_label and not placeholder:
                accessibility_results["inputs_have_labels"] = False
                break
        
        return accessibility_results

    def simulate_user_interaction_flow(self) -> list:
        """Simulate realistic user interaction patterns."""
        interaction_results = []
        
        try:
            # Test navigation flow
            for link_text in ["Datasets", "Detectors", "Detection"]:
                if link_text in self.nav_links:
                    self.click_nav_link(link_text)
                    self.wait_for_page_load()
                    interaction_results.append(f"✓ Navigated to {link_text}")
                    
                    # Return to dashboard
                    self.click_nav_link("Dashboard")
                    self.wait_for_dashboard_load()
                    interaction_results.append(f"✓ Returned to Dashboard from {link_text}")
        
        except Exception as e:
            interaction_results.append(f"✗ Navigation interaction failed: {str(e)}")
        
        try:
            # Test dashboard interactions
            if self.page.locator(self.quick_actions_selector).is_visible():
                # Test quick actions if available
                quick_actions = self.page.locator(f"{self.quick_actions_selector} button")
                if quick_actions.count() > 0:
                    first_action = quick_actions.first
                    first_action.click()
                    self.htmx_wait.wait_for_htmx_settle()
                    interaction_results.append("✓ Quick action interaction successful")
        
        except Exception as e:
            interaction_results.append(f"✗ Quick action interaction failed: {str(e)}")
        
        return interaction_results

    def get_metrics_data(self) -> list:
        """Get metrics data from dashboard."""
        metrics_data = []
        
        if self.page.locator(self.metrics_selector).is_visible():
            metrics = self.page.locator(f"{self.metrics_selector} .metric")
            
            for i in range(metrics.count()):
                metric = metrics.nth(i)
                title_element = metric.locator(".metric-title, .title")
                value_element = metric.locator(".metric-value, .value")
                
                metrics_data.append({
                    "title": title_element.text_content() or "",
                    "value": value_element.text_content() or "",
                })
        
        return metrics_data

    def capture_screenshot(self, name: str, full_page: bool = True):
        """Capture dashboard screenshot."""
        return self.screenshot_helper.capture_screenshot(f"dashboard_{name}", full_page)

    def wait_for_htmx_settle(self):
        """Wait for HTMX requests to settle."""
        self.htmx_wait.wait_for_htmx_settle()

    def wait_for_ajax_complete(self):
        """Wait for AJAX requests to complete."""
        try:
            self.page.wait_for_function(
                "() => typeof jQuery !== 'undefined' ? jQuery.active === 0 : true",
                timeout=self.default_timeout
            )
        except:
            # jQuery might not be available
            pass

    def wait_for_element_visible(self, selector: str, timeout: int = None):
        """Wait for element to be visible with retry."""
        timeout = timeout or self.default_timeout
        element = self.page.locator(selector)
        expect(element).to_be_visible(timeout=timeout)
        return element
