"""
Enhanced visualizations page object with comprehensive wait strategies and retry mechanisms.
"""

from playwright.sync_api import Page, expect
from .base_page import BasePage
from ..helpers import retry_on_failure


class VisualizationsPage(BasePage):
    """Enhanced visualizations page object with robust wait strategies."""

    def __init__(self, page: Page, base_url: str = "http://localhost:8000"):
        super().__init__(page, base_url)
        
        # Page elements
        self.visualizations_container_selector = ".visualizations-container, .viz-container"
        self.chart_selector = ".chart, .visualization"
        self.chart_type_selector = "select#chart-type"
        self.generate_button_selector = ".generate-chart, button[data-action='generate']"
        self.export_button_selector = ".export-chart, button[data-action='export']"

    @retry_on_failure(max_retries=3, delay=1.0)
    def navigate(self, path: str = "/visualizations"):
        """Navigate to visualizations page with enhanced waiting."""
        super().navigate(path)
        self.wait_for_visualizations_page_load()

    def wait_for_visualizations_page_load(self):
        """Wait for visualizations page to be fully loaded."""
        # Wait for page title
        expect(self.page).to_have_title("Visualizations", timeout=self.default_timeout)
        
        # Wait for visualizations container or empty state
        self.page.wait_for_selector(f"{self.visualizations_container_selector}, .empty-state", timeout=self.default_timeout)
        
        # Wait for HTMX to settle
        self.htmx_wait.wait_for_htmx_settle()

    @retry_on_failure(max_retries=3, delay=0.5)
    def select_chart_type(self, chart_type: str):
        """Select chart type from dropdown."""
        type_dropdown = self.wait_for_element_visible(self.chart_type_selector)
        type_dropdown.select_option(chart_type)
        self.htmx_wait.wait_for_htmx_settle()

    @retry_on_failure(max_retries=3, delay=0.5)
    def click_generate_chart(self):
        """Click generate chart button with retry logic."""
        generate_button = self.wait_for_element_visible(self.generate_button_selector)
        generate_button.click()
        self.htmx_wait.wait_for_htmx_settle()

    @retry_on_failure(max_retries=3, delay=0.5)
    def click_export_chart(self):
        """Click export chart button with retry logic."""
        export_button = self.wait_for_element_visible(self.export_button_selector)
        export_button.click()
        self.htmx_wait.wait_for_htmx_settle()

    def get_charts_list(self) -> list:
        """Get list of charts displayed."""
        charts = []
        
        if self.page.locator(self.chart_selector).count() > 0:
            chart_elements = self.page.locator(self.chart_selector)
            
            for i in range(chart_elements.count()):
                chart = chart_elements.nth(i)
                title_element = chart.locator(".chart-title, .title")
                
                charts.append({
                    "title": title_element.text_content() or f"Chart {i+1}",
                    "visible": chart.is_visible(),
                })
        
        return charts

    def verify_chart_exists(self, chart_title: str) -> bool:
        """Verify that a chart with the given title exists."""
        chart_element = self.page.locator(f"{self.chart_selector}:has-text('{chart_title}')")
        return chart_element.is_visible()

    def get_chart_count(self) -> int:
        """Get total number of charts displayed."""
        return self.page.locator(self.chart_selector).count()

    def wait_for_chart_generation(self, timeout: int = 30000):
        """Wait for chart generation to complete."""
        # Wait for loading indicator to disappear
        self.htmx_wait.wait_for_htmx_indicator_hidden(timeout=timeout)
        
        # Wait for chart to be visible
        self.page.wait_for_selector(self.chart_selector, timeout=timeout)
        
        # Wait for HTMX to settle
        self.htmx_wait.wait_for_htmx_settle()

    def verify_chart_loaded(self) -> bool:
        """Verify that at least one chart is loaded."""
        return self.page.locator(self.chart_selector).count() > 0
