"""
Enhanced detection page object with comprehensive wait strategies and retry mechanisms.
"""

from playwright.sync_api import Page, expect
from .base_page import BasePage
from ..helpers import retry_on_failure


class DetectionPage(BasePage):
    """Enhanced detection page object with robust wait strategies."""

    def __init__(self, page: Page, base_url: str = "http://localhost:8000"):
        super().__init__(page, base_url)
        
        # Page elements
        self.detection_table_selector = ".detection-table, table"
        self.start_detection_button_selector = ".start-detection, button[data-action='start']"
        self.load_file_button_selector = ".load-file, button[data-action='load']"
        self.results_table_selector = ".results-table, table"

    @retry_on_failure(max_retries=3, delay=1.0)
    def navigate(self, path: str = "/detection"):
        """Navigate to detection page with enhanced waiting."""
        super().navigate(path)
        self.wait_for_detection_page_load()

    def wait_for_detection_page_load(self):
        """Wait for detection page to be fully loaded."""
        # Wait for page title
        expect(self.page).to_have_title("Detection", timeout=self.default_timeout)
        
        # Wait for detection table or empty state
        self.page.wait_for_selector(f"{self.detection_table_selector}, .empty-state", timeout=self.default_timeout)
        
        # Wait for HTMX to settle
        self.htmx_wait.wait_for_htmx_settle()

    @retry_on_failure(max_retries=3, delay=0.5)
    def click_start_detection(self):
        """Click start detection button with retry logic."""
        start_button = self.wait_for_element_visible(self.start_detection_button_selector)
        start_button.click()
        self.htmx_wait.wait_for_htmx_settle()

    @retry_on_failure(max_retries=3, delay=0.5)
    def load_detection_file(self, file_path: str):
        """Load detection file with retry logic."""
        load_button = self.wait_for_element_visible(self.load_file_button_selector)
        load_button.set_input_files(file_path)
        self.htmx_wait.wait_for_htmx_settle()

    def get_detection_results(self) -> list:
        """Get list of detection results."""
        results = []
        
        if self.page.locator(self.results_table_selector).is_visible():
            rows = self.page.locator(f"{self.results_table_selector} tbody tr")
            
            for i in range(rows.count()):
                row = rows.nth(i)
                date_cell = row.locator("td:first-child")
                status_cell = row.locator("td:nth-child(2)")
                
                results.append({
                    "date": date_cell.text_content() or "",
                    "status": status_cell.text_content() or "",
                })
        
        return results

    def verify_detection_exists(self, date: str) -> bool:
        """Verify that a detection result exists for a given date."""
        result_row = self.page.locator(f"{self.results_table_selector} tr:has-text('{date}')")
        return result_row.is_visible()

    def verify_detection_completed(self, date: str) -> bool:
        """Verify that a detection result is completed for a given date."""
        result_row = self.page.locator(f"{self.results_table_selector} tr:has-text('{date}')")
        
        if result_row.is_visible():
            status = result_row.locator("td:nth-child(2)").text_content()
            return "Completed" in status
        return False

    def verify_detection_failed(self, date: str) -> bool:
        """Verify that a detection result failed for a given date."""
        result_row = self.page.locator(f"{self.results_table_selector} tr:has-text('{date}')")
        
        if result_row.is_visible():
            status = result_row.locator("td:nth-child(2)").text_content()
            return "Failed" in status
        return False

