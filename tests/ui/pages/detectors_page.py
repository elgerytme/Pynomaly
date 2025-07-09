"""
Enhanced detectors page object with comprehensive wait strategies and retry mechanisms.
"""

from playwright.sync_api import Page, expect
from .base_page import BasePage
from ..helpers import retry_on_failure


class DetectorsPage(BasePage):
    """Enhanced detectors page object with robust wait strategies."""

    def __init__(self, page: Page, base_url: str = "http://localhost:8000"):
        super().__init__(page, base_url)
        
        # Page elements
        self.detectors_table_selector = ".detectors-table, table"
        self.add_detector_button_selector = ".add-detector, button[data-action='add']"
        self.detector_type_selector = "select#detector-type"
        self.training_input_selector = "#training-input, input[name='trainingData']"
        self.submit_button_selector = ".submit-button, button[type='submit']"
        self.search_input_selector = ".search-input, input[type='search']"
        self.pagination_selector = ".pagination"

    @retry_on_failure(max_retries=3, delay=1.0)
    def navigate(self, path: str = "/detectors"):
        """Navigate to detectors page with enhanced waiting."""
        super().navigate(path)
        self.wait_for_detectors_page_load()

    def wait_for_detectors_page_load(self):
        """Wait for detectors page to be fully loaded."""
        # Wait for page title
        expect(self.page).to_have_title("Detectors", timeout=self.default_timeout)
        
        # Wait for detectors table or empty state
        self.page.wait_for_selector(f"{self.detectors_table_selector}, .empty-state", timeout=self.default_timeout)
        
        # Wait for HTMX to settle
        self.htmx_wait.wait_for_htmx_settle()

    @retry_on_failure(max_retries=3, delay=0.5)
    def click_add_detector(self):
        """Click add detector button with retry logic."""
        add_button = self.wait_for_element_visible(self.add_detector_button_selector)
        add_button.click()
        self.htmx_wait.wait_for_htmx_settle()

    @retry_on_failure(max_retries=3, delay=0.5)
    def select_detector_type(self, detector_type: str):
        """Select detector type from dropdown."""
        type_dropdown = self.wait_for_element_visible(self.detector_type_selector)
        type_dropdown.select_option(detector_type)
        self.htmx_wait.wait_for_htmx_settle()

    @retry_on_failure(max_retries=3, delay=0.5)
    def upload_training_file(self, file_path: str):
        """Upload training file with retry logic."""
        training_input = self.wait_for_element_visible(self.training_input_selector)
        training_input.set_input_files(file_path)
        self.htmx_wait.wait_for_htmx_settle()

    @retry_on_failure(max_retries=3, delay=0.5)
    def submit_detector_form(self):
        """Submit detector form with retry logic."""
        submit_button = self.wait_for_element_visible(self.submit_button_selector)
        submit_button.click()
        self.htmx_wait.wait_for_htmx_settle()

    def search_detectors(self, search_term: str):
        """Search detectors with HTMX waiting."""
        search_input = self.wait_for_element_visible(self.search_input_selector)
        search_input.clear()
        search_input.fill(search_term)
        
        # Wait for search results to load
        self.htmx_wait.wait_for_htmx_settle()

    def get_detectors_list(self) -> list:
        """Get list of detectors from the table."""
        detectors = []
        
        if self.page.locator(self.detectors_table_selector).is_visible():
            rows = self.page.locator(f"{self.detectors_table_selector} tbody tr")
            
            for i in range(rows.count()):
                row = rows.nth(i)
                name_cell = row.locator("td:first-child")
                type_cell = row.locator("td:nth-child(2)")
                
                detectors.append({
                    "name": name_cell.text_content() or "",
                    "type": type_cell.text_content() or "",
                })
        
        return detectors

    def delete_detector(self, detector_name: str):
        """Delete a detector by name."""
        # Find the detector row
        detector_row = self.page.locator(f"{self.detectors_table_selector} tr:has-text('{detector_name}')")
        
        if detector_row.is_visible():
            delete_button = detector_row.locator(".delete-button, button[data-action='delete']")
            delete_button.click()
            
            # Wait for confirmation modal or direct deletion
            self.htmx_wait.wait_for_htmx_settle()
            
            # If there's a confirmation modal, click confirm
            confirm_button = self.page.locator(".confirm-delete, button[data-confirm='true']")
            if confirm_button.is_visible():
                confirm_button.click()
                self.htmx_wait.wait_for_htmx_settle()

    def verify_detector_exists(self, detector_name: str) -> bool:
        """Verify that a detector exists in the list."""
        detector_row = self.page.locator(f"{self.detectors_table_selector} tr:has-text('{detector_name}')")
        return detector_row.is_visible()

    def get_detector_count(self) -> int:
        """Get total number of detectors."""
        if self.page.locator(self.detectors_table_selector).is_visible():
            rows = self.page.locator(f"{self.detectors_table_selector} tbody tr")
            return rows.count()
        return 0

    def navigate_to_next_page(self):
        """Navigate to next page of detectors."""
        next_button = self.page.locator(f"{self.pagination_selector} .next-page, .pagination-next")
        if next_button.is_visible() and next_button.is_enabled():
            next_button.click()
            self.htmx_wait.wait_for_htmx_settle()

    def navigate_to_previous_page(self):
        """Navigate to previous page of detectors."""
        prev_button = self.page.locator(f"{self.pagination_selector} .prev-page, .pagination-prev")
        if prev_button.is_visible() and prev_button.is_enabled():
            prev_button.click()
            self.htmx_wait.wait_for_htmx_settle()

