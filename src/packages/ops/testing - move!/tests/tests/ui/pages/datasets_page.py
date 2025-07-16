"""
Enhanced datasets page object with comprehensive wait strategies and retry mechanisms.
"""

from playwright.sync_api import Page, expect

from ..helpers import retry_on_failure
from .base_page import BasePage


class DatasetsPage(BasePage):
    """Enhanced datasets page object with robust wait strategies."""

    def __init__(self, page: Page, base_url: str = "http://localhost:8000"):
        super().__init__(page, base_url)

        # Page elements
        self.datasets_table_selector = ".datasets-table, table"
        self.add_dataset_button_selector = ".add-dataset, button[data-action='add']"
        self.upload_button_selector = ".upload-button, input[type='file']"
        self.dataset_name_input_selector = "#dataset-name, input[name='name']"
        self.dataset_description_input_selector = (
            "#dataset-description, textarea[name='description']"
        )
        self.submit_button_selector = ".submit-button, button[type='submit']"
        self.search_input_selector = ".search-input, input[type='search']"
        self.pagination_selector = ".pagination"

    @retry_on_failure(max_retries=3, delay=1.0)
    def navigate(self, path: str = "/datasets"):
        """Navigate to datasets page with enhanced waiting."""
        super().navigate(path)
        self.wait_for_datasets_page_load()

    def wait_for_datasets_page_load(self):
        """Wait for datasets page to be fully loaded."""
        # Wait for page title
        expect(self.page).to_have_title("Datasets", timeout=self.default_timeout)

        # Wait for datasets table or empty state
        self.page.wait_for_selector(
            f"{self.datasets_table_selector}, .empty-state",
            timeout=self.default_timeout,
        )

        # Wait for HTMX to settle
        self.htmx_wait.wait_for_htmx_settle()

    @retry_on_failure(max_retries=3, delay=0.5)
    def click_add_dataset(self):
        """Click add dataset button with retry logic."""
        add_button = self.wait_for_element_visible(self.add_dataset_button_selector)
        add_button.click()
        self.htmx_wait.wait_for_htmx_settle()

    @retry_on_failure(max_retries=3, delay=0.5)
    def upload_dataset_file(self, file_path: str):
        """Upload dataset file with retry logic."""
        upload_input = self.wait_for_element_visible(self.upload_button_selector)
        upload_input.set_input_files(file_path)
        self.htmx_wait.wait_for_htmx_settle()

    @retry_on_failure(max_retries=3, delay=0.5)
    def fill_dataset_form(self, name: str, description: str = ""):
        """Fill dataset form with retry logic."""
        if name:
            name_input = self.wait_for_element_visible(self.dataset_name_input_selector)
            name_input.clear()
            name_input.fill(name)

        if description:
            desc_input = self.wait_for_element_visible(
                self.dataset_description_input_selector
            )
            desc_input.clear()
            desc_input.fill(description)

        self.htmx_wait.wait_for_htmx_settle()

    @retry_on_failure(max_retries=3, delay=0.5)
    def submit_dataset_form(self):
        """Submit dataset form with retry logic."""
        submit_button = self.wait_for_element_visible(self.submit_button_selector)
        submit_button.click()
        self.htmx_wait.wait_for_htmx_settle()

    def search_datasets(self, search_term: str):
        """Search datasets with HTMX waiting."""
        search_input = self.wait_for_element_visible(self.search_input_selector)
        search_input.clear()
        search_input.fill(search_term)

        # Wait for search results to load
        self.htmx_wait.wait_for_htmx_settle()

    def get_datasets_list(self) -> list:
        """Get list of datasets from the table."""
        datasets = []

        if self.page.locator(self.datasets_table_selector).is_visible():
            rows = self.page.locator(f"{self.datasets_table_selector} tbody tr")

            for i in range(rows.count()):
                row = rows.nth(i)
                name_cell = row.locator("td:first-child")
                description_cell = row.locator("td:nth-child(2)")

                datasets.append(
                    {
                        "name": name_cell.text_content() or "",
                        "description": description_cell.text_content() or "",
                    }
                )

        return datasets

    def delete_dataset(self, dataset_name: str):
        """Delete a dataset by name."""
        # Find the dataset row
        dataset_row = self.page.locator(
            f"{self.datasets_table_selector} tr:has-text('{dataset_name}')"
        )

        if dataset_row.is_visible():
            delete_button = dataset_row.locator(
                ".delete-button, button[data-action='delete']"
            )
            delete_button.click()

            # Wait for confirmation modal or direct deletion
            self.htmx_wait.wait_for_htmx_settle()

            # If there's a confirmation modal, click confirm
            confirm_button = self.page.locator(
                ".confirm-delete, button[data-confirm='true']"
            )
            if confirm_button.is_visible():
                confirm_button.click()
                self.htmx_wait.wait_for_htmx_settle()

    def verify_dataset_exists(self, dataset_name: str) -> bool:
        """Verify that a dataset exists in the list."""
        dataset_row = self.page.locator(
            f"{self.datasets_table_selector} tr:has-text('{dataset_name}')"
        )
        return dataset_row.is_visible()

    def get_dataset_count(self) -> int:
        """Get total number of datasets."""
        if self.page.locator(self.datasets_table_selector).is_visible():
            rows = self.page.locator(f"{self.datasets_table_selector} tbody tr")
            return rows.count()
        return 0

    def navigate_to_next_page(self):
        """Navigate to next page of datasets."""
        next_button = self.page.locator(
            f"{self.pagination_selector} .next-page, .pagination-next"
        )
        if next_button.is_visible() and next_button.is_enabled():
            next_button.click()
            self.htmx_wait.wait_for_htmx_settle()

    def navigate_to_previous_page(self):
        """Navigate to previous page of datasets."""
        prev_button = self.page.locator(
            f"{self.pagination_selector} .prev-page, .pagination-prev"
        )
        if prev_button.is_visible() and prev_button.is_enabled():
            prev_button.click()
            self.htmx_wait.wait_for_htmx_settle()
