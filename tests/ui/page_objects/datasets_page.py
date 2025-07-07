"""Datasets page object."""

from .base_page import BasePage


class DatasetsPage(BasePage):
    """Datasets page object with specific functionality."""

    # Locators
    UPLOAD_FORM = "form[enctype='multipart/form-data'], form[data-upload]"
    FILE_INPUT = "input[type='file']"
    DATASET_NAME_INPUT = "input[name='name']"
    DATASET_DESCRIPTION_INPUT = "textarea[name='description']"
    UPLOAD_BUTTON = "button[type='submit'], input[type='submit']"
    DATASET_LIST = "#dataset-list, [data-testid='dataset-list']"
    DATASET_ITEMS = ".dataset-item, .bg-white.shadow"

    def navigate(self) -> None:
        """Navigate to datasets page."""
        self.navigate_to("/datasets")

    def upload_dataset(self, file_path: str, dataset_data: dict[str, str]) -> bool:
        """Upload a dataset file."""
        try:
            # Fill dataset information
            if self.page.locator(self.DATASET_NAME_INPUT).count() > 0:
                self.fill_input(self.DATASET_NAME_INPUT, dataset_data["name"])

            if self.page.locator(self.DATASET_DESCRIPTION_INPUT).count() > 0:
                self.fill_input(
                    self.DATASET_DESCRIPTION_INPUT, dataset_data.get("description", "")
                )

            # Upload file
            if self.page.locator(self.FILE_INPUT).count() > 0:
                self.page.locator(self.FILE_INPUT).set_input_files(file_path)

            # Submit form
            if self.page.locator(self.UPLOAD_BUTTON).count() > 0:
                self.click_element(self.UPLOAD_BUTTON)
                self.page.wait_for_timeout(3000)  # Wait for upload
                return True

        except Exception as e:
            print(f"Error uploading dataset: {e}")

        return False

    def get_dataset_list(self) -> list[dict[str, str]]:
        """Get list of datasets."""
        datasets = []

        dataset_items = self.page.locator(self.DATASET_ITEMS)

        for i in range(dataset_items.count()):
            item = dataset_items.nth(i)
            text_content = item.text_content() or ""

            # Extract dataset information
            lines = [line.strip() for line in text_content.split("\n") if line.strip()]

            dataset = {
                "name": "",
                "size": "",
                "format": "",
                "uploaded": "",
                "description": "",
            }

            # Parse dataset information
            for line in lines:
                if "Name:" in line:
                    dataset["name"] = line.replace("Name:", "").strip()
                elif "Size:" in line:
                    dataset["size"] = line.replace("Size:", "").strip()
                elif "Format:" in line:
                    dataset["format"] = line.replace("Format:", "").strip()
                elif "Uploaded:" in line:
                    dataset["uploaded"] = line.replace("Uploaded:", "").strip()
                elif len(line) > 10 and not any(
                    x in line for x in ["Name:", "Size:", "Format:", "Uploaded:"]
                ):
                    dataset["description"] = line

            # Fallback parsing
            if not dataset["name"] and lines:
                dataset["name"] = lines[0]

            datasets.append(dataset)

        return datasets

    def click_dataset_detail(self, dataset_name: str) -> None:
        """Click on a dataset to view details."""
        dataset_items = self.page.locator(self.DATASET_ITEMS)

        for i in range(dataset_items.count()):
            item = dataset_items.nth(i)
            if dataset_name in (item.text_content() or ""):
                link = item.locator("a").first
                if link.count() > 0:
                    link.click()
                else:
                    item.click()
                break

    def verify_datasets_page_layout(self) -> dict[str, bool]:
        """Verify datasets page layout."""
        return {
            "has_title": "Datasets" in (self.page.title() or ""),
            "has_upload_form": self.page.locator(self.UPLOAD_FORM).count() > 0,
            "has_file_input": self.page.locator(self.FILE_INPUT).count() > 0,
            "has_dataset_list": self.page.locator(self.DATASET_LIST).count() > 0,
            "has_name_input": self.page.locator(self.DATASET_NAME_INPUT).count() > 0,
        }

    def test_file_upload_validation(self) -> dict[str, bool]:
        """Test file upload validation."""
        results = {}

        # Test upload without file
        if self.page.locator(self.UPLOAD_BUTTON).count() > 0:
            self.click_element(self.UPLOAD_BUTTON)
            self.page.wait_for_timeout(500)

            # Check for validation messages
            validation_messages = self.page.locator(
                ".error, .invalid, [class*='error']"
            )
            results["shows_file_required_error"] = validation_messages.count() > 0

        return results

    def get_supported_file_formats(self) -> list[str]:
        """Get supported file formats from the UI."""
        formats = []

        # Look for format information in help text or labels
        help_text = self.page.locator("small, .help-text, .text-sm.text-gray-500")

        for i in range(help_text.count()):
            text = help_text.nth(i).text_content() or ""
            if any(
                ext in text.lower() for ext in [".csv", ".parquet", ".json", ".xlsx"]
            ):
                formats.append(text)

        return formats
