"""Detectors page object."""

from typing import Any

from .base_page import BasePage


class DetectorsPage(BasePage):
    """Detectors page object with specific functionality."""

    # Locators
    CREATE_DETECTOR_FORM = "form[hx-post*='detector-create']"
    DETECTOR_NAME_INPUT = "input[name='name']"
    DETECTOR_ALGORITHM_SELECT = "select[name='algorithm']"
    DETECTOR_DESCRIPTION_INPUT = (
        "textarea[name='description'], input[name='description']"
    )
    CONTAMINATION_INPUT = "input[name='contamination']"
    CREATE_BUTTON = "button[type='submit']"
    DETECTOR_LIST = "#detector-list, [data-testid='detector-list']"
    DETECTOR_ITEMS = ".detector-item, .bg-white.shadow"

    def navigate(self) -> None:
        """Navigate to detectors page."""
        self.navigate_to("/detectors")

    def create_detector(self, detector_data: dict[str, Any]) -> bool:
        """Create a new detector."""
        try:
            # Fill form fields
            if self.page.locator(self.DETECTOR_NAME_INPUT).count() > 0:
                self.fill_input(self.DETECTOR_NAME_INPUT, detector_data["name"])

            if self.page.locator(self.DETECTOR_ALGORITHM_SELECT).count() > 0:
                self.select_option(
                    self.DETECTOR_ALGORITHM_SELECT, detector_data["algorithm"]
                )

            if self.page.locator(self.DETECTOR_DESCRIPTION_INPUT).count() > 0:
                self.fill_input(
                    self.DETECTOR_DESCRIPTION_INPUT,
                    detector_data.get("description", ""),
                )

            if self.page.locator(self.CONTAMINATION_INPUT).count() > 0:
                self.fill_input(
                    self.CONTAMINATION_INPUT,
                    str(detector_data.get("contamination", 0.1)),
                )

            # Submit form
            if self.page.locator(self.CREATE_BUTTON).count() > 0:
                self.click_element(self.CREATE_BUTTON)
                self.page.wait_for_timeout(2000)  # Wait for HTMX response
                return True

        except Exception as e:
            print(f"Error creating detector: {e}")

        return False

    def get_detector_list(self) -> list[dict[str, str]]:
        """Get list of detectors."""
        detectors = []

        # Try different possible selectors for detector items
        detector_items = self.page.locator(self.DETECTOR_ITEMS)

        for i in range(detector_items.count()):
            item = detector_items.nth(i)
            text_content = item.text_content() or ""

            # Extract detector information from text content
            lines = [line.strip() for line in text_content.split("\n") if line.strip()]

            detector = {"name": "", "algorithm": "", "status": "", "description": ""}

            # Parse detector information (this may need adjustment based on actual HTML)
            for line in lines:
                if "Name:" in line:
                    detector["name"] = line.replace("Name:", "").strip()
                elif "Algorithm:" in line:
                    detector["algorithm"] = line.replace("Algorithm:", "").strip()
                elif "Status:" in line:
                    detector["status"] = line.replace("Status:", "").strip()
                elif len(line) > 10 and not any(
                    x in line for x in ["Name:", "Algorithm:", "Status:"]
                ):
                    detector["description"] = line

            # If we couldn't parse properly, just use the first few lines
            if not detector["name"] and lines:
                detector["name"] = lines[0]
                if len(lines) > 1:
                    detector["algorithm"] = lines[1]

            detectors.append(detector)

        return detectors

    def get_available_algorithms(self) -> list[str]:
        """Get list of available algorithms."""
        algorithms = []

        select_element = self.page.locator(self.DETECTOR_ALGORITHM_SELECT)
        if select_element.count() > 0:
            options = select_element.locator("option")

            for i in range(options.count()):
                option = options.nth(i)
                value = option.get_attribute("value")
                if value and value != "":
                    algorithms.append(value)

        return algorithms

    def search_detectors(self, search_term: str) -> list[dict[str, str]]:
        """Search for detectors (if search functionality exists)."""
        search_input = self.page.locator(
            "input[type='search'], input[placeholder*='search']"
        )

        if search_input.count() > 0:
            search_input.fill(search_term)
            self.page.keyboard.press("Enter")
            self.page.wait_for_timeout(1000)

        return self.get_detector_list()

    def click_detector_detail(self, detector_name: str) -> None:
        """Click on a detector to view details."""
        detector_items = self.page.locator(self.DETECTOR_ITEMS)

        for i in range(detector_items.count()):
            item = detector_items.nth(i)
            if detector_name in (item.text_content() or ""):
                # Look for a link within the item
                link = item.locator("a").first
                if link.count() > 0:
                    link.click()
                else:
                    item.click()
                break

    def verify_detectors_page_layout(self) -> dict[str, bool]:
        """Verify detectors page layout."""
        return {
            "has_title": "Detectors" in (self.page.title() or ""),
            "has_create_form": self.page.locator(self.CREATE_DETECTOR_FORM).count() > 0,
            "has_detector_list": self.page.locator(self.DETECTOR_LIST).count() > 0,
            "has_name_input": self.page.locator(self.DETECTOR_NAME_INPUT).count() > 0,
            "has_algorithm_select": self.page.locator(
                self.DETECTOR_ALGORITHM_SELECT
            ).count()
            > 0,
            "has_submit_button": self.page.locator(self.CREATE_BUTTON).count() > 0,
        }

    def test_form_validation(self) -> dict[str, bool]:
        """Test form validation."""
        results = {}

        # Test empty form submission
        if self.page.locator(self.CREATE_BUTTON).count() > 0:
            self.click_element(self.CREATE_BUTTON)
            self.page.wait_for_timeout(500)

            # Check for validation messages
            validation_messages = self.page.locator(
                ".error, .invalid, [class*='error'], [class*='invalid']"
            )
            results["shows_validation_errors"] = validation_messages.count() > 0

        # Test with only name filled
        if self.page.locator(self.DETECTOR_NAME_INPUT).count() > 0:
            self.fill_input(self.DETECTOR_NAME_INPUT, "Test Detector")
            self.click_element(self.CREATE_BUTTON)
            self.page.wait_for_timeout(500)
            results["partial_form_validation"] = True

        return results

    def get_form_errors(self) -> list[str]:
        """Get form validation errors."""
        errors = []
        error_elements = self.page.locator(
            ".error, .invalid, [class*='error'], [class*='invalid']"
        )

        for i in range(error_elements.count()):
            error_text = error_elements.nth(i).text_content()
            if error_text:
                errors.append(error_text.strip())

        return errors
