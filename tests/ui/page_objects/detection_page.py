"""Detection page object."""

from typing import Any

from .base_page import BasePage


class DetectionPage(BasePage):
    """Detection page object with specific functionality."""

    # Locators
    DETECTION_FORM = "form[hx-post*='detect-anomalies']"
    DETECTOR_SELECT = "select[name='detector_id']"
    DATASET_SELECT = "select[name='dataset_id']"
    RUN_DETECTION_BUTTON = "button[type='submit']"
    TRAIN_FORM = "form[hx-post*='train-detector']"
    TRAIN_DETECTOR_SELECT = "select[name='detector_id']"
    TRAIN_DATASET_SELECT = "select[name='dataset_id']"
    TRAIN_BUTTON = "button[hx-post*='train-detector']"
    RESULTS_CONTAINER = "#detection-results, [data-testid='results']"

    def navigate(self) -> None:
        """Navigate to detection page."""
        self.navigate_to("/detection")

    def run_detection(self, detector_name: str, dataset_name: str) -> dict[str, Any]:
        """Run anomaly detection."""
        results = {"success": False, "error": None}

        try:
            # Select detector
            if self.page.locator(self.DETECTOR_SELECT).count() > 0:
                detector_options = self.page.locator(f"{self.DETECTOR_SELECT} option")
                for i in range(detector_options.count()):
                    option = detector_options.nth(i)
                    if detector_name in (option.text_content() or ""):
                        option_value = option.get_attribute("value")
                        if option_value:
                            self.select_option(self.DETECTOR_SELECT, option_value)
                        break

            # Select dataset
            if self.page.locator(self.DATASET_SELECT).count() > 0:
                dataset_options = self.page.locator(f"{self.DATASET_SELECT} option")
                for i in range(dataset_options.count()):
                    option = dataset_options.nth(i)
                    if dataset_name in (option.text_content() or ""):
                        option_value = option.get_attribute("value")
                        if option_value:
                            self.select_option(self.DATASET_SELECT, option_value)
                        break

            # Run detection
            if self.page.locator(self.RUN_DETECTION_BUTTON).count() > 0:
                self.click_element(self.RUN_DETECTION_BUTTON)

                # Wait for results (HTMX response)
                self.page.wait_for_timeout(5000)

                # Check for results or error messages
                results_container = self.page.locator(self.RESULTS_CONTAINER)
                if results_container.count() > 0:
                    results_text = results_container.text_content() or ""
                    if (
                        "error" in results_text.lower()
                        or "failed" in results_text.lower()
                    ):
                        results["error"] = results_text
                    else:
                        results["success"] = True
                        results["results_text"] = results_text

        except Exception as e:
            results["error"] = str(e)

        return results

    def train_detector(self, detector_name: str, dataset_name: str) -> dict[str, Any]:
        """Train a detector."""
        results = {"success": False, "error": None}

        try:
            # Select detector for training
            if self.page.locator(self.TRAIN_DETECTOR_SELECT).count() > 0:
                detector_options = self.page.locator(
                    f"{self.TRAIN_DETECTOR_SELECT} option"
                )
                for i in range(detector_options.count()):
                    option = detector_options.nth(i)
                    if detector_name in (option.text_content() or ""):
                        option_value = option.get_attribute("value")
                        if option_value:
                            self.select_option(self.TRAIN_DETECTOR_SELECT, option_value)
                        break

            # Select training dataset
            if self.page.locator(self.TRAIN_DATASET_SELECT).count() > 0:
                dataset_options = self.page.locator(
                    f"{self.TRAIN_DATASET_SELECT} option"
                )
                for i in range(dataset_options.count()):
                    option = dataset_options.nth(i)
                    if dataset_name in (option.text_content() or ""):
                        option_value = option.get_attribute("value")
                        if option_value:
                            self.select_option(self.TRAIN_DATASET_SELECT, option_value)
                        break

            # Start training
            if self.page.locator(self.TRAIN_BUTTON).count() > 0:
                self.click_element(self.TRAIN_BUTTON)

                # Wait for training completion
                self.page.wait_for_timeout(10000)  # Training might take longer

                # Check for success/error messages
                alerts = self.page.locator(".alert, [class*='alert']")
                if alerts.count() > 0:
                    alert_text = alerts.first.text_content() or ""
                    if (
                        "success" in alert_text.lower()
                        or "completed" in alert_text.lower()
                    ):
                        results["success"] = True
                        results["message"] = alert_text
                    else:
                        results["error"] = alert_text

        except Exception as e:
            results["error"] = str(e)

        return results

    def get_available_detectors(self) -> list[dict[str, str]]:
        """Get available detectors for detection."""
        detectors = []

        detector_options = self.page.locator(f"{self.DETECTOR_SELECT} option")
        for i in range(detector_options.count()):
            option = detector_options.nth(i)
            value = option.get_attribute("value")
            text = option.text_content()

            if value and text and value != "":
                detectors.append(
                    {
                        "value": value,
                        "text": text,
                        "is_trained": "trained" in text.lower()
                        or "fitted" in text.lower(),
                    }
                )

        return detectors

    def get_available_datasets(self) -> list[dict[str, str]]:
        """Get available datasets for detection."""
        datasets = []

        dataset_options = self.page.locator(f"{self.DATASET_SELECT} option")
        for i in range(dataset_options.count()):
            option = dataset_options.nth(i)
            value = option.get_attribute("value")
            text = option.text_content()

            if value and text and value != "":
                datasets.append({"value": value, "text": text})

        return datasets

    def get_detection_results(self) -> dict[str, Any]:
        """Get the latest detection results."""
        results = {}

        results_container = self.page.locator(self.RESULTS_CONTAINER)
        if results_container.count() > 0:
            results_text = results_container.text_content() or ""

            # Parse common result fields
            lines = [line.strip() for line in results_text.split("\n") if line.strip()]

            for line in lines:
                if "anomalies detected" in line.lower():
                    # Extract number of anomalies
                    import re

                    match = re.search(r"(\d+)", line)
                    if match:
                        results["anomalies_count"] = int(match.group(1))
                elif "samples processed" in line.lower():
                    import re

                    match = re.search(r"(\d+)", line)
                    if match:
                        results["samples_processed"] = int(match.group(1))
                elif "anomaly rate" in line.lower():
                    import re

                    match = re.search(r"([\d.]+)%", line)
                    if match:
                        results["anomaly_rate"] = float(match.group(1))

            results["raw_text"] = results_text

        return results

    def verify_detection_page_layout(self) -> dict[str, bool]:
        """Verify detection page layout."""
        return {
            "has_title": "Detection" in (self.page.title() or ""),
            "has_detection_form": self.page.locator(self.DETECTION_FORM).count() > 0,
            "has_detector_select": self.page.locator(self.DETECTOR_SELECT).count() > 0,
            "has_dataset_select": self.page.locator(self.DATASET_SELECT).count() > 0,
            "has_run_button": self.page.locator(self.RUN_DETECTION_BUTTON).count() > 0,
            "has_train_form": self.page.locator(self.TRAIN_FORM).count() > 0,
        }
