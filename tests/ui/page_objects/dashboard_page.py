"""Dashboard page object."""

from .base_page import BasePage


class DashboardPage(BasePage):
    """Dashboard page object with specific functionality."""

    # Locators
    STATS_CARDS = ".grid .bg-white.shadow.rounded-lg"
    DETECTOR_COUNT = "[data-testid='detector-count'], .text-lg.font-semibold"
    DATASET_COUNT = "[data-testid='dataset-count'], .text-lg.font-semibold"
    RESULT_COUNT = "[data-testid='result-count'], .text-lg.font-semibold"
    RECENT_RESULTS_TABLE = "#results-table"
    REFRESH_BUTTON = "button[hx-get*='results-table']"
    QUICK_ACTIONS = ".grid .relative.rounded-lg"

    def navigate(self) -> None:
        """Navigate to dashboard."""
        self.navigate_to("/")

    def get_stats_summary(self) -> dict[str, int]:
        """Get dashboard statistics."""
        self.wait_for_load()

        # Try to get counts from stats cards
        stats = {}
        cards = self.page.locator(self.STATS_CARDS)

        for i in range(cards.count()):
            card = cards.nth(i)
            text = card.text_content() or ""

            if "Detectors" in text:
                # Extract number from the card
                count_element = card.locator(".text-lg.font-semibold")
                if count_element.count() > 0:
                    stats["detectors"] = int(count_element.text_content() or "0")
            elif "Datasets" in text:
                count_element = card.locator(".text-lg.font-semibold")
                if count_element.count() > 0:
                    stats["datasets"] = int(count_element.text_content() or "0")
            elif "Detection Results" in text:
                count_element = card.locator(".text-lg.font-semibold")
                if count_element.count() > 0:
                    stats["results"] = int(count_element.text_content() or "0")

        return stats

    def get_recent_results(self) -> list[dict[str, str]]:
        """Get recent detection results from table."""
        results = []
        table = self.page.locator(self.RECENT_RESULTS_TABLE + " table")

        if table.count() > 0:
            rows = table.locator("tbody tr")

            for i in range(rows.count()):
                row = rows.nth(i)
                cells = row.locator("td")

                if cells.count() >= 3:
                    results.append(
                        {
                            "detector": cells.nth(0).text_content() or "",
                            "dataset": cells.nth(1).text_content() or "",
                            "timestamp": cells.nth(2).text_content() or "",
                            "anomalies": (
                                cells.nth(3).text_content() or ""
                                if cells.count() > 3
                                else ""
                            ),
                        }
                    )

        return results

    def refresh_results(self) -> None:
        """Click refresh button for results."""
        refresh_btn = self.page.locator(self.REFRESH_BUTTON)
        if refresh_btn.count() > 0:
            refresh_btn.click()
            self.page.wait_for_timeout(1000)  # Wait for HTMX update

    def get_quick_actions(self) -> list[dict[str, str]]:
        """Get quick action buttons."""
        actions = []
        action_cards = self.page.locator(self.QUICK_ACTIONS)

        for i in range(action_cards.count()):
            card = action_cards.nth(i)
            link = card.locator("a").first if card.locator("a").count() > 0 else card

            title_elem = card.locator(".text-sm.font-medium")
            desc_elem = card.locator(".text-sm.text-gray-500")

            actions.append(
                {
                    "title": (
                        title_elem.text_content() or ""
                        if title_elem.count() > 0
                        else ""
                    ),
                    "description": (
                        desc_elem.text_content() or "" if desc_elem.count() > 0 else ""
                    ),
                    "href": link.get_attribute("href") or "",
                }
            )

        return actions

    def click_quick_action(self, action_title: str) -> None:
        """Click a quick action by title."""
        action_cards = self.page.locator(self.QUICK_ACTIONS)

        for i in range(action_cards.count()):
            card = action_cards.nth(i)
            title_elem = card.locator(".text-sm.font-medium")

            if title_elem.count() > 0 and action_title in (
                title_elem.text_content() or ""
            ):
                card.click()
                break

    def verify_dashboard_layout(self) -> dict[str, bool]:
        """Verify dashboard layout elements."""
        return {
            "has_title": "Dashboard" in self.page.title(),
            "has_stats_cards": self.page.locator(self.STATS_CARDS).count() >= 3,
            "has_recent_results": self.page.locator(self.RECENT_RESULTS_TABLE).count()
            > 0,
            "has_quick_actions": self.page.locator(self.QUICK_ACTIONS).count() >= 4,
            "has_refresh_button": self.page.locator(self.REFRESH_BUTTON).count() > 0,
        }

    def test_htmx_functionality(self) -> dict[str, bool]:
        """Test HTMX functionality on dashboard."""
        results = {}

        # Test refresh button
        if self.page.locator(self.REFRESH_BUTTON).count() > 0:
            self.page.locator(self.RECENT_RESULTS_TABLE).text_content()
            self.refresh_results()
            self.page.locator(self.RECENT_RESULTS_TABLE).text_content()
            results["refresh_works"] = True  # If no error occurred
        else:
            results["refresh_works"] = False

        return results

    def measure_dashboard_load_time(self) -> dict[str, float]:
        """Measure dashboard specific load times."""
        start_time = self.page.evaluate("() => performance.now()")

        # Wait for all main elements to load
        self.wait_for_element(self.STATS_CARDS)
        self.wait_for_element(self.RECENT_RESULTS_TABLE)
        self.wait_for_element(self.QUICK_ACTIONS)

        end_time = self.page.evaluate("() => performance.now()")

        return {
            "dashboard_elements_load_time": end_time - start_time,
            "stats_cards_count": self.page.locator(self.STATS_CARDS).count(),
            "quick_actions_count": self.page.locator(self.QUICK_ACTIONS).count(),
        }
