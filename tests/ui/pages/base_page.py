"""
Enhanced base page object with robust wait strategies, retry mechanisms, and screenshot/video capture.
"""

from playwright.sync_api import Page, expect
from ..helpers import retry_on_failure, HTMXWaitHelper, ScreenshotHelper, VideoHelper


class BasePage:
    """Enhanced base page object with robust wait strategies."""

    def __init__(self, page: Page, base_url: str = "http://localhost:8000"):
        self.page = page
        self.base_url = base_url
        self.default_timeout = 10000  # 10 seconds
        self.screenshot_helper = ScreenshotHelper(page)
        self.video_helper = VideoHelper(page.context)
        self.htmx_wait = HTMXWaitHelper(page, self.default_timeout)

    @retry_on_failure(max_retries=3, delay=1.0)
    def navigate(self, path: str = "/") -> None:
        """Navigate to a specific path with retry logic."""
        url = f"{self.base_url}{path}"
        self.page.goto(url, wait_until="networkidle", timeout=self.default_timeout)
        self.htmx_wait.wait_for_htmx_settle()

    def capture_screenshot_on_failure(self, test_name: str) -> None:
        """Capture screenshot and video on test failure."""
        self.screenshot_helper.capture_failure_screenshot(test_name)
        self.video_helper.capture_video_on_failure(self.page, test_name)

    @retry_on_failure(max_retries=3, delay=0.5)
    def wait_for_element_visible(self, selector: str, timeout: int | None = None):
        """Wait for element to be visible with retry logic."""
        timeout = timeout or self.default_timeout
        element = self.page.locator(selector)
        expect(element).to_be_visible(timeout=timeout)
        return element

    @retry_on_failure(max_retries=3, delay=0.5)
    def click_element(self, selector: str, timeout: int | None = None):
        """Click element with wait and retry logic."""
        timeout = timeout or self.default_timeout
        element = self.wait_for_element_visible(selector, timeout)
        element.click(timeout=timeout)
        self.htmx_wait.wait_for_htmx_settle()

    def capture_screenshot(self, name: str, full_page: bool = True):
        """Capture screenshot with consistent naming."""
        return self.screenshot_helper.capture_screenshot(name, full_page)

