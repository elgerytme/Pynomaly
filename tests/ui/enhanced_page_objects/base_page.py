"""Enhanced base page object with robust wait strategies and retry mechanisms."""

import functools
import time
from collections.abc import Callable
from typing import Any

from playwright.sync_api import Locator, Page
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import expect


def retry_on_failure(
    max_retries: int = 3, delay: float = 1.0, exponential_backoff: bool = True
):
    """Decorator for retry logic on UI interactions."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except (PlaywrightTimeoutError, AssertionError) as e:
                    last_exception = e
                    if attempt == max_retries - 1:
                        raise e

                    # Calculate delay with optional exponential backoff
                    current_delay = (
                        delay * (2**attempt) if exponential_backoff else delay
                    )
                    time.sleep(current_delay)

            raise last_exception

        return wrapper

    return decorator


class BasePage:
    """Enhanced base page object with robust wait strategies."""

    def __init__(self, page: Page, base_url: str = "http://localhost:8000"):
        self.page = page
        self.base_url = base_url
        self.default_timeout = 10000  # 10 seconds
        self.short_timeout = 5000  # 5 seconds
        self.long_timeout = 30000  # 30 seconds

    def navigate(self, path: str = ""):
        """Navigate to a specific path with retry logic."""
        url = f"{self.base_url}{path}"
        self.page.goto(url, wait_until="networkidle", timeout=self.long_timeout)
        self.wait_for_page_load()

    @retry_on_failure(max_retries=3, delay=1.0)
    def wait_for_page_load(self):
        """Wait for page to be fully loaded."""
        # Wait for DOM to be ready
        self.page.wait_for_load_state("domcontentloaded", timeout=self.default_timeout)

        # Wait for network activity to settle
        self.page.wait_for_load_state("networkidle", timeout=self.default_timeout)

        # Wait for any HTMX requests to complete
        self.wait_for_htmx_settle()

    def wait_for_htmx_settle(self, timeout: int | None = None):
        """Wait for HTMX requests to complete."""
        timeout = timeout or self.default_timeout

        try:
            # Check if HTMX is available and wait for requests to settle
            self.page.wait_for_function(
                """() => {
                    if (typeof htmx === 'undefined') return true;
                    return !document.body.hasAttribute('hx-request');
                }""",
                timeout=timeout,
            )
        except PlaywrightTimeoutError:
            # If HTMX is not available or timeout, continue
            pass

    @retry_on_failure(max_retries=3, delay=0.5)
    def wait_for_element_visible(
        self, selector: str, timeout: int | None = None
    ) -> Locator:
        """Wait for element to be visible with retry logic."""
        timeout = timeout or self.default_timeout
        locator = self.page.locator(selector)
        expect(locator).to_be_visible(timeout=timeout)
        return locator

    @retry_on_failure(max_retries=3, delay=0.5)
    def wait_for_element_clickable(
        self, selector: str, timeout: int | None = None
    ) -> Locator:
        """Wait for element to be clickable with retry logic."""
        timeout = timeout or self.default_timeout
        locator = self.page.locator(selector)
        expect(locator).to_be_visible(timeout=timeout)
        expect(locator).to_be_enabled(timeout=timeout)
        return locator

    @retry_on_failure(max_retries=3, delay=0.5)
    def click_element(self, selector: str, timeout: int | None = None):
        """Click element with wait and retry logic."""
        timeout = timeout or self.default_timeout
        element = self.wait_for_element_clickable(selector, timeout)
        element.click(timeout=timeout)

        # Wait for any resulting HTMX requests
        self.wait_for_htmx_settle()

    @retry_on_failure(max_retries=3, delay=0.5)
    def fill_input(self, selector: str, value: str, timeout: int | None = None):
        """Fill input field with wait and retry logic."""
        timeout = timeout or self.default_timeout
        element = self.wait_for_element_visible(selector, timeout)
        element.clear(timeout=timeout)
        element.fill(value, timeout=timeout)

        # Trigger change event for HTMX
        element.blur()
        self.wait_for_htmx_settle()

    @retry_on_failure(max_retries=3, delay=0.5)
    def select_option(self, selector: str, value: str, timeout: int | None = None):
        """Select option from dropdown with wait and retry logic."""
        timeout = timeout or self.default_timeout
        element = self.wait_for_element_visible(selector, timeout)
        element.select_option(value, timeout=timeout)
        self.wait_for_htmx_settle()

    def wait_for_text_content(
        self, selector: str, expected_text: str, timeout: int | None = None
    ):
        """Wait for element to contain expected text."""
        timeout = timeout or self.default_timeout
        locator = self.page.locator(selector)
        expect(locator).to_contain_text(expected_text, timeout=timeout)

    def wait_for_element_count(
        self, selector: str, count: int, timeout: int | None = None
    ):
        """Wait for specific number of elements matching selector."""
        timeout = timeout or self.default_timeout
        locator = self.page.locator(selector)
        expect(locator).to_have_count(count, timeout=timeout)

    def wait_for_url_change(self, expected_url: str, timeout: int | None = None):
        """Wait for URL to change to expected value."""
        timeout = timeout or self.default_timeout
        expect(self.page).to_have_url(expected_url, timeout=timeout)

    def scroll_to_element(self, selector: str):
        """Scroll element into view."""
        element = self.page.locator(selector)
        element.scroll_into_view_if_needed()

    def take_screenshot(self, name: str, full_page: bool = True):
        """Take screenshot with consistent naming."""
        screenshot_path = f"tests/ui/screenshots/{name}.png"
        self.page.screenshot(path=screenshot_path, full_page=full_page)
        return screenshot_path

    def wait_for_loading_complete(
        self, loading_selector: str = ".loading", timeout: int | None = None
    ):
        """Wait for loading indicators to disappear."""
        timeout = timeout or self.default_timeout
        try:
            # Wait for loading element to appear first (optional)
            loading_element = self.page.locator(loading_selector)
            if loading_element.count() > 0:
                # Then wait for it to disappear
                expect(loading_element).to_be_hidden(timeout=timeout)
        except PlaywrightTimeoutError:
            # Loading element might not appear, which is fine
            pass

    def wait_for_ajax_complete(self, timeout: int | None = None):
        """Wait for jQuery/AJAX requests to complete if jQuery is available."""
        timeout = timeout or self.default_timeout
        try:
            self.page.wait_for_function(
                "() => typeof jQuery !== 'undefined' ? jQuery.active === 0 : true",
                timeout=timeout,
            )
        except PlaywrightTimeoutError:
            # jQuery might not be available, which is fine
            pass

    def get_element_text(self, selector: str, timeout: int | None = None) -> str:
        """Get text content of element with wait."""
        timeout = timeout or self.default_timeout
        element = self.wait_for_element_visible(selector, timeout)
        return element.text_content()

    def get_element_attribute(
        self, selector: str, attribute: str, timeout: int | None = None
    ) -> str:
        """Get attribute value of element with wait."""
        timeout = timeout or self.default_timeout
        element = self.wait_for_element_visible(selector, timeout)
        return element.get_attribute(attribute)

    def is_element_visible(self, selector: str) -> bool:
        """Check if element is visible without waiting."""
        try:
            element = self.page.locator(selector)
            return element.is_visible()
        except:
            return False

    def is_element_enabled(self, selector: str) -> bool:
        """Check if element is enabled without waiting."""
        try:
            element = self.page.locator(selector)
            return element.is_enabled()
        except:
            return False

    def wait_for_download(self, trigger_action: Callable, timeout: int | None = None):
        """Wait for download to complete after triggering action."""
        timeout = timeout or self.long_timeout

        with self.page.expect_download(timeout=timeout) as download_info:
            trigger_action()

        download = download_info.value
        return download

    def wait_for_popup(self, trigger_action: Callable, timeout: int | None = None):
        """Wait for popup window after triggering action."""
        timeout = timeout or self.default_timeout

        with self.page.expect_popup(timeout=timeout) as popup_info:
            trigger_action()

        popup = popup_info.value
        return popup

    def wait_for_console_message(self, message_text: str, timeout: int | None = None):
        """Wait for specific console message."""
        timeout = timeout or self.default_timeout

        def check_message(msg):
            return message_text in msg.text

        with self.page.expect_console_message(
            predicate=check_message, timeout=timeout
        ) as msg_info:
            pass

        return msg_info.value

    def execute_script(self, script: str, *args):
        """Execute JavaScript in page context."""
        return self.page.evaluate(script, *args)

    def add_style_tag(self, content: str):
        """Add custom CSS to page."""
        self.page.add_style_tag(content=content)

    def mock_api_response(self, url_pattern: str, response_data: dict):
        """Mock API response for testing."""
        self.page.route(
            url_pattern,
            lambda route: route.fulfill(
                status=200, content_type="application/json", body=response_data
            ),
        )

    def clear_mocks(self):
        """Clear all route mocks."""
        self.page.unroute_all()

    def wait_with_custom_condition(
        self, condition_func: Callable, timeout: int | None = None
    ):
        """Wait for custom condition to be true."""
        timeout = timeout or self.default_timeout
        self.page.wait_for_function(condition_func, timeout=timeout)
