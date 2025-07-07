"""
BDD Test Configuration

This file provides pytest configuration and fixtures specifically for BDD testing
with Gherkin scenarios and step definitions.
"""

import os
from pathlib import Path

import pytest
from playwright.sync_api import Browser, BrowserContext, Page, sync_playwright
from pytest_bdd import given, parsers, then, when

# Test configuration
TEST_BASE_URL = os.getenv("PYNOMALY_BASE_URL", "http://localhost:8000")
TEST_TIMEOUT = int(os.getenv("PYNOMALY_TEST_TIMEOUT", "30000"))
HEADLESS = os.getenv("PYNOMALY_HEADLESS", "true").lower() == "true"
BROWSER_TYPE = os.getenv("BROWSER", "chromium")
DEVICE = os.getenv("DEVICE", "Desktop Chrome")

# Test markers for BDD scenarios
pytest_markers = [
    "accessibility: Accessibility compliance tests",
    "performance: Performance and Core Web Vitals tests",
    "workflow: Complete user workflow tests",
    "cross_browser: Cross-browser compatibility tests",
    "security: Security-focused scenarios",
    "ml_engineer: ML engineer workflow tests",
    "data_scientist: Data scientist workflow tests",
    "slow: Long-running test scenarios",
    "critical: Critical path tests that must pass",
    "smoke: Quick smoke test scenarios",
]


def pytest_configure(config):
    """Configure pytest with BDD-specific settings."""
    for marker in pytest_markers:
        config.addinivalue_line("markers", marker)


@pytest.fixture(scope="session")
def playwright():
    """Session-scoped Playwright instance."""
    with sync_playwright() as p:
        yield p


@pytest.fixture(scope="session")
def browser(playwright):
    """Session-scoped browser instance."""
    browser_type = getattr(playwright, BROWSER_TYPE)
    browser = browser_type.launch(
        headless=HEADLESS,
        slow_mo=50 if not HEADLESS else 0,
        args=[
            "--disable-web-security",
            "--disable-features=VizDisplayCompositor",
            "--disable-dev-shm-usage",
            "--no-sandbox",
        ],
    )
    yield browser
    browser.close()


@pytest.fixture(scope="function")
def context(browser):
    """Function-scoped browser context for test isolation."""
    context = browser.new_context(
        viewport={"width": 1920, "height": 1080},
        user_agent="Mozilla/5.0 (BDD Test) AppleWebKit/537.36 Chrome/120.0.0.0",
        ignore_https_errors=True,
        record_video_dir="test_reports/bdd/videos/" if not HEADLESS else None,
        record_video_size={"width": 1280, "height": 720},
    )
    yield context
    context.close()


@pytest.fixture(scope="function")
def page(context):
    """Function-scoped page instance for each test."""
    page = context.new_page()
    page.set_default_timeout(TEST_TIMEOUT)

    # Enable console logging in debug mode
    if not HEADLESS:
        page.on("console", lambda msg: print(f"Console: {msg.text}"))
        page.on("pageerror", lambda err: print(f"Page Error: {err}"))

    yield page

    # Take screenshot on test failure
    if hasattr(page, "_test_failed") and page._test_failed:
        screenshot_dir = Path("test_reports/bdd/screenshots")
        screenshot_dir.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=screenshot_dir / f"failed_{pytest.current_test}.png")


@pytest.fixture(scope="function")
def ui_helper(page):
    """Helper class for common UI interactions in BDD tests."""

    class UIHelper:
        def __init__(self, page: Page):
            self.page = page

        def navigate_to(self, path: str):
            """Navigate to a path relative to the base URL."""
            full_url = f"{TEST_BASE_URL}{path}"
            self.page.goto(full_url)
            self.page.wait_for_load_state("networkidle")

        def wait_for_element(self, selector: str, timeout: int = TEST_TIMEOUT):
            """Wait for an element to be visible."""
            return self.page.wait_for_selector(selector, timeout=timeout)

        def click_and_wait(self, selector: str, wait_selector: str = None):
            """Click an element and optionally wait for another element."""
            self.page.click(selector)
            if wait_selector:
                self.wait_for_element(wait_selector)

        def fill_form_field(self, selector: str, value: str):
            """Fill a form field with a value."""
            self.page.fill(selector, value)

        def upload_file(self, selector: str, file_path: str):
            """Upload a file to a file input."""
            self.page.set_input_files(selector, file_path)

        def get_text_content(self, selector: str) -> str:
            """Get text content from an element."""
            return self.page.locator(selector).text_content()

        def is_element_visible(self, selector: str) -> bool:
            """Check if an element is visible."""
            return self.page.locator(selector).is_visible()

        def wait_for_api_response(self, url_pattern: str, timeout: int = TEST_TIMEOUT):
            """Wait for a specific API response."""
            with self.page.expect_response(
                url_pattern, timeout=timeout
            ) as response_info:
                pass
            return response_info.value

        def check_accessibility(self):
            """Run basic accessibility checks."""
            # This would integrate with axe-core or similar
            # For now, just check basic ARIA attributes
            aria_issues = []

            # Check for missing alt text on images
            images = self.page.locator("img").all()
            for img in images:
                if not img.get_attribute("alt"):
                    aria_issues.append(f"Image missing alt text: {img}")

            # Check for form labels
            inputs = self.page.locator(
                "input[type='text'], input[type='email'], input[type='password']"
            ).all()
            for input_elem in inputs:
                label_id = input_elem.get_attribute("aria-labelledby")
                if not label_id and not input_elem.get_attribute("aria-label"):
                    aria_issues.append(f"Input missing label: {input_elem}")

            return aria_issues

        def measure_performance(self):
            """Measure basic performance metrics."""
            # Get navigation timing
            timing = self.page.evaluate(
                """() => {
                const nav = performance.getEntriesByType('navigation')[0];
                return {
                    loadTime: nav.loadEventEnd - nav.loadEventStart,
                    domContentLoaded: nav.domContentLoadedEventEnd - nav.domContentLoadedEventStart,
                    firstPaint: performance.getEntriesByName('first-paint')[0]?.startTime || 0,
                    firstContentfulPaint: performance.getEntriesByName('first-contentful-paint')[0]?.startTime || 0
                };
            }"""
            )
            return timing

    return UIHelper(page)


@pytest.fixture(scope="function")
def test_data():
    """Test data factory for BDD scenarios."""

    class TestDataFactory:
        @staticmethod
        def get_sample_dataset():
            """Get sample dataset for testing."""
            return {
                "name": "Test Dataset",
                "description": "Sample dataset for BDD testing",
                "format": "csv",
                "size": "10KB",
                "columns": ["feature1", "feature2", "feature3", "target"],
            }

        @staticmethod
        def get_detector_config():
            """Get sample detector configuration."""
            return {
                "name": "Test Detector",
                "algorithm": "IsolationForest",
                "contamination": 0.1,
                "parameters": {"n_estimators": 100, "max_samples": "auto"},
            }

        @staticmethod
        def get_user_credentials():
            """Get test user credentials."""
            return {
                "username": "test_user@example.com",
                "password": "TestPassword123!",
                "role": "data_scientist",
            }

    return TestDataFactory()


# Pytest hooks for BDD-specific behavior
@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Generate test reports with BDD context."""
    outcome = yield
    rep = outcome.get_result()

    # Add BDD-specific information to test reports
    if hasattr(item, "funcargs") and "page" in item.funcargs:
        page = item.funcargs["page"]
        if rep.failed:
            page._test_failed = True
            # Additional failure handling could go here


def pytest_html_report_title(report):
    """Customize HTML report title for BDD tests."""
    report.title = "Pynomaly BDD Test Report"


def pytest_collection_modifyitems(config, items):
    """Modify test collection for BDD-specific handling."""
    for item in items:
        # Add markers based on test file location or naming
        if "accessibility" in item.nodeid:
            item.add_marker(pytest.mark.accessibility)
        if "performance" in item.nodeid:
            item.add_marker(pytest.mark.performance)
        if "workflow" in item.nodeid:
            item.add_marker(pytest.mark.workflow)
        if "cross_browser" in item.nodeid:
            item.add_marker(pytest.mark.cross_browser)
