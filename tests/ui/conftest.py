"""UI Test Configuration and Fixtures."""

import os
import pytest
from pathlib import Path
from typing import Generator, Dict, Any
from datetime import datetime

from playwright.sync_api import Playwright, Browser, BrowserContext, Page


# Configuration
BASE_URL = os.getenv("PYNOMALY_BASE_URL", "http://localhost:8000")
SCREENSHOTS_DIR = Path("screenshots")
TEST_RESULTS_DIR = Path("test-results")
VISUAL_BASELINES_DIR = Path("visual-baselines")

# Create directories
SCREENSHOTS_DIR.mkdir(exist_ok=True)
TEST_RESULTS_DIR.mkdir(exist_ok=True)
VISUAL_BASELINES_DIR.mkdir(exist_ok=True)


@pytest.fixture(scope="session")
def browser_type_name() -> str:
    """Get browser type from environment or default to chromium."""
    return os.getenv("BROWSER", "chromium")


@pytest.fixture(scope="session")
def browser(playwright: Playwright, browser_type_name: str) -> Generator[Browser, None, None]:
    """Create browser instance."""
    browser_type = getattr(playwright, browser_type_name)
    browser = browser_type.launch(
        headless=True,
        args=[
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--disable-web-security",
            "--disable-features=VizDisplayCompositor"
        ]
    )
    yield browser
    browser.close()


@pytest.fixture
def context(browser: Browser) -> Generator[BrowserContext, None, None]:
    """Create browser context with custom settings."""
    context = browser.new_context(
        viewport={"width": 1920, "height": 1080},
        user_agent="Mozilla/5.0 (compatible; Pynomaly-UI-Tests/1.0)",
        ignore_https_errors=True,
        record_video_dir=str(TEST_RESULTS_DIR / "videos") if os.getenv("RECORD_VIDEO") else None,
    )
    yield context
    context.close()


@pytest.fixture
def page(context: BrowserContext) -> Generator[Page, None, None]:
    """Create a new page."""
    page = context.new_page()
    
    # Set up console logging
    page.on("console", lambda msg: print(f"Console {msg.type}: {msg.text}"))
    page.on("pageerror", lambda error: print(f"Page error: {error}"))
    
    yield page
    page.close()


@pytest.fixture
def authenticated_page(page: Page) -> Page:
    """Create an authenticated page (if auth is enabled)."""
    # For now, just return the page. In future, add login logic if needed
    return page


@pytest.fixture
def mobile_page(browser: Browser) -> Generator[Page, None, None]:
    """Create a mobile viewport page."""
    context = browser.new_context(
        viewport={"width": 375, "height": 667},  # iPhone SE
        user_agent="Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/605.1.15",
        is_mobile=True,
        has_touch=True,
    )
    page = context.new_page()
    yield page
    page.close()
    context.close()


@pytest.fixture
def tablet_page(browser: Browser) -> Generator[Page, None, None]:
    """Create a tablet viewport page."""
    context = browser.new_context(
        viewport={"width": 768, "height": 1024},  # iPad
        user_agent="Mozilla/5.0 (iPad; CPU OS 14_0 like Mac OS X) AppleWebKit/605.1.15",
        is_mobile=True,
        has_touch=True,
    )
    page = context.new_page()
    yield page
    page.close()
    context.close()


@pytest.fixture(autouse=True)
def take_screenshot_on_failure(request, page: Page):
    """Automatically take screenshot on test failure."""
    yield
    if request.node.rep_call.failed:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        test_name = request.node.name
        screenshot_path = SCREENSHOTS_DIR / f"failure_{test_name}_{timestamp}.png"
        page.screenshot(path=str(screenshot_path), full_page=True)
        print(f"Screenshot saved: {screenshot_path}")


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Hook to capture test results for screenshot fixture."""
    outcome = yield
    rep = outcome.get_result()
    setattr(item, "rep_" + rep.when, rep)


# Test data fixtures
@pytest.fixture
def sample_detector_data() -> Dict[str, Any]:
    """Sample detector data for testing."""
    return {
        "name": "Test Detector",
        "algorithm": "IsolationForest",
        "description": "Test detector for UI automation",
        "contamination": 0.1
    }


@pytest.fixture
def sample_dataset_data() -> Dict[str, Any]:
    """Sample dataset data for testing."""
    return {
        "name": "Test Dataset",
        "description": "Test dataset for UI automation",
        "features": ["feature1", "feature2", "feature3"]
    }


# Page object fixtures
@pytest.fixture
def dashboard_page(page: Page):
    """Dashboard page object."""
    from tests.ui.page_objects.dashboard_page import DashboardPage
    return DashboardPage(page, BASE_URL)


@pytest.fixture
def detectors_page(page: Page):
    """Detectors page object."""
    from tests.ui.page_objects.detectors_page import DetectorsPage
    return DetectorsPage(page, BASE_URL)


@pytest.fixture
def datasets_page(page: Page):
    """Datasets page object."""
    from tests.ui.page_objects.datasets_page import DatasetsPage
    return DatasetsPage(page, BASE_URL)


@pytest.fixture
def detection_page(page: Page):
    """Detection page object."""
    from tests.ui.page_objects.detection_page import DetectionPage
    return DetectionPage(page, BASE_URL)


@pytest.fixture
def visualizations_page(page: Page):
    """Visualizations page object."""
    from tests.ui.page_objects.visualizations_page import VisualizationsPage
    return VisualizationsPage(page, BASE_URL)