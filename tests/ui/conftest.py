"""UI Test Configuration - Playwright-specific fixtures that extend the root conftest.py."""

# Import all fixtures from root conftest
import os
import tempfile
from collections.abc import Generator
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from playwright.async_api import (
    Browser,
    BrowserContext,
    Page,
    Playwright,
)

try:
    from ..conftest import *
except ImportError:
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    try:
        from conftest import *
    except ImportError:
        # Minimal pytest configuration for standalone tests
        pass

# Enhanced Configuration
BASE_URL = os.getenv("PYNOMALY_BASE_URL", "http://localhost:8000")
SCREENSHOTS_DIR = Path("test_reports/screenshots")
VIDEOS_DIR = Path("test_reports/videos")
TRACES_DIR = Path("test_reports/traces")

# Test configuration
TEST_CONFIG = {
    "base_url": BASE_URL,
    "timeout": 30000,  # 30 seconds
    "headless": os.getenv("HEADLESS", "true").lower() == "true",
    "slow_mo": int(os.getenv("SLOW_MO", "0")),
    "viewport": {"width": 1920, "height": 1080},
    "take_screenshots": os.getenv("TAKE_SCREENSHOTS", "true").lower() == "true",
    "record_videos": os.getenv("RECORD_VIDEOS", "false").lower() == "true",
    "record_traces": os.getenv("RECORD_TRACES", "false").lower() == "true",
}

# Create directories
for directory in [SCREENSHOTS_DIR, VIDEOS_DIR, TRACES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


@pytest.fixture(scope="session")
def browser_type_name() -> str:
    """Get browser type from environment or default to chromium."""
    return os.getenv("BROWSER", "chromium")


@pytest.fixture(scope="session")
def browser(
    playwright: Playwright, browser_type_name: str
) -> Generator[Browser, None, None]:
    """Create browser instance."""
    browser_type = getattr(playwright, browser_type_name)
    browser = browser_type.launch(
        headless=TEST_CONFIG["headless"],
        args=[
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--disable-web-security",
            "--disable-features=VizDisplayCompositor",
        ],
    )
    yield browser
    browser.close()


@pytest.fixture
def context(browser: Browser) -> Generator[BrowserContext, None, None]:
    """Create browser context with custom settings."""
    context = browser.new_context(
        viewport=TEST_CONFIG["viewport"],
        user_agent="Mozilla/5.0 (compatible; Pynomaly-UI-Tests/1.0)",
        ignore_https_errors=True,
        record_video_dir=(str(VIDEOS_DIR) if TEST_CONFIG["record_videos"] else None),
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


@pytest.fixture(autouse=True)
def take_screenshot_on_failure(request, page: Page):
    """Automatically take screenshot on test failure."""
    yield
    if hasattr(request.node, "rep_call") and request.node.rep_call.failed:
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


# Sample data fixtures for UI testing
@pytest.fixture
def sample_detector_data() -> dict[str, Any]:
    """Sample detector data for testing."""
    return {
        "name": "Test Detector",
        "algorithm": "IsolationForest",
        "description": "Test detector for UI automation",
        "contamination": 0.1,
    }


@pytest.fixture
def sample_dataset_data() -> dict[str, Any]:
    """Sample dataset data for testing."""
    return {
        "name": "Test Dataset",
        "description": "Test dataset for UI automation",
        "features": ["feature1", "feature2", "feature3"],
    }


@pytest.fixture
def sample_csv_file():
    """Create sample CSV file for upload tests."""
    import numpy as np
    import pandas as pd

    # Generate sample data with anomalies
    np.random.seed(42)
    n_samples = 100

    normal_data = np.random.multivariate_normal(
        mean=[0, 0, 0],
        cov=[[1, 0.3, 0.1], [0.3, 1, 0.2], [0.1, 0.2, 1]],
        size=n_samples - 10,
    )

    anomaly_data = np.random.multivariate_normal(
        mean=[3, 3, 3], cov=[[0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]], size=10
    )

    data = np.vstack([normal_data, anomaly_data])
    df = pd.DataFrame(data, columns=["feature1", "feature2", "feature3"])
    df["timestamp"] = pd.date_range("2024-01-01", periods=n_samples, freq="1H")

    temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
    df.to_csv(temp_file.name, index=False)
    temp_file.close()

    yield temp_file.name
    # Cleanup
    os.unlink(temp_file.name)
