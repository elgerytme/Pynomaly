"""
Enhanced conftest.py for Docker-based UI testing with failure capture.
"""

import pytest
from playwright.sync_api import sync_playwright

from .test_config import (
    TEST_CONFIG,
    capture_test_artifacts,
    get_browser_config,
    get_context_config,
)


@pytest.fixture(scope="session")
def browser_type_name():
    """Get browser type from configuration."""
    return TEST_CONFIG["browser"]


@pytest.fixture(scope="session")
def browser(browser_type_name):
    """Create browser instance with enhanced configuration."""
    with sync_playwright() as p:
        browser_type = getattr(p, browser_type_name)
        browser_config = get_browser_config()

        browser = browser_type.launch(**browser_config)
        yield browser
        browser.close()


@pytest.fixture
def context(browser):
    """Create browser context with enhanced configuration."""
    context_config = get_context_config()

    # Start tracing if enabled
    if TEST_CONFIG["record_traces"]:
        context_config["trace"] = True

    context = browser.new_context(**context_config)

    # Start tracing
    if TEST_CONFIG["record_traces"]:
        context.tracing.start(screenshots=True, snapshots=True, sources=True)

    yield context

    # Stop tracing
    if TEST_CONFIG["record_traces"]:
        context.tracing.stop()

    context.close()


@pytest.fixture
def page(context):
    """Create page with enhanced error handling."""
    page = context.new_page()

    # Set up console and error logging
    page.on("console", lambda msg: print(f"Console {msg.type}: {msg.text}"))
    page.on("pageerror", lambda error: print(f"Page error: {error}"))

    # Set default timeout
    page.set_default_timeout(TEST_CONFIG["timeout"])

    yield page
    page.close()


@pytest.fixture(autouse=True)
def capture_artifacts_on_failure(request, page):
    """Automatically capture artifacts on test failure."""
    yield
    capture_test_artifacts(request, page)


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Hook to capture test results for artifact capture."""
    outcome = yield
    rep = outcome.get_result()
    setattr(item, "rep_" + rep.when, rep)


# Markers for test organization
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "browser(name): mark test to run with specific browser"
    )
    config.addinivalue_line(
        "markers", "timeout(seconds): mark test with custom timeout"
    )
    config.addinivalue_line("markers", "flaky: mark test as potentially flaky")
    config.addinivalue_line("markers", "slow: mark test as slow-running")
    config.addinivalue_line("markers", "critical: mark test as critical")


# Custom fixtures for page objects
@pytest.fixture
def dashboard_page(page):
    """Create dashboard page object."""
    from ..pages import DashboardPage

    return DashboardPage(page, TEST_CONFIG["base_url"])


@pytest.fixture
def datasets_page(page):
    """Create datasets page object."""
    from ..pages import DatasetsPage

    return DatasetsPage(page, TEST_CONFIG["base_url"])


@pytest.fixture
def detectors_page(page):
    """Create detectors page object."""
    from ..pages import DetectorsPage

    return DetectorsPage(page, TEST_CONFIG["base_url"])


@pytest.fixture
def detection_page(page):
    """Create detection page object."""
    from ..pages import DetectionPage

    return DetectionPage(page, TEST_CONFIG["base_url"])


@pytest.fixture
def visualizations_page(page):
    """Create visualizations page object."""
    from ..pages import VisualizationsPage

    return VisualizationsPage(page, TEST_CONFIG["base_url"])


# Test data fixtures
@pytest.fixture
def sample_csv_data():
    """Generate sample CSV data for testing."""
    import tempfile

    import numpy as np
    import pandas as pd

    # Generate sample data
    np.random.seed(42)
    data = {
        "feature1": np.random.normal(0, 1, 100),
        "feature2": np.random.normal(0, 1, 100),
        "feature3": np.random.normal(0, 1, 100),
    }

    # Add some anomalies
    data["feature1"][-10:] = np.random.normal(3, 0.5, 10)
    data["feature2"][-10:] = np.random.normal(3, 0.5, 10)
    data["feature3"][-10:] = np.random.normal(3, 0.5, 10)

    df = pd.DataFrame(data)

    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
    df.to_csv(temp_file.name, index=False)
    temp_file.close()

    yield temp_file.name

    # Cleanup
    import os

    os.unlink(temp_file.name)


@pytest.fixture
def wait_for_app_ready(page):
    """Wait for application to be ready."""
    try:
        page.goto(f"{TEST_CONFIG['base_url']}/health", wait_until="networkidle")
        health_response = page.text_content("body")
        if "healthy" not in health_response.lower():
            pytest.skip("Application not ready")
    except Exception:
        pytest.skip("Application not accessible")
