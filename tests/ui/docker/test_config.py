"""
Enhanced test configuration for Docker-based UI testing.
"""

import os
from datetime import datetime
from pathlib import Path

import pytest

# Test configuration
TEST_CONFIG = {
    "base_url": os.getenv("PYNOMALY_BASE_URL", "http://localhost:8000"),
    "browser": os.getenv("BROWSER", "chrome"),
    "headless": os.getenv("HEADLESS", "true").lower() == "true",
    "slow_mo": int(os.getenv("SLOW_MO", "0")),
    "timeout": int(os.getenv("TIMEOUT", "30000")),
    "take_screenshots": os.getenv("TAKE_SCREENSHOTS", "true").lower() == "true",
    "record_videos": os.getenv("RECORD_VIDEOS", "true").lower() == "true",
    "record_traces": os.getenv("RECORD_TRACES", "false").lower() == "true",
    "parallel_workers": int(os.getenv("PARALLEL_WORKERS", "2")),
}

# Artifact directories
SCREENSHOTS_DIR = Path("test_reports/screenshots")
VIDEOS_DIR = Path("test_reports/videos")
TRACES_DIR = Path("test_reports/traces")
REPORTS_DIR = Path("test_reports")

# Create directories
for directory in [SCREENSHOTS_DIR, VIDEOS_DIR, TRACES_DIR, REPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


def get_browser_config():
    """Get browser configuration based on environment."""
    browser_name = TEST_CONFIG["browser"]

    config = {
        "headless": TEST_CONFIG["headless"],
        "slow_mo": TEST_CONFIG["slow_mo"],
        "args": [
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--disable-web-security",
            "--disable-features=VizDisplayCompositor",
        ],
    }

    if browser_name == "chrome":
        config["args"].extend(
            [
                "--disable-gpu",
                "--disable-background-timer-throttling",
                "--disable-backgrounding-occluded-windows",
                "--disable-renderer-backgrounding",
            ]
        )
    elif browser_name == "firefox":
        config["args"].extend(
            [
                "--disable-gpu",
                "--disable-background-timer-throttling",
            ]
        )

    return config


def get_context_config():
    """Get browser context configuration."""
    config = {
        "viewport": {"width": 1920, "height": 1080},
        "user_agent": f"Mozilla/5.0 (compatible; Pynomaly-UI-Tests-{TEST_CONFIG['browser']}/1.0)",
        "ignore_https_errors": True,
        "record_video_dir": str(VIDEOS_DIR) if TEST_CONFIG["record_videos"] else None,
        "record_video_size": {"width": 1920, "height": 1080},
    }

    return config


def capture_test_artifacts(request, page):
    """Capture test artifacts on failure."""
    if hasattr(request.node, "rep_call") and request.node.rep_call.failed:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        test_name = request.node.name
        browser_name = TEST_CONFIG["browser"]

        # Capture screenshot
        if TEST_CONFIG["take_screenshots"]:
            screenshot_path = (
                SCREENSHOTS_DIR / f"{browser_name}_{test_name}_{timestamp}.png"
            )
            page.screenshot(path=str(screenshot_path), full_page=True)
            print(f"Screenshot saved: {screenshot_path}")

        # Capture video
        if TEST_CONFIG["record_videos"] and page.video:
            video_path = page.video.path()
            if video_path:
                final_video_path = (
                    VIDEOS_DIR / f"{browser_name}_{test_name}_{timestamp}.webm"
                )
                import shutil

                shutil.copy(video_path, final_video_path)
                print(f"Video saved: {final_video_path}")

        # Capture trace
        if TEST_CONFIG["record_traces"]:
            trace_path = TRACES_DIR / f"{browser_name}_{test_name}_{timestamp}.zip"
            page.context.tracing.stop(path=str(trace_path))
            print(f"Trace saved: {trace_path}")


def pytest_configure(config):
    """Configure pytest for UI testing."""
    # Set up parallel execution
    if TEST_CONFIG["parallel_workers"] > 1:
        config.option.numprocesses = TEST_CONFIG["parallel_workers"]

    # Set up HTML reporting
    config.option.htmlpath = REPORTS_DIR / f"report_{TEST_CONFIG['browser']}.html"

    # Set up JSON reporting
    config.option.json_report_file = (
        REPORTS_DIR / f"report_{TEST_CONFIG['browser']}.json"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test items for better organization."""
    for item in items:
        # Add browser marker
        item.add_marker(pytest.mark.browser(TEST_CONFIG["browser"]))

        # Add timeout marker
        item.add_marker(pytest.mark.timeout(TEST_CONFIG["timeout"] / 1000))

        # Add retry marker for flaky tests
        if "flaky" in item.keywords:
            item.add_marker(pytest.mark.flaky(reruns=3, reruns_delay=1))
