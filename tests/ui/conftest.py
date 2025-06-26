"""UI Test Configuration and Fixtures - Enhanced for Production Readiness."""

import asyncio
import json
import os
import tempfile
import time
from collections.abc import Generator
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncGenerator

import pytest
from playwright.async_api import (
    Browser,
    BrowserContext,
    Page,
    Playwright,
    async_playwright,
)
from playwright.sync_api import Browser as SyncBrowser, BrowserContext as SyncBrowserContext, Page as SyncPage, Playwright as SyncPlaywright

# Enhanced Configuration
BASE_URL = os.getenv("PYNOMALY_BASE_URL", "http://localhost:8000")
SCREENSHOTS_DIR = Path("test_reports/screenshots")
VIDEOS_DIR = Path("test_reports/videos")
TRACES_DIR = Path("test_reports/traces")
VISUAL_BASELINES_DIR = Path("test_reports/visual-baselines")
ACCESSIBILITY_REPORTS_DIR = Path("test_reports/accessibility")
PERFORMANCE_REPORTS_DIR = Path("test_reports/performance")

# Test configuration with comprehensive options
TEST_CONFIG = {
    "base_url": BASE_URL,
    "timeout": 30000,  # 30 seconds
    "headless": os.getenv("HEADLESS", "true").lower() == "true",
    "slow_mo": int(os.getenv("SLOW_MO", "0")),
    "viewport": {"width": 1920, "height": 1080},
    "browsers": ["chromium", "firefox", "webkit"],
    "mobile_devices": ["iPhone 12", "Pixel 5", "iPad Pro"],
    "desktop_devices": [
        {"width": 1920, "height": 1080, "name": "Desktop HD"},
        {"width": 1366, "height": 768, "name": "Desktop Standard"},
        {"width": 1440, "height": 900, "name": "Desktop Large"},
    ],
    "take_screenshots": os.getenv("TAKE_SCREENSHOTS", "true").lower() == "true",
    "record_videos": os.getenv("RECORD_VIDEOS", "false").lower() == "true",
    "record_traces": os.getenv("RECORD_TRACES", "false").lower() == "true",
    "visual_testing": os.getenv("VISUAL_TESTING", "false").lower() == "true",
    "accessibility_testing": os.getenv("ACCESSIBILITY_TESTING", "true").lower() == "true",
    "performance_testing": os.getenv("PERFORMANCE_TESTING", "true").lower() == "true",
}

# Create all necessary directories
for directory in [SCREENSHOTS_DIR, VIDEOS_DIR, TRACES_DIR, VISUAL_BASELINES_DIR, 
                  ACCESSIBILITY_REPORTS_DIR, PERFORMANCE_REPORTS_DIR]:
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
        headless=True,
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
        viewport={"width": 1920, "height": 1080},
        user_agent="Mozilla/5.0 (compatible; Pynomaly-UI-Tests/1.0)",
        ignore_https_errors=True,
        record_video_dir=(
            str(TEST_RESULTS_DIR / "videos") if os.getenv("RECORD_VIDEO") else None
        ),
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


# Enhanced Testing Infrastructure
class UITestHelper:
    """Enhanced helper class for UI testing operations with comprehensive features."""
    
    def __init__(self, page: Page, config: dict):
        self.page = page
        self.config = config
        
    async def wait_for_htmx(self, timeout: int = 5000):
        """Wait for HTMX requests to complete."""
        try:
            await self.page.wait_for_function(
                "() => !document.body.classList.contains('htmx-request')",
                timeout=timeout
            )
        except:
            # If HTMX is not present, just continue
            pass
    
    async def wait_for_loading(self, timeout: int = 10000):
        """Wait for loading indicators to disappear."""
        try:
            await self.page.wait_for_selector(
                "[data-loading], .loading, .spinner, .htmx-indicator",
                state="detached",
                timeout=timeout
            )
        except:
            pass  # No loading indicators found
    
    async def take_screenshot(self, name: str, full_page: bool = True):
        """Take screenshot with timestamp and proper naming."""
        if not self.config.get("take_screenshots", True):
            return None
            
        timestamp = int(time.time())
        filename = f"{name}_{timestamp}.png"
        filepath = SCREENSHOTS_DIR / filename
        
        await self.page.screenshot(
            path=str(filepath),
            full_page=full_page
        )
        
        return filepath
    
    async def wait_for_url_change(self, timeout: int = 5000):
        """Wait for URL to change (useful for navigation)."""
        current_url = self.page.url
        await self.page.wait_for_function(
            f"() => window.location.href !== '{current_url}'",
            timeout=timeout
        )
    
    async def fill_form_field(self, selector: str, value: str, clear: bool = True):
        """Fill form field with proper waiting and validation."""
        await self.page.wait_for_selector(selector, state="visible")
        if clear:
            await self.page.fill(selector, "")
        await self.page.fill(selector, value)
        
        # Verify the value was set
        actual_value = await self.page.input_value(selector)
        assert actual_value == value, f"Expected '{value}' but got '{actual_value}'"
    
    async def click_and_wait(self, selector: str, wait_for: str = None):
        """Click element and wait for response."""
        await self.page.wait_for_selector(selector, state="visible")
        await self.page.click(selector)
        
        if wait_for:
            await self.page.wait_for_selector(wait_for, state="visible")
        else:
            await self.wait_for_htmx()
            await self.wait_for_loading()
    
    async def upload_file(self, selector: str, file_path: str):
        """Upload file to input element with validation."""
        await self.page.wait_for_selector(selector)
        file_input = await self.page.query_selector(selector)
        await file_input.set_input_files(file_path)
    
    async def get_table_data(self, table_selector: str) -> list:
        """Extract data from table with comprehensive parsing."""
        await self.page.wait_for_selector(table_selector)
        
        rows = await self.page.query_selector_all(f"{table_selector} tbody tr")
        data = []
        
        for row in rows:
            cells = await row.query_selector_all("td")
            row_data = []
            for cell in cells:
                text = await cell.text_content()
                row_data.append(text.strip() if text else "")
            data.append(row_data)
        
        return data
    
    async def check_accessibility(self) -> dict:
        """Run comprehensive accessibility checks with axe-core."""
        try:
            # Inject axe-core
            await self.page.add_script_tag(
                url="https://unpkg.com/axe-core@4.7.0/axe.min.js"
            )
            
            # Run axe scan
            results = await self.page.evaluate("""
                async () => {
                    return await axe.run();
                }
            """)
            
            # Save accessibility report
            if self.config.get("accessibility_testing", True):
                timestamp = int(time.time())
                report_file = ACCESSIBILITY_REPORTS_DIR / f"accessibility_report_{timestamp}.json"
                with open(report_file, 'w') as f:
                    json.dump(results, f, indent=2)
            
            return results
        except Exception as e:
            print(f"Accessibility check failed: {e}")
            return {"violations": [], "passes": [], "error": str(e)}
    
    async def check_performance(self) -> dict:
        """Collect performance metrics including Core Web Vitals."""
        try:
            # Get navigation timing
            navigation_timing = await self.page.evaluate("""
                () => {
                    const navigation = performance.getEntriesByType('navigation')[0];
                    const paint = performance.getEntriesByType('paint');
                    
                    return {
                        domContentLoaded: navigation.domContentLoadedEventEnd - navigation.domContentLoadedEventStart,
                        loadComplete: navigation.loadEventEnd - navigation.loadEventStart,
                        firstPaint: paint.find(p => p.name === 'first-paint')?.startTime || 0,
                        firstContentfulPaint: paint.find(p => p.name === 'first-contentful-paint')?.startTime || 0,
                        totalLoadTime: navigation.loadEventEnd - navigation.fetchStart,
                        timeToInteractive: navigation.domInteractive - navigation.fetchStart,
                        pageSize: navigation.transferSize || 0
                    };
                }
            """)
            
            # Get Core Web Vitals
            web_vitals = await self.page.evaluate("""
                () => {
                    return new Promise((resolve) => {
                        let vitals = {};
                        
                        // LCP (Largest Contentful Paint)
                        new PerformanceObserver((entryList) => {
                            const entries = entryList.getEntries();
                            const lastEntry = entries[entries.length - 1];
                            vitals.lcp = lastEntry.startTime;
                        }).observe({entryTypes: ['largest-contentful-paint']});
                        
                        // FID (First Input Delay) - simulation
                        vitals.fid = 0;
                        
                        // CLS (Cumulative Layout Shift)
                        let clsValue = 0;
                        new PerformanceObserver((entryList) => {
                            for (const entry of entryList.getEntries()) {
                                if (!entry.hadRecentInput) {
                                    clsValue += entry.value;
                                }
                            }
                            vitals.cls = clsValue;
                        }).observe({entryTypes: ['layout-shift']});
                        
                        setTimeout(() => resolve(vitals), 2000);
                    });
                }
            """)
            
            performance_data = {**navigation_timing, **web_vitals}
            
            # Save performance report
            if self.config.get("performance_testing", True):
                timestamp = int(time.time())
                report_file = PERFORMANCE_REPORTS_DIR / f"performance_report_{timestamp}.json"
                with open(report_file, 'w') as f:
                    json.dump(performance_data, f, indent=2)
            
            return performance_data
        except Exception as e:
            print(f"Performance check failed: {e}")
            return {"error": str(e)}
    
    async def validate_responsive_design(self, breakpoints: list = None):
        """Validate responsive design across different viewport sizes."""
        if breakpoints is None:
            breakpoints = [
                {"width": 320, "height": 568, "name": "Mobile"},
                {"width": 768, "height": 1024, "name": "Tablet"},
                {"width": 1024, "height": 768, "name": "Desktop Small"},
                {"width": 1920, "height": 1080, "name": "Desktop Large"},
            ]
        
        results = {}
        original_viewport = self.page.viewport_size
        
        for breakpoint in breakpoints:
            await self.page.set_viewport_size(breakpoint)
            await self.page.wait_for_timeout(500)  # Allow layout to settle
            
            # Check if content is visible and accessible
            is_content_visible = await self.page.evaluate("""
                () => {
                    const body = document.body;
                    const hasOverflow = body.scrollWidth > window.innerWidth;
                    const hasHiddenContent = Array.from(document.querySelectorAll('*')).some(
                        el => getComputedStyle(el).overflow === 'hidden' && el.scrollWidth > el.clientWidth
                    );
                    return {
                        hasHorizontalOverflow: hasOverflow,
                        hasHiddenContent: hasHiddenContent,
                        viewportWidth: window.innerWidth,
                        viewportHeight: window.innerHeight
                    };
                }
            """)
            
            results[breakpoint["name"]] = {
                "viewport": breakpoint,
                "content_analysis": is_content_visible,
                "screenshot": await self.take_screenshot(f"responsive_{breakpoint['name'].lower()}")
            }
        
        # Restore original viewport
        if original_viewport:
            await self.page.set_viewport_size(original_viewport)
        
        return results


class PerformanceMonitor:
    """Enhanced performance monitoring with Core Web Vitals tracking."""
    
    def __init__(self, page: Page):
        self.page = page
        self.metrics = {}
        self.start_time = None
    
    async def start_monitoring(self):
        """Initialize performance monitoring."""
        self.start_time = time.time()
        
        # Inject performance monitoring script
        await self.page.add_init_script("""
            window.performanceMetrics = {
                navigationStart: performance.timeOrigin,
                entries: [],
                vitals: {}
            };
            
            // Performance Observer for all entry types
            if ('PerformanceObserver' in window) {
                const observer = new PerformanceObserver((list) => {
                    window.performanceMetrics.entries.push(...list.getEntries());
                });
                observer.observe({entryTypes: ['navigation', 'measure', 'paint', 'largest-contentful-paint', 'layout-shift']});
            }
        """)
    
    async def get_comprehensive_metrics(self) -> dict:
        """Get comprehensive performance metrics."""
        end_time = time.time()
        monitoring_duration = end_time - (self.start_time or end_time)
        
        try:
            metrics = await self.page.evaluate("""
                () => {
                    const navigation = performance.getEntriesByType('navigation')[0];
                    const paint = performance.getEntriesByType('paint');
                    const lcpEntries = performance.getEntriesByType('largest-contentful-paint');
                    const layoutShifts = performance.getEntriesByType('layout-shift');
                    
                    // Calculate CLS
                    let cls = 0;
                    layoutShifts.forEach(entry => {
                        if (!entry.hadRecentInput) {
                            cls += entry.value;
                        }
                    });
                    
                    return {
                        // Navigation timing
                        domContentLoaded: navigation?.domContentLoadedEventEnd - navigation?.domContentLoadedEventStart || 0,
                        loadComplete: navigation?.loadEventEnd - navigation?.loadEventStart || 0,
                        totalLoadTime: navigation?.loadEventEnd - navigation?.fetchStart || 0,
                        
                        // Paint timing
                        firstPaint: paint.find(p => p.name === 'first-paint')?.startTime || 0,
                        firstContentfulPaint: paint.find(p => p.name === 'first-contentful-paint')?.startTime || 0,
                        
                        // Core Web Vitals
                        largestContentfulPaint: lcpEntries[lcpEntries.length - 1]?.startTime || 0,
                        cumulativeLayoutShift: cls,
                        firstInputDelay: 0, // Requires actual user interaction
                        
                        // Resource info
                        transferSize: navigation?.transferSize || 0,
                        resourceCount: performance.getEntriesByType('resource').length,
                        
                        // Memory usage (if available)
                        memoryUsage: (performance as any).memory ? {
                            usedJSHeapSize: (performance as any).memory.usedJSHeapSize,
                            totalJSHeapSize: (performance as any).memory.totalJSHeapSize,
                            jsHeapSizeLimit: (performance as any).memory.jsHeapSizeLimit
                        } : null
                    };
                }
            """)
            
            metrics["monitoringDuration"] = monitoring_duration
            return metrics
        except Exception as e:
            return {"error": str(e), "monitoringDuration": monitoring_duration}


# Enhanced fixtures
@pytest.fixture
async def ui_helper(page: Page) -> UITestHelper:
    """Create enhanced UI test helper instance."""
    return UITestHelper(page, TEST_CONFIG)


@pytest.fixture
async def performance_monitor(page: Page) -> PerformanceMonitor:
    """Create performance monitor instance."""
    monitor = PerformanceMonitor(page)
    await monitor.start_monitoring()
    return monitor


@pytest.fixture
def test_config():
    """Provide test configuration."""
    return TEST_CONFIG


# Cross-browser testing fixtures
@pytest.fixture(params=["chromium", "firefox", "webkit"])
async def cross_browser_page(playwright: Playwright, request) -> AsyncGenerator[Page, None]:
    """Create page for cross-browser testing."""
    browser_name = request.param
    browser_type = getattr(playwright, browser_name)
    
    browser = await browser_type.launch(
        headless=TEST_CONFIG["headless"],
        slow_mo=TEST_CONFIG["slow_mo"]
    )
    
    context = await browser.new_context(
        viewport=TEST_CONFIG["viewport"],
        record_video_dir=str(VIDEOS_DIR) if TEST_CONFIG["record_videos"] else None,
    )
    
    if TEST_CONFIG["record_traces"]:
        await context.tracing.start(screenshots=True, snapshots=True, sources=True)
    
    page = await context.new_page()
    page.browser_name = browser_name  # Store for test identification
    
    yield page
    
    if TEST_CONFIG["record_traces"]:
        trace_file = TRACES_DIR / f"trace_{browser_name}_{int(time.time())}.zip"
        await context.tracing.stop(path=str(trace_file))
    
    await context.close()
    await browser.close()


# Mobile device testing
@pytest.fixture(params=["iPhone 12", "Pixel 5", "iPad Pro"])
async def mobile_device_page(playwright: Playwright, request) -> AsyncGenerator[Page, None]:
    """Create page with mobile device emulation."""
    device_name = request.param
    device = playwright.devices[device_name]
    
    browser = await playwright.chromium.launch(headless=TEST_CONFIG["headless"])
    context = await browser.new_context(**device)
    page = await context.new_page()
    page.device_name = device_name  # Store for test identification
    
    yield page
    
    await context.close()
    await browser.close()


# Utility functions for test data
def create_sample_csv():
    """Create sample CSV file for testing uploads."""
    import pandas as pd
    import numpy as np
    
    # Generate sample data with anomalies
    np.random.seed(42)
    n_samples = 100
    
    normal_data = np.random.multivariate_normal(
        mean=[0, 0, 0],
        cov=[[1, 0.3, 0.1], [0.3, 1, 0.2], [0.1, 0.2, 1]],
        size=n_samples - 10
    )
    
    anomaly_data = np.random.multivariate_normal(
        mean=[3, 3, 3],
        cov=[[0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]],
        size=10
    )
    
    data = np.vstack([normal_data, anomaly_data])
    df = pd.DataFrame(data, columns=['feature1', 'feature2', 'feature3'])
    df['timestamp'] = pd.date_range('2024-01-01', periods=n_samples, freq='1H')
    
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    df.to_csv(temp_file.name, index=False)
    temp_file.close()
    
    return temp_file.name


@pytest.fixture
def sample_csv_file():
    """Create sample CSV file for upload tests."""
    file_path = create_sample_csv()
    yield file_path
    # Cleanup
    os.unlink(file_path)
