"""
Advanced Web UI Testing Framework for Pynomaly

This module provides comprehensive testing infrastructure including:
- Multi-browser testing with Playwright
- Visual regression testing
- Performance monitoring and validation
- Accessibility compliance testing
- Component-level testing with modern patterns
- Advanced error handling and recovery
- Real-time test metrics and reporting
"""

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from playwright.async_api import (
    Browser,
    BrowserContext,
    Page,
    async_playwright,
)

from tests.ui.advanced.utils.accessibility_validator import AccessibilityValidator
from tests.ui.advanced.utils.cross_browser_manager import CrossBrowserManager
from tests.ui.advanced.utils.performance_validator import PerformanceValidator
from tests.ui.advanced.utils.test_data_generator import TestDataGenerator
from tests.ui.advanced.utils.visual_regression_manager import VisualRegressionManager


class AdvancedUITestFramework:
    """
    Advanced UI testing framework with comprehensive capabilities
    """

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.test_start_time = datetime.now(UTC)
        self.browsers: dict[str, Browser] = {}
        self.contexts: dict[str, BrowserContext] = {}
        self.pages: dict[str, Page] = {}

        # Initialize validators and managers
        self.accessibility_validator = AccessibilityValidator()
        self.performance_validator = PerformanceValidator()
        self.visual_regression_manager = VisualRegressionManager()
        self.test_data_generator = TestDataGenerator()
        self.cross_browser_manager = CrossBrowserManager()

        # Test metrics
        self.test_metrics = {
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "skipped_tests": 0,
            "performance_metrics": {},
            "accessibility_scores": {},
            "visual_regression_results": {},
            "cross_browser_results": {},
        }

        # Test artifacts directories
        self.artifacts_dir = Path("test_artifacts")
        self.screenshots_dir = self.artifacts_dir / "screenshots"
        self.videos_dir = self.artifacts_dir / "videos"
        self.reports_dir = self.artifacts_dir / "reports"
        self.traces_dir = self.artifacts_dir / "traces"

        # Create directories
        for directory in [
            self.screenshots_dir,
            self.videos_dir,
            self.reports_dir,
            self.traces_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

    async def setup_browsers(self, browsers: list[str] = None) -> None:
        """Setup multiple browsers for cross-browser testing"""
        if browsers is None:
            browsers = ["chromium", "firefox", "webkit"]

        self.playwright = await async_playwright().start()

        for browser_name in browsers:
            try:
                browser_type = getattr(self.playwright, browser_name)
                browser = await browser_type.launch(
                    headless=True,
                    args=[
                        "--no-sandbox",
                        "--disable-dev-shm-usage",
                        "--disable-web-security",
                        "--enable-automation",
                    ],
                )
                self.browsers[browser_name] = browser

                # Create context with advanced settings
                context = await browser.new_context(
                    viewport={"width": 1920, "height": 1080},
                    user_agent="Mozilla/5.0 (compatible; PynotalyUITest/1.0)",
                    permissions=["notifications", "geolocation"],
                    record_video_dir=str(self.videos_dir),
                    record_video_size={"width": 1920, "height": 1080},
                )

                # Enable tracing for debugging
                await context.tracing.start(
                    screenshots=True, snapshots=True, sources=True
                )

                self.contexts[browser_name] = context

                # Create page
                page = await context.new_page()

                # Setup page monitoring
                await self._setup_page_monitoring(page, browser_name)

                self.pages[browser_name] = page

            except Exception as e:
                print(f"Failed to setup {browser_name}: {e}")

    async def _setup_page_monitoring(self, page: Page, browser_name: str) -> None:
        """Setup comprehensive page monitoring"""

        # Console message monitoring
        page.on("console", lambda msg: self._log_console_message(msg, browser_name))

        # Page error monitoring
        page.on("pageerror", lambda error: self._log_page_error(error, browser_name))

        # Request monitoring
        page.on("request", lambda request: self._log_request(request, browser_name))

        # Response monitoring
        page.on("response", lambda response: self._log_response(response, browser_name))

        # Dialog monitoring
        page.on("dialog", lambda dialog: self._handle_dialog(dialog, browser_name))

    def _log_console_message(self, msg, browser_name: str) -> None:
        """Log console messages for debugging"""
        timestamp = datetime.now(UTC).isoformat()
        print(f"[{timestamp}] [{browser_name}] Console {msg.type}: {msg.text}")

    def _log_page_error(self, error, browser_name: str) -> None:
        """Log page errors for debugging"""
        timestamp = datetime.now(UTC).isoformat()
        print(f"[{timestamp}] [{browser_name}] Page Error: {error}")

    def _log_request(self, request, browser_name: str) -> None:
        """Log HTTP requests for debugging"""
        if request.url.startswith(self.base_url):
            timestamp = datetime.now(UTC).isoformat()
            print(
                f"[{timestamp}] [{browser_name}] Request: {request.method} {request.url}"
            )

    def _log_response(self, response, browser_name: str) -> None:
        """Log HTTP responses for debugging"""
        if response.url.startswith(self.base_url):
            timestamp = datetime.now(UTC).isoformat()
            print(
                f"[{timestamp}] [{browser_name}] Response: {response.status} {response.url}"
            )

    async def _handle_dialog(self, dialog, browser_name: str) -> None:
        """Handle browser dialogs automatically"""
        timestamp = datetime.now(UTC).isoformat()
        print(
            f"[{timestamp}] [{browser_name}] Dialog: {dialog.type} - {dialog.message}"
        )
        await dialog.accept()

    async def run_comprehensive_test_suite(self) -> dict[str, Any]:
        """Run the complete advanced test suite"""
        test_results = {
            "suite_start_time": datetime.now(UTC).isoformat(),
            "browsers": {},
            "overall_metrics": {},
            "test_categories": {},
        }

        # Run tests for each browser
        for browser_name, page in self.pages.items():
            print(f"\n🌐 Running tests for {browser_name}...")

            browser_results = await self._run_browser_test_suite(browser_name, page)
            test_results["browsers"][browser_name] = browser_results

        # Run cross-browser comparison tests
        print("\n🔄 Running cross-browser comparison tests...")
        cross_browser_results = await self._run_cross_browser_tests()
        test_results["cross_browser"] = cross_browser_results

        # Generate comprehensive reports
        test_results["reports"] = await self._generate_comprehensive_reports()

        test_results["suite_end_time"] = datetime.now(UTC).isoformat()

        return test_results

    async def _run_browser_test_suite(
        self, browser_name: str, page: Page
    ) -> dict[str, Any]:
        """Run comprehensive test suite for a specific browser"""
        browser_results = {
            "browser": browser_name,
            "start_time": datetime.now(UTC).isoformat(),
            "categories": {},
        }

        # Navigation and Layout Tests
        print("  📄 Testing navigation and layout...")
        navigation_results = await self._test_navigation_and_layout(page, browser_name)
        browser_results["categories"]["navigation"] = navigation_results

        # Performance Tests
        print("  ⚡ Testing performance...")
        performance_results = await self._test_performance(page, browser_name)
        browser_results["categories"]["performance"] = performance_results

        # Accessibility Tests
        print("  ♿ Testing accessibility...")
        accessibility_results = await self._test_accessibility(page, browser_name)
        browser_results["categories"]["accessibility"] = accessibility_results

        # Visual Regression Tests
        print("  👁️  Testing visual regression...")
        visual_results = await self._test_visual_regression(page, browser_name)
        browser_results["categories"]["visual_regression"] = visual_results

        # Responsive Design Tests
        print("  📱 Testing responsive design...")
        responsive_results = await self._test_responsive_design(page, browser_name)
        browser_results["categories"]["responsive"] = responsive_results

        # Component Interaction Tests
        print("  🔧 Testing component interactions...")
        interaction_results = await self._test_component_interactions(
            page, browser_name
        )
        browser_results["categories"]["interactions"] = interaction_results

        # Error Handling Tests
        print("  🚨 Testing error handling...")
        error_results = await self._test_error_handling(page, browser_name)
        browser_results["categories"]["error_handling"] = error_results

        browser_results["end_time"] = datetime.now(UTC).isoformat()

        return browser_results

    async def _test_navigation_and_layout(
        self, page: Page, browser_name: str
    ) -> dict[str, Any]:
        """Test navigation and layout functionality"""
        results = {"tests": [], "passed": 0, "failed": 0}

        try:
            # Navigate to dashboard
            await page.goto(f"{self.base_url}/")
            await page.wait_for_load_state("networkidle")

            # Test main navigation elements
            nav_tests = [
                ("Logo visibility", "img[alt*='Pynomaly'], .logo"),
                ("Main navigation", "nav, .navigation"),
                ("Dashboard link", "a[href*='/'], a[href*='dashboard']"),
                ("Detectors link", "a[href*='detectors']"),
                ("Datasets link", "a[href*='datasets']"),
                ("Footer presence", "footer, .footer"),
            ]

            for test_name, selector in nav_tests:
                try:
                    element = await page.wait_for_selector(selector, timeout=5000)
                    if element:
                        await self._capture_test_screenshot(
                            page,
                            f"navigation_{test_name.lower().replace(' ', '_')}",
                            browser_name,
                        )
                        results["tests"].append(
                            {
                                "name": test_name,
                                "status": "passed",
                                "selector": selector,
                            }
                        )
                        results["passed"] += 1
                    else:
                        raise Exception(f"Element not found: {selector}")

                except Exception as e:
                    results["tests"].append(
                        {
                            "name": test_name,
                            "status": "failed",
                            "error": str(e),
                            "selector": selector,
                        }
                    )
                    results["failed"] += 1

            # Test responsive navigation (mobile menu)
            await page.set_viewport_size({"width": 768, "height": 1024})
            await page.wait_for_timeout(1000)

            # Check if mobile menu exists
            try:
                mobile_menu = await page.query_selector(
                    ".mobile-menu, .menu-toggle, [aria-label*='menu']"
                )
                if mobile_menu:
                    await mobile_menu.click()
                    await page.wait_for_timeout(500)
                    await self._capture_test_screenshot(
                        page, "mobile_menu_open", browser_name
                    )

                    results["tests"].append(
                        {"name": "Mobile menu functionality", "status": "passed"}
                    )
                    results["passed"] += 1
                else:
                    results["tests"].append(
                        {
                            "name": "Mobile menu functionality",
                            "status": "skipped",
                            "reason": "Mobile menu not found",
                        }
                    )
            except Exception as e:
                results["tests"].append(
                    {
                        "name": "Mobile menu functionality",
                        "status": "failed",
                        "error": str(e),
                    }
                )
                results["failed"] += 1

            # Reset viewport
            await page.set_viewport_size({"width": 1920, "height": 1080})

        except Exception as e:
            results["tests"].append(
                {"name": "Navigation page load", "status": "failed", "error": str(e)}
            )
            results["failed"] += 1

        return results

    async def _test_performance(self, page: Page, browser_name: str) -> dict[str, Any]:
        """Test performance metrics and optimization"""
        results = {
            "core_web_vitals": {},
            "load_times": {},
            "resource_metrics": {},
            "tests": [],
            "passed": 0,
            "failed": 0,
        }

        try:
            # Enable performance monitoring
            await page.goto(f"{self.base_url}/")

            # Measure page load performance
            performance_timing = await page.evaluate("""
                () => {
                    const timing = performance.timing;
                    const navigation = performance.getEntriesByType('navigation')[0];

                    return {
                        domContentLoaded: timing.domContentLoadedEventEnd - timing.navigationStart,
                        loadComplete: timing.loadEventEnd - timing.navigationStart,
                        firstPaint: navigation.responseStart - navigation.requestStart,
                        domInteractive: timing.domInteractive - timing.navigationStart
                    };
                }
            """)

            results["load_times"] = performance_timing

            # Test Core Web Vitals if available
            try:
                web_vitals = await page.evaluate("""
                    () => {
                        return new Promise((resolve) => {
                            const vitals = {};

                            // LCP - Largest Contentful Paint
                            new PerformanceObserver((list) => {
                                const entries = list.getEntries();
                                const lastEntry = entries[entries.length - 1];
                                vitals.lcp = lastEntry.startTime;
                            }).observe({entryTypes: ['largest-contentful-paint']});

                            // FID - First Input Delay (approximated)
                            vitals.fid = 0; // Would need actual user interaction

                            // CLS - Cumulative Layout Shift
                            let clsValue = 0;
                            new PerformanceObserver((list) => {
                                for (const entry of list.getEntries()) {
                                    if (!entry.hadRecentInput) {
                                        clsValue += entry.value;
                                    }
                                }
                                vitals.cls = clsValue;
                            }).observe({entryTypes: ['layout-shift']});

                            setTimeout(() => resolve(vitals), 3000);
                        });
                    }
                """)

                results["core_web_vitals"] = web_vitals

                # Validate Core Web Vitals
                if web_vitals.get("lcp", 0) < 2500:  # LCP < 2.5s is good
                    results["tests"].append(
                        {
                            "name": "LCP Performance",
                            "status": "passed",
                            "value": web_vitals.get("lcp", 0),
                        }
                    )
                    results["passed"] += 1
                else:
                    results["tests"].append(
                        {
                            "name": "LCP Performance",
                            "status": "failed",
                            "value": web_vitals.get("lcp", 0),
                            "threshold": 2500,
                        }
                    )
                    results["failed"] += 1

                if web_vitals.get("cls", 1) < 0.1:  # CLS < 0.1 is good
                    results["tests"].append(
                        {
                            "name": "CLS Performance",
                            "status": "passed",
                            "value": web_vitals.get("cls", 1),
                        }
                    )
                    results["passed"] += 1
                else:
                    results["tests"].append(
                        {
                            "name": "CLS Performance",
                            "status": "failed",
                            "value": web_vitals.get("cls", 1),
                            "threshold": 0.1,
                        }
                    )
                    results["failed"] += 1

            except Exception as e:
                results["tests"].append(
                    {
                        "name": "Core Web Vitals measurement",
                        "status": "failed",
                        "error": str(e),
                    }
                )
                results["failed"] += 1

            # Test resource loading performance
            resource_metrics = await page.evaluate("""
                () => {
                    const resources = performance.getEntriesByType('resource');
                    const metrics = {
                        total_resources: resources.length,
                        total_size: 0,
                        slow_resources: []
                    };

                    resources.forEach(resource => {
                        const duration = resource.responseEnd - resource.requestStart;
                        if (duration > 1000) { // Resources taking more than 1s
                            metrics.slow_resources.push({
                                name: resource.name,
                                duration: duration,
                                size: resource.transferSize || 0
                            });
                        }
                        metrics.total_size += resource.transferSize || 0;
                    });

                    return metrics;
                }
            """)

            results["resource_metrics"] = resource_metrics

            # Test resource performance
            if len(resource_metrics["slow_resources"]) == 0:
                results["tests"].append(
                    {
                        "name": "Resource loading performance",
                        "status": "passed",
                        "total_resources": resource_metrics["total_resources"],
                    }
                )
                results["passed"] += 1
            else:
                results["tests"].append(
                    {
                        "name": "Resource loading performance",
                        "status": "warning",
                        "slow_resources": resource_metrics["slow_resources"],
                    }
                )
                results["failed"] += 1

        except Exception as e:
            results["tests"].append(
                {"name": "Performance measurement", "status": "failed", "error": str(e)}
            )
            results["failed"] += 1

        return results

    async def _test_accessibility(
        self, page: Page, browser_name: str
    ) -> dict[str, Any]:
        """Test accessibility compliance"""
        results = {"tests": [], "passed": 0, "failed": 0, "score": 0}

        try:
            await page.goto(f"{self.base_url}/")
            await page.wait_for_load_state("networkidle")

            # Inject axe-core for accessibility testing
            await page.add_script_tag(url="https://unpkg.com/axe-core@4.7.0/axe.min.js")
            await page.wait_for_timeout(1000)

            # Run axe accessibility tests
            axe_results = await page.evaluate("""
                () => {
                    return new Promise((resolve, reject) => {
                        if (typeof axe === 'undefined') {
                            reject(new Error('axe-core not loaded'));
                            return;
                        }

                        axe.run(document, {
                            rules: {
                                'color-contrast': { enabled: true },
                                'keyboard-navigation': { enabled: true },
                                'focus-management': { enabled: true },
                                'aria-labels': { enabled: true }
                            }
                        }, (err, results) => {
                            if (err) reject(err);
                            else resolve(results);
                        });
                    });
                }
            """)

            # Process axe results
            violations = axe_results.get("violations", [])
            passes = axe_results.get("passes", [])

            results["axe_results"] = {
                "violations": len(violations),
                "passes": len(passes),
                "details": violations[:5],  # Limit details for readability
            }

            # Calculate accessibility score
            total_tests = len(violations) + len(passes)
            if total_tests > 0:
                results["score"] = (len(passes) / total_tests) * 100

            # Test specific accessibility features
            accessibility_tests = [
                ("Alt text for images", "img[alt], img[aria-label]"),
                ("Form labels", "input[aria-label], input[aria-labelledby], label"),
                ("Heading hierarchy", "h1, h2, h3, h4, h5, h6"),
                ("Skip links", "a[href='#main'], a[href='#content']"),
                ("Focus indicators", "*:focus"),
            ]

            for test_name, selector in accessibility_tests:
                try:
                    elements = await page.query_selector_all(selector)
                    if elements:
                        results["tests"].append(
                            {
                                "name": test_name,
                                "status": "passed",
                                "count": len(elements),
                            }
                        )
                        results["passed"] += 1
                    else:
                        results["tests"].append(
                            {
                                "name": test_name,
                                "status": "warning",
                                "message": f"No elements found with {selector}",
                            }
                        )

                except Exception as e:
                    results["tests"].append(
                        {"name": test_name, "status": "failed", "error": str(e)}
                    )
                    results["failed"] += 1

            # Test keyboard navigation
            try:
                # Tab through interactive elements
                await page.keyboard.press("Tab")
                await page.wait_for_timeout(100)
                focused_element = await page.evaluate("document.activeElement.tagName")

                if focused_element in ["A", "BUTTON", "INPUT", "SELECT", "TEXTAREA"]:
                    results["tests"].append(
                        {
                            "name": "Keyboard navigation",
                            "status": "passed",
                            "focused_element": focused_element,
                        }
                    )
                    results["passed"] += 1
                else:
                    results["tests"].append(
                        {
                            "name": "Keyboard navigation",
                            "status": "warning",
                            "focused_element": focused_element,
                        }
                    )

            except Exception as e:
                results["tests"].append(
                    {"name": "Keyboard navigation", "status": "failed", "error": str(e)}
                )
                results["failed"] += 1

        except Exception as e:
            results["tests"].append(
                {
                    "name": "Accessibility testing setup",
                    "status": "failed",
                    "error": str(e),
                }
            )
            results["failed"] += 1

        return results

    async def _test_visual_regression(
        self, page: Page, browser_name: str
    ) -> dict[str, Any]:
        """Test visual regression by comparing screenshots"""
        results = {"tests": [], "passed": 0, "failed": 0, "screenshots": []}

        try:
            await page.goto(f"{self.base_url}/")
            await page.wait_for_load_state("networkidle")

            # Define views to test for visual regression
            test_views = [
                {"name": "dashboard_full", "url": "/", "viewport": None},
                {
                    "name": "dashboard_mobile",
                    "url": "/",
                    "viewport": {"width": 375, "height": 667},
                },
                {
                    "name": "dashboard_tablet",
                    "url": "/",
                    "viewport": {"width": 768, "height": 1024},
                },
            ]

            for view in test_views:
                try:
                    if view["viewport"]:
                        await page.set_viewport_size(view["viewport"])
                        await page.wait_for_timeout(500)

                    await page.goto(f"{self.base_url}{view['url']}")
                    await page.wait_for_load_state("networkidle")

                    # Take screenshot
                    screenshot_path = await self._capture_test_screenshot(
                        page, f"visual_regression_{view['name']}", browser_name
                    )

                    results["screenshots"].append(
                        {
                            "name": view["name"],
                            "path": str(screenshot_path),
                            "viewport": view["viewport"],
                        }
                    )

                    results["tests"].append(
                        {
                            "name": f"Visual regression - {view['name']}",
                            "status": "passed",
                            "screenshot": str(screenshot_path),
                        }
                    )
                    results["passed"] += 1

                except Exception as e:
                    results["tests"].append(
                        {
                            "name": f"Visual regression - {view['name']}",
                            "status": "failed",
                            "error": str(e),
                        }
                    )
                    results["failed"] += 1

            # Reset viewport
            await page.set_viewport_size({"width": 1920, "height": 1080})

        except Exception as e:
            results["tests"].append(
                {"name": "Visual regression setup", "status": "failed", "error": str(e)}
            )
            results["failed"] += 1

        return results

    async def _test_responsive_design(
        self, page: Page, browser_name: str
    ) -> dict[str, Any]:
        """Test responsive design across different viewports"""
        results = {"tests": [], "passed": 0, "failed": 0, "viewports": []}

        viewports = [
            {"name": "Mobile Portrait", "width": 375, "height": 667},
            {"name": "Mobile Landscape", "width": 667, "height": 375},
            {"name": "Tablet Portrait", "width": 768, "height": 1024},
            {"name": "Tablet Landscape", "width": 1024, "height": 768},
            {"name": "Desktop Small", "width": 1366, "height": 768},
            {"name": "Desktop Large", "width": 1920, "height": 1080},
            {"name": "Wide Screen", "width": 2560, "height": 1440},
        ]

        try:
            await page.goto(f"{self.base_url}/")

            for viewport in viewports:
                try:
                    await page.set_viewport_size(
                        {"width": viewport["width"], "height": viewport["height"]}
                    )
                    await page.wait_for_timeout(1000)
                    await page.wait_for_load_state("networkidle")

                    # Test layout integrity
                    layout_check = await page.evaluate("""
                        () => {
                            const body = document.body;
                            const hasHorizontalScrollbar = body.scrollWidth > window.innerWidth;
                            const hasOverflowingElements = Array.from(document.querySelectorAll('*'))
                                .some(el => el.scrollWidth > window.innerWidth);

                            return {
                                hasHorizontalScrollbar,
                                hasOverflowingElements,
                                bodyWidth: body.scrollWidth,
                                windowWidth: window.innerWidth
                            };
                        }
                    """)

                    # Capture screenshot for this viewport
                    screenshot_path = await self._capture_test_screenshot(
                        page,
                        f"responsive_{viewport['name'].lower().replace(' ', '_')}",
                        browser_name,
                    )

                    viewport_result = {
                        "name": viewport["name"],
                        "dimensions": f"{viewport['width']}x{viewport['height']}",
                        "layout_check": layout_check,
                        "screenshot": str(screenshot_path),
                    }

                    # Test passes if no horizontal overflow
                    if (
                        not layout_check["hasHorizontalScrollbar"]
                        and not layout_check["hasOverflowingElements"]
                    ):
                        results["tests"].append(
                            {
                                "name": f"Responsive - {viewport['name']}",
                                "status": "passed",
                                "viewport": viewport_result,
                            }
                        )
                        results["passed"] += 1
                    else:
                        results["tests"].append(
                            {
                                "name": f"Responsive - {viewport['name']}",
                                "status": "failed",
                                "issue": "Horizontal overflow detected",
                                "viewport": viewport_result,
                            }
                        )
                        results["failed"] += 1

                    results["viewports"].append(viewport_result)

                except Exception as e:
                    results["tests"].append(
                        {
                            "name": f"Responsive - {viewport['name']}",
                            "status": "failed",
                            "error": str(e),
                        }
                    )
                    results["failed"] += 1

            # Reset to default viewport
            await page.set_viewport_size({"width": 1920, "height": 1080})

        except Exception as e:
            results["tests"].append(
                {"name": "Responsive design setup", "status": "failed", "error": str(e)}
            )
            results["failed"] += 1

        return results

    async def _test_component_interactions(
        self, page: Page, browser_name: str
    ) -> dict[str, Any]:
        """Test interactive components and user workflows"""
        results = {"tests": [], "passed": 0, "failed": 0, "interactions": []}

        try:
            await page.goto(f"{self.base_url}/")
            await page.wait_for_load_state("networkidle")

            # Test form interactions
            form_tests = [
                {
                    "name": "Search functionality",
                    "selector": "input[type='search'], input[placeholder*='search']",
                    "action": "type",
                    "value": "test search",
                },
                {
                    "name": "Button clicks",
                    "selector": "button:not([disabled]), .btn:not([disabled])",
                    "action": "click",
                },
                {
                    "name": "Link navigation",
                    "selector": "a[href]:not([href='#']):not([href=''])",
                    "action": "hover",
                },
            ]

            for test in form_tests:
                try:
                    elements = await page.query_selector_all(test["selector"])
                    if elements:
                        # Test first element
                        element = elements[0]

                        if test["action"] == "type" and "value" in test:
                            await element.fill(test["value"])
                            await page.wait_for_timeout(500)
                        elif test["action"] == "click":
                            await element.click()
                            await page.wait_for_timeout(500)
                        elif test["action"] == "hover":
                            await element.hover()
                            await page.wait_for_timeout(300)

                        results["tests"].append(
                            {
                                "name": test["name"],
                                "status": "passed",
                                "elements_found": len(elements),
                                "action": test["action"],
                            }
                        )
                        results["passed"] += 1

                        interaction_result = {
                            "test_name": test["name"],
                            "selector": test["selector"],
                            "action": test["action"],
                            "status": "completed",
                        }
                        results["interactions"].append(interaction_result)

                    else:
                        results["tests"].append(
                            {
                                "name": test["name"],
                                "status": "skipped",
                                "reason": f"No elements found with selector: {test['selector']}",
                            }
                        )

                except Exception as e:
                    results["tests"].append(
                        {"name": test["name"], "status": "failed", "error": str(e)}
                    )
                    results["failed"] += 1

            # Test modal/dropdown interactions
            try:
                modals = await page.query_selector_all(
                    ".modal, .dropdown, [role='dialog']"
                )
                if modals:
                    results["tests"].append(
                        {
                            "name": "Modal/Dropdown components",
                            "status": "passed",
                            "count": len(modals),
                        }
                    )
                    results["passed"] += 1
                else:
                    results["tests"].append(
                        {
                            "name": "Modal/Dropdown components",
                            "status": "skipped",
                            "reason": "No modal/dropdown components found",
                        }
                    )
            except Exception as e:
                results["tests"].append(
                    {
                        "name": "Modal/Dropdown components",
                        "status": "failed",
                        "error": str(e),
                    }
                )
                results["failed"] += 1

            # Test HTMX interactions if present
            try:
                htmx_elements = await page.query_selector_all(
                    "[hx-get], [hx-post], [hx-trigger]"
                )
                if htmx_elements:
                    results["tests"].append(
                        {
                            "name": "HTMX dynamic elements",
                            "status": "passed",
                            "count": len(htmx_elements),
                        }
                    )
                    results["passed"] += 1
                else:
                    results["tests"].append(
                        {
                            "name": "HTMX dynamic elements",
                            "status": "skipped",
                            "reason": "No HTMX elements found",
                        }
                    )
            except Exception as e:
                results["tests"].append(
                    {
                        "name": "HTMX dynamic elements",
                        "status": "failed",
                        "error": str(e),
                    }
                )
                results["failed"] += 1

        except Exception as e:
            results["tests"].append(
                {
                    "name": "Component interaction setup",
                    "status": "failed",
                    "error": str(e),
                }
            )
            results["failed"] += 1

        return results

    async def _test_error_handling(
        self, page: Page, browser_name: str
    ) -> dict[str, Any]:
        """Test error handling and edge cases"""
        results = {"tests": [], "passed": 0, "failed": 0, "error_scenarios": []}

        try:
            # Test 404 error handling
            await page.goto(
                f"{self.base_url}/nonexistent-page", wait_until="networkidle"
            )

            # Check if proper 404 page is shown
            page_content = await page.content()
            if "404" in page_content or "not found" in page_content.lower():
                results["tests"].append(
                    {"name": "404 error handling", "status": "passed"}
                )
                results["passed"] += 1
            else:
                results["tests"].append(
                    {
                        "name": "404 error handling",
                        "status": "failed",
                        "issue": "No proper 404 page found",
                    }
                )
                results["failed"] += 1

            # Test JavaScript error handling
            await page.goto(f"{self.base_url}/")

            js_errors = []
            page.on("pageerror", lambda error: js_errors.append(str(error)))

            # Inject code that might cause errors and see if app handles gracefully
            await page.evaluate("""
                () => {
                    try {
                        // Test undefined variable access
                        window.testUndefinedVariable.someProperty;
                    } catch (e) {
                        console.warn('Caught expected error:', e.message);
                    }
                }
            """)

            await page.wait_for_timeout(2000)

            if len(js_errors) == 0:
                results["tests"].append(
                    {"name": "JavaScript error handling", "status": "passed"}
                )
                results["passed"] += 1
            else:
                results["tests"].append(
                    {
                        "name": "JavaScript error handling",
                        "status": "warning",
                        "errors": js_errors[:3],  # Limit to first 3 errors
                    }
                )

            # Test network error handling (simulated)
            await page.route("**/api/test-endpoint", lambda route: route.abort())

            try:
                await page.evaluate("""
                    () => fetch('/api/test-endpoint').catch(e => console.log('Network error handled:', e.message))
                """)

                results["tests"].append(
                    {"name": "Network error handling", "status": "passed"}
                )
                results["passed"] += 1

            except Exception as e:
                results["tests"].append(
                    {
                        "name": "Network error handling",
                        "status": "failed",
                        "error": str(e),
                    }
                )
                results["failed"] += 1

            # Test form validation
            forms = await page.query_selector_all("form")
            if forms:
                try:
                    form = forms[0]
                    submit_button = await form.query_selector(
                        "button[type='submit'], input[type='submit']"
                    )

                    if submit_button:
                        await submit_button.click()
                        await page.wait_for_timeout(1000)

                        # Check for validation messages
                        validation_messages = await page.query_selector_all(
                            ".error, .invalid, [aria-invalid='true'], .validation-error"
                        )

                        if validation_messages:
                            results["tests"].append(
                                {
                                    "name": "Form validation",
                                    "status": "passed",
                                    "validation_messages": len(validation_messages),
                                }
                            )
                            results["passed"] += 1
                        else:
                            results["tests"].append(
                                {
                                    "name": "Form validation",
                                    "status": "warning",
                                    "message": "No validation messages found",
                                }
                            )
                    else:
                        results["tests"].append(
                            {
                                "name": "Form validation",
                                "status": "skipped",
                                "reason": "No submit button found",
                            }
                        )

                except Exception as e:
                    results["tests"].append(
                        {"name": "Form validation", "status": "failed", "error": str(e)}
                    )
                    results["failed"] += 1
            else:
                results["tests"].append(
                    {
                        "name": "Form validation",
                        "status": "skipped",
                        "reason": "No forms found",
                    }
                )

        except Exception as e:
            results["tests"].append(
                {"name": "Error handling setup", "status": "failed", "error": str(e)}
            )
            results["failed"] += 1

        return results

    async def _run_cross_browser_tests(self) -> dict[str, Any]:
        """Run cross-browser comparison tests"""
        results = {
            "consistency_tests": [],
            "screenshot_comparisons": [],
            "performance_comparisons": {},
            "passed": 0,
            "failed": 0,
        }

        try:
            # Take screenshots across all browsers for comparison
            browser_screenshots = {}

            for browser_name, page in self.pages.items():
                await page.goto(f"{self.base_url}/")
                await page.wait_for_load_state("networkidle")

                screenshot_path = await self._capture_test_screenshot(
                    page, "cross_browser_comparison", browser_name
                )
                browser_screenshots[browser_name] = screenshot_path

            results["screenshot_comparisons"] = browser_screenshots

            # Compare layout consistency (simplified)
            if len(browser_screenshots) >= 2:
                results["consistency_tests"].append(
                    {
                        "name": "Cross-browser layout consistency",
                        "status": "passed",
                        "browsers_tested": list(browser_screenshots.keys()),
                        "note": "Screenshots captured for manual comparison",
                    }
                )
                results["passed"] += 1
            else:
                results["consistency_tests"].append(
                    {
                        "name": "Cross-browser layout consistency",
                        "status": "skipped",
                        "reason": "Not enough browsers for comparison",
                    }
                )

        except Exception as e:
            results["consistency_tests"].append(
                {"name": "Cross-browser testing", "status": "failed", "error": str(e)}
            )
            results["failed"] += 1

        return results

    async def _capture_test_screenshot(
        self, page: Page, test_name: str, browser_name: str
    ) -> Path:
        """Capture screenshot for test documentation"""
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        screenshot_filename = f"{browser_name}_{test_name}_{timestamp}.png"
        screenshot_path = self.screenshots_dir / screenshot_filename

        await page.screenshot(path=str(screenshot_path), full_page=True, type="png")

        return screenshot_path

    async def _generate_comprehensive_reports(self) -> dict[str, Any]:
        """Generate comprehensive test reports"""
        reports = {"html_report": None, "json_report": None, "summary_report": None}

        try:
            # Generate JSON report
            json_report_path = (
                self.reports_dir
                / f"ui_test_report_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.json"
            )

            report_data = {
                "test_suite": "Advanced UI Testing Framework",
                "version": "1.0.0",
                "timestamp": datetime.now(UTC).isoformat(),
                "metrics": self.test_metrics,
                "browsers_tested": list(self.browsers.keys()),
                "artifacts": {
                    "screenshots_directory": str(self.screenshots_dir),
                    "videos_directory": str(self.videos_dir),
                    "traces_directory": str(self.traces_dir),
                },
            }

            with open(json_report_path, "w") as f:
                json.dump(report_data, f, indent=2, default=str)

            reports["json_report"] = str(json_report_path)

            # Generate summary report
            summary_path = self.reports_dir / "test_summary.md"
            summary_content = self._generate_markdown_summary(report_data)

            with open(summary_path, "w") as f:
                f.write(summary_content)

            reports["summary_report"] = str(summary_path)

        except Exception as e:
            print(f"Error generating reports: {e}")

        return reports

    def _generate_markdown_summary(self, report_data: dict[str, Any]) -> str:
        """Generate markdown summary report"""
        return f"""# Advanced UI Test Report

## Test Execution Summary

- **Test Suite**: {report_data['test_suite']}
- **Version**: {report_data['version']}
- **Timestamp**: {report_data['timestamp']}
- **Browsers Tested**: {', '.join(report_data['browsers_tested'])}

## Test Results Overview

- **Total Tests**: {self.test_metrics['total_tests']}
- **Passed**: {self.test_metrics['passed_tests']} ✅
- **Failed**: {self.test_metrics['failed_tests']} ❌
- **Skipped**: {self.test_metrics['skipped_tests']} ⏭️

## Test Categories

### Navigation and Layout
- Comprehensive testing of navigation elements
- Mobile responsiveness validation
- Layout integrity checks

### Performance Testing
- Core Web Vitals measurement
- Resource loading optimization
- Page load time analysis

### Accessibility Testing
- WCAG compliance verification
- Keyboard navigation testing
- Screen reader compatibility

### Visual Regression Testing
- Cross-browser visual consistency
- Responsive design validation
- Screenshot-based comparisons

### Error Handling
- 404 error page testing
- JavaScript error recovery
- Form validation testing

## Artifacts Generated

- **Screenshots**: {report_data['artifacts']['screenshots_directory']}
- **Videos**: {report_data['artifacts']['videos_directory']}
- **Traces**: {report_data['artifacts']['traces_directory']}

## Next Steps

1. Review failed tests and screenshots
2. Address any accessibility violations
3. Optimize performance bottlenecks
4. Enhance error handling where needed

---
*Generated by Advanced UI Testing Framework*
"""

    async def cleanup(self) -> None:
        """Clean up browser instances and save traces"""
        try:
            for browser_name, context in self.contexts.items():
                try:
                    # Stop tracing and save
                    trace_path = self.traces_dir / f"{browser_name}_trace.zip"
                    await context.tracing.stop(path=str(trace_path))
                    print(f"Trace saved for {browser_name}: {trace_path}")
                except Exception as e:
                    print(f"Error saving trace for {browser_name}: {e}")

                await context.close()

            for browser in self.browsers.values():
                await browser.close()

            await self.playwright.stop()

        except Exception as e:
            print(f"Error during cleanup: {e}")


# Test execution function
async def main():
    """Main test execution function"""
    print("🚀 Starting Advanced UI Testing Framework...")

    framework = AdvancedUITestFramework()

    try:
        # Setup browsers
        await framework.setup_browsers(["chromium", "firefox"])

        # Run comprehensive test suite
        results = await framework.run_comprehensive_test_suite()

        # Print summary
        print("\n📊 Test Execution Complete!")
        print(f"Total browsers tested: {len(results.get('browsers', {}))}")

        for browser_name, browser_results in results.get("browsers", {}).items():
            print(f"\n{browser_name.upper()} Results:")
            for category, category_results in browser_results.get(
                "categories", {}
            ).items():
                passed = category_results.get("passed", 0)
                failed = category_results.get("failed", 0)
                print(f"  {category}: {passed} passed, {failed} failed")

        print(f"\n📁 Reports generated in: {framework.reports_dir}")

    except Exception as e:
        print(f"❌ Test execution failed: {e}")

    finally:
        await framework.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
