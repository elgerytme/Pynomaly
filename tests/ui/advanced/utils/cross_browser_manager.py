"""
Cross Browser Manager for Advanced UI Testing

Provides comprehensive cross-browser testing capabilities including:
- Multi-browser session management
- Browser compatibility validation
- Feature detection and polyfill testing
- Performance comparison across browsers
- Responsive design validation
"""

from datetime import UTC, datetime
from typing import Any

from playwright.async_api import Browser, BrowserContext, Page, async_playwright


class CrossBrowserManager:
    """
    Manages cross-browser testing scenarios and comparisons
    """

    def __init__(self):
        self.playwright = None
        self.browsers: dict[str, Browser] = {}
        self.contexts: dict[str, BrowserContext] = {}
        self.pages: dict[str, Page] = {}

        self.browser_configs = {
            "chromium": {
                "name": "Chromium",
                "engine": "Blink",
                "capabilities": ["modern_js", "css_grid", "webgl", "service_workers"],
                "mobile_simulation": True,
                "dev_tools": True,
            },
            "firefox": {
                "name": "Firefox",
                "engine": "Gecko",
                "capabilities": ["modern_js", "css_grid", "webgl", "service_workers"],
                "mobile_simulation": True,
                "dev_tools": True,
            },
            "webkit": {
                "name": "WebKit/Safari",
                "engine": "WebKit",
                "capabilities": ["modern_js", "css_grid", "webgl", "service_workers"],
                "mobile_simulation": True,
                "dev_tools": False,
            },
        }

        self.test_results: dict[str, Any] = {}
        self.comparison_results: dict[str, Any] = {}

    async def initialize_browsers(self, browsers: list[str] = None) -> dict[str, bool]:
        """Initialize specified browsers for testing"""
        if browsers is None:
            browsers = ["chromium", "firefox", "webkit"]

        self.playwright = await async_playwright().start()
        initialization_results = {}

        for browser_name in browsers:
            try:
                success = await self._initialize_single_browser(browser_name)
                initialization_results[browser_name] = success
            except Exception as e:
                print(f"Failed to initialize {browser_name}: {e}")
                initialization_results[browser_name] = False

        return initialization_results

    async def _initialize_single_browser(self, browser_name: str) -> bool:
        """Initialize a single browser instance"""
        try:
            browser_type = getattr(self.playwright, browser_name)
            browser = await browser_type.launch(
                headless=True,
                args=[
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-web-security",
                    "--enable-automation",
                    "--disable-background-timer-throttling",
                    "--disable-backgrounding-occluded-windows",
                    "--disable-renderer-backgrounding",
                ],
            )

            # Create context with realistic settings
            context = await browser.new_context(
                viewport={"width": 1920, "height": 1080},
                user_agent=f"Mozilla/5.0 (compatible; PynotalyUITest/{browser_name}/1.0)",
                permissions=["notifications"],
                geolocation={
                    "latitude": 37.7749,
                    "longitude": -122.4194,
                },  # San Francisco
                timezone_id="America/Los_Angeles",
                locale="en-US",
            )

            # Create page
            page = await context.new_page()

            # Store instances
            self.browsers[browser_name] = browser
            self.contexts[browser_name] = context
            self.pages[browser_name] = page

            return True

        except Exception as e:
            print(f"Error initializing {browser_name}: {e}")
            return False

    async def run_cross_browser_compatibility_tests(
        self, base_url: str
    ) -> dict[str, Any]:
        """Run comprehensive cross-browser compatibility tests"""
        results = {
            "timestamp": datetime.now(UTC).isoformat(),
            "browsers_tested": list(self.pages.keys()),
            "test_categories": {
                "basic_functionality": {},
                "feature_support": {},
                "performance_comparison": {},
                "visual_consistency": {},
                "responsive_behavior": {},
            },
            "overall_compatibility": {},
        }

        # Test basic functionality across browsers
        print("🌐 Testing basic functionality across browsers...")
        basic_results = await self._test_basic_functionality(base_url)
        results["test_categories"]["basic_functionality"] = basic_results

        # Test feature support
        print("🔧 Testing feature support across browsers...")
        feature_results = await self._test_feature_support()
        results["test_categories"]["feature_support"] = feature_results

        # Compare performance
        print("⚡ Comparing performance across browsers...")
        performance_results = await self._compare_performance(base_url)
        results["test_categories"]["performance_comparison"] = performance_results

        # Test visual consistency
        print("👁️  Testing visual consistency...")
        visual_results = await self._test_visual_consistency(base_url)
        results["test_categories"]["visual_consistency"] = visual_results

        # Test responsive behavior
        print("📱 Testing responsive behavior...")
        responsive_results = await self._test_responsive_behavior(base_url)
        results["test_categories"]["responsive_behavior"] = responsive_results

        # Calculate overall compatibility score
        results["overall_compatibility"] = self._calculate_compatibility_score(
            results["test_categories"]
        )

        self.test_results = results
        return results

    async def _test_basic_functionality(self, base_url: str) -> dict[str, Any]:
        """Test basic functionality across all browsers"""
        results = {}

        for browser_name, page in self.pages.items():
            browser_results = {
                "page_load": False,
                "navigation": False,
                "forms": False,
                "javascript": False,
                "errors": [],
            }

            try:
                # Test page load
                response = await page.goto(base_url, wait_until="networkidle")
                if response and response.status < 400:
                    browser_results["page_load"] = True

                # Test navigation
                nav_links = await page.query_selector_all("a[href], button")
                if nav_links:
                    browser_results["navigation"] = True

                # Test form functionality
                forms = await page.query_selector_all("form")
                if forms:
                    browser_results["forms"] = True

                # Test JavaScript execution
                js_result = await page.evaluate("() => typeof window !== 'undefined'")
                if js_result:
                    browser_results["javascript"] = True

            except Exception as e:
                browser_results["errors"].append(str(e))

            results[browser_name] = browser_results

        return results

    async def _test_feature_support(self) -> dict[str, Any]:
        """Test modern web features support across browsers"""
        results = {}

        feature_tests = {
            "es6_support": "() => { try { eval('const x = () => 1'); return true; } catch(e) { return false; } }",
            "fetch_api": "() => typeof fetch !== 'undefined'",
            "local_storage": "() => typeof localStorage !== 'undefined'",
            "session_storage": "() => typeof sessionStorage !== 'undefined'",
            "geolocation": "() => typeof navigator.geolocation !== 'undefined'",
            "webgl": "() => { const canvas = document.createElement('canvas'); return !!canvas.getContext('webgl'); }",
            "websockets": "() => typeof WebSocket !== 'undefined'",
            "service_workers": "() => 'serviceWorker' in navigator",
            "push_notifications": "() => 'Notification' in window",
            "web_workers": "() => typeof Worker !== 'undefined'",
            "css_grid": "() => CSS.supports('display', 'grid')",
            "css_flexbox": "() => CSS.supports('display', 'flex')",
            "css_variables": "() => CSS.supports('color', 'var(--test)')",
            "intersection_observer": "() => 'IntersectionObserver' in window",
            "resize_observer": "() => 'ResizeObserver' in window",
        }

        for browser_name, page in self.pages.items():
            browser_features = {}

            for feature_name, test_code in feature_tests.items():
                try:
                    supported = await page.evaluate(test_code)
                    browser_features[feature_name] = supported
                except Exception as e:
                    browser_features[feature_name] = False
                    print(f"Feature test {feature_name} failed in {browser_name}: {e}")

            results[browser_name] = browser_features

        return results

    async def _compare_performance(self, base_url: str) -> dict[str, Any]:
        """Compare performance metrics across browsers"""
        results = {}

        for browser_name, page in self.pages.items():
            performance_metrics = {
                "load_time": 0,
                "dom_content_loaded": 0,
                "first_paint": 0,
                "memory_usage": 0,
                "cpu_usage": 0,
            }

            try:
                # Navigate and measure timing
                start_time = datetime.now(UTC)
                await page.goto(base_url, wait_until="networkidle")
                end_time = datetime.now(UTC)

                performance_metrics["load_time"] = (
                    end_time - start_time
                ).total_seconds() * 1000

                # Get browser performance metrics
                browser_metrics = await page.evaluate("""
                    () => {
                        const timing = performance.timing;
                        const navigation = performance.getEntriesByType('navigation')[0];

                        return {
                            domContentLoaded: timing.domContentLoadedEventEnd - timing.navigationStart,
                            firstPaint: navigation ? navigation.responseStart - navigation.requestStart : 0,
                            resourceCount: performance.getEntriesByType('resource').length
                        };
                    }
                """)

                performance_metrics.update(browser_metrics)

                # Get memory usage (if available)
                try:
                    memory_info = await page.evaluate("() => performance.memory || {}")
                    if memory_info:
                        performance_metrics["memory_usage"] = memory_info.get(
                            "usedJSHeapSize", 0
                        )
                except Exception:
                    pass  # Memory API not available

            except Exception as e:
                performance_metrics["error"] = str(e)

            results[browser_name] = performance_metrics

        return results

    async def _test_visual_consistency(self, base_url: str) -> dict[str, Any]:
        """Test visual consistency across browsers"""
        results = {
            "screenshots": {},
            "layout_differences": [],
            "font_rendering": {},
            "color_differences": [],
        }

        # Take screenshots in each browser
        for browser_name, page in self.pages.items():
            try:
                await page.goto(base_url, wait_until="networkidle")

                # Take full page screenshot
                screenshot_path = f"test_artifacts/cross_browser_{browser_name}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.png"
                await page.screenshot(path=screenshot_path, full_page=True)
                results["screenshots"][browser_name] = screenshot_path

                # Get layout information
                layout_info = await page.evaluate("""
                    () => {
                        const body = document.body;
                        const main = document.querySelector('main, .main, #main') || body;

                        return {
                            bodyWidth: body.offsetWidth,
                            bodyHeight: body.offsetHeight,
                            mainWidth: main.offsetWidth,
                            mainHeight: main.offsetHeight,
                            scrollHeight: body.scrollHeight,
                            scrollWidth: body.scrollWidth
                        };
                    }
                """)

                results["font_rendering"][browser_name] = layout_info

            except Exception as e:
                results["screenshots"][browser_name] = f"Error: {e}"

        # Compare layouts (simplified)
        layout_values = list(results["font_rendering"].values())
        if len(layout_values) > 1:
            first_layout = layout_values[0]
            for i, layout in enumerate(layout_values[1:], 1):
                browser_name = list(results["font_rendering"].keys())[i]
                differences = []

                for key, value in layout.items():
                    if isinstance(value, (int, float)) and isinstance(
                        first_layout.get(key), (int, float)
                    ):
                        diff_percentage = (
                            abs(value - first_layout[key])
                            / max(first_layout[key], 1)
                            * 100
                        )
                        if diff_percentage > 5:  # More than 5% difference
                            differences.append(
                                {
                                    "property": key,
                                    "difference_percentage": diff_percentage,
                                    "values": {
                                        "reference": first_layout[key],
                                        "current": value,
                                    },
                                }
                            )

                if differences:
                    results["layout_differences"].append(
                        {"browser": browser_name, "differences": differences}
                    )

        return results

    async def _test_responsive_behavior(self, base_url: str) -> dict[str, Any]:
        """Test responsive behavior across browsers"""
        results = {}

        viewports = [
            {"name": "mobile_portrait", "width": 375, "height": 667},
            {"name": "mobile_landscape", "width": 667, "height": 375},
            {"name": "tablet", "width": 768, "height": 1024},
            {"name": "desktop", "width": 1920, "height": 1080},
        ]

        for browser_name, page in self.pages.items():
            browser_responsive_results = {}

            for viewport in viewports:
                try:
                    # Set viewport
                    await page.set_viewport_size(
                        {"width": viewport["width"], "height": viewport["height"]}
                    )
                    await page.goto(base_url, wait_until="networkidle")
                    await page.wait_for_timeout(1000)  # Allow layout to settle

                    # Test responsive behavior
                    responsive_check = await page.evaluate("""
                        () => {
                            const body = document.body;
                            const hasHorizontalScroll = body.scrollWidth > window.innerWidth;
                            const hasVisibleContent = body.scrollHeight > 100;

                            // Check for responsive elements
                            const mediaQueries = Array.from(document.styleSheets)
                                .flatMap(sheet => {
                                    try {
                                        return Array.from(sheet.cssRules || []);
                                    } catch(e) {
                                        return [];
                                    }
                                })
                                .filter(rule => rule.type === CSSRule.MEDIA_RULE)
                                .length;

                            return {
                                hasHorizontalScroll,
                                hasVisibleContent,
                                mediaQueriesCount: mediaQueries,
                                viewportWidth: window.innerWidth,
                                viewportHeight: window.innerHeight,
                                bodyWidth: body.offsetWidth,
                                bodyHeight: body.offsetHeight
                            };
                        }
                    """)

                    browser_responsive_results[viewport["name"]] = {
                        "viewport": viewport,
                        "check_results": responsive_check,
                        "passes_responsive_test": not responsive_check[
                            "hasHorizontalScroll"
                        ]
                        and responsive_check["hasVisibleContent"],
                    }

                except Exception as e:
                    browser_responsive_results[viewport["name"]] = {
                        "viewport": viewport,
                        "error": str(e),
                        "passes_responsive_test": False,
                    }

            results[browser_name] = browser_responsive_results

        return results

    def _calculate_compatibility_score(
        self, test_categories: dict[str, Any]
    ) -> dict[str, Any]:
        """Calculate overall compatibility score across browsers"""
        browser_scores = {}

        for browser_name in self.pages.keys():
            scores = []

            # Basic functionality score
            basic = test_categories.get("basic_functionality", {}).get(browser_name, {})
            basic_score = (
                sum(
                    [
                        basic.get("page_load", False),
                        basic.get("navigation", False),
                        basic.get("forms", False),
                        basic.get("javascript", False),
                    ]
                )
                / 4
                * 100
            )
            scores.append(basic_score)

            # Feature support score
            features = test_categories.get("feature_support", {}).get(browser_name, {})
            if features:
                feature_score = sum(features.values()) / len(features) * 100
                scores.append(feature_score)

            # Responsive score
            responsive = test_categories.get("responsive_behavior", {}).get(
                browser_name, {}
            )
            if responsive:
                responsive_tests = [
                    result.get("passes_responsive_test", False)
                    for result in responsive.values()
                    if isinstance(result, dict)
                ]
                if responsive_tests:
                    responsive_score = (
                        sum(responsive_tests) / len(responsive_tests) * 100
                    )
                    scores.append(responsive_score)

            # Calculate overall score
            browser_scores[browser_name] = {
                "overall_score": sum(scores) / len(scores) if scores else 0,
                "category_scores": {
                    "basic_functionality": basic_score,
                    "feature_support": feature_score if features else 0,
                    "responsive_design": responsive_score
                    if "responsive_score" in locals()
                    else 0,
                },
            }

        # Calculate cross-browser consistency
        overall_scores = [
            browser_scores[browser]["overall_score"] for browser in browser_scores
        ]
        consistency_score = (
            100 - (max(overall_scores) - min(overall_scores)) if overall_scores else 0
        )

        return {
            "browser_scores": browser_scores,
            "consistency_score": consistency_score,
            "average_compatibility": sum(overall_scores) / len(overall_scores)
            if overall_scores
            else 0,
            "recommendation": self._generate_compatibility_recommendation(
                browser_scores, consistency_score
            ),
        }

    def _generate_compatibility_recommendation(
        self, browser_scores: dict[str, Any], consistency_score: float
    ) -> str:
        """Generate compatibility recommendations"""
        avg_score = sum(
            browser["overall_score"] for browser in browser_scores.values()
        ) / len(browser_scores)

        if avg_score >= 90 and consistency_score >= 90:
            return "Excellent cross-browser compatibility. No major issues detected."
        elif avg_score >= 80 and consistency_score >= 80:
            return "Good cross-browser compatibility with minor differences. Consider testing edge cases."
        elif avg_score >= 70 or consistency_score >= 70:
            return "Moderate compatibility issues. Some browsers may need specific attention."
        else:
            return "Significant compatibility issues detected. Major browser-specific fixes required."

    async def run_feature_detection_tests(self) -> dict[str, Any]:
        """Run comprehensive feature detection tests"""
        results = {
            "html5_features": {},
            "css3_features": {},
            "javascript_features": {},
            "api_features": {},
            "performance_features": {},
        }

        # HTML5 features
        html5_tests = {
            "canvas": "() => !!document.createElement('canvas').getContext",
            "video": "() => !!document.createElement('video').canPlayType",
            "audio": "() => !!document.createElement('audio').canPlayType",
            "svg": "() => !!document.createElementNS && !!document.createElementNS('http://www.w3.org/2000/svg', 'svg').createSVGRect",
            "webgl": "() => { const canvas = document.createElement('canvas'); return !!canvas.getContext('webgl') || !!canvas.getContext('experimental-webgl'); }",
            "form_validation": "() => 'checkValidity' in document.createElement('input')",
        }

        # CSS3 features
        css3_tests = {
            "flexbox": "() => CSS.supports('display', 'flex')",
            "grid": "() => CSS.supports('display', 'grid')",
            "transforms": "() => CSS.supports('transform', 'scale(1)')",
            "transitions": "() => CSS.supports('transition', 'all 1s')",
            "animations": "() => CSS.supports('animation', 'name 1s')",
            "custom_properties": "() => CSS.supports('color', 'var(--test)')",
            "backdrop_filter": "() => CSS.supports('backdrop-filter', 'blur(5px)')",
        }

        # JavaScript features
        js_tests = {
            "es6_classes": "() => { try { eval('class Test {}'); return true; } catch(e) { return false; } }",
            "arrow_functions": "() => { try { eval('() => {}'); return true; } catch(e) { return false; } }",
            "destructuring": "() => { try { eval('const {a} = {a:1}'); return true; } catch(e) { return false; } }",
            "template_literals": "() => { try { eval('`test`'); return true; } catch(e) { return false; } }",
            "async_await": "() => { try { eval('async function test() { await 1; }'); return true; } catch(e) { return false; } }",
            "modules": "() => 'import' in document.createElement('script')",
            "proxy": "() => typeof Proxy !== 'undefined'",
            "symbols": "() => typeof Symbol !== 'undefined'",
        }

        # API features
        api_tests = {
            "fetch": "() => typeof fetch !== 'undefined'",
            "websockets": "() => typeof WebSocket !== 'undefined'",
            "sse": "() => typeof EventSource !== 'undefined'",
            "webrtc": "() => typeof RTCPeerConnection !== 'undefined'",
            "geolocation": "() => typeof navigator.geolocation !== 'undefined'",
            "notifications": "() => 'Notification' in window",
            "service_workers": "() => 'serviceWorker' in navigator",
            "payment_request": "() => typeof PaymentRequest !== 'undefined'",
            "web_workers": "() => typeof Worker !== 'undefined'",
            "shared_workers": "() => typeof SharedWorker !== 'undefined'",
        }

        # Performance features
        performance_tests = {
            "performance_api": "() => typeof performance !== 'undefined'",
            "performance_observer": "() => typeof PerformanceObserver !== 'undefined'",
            "intersection_observer": "() => typeof IntersectionObserver !== 'undefined'",
            "resize_observer": "() => typeof ResizeObserver !== 'undefined'",
            "mutation_observer": "() => typeof MutationObserver !== 'undefined'",
            "requestAnimationFrame": "() => typeof requestAnimationFrame !== 'undefined'",
            "requestIdleCallback": "() => typeof requestIdleCallback !== 'undefined'",
        }

        test_suites = {
            "html5_features": html5_tests,
            "css3_features": css3_tests,
            "javascript_features": js_tests,
            "api_features": api_tests,
            "performance_features": performance_tests,
        }

        for suite_name, tests in test_suites.items():
            results[suite_name] = {}

            for browser_name, page in self.pages.items():
                browser_results = {}

                for test_name, test_code in tests.items():
                    try:
                        supported = await page.evaluate(test_code)
                        browser_results[test_name] = supported
                    except Exception as e:
                        browser_results[test_name] = False
                        print(f"Feature test {test_name} failed in {browser_name}: {e}")

                results[suite_name][browser_name] = browser_results

        return results

    async def cleanup(self) -> None:
        """Clean up browser instances"""
        try:
            for context in self.contexts.values():
                await context.close()

            for browser in self.browsers.values():
                await browser.close()

            if self.playwright:
                await self.playwright.stop()

        except Exception as e:
            print(f"Error during cleanup: {e}")

    def generate_compatibility_report(self) -> str:
        """Generate detailed compatibility report"""
        if not self.test_results:
            return "No test results available. Run tests first."

        report = f"""# Cross-Browser Compatibility Report

## Test Summary
- **Timestamp**: {self.test_results['timestamp']}
- **Browsers Tested**: {', '.join(self.test_results['browsers_tested'])}
- **Overall Compatibility**: {self.test_results['overall_compatibility']['average_compatibility']:.1f}%
- **Consistency Score**: {self.test_results['overall_compatibility']['consistency_score']:.1f}%

## Browser Scores
"""

        for browser, scores in self.test_results["overall_compatibility"][
            "browser_scores"
        ].items():
            report += f"\n### {browser.title()}\n"
            report += f"- **Overall Score**: {scores['overall_score']:.1f}%\n"
            report += f"- **Basic Functionality**: {scores['category_scores']['basic_functionality']:.1f}%\n"
            report += f"- **Feature Support**: {scores['category_scores']['feature_support']:.1f}%\n"
            report += f"- **Responsive Design**: {scores['category_scores']['responsive_design']:.1f}%\n"

        report += f"""
## Recommendation
{self.test_results['overall_compatibility']['recommendation']}

## Detailed Results

### Visual Consistency
"""

        visual_results = self.test_results["test_categories"]["visual_consistency"]
        if visual_results["layout_differences"]:
            report += "**Layout Differences Detected:**\n"
            for diff in visual_results["layout_differences"]:
                report += f"- {diff['browser']}: {len(diff['differences'])} differences found\n"
        else:
            report += "No significant layout differences detected.\n"

        report += """
### Performance Comparison
"""

        performance = self.test_results["test_categories"]["performance_comparison"]
        for browser, metrics in performance.items():
            report += (
                f"- **{browser}**: Load time {metrics.get('load_time', 0):.0f}ms\n"
            )

        return report
