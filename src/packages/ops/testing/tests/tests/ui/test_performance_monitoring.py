"""Performance Monitoring and Core Web Vitals Testing."""

import time
from typing import Any

from playwright.sync_api import Page


class TestPerformanceMonitoring:
    """Test suite for performance monitoring and Core Web Vitals."""

    def test_page_load_performance(self, page: Page):
        """Test page load performance metrics."""
        # Start performance monitoring
        page.goto("http://pynomaly-app:8000/web/", wait_until="networkidle")

        # Get performance metrics
        metrics = page.evaluate(
            """
            () => {
                const navigation = performance.getEntriesByType('navigation')[0];
                const paint = performance.getEntriesByType('paint');

                const metrics = {
                    // Core Web Vitals
                    domContentLoaded: navigation.domContentLoadedEventEnd - navigation.domContentLoadedEventStart,
                    loadComplete: navigation.loadEventEnd - navigation.loadEventStart,
                    firstPaint: 0,
                    firstContentfulPaint: 0,

                    // Detailed timing
                    dnsLookup: navigation.domainLookupEnd - navigation.domainLookupStart,
                    tcpConnection: navigation.connectEnd - navigation.connectStart,
                    serverResponse: navigation.responseEnd - navigation.requestStart,
                    domProcessing: navigation.domInteractive - navigation.responseEnd,
                    resourceLoad: navigation.loadEventStart - navigation.domContentLoadedEventEnd
                };

                // Paint timing
                paint.forEach(entry => {
                    if (entry.name === 'first-paint') {
                        metrics.firstPaint = entry.startTime;
                    } else if (entry.name === 'first-contentful-paint') {
                        metrics.firstContentfulPaint = entry.startTime;
                    }
                });

                return metrics;
            }
        """
        )

        # Performance assertions
        assert (
            metrics["domContentLoaded"] < 1000
        ), f"DOM content loaded too slow: {metrics['domContentLoaded']}ms"
        assert (
            metrics["loadComplete"] < 3000
        ), f"Page load too slow: {metrics['loadComplete']}ms"
        assert (
            metrics["firstContentfulPaint"] < 2000
        ), f"FCP too slow: {metrics['firstContentfulPaint']}ms"

        # Take screenshot with performance overlay
        page.screenshot(path="screenshots/performance_dashboard.png")

        return metrics

    def test_interactive_performance(self, dashboard_page):
        """Test interactive element performance."""
        dashboard_page.navigate()

        # Test navigation click performance
        start_time = time.time()

        nav_link = dashboard_page.page.locator("nav a[href='/web/detectors']")
        nav_link.click()
        dashboard_page.page.wait_for_load_state("networkidle")

        navigation_time = (time.time() - start_time) * 1000

        assert navigation_time < 1000, f"Navigation too slow: {navigation_time}ms"

        # Test form interaction performance
        dashboard_page.navigate_to("/detectors")

        form_input = dashboard_page.page.locator("input[name='name']")
        if form_input.count() > 0:
            start_time = time.time()
            form_input.fill("Performance Test Detector")
            input_response = (time.time() - start_time) * 1000

            assert (
                input_response < 100
            ), f"Form input response too slow: {input_response}ms"

    def test_chart_rendering_performance(self, visualizations_page):
        """Test chart rendering performance."""
        start_time = time.time()

        visualizations_page.navigate()
        charts_loaded = visualizations_page.wait_for_charts_to_load(timeout=10000)

        chart_load_time = (time.time() - start_time) * 1000

        if charts_loaded:
            assert (
                chart_load_time < 5000
            ), f"Chart rendering too slow: {chart_load_time}ms"

            # Test chart interaction performance
            charts = visualizations_page.get_available_charts()

            if charts:
                # Test hover performance on first chart
                first_chart = visualizations_page.page.locator("svg, canvas").first
                if first_chart.count() > 0:
                    start_time = time.time()
                    first_chart.hover()
                    hover_response = (time.time() - start_time) * 1000

                    assert (
                        hover_response < 200
                    ), f"Chart hover response too slow: {hover_response}ms"

    def test_htmx_performance(self, dashboard_page):
        """Test HTMX interaction performance."""
        dashboard_page.navigate()

        refresh_button = dashboard_page.page.locator(dashboard_page.REFRESH_BUTTON)

        if refresh_button.count() > 0:
            start_time = time.time()

            refresh_button.click()
            # Wait for HTMX to complete
            dashboard_page.page.wait_for_timeout(2000)

            htmx_response_time = (time.time() - start_time) * 1000

            assert (
                htmx_response_time < 2000
            ), f"HTMX response too slow: {htmx_response_time}ms"

    def test_memory_usage(self, page: Page):
        """Test memory usage and potential leaks."""
        page.goto("http://pynomaly-app:8000/web/")
        page.wait_for_load_state("networkidle")

        # Get initial memory usage
        initial_memory = page.evaluate(
            """
            () => {
                if (performance.memory) {
                    return {
                        usedJSHeapSize: performance.memory.usedJSHeapSize,
                        totalJSHeapSize: performance.memory.totalJSHeapSize,
                        jsHeapSizeLimit: performance.memory.jsHeapSizeLimit
                    };
                }
                return null;
            }
        """
        )

        if initial_memory:
            # Navigate through several pages
            pages = ["/detectors", "/datasets", "/detection", "/visualizations", "/"]

            for page_path in pages:
                page.goto(f"http://pynomaly-app:8000/web{page_path}")
                page.wait_for_load_state("networkidle")
                page.wait_for_timeout(1000)

            # Get final memory usage
            final_memory = page.evaluate(
                """
                () => {
                    if (performance.memory) {
                        return {
                            usedJSHeapSize: performance.memory.usedJSHeapSize,
                            totalJSHeapSize: performance.memory.totalJSHeapSize,
                            jsHeapSizeLimit: performance.memory.jsHeapSizeLimit
                        };
                    }
                    return null;
                }
            """
            )

            if final_memory:
                memory_increase = (
                    final_memory["usedJSHeapSize"] - initial_memory["usedJSHeapSize"]
                )
                memory_increase_mb = memory_increase / (1024 * 1024)

                # Memory increase should be reasonable (less than 20MB)
                assert (
                    memory_increase_mb < 20
                ), f"Excessive memory usage increase: {memory_increase_mb:.2f}MB"

    def test_resource_loading_efficiency(self, page: Page):
        """Test resource loading efficiency."""
        # Monitor network requests
        requests = []

        def handle_request(request):
            requests.append(
                {
                    "url": request.url,
                    "method": request.method,
                    "resource_type": request.resource_type,
                }
            )

        page.on("request", handle_request)

        page.goto("http://pynomaly-app:8000/web/")
        page.wait_for_load_state("networkidle")

        # Analyze requests
        total_requests = len(requests)
        js_requests = len([r for r in requests if r["resource_type"] == "script"])
        css_requests = len([r for r in requests if r["resource_type"] == "stylesheet"])
        len([r for r in requests if r["resource_type"] == "image"])

        # Performance assertions
        assert total_requests < 50, f"Too many requests: {total_requests}"
        assert js_requests < 10, f"Too many JS files: {js_requests}"
        assert css_requests < 5, f"Too many CSS files: {css_requests}"

        # Check for efficient resource loading
        external_cdn_requests = [
            r for r in requests if "cdn." in r["url"] or "unpkg.com" in r["url"]
        ]

        # Should use CDN for common libraries
        assert len(external_cdn_requests) > 0, "Should use CDN for external libraries"

    def test_core_web_vitals(self, page: Page):
        """Test Core Web Vitals metrics."""
        page.goto("http://pynomaly-app:8000/web/")

        # Wait for page to fully load
        page.wait_for_load_state("networkidle")
        page.wait_for_timeout(3000)  # Allow time for metrics to stabilize

        # Get Core Web Vitals
        vitals = page.evaluate(
            """
            () => new Promise((resolve) => {
                let vitals = {};

                // Largest Contentful Paint
                new PerformanceObserver((entryList) => {
                    const entries = entryList.getEntries();
                    const lastEntry = entries[entries.length - 1];
                    vitals.lcp = lastEntry.startTime;
                }).observe({entryTypes: ['largest-contentful-paint']});

                // First Input Delay (simulated)
                vitals.fid = 0; // Will be 0 in automated tests

                // Cumulative Layout Shift
                let clsValue = 0;
                new PerformanceObserver((entryList) => {
                    for (const entry of entryList.getEntries()) {
                        if (!entry.hadRecentInput) {
                            clsValue += entry.value;
                        }
                    }
                    vitals.cls = clsValue;
                }).observe({entryTypes: ['layout-shift']});

                // Give time for observers to collect data
                setTimeout(() => {
                    resolve(vitals);
                }, 2000);
            })
        """
        )

        # Core Web Vitals thresholds (Google recommendations)
        if "lcp" in vitals and vitals["lcp"] > 0:
            assert (
                vitals["lcp"] < 2500
            ), f"LCP too slow: {vitals['lcp']}ms (should be < 2.5s)"

        if "cls" in vitals:
            assert (
                vitals["cls"] < 0.1
            ), f"CLS too high: {vitals['cls']} (should be < 0.1)"

        # First Input Delay is hard to test in automation, but we can check for it
        # assert vitals.get("fid", 0) < 100, f"FID too slow: {vitals['fid']}ms"

    def generate_performance_report(self, page: Page) -> dict[str, Any]:
        """Generate comprehensive performance report."""
        page.goto("http://pynomaly-app:8000/web/")
        page.wait_for_load_state("networkidle")

        # Collect comprehensive performance data
        performance_data = page.evaluate(
            """
            () => {
                const navigation = performance.getEntriesByType('navigation')[0];
                const paint = performance.getEntriesByType('paint');
                const resources = performance.getEntriesByType('resource');

                // Navigation timing
                const timing = {
                    navigation: {
                        dnsLookup: navigation.domainLookupEnd - navigation.domainLookupStart,
                        tcpConnection: navigation.connectEnd - navigation.connectStart,
                        serverResponse: navigation.responseEnd - navigation.requestStart,
                        domProcessing: navigation.domInteractive - navigation.responseEnd,
                        resourceLoad: navigation.loadEventStart - navigation.domContentLoadedEventEnd,
                        totalLoad: navigation.loadEventEnd - navigation.navigationStart
                    },
                    paint: {},
                    resources: {
                        total: resources.length,
                        by_type: {}
                    }
                };

                // Paint timing
                paint.forEach(entry => {
                    timing.paint[entry.name] = entry.startTime;
                });

                // Resource analysis
                const resourceTypes = {};
                resources.forEach(resource => {
                    const type = resource.initiatorType || 'other';
                    if (!resourceTypes[type]) {
                        resourceTypes[type] = { count: 0, totalSize: 0, totalTime: 0 };
                    }
                    resourceTypes[type].count++;
                    resourceTypes[type].totalTime += resource.duration;
                    if (resource.transferSize) {
                        resourceTypes[type].totalSize += resource.transferSize;
                    }
                });

                timing.resources.by_type = resourceTypes;

                // Memory usage (if available)
                if (performance.memory) {
                    timing.memory = {
                        used: performance.memory.usedJSHeapSize,
                        total: performance.memory.totalJSHeapSize,
                        limit: performance.memory.jsHeapSizeLimit
                    };
                }

                return timing;
            }
        """
        )

        return performance_data
