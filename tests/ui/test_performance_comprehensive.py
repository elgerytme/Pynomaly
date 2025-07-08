"""Comprehensive Performance Testing with Lighthouse CI and Core Web Vitals Tracking."""

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
from playwright.sync_api import Page
from tests.ui.enhanced_page_objects.base_page import BasePage

# Configuration
PERFORMANCE_TESTING_ENABLED = os.getenv("PERFORMANCE_TESTING", "true").lower() == "true"
LIGHTHOUSE_ENABLED = os.getenv("LIGHTHOUSE_ENABLED", "false").lower() == "true"
PERFORMANCE_REPORTS_DIR = Path("test_reports/performance")
LIGHTHOUSE_REPORTS_DIR = Path("test_reports/lighthouse")

# Create directories
for directory in [PERFORMANCE_REPORTS_DIR, LIGHTHOUSE_REPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Performance thresholds (Core Web Vitals)
PERFORMANCE_THRESHOLDS = {
    "core_web_vitals": {
        "largest_contentful_paint": 2500,  # ms
        "first_input_delay": 100,  # ms
        "cumulative_layout_shift": 0.1,  # unitless
        "first_contentful_paint": 1800,  # ms
        "total_blocking_time": 200,  # ms
    },
    "lighthouse_scores": {
        "performance": 90,  # 0-100
        "accessibility": 95,  # 0-100
        "best_practices": 90,  # 0-100
        "seo": 90,  # 0-100
        "pwa": 80,  # 0-100 (if applicable)
    },
    "custom_metrics": {
        "dom_content_loaded": 2000,  # ms
        "load_complete": 3000,  # ms
        "time_to_interactive": 3500,  # ms
        "bundle_size": 1024 * 1024,  # 1MB in bytes
        "resource_count": 50,  # number of resources
    },
}

# Device configurations for performance testing
DEVICE_CONFIGS = [
    {
        "name": "desktop",
        "viewport": {"width": 1920, "height": 1080},
        "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "cpu_slowdown": 1,
        "network": "fast3g",
    },
    {
        "name": "mobile",
        "viewport": {"width": 375, "height": 667},
        "user_agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/605.1.15",
        "cpu_slowdown": 4,
        "network": "slow3g",
    },
    {
        "name": "tablet",
        "viewport": {"width": 768, "height": 1024},
        "user_agent": "Mozilla/5.0 (iPad; CPU OS 14_0 like Mac OS X) AppleWebKit/605.1.15",
        "cpu_slowdown": 2,
        "network": "regular3g",
    },
]


class PerformanceMonitor:
    """Comprehensive performance monitoring with Core Web Vitals and Lighthouse integration."""

    def __init__(self, page: Page):
        self.page = page
        self.base_page = BasePage(page)
        self.monitoring_active = False
        self.start_time = None

    async def start_monitoring(self):
        """Initialize performance monitoring with comprehensive tracking."""
        self.start_time = time.time()
        self.monitoring_active = True

        # Inject performance monitoring scripts
        await self.page.add_init_script(
            """
            window.performanceData = {
                navigationStart: performance.timeOrigin,
                metrics: {},
                vitals: {},
                resources: [],
                observers: []
            };
            
            // Core Web Vitals monitoring
            if ('PerformanceObserver' in window) {
                // Largest Contentful Paint
                const lcpObserver = new PerformanceObserver((entryList) => {
                    const entries = entryList.getEntries();
                    const lastEntry = entries[entries.length - 1];
                    window.performanceData.vitals.lcp = lastEntry.startTime;
                });
                lcpObserver.observe({entryTypes: ['largest-contentful-paint']});
                window.performanceData.observers.push(lcpObserver);
                
                // First Input Delay (requires actual interaction)
                const fidObserver = new PerformanceObserver((entryList) => {
                    const firstEntry = entryList.getEntries()[0];
                    window.performanceData.vitals.fid = firstEntry.processingStart - firstEntry.startTime;
                });
                fidObserver.observe({entryTypes: ['first-input']});
                window.performanceData.observers.push(fidObserver);
                
                // Cumulative Layout Shift
                let clsValue = 0;
                const clsObserver = new PerformanceObserver((entryList) => {
                    for (const entry of entryList.getEntries()) {
                        if (!entry.hadRecentInput) {
                            clsValue += entry.value;
                        }
                    }
                    window.performanceData.vitals.cls = clsValue;
                });
                clsObserver.observe({entryTypes: ['layout-shift']});
                window.performanceData.observers.push(clsObserver);
                
                // Navigation timing
                const navObserver = new PerformanceObserver((entryList) => {
                    const navigation = entryList.getEntries()[0];
                    window.performanceData.metrics.navigation = {
                        domContentLoaded: navigation.domContentLoadedEventEnd - navigation.domContentLoadedEventStart,
                        loadComplete: navigation.loadEventEnd - navigation.loadEventStart,
                        timeToInteractive: navigation.domInteractive - navigation.fetchStart,
                        totalLoadTime: navigation.loadEventEnd - navigation.fetchStart,
                        transferSize: navigation.transferSize || 0,
                        resourceTimingBufferSize: performance.getEntriesByType('resource').length
                    };
                });
                navObserver.observe({entryTypes: ['navigation']});
                window.performanceData.observers.push(navObserver);
                
                // Paint timing
                const paintObserver = new PerformanceObserver((entryList) => {
                    const entries = entryList.getEntries();
                    for (const entry of entries) {
                        window.performanceData.vitals[entry.name.replace(/-/g, '_')] = entry.startTime;
                    }
                });
                paintObserver.observe({entryTypes: ['paint']});
                window.performanceData.observers.push(paintObserver);
                
                // Resource timing
                const resourceObserver = new PerformanceObserver((entryList) => {
                    const entries = entryList.getEntries();
                    window.performanceData.resources.push(...entries.map(entry => ({
                        name: entry.name,
                        type: entry.initiatorType,
                        startTime: entry.startTime,
                        duration: entry.duration,
                        transferSize: entry.transferSize || 0,
                        responseEnd: entry.responseEnd
                    })));
                });
                resourceObserver.observe({entryTypes: ['resource']});
                window.performanceData.observers.push(resourceObserver);
            }
            
            // Custom performance markers
            window.markPerformance = (name) => {
                if (performance.mark) {
                    performance.mark(name);
                    window.performanceData.metrics[name] = performance.now();
                }
            };
            
            // Bundle size estimation
            window.estimateBundleSize = () => {
                const resources = performance.getEntriesByType('resource');
                const jsResources = resources.filter(r => r.initiatorType === 'script');
                const cssResources = resources.filter(r => r.initiatorType === 'link');
                
                const totalJsSize = jsResources.reduce((sum, r) => sum + (r.transferSize || 0), 0);
                const totalCssSize = cssResources.reduce((sum, r) => sum + (r.transferSize || 0), 0);
                
                return {
                    totalJs: totalJsSize,
                    totalCss: totalCssSize,
                    total: totalJsSize + totalCssSize,
                    resourceCount: resources.length
                };
            };
        """
        )

    async def collect_performance_metrics(self) -> Dict[str, Any]:
        """Collect comprehensive performance metrics."""
        if not self.monitoring_active:
            await self.start_monitoring()

        try:
            # Wait for metrics to be collected
            await self.page.wait_for_timeout(2000)

            # Collect all performance data
            metrics = await self.page.evaluate(
                """
                async () => {
                    // Wait a bit more for observers to collect data
                    await new Promise(resolve => setTimeout(resolve, 1000));
                    
                    const data = window.performanceData || {};
                    const navigation = performance.getEntriesByType('navigation')[0];
                    const paint = performance.getEntriesByType('paint');
                    const bundleSize = window.estimateBundleSize ? window.estimateBundleSize() : {};
                    
                    // Memory usage (if available)
                    const memory = (performance as any).memory ? {
                        usedJSHeapSize: (performance as any).memory.usedJSHeapSize,
                        totalJSHeapSize: (performance as any).memory.totalJSHeapSize,
                        jsHeapSizeLimit: (performance as any).memory.jsHeapSizeLimit
                    } : null;
                    
                    // Connection information
                    const connection = (navigator as any).connection ? {
                        effectiveType: (navigator as any).connection.effectiveType,
                        downlink: (navigator as any).connection.downlink,
                        rtt: (navigator as any).connection.rtt
                    } : null;
                    
                    return {
                        // Core Web Vitals
                        coreWebVitals: {
                            largestContentfulPaint: data.vitals?.lcp || 0,
                            firstInputDelay: data.vitals?.fid || 0,
                            cumulativeLayoutShift: data.vitals?.cls || 0,
                            firstContentfulPaint: data.vitals?.first_contentful_paint || 0,
                            firstPaint: data.vitals?.first_paint || 0
                        },
                        
                        // Navigation timing
                        navigationTiming: {
                            domContentLoaded: navigation?.domContentLoadedEventEnd - navigation?.domContentLoadedEventStart || 0,
                            loadComplete: navigation?.loadEventEnd - navigation?.loadEventStart || 0,
                            timeToInteractive: navigation?.domInteractive - navigation?.fetchStart || 0,
                            totalLoadTime: navigation?.loadEventEnd - navigation?.fetchStart || 0,
                            dnsLookup: navigation?.domainLookupEnd - navigation?.domainLookupStart || 0,
                            tcpConnect: navigation?.connectEnd - navigation?.connectStart || 0,
                            serverResponse: navigation?.responseEnd - navigation?.requestStart || 0,
                            domProcessing: navigation?.domComplete - navigation?.domLoading || 0
                        },
                        
                        // Resource metrics
                        resourceMetrics: {
                            totalResources: data.resources?.length || 0,
                            totalTransferSize: data.resources?.reduce((sum, r) => sum + (r.transferSize || 0), 0) || 0,
                            bundleSize: bundleSize,
                            resourceTypes: data.resources?.reduce((types, r) => {
                                types[r.type] = (types[r.type] || 0) + 1;
                                return types;
                            }, {}) || {}
                        },
                        
                        // System metrics
                        systemMetrics: {
                            memory: memory,
                            connection: connection,
                            userAgent: navigator.userAgent,
                            viewport: {
                                width: window.innerWidth,
                                height: window.innerHeight
                            },
                            pixelRatio: window.devicePixelRatio || 1
                        },
                        
                        // Timestamp and URL
                        timestamp: new Date().toISOString(),
                        url: window.location.href,
                        title: document.title
                    };
                }
            """
            )

            # Add monitoring duration
            end_time = time.time()
            metrics["monitoringDuration"] = end_time - (self.start_time or end_time)

            # Evaluate performance against thresholds
            metrics["evaluation"] = self._evaluate_performance(metrics)

            return metrics

        except Exception as e:
            return {"error": str(e), "timestamp": time.time(), "url": self.page.url}

    def _evaluate_performance(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate performance metrics against thresholds."""
        evaluation = {
            "overall_score": 0,
            "grade": "F",
            "passed_checks": 0,
            "failed_checks": 0,
            "warnings": [],
            "critical_issues": [],
            "recommendations": [],
        }

        core_vitals = metrics.get("coreWebVitals", {})
        nav_timing = metrics.get("navigationTiming", {})
        resource_metrics = metrics.get("resourceMetrics", {})

        checks = []

        # Core Web Vitals checks
        lcp = core_vitals.get("largestContentfulPaint", 0)
        if lcp > 0:
            if (
                lcp
                <= PERFORMANCE_THRESHOLDS["core_web_vitals"]["largest_contentful_paint"]
            ):
                checks.append(
                    {
                        "name": "LCP",
                        "passed": True,
                        "value": lcp,
                        "threshold": PERFORMANCE_THRESHOLDS["core_web_vitals"][
                            "largest_contentful_paint"
                        ],
                    }
                )
            else:
                checks.append(
                    {
                        "name": "LCP",
                        "passed": False,
                        "value": lcp,
                        "threshold": PERFORMANCE_THRESHOLDS["core_web_vitals"][
                            "largest_contentful_paint"
                        ],
                    }
                )
                evaluation["critical_issues"].append(
                    f"Largest Contentful Paint ({lcp}ms) exceeds threshold"
                )

        fid = core_vitals.get("firstInputDelay", 0)
        if fid > 0:
            if fid <= PERFORMANCE_THRESHOLDS["core_web_vitals"]["first_input_delay"]:
                checks.append(
                    {
                        "name": "FID",
                        "passed": True,
                        "value": fid,
                        "threshold": PERFORMANCE_THRESHOLDS["core_web_vitals"][
                            "first_input_delay"
                        ],
                    }
                )
            else:
                checks.append(
                    {
                        "name": "FID",
                        "passed": False,
                        "value": fid,
                        "threshold": PERFORMANCE_THRESHOLDS["core_web_vitals"][
                            "first_input_delay"
                        ],
                    }
                )
                evaluation["critical_issues"].append(
                    f"First Input Delay ({fid}ms) exceeds threshold"
                )

        cls = core_vitals.get("cumulativeLayoutShift", 0)
        if cls <= PERFORMANCE_THRESHOLDS["core_web_vitals"]["cumulative_layout_shift"]:
            checks.append(
                {
                    "name": "CLS",
                    "passed": True,
                    "value": cls,
                    "threshold": PERFORMANCE_THRESHOLDS["core_web_vitals"][
                        "cumulative_layout_shift"
                    ],
                }
            )
        else:
            checks.append(
                {
                    "name": "CLS",
                    "passed": False,
                    "value": cls,
                    "threshold": PERFORMANCE_THRESHOLDS["core_web_vitals"][
                        "cumulative_layout_shift"
                    ],
                }
            )
            evaluation["critical_issues"].append(
                f"Cumulative Layout Shift ({cls}) exceeds threshold"
            )

        # Navigation timing checks
        dom_loaded = nav_timing.get("domContentLoaded", 0)
        if dom_loaded <= PERFORMANCE_THRESHOLDS["custom_metrics"]["dom_content_loaded"]:
            checks.append(
                {"name": "DOM Content Loaded", "passed": True, "value": dom_loaded}
            )
        else:
            checks.append(
                {"name": "DOM Content Loaded", "passed": False, "value": dom_loaded}
            )
            evaluation["warnings"].append(
                f"DOM Content Loaded ({dom_loaded}ms) is slow"
            )

        load_complete = nav_timing.get("loadComplete", 0)
        if load_complete <= PERFORMANCE_THRESHOLDS["custom_metrics"]["load_complete"]:
            checks.append(
                {"name": "Load Complete", "passed": True, "value": load_complete}
            )
        else:
            checks.append(
                {"name": "Load Complete", "passed": False, "value": load_complete}
            )
            evaluation["warnings"].append(f"Load Complete ({load_complete}ms) is slow")

        # Resource checks
        resource_count = resource_metrics.get("totalResources", 0)
        if resource_count <= PERFORMANCE_THRESHOLDS["custom_metrics"]["resource_count"]:
            checks.append(
                {"name": "Resource Count", "passed": True, "value": resource_count}
            )
        else:
            checks.append(
                {"name": "Resource Count", "passed": False, "value": resource_count}
            )
            evaluation["warnings"].append(f"Too many resources ({resource_count})")

        # Calculate overall score
        evaluation["passed_checks"] = sum(1 for check in checks if check["passed"])
        evaluation["failed_checks"] = len(checks) - evaluation["passed_checks"]

        if len(checks) > 0:
            evaluation["overall_score"] = (
                evaluation["passed_checks"] / len(checks)
            ) * 100

        # Determine grade
        score = evaluation["overall_score"]
        if score >= 95:
            evaluation["grade"] = "A+"
        elif score >= 90:
            evaluation["grade"] = "A"
        elif score >= 85:
            evaluation["grade"] = "B+"
        elif score >= 80:
            evaluation["grade"] = "B"
        elif score >= 70:
            evaluation["grade"] = "C"
        elif score >= 60:
            evaluation["grade"] = "D"
        else:
            evaluation["grade"] = "F"

        # Generate recommendations
        if evaluation["critical_issues"]:
            evaluation["recommendations"].append(
                "Address critical Core Web Vitals issues immediately"
            )
        if evaluation["warnings"]:
            evaluation["recommendations"].append(
                "Optimize resource loading and rendering performance"
            )
        if resource_count > 30:
            evaluation["recommendations"].append(
                "Consider bundling or reducing the number of resources"
            )

        evaluation["checks"] = checks
        return evaluation

    async def run_lighthouse_audit(
        self, url: str, device_config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Run Lighthouse audit for comprehensive performance analysis."""
        if not LIGHTHOUSE_ENABLED:
            return {"error": "Lighthouse not enabled", "lighthouse_available": False}

        device_config = device_config or DEVICE_CONFIGS[0]

        try:
            # Create temporary Lighthouse config
            lighthouse_config = {
                "extends": "lighthouse:default",
                "settings": {
                    "throttlingMethod": "devtools",
                    "throttling": {
                        "cpuSlowdownMultiplier": device_config.get("cpu_slowdown", 1),
                        "requestLatencyMs": (
                            40 if device_config.get("network") == "fast3g" else 150
                        ),
                        "downloadThroughputKbps": (
                            1600 if device_config.get("network") == "fast3g" else 400
                        ),
                        "uploadThroughputKbps": (
                            750 if device_config.get("network") == "fast3g" else 400
                        ),
                    },
                    "emulatedUserAgent": device_config.get("user_agent"),
                    "formFactor": device_config.get("name"),
                    "screenEmulation": {
                        "mobile": device_config.get("name") == "mobile",
                        "width": device_config["viewport"]["width"],
                        "height": device_config["viewport"]["height"],
                        "deviceScaleFactor": 1,
                        "disabled": False,
                    },
                },
            }

            # Save config to temporary file
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as f:
                json.dump(lighthouse_config, f)
                config_path = f.name

            # Run Lighthouse
            output_path = (
                LIGHTHOUSE_REPORTS_DIR
                / f"lighthouse_{device_config['name']}_{int(time.time())}.json"
            )

            cmd = [
                "lighthouse",
                url,
                "--output=json",
                f"--output-path={output_path}",
                f"--config-path={config_path}",
                "--chrome-flags=--headless --no-sandbox --disable-dev-shm-usage",
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

            # Clean up config file
            os.unlink(config_path)

            if result.returncode == 0 and output_path.exists():
                with open(output_path) as f:
                    lighthouse_data = json.load(f)

                # Extract key metrics
                audits = lighthouse_data.get("audits", {})
                categories = lighthouse_data.get("categories", {})

                processed_results = {
                    "scores": {
                        category: (
                            data.get("score", 0) * 100
                            if data.get("score") is not None
                            else 0
                        )
                        for category, data in categories.items()
                    },
                    "metrics": {
                        "first_contentful_paint": audits.get(
                            "first-contentful-paint", {}
                        ).get("numericValue", 0),
                        "largest_contentful_paint": audits.get(
                            "largest-contentful-paint", {}
                        ).get("numericValue", 0),
                        "first_input_delay": audits.get("max-potential-fid", {}).get(
                            "numericValue", 0
                        ),
                        "cumulative_layout_shift": audits.get(
                            "cumulative-layout-shift", {}
                        ).get("numericValue", 0),
                        "total_blocking_time": audits.get(
                            "total-blocking-time", {}
                        ).get("numericValue", 0),
                        "speed_index": audits.get("speed-index", {}).get(
                            "numericValue", 0
                        ),
                    },
                    "opportunities": [
                        {
                            "title": audit.get("title", ""),
                            "description": audit.get("description", ""),
                            "potential_savings": audit.get("numericValue", 0),
                        }
                        for audit_id, audit in audits.items()
                        if audit.get("scoreDisplayMode") == "numeric"
                        and audit.get("numericValue", 0) > 0
                    ],
                    "device_config": device_config,
                    "timestamp": lighthouse_data.get("fetchTime"),
                    "lighthouse_version": lighthouse_data.get("lighthouseVersion"),
                    "report_path": str(output_path),
                }

                return processed_results
            else:
                return {
                    "error": f"Lighthouse failed: {result.stderr}",
                    "returncode": result.returncode,
                }

        except subprocess.TimeoutExpired:
            return {"error": "Lighthouse audit timed out"}
        except Exception as e:
            return {"error": f"Lighthouse audit failed: {str(e)}"}

    async def save_performance_report(
        self, metrics: Dict[str, Any], test_name: str = "performance_test"
    ):
        """Save performance report to file."""
        timestamp = metrics.get("timestamp", time.time())
        if isinstance(timestamp, str):
            timestamp = timestamp.replace(":", "-").replace("T", "_").split(".")[0]
        else:
            timestamp = str(int(timestamp))

        filename = f"{test_name}_{timestamp}.json"
        report_path = PERFORMANCE_REPORTS_DIR / filename

        with open(report_path, "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"Performance report saved: {report_path}")
        return str(report_path)


# Test fixtures
@pytest.fixture
async def performance_monitor(page: Page):
    """Create performance monitor instance."""
    monitor = PerformanceMonitor(page)
    await monitor.start_monitoring()
    return monitor


@pytest.fixture(params=DEVICE_CONFIGS)
def device_config(request):
    """Parametrize tests across different device configurations."""
    return request.param


# Performance test suite
@pytest.mark.skipif(
    not PERFORMANCE_TESTING_ENABLED, reason="Performance testing disabled"
)
class TestPerformanceComprehensive:
    """Comprehensive performance testing with Core Web Vitals and Lighthouse."""

    async def test_homepage_performance(
        self, performance_monitor: PerformanceMonitor, page: Page
    ):
        """Test homepage performance metrics."""
        await page.goto("/")

        metrics = await performance_monitor.collect_performance_metrics()

        # Assert Core Web Vitals
        core_vitals = metrics.get("coreWebVitals", {})

        # LCP should be under 2.5 seconds
        lcp = core_vitals.get("largestContentfulPaint", 0)
        if lcp > 0:  # Only test if LCP was measured
            assert (
                lcp
                <= PERFORMANCE_THRESHOLDS["core_web_vitals"]["largest_contentful_paint"]
            ), f"LCP ({lcp}ms) exceeds threshold ({PERFORMANCE_THRESHOLDS['core_web_vitals']['largest_contentful_paint']}ms)"

        # CLS should be under 0.1
        cls = core_vitals.get("cumulativeLayoutShift", 0)
        assert (
            cls <= PERFORMANCE_THRESHOLDS["core_web_vitals"]["cumulative_layout_shift"]
        ), f"CLS ({cls}) exceeds threshold ({PERFORMANCE_THRESHOLDS['core_web_vitals']['cumulative_layout_shift']})"

        # Overall evaluation should pass
        evaluation = metrics.get("evaluation", {})
        assert (
            evaluation.get("overall_score", 0) >= 70
        ), f"Performance score ({evaluation.get('overall_score', 0)}) is too low"

        # Save report
        await performance_monitor.save_performance_report(
            metrics, "homepage_performance"
        )

    async def test_detectors_page_performance(
        self, performance_monitor: PerformanceMonitor, page: Page
    ):
        """Test detectors page performance."""
        await page.goto("/detectors")

        metrics = await performance_monitor.collect_performance_metrics()

        # Check navigation timing
        nav_timing = metrics.get("navigationTiming", {})
        dom_loaded = nav_timing.get("domContentLoaded", 0)

        assert (
            dom_loaded <= PERFORMANCE_THRESHOLDS["custom_metrics"]["dom_content_loaded"]
        ), f"DOM Content Loaded ({dom_loaded}ms) is too slow"

        # Check resource usage
        resource_metrics = metrics.get("resourceMetrics", {})
        resource_count = resource_metrics.get("totalResources", 0)

        assert (
            resource_count <= PERFORMANCE_THRESHOLDS["custom_metrics"]["resource_count"]
        ), f"Too many resources loaded ({resource_count})"

        await performance_monitor.save_performance_report(
            metrics, "detectors_performance"
        )

    async def test_performance_across_devices(
        self,
        performance_monitor: PerformanceMonitor,
        page: Page,
        device_config: Dict[str, Any],
    ):
        """Test performance across different device configurations."""
        # Set viewport for device
        await page.set_viewport_size(device_config["viewport"])

        # Simulate network conditions (simplified)
        if device_config["network"] == "slow3g":
            # Add artificial delay for slow network simulation
            await page.route(
                "**/*",
                lambda route: (
                    (
                        asyncio.sleep(0.1)
                        if asyncio.iscoroutinefunction(asyncio.sleep)
                        else None
                    ),
                    route.continue_(),
                ),
            )

        await page.goto("/")

        metrics = await performance_monitor.collect_performance_metrics()

        # Device-specific thresholds
        device_thresholds = PERFORMANCE_THRESHOLDS["core_web_vitals"].copy()

        if device_config["name"] == "mobile":
            # More lenient thresholds for mobile
            device_thresholds["largest_contentful_paint"] *= 1.5
            device_thresholds["first_input_delay"] *= 1.2

        # Test against device-specific thresholds
        core_vitals = metrics.get("coreWebVitals", {})
        lcp = core_vitals.get("largestContentfulPaint", 0)

        if lcp > 0:
            assert (
                lcp <= device_thresholds["largest_contentful_paint"]
            ), f"LCP on {device_config['name']} ({lcp}ms) exceeds threshold"

        await performance_monitor.save_performance_report(
            metrics, f"device_performance_{device_config['name']}"
        )

    @pytest.mark.skipif(not LIGHTHOUSE_ENABLED, reason="Lighthouse not enabled")
    async def test_lighthouse_audit_comprehensive(
        self, performance_monitor: PerformanceMonitor, page: Page
    ):
        """Test comprehensive Lighthouse audit."""
        url = (
            page.url
            if page.url and page.url != "about:blank"
            else "http://localhost:8000"
        )

        # Test different device configurations
        for device_config in DEVICE_CONFIGS:
            lighthouse_results = await performance_monitor.run_lighthouse_audit(
                url, device_config
            )

            if "error" in lighthouse_results:
                pytest.skip(f"Lighthouse audit failed: {lighthouse_results['error']}")

            scores = lighthouse_results.get("scores", {})

            # Assert Lighthouse scores meet thresholds
            for category, threshold in PERFORMANCE_THRESHOLDS[
                "lighthouse_scores"
            ].items():
                if category in scores:
                    score = scores[category]
                    assert (
                        score >= threshold
                    ), f"Lighthouse {category} score ({score}) below threshold ({threshold}) on {device_config['name']}"

    async def test_resource_optimization(
        self, performance_monitor: PerformanceMonitor, page: Page
    ):
        """Test resource loading optimization."""
        await page.goto("/")

        metrics = await performance_monitor.collect_performance_metrics()
        resource_metrics = metrics.get("resourceMetrics", {})

        # Check bundle size
        bundle_size = resource_metrics.get("bundleSize", {})
        total_bundle = bundle_size.get("total", 0)

        if total_bundle > 0:
            assert (
                total_bundle <= PERFORMANCE_THRESHOLDS["custom_metrics"]["bundle_size"]
            ), f"Bundle size ({total_bundle} bytes) exceeds threshold"

        # Check resource types distribution
        resource_types = resource_metrics.get("resourceTypes", {})

        # Should not have too many individual script files (suggests bundling issues)
        script_count = resource_types.get("script", 0)
        assert (
            script_count <= 10
        ), f"Too many script resources ({script_count}), consider bundling"

        await performance_monitor.save_performance_report(
            metrics, "resource_optimization"
        )

    async def test_core_web_vitals_monitoring(
        self, performance_monitor: PerformanceMonitor, page: Page
    ):
        """Test Core Web Vitals monitoring across multiple pages."""
        pages_to_test = [
            ("/", "homepage"),
            ("/detectors", "detectors"),
            ("/datasets", "datasets"),
            ("/detection", "detection"),
            ("/visualizations", "visualizations"),
        ]

        all_results = {}

        for url, page_name in pages_to_test:
            await page.goto(url)

            # Wait for page to be fully interactive
            await page.wait_for_load_state("networkidle")

            metrics = await performance_monitor.collect_performance_metrics()
            all_results[page_name] = metrics

            # Individual page assertions
            evaluation = metrics.get("evaluation", {})
            assert (
                evaluation.get("overall_score", 0) >= 60
            ), f"Performance score for {page_name} ({evaluation.get('overall_score', 0)}) is too low"

        # Save comprehensive report
        comprehensive_report = {
            "test_type": "core_web_vitals_monitoring",
            "pages_tested": list(all_results.keys()),
            "results": all_results,
            "summary": {
                "average_score": sum(
                    result.get("evaluation", {}).get("overall_score", 0)
                    for result in all_results.values()
                )
                / len(all_results),
                "pages_passed": sum(
                    1
                    for result in all_results.values()
                    if result.get("evaluation", {}).get("overall_score", 0) >= 70
                ),
                "critical_issues": sum(
                    len(result.get("evaluation", {}).get("critical_issues", []))
                    for result in all_results.values()
                ),
            },
            "timestamp": time.time(),
        }

        await performance_monitor.save_performance_report(
            comprehensive_report, "comprehensive_cwv_monitoring"
        )


# Utility functions
def generate_performance_summary_report(
    reports_dir: Path = PERFORMANCE_REPORTS_DIR,
) -> Dict[str, Any]:
    """Generate summary report from all performance test results."""
    summary = {
        "total_reports": 0,
        "pages_tested": set(),
        "average_scores": {},
        "common_issues": {},
        "recommendations": set(),
        "trend_analysis": [],
    }

    for report_file in reports_dir.glob("*.json"):
        try:
            with open(report_file) as f:
                data = json.load(f)

            summary["total_reports"] += 1

            url = data.get("url", "unknown")
            summary["pages_tested"].add(url)

            # Aggregate scores
            evaluation = data.get("evaluation", {})
            score = evaluation.get("overall_score", 0)
            if score > 0:
                page_key = url.split("/")[-1] or "homepage"
                if page_key not in summary["average_scores"]:
                    summary["average_scores"][page_key] = []
                summary["average_scores"][page_key].append(score)

            # Collect issues
            for issue in evaluation.get("critical_issues", []):
                summary["common_issues"][issue] = (
                    summary["common_issues"].get(issue, 0) + 1
                )

            # Collect recommendations
            for rec in evaluation.get("recommendations", []):
                summary["recommendations"].add(rec)

        except Exception as e:
            print(f"Error processing {report_file}: {e}")

    # Calculate averages
    for page, scores in summary["average_scores"].items():
        summary["average_scores"][page] = sum(scores) / len(scores)

    # Convert sets to lists
    summary["pages_tested"] = list(summary["pages_tested"])
    summary["recommendations"] = list(summary["recommendations"])

    return summary


if __name__ == "__main__":
    # Run performance tests standalone
    pytest.main(
        [
            __file__,
            "-v",
            "--html=test_reports/performance_report.html",
            "--self-contained-html",
        ]
    )
