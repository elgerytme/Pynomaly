"""
Performance Validator for Advanced UI Testing

Provides comprehensive performance testing including:
- Core Web Vitals measurement and validation
- Resource loading optimization analysis
- Network performance monitoring
- JavaScript execution profiling
- Memory usage tracking
"""

import time
from typing import Any, Dict, List, Optional
from playwright.async_api import Page


class PerformanceValidator:
    """
    Comprehensive performance validation for web applications
    """

    def __init__(self):
        self.thresholds = {
            "core_web_vitals": {
                "lcp": {"good": 2500, "needs_improvement": 4000},  # ms
                "fid": {"good": 100, "needs_improvement": 300},    # ms
                "cls": {"good": 0.1, "needs_improvement": 0.25},   # score
                "fcp": {"good": 1800, "needs_improvement": 3000},  # ms
                "ttfb": {"good": 800, "needs_improvement": 1800}   # ms
            },
            "page_metrics": {
                "dom_content_loaded": 3000,  # ms
                "load_complete": 5000,       # ms
                "first_paint": 2000,         # ms
                "interactive": 4000          # ms
            },
            "resource_metrics": {
                "total_requests": 50,
                "total_size_mb": 2.0,
                "slow_resource_threshold": 1000,  # ms
                "failed_requests_threshold": 0.05  # 5%
            },
            "javascript_metrics": {
                "execution_time": 1000,      # ms
                "heap_size_mb": 50,          # MB
                "event_listeners": 100       # count
            }
        }

    async def validate_page_performance(self, page: Page, test_name: str = "performance_test") -> Dict[str, Any]:
        """
        Perform comprehensive performance validation
        
        Args:
            page: Playwright page object
            test_name: Name of the test for reporting
            
        Returns:
            Performance validation results with metrics and recommendations
        """
        results = {
            "test_name": test_name,
            "overall_score": 0,
            "performance_grade": "F",
            "metrics": {},
            "violations": [],
            "recommendations": [],
            "timestamp": time.time()
        }
        
        try:
            # Measure Core Web Vitals
            core_vitals = await self._measure_core_web_vitals(page)
            results["metrics"]["core_web_vitals"] = core_vitals
            
            # Measure basic page timing
            page_timing = await self._measure_page_timing(page)
            results["metrics"]["page_timing"] = page_timing
            
            # Analyze resource loading
            resource_analysis = await self._analyze_resource_loading(page)
            results["metrics"]["resource_analysis"] = resource_analysis
            
            # Measure JavaScript performance
            js_performance = await self._measure_javascript_performance(page)
            results["metrics"]["javascript_performance"] = js_performance
            
            # Analyze network efficiency
            network_analysis = await self._analyze_network_efficiency(page)
            results["metrics"]["network_analysis"] = network_analysis
            
            # Calculate overall performance score
            results["overall_score"] = self._calculate_performance_score(results["metrics"])
            results["performance_grade"] = self._get_performance_grade(results["overall_score"])
            
            # Generate violations and recommendations
            results["violations"] = self._identify_performance_violations(results["metrics"])
            results["recommendations"] = self._generate_performance_recommendations(results["metrics"])
            
        except Exception as e:
            results["error"] = str(e)
            results["overall_score"] = 0
        
        return results

    async def _measure_core_web_vitals(self, page: Page) -> Dict[str, Any]:
        """Measure Core Web Vitals using Performance Observer API"""
        vitals_results = {
            "lcp": None,
            "fid": None,
            "cls": None,
            "fcp": None,
            "ttfb": None,
            "measurement_status": "pending"
        }
        
        try:
            # Inject performance measurement script
            vitals_data = await page.evaluate("""
                () => {
                    return new Promise((resolve) => {
                        const vitals = {};
                        let measurementCount = 0;
                        const expectedMeasurements = 4; // LCP, CLS, FCP, TTFB
                        
                        const checkComplete = () => {
                            measurementCount++;
                            if (measurementCount >= expectedMeasurements) {
                                resolve(vitals);
                            }
                        };
                        
                        // Largest Contentful Paint
                        new PerformanceObserver((list) => {
                            const entries = list.getEntries();
                            const lastEntry = entries[entries.length - 1];
                            vitals.lcp = lastEntry.startTime;
                            checkComplete();
                        }).observe({entryTypes: ['largest-contentful-paint']});
                        
                        // First Contentful Paint
                        new PerformanceObserver((list) => {
                            const entries = list.getEntries();
                            const fcpEntry = entries.find(entry => entry.name === 'first-contentful-paint');
                            if (fcpEntry) {
                                vitals.fcp = fcpEntry.startTime;
                            }
                            checkComplete();
                        }).observe({entryTypes: ['paint']});
                        
                        // Cumulative Layout Shift
                        let clsValue = 0;
                        new PerformanceObserver((list) => {
                            for (const entry of list.getEntries()) {
                                if (!entry.hadRecentInput) {
                                    clsValue += entry.value;
                                }
                            }
                            vitals.cls = clsValue;
                        }).observe({entryTypes: ['layout-shift']});
                        
                        // Time to First Byte (from Navigation Timing)
                        const navigation = performance.getEntriesByType('navigation')[0];
                        if (navigation) {
                            vitals.ttfb = navigation.responseStart - navigation.requestStart;
                        }
                        checkComplete();
                        
                        // First Input Delay measurement setup (will be 0 without actual interaction)
                        vitals.fid = 0;
                        
                        // Timeout to ensure we don't wait forever
                        setTimeout(() => {
                            checkComplete();
                            checkComplete(); // Extra calls to meet threshold
                        }, 3000);
                    });
                }
            """)
            
            vitals_results.update(vitals_data)
            vitals_results["measurement_status"] = "completed"
            
            # Validate measurements against thresholds
            for metric, value in vitals_data.items():
                if value is not None and metric in self.thresholds["core_web_vitals"]:
                    thresholds = self.thresholds["core_web_vitals"][metric]
                    if value <= thresholds["good"]:
                        vitals_results[f"{metric}_status"] = "good"
                    elif value <= thresholds["needs_improvement"]:
                        vitals_results[f"{metric}_status"] = "needs_improvement"
                    else:
                        vitals_results[f"{metric}_status"] = "poor"
            
        except Exception as e:
            vitals_results["error"] = str(e)
            vitals_results["measurement_status"] = "failed"
        
        return vitals_results

    async def _measure_page_timing(self, page: Page) -> Dict[str, Any]:
        """Measure basic page load timing metrics"""
        timing_results = {}
        
        try:
            timing_data = await page.evaluate("""
                () => {
                    const timing = performance.timing;
                    const navigation = performance.getEntriesByType('navigation')[0];
                    
                    return {
                        // Basic timing
                        navigation_start: timing.navigationStart,
                        dom_content_loaded: timing.domContentLoadedEventEnd - timing.navigationStart,
                        load_complete: timing.loadEventEnd - timing.navigationStart,
                        dom_interactive: timing.domInteractive - timing.navigationStart,
                        
                        // Network timing
                        dns_lookup: timing.domainLookupEnd - timing.domainLookupStart,
                        tcp_connect: timing.connectEnd - timing.connectStart,
                        ssl_handshake: timing.connectEnd - timing.secureConnectionStart,
                        server_response: timing.responseEnd - timing.requestStart,
                        
                        // Processing timing
                        dom_processing: timing.domComplete - timing.domLoading,
                        
                        // Resource timing summary
                        resource_count: performance.getEntriesByType('resource').length,
                        
                        // Modern metrics from Navigation Timing API Level 2
                        redirect_time: navigation ? navigation.redirectEnd - navigation.redirectStart : 0,
                        cache_seek_time: navigation ? navigation.domainLookupStart - navigation.fetchStart : 0,
                        request_time: navigation ? navigation.responseEnd - navigation.requestStart : 0,
                        response_time: navigation ? navigation.responseEnd - navigation.responseStart : 0
                    };
                }
            """)
            
            timing_results = timing_data
            
            # Add performance assessment
            timing_results["assessments"] = {}
            
            # Assess key metrics
            if timing_data["dom_content_loaded"] <= self.thresholds["page_metrics"]["dom_content_loaded"]:
                timing_results["assessments"]["dom_content_loaded"] = "good"
            else:
                timing_results["assessments"]["dom_content_loaded"] = "poor"
            
            if timing_data["load_complete"] <= self.thresholds["page_metrics"]["load_complete"]:
                timing_results["assessments"]["load_complete"] = "good"
            else:
                timing_results["assessments"]["load_complete"] = "poor"
            
        except Exception as e:
            timing_results["error"] = str(e)
        
        return timing_results

    async def _analyze_resource_loading(self, page: Page) -> Dict[str, Any]:
        """Analyze resource loading performance and efficiency"""
        resource_results = {}
        
        try:
            resource_data = await page.evaluate("""
                () => {
                    const resources = performance.getEntriesByType('resource');
                    const analysis = {
                        total_resources: resources.length,
                        total_size: 0,
                        total_transfer_size: 0,
                        resource_types: {},
                        slow_resources: [],
                        failed_resources: [],
                        cached_resources: 0,
                        compressed_resources: 0
                    };
                    
                    resources.forEach(resource => {
                        const duration = resource.responseEnd - resource.requestStart;
                        const size = resource.transferSize || 0;
                        const encodedSize = resource.encodedBodySize || 0;
                        const decodedSize = resource.decodedBodySize || 0;
                        
                        // Track total sizes
                        analysis.total_size += decodedSize;
                        analysis.total_transfer_size += size;
                        
                        // Categorize by resource type
                        const extension = resource.name.split('.').pop().split('?')[0].toLowerCase();
                        if (!analysis.resource_types[extension]) {
                            analysis.resource_types[extension] = {
                                count: 0,
                                total_size: 0,
                                total_duration: 0
                            };
                        }
                        analysis.resource_types[extension].count++;
                        analysis.resource_types[extension].total_size += decodedSize;
                        analysis.resource_types[extension].total_duration += duration;
                        
                        // Check for slow resources (>1s)
                        if (duration > 1000) {
                            analysis.slow_resources.push({
                                name: resource.name.split('/').pop(),
                                duration: Math.round(duration),
                                size: Math.round(decodedSize / 1024), // KB
                                type: extension
                            });
                        }
                        
                        // Check for cached resources
                        if (size === 0 && encodedSize > 0) {
                            analysis.cached_resources++;
                        }
                        
                        // Check for compressed resources
                        if (encodedSize > 0 && decodedSize > encodedSize) {
                            analysis.compressed_resources++;
                        }
                        
                        // Check for failed resources (this is a simplified check)
                        if (duration === 0 && size === 0) {
                            analysis.failed_resources.push(resource.name);
                        }
                    });
                    
                    // Calculate percentages
                    analysis.cache_hit_rate = (analysis.cached_resources / analysis.total_resources) * 100;
                    analysis.compression_rate = (analysis.compressed_resources / analysis.total_resources) * 100;
                    analysis.total_size_mb = analysis.total_size / (1024 * 1024);
                    analysis.total_transfer_mb = analysis.total_transfer_size / (1024 * 1024);
                    
                    return analysis;
                }
            """)
            
            resource_results = resource_data
            
            # Add performance assessments
            resource_results["assessments"] = {}
            
            # Assess total requests
            if resource_data["total_resources"] <= self.thresholds["resource_metrics"]["total_requests"]:
                resource_results["assessments"]["request_count"] = "good"
            else:
                resource_results["assessments"]["request_count"] = "poor"
            
            # Assess total size
            if resource_data["total_size_mb"] <= self.thresholds["resource_metrics"]["total_size_mb"]:
                resource_results["assessments"]["total_size"] = "good"
            else:
                resource_results["assessments"]["total_size"] = "poor"
            
            # Assess cache performance
            if resource_data["cache_hit_rate"] >= 20:  # 20% cache hit rate
                resource_results["assessments"]["caching"] = "good"
            else:
                resource_results["assessments"]["caching"] = "poor"
            
        except Exception as e:
            resource_results["error"] = str(e)
        
        return resource_results

    async def _measure_javascript_performance(self, page: Page) -> Dict[str, Any]:
        """Measure JavaScript execution performance and memory usage"""
        js_results = {}
        
        try:
            js_data = await page.evaluate("""
                () => {
                    const measurements = {
                        execution_metrics: {},
                        memory_metrics: {},
                        dom_metrics: {},
                        event_metrics: {}
                    };
                    
                    // Memory metrics (if available)
                    if (performance.memory) {
                        measurements.memory_metrics = {
                            used_heap_size: performance.memory.usedJSHeapSize,
                            total_heap_size: performance.memory.totalJSHeapSize,
                            heap_size_limit: performance.memory.jsHeapSizeLimit,
                            used_heap_mb: Math.round(performance.memory.usedJSHeapSize / (1024 * 1024)),
                            total_heap_mb: Math.round(performance.memory.totalJSHeapSize / (1024 * 1024))
                        };
                    }
                    
                    // DOM metrics
                    measurements.dom_metrics = {
                        total_elements: document.querySelectorAll('*').length,
                        scripts_count: document.querySelectorAll('script').length,
                        stylesheets_count: document.querySelectorAll('link[rel="stylesheet"]').length,
                        images_count: document.querySelectorAll('img').length,
                        forms_count: document.querySelectorAll('form').length
                    };
                    
                    // Event listener metrics (simplified)
                    const elementsWithEvents = document.querySelectorAll('[onclick], [onload], [onchange]');
                    measurements.event_metrics = {
                        inline_event_handlers: elementsWithEvents.length,
                        estimated_listeners: elementsWithEvents.length // Simplified estimate
                    };
                    
                    // JavaScript execution timing (measure simple operations)
                    const start = performance.now();
                    for (let i = 0; i < 10000; i++) {
                        Math.random() * 100;
                    }
                    const executionTime = performance.now() - start;
                    
                    measurements.execution_metrics = {
                        simple_operation_time: executionTime,
                        estimated_js_performance: executionTime < 10 ? 'good' : executionTime < 50 ? 'fair' : 'poor'
                    };
                    
                    return measurements;
                }
            """)
            
            js_results = js_data
            
            # Add performance assessments
            js_results["assessments"] = {}
            
            # Assess memory usage
            if js_data["memory_metrics"].get("used_heap_mb", 0) <= self.thresholds["javascript_metrics"]["heap_size_mb"]:
                js_results["assessments"]["memory_usage"] = "good"
            else:
                js_results["assessments"]["memory_usage"] = "poor"
            
            # Assess DOM complexity
            total_elements = js_data["dom_metrics"]["total_elements"]
            if total_elements <= 1000:
                js_results["assessments"]["dom_complexity"] = "good"
            elif total_elements <= 3000:
                js_results["assessments"]["dom_complexity"] = "fair"
            else:
                js_results["assessments"]["dom_complexity"] = "poor"
            
        except Exception as e:
            js_results["error"] = str(e)
        
        return js_results

    async def _analyze_network_efficiency(self, page: Page) -> Dict[str, Any]:
        """Analyze network efficiency and optimization opportunities"""
        network_results = {}
        
        try:
            network_data = await page.evaluate("""
                () => {
                    const resources = performance.getEntriesByType('resource');
                    const analysis = {
                        http_versions: {},
                        compression_analysis: {},
                        caching_analysis: {},
                        connection_analysis: {},
                        efficiency_metrics: {}
                    };
                    
                    let totalUncompressedSize = 0;
                    let totalCompressedSize = 0;
                    let totalConnections = 0;
                    let cachedResources = 0;
                    
                    resources.forEach(resource => {
                        const transferSize = resource.transferSize || 0;
                        const decodedSize = resource.decodedBodySize || 0;
                        const encodedSize = resource.encodedBodySize || 0;
                        
                        // HTTP version analysis
                        const protocol = resource.nextHopProtocol || 'unknown';
                        analysis.http_versions[protocol] = (analysis.http_versions[protocol] || 0) + 1;
                        
                        // Compression analysis
                        if (decodedSize > 0) {
                            totalUncompressedSize += decodedSize;
                            totalCompressedSize += encodedSize;
                            
                            if (encodedSize < decodedSize) {
                                const compressionRatio = ((decodedSize - encodedSize) / decodedSize) * 100;
                                if (!analysis.compression_analysis.compressed_resources) {
                                    analysis.compression_analysis.compressed_resources = [];
                                }
                                analysis.compression_analysis.compressed_resources.push({
                                    name: resource.name.split('/').pop(),
                                    ratio: Math.round(compressionRatio)
                                });
                            }
                        }
                        
                        // Caching analysis
                        if (transferSize === 0 && decodedSize > 0) {
                            cachedResources++;
                        }
                        
                        // Connection analysis (simplified)
                        if (resource.connectStart && resource.connectEnd) {
                            totalConnections++;
                        }
                    });
                    
                    // Calculate efficiency metrics
                    analysis.efficiency_metrics = {
                        total_resources: resources.length,
                        cached_percentage: (cachedResources / resources.length) * 100,
                        compression_ratio: totalUncompressedSize > 0 ? 
                            ((totalUncompressedSize - totalCompressedSize) / totalUncompressedSize) * 100 : 0,
                        total_uncompressed_mb: totalUncompressedSize / (1024 * 1024),
                        total_compressed_mb: totalCompressedSize / (1024 * 1024),
                        bandwidth_saved_mb: (totalUncompressedSize - totalCompressedSize) / (1024 * 1024),
                        estimated_connections: totalConnections
                    };
                    
                    return analysis;
                }
            """)
            
            network_results = network_data
            
            # Add efficiency assessments
            network_results["assessments"] = {}
            
            # Assess compression efficiency
            compression_ratio = network_data["efficiency_metrics"]["compression_ratio"]
            if compression_ratio >= 60:
                network_results["assessments"]["compression"] = "excellent"
            elif compression_ratio >= 30:
                network_results["assessments"]["compression"] = "good"
            else:
                network_results["assessments"]["compression"] = "poor"
            
            # Assess caching efficiency
            cached_percentage = network_data["efficiency_metrics"]["cached_percentage"]
            if cached_percentage >= 50:
                network_results["assessments"]["caching"] = "excellent"
            elif cached_percentage >= 20:
                network_results["assessments"]["caching"] = "good"
            else:
                network_results["assessments"]["caching"] = "poor"
            
        except Exception as e:
            network_results["error"] = str(e)
        
        return network_results

    def _calculate_performance_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate overall performance score (0-100)"""
        scores = []
        weights = {
            "core_web_vitals": 0.4,
            "page_timing": 0.25,
            "resource_analysis": 0.2,
            "javascript_performance": 0.1,
            "network_analysis": 0.05
        }
        
        try:
            # Core Web Vitals score
            if "core_web_vitals" in metrics:
                cwv_score = self._score_core_web_vitals(metrics["core_web_vitals"])
                scores.append(("core_web_vitals", cwv_score))
            
            # Page timing score
            if "page_timing" in metrics:
                timing_score = self._score_page_timing(metrics["page_timing"])
                scores.append(("page_timing", timing_score))
            
            # Resource analysis score
            if "resource_analysis" in metrics:
                resource_score = self._score_resource_analysis(metrics["resource_analysis"])
                scores.append(("resource_analysis", resource_score))
            
            # JavaScript performance score
            if "javascript_performance" in metrics:
                js_score = self._score_javascript_performance(metrics["javascript_performance"])
                scores.append(("javascript_performance", js_score))
            
            # Network analysis score
            if "network_analysis" in metrics:
                network_score = self._score_network_analysis(metrics["network_analysis"])
                scores.append(("network_analysis", network_score))
            
            # Calculate weighted average
            total_score = 0
            total_weight = 0
            
            for category, score in scores:
                weight = weights.get(category, 0.1)
                total_score += score * weight
                total_weight += weight
            
            return total_score / total_weight if total_weight > 0 else 0
            
        except Exception:
            return 0

    def _score_core_web_vitals(self, cwv_metrics: Dict[str, Any]) -> float:
        """Score Core Web Vitals (0-100)"""
        scores = []
        
        # LCP scoring
        lcp = cwv_metrics.get("lcp")
        if lcp is not None:
            if lcp <= 2500:
                scores.append(100)
            elif lcp <= 4000:
                scores.append(50)
            else:
                scores.append(0)
        
        # CLS scoring
        cls = cwv_metrics.get("cls")
        if cls is not None:
            if cls <= 0.1:
                scores.append(100)
            elif cls <= 0.25:
                scores.append(50)
            else:
                scores.append(0)
        
        # FCP scoring
        fcp = cwv_metrics.get("fcp")
        if fcp is not None:
            if fcp <= 1800:
                scores.append(100)
            elif fcp <= 3000:
                scores.append(50)
            else:
                scores.append(0)
        
        return sum(scores) / len(scores) if scores else 50

    def _score_page_timing(self, timing_metrics: Dict[str, Any]) -> float:
        """Score page timing metrics (0-100)"""
        scores = []
        
        # DOM Content Loaded
        dcl = timing_metrics.get("dom_content_loaded")
        if dcl is not None:
            if dcl <= 2000:
                scores.append(100)
            elif dcl <= 4000:
                scores.append(50)
            else:
                scores.append(0)
        
        # Load Complete
        load_complete = timing_metrics.get("load_complete")
        if load_complete is not None:
            if load_complete <= 3000:
                scores.append(100)
            elif load_complete <= 6000:
                scores.append(50)
            else:
                scores.append(0)
        
        return sum(scores) / len(scores) if scores else 50

    def _score_resource_analysis(self, resource_metrics: Dict[str, Any]) -> float:
        """Score resource loading efficiency (0-100)"""
        scores = []
        
        # Total size
        total_size_mb = resource_metrics.get("total_size_mb", 0)
        if total_size_mb <= 1:
            scores.append(100)
        elif total_size_mb <= 3:
            scores.append(75)
        elif total_size_mb <= 5:
            scores.append(50)
        else:
            scores.append(25)
        
        # Request count
        total_resources = resource_metrics.get("total_resources", 0)
        if total_resources <= 30:
            scores.append(100)
        elif total_resources <= 60:
            scores.append(75)
        elif total_resources <= 100:
            scores.append(50)
        else:
            scores.append(25)
        
        # Cache hit rate
        cache_hit_rate = resource_metrics.get("cache_hit_rate", 0)
        if cache_hit_rate >= 50:
            scores.append(100)
        elif cache_hit_rate >= 20:
            scores.append(75)
        else:
            scores.append(50)
        
        return sum(scores) / len(scores) if scores else 50

    def _score_javascript_performance(self, js_metrics: Dict[str, Any]) -> float:
        """Score JavaScript performance (0-100)"""
        scores = []
        
        # Memory usage
        memory_metrics = js_metrics.get("memory_metrics", {})
        used_heap_mb = memory_metrics.get("used_heap_mb", 0)
        if used_heap_mb <= 20:
            scores.append(100)
        elif used_heap_mb <= 50:
            scores.append(75)
        else:
            scores.append(50)
        
        # DOM complexity
        dom_metrics = js_metrics.get("dom_metrics", {})
        total_elements = dom_metrics.get("total_elements", 0)
        if total_elements <= 500:
            scores.append(100)
        elif total_elements <= 1500:
            scores.append(75)
        else:
            scores.append(50)
        
        return sum(scores) / len(scores) if scores else 75

    def _score_network_analysis(self, network_metrics: Dict[str, Any]) -> float:
        """Score network efficiency (0-100)"""
        scores = []
        
        efficiency_metrics = network_metrics.get("efficiency_metrics", {})
        
        # Compression ratio
        compression_ratio = efficiency_metrics.get("compression_ratio", 0)
        if compression_ratio >= 60:
            scores.append(100)
        elif compression_ratio >= 30:
            scores.append(75)
        else:
            scores.append(50)
        
        # Cached percentage
        cached_percentage = efficiency_metrics.get("cached_percentage", 0)
        if cached_percentage >= 40:
            scores.append(100)
        elif cached_percentage >= 20:
            scores.append(75)
        else:
            scores.append(50)
        
        return sum(scores) / len(scores) if scores else 75

    def _get_performance_grade(self, score: float) -> str:
        """Convert numeric score to letter grade"""
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"

    def _identify_performance_violations(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify performance violations based on thresholds"""
        violations = []
        
        try:
            # Check Core Web Vitals violations
            cwv = metrics.get("core_web_vitals", {})
            for metric in ["lcp", "cls", "fcp"]:
                value = cwv.get(metric)
                if value is not None and metric in self.thresholds["core_web_vitals"]:
                    threshold = self.thresholds["core_web_vitals"][metric]["good"]
                    if value > threshold:
                        violations.append({
                            "category": "Core Web Vitals",
                            "metric": metric.upper(),
                            "value": value,
                            "threshold": threshold,
                            "severity": "high" if value > self.thresholds["core_web_vitals"][metric]["needs_improvement"] else "medium"
                        })
            
            # Check page timing violations
            timing = metrics.get("page_timing", {})
            for metric, threshold in self.thresholds["page_metrics"].items():
                value = timing.get(metric)
                if value is not None and value > threshold:
                    violations.append({
                        "category": "Page Timing",
                        "metric": metric.replace("_", " ").title(),
                        "value": value,
                        "threshold": threshold,
                        "severity": "medium"
                    })
            
            # Check resource violations
            resource = metrics.get("resource_analysis", {})
            if resource.get("total_size_mb", 0) > self.thresholds["resource_metrics"]["total_size_mb"]:
                violations.append({
                    "category": "Resource Loading",
                    "metric": "Total Size",
                    "value": f"{resource['total_size_mb']:.2f} MB",
                    "threshold": f"{self.thresholds['resource_metrics']['total_size_mb']} MB",
                    "severity": "high"
                })
            
            if resource.get("total_resources", 0) > self.thresholds["resource_metrics"]["total_requests"]:
                violations.append({
                    "category": "Resource Loading",
                    "metric": "Request Count",
                    "value": resource["total_resources"],
                    "threshold": self.thresholds["resource_metrics"]["total_requests"],
                    "severity": "medium"
                })
            
        except Exception as e:
            violations.append({
                "category": "Analysis Error",
                "metric": "Violation Detection",
                "error": str(e),
                "severity": "low"
            })
        
        return violations

    def _generate_performance_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """Generate actionable performance recommendations"""
        recommendations = []
        
        try:
            # Core Web Vitals recommendations
            cwv = metrics.get("core_web_vitals", {})
            if cwv.get("lcp", 0) > 2500:
                recommendations.append("Optimize Largest Contentful Paint by reducing server response times, optimizing critical resources, and removing render-blocking resources")
            
            if cwv.get("cls", 0) > 0.1:
                recommendations.append("Improve Cumulative Layout Shift by setting size attributes on images and videos, avoiding dynamically injected content, and using CSS aspect ratios")
            
            if cwv.get("fcp", 0) > 1800:
                recommendations.append("Optimize First Contentful Paint by minimizing render-blocking resources, optimizing fonts, and reducing server response time")
            
            # Resource optimization recommendations
            resource = metrics.get("resource_analysis", {})
            if resource.get("total_size_mb", 0) > 2:
                recommendations.append("Reduce total page size by compressing images, minifying CSS/JS, and enabling gzip/brotli compression")
            
            if resource.get("total_resources", 0) > 50:
                recommendations.append("Reduce number of HTTP requests by combining files, using CSS sprites, and implementing resource bundling")
            
            if resource.get("cache_hit_rate", 0) < 20:
                recommendations.append("Improve caching strategy by setting proper cache headers and implementing service workers for offline capabilities")
            
            # JavaScript performance recommendations
            js = metrics.get("javascript_performance", {})
            if js.get("memory_metrics", {}).get("used_heap_mb", 0) > 50:
                recommendations.append("Optimize JavaScript memory usage by removing unused code, implementing lazy loading, and avoiding memory leaks")
            
            dom_complexity = js.get("dom_metrics", {}).get("total_elements", 0)
            if dom_complexity > 1500:
                recommendations.append("Reduce DOM complexity by minimizing nested elements, removing unused HTML, and implementing virtual scrolling for large lists")
            
            # Network optimization recommendations
            network = metrics.get("network_analysis", {})
            if network.get("efficiency_metrics", {}).get("compression_ratio", 0) < 30:
                recommendations.append("Enable better compression by configuring gzip/brotli on the server and optimizing text-based resources")
            
            # Generic recommendations if no specific issues found
            if not recommendations:
                recommendations.extend([
                    "Consider implementing a Content Delivery Network (CDN) for global performance",
                    "Optimize images by using modern formats (WebP, AVIF) and responsive images",
                    "Implement critical CSS inlining and defer non-critical styles",
                    "Use HTTP/2 or HTTP/3 for improved connection efficiency"
                ])
            
        except Exception:
            recommendations.append("Unable to generate specific recommendations due to analysis error")
        
        return recommendations

    def generate_performance_report(self, results: Dict[str, Any]) -> str:
        """Generate detailed performance report"""
        report = f"""
# Performance Validation Report

## Overall Results
- **Performance Score**: {results['overall_score']:.1f}/100
- **Performance Grade**: {results['performance_grade']}
- **Test**: {results['test_name']}

## Core Web Vitals
"""
        
        cwv = results["metrics"].get("core_web_vitals", {})
        for metric in ["lcp", "fcp", "cls"]:
            value = cwv.get(metric)
            status = cwv.get(f"{metric}_status", "unknown")
            if value is not None:
                unit = "ms" if metric in ["lcp", "fcp"] else ""
                report += f"- **{metric.upper()}**: {value:.1f}{unit} ({status})\n"
        
        report += "\n## Performance Violations\n"
        if results["violations"]:
            for violation in results["violations"]:
                report += f"- **{violation['category']}**: {violation['metric']} = {violation['value']} (threshold: {violation['threshold']})\n"
        else:
            report += "- No performance violations detected\n"
        
        report += "\n## Recommendations\n"
        for i, recommendation in enumerate(results["recommendations"], 1):
            report += f"{i}. {recommendation}\n"
        
        # Add detailed metrics
        report += "\n## Detailed Metrics\n"
        
        # Page timing
        timing = results["metrics"].get("page_timing", {})
        if timing:
            report += f"\n### Page Timing\n"
            report += f"- DOM Content Loaded: {timing.get('dom_content_loaded', 'N/A')}ms\n"
            report += f"- Load Complete: {timing.get('load_complete', 'N/A')}ms\n"
            report += f"- DOM Interactive: {timing.get('dom_interactive', 'N/A')}ms\n"
        
        # Resource analysis
        resource = results["metrics"].get("resource_analysis", {})
        if resource:
            report += f"\n### Resource Analysis\n"
            report += f"- Total Resources: {resource.get('total_resources', 'N/A')}\n"
            report += f"- Total Size: {resource.get('total_size_mb', 0):.2f} MB\n"
            report += f"- Cache Hit Rate: {resource.get('cache_hit_rate', 0):.1f}%\n"
            
            slow_resources = resource.get("slow_resources", [])
            if slow_resources:
                report += f"- Slow Resources ({len(slow_resources)}):\n"
                for res in slow_resources[:5]:  # Show top 5
                    report += f"  - {res['name']}: {res['duration']}ms ({res['size']}KB)\n"
        
        return report