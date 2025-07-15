"""
Performance Regression Testing Framework for CI/CD Pipeline.

This module provides comprehensive performance regression detection with automated
baseline management, trend analysis, and intelligent alerting for the Pynomaly system.
"""

from __future__ import annotations

import asyncio
import json
import logging
import statistics
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import numpy as np
import pandas as pd
import psutil
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Individual performance metric data point."""
    
    name: str
    value: float
    unit: str
    timestamp: datetime
    context: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "timestamp": self.timestamp.isoformat(),
            "context": self.context,
            "tags": self.tags
        }


@dataclass
class PerformanceBaseline:
    """Performance baseline for regression detection."""
    
    metric_name: str
    mean: float
    std: float
    p50: float
    p95: float
    p99: float
    sample_size: int
    established_at: datetime
    environment: Dict[str, Any] = field(default_factory=dict)
    
    def is_regression(self, value: float, threshold_std: float = 2.0) -> bool:
        """Check if value represents a performance regression."""
        return value > self.mean + (threshold_std * self.std)
    
    def is_improvement(self, value: float, threshold_std: float = 2.0) -> bool:
        """Check if value represents a performance improvement."""
        return value < self.mean - (threshold_std * self.std)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "metric_name": self.metric_name,
            "mean": self.mean,
            "std": self.std,
            "p50": self.p50,
            "p95": self.p95,
            "p99": self.p99,
            "sample_size": self.sample_size,
            "established_at": self.established_at.isoformat(),
            "environment": self.environment
        }


@dataclass
class RegressionResult:
    """Result of regression analysis."""
    
    metric_name: str
    current_value: float
    baseline_mean: float
    deviation_std: float
    is_regression: bool
    is_improvement: bool
    confidence: float
    severity: str  # 'low', 'medium', 'high', 'critical'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "metric_name": self.metric_name,
            "current_value": self.current_value,
            "baseline_mean": self.baseline_mean,
            "deviation_std": self.deviation_std,
            "is_regression": self.is_regression,
            "is_improvement": self.is_improvement,
            "confidence": self.confidence,
            "severity": self.severity
        }


class PerformanceTest(ABC):
    """Abstract base class for performance tests."""
    
    def __init__(self, name: str, tags: List[str] = None):
        self.name = name
        self.tags = tags or []
        self.metrics: List[PerformanceMetric] = []
    
    @abstractmethod
    async def run(self) -> List[PerformanceMetric]:
        """Execute the performance test and return metrics."""
        pass
    
    def add_metric(self, name: str, value: float, unit: str, **context) -> None:
        """Add a performance metric."""
        metric = PerformanceMetric(
            name=name,
            value=value,
            unit=unit,
            timestamp=datetime.now(),
            context=context,
            tags=self.tags
        )
        self.metrics.append(metric)


class APIPerformanceTest(PerformanceTest):
    """Performance test for API endpoints."""
    
    def __init__(self, name: str, endpoint: str, method: str = "GET", 
                 payload: Dict = None, concurrent_users: int = 1,
                 duration_seconds: int = 10):
        super().__init__(name, tags=["api", "endpoint"])
        self.endpoint = endpoint
        self.method = method
        self.payload = payload or {}
        self.concurrent_users = concurrent_users
        self.duration_seconds = duration_seconds
    
    async def run(self) -> List[PerformanceMetric]:
        """Run API performance test."""
        import aiohttp
        
        response_times = []
        error_count = 0
        success_count = 0
        
        start_time = time.time()
        end_time = start_time + self.duration_seconds
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            
            while time.time() < end_time:
                for _ in range(self.concurrent_users):
                    task = self._make_request(session)
                    tasks.append(task)
                
                # Process batch of requests
                if len(tasks) >= 10:
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    for result in results:
                        if isinstance(result, Exception):
                            error_count += 1
                        else:
                            response_times.append(result)
                            success_count += 1
                    
                    tasks.clear()
                
                # Small delay to prevent overwhelming
                await asyncio.sleep(0.1)
            
            # Process remaining tasks
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for result in results:
                    if isinstance(result, Exception):
                        error_count += 1
                    else:
                        response_times.append(result)
                        success_count += 1
        
        # Calculate metrics
        if response_times:
            self.add_metric("response_time_mean", statistics.mean(response_times), "ms")
            self.add_metric("response_time_p50", statistics.median(response_times), "ms")
            self.add_metric("response_time_p95", np.percentile(response_times, 95), "ms")
            self.add_metric("response_time_p99", np.percentile(response_times, 99), "ms")
        
        total_requests = success_count + error_count
        if total_requests > 0:
            self.add_metric("error_rate", (error_count / total_requests) * 100, "percent")
            self.add_metric("throughput", total_requests / self.duration_seconds, "rps")
        
        self.add_metric("total_requests", total_requests, "count")
        self.add_metric("success_count", success_count, "count")
        self.add_metric("error_count", error_count, "count")
        
        return self.metrics
    
    async def _make_request(self, session: aiohttp.ClientSession) -> float:
        """Make individual HTTP request and return response time."""
        start = time.time()
        
        try:
            if self.method.upper() == "GET":
                async with session.get(self.endpoint) as response:
                    await response.text()
            elif self.method.upper() == "POST":
                async with session.post(self.endpoint, json=self.payload) as response:
                    await response.text()
            else:
                raise ValueError(f"Unsupported HTTP method: {self.method}")
            
            return (time.time() - start) * 1000  # Convert to milliseconds
            
        except Exception as e:
            logger.warning(f"Request failed: {e}")
            raise


class DatabasePerformanceTest(PerformanceTest):
    """Performance test for database operations."""
    
    def __init__(self, name: str, query: str, iterations: int = 100):
        super().__init__(name, tags=["database", "query"])
        self.query = query
        self.iterations = iterations
    
    async def run(self) -> List[PerformanceMetric]:
        """Run database performance test."""
        # Mock database operations - replace with actual DB connection
        execution_times = []
        
        for _ in range(self.iterations):
            start = time.time()
            
            # Simulate database query execution
            await asyncio.sleep(0.001 + np.random.exponential(0.002))
            
            execution_time = (time.time() - start) * 1000
            execution_times.append(execution_time)
        
        # Calculate metrics
        self.add_metric("query_time_mean", statistics.mean(execution_times), "ms")
        self.add_metric("query_time_p50", statistics.median(execution_times), "ms")
        self.add_metric("query_time_p95", np.percentile(execution_times, 95), "ms")
        self.add_metric("query_time_p99", np.percentile(execution_times, 99), "ms")
        self.add_metric("query_iterations", self.iterations, "count")
        
        return self.metrics


class SystemResourceTest(PerformanceTest):
    """Test system resource utilization."""
    
    def __init__(self, name: str, duration_seconds: int = 10):
        super().__init__(name, tags=["system", "resources"])
        self.duration_seconds = duration_seconds
    
    async def run(self) -> List[PerformanceMetric]:
        """Monitor system resources during test duration."""
        cpu_samples = []
        memory_samples = []
        
        start_time = time.time()
        
        while time.time() - start_time < self.duration_seconds:
            cpu_percent = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory()
            
            cpu_samples.append(cpu_percent)
            memory_samples.append(memory.percent)
            
            await asyncio.sleep(1)
        
        # Calculate metrics
        self.add_metric("cpu_usage_mean", statistics.mean(cpu_samples), "percent")
        self.add_metric("cpu_usage_max", max(cpu_samples), "percent")
        self.add_metric("memory_usage_mean", statistics.mean(memory_samples), "percent")
        self.add_metric("memory_usage_max", max(memory_samples), "percent")
        
        return self.metrics


class BaselineManager:
    """Manages performance baselines for regression detection."""
    
    def __init__(self, storage_path: Path):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.baselines: Dict[str, PerformanceBaseline] = {}
        self.load_baselines()
    
    def load_baselines(self) -> None:
        """Load existing baselines from storage."""
        baseline_file = self.storage_path / "baselines.json"
        
        if baseline_file.exists():
            try:
                with open(baseline_file, 'r') as f:
                    data = json.load(f)
                
                for baseline_data in data.get('baselines', []):
                    baseline = PerformanceBaseline(
                        metric_name=baseline_data['metric_name'],
                        mean=baseline_data['mean'],
                        std=baseline_data['std'],
                        p50=baseline_data['p50'],
                        p95=baseline_data['p95'],
                        p99=baseline_data['p99'],
                        sample_size=baseline_data['sample_size'],
                        established_at=datetime.fromisoformat(baseline_data['established_at']),
                        environment=baseline_data.get('environment', {})
                    )
                    self.baselines[baseline.metric_name] = baseline
                    
            except Exception as e:
                logger.error(f"Failed to load baselines: {e}")
    
    def save_baselines(self) -> None:
        """Save baselines to storage."""
        baseline_file = self.storage_path / "baselines.json"
        
        data = {
            "baselines": [baseline.to_dict() for baseline in self.baselines.values()],
            "updated_at": datetime.now().isoformat()
        }
        
        with open(baseline_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def establish_baseline(self, metric_name: str, values: List[float], 
                          environment: Dict[str, Any] = None) -> PerformanceBaseline:
        """Establish a new baseline from historical data."""
        if len(values) < 10:
            raise ValueError("Need at least 10 data points to establish baseline")
        
        values = np.array(values)
        
        baseline = PerformanceBaseline(
            metric_name=metric_name,
            mean=float(np.mean(values)),
            std=float(np.std(values)),
            p50=float(np.percentile(values, 50)),
            p95=float(np.percentile(values, 95)),
            p99=float(np.percentile(values, 99)),
            sample_size=len(values),
            established_at=datetime.now(),
            environment=environment or {}
        )
        
        self.baselines[metric_name] = baseline
        self.save_baselines()
        
        logger.info(f"Established baseline for {metric_name}: mean={baseline.mean:.2f}, std={baseline.std:.2f}")
        return baseline
    
    def update_baseline(self, metric_name: str, new_values: List[float]) -> None:
        """Update existing baseline with new data points."""
        if metric_name not in self.baselines:
            raise ValueError(f"No baseline exists for metric: {metric_name}")
        
        baseline = self.baselines[metric_name]
        
        # Use exponential moving average for baseline updates
        alpha = 0.1  # Smoothing factor
        
        for value in new_values:
            baseline.mean = alpha * value + (1 - alpha) * baseline.mean
        
        # Recalculate standard deviation with recent data
        recent_window = 50  # Consider last 50 values
        if len(new_values) >= recent_window:
            baseline.std = float(np.std(new_values[-recent_window:]))
        
        self.save_baselines()


class RegressionDetector:
    """Detects performance regressions using statistical analysis."""
    
    def __init__(self, baseline_manager: BaselineManager):
        self.baseline_manager = baseline_manager
    
    def analyze(self, metrics: List[PerformanceMetric]) -> List[RegressionResult]:
        """Analyze metrics for regressions and improvements."""
        results = []
        
        for metric in metrics:
            if metric.name not in self.baseline_manager.baselines:
                logger.warning(f"No baseline for metric: {metric.name}")
                continue
            
            baseline = self.baseline_manager.baselines[metric.name]
            result = self._analyze_metric(metric, baseline)
            results.append(result)
        
        return results
    
    def _analyze_metric(self, metric: PerformanceMetric, 
                       baseline: PerformanceBaseline) -> RegressionResult:
        """Analyze individual metric against baseline."""
        deviation = (metric.value - baseline.mean) / baseline.std if baseline.std > 0 else 0
        
        # Determine regression/improvement with confidence levels
        is_regression = baseline.is_regression(metric.value, threshold_std=2.0)
        is_improvement = baseline.is_improvement(metric.value, threshold_std=2.0)
        
        # Calculate confidence using statistical significance
        confidence = min(abs(deviation) / 2.0, 1.0)  # Normalize to 0-1
        
        # Determine severity
        severity = self._calculate_severity(deviation, confidence)
        
        return RegressionResult(
            metric_name=metric.name,
            current_value=metric.value,
            baseline_mean=baseline.mean,
            deviation_std=deviation,
            is_regression=is_regression,
            is_improvement=is_improvement,
            confidence=confidence,
            severity=severity
        )
    
    def _calculate_severity(self, deviation: float, confidence: float) -> str:
        """Calculate regression severity based on deviation and confidence."""
        abs_deviation = abs(deviation)
        
        if abs_deviation >= 4.0 and confidence >= 0.9:
            return "critical"
        elif abs_deviation >= 3.0 and confidence >= 0.8:
            return "high"
        elif abs_deviation >= 2.0 and confidence >= 0.6:
            return "medium"
        else:
            return "low"


class PerformanceRegressionFramework:
    """Main framework for performance regression testing."""
    
    def __init__(self, baseline_storage_path: str = "performance_baselines"):
        self.baseline_manager = BaselineManager(Path(baseline_storage_path))
        self.regression_detector = RegressionDetector(self.baseline_manager)
        self.tests: List[PerformanceTest] = []
        self.results_history: List[Dict[str, Any]] = []
    
    def add_test(self, test: PerformanceTest) -> None:
        """Add a performance test to the framework."""
        self.tests.append(test)
    
    async def run_tests(self) -> Dict[str, Any]:
        """Execute all performance tests and analyze results."""
        logger.info(f"Running {len(self.tests)} performance tests...")
        
        start_time = time.time()
        all_metrics = []
        test_results = {}
        
        for test in self.tests:
            logger.info(f"Running test: {test.name}")
            
            try:
                metrics = await test.run()
                all_metrics.extend(metrics)
                test_results[test.name] = {
                    "status": "success",
                    "metrics": [m.to_dict() for m in metrics]
                }
                
            except Exception as e:
                logger.error(f"Test {test.name} failed: {e}")
                test_results[test.name] = {
                    "status": "failed",
                    "error": str(e)
                }
        
        # Analyze for regressions
        regression_results = self.regression_detector.analyze(all_metrics)
        
        # Prepare final results
        results = {
            "run_id": str(uuid4()),
            "timestamp": datetime.now().isoformat(),
            "duration_seconds": time.time() - start_time,
            "total_tests": len(self.tests),
            "successful_tests": len([r for r in test_results.values() if r["status"] == "success"]),
            "total_metrics": len(all_metrics),
            "test_results": test_results,
            "regression_analysis": {
                "total_regressions": len([r for r in regression_results if r.is_regression]),
                "total_improvements": len([r for r in regression_results if r.is_improvement]),
                "critical_regressions": len([r for r in regression_results if r.severity == "critical"]),
                "results": [r.to_dict() for r in regression_results]
            },
            "environment": {
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
                "cpu_count": psutil.cpu_count(),
                "memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
                "platform": psutil.platform()
            }
        }
        
        self.results_history.append(results)
        return results
    
    def establish_baselines_from_history(self, metric_names: List[str] = None) -> None:
        """Establish baselines from historical test results."""
        if not self.results_history:
            logger.warning("No historical data available for baseline establishment")
            return
        
        # Collect historical metric values
        metric_values = {}
        
        for result in self.results_history:
            for test_name, test_data in result.get("test_results", {}).items():
                if test_data.get("status") != "success":
                    continue
                
                for metric_data in test_data.get("metrics", []):
                    metric_name = metric_data["name"]
                    
                    if metric_names and metric_name not in metric_names:
                        continue
                    
                    if metric_name not in metric_values:
                        metric_values[metric_name] = []
                    
                    metric_values[metric_name].append(metric_data["value"])
        
        # Establish baselines
        for metric_name, values in metric_values.items():
            if len(values) >= 10:  # Minimum required for baseline
                try:
                    self.baseline_manager.establish_baseline(
                        metric_name, 
                        values,
                        environment={"source": "historical_data"}
                    )
                except Exception as e:
                    logger.error(f"Failed to establish baseline for {metric_name}: {e}")
    
    def get_regression_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a summary of regression analysis results."""
        regression_data = results.get("regression_analysis", {})
        regression_results = regression_data.get("results", [])
        
        # Group by severity
        by_severity = {"critical": [], "high": [], "medium": [], "low": []}
        for result in regression_results:
            if result["is_regression"]:
                by_severity[result["severity"]].append(result)
        
        # Calculate impact scores
        total_regressions = len([r for r in regression_results if r["is_regression"]])
        total_improvements = len([r for r in regression_results if r["is_improvement"]])
        
        return {
            "total_regressions": total_regressions,
            "total_improvements": total_improvements,
            "regressions_by_severity": {k: len(v) for k, v in by_severity.items()},
            "has_critical_regressions": len(by_severity["critical"]) > 0,
            "regression_rate": (total_regressions / len(regression_results) * 100) if regression_results else 0,
            "overall_status": self._determine_overall_status(by_severity)
        }
    
    def _determine_overall_status(self, by_severity: Dict[str, List]) -> str:
        """Determine overall regression test status."""
        if by_severity["critical"]:
            return "CRITICAL"
        elif by_severity["high"]:
            return "FAILED"
        elif by_severity["medium"]:
            return "WARNING"
        else:
            return "PASSED"


# Example usage and test setup
async def create_example_tests() -> List[PerformanceTest]:
    """Create example performance tests for the framework."""
    tests = [
        APIPerformanceTest(
            name="health_endpoint_test",
            endpoint="http://localhost:8000/health",
            concurrent_users=5,
            duration_seconds=10
        ),
        APIPerformanceTest(
            name="dashboard_api_test", 
            endpoint="http://localhost:8000/api/v1/dashboard",
            concurrent_users=3,
            duration_seconds=15
        ),
        DatabasePerformanceTest(
            name="user_query_test",
            query="SELECT * FROM users WHERE active = true",
            iterations=100
        ),
        SystemResourceTest(
            name="system_resource_test",
            duration_seconds=10
        )
    ]
    
    return tests


if __name__ == "__main__":
    import sys
    
    async def main():
        """Example usage of the performance regression framework."""
        framework = PerformanceRegressionFramework()
        
        # Add example tests
        tests = await create_example_tests()
        for test in tests:
            framework.add_test(test)
        
        # Run tests
        results = await framework.run_tests()
        
        # Print summary
        summary = framework.get_regression_summary(results)
        print(f"Performance Test Results:")
        print(f"Status: {summary['overall_status']}")
        print(f"Total Regressions: {summary['total_regressions']}")
        print(f"Total Improvements: {summary['total_improvements']}")
        print(f"Critical Regressions: {summary['regressions_by_severity']['critical']}")
        
        # Save results
        results_file = Path("performance_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {results_file}")
        
        # Exit with error code if critical regressions found
        if summary["has_critical_regressions"]:
            sys.exit(1)
    
    asyncio.run(main())