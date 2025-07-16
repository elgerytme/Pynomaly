"""Enhanced Performance testing framework for GitHub Issue #164: Phase 6.1 Integration Testing - End-to-End Validation."""

import pytest
import asyncio
import time
import psutil
import numpy as np
import threading
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import asynccontextmanager
import gc
import json

from pynomaly.domain.entities.detector import Detector
from pynomaly.domain.services.advanced_classification_service import AdvancedClassificationService
from pynomaly.domain.services.threshold_severity_classifier import ThresholdSeverityClassifier
from pynomaly.domain.value_objects import ContaminationRate


@dataclass
class PerformanceBenchmark:
    """Performance benchmark thresholds and criteria."""
    
    max_latency_ms: float = 1000.0
    max_memory_mb: float = 500.0
    max_cpu_percent: float = 80.0
    min_throughput_ops_per_sec: float = 100.0
    max_error_rate_percent: float = 1.0
    max_p95_latency_ms: float = 2000.0
    max_p99_latency_ms: float = 5000.0
    memory_leak_threshold_mb: float = 50.0


@dataclass
class LoadTestScenario:
    """Load testing scenario configuration."""
    
    name: str
    concurrent_users: int
    duration_seconds: int
    ramp_up_seconds: int
    operations_per_user: int
    data_size_per_operation: int
    expected_throughput: float
    failure_tolerance_percent: float = 5.0


@dataclass
class DetailedPerformanceMetrics:
    """Comprehensive performance metrics collection."""
    
    test_name: str
    execution_time_ms: float
    memory_usage_mb: float
    peak_memory_mb: float
    cpu_usage_percent: float
    throughput_ops_per_sec: float
    latency_measurements: List[float] = field(default_factory=list)
    error_count: int = 0
    success_count: int = 0
    gc_collections: int = 0
    cache_hit_rate: Optional[float] = None
    
    @property
    def p50_latency_ms(self) -> float:
        """50th percentile latency."""
        return statistics.median(self.latency_measurements) if self.latency_measurements else 0.0
    
    @property
    def p95_latency_ms(self) -> float:
        """95th percentile latency."""
        return statistics.quantiles(self.latency_measurements, n=20)[18] if len(self.latency_measurements) > 20 else 0.0
    
    @property
    def p99_latency_ms(self) -> float:
        """99th percentile latency."""
        return statistics.quantiles(self.latency_measurements, n=100)[98] if len(self.latency_measurements) > 100 else 0.0
    
    @property
    def error_rate_percent(self) -> float:
        """Error rate percentage."""
        total_operations = self.success_count + self.error_count
        return (self.error_count / total_operations * 100) if total_operations > 0 else 0.0
    
    @property
    def success_rate_percent(self) -> float:
        """Success rate percentage."""
        return 100.0 - self.error_rate_percent


@dataclass
class StressTestResult:
    """Stress testing result analysis."""
    
    breaking_point_users: int
    max_stable_throughput: float
    degradation_threshold_users: int
    memory_stability: bool
    cpu_stability: bool
    error_escalation_point: int


@dataclass
class PerformanceMetrics:
    """Performance metrics data class."""
    
    execution_time: float
    memory_usage_mb: float
    cpu_usage_percent: float
    throughput_ops_per_sec: float
    peak_memory_mb: float
    error_count: int = 0
    success_count: int = 0


@dataclass
class LoadTestConfig:
    """Load testing configuration."""
    
    concurrent_users: int = 10
    duration_seconds: int = 60
    ramp_up_seconds: int = 10
    target_throughput: float = 100.0  # operations per second
    max_response_time_ms: float = 1000.0
    max_error_rate_percent: float = 1.0


class PerformanceMonitor:
    """Performance monitoring utility."""
    
    def __init__(self):
        self.start_time: Optional[float] = None
        self.start_memory: Optional[float] = None
        self.peak_memory: float = 0.0
        self.measurements: List[Dict[str, float]] = []
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None

    def start_monitoring(self, interval_seconds: float = 1.0):
        """Start performance monitoring."""
        self.start_time = time.time()
        process = psutil.Process()
        self.start_memory = process.memory_info().rss / 1024 / 1024  # MB
        self.peak_memory = self.start_memory
        self.monitoring = True
        self.measurements = []
        
        def monitor_loop():
            while self.monitoring:
                try:
                    memory_mb = process.memory_info().rss / 1024 / 1024
                    cpu_percent = process.cpu_percent()
                    self.peak_memory = max(self.peak_memory, memory_mb)
                    
                    self.measurements.append({
                        "timestamp": time.time(),
                        "memory_mb": memory_mb,
                        "cpu_percent": cpu_percent,
                    })
                    
                    time.sleep(interval_seconds)
                except Exception:
                    break
        
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()

    def stop_monitoring(self) -> PerformanceMetrics:
        """Stop monitoring and return metrics."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        
        end_time = time.time()
        execution_time = end_time - (self.start_time or end_time)
        
        if self.measurements:
            avg_memory = sum(m["memory_mb"] for m in self.measurements) / len(self.measurements)
            avg_cpu = sum(m["cpu_percent"] for m in self.measurements) / len(self.measurements)
        else:
            avg_memory = self.start_memory or 0.0
            avg_cpu = 0.0
        
        return PerformanceMetrics(
            execution_time=execution_time,
            memory_usage_mb=avg_memory,
            cpu_usage_percent=avg_cpu,
            throughput_ops_per_sec=0.0,  # To be calculated separately
            peak_memory_mb=self.peak_memory,
        )


class EnhancedPerformanceTestFramework:
    """Comprehensive performance testing framework for GitHub Issue #164."""
    
    def __init__(self, benchmark: Optional[PerformanceBenchmark] = None):
        self.monitor = PerformanceMonitor()
        self.results: List[DetailedPerformanceMetrics] = []
        self.benchmark = benchmark or PerformanceBenchmark()
        self.stress_test_results: List[StressTestResult] = []
        self.load_test_scenarios: List[LoadTestScenario] = []
    
    async def run_comprehensive_performance_suite(
        self,
        test_function: Callable,
        test_name: str,
        scenarios: List[LoadTestScenario],
        **kwargs
    ) -> Dict[str, Any]:
        """Run comprehensive performance test suite including load, stress, and endurance tests."""
        
        suite_results = {
            "test_name": test_name,
            "timestamp": datetime.utcnow().isoformat(),
            "benchmark_criteria": self.benchmark,
            "scenario_results": {},
            "stress_test_results": [],
            "performance_grade": "F",
            "recommendations": []
        }
        
        # Run baseline performance test
        baseline_metrics = await self.run_baseline_performance_test(
            test_function, f"{test_name}_baseline", **kwargs
        )
        suite_results["baseline_performance"] = baseline_metrics
        
        # Run load test scenarios
        for scenario in scenarios:
            scenario_result = await self.run_load_test_scenario(
                test_function, scenario, **kwargs
            )
            suite_results["scenario_results"][scenario.name] = scenario_result
        
        # Run stress testing to find breaking points
        stress_result = await self.run_stress_test(
            test_function, f"{test_name}_stress", **kwargs
        )
        suite_results["stress_test_results"] = stress_result
        
        # Run endurance testing
        endurance_result = await self.run_endurance_test(
            test_function, f"{test_name}_endurance", duration_minutes=10, **kwargs
        )
        suite_results["endurance_test"] = endurance_result
        
        # Generate performance grade and recommendations
        performance_grade = self._calculate_comprehensive_grade(suite_results)
        recommendations = self._generate_performance_recommendations(suite_results)
        
        suite_results["performance_grade"] = performance_grade
        suite_results["recommendations"] = recommendations
        
        return suite_results
    
    async def run_baseline_performance_test(
        self,
        test_function: Callable,
        test_name: str,
        iterations: int = 100,
        warmup_iterations: int = 10,
        **kwargs
    ) -> DetailedPerformanceMetrics:
        """Run baseline performance test to establish performance characteristics."""
        
        print(f"🏁 Running baseline performance test: {test_name}")
        
        # Warmup phase
        for _ in range(warmup_iterations):
            try:
                await test_function(**kwargs)
            except Exception:
                pass
        
        # Force garbage collection before test
        gc.collect()
        initial_gc_count = len(gc.get_objects())
        
        # Start monitoring
        self.monitor.start_monitoring()
        latency_measurements = []
        
        start_time = time.perf_counter()
        success_count = 0
        error_count = 0
        
        for i in range(iterations):
            operation_start = time.perf_counter()
            try:
                await test_function(**kwargs)
                operation_end = time.perf_counter()
                latency_measurements.append((operation_end - operation_start) * 1000)
                success_count += 1
            except Exception as e:
                error_count += 1
                if error_count > iterations * 0.1:  # More than 10% errors
                    print(f"⚠️ High error rate detected: {error_count}/{i+1}")
                    break
        
        end_time = time.perf_counter()
        
        # Stop monitoring and collect metrics
        monitor_metrics = self.monitor.stop_monitoring()
        final_gc_count = len(gc.get_objects())
        
        total_time = end_time - start_time
        throughput = success_count / total_time if total_time > 0 else 0.0
        
        metrics = DetailedPerformanceMetrics(
            test_name=test_name,
            execution_time_ms=total_time * 1000,
            memory_usage_mb=monitor_metrics.memory_usage_mb,
            peak_memory_mb=monitor_metrics.peak_memory_mb,
            cpu_usage_percent=monitor_metrics.cpu_usage_percent,
            throughput_ops_per_sec=throughput,
            latency_measurements=latency_measurements,
            error_count=error_count,
            success_count=success_count,
            gc_collections=final_gc_count - initial_gc_count
        )
        
        self.results.append(metrics)
        self._print_baseline_results(metrics)
        
        return metrics
    
    async def run_load_test_scenario(
        self,
        test_function: Callable,
        scenario: LoadTestScenario,
        **kwargs
    ) -> Dict[str, Any]:
        """Run specific load test scenario with concurrent users."""
        
        print(f"🔥 Running load test scenario: {scenario.name}")
        print(f"   Users: {scenario.concurrent_users}, Duration: {scenario.duration_seconds}s")
        
        scenario_results = {
            "scenario": scenario,
            "start_time": datetime.utcnow(),
            "user_results": [],
            "aggregate_metrics": {},
            "performance_degradation": {}
        }
        
        # Start system monitoring
        self.monitor.start_monitoring()
        
        # Create and start user tasks with ramp-up
        user_tasks = []
        for user_id in range(scenario.concurrent_users):
            # Stagger user start times for ramp-up
            if scenario.ramp_up_seconds > 0:
                delay = (user_id / scenario.concurrent_users) * scenario.ramp_up_seconds
                await asyncio.sleep(delay / scenario.concurrent_users)
            
            task = asyncio.create_task(
                self._run_load_test_user_session(
                    test_function,
                    scenario,
                    user_id,
                    **kwargs
                )
            )
            user_tasks.append(task)
        
        # Wait for all user sessions to complete
        user_results = await asyncio.gather(*user_tasks, return_exceptions=True)
        
        # Stop monitoring
        system_metrics = self.monitor.stop_monitoring()
        
        # Process and analyze results
        successful_results = [r for r in user_results if not isinstance(r, Exception)]
        failed_results = [r for r in user_results if isinstance(r, Exception)]
        
        if successful_results:
            aggregate_metrics = self._calculate_aggregate_load_metrics(
                successful_results, system_metrics, scenario
            )
        else:
            aggregate_metrics = {"error": "All users failed"}
        
        scenario_results.update({
            "user_results": successful_results,
            "failed_users": len(failed_results),
            "aggregate_metrics": aggregate_metrics,
            "end_time": datetime.utcnow(),
            "system_metrics": system_metrics
        })
        
        self._print_load_test_results(scenario, aggregate_metrics)
        
        return scenario_results
    
    async def run_stress_test(
        self,
        test_function: Callable,
        test_name: str,
        max_users: int = 100,
        step_size: int = 10,
        step_duration: int = 30,
        **kwargs
    ) -> StressTestResult:
        """Run stress test to find system breaking points."""
        
        print(f"🔬 Running stress test: {test_name}")
        print(f"   Max users: {max_users}, Step size: {step_size}")
        
        stress_metrics = []
        current_users = step_size
        breaking_point_users = max_users
        degradation_threshold_users = max_users
        
        while current_users <= max_users:
            print(f"   Testing with {current_users} users...")
            
            # Create stress test scenario
            stress_scenario = LoadTestScenario(
                name=f"stress_{current_users}_users",
                concurrent_users=current_users,
                duration_seconds=step_duration,
                ramp_up_seconds=5,
                operations_per_user=10,
                data_size_per_operation=1000,
                expected_throughput=current_users * 2,
                failure_tolerance_percent=10.0
            )
            
            try:
                scenario_result = await self.run_load_test_scenario(
                    test_function, stress_scenario, **kwargs
                )
                
                metrics = scenario_result["aggregate_metrics"]
                stress_metrics.append({
                    "users": current_users,
                    "metrics": metrics,
                    "stable": self._is_performance_stable(metrics)
                })
                
                # Check for performance degradation
                if not self._is_performance_stable(metrics):
                    if degradation_threshold_users == max_users:
                        degradation_threshold_users = current_users
                
                # Check for system failure
                if metrics.get("error_rate_percent", 0) > 20:
                    breaking_point_users = current_users
                    break
                
            except Exception as e:
                print(f"   💥 System failure at {current_users} users: {e}")
                breaking_point_users = current_users
                break
            
            current_users += step_size
        
        # Analyze stress test results
        max_stable_throughput = max(
            (m["metrics"].get("throughput_ops_per_sec", 0) for m in stress_metrics),
            default=0
        )
        
        memory_stability = all(
            m["metrics"].get("peak_memory_mb", 0) < self.benchmark.max_memory_mb
            for m in stress_metrics
        )
        
        cpu_stability = all(
            m["metrics"].get("avg_cpu_percent", 0) < self.benchmark.max_cpu_percent
            for m in stress_metrics
        )
        
        error_escalation_point = next(
            (m["users"] for m in stress_metrics 
             if m["metrics"].get("error_rate_percent", 0) > self.benchmark.max_error_rate_percent),
            max_users
        )
        
        result = StressTestResult(
            breaking_point_users=breaking_point_users,
            max_stable_throughput=max_stable_throughput,
            degradation_threshold_users=degradation_threshold_users,
            memory_stability=memory_stability,
            cpu_stability=cpu_stability,
            error_escalation_point=error_escalation_point
        )
        
        self.stress_test_results.append(result)
        self._print_stress_test_results(result)
        
        return result
    
    async def run_endurance_test(
        self,
        test_function: Callable,
        test_name: str,
        duration_minutes: int = 30,
        user_count: int = 10,
        **kwargs
    ) -> Dict[str, Any]:
        """Run endurance test to detect memory leaks and performance degradation over time."""
        
        print(f"⏰ Running endurance test: {test_name}")
        print(f"   Duration: {duration_minutes} minutes, Users: {user_count}")
        
        endurance_scenario = LoadTestScenario(
            name=f"endurance_{duration_minutes}m",
            concurrent_users=user_count,
            duration_seconds=duration_minutes * 60,
            ramp_up_seconds=30,
            operations_per_user=1000,
            data_size_per_operation=500,
            expected_throughput=user_count * 5,
            failure_tolerance_percent=5.0
        )
        
        # Track metrics over time for trend analysis
        time_series_metrics = []
        start_time = time.time()
        
        result = await self.run_load_test_scenario(
            test_function, endurance_scenario, **kwargs
        )
        
        # Analyze for memory leaks and performance degradation
        memory_trend = self._analyze_memory_trend(result)
        performance_trend = self._analyze_performance_trend(result)
        
        endurance_analysis = {
            "test_name": test_name,
            "duration_minutes": duration_minutes,
            "result": result,
            "memory_trend": memory_trend,
            "performance_trend": performance_trend,
            "memory_leak_detected": memory_trend.get("leak_detected", False),
            "performance_degradation": performance_trend.get("degradation_detected", False)
        }
        
        self._print_endurance_test_results(endurance_analysis)
        
        return endurance_analysis
    
    def _calculate_comprehensive_grade(self, suite_results: Dict[str, Any]) -> str:
        """Calculate comprehensive performance grade based on all test results."""
        
        scores = []
        
        # Baseline performance score (30%)
        baseline = suite_results.get("baseline_performance")
        if baseline:
            baseline_score = self._score_baseline_performance(baseline)
            scores.append(("baseline", baseline_score, 0.3))
        
        # Load test scenarios score (40%)
        scenario_results = suite_results.get("scenario_results", {})
        if scenario_results:
            load_score = self._score_load_test_results(scenario_results)
            scores.append(("load", load_score, 0.4))
        
        # Stress test score (20%)
        stress_results = suite_results.get("stress_test_results")
        if stress_results:
            stress_score = self._score_stress_test_results(stress_results)
            scores.append(("stress", stress_score, 0.2))
        
        # Endurance test score (10%)
        endurance_results = suite_results.get("endurance_test")
        if endurance_results:
            endurance_score = self._score_endurance_test_results(endurance_results)
            scores.append(("endurance", endurance_score, 0.1))
        
        # Calculate weighted average
        if scores:
            weighted_score = sum(score * weight for _, score, weight in scores)
            total_weight = sum(weight for _, _, weight in scores)
            final_score = weighted_score / total_weight if total_weight > 0 else 0
        else:
            final_score = 0
        
        # Convert to letter grade
        if final_score >= 90: return "A"
        elif final_score >= 80: return "B"
        elif final_score >= 70: return "C"
        elif final_score >= 60: return "D"
        else: return "F"

    async def run_performance_test(
        self,
        test_function,
        test_name: str,
        iterations: int = 100,
        warmup_iterations: int = 10,
        **kwargs
    ) -> PerformanceMetrics:
        """Run a performance test with monitoring."""
        print(f"Running performance test: {test_name}")
        
        # Warmup
        for _ in range(warmup_iterations):
            try:
                await test_function(**kwargs)
            except Exception:
                pass
        
        # Start monitoring
        self.monitor.start_monitoring()
        
        # Run test
        start_time = time.time()
        success_count = 0
        error_count = 0
        
        for i in range(iterations):
            try:
                await test_function(**kwargs)
                success_count += 1
            except Exception as e:
                error_count += 1
                if error_count > iterations * 0.1:  # More than 10% errors
                    break
        
        end_time = time.time()
        
        # Stop monitoring
        metrics = self.monitor.stop_monitoring()
        
        # Calculate throughput
        total_time = end_time - start_time
        metrics.throughput_ops_per_sec = success_count / total_time if total_time > 0 else 0.0
        metrics.success_count = success_count
        metrics.error_count = error_count
        
        self.results.append(metrics)
        
        print(f"Test {test_name} completed:")
        print(f"  - Execution time: {metrics.execution_time:.2f}s")
        print(f"  - Throughput: {metrics.throughput_ops_per_sec:.2f} ops/sec")
        print(f"  - Memory usage: {metrics.memory_usage_mb:.2f} MB")
        print(f"  - Success rate: {success_count}/{iterations} ({100*success_count/iterations:.1f}%)")
        
        return metrics

    async def run_load_test(
        self,
        test_function,
        config: LoadTestConfig,
        test_name: str = "load_test",
        **kwargs
    ) -> Dict[str, Any]:
        """Run a load test with multiple concurrent users."""
        print(f"Running load test: {test_name}")
        print(f"Configuration: {config.concurrent_users} users, {config.duration_seconds}s duration")
        
        results = {
            "config": config,
            "start_time": datetime.utcnow(),
            "user_results": [],
            "aggregate_metrics": {},
        }
        
        # Start monitoring
        self.monitor.start_monitoring()
        
        # Create user tasks
        user_tasks = []
        for user_id in range(config.concurrent_users):
            if config.ramp_up_seconds > 0:
                delay = (user_id / config.concurrent_users) * config.ramp_up_seconds
                await asyncio.sleep(delay / config.concurrent_users)
            
            task = asyncio.create_task(
                self._run_user_session(
                    test_function,
                    config.duration_seconds,
                    user_id,
                    **kwargs
                )
            )
            user_tasks.append(task)
        
        # Wait for all users to complete
        user_results = await asyncio.gather(*user_tasks, return_exceptions=True)
        
        # Stop monitoring
        overall_metrics = self.monitor.stop_monitoring()
        
        # Process results
        successful_results = [r for r in user_results if not isinstance(r, Exception)]
        failed_results = [r for r in user_results if isinstance(r, Exception)]
        
        # Calculate aggregate metrics
        if successful_results:
            total_operations = sum(r["operations"] for r in successful_results)
            total_errors = sum(r["errors"] for r in successful_results)
            avg_response_time = sum(r["avg_response_time"] for r in successful_results) / len(successful_results)
            
            aggregate_metrics = {
                "total_users": config.concurrent_users,
                "successful_users": len(successful_results),
                "failed_users": len(failed_results),
                "total_operations": total_operations,
                "total_errors": total_errors,
                "error_rate_percent": (total_errors / total_operations * 100) if total_operations > 0 else 0,
                "avg_response_time_ms": avg_response_time * 1000,
                "throughput_ops_per_sec": total_operations / config.duration_seconds,
                "overall_execution_time": overall_metrics.execution_time,
                "peak_memory_mb": overall_metrics.peak_memory_mb,
                "avg_cpu_percent": overall_metrics.cpu_usage_percent,
            }
        else:
            aggregate_metrics = {"error": "All users failed"}
        
        results["user_results"] = successful_results
        results["aggregate_metrics"] = aggregate_metrics
        results["end_time"] = datetime.utcnow()
        
        print(f"Load test {test_name} completed:")
        print(f"  - Successful users: {aggregate_metrics.get('successful_users', 0)}/{config.concurrent_users}")
        print(f"  - Total operations: {aggregate_metrics.get('total_operations', 0)}")
        print(f"  - Error rate: {aggregate_metrics.get('error_rate_percent', 0):.2f}%")
        print(f"  - Throughput: {aggregate_metrics.get('throughput_ops_per_sec', 0):.2f} ops/sec")
        print(f"  - Avg response time: {aggregate_metrics.get('avg_response_time_ms', 0):.2f} ms")
        
        return results

    async def _run_user_session(
        self,
        test_function,
        duration_seconds: int,
        user_id: int,
        **kwargs
    ) -> Dict[str, Any]:
        """Run a single user session."""
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        operations = 0
        errors = 0
        response_times = []
        
        while time.time() < end_time:
            operation_start = time.time()
            try:
                await test_function(user_id=user_id, **kwargs)
                operations += 1
                response_times.append(time.time() - operation_start)
            except Exception:
                errors += 1
            
            # Small delay to prevent overwhelming the system
            await asyncio.sleep(0.01)
        
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0.0
        
        return {
            "user_id": user_id,
            "operations": operations,
            "errors": errors,
            "avg_response_time": avg_response_time,
            "total_time": time.time() - start_time,
        }

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of all performance test results."""
        if not self.results:
            return {"message": "No performance tests run"}
        
        summary = {
            "total_tests": len(self.results),
            "avg_execution_time": sum(r.execution_time for r in self.results) / len(self.results),
            "avg_memory_usage_mb": sum(r.memory_usage_mb for r in self.results) / len(self.results),
            "avg_cpu_usage_percent": sum(r.cpu_usage_percent for r in self.results) / len(self.results),
            "avg_throughput": sum(r.throughput_ops_per_sec for r in self.results) / len(self.results),
            "peak_memory_mb": max(r.peak_memory_mb for r in self.results),
            "total_success_count": sum(r.success_count for r in self.results),
            "total_error_count": sum(r.error_count for r in self.results),
        }
        
        if summary["total_success_count"] + summary["total_error_count"] > 0:
            summary["overall_success_rate"] = (
                summary["total_success_count"] / 
                (summary["total_success_count"] + summary["total_error_count"]) * 100
            )
        
        return summary


class TestPerformanceFramework:
    """Test suite for the performance framework."""

    @pytest.fixture
    def performance_framework(self):
        """Performance testing framework."""
        return PerformanceTestFramework()

    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        np.random.seed(42)
        return np.random.normal(0, 1, (100, 10))

    @pytest.fixture
    def classification_service(self):
        """Classification service for testing."""
        severity_classifier = ThresholdSeverityClassifier()
        return AdvancedClassificationService(
            severity_classifier=severity_classifier,
            enable_hierarchical=True,
            enable_multiclass=True,
        )

    async def mock_classification_operation(self, classification_service, sample_data, user_id=None):
        """Mock classification operation for performance testing."""
        detector = Detector(
            name=f"perf_test_detector_{user_id or 0}",
            algorithm_name="isolation_forest",
            contamination_rate=ContaminationRate.from_value(0.1),
            parameters={"n_estimators": 50, "random_state": 42},
        )
        
        # Simulate processing a small batch
        data_batch = sample_data[:5]
        anomaly_scores = np.random.uniform(0, 1, len(data_batch))
        
        for i, score in enumerate(anomaly_scores):
            feature_data = {f"feature_{j}": data_batch[i, j] for j in range(data_batch.shape[1])}
            
            classification = classification_service.classify_anomaly(
                anomaly_score=score,
                detector=detector,
                feature_data=feature_data,
                context_data={"timestamp": datetime.utcnow()},
            )
            
            # Simulate some processing
            await asyncio.sleep(0.001)

    @pytest.mark.asyncio
    async def test_classification_performance(
        self, performance_framework, classification_service, sample_data
    ):
        """Test classification performance."""
        metrics = await performance_framework.run_performance_test(
            test_function=self.mock_classification_operation,
            test_name="classification_performance",
            iterations=50,
            warmup_iterations=5,
            classification_service=classification_service,
            sample_data=sample_data,
        )
        
        # Performance assertions
        assert metrics.execution_time < 30.0  # Should complete within 30 seconds
        assert metrics.memory_usage_mb < 500.0  # Should use less than 500MB
        assert metrics.throughput_ops_per_sec > 1.0  # At least 1 operation per second
        assert metrics.error_count == 0  # No errors expected

    @pytest.mark.asyncio
    async def test_load_testing(
        self, performance_framework, classification_service, sample_data
    ):
        """Test load testing capabilities."""
        load_config = LoadTestConfig(
            concurrent_users=5,
            duration_seconds=10,
            ramp_up_seconds=2,
            target_throughput=10.0,
            max_response_time_ms=2000.0,
            max_error_rate_percent=5.0,
        )
        
        results = await performance_framework.run_load_test(
            test_function=self.mock_classification_operation,
            config=load_config,
            test_name="classification_load_test",
            classification_service=classification_service,
            sample_data=sample_data,
        )
        
        # Load test assertions
        aggregate = results["aggregate_metrics"]
        assert aggregate["successful_users"] >= 4  # At least 4/5 users successful
        assert aggregate["error_rate_percent"] <= load_config.max_error_rate_percent
        assert aggregate["avg_response_time_ms"] <= load_config.max_response_time_ms
        assert aggregate["throughput_ops_per_sec"] > 0

    @pytest.mark.asyncio
    async def test_memory_usage_monitoring(
        self, performance_framework, classification_service, sample_data
    ):
        """Test memory usage monitoring."""
        # Run a memory-intensive operation
        async def memory_intensive_operation(**kwargs):
            # Create large data structures
            large_data = np.random.normal(0, 1, (1000, 100))
            
            # Process through classification
            await self.mock_classification_operation(
                classification_service, large_data[:10], **kwargs
            )
            
            # Keep reference to prevent garbage collection
            return large_data
        
        metrics = await performance_framework.run_performance_test(
            test_function=memory_intensive_operation,
            test_name="memory_usage_test",
            iterations=5,
            warmup_iterations=1,
            classification_service=classification_service,
            sample_data=sample_data,
        )
        
        # Memory assertions
        assert metrics.peak_memory_mb > metrics.memory_usage_mb  # Peak should be higher than average
        assert metrics.peak_memory_mb < 1000.0  # Should not exceed 1GB

    @pytest.mark.asyncio
    async def test_performance_regression_detection(
        self, performance_framework, classification_service, sample_data
    ):
        """Test performance regression detection."""
        # Run baseline test
        baseline_metrics = await performance_framework.run_performance_test(
            test_function=self.mock_classification_operation,
            test_name="baseline_performance",
            iterations=20,
            classification_service=classification_service,
            sample_data=sample_data,
        )
        
        # Run comparison test (should be similar)
        comparison_metrics = await performance_framework.run_performance_test(
            test_function=self.mock_classification_operation,
            test_name="comparison_performance",
            iterations=20,
            classification_service=classification_service,
            sample_data=sample_data,
        )
        
        # Regression detection
        throughput_change = abs(
            comparison_metrics.throughput_ops_per_sec - baseline_metrics.throughput_ops_per_sec
        ) / baseline_metrics.throughput_ops_per_sec
        
        memory_change = abs(
            comparison_metrics.memory_usage_mb - baseline_metrics.memory_usage_mb
        ) / baseline_metrics.memory_usage_mb
        
        # Assertions for regression detection
        assert throughput_change < 0.2  # Less than 20% change in throughput
        assert memory_change < 0.3  # Less than 30% change in memory usage

    @pytest.mark.asyncio
    async def test_performance_summary_generation(
        self, performance_framework, classification_service, sample_data
    ):
        """Test performance summary generation."""
        # Run multiple tests
        for i in range(3):
            await performance_framework.run_performance_test(
                test_function=self.mock_classification_operation,
                test_name=f"summary_test_{i}",
                iterations=10,
                classification_service=classification_service,
                sample_data=sample_data,
            )
        
        # Generate summary
        summary = performance_framework.get_performance_summary()
        
        # Summary assertions
        assert summary["total_tests"] == 3
        assert "avg_execution_time" in summary
        assert "avg_memory_usage_mb" in summary
        assert "avg_throughput" in summary
        assert "overall_success_rate" in summary
        assert summary["total_error_count"] == 0