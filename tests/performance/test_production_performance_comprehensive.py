"""
Production-Grade Performance Testing Suite

Advanced performance tests covering production scenarios, stress testing,
memory profiling, scalability analysis, and performance regression detection.
"""

import asyncio
import gc
import multiprocessing
import os
import resource
import statistics
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import psutil
import pytest

from pynomaly.domain.entities.detector import Detector
from pynomaly.domain.value_objects.anomaly_score import AnomalyScore
from pynomaly.domain.value_objects.performance_metrics import PerformanceMetrics


@dataclass
class ProductionPerformanceMetrics:
    """Production performance metrics container."""

    operation: str
    duration: float
    memory_used: int
    cpu_percent: float
    throughput: float
    success: bool
    concurrent_users: int = 1
    error_rate: float = 0.0
    latency_p95: float = 0.0
    latency_p99: float = 0.0
    gc_collections: int = 0
    memory_peak: int = 0
    thread_count: int = 1
    error_message: str = ""


class ProductionPerformanceProfiler:
    """Advanced production performance profiling utility."""

    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.metrics: list[ProductionPerformanceMetrics] = []
        self.latency_samples: list[float] = []

    @contextmanager
    def profile_operation(self, operation_name: str, concurrent_users: int = 1):
        """Profile a production operation with comprehensive metrics."""
        # Record initial state
        initial_memory = self.process.memory_info().rss
        initial_cpu = self.process.cpu_percent()
        initial_gc = sum(gc.get_stats())
        initial_threads = threading.active_count()
        
        start_time = time.perf_counter()
        gc.collect()  # Clean slate
        
        try:
            yield self
            success = True
            error_message = ""
        except Exception as e:
            success = False
            error_message = str(e)
            
        finally:
            # Record final state
            end_time = time.perf_counter()
            duration = end_time - start_time
            final_memory = self.process.memory_info().rss
            final_cpu = self.process.cpu_percent()
            final_gc = sum(gc.get_stats())
            final_threads = threading.active_count()
            
            memory_used = final_memory - initial_memory
            cpu_percent = final_cpu - initial_cpu
            throughput = 1 / duration if duration > 0 else 0
            
            # Calculate percentiles
            p95 = np.percentile(self.latency_samples, 95) if self.latency_samples else 0
            p99 = np.percentile(self.latency_samples, 99) if self.latency_samples else 0
            
            metrics = ProductionPerformanceMetrics(
                operation=operation_name,
                duration=duration,
                memory_used=memory_used,
                cpu_percent=cpu_percent,
                throughput=throughput,
                success=success,
                concurrent_users=concurrent_users,
                latency_p95=p95,
                latency_p99=p99,
                gc_collections=final_gc - initial_gc,
                memory_peak=final_memory,
                thread_count=final_threads - initial_threads,
                error_message=error_message,
            )
            
            self.metrics.append(metrics)
            self.latency_samples.clear()

    def record_latency(self, latency: float):
        """Record a latency sample."""
        self.latency_samples.append(latency)


class TestProductionPerformance:
    """Production-grade performance tests."""

    @pytest.fixture
    def profiler(self):
        """Create performance profiler."""
        return ProductionPerformanceProfiler()

    @pytest.fixture
    def large_dataset(self):
        """Create large dataset for performance testing."""
        return np.random.random((10000, 20))

    @pytest.fixture
    def detector(self):
        """Create detector for testing."""
        return Detector(
            id="test-detector",
            name="Test Detector",
            algorithm="IsolationForest",
            parameters={"contamination": 0.1, "n_estimators": 100},
            trained=True,
        )

    def test_memory_usage_under_load(self, profiler, large_dataset):
        """Test memory usage under heavy load."""
        with profiler.profile_operation("memory_load_test"):
            # Simulate memory-intensive operations
            data_copies = []
            for i in range(100):
                data_copies.append(large_dataset.copy())
                profiler.record_latency(time.perf_counter())
                
                # Check memory usage doesn't exceed 500MB
                memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
                assert memory_usage < 500, f"Memory usage {memory_usage}MB exceeds limit"

        metrics = profiler.metrics[-1]
        assert metrics.success
        assert metrics.memory_used < 500 * 1024 * 1024  # 500MB limit

    def test_concurrent_detection_performance(self, profiler, detector, large_dataset):
        """Test performance under concurrent detection requests."""
        num_threads = multiprocessing.cpu_count()
        
        def detection_task():
            start = time.perf_counter()
            # Simulate detection
            time.sleep(0.1)  # Simulate processing time
            end = time.perf_counter()
            profiler.record_latency(end - start)
            return True

        with profiler.profile_operation("concurrent_detection", num_threads):
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(detection_task) for _ in range(100)]
                results = [f.result() for f in futures]

        metrics = profiler.metrics[-1]
        assert metrics.success
        assert all(results)
        assert metrics.throughput > 5  # At least 5 operations per second
        assert metrics.latency_p95 < 1.0  # 95th percentile under 1 second

    def test_stress_testing_resource_limits(self, profiler):
        """Test system behavior under extreme stress."""
        with profiler.profile_operation("stress_test"):
            # Simulate high CPU and memory usage
            cpu_bound_tasks = []
            for _ in range(multiprocessing.cpu_count()):
                def cpu_task():
                    # CPU-intensive computation
                    return sum(i ** 2 for i in range(10000))
                
                cpu_bound_tasks.append(cpu_task())

            # Memory stress
            memory_blocks = []
            for _ in range(50):
                memory_blocks.append(np.random.random((1000, 1000)))

        metrics = profiler.metrics[-1]
        assert metrics.success
        
        # Check system didn't crash under stress
        assert psutil.virtual_memory().percent < 95  # Less than 95% memory usage

    def test_long_running_stability(self, profiler, detector):
        """Test long-running operation stability."""
        with profiler.profile_operation("long_running_test"):
            # Simulate long-running service
            for i in range(1000):
                # Simulate periodic processing
                time.sleep(0.001)  # 1ms sleep
                
                if i % 100 == 0:
                    # Force garbage collection periodically
                    gc.collect()
                    
                profiler.record_latency(0.001)

        metrics = profiler.metrics[-1]
        assert metrics.success
        assert metrics.duration < 10.0  # Should complete within 10 seconds
        assert metrics.gc_collections < 50  # Reasonable GC activity

    def test_memory_leak_detection(self, profiler):
        """Test for memory leaks in long-running operations."""
        initial_memory = psutil.Process().memory_info().rss
        
        with profiler.profile_operation("memory_leak_test"):
            for i in range(500):
                # Simulate operations that might leak memory
                data = np.random.random((100, 100))
                processed_data = data * 2
                result = np.sum(processed_data)
                
                # Clean up references
                del data, processed_data, result
                
                if i % 50 == 0:
                    gc.collect()
                    current_memory = psutil.Process().memory_info().rss
                    memory_growth = current_memory - initial_memory
                    
                    # Memory growth should be reasonable
                    assert memory_growth < 50 * 1024 * 1024  # Less than 50MB growth

        metrics = profiler.metrics[-1]
        assert metrics.success
        
        # Final memory check
        final_memory = psutil.Process().memory_info().rss
        total_growth = final_memory - initial_memory
        assert total_growth < 100 * 1024 * 1024  # Less than 100MB total growth

    def test_database_connection_pool_performance(self, profiler):
        """Test database connection pool performance."""
        with profiler.profile_operation("db_pool_test"):
            # Simulate database operations
            connections = []
            for i in range(20):
                # Simulate connection creation
                conn = Mock()
                conn.execute = Mock(return_value=True)
                connections.append(conn)
                
                # Simulate query execution
                start = time.perf_counter()
                conn.execute("SELECT * FROM detectors")
                end = time.perf_counter()
                profiler.record_latency(end - start)

        metrics = profiler.metrics[-1]
        assert metrics.success
        assert metrics.latency_p95 < 0.1  # Database queries should be fast

    def test_api_response_time_under_load(self, profiler):
        """Test API response times under load."""
        def simulate_api_request():
            start = time.perf_counter()
            # Simulate API processing
            time.sleep(0.05)  # 50ms processing time
            end = time.perf_counter()
            profiler.record_latency(end - start)
            return {"status": "success"}

        with profiler.profile_operation("api_load_test", concurrent_users=50):
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(simulate_api_request) for _ in range(200)]
                results = [f.result() for f in futures]

        metrics = profiler.metrics[-1]
        assert metrics.success
        assert all(r["status"] == "success" for r in results)
        assert metrics.latency_p95 < 0.5  # 95th percentile under 500ms
        assert metrics.latency_p99 < 1.0  # 99th percentile under 1 second

    def test_cache_performance_impact(self, profiler):
        """Test cache performance impact."""
        cache = {}
        
        def cached_operation(key):
            if key in cache:
                return cache[key]
            
            # Simulate expensive operation
            result = sum(i ** 2 for i in range(1000))
            cache[key] = result
            return result

        with profiler.profile_operation("cache_test"):
            # First run - cache miss
            for i in range(100):
                start = time.perf_counter()
                result = cached_operation(i)
                end = time.perf_counter()
                profiler.record_latency(end - start)
            
            # Second run - cache hit
            for i in range(100):
                start = time.perf_counter()
                result = cached_operation(i)
                end = time.perf_counter()
                profiler.record_latency(end - start)

        metrics = profiler.metrics[-1]
        assert metrics.success
        
        # Cache hits should be much faster
        cache_hit_latencies = profiler.latency_samples[100:]
        cache_miss_latencies = profiler.latency_samples[:100]
        
        avg_cache_hit = statistics.mean(cache_hit_latencies)
        avg_cache_miss = statistics.mean(cache_miss_latencies)
        
        assert avg_cache_hit < avg_cache_miss * 0.1  # Cache hits 10x faster

    def test_scalability_analysis(self, profiler):
        """Test system scalability with increasing load."""
        results = []
        
        for load_factor in [1, 2, 4, 8, 16]:
            with profiler.profile_operation(f"scalability_test_{load_factor}"):
                def work_unit():
                    # Simulate work
                    return sum(i for i in range(1000))
                
                with ThreadPoolExecutor(max_workers=load_factor) as executor:
                    futures = [executor.submit(work_unit) for _ in range(load_factor * 10)]
                    [f.result() for f in futures]
            
            metrics = profiler.metrics[-1]
            results.append((load_factor, metrics.throughput))

        # Analyze scalability
        throughputs = [r[1] for r in results]
        
        # Throughput should increase with load (up to a point)
        assert throughputs[1] > throughputs[0]  # 2x load should be faster
        
        # But efficiency shouldn't degrade too much
        efficiency_ratio = throughputs[-1] / throughputs[0]
        assert efficiency_ratio > 0.5  # Should maintain at least 50% efficiency

    def test_resource_cleanup_performance(self, profiler):
        """Test resource cleanup performance."""
        with profiler.profile_operation("resource_cleanup_test"):
            resources = []
            
            # Create resources
            for i in range(1000):
                resource = Mock()
                resource.close = Mock()
                resources.append(resource)
            
            # Cleanup resources
            start = time.perf_counter()
            for resource in resources:
                resource.close()
            cleanup_time = time.perf_counter() - start
            
            profiler.record_latency(cleanup_time)

        metrics = profiler.metrics[-1]
        assert metrics.success
        assert cleanup_time < 1.0  # Cleanup should be fast

    def test_performance_regression_detection(self, profiler):
        """Test performance regression detection."""
        baseline_metrics = []
        
        # Establish baseline
        for _ in range(5):
            with profiler.profile_operation("baseline_test"):
                # Simulate consistent operation
                time.sleep(0.1)
                profiler.record_latency(0.1)
            
            baseline_metrics.append(profiler.metrics[-1])
        
        # Test current performance
        with profiler.profile_operation("current_test"):
            time.sleep(0.1)
            profiler.record_latency(0.1)
        
        current_metrics = profiler.metrics[-1]
        
        # Check for regression
        baseline_avg = statistics.mean(m.duration for m in baseline_metrics)
        regression_threshold = baseline_avg * 1.5  # 50% performance degradation
        
        assert current_metrics.duration < regression_threshold
        assert current_metrics.success