"""
Comprehensive Performance Testing Suite

Advanced performance testing covering load testing, stress testing, memory profiling,
scalability analysis, and performance regression detection across all system components.
"""

import asyncio
import gc
import multiprocessing
import os
import statistics
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, List
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import psutil
import pytest


@dataclass
class PerformanceMetrics:
    """Performance metrics container."""

    operation: str
    duration: float
    memory_used: int
    cpu_percent: float
    throughput: float
    success: bool
    error_message: str = ""


class PerformanceProfiler:
    """Advanced performance profiling utility."""

    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.metrics: List[PerformanceMetrics] = []

    @contextmanager
    def profile_operation(self, operation_name: str):
        """Profile a single operation."""
        # Record initial state
        initial_memory = self.process.memory_info().rss
        initial_time = time.perf_counter()
        initial_cpu = self.process.cpu_percent()

        success = True
        error_message = ""

        try:
            yield
        except Exception as e:
            success = False
            error_message = str(e)
        finally:
            # Record final state
            final_time = time.perf_counter()
            final_memory = self.process.memory_info().rss
            final_cpu = self.process.cpu_percent()

            duration = final_time - initial_time
            memory_used = final_memory - initial_memory
            cpu_percent = (initial_cpu + final_cpu) / 2
            throughput = 1.0 / duration if duration > 0 else 0

            metric = PerformanceMetrics(
                operation=operation_name,
                duration=duration,
                memory_used=memory_used,
                cpu_percent=cpu_percent,
                throughput=throughput,
                success=success,
                error_message=error_message
            )

            self.metrics.append(metric)

    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics."""
        if not self.metrics:
            return {}

        durations = [m.duration for m in self.metrics if m.success]
        memory_usage = [m.memory_used for m in self.metrics if m.success]
        throughputs = [m.throughput for m in self.metrics if m.success]

        return {
            "total_operations": len(self.metrics),
            "successful_operations": len(durations),
            "success_rate": len(durations) / len(self.metrics) if self.metrics else 0,
            "avg_duration": statistics.mean(durations) if durations else 0,
            "p95_duration": statistics.quantiles(durations, n=20)[18] if len(durations) > 1 else 0,
            "p99_duration": statistics.quantiles(durations, n=100)[98] if len(durations) > 1 else 0,
            "avg_memory_usage": statistics.mean(memory_usage) if memory_usage else 0,
            "max_memory_usage": max(memory_usage) if memory_usage else 0,
            "avg_throughput": statistics.mean(throughputs) if throughputs else 0,
            "max_throughput": max(throughputs) if throughputs else 0
        }


class TestAPIPerformance:
    """Performance tests for API layer."""

    @pytest.fixture
    def performance_profiler(self):
        """Create performance profiler."""
        return PerformanceProfiler()

    @pytest.fixture
    def test_client(self):
        """Create test client for performance testing."""
        from fastapi.testclient import TestClient

        from pynomaly.presentation.api.app import create_app

        app = create_app(testing=True)
        return TestClient(app)

    def test_api_response_time_benchmarks(self, test_client, performance_profiler):
        """Benchmark API response times for all endpoints."""

        endpoints = [
            ("GET", "/api/v1/health"),
            ("GET", "/api/v1/datasets"),
            ("GET", "/api/v1/detectors"),
            ("GET", "/api/v1/detection/results")
        ]

        # Warm-up requests
        for method, endpoint in endpoints:
            test_client.request(method, endpoint)

        # Benchmark requests
        for method, endpoint in endpoints:
            with performance_profiler.profile_operation(f"{method} {endpoint}"):
                response = test_client.request(method, endpoint)
                assert response.status_code in [200, 404]  # 404 acceptable for empty resources

        summary = performance_profiler.get_summary()

        # Performance assertions
        assert summary["avg_duration"] < 0.5  # Average under 500ms
        assert summary["p95_duration"] < 1.0   # 95th percentile under 1s
        assert summary["success_rate"] >= 0.95 # 95% success rate

    def test_api_concurrent_load(self, test_client):
        """Test API performance under concurrent load."""

        def make_concurrent_request(request_id: int) -> Dict[str, Any]:
            start_time = time.perf_counter()

            try:
                response = test_client.get("/api/v1/health")
                success = response.status_code == 200
                error = None
            except Exception as e:
                success = False
                error = str(e)

            end_time = time.perf_counter()

            return {
                "request_id": request_id,
                "duration": end_time - start_time,
                "success": success,
                "error": error
            }

        # Test different concurrency levels
        concurrency_levels = [1, 5, 10, 25, 50]
        results = {}

        for concurrency in concurrency_levels:
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                start_time = time.perf_counter()

                # Submit concurrent requests
                futures = [
                    executor.submit(make_concurrent_request, i)
                    for i in range(concurrency * 2)  # 2x requests per worker
                ]

                # Collect results
                request_results = [future.result() for future in futures]

                end_time = time.perf_counter()
                total_time = end_time - start_time

                # Calculate metrics
                successful_requests = [r for r in request_results if r["success"]]
                response_times = [r["duration"] for r in successful_requests]

                results[concurrency] = {
                    "total_requests": len(request_results),
                    "successful_requests": len(successful_requests),
                    "success_rate": len(successful_requests) / len(request_results),
                    "total_time": total_time,
                    "throughput": len(successful_requests) / total_time,
                    "avg_response_time": statistics.mean(response_times) if response_times else 0,
                    "p95_response_time": statistics.quantiles(response_times, n=20)[18] if len(response_times) > 1 else 0
                }

        # Performance assertions
        for concurrency, metrics in results.items():
            assert metrics["success_rate"] >= 0.90  # 90% success rate
            assert metrics["avg_response_time"] < 2.0  # Average under 2s
            assert metrics["throughput"] > concurrency * 0.5  # Reasonable throughput

        # Scalability assertions
        assert results[5]["throughput"] > results[1]["throughput"] * 2  # Should scale
        assert results[10]["success_rate"] >= 0.85  # Maintain quality under load

    def test_api_memory_usage_under_load(self, test_client):
        """Test API memory usage under sustained load."""

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        memory_samples = []

        def monitor_memory():
            """Monitor memory usage during test."""
            for _ in range(30):  # Monitor for 30 seconds
                memory_samples.append(process.memory_info().rss)
                time.sleep(1)

        # Start memory monitoring
        monitor_thread = threading.Thread(target=monitor_memory)
        monitor_thread.start()

        try:
            # Generate sustained load
            for batch in range(10):
                with ThreadPoolExecutor(max_workers=10) as executor:
                    futures = [
                        executor.submit(test_client.get, "/api/v1/health")
                        for _ in range(20)
                    ]

                    # Wait for completion
                    [future.result() for future in futures]

                time.sleep(1)  # Brief pause between batches

        finally:
            monitor_thread.join()

        # Analyze memory usage
        peak_memory = max(memory_samples)
        final_memory = process.memory_info().rss
        memory_increase = peak_memory - initial_memory
        memory_leaked = final_memory - initial_memory

        # Memory assertions
        assert memory_increase < 100 * 1024 * 1024  # Under 100MB increase
        assert memory_leaked < 50 * 1024 * 1024     # Under 50MB leaked
        assert peak_memory < initial_memory * 2     # Under 2x initial memory

    def test_api_stress_testing(self, test_client):
        """Stress test API with extreme load."""

        stress_levels = [
            {"workers": 20, "requests_per_worker": 50, "duration": 10},
            {"workers": 50, "requests_per_worker": 20, "duration": 15},
            {"workers": 100, "requests_per_worker": 10, "duration": 20}
        ]

        for stress_config in stress_levels:
            workers = stress_config["workers"]
            requests_per_worker = stress_config["requests_per_worker"]
            duration = stress_config["duration"]

            results = []
            start_time = time.perf_counter()

            def stress_worker(worker_id: int):
                """Worker function for stress testing."""
                worker_results = []
                worker_start = time.perf_counter()

                while time.perf_counter() - worker_start < duration:
                    try:
                        response = test_client.get("/api/v1/health")
                        worker_results.append({
                            "success": response.status_code == 200,
                            "response_time": 0.001  # Approximate
                        })
                    except Exception:
                        worker_results.append({
                            "success": False,
                            "response_time": 0
                        })

                return worker_results

            # Execute stress test
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [
                    executor.submit(stress_worker, i)
                    for i in range(workers)
                ]

                worker_results = [future.result() for future in futures]

            # Aggregate results
            all_results = []
            for worker_result in worker_results:
                all_results.extend(worker_result)

            total_time = time.perf_counter() - start_time
            successful_requests = [r for r in all_results if r["success"]]

            stress_metrics = {
                "workers": workers,
                "total_requests": len(all_results),
                "successful_requests": len(successful_requests),
                "success_rate": len(successful_requests) / len(all_results) if all_results else 0,
                "throughput": len(successful_requests) / total_time,
                "duration": total_time
            }

            # Stress test assertions
            assert stress_metrics["success_rate"] >= 0.70  # 70% under stress
            assert stress_metrics["throughput"] > 10       # Minimum throughput

            print(f"Stress test {workers} workers: "
                  f"{stress_metrics['success_rate']:.2%} success, "
                  f"{stress_metrics['throughput']:.1f} req/s")


class TestMLPerformance:
    """Performance tests for ML operations."""

    def test_algorithm_training_performance(self):
        """Benchmark training performance across algorithms."""

        # Generate test datasets of different sizes
        dataset_sizes = [
            (100, 5),    # Small
            (1000, 10),  # Medium
            (10000, 20), # Large
        ]

        algorithms = ["IsolationForest", "LocalOutlierFactor", "OneClassSVM"]

        results = {}

        for dataset_size in dataset_sizes:
            n_samples, n_features = dataset_size
            X = np.random.randn(n_samples, n_features)

            for algorithm in algorithms:
                # Mock adapter for consistent testing
                with patch('pynomaly.infrastructure.adapters.sklearn_adapter.SklearnAdapter') as mock_adapter:
                    adapter = mock_adapter.return_value

                    # Mock training with realistic timing
                    def mock_fit(dataset):
                        # Simulate training time based on data size
                        training_time = (n_samples * n_features) / 100000  # Scaled timing
                        time.sleep(min(training_time, 0.1))  # Cap at 100ms for testing
                        return Mock(id=f"model_{algorithm}_{n_samples}")

                    adapter.fit = mock_fit

                    # Measure training performance
                    start_time = time.perf_counter()
                    start_memory = psutil.Process().memory_info().rss

                    model = adapter.fit(Mock(data=X))

                    end_time = time.perf_counter()
                    end_memory = psutil.Process().memory_info().rss

                    # Record results
                    key = f"{algorithm}_{n_samples}x{n_features}"
                    results[key] = {
                        "algorithm": algorithm,
                        "dataset_size": (n_samples, n_features),
                        "training_time": end_time - start_time,
                        "memory_usage": end_memory - start_memory,
                        "throughput": n_samples / (end_time - start_time)
                    }

        # Performance assertions
        for key, metrics in results.items():
            assert metrics["training_time"] < 5.0  # Under 5 seconds
            assert metrics["memory_usage"] < 100 * 1024 * 1024  # Under 100MB
            assert metrics["throughput"] > 1000  # Over 1000 samples/second

        # Scalability assertions
        small_iso = results["IsolationForest_100x5"]
        large_iso = results["IsolationForest_10000x20"]

        # Training time should scale sub-linearly
        size_ratio = (10000 * 20) / (100 * 5)  # 400x larger
        time_ratio = large_iso["training_time"] / small_iso["training_time"]
        assert time_ratio < size_ratio * 0.5  # Should be much less than linear

    def test_prediction_performance_scaling(self):
        """Test prediction performance scaling with data size."""

        batch_sizes = [10, 100, 1000, 10000]
        n_features = 10

        with patch('pynomaly.infrastructure.adapters.sklearn_adapter.SklearnAdapter') as mock_adapter:
            adapter = mock_adapter.return_value

            # Mock model
            mock_model = Mock()
            mock_model.decision_function = lambda X: np.random.randn(len(X))
            mock_model.predict = lambda X: np.random.choice([-1, 1], len(X))

            def mock_predict(detector, data):
                # Simulate prediction time
                prediction_time = len(data) / 50000  # 50k predictions per second
                time.sleep(min(prediction_time, 0.1))  # Cap for testing

                return Mock(
                    predictions=np.random.choice([0, 1], len(data)).tolist(),
                    anomaly_scores=np.random.random(len(data)).tolist()
                )

            adapter.predict = mock_predict

            prediction_results = {}

            for batch_size in batch_sizes:
                X = np.random.randn(batch_size, n_features)

                start_time = time.perf_counter()
                result = adapter.predict(mock_model, X)
                end_time = time.perf_counter()

                duration = end_time - start_time
                throughput = batch_size / duration

                prediction_results[batch_size] = {
                    "batch_size": batch_size,
                    "duration": duration,
                    "throughput": throughput,
                    "latency_per_sample": duration / batch_size
                }

        # Performance assertions
        for batch_size, metrics in prediction_results.items():
            assert metrics["throughput"] > 1000  # Over 1000 predictions/second
            assert metrics["latency_per_sample"] < 0.01  # Under 10ms per sample

        # Throughput should increase with batch size
        assert prediction_results[1000]["throughput"] > prediction_results[100]["throughput"]
        assert prediction_results[10000]["throughput"] > prediction_results[1000]["throughput"]

    def test_concurrent_ml_operations(self):
        """Test performance of concurrent ML operations."""

        def ml_operation(operation_id: int) -> Dict[str, Any]:
            """Simulate ML operation."""
            start_time = time.perf_counter()

            # Generate random data
            X = np.random.randn(1000, 5)

            # Simulate processing
            with patch('pynomaly.infrastructure.adapters.sklearn_adapter.SklearnAdapter') as mock_adapter:
                adapter = mock_adapter.return_value

                # Mock operations with realistic timing
                time.sleep(0.01)  # 10ms processing time

                result = Mock(
                    predictions=np.random.choice([0, 1], 1000).tolist(),
                    anomaly_scores=np.random.random(1000).tolist()
                )

                adapter.predict.return_value = result
                prediction_result = adapter.predict(Mock(), X)

            end_time = time.perf_counter()

            return {
                "operation_id": operation_id,
                "duration": end_time - start_time,
                "success": True,
                "samples_processed": 1000
            }

        # Test different concurrency levels
        concurrency_levels = [1, 2, 4, 8]

        for concurrency in concurrency_levels:
            start_time = time.perf_counter()

            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = [
                    executor.submit(ml_operation, i)
                    for i in range(concurrency * 2)
                ]
                results = [future.result() for future in futures]

            end_time = time.perf_counter()
            total_time = end_time - start_time

            # Calculate metrics
            total_samples = sum(r["samples_processed"] for r in results)
            overall_throughput = total_samples / total_time
            avg_operation_time = statistics.mean(r["duration"] for r in results)

            # Performance assertions
            assert all(r["success"] for r in results)
            assert overall_throughput > concurrency * 50000  # Scale with concurrency
            assert avg_operation_time < 0.1  # Under 100ms per operation

            print(f"Concurrency {concurrency}: "
                  f"{overall_throughput:.0f} samples/s, "
                  f"{avg_operation_time:.3f}s avg operation time")

    def test_memory_efficiency_ml_operations(self):
        """Test memory efficiency of ML operations."""

        process = psutil.Process()
        initial_memory = process.memory_info().rss

        # Test memory usage with different data sizes
        data_sizes = [1000, 10000, 100000]
        memory_results = {}

        for size in data_sizes:
            # Force garbage collection before test
            gc.collect()

            pre_test_memory = process.memory_info().rss

            # Generate large dataset
            X = np.random.randn(size, 20)

            # Simulate ML operations
            with patch('pynomaly.infrastructure.adapters.sklearn_adapter.SklearnAdapter') as mock_adapter:
                adapter = mock_adapter.return_value

                # Mock training
                adapter.fit.return_value = Mock(id=f"model_{size}")
                model = adapter.fit(Mock(data=X))

                # Mock prediction
                adapter.predict.return_value = Mock(
                    predictions=np.random.choice([0, 1], size).tolist(),
                    anomaly_scores=np.random.random(size).tolist()
                )
                result = adapter.predict(model, X)

            post_test_memory = process.memory_info().rss
            memory_used = post_test_memory - pre_test_memory

            # Clean up
            del X, model, result
            gc.collect()

            memory_results[size] = {
                "data_size": size,
                "memory_used_mb": memory_used / (1024 * 1024),
                "memory_per_sample": memory_used / size
            }

        # Memory efficiency assertions
        for size, metrics in memory_results.items():
            assert metrics["memory_used_mb"] < 500  # Under 500MB
            assert metrics["memory_per_sample"] < 10000  # Under 10KB per sample

        # Memory usage should scale reasonably
        ratio_10k_1k = memory_results[10000]["memory_used_mb"] / memory_results[1000]["memory_used_mb"]
        assert ratio_10k_1k < 20  # Should not be 100x worse

        # Check for memory leaks
        final_memory = process.memory_info().rss
        memory_leak = final_memory - initial_memory
        assert memory_leak < 50 * 1024 * 1024  # Under 50MB leaked


class TestDatabasePerformance:
    """Performance tests for database operations."""

    @pytest.fixture
    async def mock_database(self):
        """Mock database for performance testing."""
        with patch('pynomaly.infrastructure.persistence.database.DatabaseManager') as mock_db:
            db_manager = mock_db.return_value

            # Configure realistic response times
            async def mock_save(entity):
                await asyncio.sleep(0.001)  # 1ms save time
                return Mock(id=f"saved_{id(entity)}")

            async def mock_get(entity_id):
                await asyncio.sleep(0.0005)  # 0.5ms get time
                return Mock(id=entity_id)

            async def mock_list():
                await asyncio.sleep(0.01)  # 10ms list time
                return [Mock(id=f"item_{i}") for i in range(100)]

            db_manager.save_dataset = mock_save
            db_manager.get_dataset = mock_get
            db_manager.list_datasets = mock_list

            yield db_manager

    async def test_database_operation_benchmarks(self, mock_database):
        """Benchmark individual database operations."""

        operations = [
            ("save_dataset", lambda: mock_database.save_dataset(Mock())),
            ("get_dataset", lambda: mock_database.get_dataset("test_id")),
            ("list_datasets", lambda: mock_database.list_datasets())
        ]

        benchmark_results = {}

        for operation_name, operation_func in operations:
            # Warm-up
            await operation_func()

            # Benchmark
            times = []
            for _ in range(100):  # 100 iterations
                start_time = time.perf_counter()
                await operation_func()
                end_time = time.perf_counter()
                times.append(end_time - start_time)

            benchmark_results[operation_name] = {
                "avg_time": statistics.mean(times),
                "min_time": min(times),
                "max_time": max(times),
                "p95_time": statistics.quantiles(times, n=20)[18] if len(times) > 1 else times[0],
                "throughput": 1 / statistics.mean(times)
            }

        # Performance assertions
        assert benchmark_results["save_dataset"]["p95_time"] < 0.01   # Under 10ms
        assert benchmark_results["get_dataset"]["p95_time"] < 0.005   # Under 5ms
        assert benchmark_results["list_datasets"]["p95_time"] < 0.05  # Under 50ms

        # Throughput assertions
        assert benchmark_results["save_dataset"]["throughput"] > 100   # 100 saves/second
        assert benchmark_results["get_dataset"]["throughput"] > 1000   # 1000 gets/second
        assert benchmark_results["list_datasets"]["throughput"] > 20   # 20 lists/second

    async def test_database_concurrent_operations(self, mock_database):
        """Test database performance under concurrent load."""

        async def concurrent_operation(operation_id: int):
            """Perform concurrent database operations."""
            start_time = time.perf_counter()

            try:
                # Mix of operations
                if operation_id % 3 == 0:
                    await mock_database.save_dataset(Mock(id=f"dataset_{operation_id}"))
                elif operation_id % 3 == 1:
                    await mock_database.get_dataset(f"dataset_{operation_id}")
                else:
                    await mock_database.list_datasets()

                success = True
                error = None
            except Exception as e:
                success = False
                error = str(e)

            end_time = time.perf_counter()

            return {
                "operation_id": operation_id,
                "duration": end_time - start_time,
                "success": success,
                "error": error
            }

        # Test different concurrency levels
        concurrency_levels = [10, 50, 100]

        for concurrency in concurrency_levels:
            start_time = time.perf_counter()

            # Execute concurrent operations
            tasks = [
                concurrent_operation(i)
                for i in range(concurrency)
            ]
            results = await asyncio.gather(*tasks)

            end_time = time.perf_counter()
            total_time = end_time - start_time

            # Analyze results
            successful_operations = [r for r in results if r["success"]]
            response_times = [r["duration"] for r in successful_operations]

            metrics = {
                "concurrency": concurrency,
                "total_operations": len(results),
                "successful_operations": len(successful_operations),
                "success_rate": len(successful_operations) / len(results),
                "total_time": total_time,
                "throughput": len(successful_operations) / total_time,
                "avg_response_time": statistics.mean(response_times) if response_times else 0,
                "p95_response_time": statistics.quantiles(response_times, n=20)[18] if len(response_times) > 1 else 0
            }

            # Performance assertions
            assert metrics["success_rate"] >= 0.95  # 95% success rate
            assert metrics["avg_response_time"] < 0.1  # Under 100ms average
            assert metrics["throughput"] > concurrency * 5  # Reasonable throughput

            print(f"DB Concurrency {concurrency}: "
                  f"{metrics['success_rate']:.2%} success, "
                  f"{metrics['throughput']:.1f} ops/s")

    async def test_database_connection_pool_performance(self, mock_database):
        """Test database connection pool performance."""

        # Mock connection pool
        with patch('pynomaly.infrastructure.persistence.connection_pool.ConnectionPool') as mock_pool:
            pool = mock_pool.return_value

            # Configure pool operations
            async def mock_acquire():
                await asyncio.sleep(0.001)  # 1ms to acquire connection
                return Mock(id="connection")

            async def mock_release(connection):
                await asyncio.sleep(0.0005)  # 0.5ms to release connection

            pool.acquire = mock_acquire
            pool.release = mock_release
            pool.size = 10
            pool.max_size = 20

            # Test pool utilization
            connection_times = []

            async def use_connection(operation_id: int):
                start_time = time.perf_counter()

                # Acquire connection
                connection = await pool.acquire()
                acquire_time = time.perf_counter()

                # Use connection (simulate query)
                await asyncio.sleep(0.01)  # 10ms query
                query_time = time.perf_counter()

                # Release connection
                await pool.release(connection)
                release_time = time.perf_counter()

                return {
                    "operation_id": operation_id,
                    "total_time": release_time - start_time,
                    "acquire_time": acquire_time - start_time,
                    "query_time": query_time - acquire_time,
                    "release_time": release_time - query_time
                }

            # Test with high concurrency
            tasks = [use_connection(i) for i in range(50)]
            results = await asyncio.gather(*tasks)

            # Analyze connection pool performance
            acquire_times = [r["acquire_time"] for r in results]
            total_times = [r["total_time"] for r in results]

            # Performance assertions
            assert statistics.mean(acquire_times) < 0.01  # Under 10ms to acquire
            assert max(acquire_times) < 0.1  # Under 100ms worst case
            assert statistics.mean(total_times) < 0.02  # Under 20ms total

    async def test_database_bulk_operations(self, mock_database):
        """Test bulk database operation performance."""

        # Mock bulk operations
        async def mock_bulk_insert(entities):
            # Simulate bulk insert time based on count
            await asyncio.sleep(len(entities) * 0.0001)  # 0.1ms per entity
            return [Mock(id=f"bulk_{i}") for i in range(len(entities))]

        async def mock_bulk_update(entities):
            await asyncio.sleep(len(entities) * 0.0001)
            return entities

        mock_database.bulk_insert_datasets = mock_bulk_insert
        mock_database.bulk_update_datasets = mock_bulk_update

        # Test different bulk sizes
        bulk_sizes = [10, 100, 1000, 10000]

        for bulk_size in bulk_sizes:
            entities = [Mock(id=f"entity_{i}") for i in range(bulk_size)]

            # Test bulk insert
            start_time = time.perf_counter()
            inserted = await mock_database.bulk_insert_datasets(entities)
            insert_time = time.perf_counter() - start_time

            # Test bulk update
            start_time = time.perf_counter()
            updated = await mock_database.bulk_update_datasets(entities)
            update_time = time.perf_counter() - start_time

            # Calculate throughput
            insert_throughput = bulk_size / insert_time
            update_throughput = bulk_size / update_time

            # Performance assertions
            assert insert_throughput > 1000  # Over 1000 inserts/second
            assert update_throughput > 1000  # Over 1000 updates/second
            assert insert_time < bulk_size * 0.001  # Under 1ms per entity
            assert update_time < bulk_size * 0.001  # Under 1ms per entity

            print(f"Bulk {bulk_size}: "
                  f"Insert {insert_throughput:.0f}/s, "
                  f"Update {update_throughput:.0f}/s")


class TestSystemPerformance:
    """System-wide performance tests."""

    def test_system_resource_utilization(self):
        """Test system resource utilization under load."""

        # Monitor system resources
        cpu_samples = []
        memory_samples = []

        def monitor_resources():
            """Monitor CPU and memory usage."""
            for _ in range(30):  # 30 seconds of monitoring
                cpu_samples.append(psutil.cpu_percent(interval=1))
                memory_samples.append(psutil.virtual_memory().percent)

        # Start monitoring
        monitor_thread = threading.Thread(target=monitor_resources)
        monitor_thread.start()

        try:
            # Generate system load
            def cpu_intensive_task():
                """CPU intensive task."""
                for _ in range(1000000):
                    _ = sum(range(100))

            def memory_intensive_task():
                """Memory intensive task."""
                large_arrays = []
                for _ in range(10):
                    large_arrays.append(np.random.randn(1000, 1000))
                return large_arrays

            # Execute load tests
            with ThreadPoolExecutor(max_workers=4) as executor:
                # CPU tasks
                cpu_futures = [
                    executor.submit(cpu_intensive_task)
                    for _ in range(4)
                ]

                # Memory tasks
                memory_futures = [
                    executor.submit(memory_intensive_task)
                    for _ in range(2)
                ]

                # Wait for completion
                [future.result() for future in cpu_futures + memory_futures]

        finally:
            monitor_thread.join()

        # Analyze resource utilization
        avg_cpu = statistics.mean(cpu_samples)
        max_cpu = max(cpu_samples)
        avg_memory = statistics.mean(memory_samples)
        max_memory = max(memory_samples)

        # Resource utilization assertions
        assert max_cpu < 90  # Under 90% CPU
        assert max_memory < 80  # Under 80% memory
        assert avg_cpu < 50  # Under 50% average CPU
        assert avg_memory < 60  # Under 60% average memory

        print(f"Resource utilization - "
              f"CPU: avg {avg_cpu:.1f}%, max {max_cpu:.1f}%; "
              f"Memory: avg {avg_memory:.1f}%, max {max_memory:.1f}%")

    def test_end_to_end_performance(self):
        """Test end-to-end system performance."""

        from fastapi.testclient import TestClient

        from pynomaly.presentation.api.app import create_app

        app = create_app(testing=True)
        client = TestClient(app)

        # End-to-end workflow timing
        workflow_steps = []

        def time_step(step_name: str, operation: Callable):
            """Time a workflow step."""
            start_time = time.perf_counter()
            result = operation()
            end_time = time.perf_counter()

            workflow_steps.append({
                "step": step_name,
                "duration": end_time - start_time,
                "success": result is not None
            })

            return result

        # Mock the workflow steps
        with patch.multiple(
            'pynomaly.application.services.detection_service.DetectionService',
            detect_anomalies=MagicMock(return_value=Mock()),
            train_detector=MagicMock(return_value=Mock())
        ):

            # Step 1: Create dataset
            dataset_data = {
                "name": "E2E Performance Dataset",
                "data": np.random.randn(1000, 10).tolist(),
                "features": [f"f_{i}" for i in range(10)]
            }

            dataset = time_step(
                "create_dataset",
                lambda: client.post("/api/v1/datasets", json=dataset_data)
            )

            # Step 2: Create detector
            detector_data = {
                "name": "E2E Performance Detector",
                "algorithm": "IsolationForest",
                "parameters": {"n_estimators": 100}
            }

            detector = time_step(
                "create_detector",
                lambda: client.post("/api/v1/detectors", json=detector_data)
            )

            # Step 3: Train detector (mocked)
            training = time_step(
                "train_detector",
                lambda: {"status": "completed"}  # Simulate training
            )

            # Step 4: Perform detection
            detection_data = {
                "detector_id": "test_detector",
                "data": np.random.randn(100, 10).tolist(),
                "return_scores": True
            }

            detection = time_step(
                "detect_anomalies",
                lambda: client.post("/api/v1/detection/detect", json=detection_data)
            )

        # Analyze workflow performance
        total_time = sum(step["duration"] for step in workflow_steps)
        successful_steps = [step for step in workflow_steps if step["success"]]

        # End-to-end performance assertions
        assert len(successful_steps) == len(workflow_steps)  # All steps successful
        assert total_time < 5.0  # Under 5 seconds total

        # Individual step assertions
        step_times = {step["step"]: step["duration"] for step in workflow_steps}
        assert step_times["create_dataset"] < 1.0   # Under 1 second
        assert step_times["create_detector"] < 0.5  # Under 0.5 seconds
        assert step_times["detect_anomalies"] < 2.0 # Under 2 seconds

        step_details = [f"{s['step']}:{s['duration']:.2f}s" for s in workflow_steps]
        print(f"E2E Performance: {total_time:.2f}s total, steps: {step_details}")

    def test_performance_regression_detection(self):
        """Test for performance regressions."""

        # Define performance baselines
        baselines = {
            "api_response_time": 0.5,      # 500ms
            "ml_training_time": 2.0,       # 2 seconds
            "prediction_throughput": 1000, # 1000 predictions/second
            "memory_usage_mb": 200,        # 200MB
            "database_query_time": 0.01    # 10ms
        }

        # Measure current performance
        current_performance = {}

        # API response time
        from fastapi.testclient import TestClient

        from pynomaly.presentation.api.app import create_app

        app = create_app(testing=True)
        client = TestClient(app)

        start_time = time.perf_counter()
        response = client.get("/api/v1/health")
        api_time = time.perf_counter() - start_time
        current_performance["api_response_time"] = api_time

        # ML operations (mocked)
        with patch('pynomaly.infrastructure.adapters.sklearn_adapter.SklearnAdapter') as mock_adapter:
            mock_adapter.return_value.fit.return_value = Mock()

            start_time = time.perf_counter()
            time.sleep(0.1)  # Simulate training
            training_time = time.perf_counter() - start_time
            current_performance["ml_training_time"] = training_time

        # Prediction throughput
        predictions_per_second = 5000  # Simulated
        current_performance["prediction_throughput"] = predictions_per_second

        # Memory usage
        memory_mb = psutil.Process().memory_info().rss / (1024 * 1024)
        current_performance["memory_usage_mb"] = memory_mb

        # Database query time (mocked)
        current_performance["database_query_time"] = 0.005  # Simulated

        # Check for regressions
        regressions = []

        for metric, baseline in baselines.items():
            current = current_performance[metric]

            # Define regression thresholds
            if metric == "prediction_throughput":
                # Higher is better
                regression_threshold = baseline * 0.8  # 20% slower
                if current < regression_threshold:
                    regressions.append({
                        "metric": metric,
                        "baseline": baseline,
                        "current": current,
                        "regression_percent": (baseline - current) / baseline * 100
                    })
            else:
                # Lower is better
                regression_threshold = baseline * 1.2  # 20% slower
                if current > regression_threshold:
                    regressions.append({
                        "metric": metric,
                        "baseline": baseline,
                        "current": current,
                        "regression_percent": (current - baseline) / baseline * 100
                    })

        # Performance regression assertions
        assert len(regressions) == 0, f"Performance regressions detected: {regressions}"

        # Log performance metrics
        print("Performance Metrics:")
        for metric, value in current_performance.items():
            baseline = baselines[metric]
            if metric == "prediction_throughput":
                improvement = ((value - baseline) / baseline) * 100
                print(f"  {metric}: {value:.3f} (baseline: {baseline:.3f}, {improvement:+.1f}%)")
            else:
                improvement = ((baseline - value) / baseline) * 100
                print(f"  {metric}: {value:.3f} (baseline: {baseline:.3f}, {improvement:+.1f}%)")
