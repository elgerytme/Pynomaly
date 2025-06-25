"""Load testing and performance benchmarks - Phase 4 Performance Testing."""

from __future__ import annotations

import asyncio
import gc
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import psutil
import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient

from pynomaly.infrastructure.config import create_container
from pynomaly.presentation.api.app import create_app


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""

    operation_name: str
    start_time: float
    end_time: float
    memory_before_mb: float
    memory_after_mb: float
    success: bool
    error_message: str | None = None

    @property
    def duration_ms(self) -> float:
        """Duration in milliseconds."""
        return (self.end_time - self.start_time) * 1000

    @property
    def memory_delta_mb(self) -> float:
        """Memory usage delta in MB."""
        return self.memory_after_mb - self.memory_before_mb


@dataclass
class LoadTestResults:
    """Results from load testing."""

    total_requests: int
    successful_requests: int
    failed_requests: int
    total_duration_seconds: float
    avg_response_time_ms: float
    min_response_time_ms: float
    max_response_time_ms: float
    p95_response_time_ms: float
    requests_per_second: float
    memory_usage_mb: float
    cpu_usage_percent: float
    error_messages: list[str]


class PerformanceMonitor:
    """Monitor system performance during tests."""

    def __init__(self):
        self.metrics: list[PerformanceMetrics] = []
        self.start_time: float | None = None
        self.process = psutil.Process()

    def start_monitoring(self):
        """Start performance monitoring."""
        self.start_time = time.time()
        self.metrics.clear()
        gc.collect()  # Clean up before monitoring

    @contextmanager
    def measure_operation(self, operation_name: str):
        """Context manager to measure a single operation."""
        memory_before = self.process.memory_info().rss / 1024 / 1024
        start_time = time.time()
        success = True
        error_message = None

        try:
            yield
        except Exception as e:
            success = False
            error_message = str(e)
            raise
        finally:
            end_time = time.time()
            memory_after = self.process.memory_info().rss / 1024 / 1024

            metric = PerformanceMetrics(
                operation_name=operation_name,
                start_time=start_time,
                end_time=end_time,
                memory_before_mb=memory_before,
                memory_after_mb=memory_after,
                success=success,
                error_message=error_message,
            )
            self.metrics.append(metric)

    def get_summary(self) -> dict[str, Any]:
        """Get performance summary."""
        if not self.metrics:
            return {}

        successful_metrics = [m for m in self.metrics if m.success]
        failed_metrics = [m for m in self.metrics if not m.success]

        durations = [m.duration_ms for m in successful_metrics]
        memory_deltas = [m.memory_delta_mb for m in self.metrics]

        summary = {
            "total_operations": len(self.metrics),
            "successful_operations": len(successful_metrics),
            "failed_operations": len(failed_metrics),
            "success_rate": len(successful_metrics) / len(self.metrics)
            if self.metrics
            else 0,
        }

        if durations:
            summary.update(
                {
                    "avg_duration_ms": statistics.mean(durations),
                    "min_duration_ms": min(durations),
                    "max_duration_ms": max(durations),
                    "p95_duration_ms": np.percentile(durations, 95),
                    "p99_duration_ms": np.percentile(durations, 99),
                }
            )

        if memory_deltas:
            summary.update(
                {
                    "avg_memory_delta_mb": statistics.mean(memory_deltas),
                    "max_memory_delta_mb": max(memory_deltas),
                    "total_memory_change_mb": sum(memory_deltas),
                }
            )

        if failed_metrics:
            summary["error_messages"] = [
                m.error_message for m in failed_metrics if m.error_message
            ]

        return summary


@pytest.fixture
def performance_monitor():
    """Performance monitoring fixture."""
    return PerformanceMonitor()


@pytest.fixture
def test_container():
    """Test container with performance optimizations."""
    container = create_container()
    # Configure for performance testing
    container.config.override(
        {
            "database_url": "sqlite:///:memory:",
            "redis_url": "redis://localhost:6379/15",  # Test database
            "max_concurrent_detections": 10,
            "log_level": "WARNING",  # Reduce logging overhead
            "enable_metrics": False,  # Disable metrics collection during tests
        }
    )
    return container


@pytest.fixture
def api_client(test_container):
    """API client for load testing."""
    app = create_app(test_container)
    return TestClient(app)


@pytest.fixture
async def async_api_client(test_container):
    """Async API client for load testing."""
    app = create_app(test_container)
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


def generate_test_dataset(
    n_samples: int, n_features: int, with_anomalies: bool = False
) -> pd.DataFrame:
    """Generate test dataset for performance testing."""
    np.random.seed(42)  # Reproducible data

    # Generate normal data
    data = np.random.normal(0, 1, (n_samples, n_features))

    if with_anomalies:
        # Add some anomalies (10% of data)
        n_anomalies = max(1, n_samples // 10)
        anomaly_indices = np.random.choice(n_samples, n_anomalies, replace=False)
        data[anomaly_indices] = np.random.normal(5, 2, (n_anomalies, n_features))

    feature_names = [f"feature_{i}" for i in range(n_features)]
    return pd.DataFrame(data, columns=feature_names)


class TestAPILoadTesting:
    """Load testing for API endpoints."""

    def test_concurrent_dataset_uploads(
        self, api_client: TestClient, performance_monitor: PerformanceMonitor
    ):
        """Test concurrent dataset uploads."""
        performance_monitor.start_monitoring()

        # Generate test datasets
        datasets = []
        for i in range(10):
            dataset = generate_test_dataset(100, 5)
            csv_data = dataset.to_csv(index=False)
            datasets.append((f"load_test_dataset_{i}", csv_data))

        def upload_dataset(name_and_data):
            name, csv_data = name_and_data
            with performance_monitor.measure_operation(f"upload_{name}"):
                response = api_client.post(
                    "/api/datasets/",
                    files={"file": (f"{name}.csv", csv_data.encode(), "text/csv")},
                    data={"name": name, "description": f"Load test dataset {name}"},
                )
                return (
                    response.status_code,
                    response.json() if response.status_code == 201 else None,
                )

        # Execute concurrent uploads
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(upload_dataset, dataset) for dataset in datasets]
            results = [future.result() for future in as_completed(futures)]

        total_time = time.time() - start_time

        # Analyze results
        successful_uploads = sum(1 for status_code, _ in results if status_code == 201)
        dataset_ids = [
            data["id"] for status_code, data in results if data and status_code == 201
        ]

        summary = performance_monitor.get_summary()

        # Assertions
        assert successful_uploads >= 8  # At least 80% success rate
        assert total_time < 30  # Should complete within 30 seconds
        assert summary["success_rate"] >= 0.8
        assert summary["avg_duration_ms"] < 5000  # Average under 5 seconds

        print(
            f"Concurrent uploads: {successful_uploads}/{len(datasets)} successful in {total_time:.2f}s"
        )
        print(f"Performance summary: {summary}")

        # Clean up
        for dataset_id in dataset_ids:
            try:
                api_client.delete(f"/api/datasets/{dataset_id}")
            except:
                pass  # Ignore cleanup errors

    def test_detector_creation_load(
        self, api_client: TestClient, performance_monitor: PerformanceMonitor
    ):
        """Test detector creation under load."""
        performance_monitor.start_monitoring()

        algorithms = ["IsolationForest", "LocalOutlierFactor", "OneClassSVM"]
        detector_configs = []

        # Generate detector configurations
        for i in range(15):
            algorithm = algorithms[i % len(algorithms)]
            config = {
                "name": f"Load Test Detector {i}",
                "algorithm": algorithm,
                "hyperparameters": {"contamination": 0.1, "random_state": 42 + i},
            }
            detector_configs.append(config)

        def create_detector(config):
            with performance_monitor.measure_operation(
                f"create_detector_{config['name']}"
            ):
                response = api_client.post("/api/detectors/", json=config)
                return (
                    response.status_code,
                    response.json() if response.status_code == 201 else None,
                )

        # Execute concurrent detector creation
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [
                executor.submit(create_detector, config) for config in detector_configs
            ]
            results = [future.result() for future in as_completed(futures)]

        total_time = time.time() - start_time

        # Analyze results
        successful_creations = sum(
            1 for status_code, _ in results if status_code == 201
        )
        detector_ids = [
            data["id"] for status_code, data in results if data and status_code == 201
        ]

        summary = performance_monitor.get_summary()

        # Assertions
        assert successful_creations >= 12  # At least 80% success rate
        assert total_time < 20  # Should complete within 20 seconds
        assert summary["success_rate"] >= 0.8
        assert summary["avg_duration_ms"] < 2000  # Average under 2 seconds

        print(
            f"Detector creation: {successful_creations}/{len(detector_configs)} successful in {total_time:.2f}s"
        )

        # Clean up
        for detector_id in detector_ids:
            try:
                api_client.delete(f"/api/detectors/{detector_id}")
            except:
                pass

    @pytest.mark.asyncio
    async def test_high_throughput_health_checks(self, async_api_client: AsyncClient):
        """Test high-throughput health check requests."""
        num_requests = 100
        concurrent_requests = 20

        async def make_health_request():
            start_time = time.time()
            response = await async_api_client.get("/api/health/")
            end_time = time.time()
            return {
                "status_code": response.status_code,
                "duration_ms": (end_time - start_time) * 1000,
                "success": response.status_code == 200,
            }

        # Execute high-throughput requests
        start_time = time.time()

        # Use semaphore to limit concurrency
        semaphore = asyncio.Semaphore(concurrent_requests)

        async def bounded_request():
            async with semaphore:
                return await make_health_request()

        tasks = [bounded_request() for _ in range(num_requests)]
        results = await asyncio.gather(*tasks)

        total_time = time.time() - start_time

        # Analyze results
        successful_requests = sum(1 for result in results if result["success"])
        response_times = [
            result["duration_ms"] for result in results if result["success"]
        ]

        # Calculate statistics
        avg_response_time = statistics.mean(response_times) if response_times else 0
        p95_response_time = np.percentile(response_times, 95) if response_times else 0
        requests_per_second = num_requests / total_time

        # Assertions
        assert successful_requests >= num_requests * 0.95  # 95% success rate
        assert avg_response_time < 100  # Average under 100ms
        assert p95_response_time < 500  # 95th percentile under 500ms
        assert requests_per_second > 10  # At least 10 RPS

        print(
            f"Health check throughput: {successful_requests}/{num_requests} successful"
        )
        print(
            f"RPS: {requests_per_second:.1f}, Avg: {avg_response_time:.1f}ms, P95: {p95_response_time:.1f}ms"
        )

    def test_mixed_workload_simulation(
        self, api_client: TestClient, performance_monitor: PerformanceMonitor
    ):
        """Simulate mixed workload with different operations."""
        performance_monitor.start_monitoring()

        # Prepare test data
        dataset = generate_test_dataset(500, 10, with_anomalies=True)
        csv_data = dataset.to_csv(index=False)

        # Upload initial dataset
        upload_response = api_client.post(
            "/api/datasets/",
            files={"file": ("workload_dataset.csv", csv_data.encode(), "text/csv")},
            data={"name": "Workload Test Dataset"},
        )
        assert upload_response.status_code == 201
        dataset_id = upload_response.json()["id"]

        # Create initial detector
        detector_response = api_client.post(
            "/api/detectors/",
            json={
                "name": "Workload Test Detector",
                "algorithm": "IsolationForest",
                "hyperparameters": {"contamination": 0.1, "random_state": 42},
            },
        )
        assert detector_response.status_code == 201
        detector_id = detector_response.json()["id"]

        # Train detector
        train_response = api_client.post(
            f"/api/detectors/{detector_id}/train", json={"dataset_id": dataset_id}
        )
        assert train_response.status_code == 200

        # Define workload operations
        def health_check():
            with performance_monitor.measure_operation("health_check"):
                response = api_client.get("/api/health/")
                return response.status_code == 200

        def list_datasets():
            with performance_monitor.measure_operation("list_datasets"):
                response = api_client.get("/api/datasets/")
                return response.status_code == 200

        def list_detectors():
            with performance_monitor.measure_operation("list_detectors"):
                response = api_client.get("/api/detectors/")
                return response.status_code == 200

        def get_detector_details():
            with performance_monitor.measure_operation("get_detector_details"):
                response = api_client.get(f"/api/detectors/{detector_id}")
                return response.status_code == 200

        def run_prediction():
            with performance_monitor.measure_operation("run_prediction"):
                response = api_client.post(
                    f"/api/detectors/{detector_id}/predict",
                    json={"dataset_id": dataset_id},
                )
                return response.status_code == 200

        # Workload distribution (weights)
        operations = [
            (health_check, 30),  # 30% health checks
            (list_datasets, 20),  # 20% dataset listings
            (list_detectors, 20),  # 20% detector listings
            (get_detector_details, 15),  # 15% detector details
            (run_prediction, 15),  # 15% predictions
        ]

        # Generate operation sequence
        operation_sequence = []
        for operation, weight in operations:
            operation_sequence.extend([operation] * weight)

        # Shuffle for realistic mixed workload
        np.random.shuffle(operation_sequence)

        # Execute mixed workload
        start_time = time.time()

        def execute_operations(ops_subset):
            results = []
            for operation in ops_subset:
                try:
                    success = operation()
                    results.append(success)
                except Exception as e:
                    print(f"Operation failed: {e}")
                    results.append(False)
            return results

        # Split operations across threads
        chunk_size = len(operation_sequence) // 4
        operation_chunks = [
            operation_sequence[i : i + chunk_size]
            for i in range(0, len(operation_sequence), chunk_size)
        ]

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(execute_operations, chunk) for chunk in operation_chunks
            ]
            all_results = []
            for future in as_completed(futures):
                all_results.extend(future.result())

        total_time = time.time() - start_time

        # Analyze results
        successful_operations = sum(all_results)
        total_operations = len(all_results)

        summary = performance_monitor.get_summary()

        # Assertions
        assert successful_operations >= total_operations * 0.9  # 90% success rate
        assert total_time < 60  # Should complete within 1 minute
        assert summary["success_rate"] >= 0.9

        print(
            f"Mixed workload: {successful_operations}/{total_operations} successful in {total_time:.2f}s"
        )
        print(f"Operations per second: {total_operations / total_time:.1f}")
        print(f"Performance summary: {summary}")

        # Clean up
        api_client.delete(f"/api/detectors/{detector_id}")
        api_client.delete(f"/api/datasets/{dataset_id}")


class TestMemoryAndResourceTesting:
    """Test memory usage and resource management under load."""

    def test_memory_leak_detection(self, api_client: TestClient):
        """Test for memory leaks during repeated operations."""
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_measurements = [initial_memory]

        # Perform many operations that could potentially leak memory
        created_resources = []

        try:
            for iteration in range(50):
                # Create dataset
                dataset = generate_test_dataset(100, 5)
                csv_data = dataset.to_csv(index=False)

                upload_response = api_client.post(
                    "/api/datasets/",
                    files={
                        "file": (
                            f"leak_test_{iteration}.csv",
                            csv_data.encode(),
                            "text/csv",
                        )
                    },
                    data={"name": f"Leak Test Dataset {iteration}"},
                )

                if upload_response.status_code == 201:
                    dataset_id = upload_response.json()["id"]

                    # Create detector
                    detector_response = api_client.post(
                        "/api/detectors/",
                        json={
                            "name": f"Leak Test Detector {iteration}",
                            "algorithm": "IsolationForest",
                            "hyperparameters": {
                                "contamination": 0.1,
                                "random_state": iteration,
                            },
                        },
                    )

                    if detector_response.status_code == 201:
                        detector_id = detector_response.json()["id"]
                        created_resources.append((dataset_id, detector_id))

                        # Train and predict
                        train_response = api_client.post(
                            f"/api/detectors/{detector_id}/train",
                            json={"dataset_id": dataset_id},
                        )

                        if train_response.status_code == 200:
                            api_client.post(
                                f"/api/detectors/{detector_id}/predict",
                                json={"dataset_id": dataset_id},
                            )

                # Measure memory every 10 iterations
                if iteration % 10 == 0:
                    current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                    memory_measurements.append(current_memory)

                    # Force garbage collection
                    gc.collect()

        finally:
            # Clean up all created resources
            for dataset_id, detector_id in created_resources:
                try:
                    api_client.delete(f"/api/detectors/{detector_id}")
                    api_client.delete(f"/api/datasets/{dataset_id}")
                except:
                    pass

        # Force final garbage collection
        gc.collect()
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024

        # Analyze memory usage
        memory_increase = final_memory - initial_memory
        max_memory = max(memory_measurements)
        memory_trend = np.polyfit(
            range(len(memory_measurements)), memory_measurements, 1
        )[0]

        print(
            f"Memory usage: Initial: {initial_memory:.1f}MB, Final: {final_memory:.1f}MB"
        )
        print(f"Memory increase: {memory_increase:.1f}MB, Max: {max_memory:.1f}MB")
        print(f"Memory trend (slope): {memory_trend:.3f} MB/measurement")

        # Assertions
        assert memory_increase < 200  # Should not increase by more than 200MB
        assert memory_trend < 5  # Should not have strong upward trend
        assert max_memory - initial_memory < 500  # Peak usage should be reasonable

    def test_concurrent_resource_exhaustion(self, api_client: TestClient):
        """Test behavior under resource exhaustion scenarios."""

        # Test with many concurrent operations that could exhaust resources
        def resource_intensive_operation(thread_id: int):
            try:
                # Create large dataset
                large_dataset = generate_test_dataset(5000, 20, with_anomalies=True)
                csv_data = large_dataset.to_csv(index=False)

                # Upload
                upload_response = api_client.post(
                    "/api/datasets/",
                    files={
                        "file": (
                            f"intensive_{thread_id}.csv",
                            csv_data.encode(),
                            "text/csv",
                        )
                    },
                    data={"name": f"Resource Intensive Dataset {thread_id}"},
                )

                if upload_response.status_code != 201:
                    return {
                        "success": False,
                        "error": "Upload failed",
                        "thread_id": thread_id,
                    }

                dataset_id = upload_response.json()["id"]

                # Create detector
                detector_response = api_client.post(
                    "/api/detectors/",
                    json={
                        "name": f"Intensive Detector {thread_id}",
                        "algorithm": "IsolationForest",
                        "hyperparameters": {
                            "contamination": 0.1,
                            "n_estimators": 200,  # More intensive
                            "random_state": thread_id,
                        },
                    },
                )

                if detector_response.status_code != 201:
                    api_client.delete(f"/api/datasets/{dataset_id}")
                    return {
                        "success": False,
                        "error": "Detector creation failed",
                        "thread_id": thread_id,
                    }

                detector_id = detector_response.json()["id"]

                # Train (intensive operation)
                train_start = time.time()
                train_response = api_client.post(
                    f"/api/detectors/{detector_id}/train",
                    json={"dataset_id": dataset_id},
                )
                train_time = time.time() - train_start

                # Clean up
                api_client.delete(f"/api/detectors/{detector_id}")
                api_client.delete(f"/api/datasets/{dataset_id}")

                if train_response.status_code == 200:
                    return {
                        "success": True,
                        "thread_id": thread_id,
                        "train_time": train_time,
                    }
                else:
                    return {
                        "success": False,
                        "error": f"Training failed: {train_response.status_code}",
                        "thread_id": thread_id,
                    }

            except Exception as e:
                return {"success": False, "error": str(e), "thread_id": thread_id}

        # Launch many concurrent intensive operations
        num_threads = 8
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(resource_intensive_operation, i)
                for i in range(num_threads)
            ]
            results = [
                future.result(timeout=120)
                for future in as_completed(futures, timeout=120)
            ]

        total_time = time.time() - start_time

        # Analyze results
        successful_operations = sum(1 for result in results if result["success"])
        failed_operations = [result for result in results if not result["success"]]

        if successful_operations > 0:
            successful_results = [result for result in results if result["success"]]
            avg_train_time = statistics.mean(
                [r["train_time"] for r in successful_results]
            )
            max_train_time = max([r["train_time"] for r in successful_results])
        else:
            avg_train_time = max_train_time = 0

        print(
            f"Resource exhaustion test: {successful_operations}/{num_threads} successful"
        )
        print(f"Total time: {total_time:.1f}s, Avg train time: {avg_train_time:.1f}s")

        if failed_operations:
            print(f"Failed operations: {[r['error'] for r in failed_operations[:3]]}")

        # Assertions - system should handle load gracefully
        assert successful_operations >= num_threads * 0.5  # At least 50% should succeed
        assert total_time < 180  # Should complete within 3 minutes

        # If operations succeed, they should complete in reasonable time
        if avg_train_time > 0:
            assert avg_train_time < 60  # Average training under 1 minute
            assert max_train_time < 120  # Maximum training under 2 minutes

    def test_garbage_collection_effectiveness(self, api_client: TestClient):
        """Test effectiveness of garbage collection during operations."""
        # Monitor memory before operations
        gc.collect()
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024

        # Perform operations that create temporary objects
        temp_resource_ids = []

        for i in range(20):
            # Create temporary dataset
            dataset = generate_test_dataset(1000, 10)
            csv_data = dataset.to_csv(index=False)

            upload_response = api_client.post(
                "/api/datasets/",
                files={"file": (f"gc_test_{i}.csv", csv_data.encode(), "text/csv")},
                data={"name": f"GC Test Dataset {i}"},
            )

            if upload_response.status_code == 201:
                dataset_id = upload_response.json()["id"]
                temp_resource_ids.append(("dataset", dataset_id))

                # Create temporary detector
                detector_response = api_client.post(
                    "/api/detectors/",
                    json={
                        "name": f"GC Test Detector {i}",
                        "algorithm": "IsolationForest",
                        "hyperparameters": {"contamination": 0.1},
                    },
                )

                if detector_response.status_code == 201:
                    detector_id = detector_response.json()["id"]
                    temp_resource_ids.append(("detector", detector_id))

        # Measure memory after creating resources
        memory_after_creation = psutil.Process().memory_info().rss / 1024 / 1024

        # Delete all temporary resources
        for resource_type, resource_id in temp_resource_ids:
            if resource_type == "detector":
                api_client.delete(f"/api/detectors/{resource_id}")
            elif resource_type == "dataset":
                api_client.delete(f"/api/datasets/{resource_id}")

        # Force garbage collection
        gc.collect()

        # Measure memory after cleanup
        memory_after_cleanup = psutil.Process().memory_info().rss / 1024 / 1024

        # Calculate memory metrics
        memory_increase = memory_after_creation - initial_memory
        memory_recovered = memory_after_creation - memory_after_cleanup
        recovery_rate = memory_recovered / memory_increase if memory_increase > 0 else 0

        print(
            f"Memory: Initial: {initial_memory:.1f}MB, After creation: {memory_after_creation:.1f}MB"
        )
        print(f"Memory: After cleanup: {memory_after_cleanup:.1f}MB")
        print(
            f"Memory increase: {memory_increase:.1f}MB, Recovered: {memory_recovered:.1f}MB"
        )
        print(f"Recovery rate: {recovery_rate:.1%}")

        # Assertions
        assert (
            memory_increase < 500
        )  # Creating resources shouldn't use excessive memory
        assert recovery_rate > 0.5  # Should recover at least 50% of memory
        assert (
            memory_after_cleanup - initial_memory < 100
        )  # Final increase should be minimal


class TestStressAndReliability:
    """Stress testing and reliability under extreme conditions."""

    def test_sustained_load_stability(self, api_client: TestClient):
        """Test system stability under sustained load."""
        # Run sustained operations for a period of time
        duration_seconds = 30  # Adjust based on test environment
        start_time = time.time()

        operation_count = 0
        error_count = 0
        response_times = []

        # Prepare test data
        dataset = generate_test_dataset(200, 5)
        csv_data = dataset.to_csv(index=False)

        # Upload base dataset
        upload_response = api_client.post(
            "/api/datasets/",
            files={
                "file": ("sustained_load_dataset.csv", csv_data.encode(), "text/csv")
            },
            data={"name": "Sustained Load Dataset"},
        )
        assert upload_response.status_code == 201
        dataset_id = upload_response.json()["id"]

        # Create base detector
        detector_response = api_client.post(
            "/api/detectors/",
            json={
                "name": "Sustained Load Detector",
                "algorithm": "IsolationForest",
                "hyperparameters": {"contamination": 0.1, "random_state": 42},
            },
        )
        assert detector_response.status_code == 201
        detector_id = detector_response.json()["id"]

        # Train detector
        train_response = api_client.post(
            f"/api/detectors/{detector_id}/train", json={"dataset_id": dataset_id}
        )
        assert train_response.status_code == 200

        # Run sustained operations
        operations = [
            lambda: api_client.get("/api/health/"),
            lambda: api_client.get("/api/datasets/"),
            lambda: api_client.get("/api/detectors/"),
            lambda: api_client.get(f"/api/detectors/{detector_id}"),
            lambda: api_client.post(
                f"/api/detectors/{detector_id}/predict", json={"dataset_id": dataset_id}
            ),
        ]

        while time.time() - start_time < duration_seconds:
            # Select random operation
            operation = np.random.choice(operations)

            # Execute operation
            op_start = time.time()
            try:
                response = operation()
                op_end = time.time()

                response_time = (op_end - op_start) * 1000
                response_times.append(response_time)

                if response.status_code >= 400:
                    error_count += 1

                operation_count += 1

            except Exception as e:
                error_count += 1
                operation_count += 1
                print(f"Operation error: {e}")

            # Small delay to prevent overwhelming
            time.sleep(0.01)

        total_time = time.time() - start_time

        # Calculate metrics
        success_rate = (
            (operation_count - error_count) / operation_count
            if operation_count > 0
            else 0
        )
        operations_per_second = operation_count / total_time
        avg_response_time = statistics.mean(response_times) if response_times else 0
        p95_response_time = np.percentile(response_times, 95) if response_times else 0

        print(f"Sustained load: {operation_count} operations in {total_time:.1f}s")
        print(f"Success rate: {success_rate:.1%}, OPS: {operations_per_second:.1f}")
        print(
            f"Avg response: {avg_response_time:.1f}ms, P95: {p95_response_time:.1f}ms"
        )

        # Assertions
        assert success_rate >= 0.95  # 95% success rate
        assert operations_per_second > 5  # At least 5 operations per second
        assert avg_response_time < 1000  # Average under 1 second
        assert p95_response_time < 5000  # 95th percentile under 5 seconds

        # Clean up
        api_client.delete(f"/api/detectors/{detector_id}")
        api_client.delete(f"/api/datasets/{dataset_id}")

    def test_rapid_scaling_scenario(self, api_client: TestClient):
        """Test rapid scaling up and down of operations."""
        phases = [
            ("ramp_up", 5, 1),  # 5 seconds, 1 operation per second
            ("peak", 10, 5),  # 10 seconds, 5 operations per second
            ("ramp_down", 5, 1),  # 5 seconds, 1 operation per second
        ]

        # Prepare test resources
        dataset = generate_test_dataset(100, 3)
        csv_data = dataset.to_csv(index=False)

        upload_response = api_client.post(
            "/api/datasets/",
            files={"file": ("scaling_dataset.csv", csv_data.encode(), "text/csv")},
            data={"name": "Scaling Test Dataset"},
        )
        assert upload_response.status_code == 201
        dataset_id = upload_response.json()["id"]

        detector_response = api_client.post(
            "/api/detectors/",
            json={
                "name": "Scaling Test Detector",
                "algorithm": "IsolationForest",
                "hyperparameters": {"contamination": 0.1},
            },
        )
        assert detector_response.status_code == 201
        detector_id = detector_response.json()["id"]

        # Train detector
        train_response = api_client.post(
            f"/api/detectors/{detector_id}/train", json={"dataset_id": dataset_id}
        )
        assert train_response.status_code == 200

        # Execute scaling phases
        total_operations = 0
        total_errors = 0
        phase_results = {}

        for phase_name, duration, ops_per_second in phases:
            print(
                f"Starting phase: {phase_name} ({ops_per_second} ops/sec for {duration}s)"
            )

            phase_start = time.time()
            phase_operations = 0
            phase_errors = 0
            phase_response_times = []

            target_interval = 1.0 / ops_per_second
            last_operation_time = phase_start

            while time.time() - phase_start < duration:
                current_time = time.time()

                # Check if it's time for next operation
                if current_time - last_operation_time >= target_interval:
                    op_start = time.time()
                    try:
                        # Alternate between different operations
                        if phase_operations % 3 == 0:
                            response = api_client.get("/api/health/")
                        elif phase_operations % 3 == 1:
                            response = api_client.get(f"/api/detectors/{detector_id}")
                        else:
                            response = api_client.post(
                                f"/api/detectors/{detector_id}/predict",
                                json={"dataset_id": dataset_id},
                            )

                        op_end = time.time()
                        phase_response_times.append((op_end - op_start) * 1000)

                        if response.status_code >= 400:
                            phase_errors += 1

                        phase_operations += 1
                        last_operation_time = current_time

                    except Exception as e:
                        phase_errors += 1
                        phase_operations += 1
                        print(f"Phase {phase_name} error: {e}")

                time.sleep(0.001)  # Small sleep to prevent busy waiting

            # Record phase results
            phase_duration = time.time() - phase_start
            phase_success_rate = (
                (phase_operations - phase_errors) / phase_operations
                if phase_operations > 0
                else 0
            )
            phase_avg_response = (
                statistics.mean(phase_response_times) if phase_response_times else 0
            )

            phase_results[phase_name] = {
                "operations": phase_operations,
                "errors": phase_errors,
                "success_rate": phase_success_rate,
                "avg_response_ms": phase_avg_response,
                "duration": phase_duration,
                "actual_ops_per_sec": phase_operations / phase_duration,
            }

            total_operations += phase_operations
            total_errors += phase_errors

            print(
                f"Phase {phase_name}: {phase_operations} ops, {phase_success_rate:.1%} success"
            )

        # Analyze overall results
        overall_success_rate = (
            (total_operations - total_errors) / total_operations
            if total_operations > 0
            else 0
        )

        print(f"Scaling test complete: {total_operations} total operations")
        print(f"Overall success rate: {overall_success_rate:.1%}")

        for phase_name, results in phase_results.items():
            print(
                f"{phase_name}: {results['actual_ops_per_sec']:.1f} ops/sec, "
                f"{results['avg_response_ms']:.1f}ms avg"
            )

        # Assertions
        assert overall_success_rate >= 0.9  # 90% overall success rate

        # Each phase should meet minimum requirements
        for phase_name, results in phase_results.items():
            assert results["success_rate"] >= 0.85  # 85% success rate per phase
            assert results["avg_response_ms"] < 2000  # Average under 2 seconds

        # Peak phase should achieve higher throughput
        peak_results = phase_results["peak"]
        ramp_up_results = phase_results["ramp_up"]
        assert (
            peak_results["actual_ops_per_sec"] > ramp_up_results["actual_ops_per_sec"]
        )

        # Clean up
        api_client.delete(f"/api/detectors/{detector_id}")
        api_client.delete(f"/api/datasets/{dataset_id}")


if __name__ == "__main__":
    # Enable running load tests directly
    pytest.main([__file__, "-v", "-s"])
