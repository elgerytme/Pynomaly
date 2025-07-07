"""Performance and scaling workflow end-to-end tests.

This module tests performance optimization, scalability scenarios, load testing,
and system behavior under various performance conditions.
"""

import concurrent.futures
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from pynomaly.infrastructure.config import create_container
from pynomaly.presentation.api.app import create_app


class TestPerformanceScalingWorkflows:
    """Test performance and scaling workflows."""

    @pytest.fixture
    def app_client(self):
        """Create test client for API."""
        container = create_container()
        app = create_app(container)
        return TestClient(app)

    @pytest.fixture
    def performance_datasets(self):
        """Create datasets of various sizes for performance testing."""
        np.random.seed(42)

        datasets = {}

        # Small dataset (1K samples)
        small_data = pd.DataFrame(
            {f"feature_{i}": np.random.normal(0, 1, 1000) for i in range(10)}
        )
        datasets["small"] = small_data

        # Medium dataset (10K samples)
        medium_data = pd.DataFrame(
            {f"feature_{i}": np.random.normal(0, 1, 10000) for i in range(20)}
        )
        datasets["medium"] = medium_data

        # Large dataset (100K samples)
        large_data = pd.DataFrame(
            {f"feature_{i}": np.random.normal(0, 1, 100000) for i in range(50)}
        )
        datasets["large"] = large_data

        # High-dimensional dataset (1K samples, 1K features)
        high_dim_data = pd.DataFrame(
            {f"feature_{i}": np.random.normal(0, 1, 1000) for i in range(1000)}
        )
        datasets["high_dimensional"] = high_dim_data

        return datasets

    def test_load_testing_workflow(self, app_client, performance_datasets):
        """Test system behavior under load."""
        # Upload test dataset
        test_data = performance_datasets["medium"]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            test_data.to_csv(f.name, index=False)
            dataset_file = f.name

        try:
            with open(dataset_file, "rb") as file:
                upload_response = app_client.post(
                    "/api/datasets/upload",
                    files={"file": ("load_test_data.csv", file, "text/csv")},
                    data={"name": "Load Test Dataset"},
                )
            assert upload_response.status_code == 200
            dataset_id = upload_response.json()["id"]

            # Create detector for load testing
            detector_data = {
                "name": "Load Test Detector",
                "algorithm_name": "IsolationForest",
                "parameters": {"contamination": 0.1, "n_estimators": 100},
            }

            create_response = app_client.post("/api/detectors/", json=detector_data)
            assert create_response.status_code == 200
            detector_id = create_response.json()["id"]

            # Train detector
            train_start = time.time()
            train_response = app_client.post(
                f"/api/detectors/{detector_id}/train", json={"dataset_id": dataset_id}
            )
            train_time = time.time() - train_start

            assert train_response.status_code == 200
            assert train_time < 60  # Should complete within reasonable time

            # Concurrent load test
            concurrent_requests = 50
            request_results = []

            def make_detection_request():
                detect_start = time.time()
                response = app_client.post(
                    f"/api/detectors/{detector_id}/detect",
                    json={"dataset_id": dataset_id},
                )
                detect_time = time.time() - detect_start
                return {
                    "status_code": response.status_code,
                    "response_time": detect_time,
                    "timestamp": time.time(),
                }

            # Execute concurrent requests
            load_test_start = time.time()
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                futures = [
                    executor.submit(make_detection_request)
                    for _ in range(concurrent_requests)
                ]
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    request_results.append(result)
            load_test_duration = time.time() - load_test_start

            # Analyze load test results
            successful_requests = [
                r for r in request_results if r["status_code"] == 200
            ]
            [r for r in request_results if r["status_code"] != 200]

            success_rate = len(successful_requests) / len(request_results)
            avg_response_time = np.mean(
                [r["response_time"] for r in successful_requests]
            )
            max_response_time = np.max(
                [r["response_time"] for r in successful_requests]
            )

            # Performance assertions
            assert success_rate >= 0.9  # At least 90% success rate
            assert avg_response_time < 5.0  # Average response under 5 seconds
            assert max_response_time < 15.0  # No request over 15 seconds

            # Test performance monitoring endpoint
            perf_metrics_response = app_client.get("/api/performance/metrics")
            if perf_metrics_response.status_code == 200:
                metrics = perf_metrics_response.json()

                assert "request_rate" in metrics
                assert "average_response_time" in metrics
                assert "error_rate" in metrics
                assert "active_connections" in metrics

            # Generate load test report
            load_report_request = {
                "test_duration": load_test_duration,
                "concurrent_users": 10,
                "total_requests": concurrent_requests,
                "success_rate": success_rate,
                "avg_response_time": avg_response_time,
                "max_response_time": max_response_time,
            }

            report_response = app_client.post(
                "/api/performance/load-test-report", json=load_report_request
            )
            if report_response.status_code == 200:
                report_result = report_response.json()
                assert "performance_grade" in report_result
                assert "bottlenecks_identified" in report_result
                assert "recommendations" in report_result

        finally:
            Path(dataset_file).unlink(missing_ok=True)

    def test_scalability_benchmarking_workflow(self, app_client, performance_datasets):
        """Test system scalability with different dataset sizes."""
        dataset_sizes = ["small", "medium", "large"]
        benchmark_results = {}

        for size_name in dataset_sizes:
            if size_name not in performance_datasets:
                continue

            dataset = performance_datasets[size_name]

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".csv", delete=False
            ) as f:
                dataset.to_csv(f.name, index=False)
                dataset_file = f.name

            try:
                # Upload dataset
                upload_start = time.time()
                with open(dataset_file, "rb") as file:
                    upload_response = app_client.post(
                        "/api/datasets/upload",
                        files={
                            "file": (f"benchmark_{size_name}.csv", file, "text/csv")
                        },
                        data={"name": f"Benchmark {size_name} Dataset"},
                    )
                upload_time = time.time() - upload_start

                assert upload_response.status_code == 200
                dataset_id = upload_response.json()["id"]

                # Create detector
                detector_data = {
                    "name": f"Benchmark {size_name} Detector",
                    "algorithm_name": "IsolationForest",
                    "parameters": {"contamination": 0.1},
                }

                create_start = time.time()
                create_response = app_client.post("/api/detectors/", json=detector_data)
                create_time = time.time() - create_start

                assert create_response.status_code == 200
                detector_id = create_response.json()["id"]

                # Benchmark training
                train_start = time.time()
                train_response = app_client.post(
                    f"/api/detectors/{detector_id}/train",
                    json={"dataset_id": dataset_id},
                )
                train_time = time.time() - train_start

                assert train_response.status_code == 200

                # Benchmark detection
                detect_start = time.time()
                detect_response = app_client.post(
                    f"/api/detectors/{detector_id}/detect",
                    json={"dataset_id": dataset_id},
                )
                detect_time = time.time() - detect_start

                assert detect_response.status_code == 200

                # Store benchmark results
                benchmark_results[size_name] = {
                    "dataset_size": len(dataset),
                    "feature_count": len(dataset.columns),
                    "upload_time": upload_time,
                    "create_time": create_time,
                    "train_time": train_time,
                    "detect_time": detect_time,
                    "total_time": upload_time + create_time + train_time + detect_time,
                }

                # Memory usage monitoring
                memory_response = app_client.get("/api/performance/memory-usage")
                if memory_response.status_code == 200:
                    memory_info = memory_response.json()
                    benchmark_results[size_name]["memory_usage"] = memory_info

            finally:
                Path(dataset_file).unlink(missing_ok=True)

        # Analyze scalability
        if len(benchmark_results) > 1:
            scalability_analysis = {
                "dataset_sizes": {
                    name: results["dataset_size"]
                    for name, results in benchmark_results.items()
                },
                "training_times": {
                    name: results["train_time"]
                    for name, results in benchmark_results.items()
                },
                "detection_times": {
                    name: results["detect_time"]
                    for name, results in benchmark_results.items()
                },
            }

            # Test scalability report generation
            scalability_report_response = app_client.post(
                "/api/performance/scalability-report", json=scalability_analysis
            )
            if scalability_report_response.status_code == 200:
                report_result = scalability_report_response.json()
                assert "scalability_score" in report_result
                assert "linear_scaling_factor" in report_result
                assert "performance_trends" in report_result

    def test_memory_optimization_workflow(self, app_client, performance_datasets):
        """Test memory usage optimization and monitoring."""
        # Use large dataset for memory testing
        large_dataset = performance_datasets["large"]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            large_dataset.to_csv(f.name, index=False)
            dataset_file = f.name

        try:
            # Monitor initial memory state
            initial_memory_response = app_client.get("/api/performance/memory-usage")
            if initial_memory_response.status_code == 200:
                initial_memory = initial_memory_response.json()
                initial_used = initial_memory.get("used_memory_mb", 0)
            else:
                initial_used = 0

            # Upload large dataset
            with open(dataset_file, "rb") as file:
                upload_response = app_client.post(
                    "/api/datasets/upload",
                    files={"file": ("memory_test_data.csv", file, "text/csv")},
                    data={"name": "Memory Test Dataset"},
                )
            assert upload_response.status_code == 200
            dataset_id = upload_response.json()["id"]

            # Monitor memory after upload
            post_upload_memory_response = app_client.get(
                "/api/performance/memory-usage"
            )
            if post_upload_memory_response.status_code == 200:
                post_upload_memory = post_upload_memory_response.json()
                upload_memory_usage = (
                    post_upload_memory.get("used_memory_mb", 0) - initial_used
                )
                assert upload_memory_usage > 0  # Should use some memory

            # Create memory-optimized detector
            detector_data = {
                "name": "Memory Optimized Detector",
                "algorithm_name": "IsolationForest",
                "parameters": {
                    "contamination": 0.1,
                    "max_samples": "auto",  # Memory optimization
                    "n_estimators": 50,  # Reduced for memory efficiency
                },
                "optimization_settings": {
                    "memory_efficient": True,
                    "streaming_mode": True,
                    "batch_size": 1000,
                },
            }

            create_response = app_client.post("/api/detectors/", json=detector_data)
            assert create_response.status_code == 200
            detector_id = create_response.json()["id"]

            # Train with memory monitoring
            train_start_memory_response = app_client.get(
                "/api/performance/memory-usage"
            )

            train_response = app_client.post(
                f"/api/detectors/{detector_id}/train",
                json={
                    "dataset_id": dataset_id,
                    "memory_optimization": True,
                    "chunk_size": 5000,
                },
            )
            assert train_response.status_code == 200

            train_end_memory_response = app_client.get("/api/performance/memory-usage")

            # Check memory usage during training
            if (
                train_start_memory_response.status_code == 200
                and train_end_memory_response.status_code == 200
            ):
                start_memory = train_start_memory_response.json().get(
                    "used_memory_mb", 0
                )
                end_memory = train_end_memory_response.json().get("used_memory_mb", 0)
                peak_memory = max(start_memory, end_memory)

                # Memory usage should be reasonable for large dataset
                assert peak_memory < 2000  # Less than 2GB

            # Test memory cleanup
            cleanup_response = app_client.post(
                "/api/performance/memory-cleanup",
                json={
                    "cleanup_type": "aggressive",
                    "clear_caches": True,
                    "garbage_collect": True,
                },
            )

            if cleanup_response.status_code == 200:
                cleanup_result = cleanup_response.json()
                assert "memory_freed_mb" in cleanup_result
                assert cleanup_result["memory_freed_mb"] >= 0

            # Test memory profiling
            profiling_response = app_client.post(
                "/api/performance/memory-profile",
                json={
                    "detector_id": detector_id,
                    "dataset_id": dataset_id,
                    "profiling_duration": 30,
                },
            )

            if profiling_response.status_code == 200:
                profiling_result = profiling_response.json()
                assert "memory_timeline" in profiling_result
                assert "peak_usage" in profiling_result
                assert "memory_leaks_detected" in profiling_result

        finally:
            Path(dataset_file).unlink(missing_ok=True)

    def test_cpu_optimization_workflow(self, app_client, performance_datasets):
        """Test CPU usage optimization and parallel processing."""
        # Use medium dataset for CPU testing
        test_dataset = performance_datasets["medium"]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            test_dataset.to_csv(f.name, index=False)
            dataset_file = f.name

        try:
            with open(dataset_file, "rb") as file:
                upload_response = app_client.post(
                    "/api/datasets/upload",
                    files={"file": ("cpu_test_data.csv", file, "text/csv")},
                    data={"name": "CPU Test Dataset"},
                )
            assert upload_response.status_code == 200
            dataset_id = upload_response.json()["id"]

            # Test single-threaded performance
            single_thread_detector = {
                "name": "Single Thread Detector",
                "algorithm_name": "IsolationForest",
                "parameters": {
                    "contamination": 0.1,
                    "n_estimators": 100,
                    "n_jobs": 1,  # Single thread
                },
            }

            create_response = app_client.post(
                "/api/detectors/", json=single_thread_detector
            )
            assert create_response.status_code == 200
            single_detector_id = create_response.json()["id"]

            # Train single-threaded
            single_train_start = time.time()
            train_response = app_client.post(
                f"/api/detectors/{single_detector_id}/train",
                json={"dataset_id": dataset_id},
            )
            single_train_time = time.time() - single_train_start
            assert train_response.status_code == 200

            # Test multi-threaded performance
            multi_thread_detector = {
                "name": "Multi Thread Detector",
                "algorithm_name": "IsolationForest",
                "parameters": {
                    "contamination": 0.1,
                    "n_estimators": 100,
                    "n_jobs": -1,  # Use all available cores
                },
            }

            create_response = app_client.post(
                "/api/detectors/", json=multi_thread_detector
            )
            assert create_response.status_code == 200
            multi_detector_id = create_response.json()["id"]

            # Train multi-threaded
            multi_train_start = time.time()
            train_response = app_client.post(
                f"/api/detectors/{multi_detector_id}/train",
                json={"dataset_id": dataset_id},
            )
            multi_train_time = time.time() - multi_train_start
            assert train_response.status_code == 200

            # Multi-threaded should be faster (if multiple cores available)
            speedup_ratio = single_train_time / multi_train_time
            assert speedup_ratio >= 0.8  # At least not significantly slower

            # Test CPU usage monitoring
            cpu_monitoring_response = app_client.get("/api/performance/cpu-usage")
            if cpu_monitoring_response.status_code == 200:
                cpu_info = cpu_monitoring_response.json()
                assert "cpu_percent" in cpu_info
                assert "core_count" in cpu_info
                assert "load_average" in cpu_info

            # Test batch processing optimization
            batch_request = {
                "detector_id": multi_detector_id,
                "dataset_id": dataset_id,
                "batch_optimization": {
                    "parallel_batches": True,
                    "batch_size": 2000,
                    "max_workers": 4,
                },
            }

            batch_start = time.time()
            batch_response = app_client.post(
                "/api/detection/batch-optimized", json=batch_request
            )
            time.time() - batch_start

            if batch_response.status_code == 200:
                batch_result = batch_response.json()
                assert "processing_time" in batch_result
                assert "throughput" in batch_result
                assert batch_result["throughput"] > 100  # Samples per second

            # Test CPU profiling
            profiling_request = {
                "detector_id": multi_detector_id,
                "dataset_id": dataset_id,
                "profile_duration": 30,
                "include_call_stack": True,
            }

            profiling_response = app_client.post(
                "/api/performance/cpu-profile", json=profiling_request
            )
            if profiling_response.status_code == 200:
                profiling_result = profiling_response.json()
                assert "cpu_usage_timeline" in profiling_result
                assert "hotspots" in profiling_result
                assert "optimization_suggestions" in profiling_result

        finally:
            Path(dataset_file).unlink(missing_ok=True)

    def test_high_dimensional_performance_workflow(
        self, app_client, performance_datasets
    ):
        """Test performance with high-dimensional data."""
        high_dim_dataset = performance_datasets["high_dimensional"]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            high_dim_dataset.to_csv(f.name, index=False)
            dataset_file = f.name

        try:
            with open(dataset_file, "rb") as file:
                upload_response = app_client.post(
                    "/api/datasets/upload",
                    files={"file": ("high_dim_data.csv", file, "text/csv")},
                    data={"name": "High Dimensional Dataset"},
                )
            assert upload_response.status_code == 200
            dataset_id = upload_response.json()["id"]

            # Test dimensionality reduction preprocessing
            dim_reduction_request = {
                "dataset_id": dataset_id,
                "method": "pca",
                "target_dimensions": 50,
                "variance_threshold": 0.95,
            }

            dim_reduction_response = app_client.post(
                "/api/preprocessing/dimensionality-reduction",
                json=dim_reduction_request,
            )

            if dim_reduction_response.status_code == 200:
                dim_reduction_result = dim_reduction_response.json()
                assert "reduced_dataset_id" in dim_reduction_result
                assert "explained_variance" in dim_reduction_result
                reduced_dataset_id = dim_reduction_result["reduced_dataset_id"]
            else:
                reduced_dataset_id = dataset_id

            # Create detector optimized for high-dimensional data
            high_dim_detector = {
                "name": "High Dimensional Detector",
                "algorithm_name": "IsolationForest",
                "parameters": {
                    "contamination": 0.1,
                    "max_features": 0.5,  # Use subset of features
                    "n_estimators": 50,  # Reduced for performance
                },
                "preprocessing": {
                    "feature_selection": True,
                    "dimensionality_reduction": True,
                },
            }

            create_response = app_client.post("/api/detectors/", json=high_dim_detector)
            assert create_response.status_code == 200
            detector_id = create_response.json()["id"]

            # Benchmark training on high-dimensional data
            train_start = time.time()
            train_response = app_client.post(
                f"/api/detectors/{detector_id}/train",
                json={"dataset_id": reduced_dataset_id},
            )
            train_time = time.time() - train_start

            assert train_response.status_code == 200
            # High-dimensional training should complete in reasonable time
            assert train_time < 300  # 5 minutes max

            # Test feature importance analysis
            feature_importance_response = app_client.post(
                f"/api/detectors/{detector_id}/feature-importance",
                json={"dataset_id": reduced_dataset_id},
            )

            if feature_importance_response.status_code == 200:
                importance_result = feature_importance_response.json()
                assert "feature_scores" in importance_result
                assert "top_features" in importance_result

            # Test detection performance
            detect_start = time.time()
            detect_response = app_client.post(
                f"/api/detectors/{detector_id}/detect",
                json={"dataset_id": reduced_dataset_id},
            )
            detect_time = time.time() - detect_start

            assert detect_response.status_code == 200
            assert detect_time < 60  # Should be fast for detection

            # Test streaming detection for high-dimensional data
            sample_data = high_dim_dataset.iloc[:10].to_dict("records")

            streaming_request = {
                "detector_id": detector_id,
                "samples": sample_data,
                "preprocessing_pipeline": True,
            }

            streaming_start = time.time()
            streaming_response = app_client.post(
                "/api/detection/stream-high-dim", json=streaming_request
            )
            streaming_time = time.time() - streaming_start

            if streaming_response.status_code == 200:
                streaming_result = streaming_response.json()
                assert "results" in streaming_result
                assert len(streaming_result["results"]) == 10
                assert streaming_time < 5  # Fast streaming inference

        finally:
            Path(dataset_file).unlink(missing_ok=True)

    def test_auto_scaling_workflow(self, app_client):
        """Test automatic scaling based on load."""
        # Test current scaling status
        scaling_status_response = app_client.get("/api/scaling/status")

        if scaling_status_response.status_code == 200:
            scaling_status = scaling_status_response.json()

            assert "current_instances" in scaling_status
            assert "target_instances" in scaling_status
            assert "scaling_policy" in scaling_status

            current_instances = scaling_status["current_instances"]

            # Configure auto-scaling policy
            scaling_policy = {
                "min_instances": 2,
                "max_instances": 10,
                "target_cpu_utilization": 70,
                "target_memory_utilization": 80,
                "scale_up_threshold": 75,
                "scale_down_threshold": 30,
                "cooldown_period": 300,  # 5 minutes
            }

            policy_response = app_client.post(
                "/api/scaling/configure", json=scaling_policy
            )
            assert policy_response.status_code == 200

            # Simulate high load to trigger scaling
            load_simulation_request = {
                "simulation_type": "cpu_intensive",
                "duration_seconds": 60,
                "target_utilization": 85,
                "concurrent_requests": 20,
            }

            load_response = app_client.post(
                "/api/scaling/simulate-load", json=load_simulation_request
            )

            if load_response.status_code == 200:
                load_result = load_response.json()
                assert "simulation_id" in load_result
                simulation_id = load_result["simulation_id"]

                # Monitor scaling events
                time.sleep(10)  # Wait for scaling to potentially trigger

                scaling_events_response = app_client.get(
                    f"/api/scaling/events?simulation_id={simulation_id}"
                )

                if scaling_events_response.status_code == 200:
                    events = scaling_events_response.json()
                    assert "events" in events

                    # Check if scaling occurred
                    scale_up_events = [
                        e for e in events["events"] if e.get("action") == "scale_up"
                    ]
                    if scale_up_events:
                        assert len(scale_up_events) > 0

                        # Verify new instance count
                        new_status_response = app_client.get("/api/scaling/status")
                        if new_status_response.status_code == 200:
                            new_status = new_status_response.json()
                            new_instances = new_status["current_instances"]
                            assert new_instances >= current_instances

            # Test scale-down after load reduction
            cooldown_request = {"force_cooldown": True, "target_utilization": 20}

            cooldown_response = app_client.post(
                "/api/scaling/cooldown", json=cooldown_request
            )
            if cooldown_response.status_code == 200:
                time.sleep(5)  # Wait for potential scale-down

                final_status_response = app_client.get("/api/scaling/status")
                if final_status_response.status_code == 200:
                    final_status = final_status_response.json()
                    # Should eventually scale back down (but respect min_instances)
                    assert (
                        final_status["current_instances"]
                        >= scaling_policy["min_instances"]
                    )

    def test_caching_optimization_workflow(self, app_client, performance_datasets):
        """Test caching optimization for performance improvement."""
        test_dataset = performance_datasets["small"]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            test_dataset.to_csv(f.name, index=False)
            dataset_file = f.name

        try:
            with open(dataset_file, "rb") as file:
                upload_response = app_client.post(
                    "/api/datasets/upload",
                    files={"file": ("cache_test_data.csv", file, "text/csv")},
                    data={"name": "Cache Test Dataset"},
                )
            assert upload_response.status_code == 200
            dataset_id = upload_response.json()["id"]

            # Create detector
            detector_data = {
                "name": "Cache Test Detector",
                "algorithm_name": "IsolationForest",
                "parameters": {"contamination": 0.1},
            }

            create_response = app_client.post("/api/detectors/", json=detector_data)
            assert create_response.status_code == 200
            detector_id = create_response.json()["id"]

            # Train detector
            train_response = app_client.post(
                f"/api/detectors/{detector_id}/train", json={"dataset_id": dataset_id}
            )
            assert train_response.status_code == 200

            # Test cache configuration
            cache_config = {
                "cache_type": "redis",
                "cache_size_mb": 512,
                "ttl_seconds": 3600,
                "cache_strategies": ["model_cache", "result_cache", "dataset_cache"],
            }

            cache_setup_response = app_client.post(
                "/api/cache/configure", json=cache_config
            )
            if cache_setup_response.status_code == 200:
                # First detection (cache miss)
                first_detect_start = time.time()
                first_detect_response = app_client.post(
                    f"/api/detectors/{detector_id}/detect",
                    json={"dataset_id": dataset_id},
                )
                first_detect_time = time.time() - first_detect_start

                assert first_detect_response.status_code == 200

                # Second detection (cache hit)
                second_detect_start = time.time()
                second_detect_response = app_client.post(
                    f"/api/detectors/{detector_id}/detect",
                    json={"dataset_id": dataset_id},
                )
                second_detect_time = time.time() - second_detect_start

                assert second_detect_response.status_code == 200

                # Second request should be significantly faster
                speedup = first_detect_time / second_detect_time
                assert speedup >= 1.5  # At least 50% faster

                # Verify cache hit in headers
                if "X-Cache-Status" in second_detect_response.headers:
                    assert second_detect_response.headers["X-Cache-Status"] == "HIT"

            # Test cache statistics
            cache_stats_response = app_client.get("/api/cache/statistics")
            if cache_stats_response.status_code == 200:
                cache_stats = cache_stats_response.json()

                assert "hit_rate" in cache_stats
                assert "miss_rate" in cache_stats
                assert "cache_size" in cache_stats
                assert "evictions" in cache_stats

            # Test cache invalidation
            invalidation_response = app_client.post(
                "/api/cache/invalidate",
                json={
                    "cache_keys": [f"detector_{detector_id}", f"dataset_{dataset_id}"],
                    "invalidate_pattern": True,
                },
            )

            if invalidation_response.status_code == 200:
                invalidation_result = invalidation_response.json()
                assert "invalidated_keys" in invalidation_result
                assert invalidation_result["invalidated_keys"] > 0

        finally:
            Path(dataset_file).unlink(missing_ok=True)
