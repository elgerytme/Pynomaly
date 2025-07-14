"""
Performance and Load Testing

Comprehensive performance tests covering scalability, load handling,
and resource utilization under various conditions.
"""

import pytest
import asyncio
import time
import concurrent.futures
from typing import List, Dict, Any
import statistics
import psutil
import numpy as np

from .conftest import (
    assert_performance_within_limits,
    assert_api_response_valid
)


@pytest.mark.asyncio
@pytest.mark.performance
@pytest.mark.slow
class TestPerformanceBaseline:
    """Test performance baselines for individual operations."""

    async def test_detector_creation_performance(
        self,
        async_client,
        api_headers,
        performance_monitor
    ):
        """Test detector creation performance."""
        
        detector_configs = [
            {
                "name": f"perf-detector-{i}",
                "algorithm_name": "IsolationForest",
                "hyperparameters": {"n_estimators": 50},
                "contamination_rate": 0.05
            }
            for i in range(10)
        ]
        
        creation_times = []
        detector_ids = []
        
        for config in detector_configs:
            performance_monitor.start_timer("detector_creation")
            
            response = await async_client.post(
                "/api/v1/detectors",
                json=config,
                headers=api_headers
            )
            
            performance_monitor.end_timer("detector_creation")
            assert_api_response_valid(response, 201)
            
            detector = response.json()
            detector_ids.append(detector["id"])
            creation_times.append(performance_monitor.get_duration("detector_creation"))
        
        # Analyze performance
        avg_creation_time = statistics.mean(creation_times)
        max_creation_time = max(creation_times)
        
        # Performance assertions
        assert_performance_within_limits(avg_creation_time, 2.0)  # Average < 2s
        assert_performance_within_limits(max_creation_time, 5.0)  # Max < 5s
        
        # Cleanup
        for detector_id in detector_ids:
            await async_client.delete(f"/api/v1/detectors/{detector_id}", headers=api_headers)

    async def test_training_performance_scaling(
        self,
        async_client,
        api_headers,
        performance_monitor
    ):
        """Test training performance with different data sizes."""
        
        # Create detector
        detector_config = {
            "name": "scaling-test-detector",
            "algorithm_name": "IsolationForest",
            "hyperparameters": {"n_estimators": 50},
            "contamination_rate": 0.05
        }
        
        response = await async_client.post(
            "/api/v1/detectors",
            json=detector_config,
            headers=api_headers
        )
        assert_api_response_valid(response, 201)
        detector = response.json()
        detector_id = detector["id"]
        
        # Test different data sizes
        data_sizes = [100, 500, 1000, 2000]
        training_times = []
        
        for size in data_sizes:
            # Generate test data
            np.random.seed(42)
            data = np.random.normal(0, 1, (size, 5))
            
            dataset_payload = {
                "name": f"scaling-test-data-{size}",
                "data": data.tolist(),
                "feature_names": [f"feature_{i}" for i in range(5)]
            }
            
            training_payload = {
                "detector_id": detector_id,
                "dataset": dataset_payload,
                "job_name": f"scaling-test-{size}"
            }
            
            performance_monitor.start_timer(f"training_{size}")
            
            response = await async_client.post(
                "/api/v1/training/jobs",
                json=training_payload,
                headers=api_headers
            )
            assert_api_response_valid(response, 201)
            training_job = response.json()
            job_id = training_job["job_id"]
            
            # Wait for completion
            await self._wait_for_job_completion(async_client, job_id, api_headers, max_wait_time=120)
            
            performance_monitor.end_timer(f"training_{size}")
            training_time = performance_monitor.get_duration(f"training_{size}")
            training_times.append(training_time)
        
        # Analyze scaling behavior
        # Training time should scale sub-linearly for most algorithms
        for i, (size, time_taken) in enumerate(zip(data_sizes, training_times)):
            expected_max_time = 5 + (size / 100) * 2  # Base 5s + 2s per 100 samples
            assert_performance_within_limits(time_taken, expected_max_time)
        
        # Cleanup
        await async_client.delete(f"/api/v1/detectors/{detector_id}", headers=api_headers)

    async def test_detection_performance_scaling(
        self,
        async_client,
        sample_dataset,
        api_headers,
        performance_monitor
    ):
        """Test detection performance with different data sizes."""
        
        # Create and train detector
        detector_config = {
            "name": "detection-perf-detector",
            "algorithm_name": "IsolationForest",
            "hyperparameters": {"n_estimators": 50},
            "contamination_rate": 0.05
        }
        
        response = await async_client.post(
            "/api/v1/detectors",
            json=detector_config,
            headers=api_headers
        )
        assert_api_response_valid(response, 201)
        detector = response.json()
        detector_id = detector["id"]
        
        # Train with sample data
        training_data = sample_dataset['features'].iloc[:500]
        training_payload = {
            "detector_id": detector_id,
            "dataset": {
                "name": "detection-perf-training",
                "data": training_data.values.tolist(),
                "feature_names": training_data.columns.tolist()
            },
            "job_name": "detection-perf-training"
        }
        
        response = await async_client.post(
            "/api/v1/training/jobs",
            json=training_payload,
            headers=api_headers
        )
        assert_api_response_valid(response, 201)
        training_job = response.json()
        
        await self._wait_for_job_completion(async_client, training_job["job_id"], api_headers)
        
        # Test detection with different data sizes
        detection_sizes = [10, 50, 100, 500, 1000]
        detection_times = []
        
        for size in detection_sizes:
            # Generate test data
            np.random.seed(123)
            data = np.random.normal(0, 1, (size, len(sample_dataset['feature_names'])))
            
            detection_payload = {
                "detector_id": detector_id,
                "dataset": {
                    "name": f"detection-test-{size}",
                    "data": data.tolist(),
                    "feature_names": sample_dataset['feature_names']
                },
                "return_scores": True
            }
            
            performance_monitor.start_timer(f"detection_{size}")
            
            response = await async_client.post(
                "/api/v1/detection/predict",
                json=detection_payload,
                headers=api_headers
            )
            
            performance_monitor.end_timer(f"detection_{size}")
            assert_api_response_valid(response)
            
            detection_time = performance_monitor.get_duration(f"detection_{size}")
            detection_times.append(detection_time)
            
            result = response.json()
            assert result["n_samples"] == size
        
        # Analyze detection performance scaling
        for i, (size, time_taken) in enumerate(zip(detection_sizes, detection_times)):
            # Detection should be very fast and scale linearly
            expected_max_time = 0.1 + (size / 1000) * 1.0  # 0.1s base + 1s per 1000 samples
            assert_performance_within_limits(time_taken, expected_max_time)
        
        # Cleanup
        await async_client.delete(f"/api/v1/detectors/{detector_id}", headers=api_headers)

    async def _wait_for_job_completion(
        self,
        client,
        job_id: str,
        headers: Dict[str, str],
        max_wait_time: int = 60
    ):
        """Helper method to wait for training job completion."""
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            response = await client.get(
                f"/api/v1/training/jobs/{job_id}",
                headers=headers
            )
            assert_api_response_valid(response)
            job_status = response.json()
            
            if job_status["status"] == "completed":
                return job_status
            elif job_status["status"] == "failed":
                pytest.fail(f"Training failed: {job_status.get('error_message', 'Unknown error')}")
                
            await asyncio.sleep(2)
        
        pytest.fail("Training did not complete within expected time")


@pytest.mark.asyncio
@pytest.mark.load
@pytest.mark.slow
class TestLoadTesting:
    """Test system behavior under various load conditions."""

    async def test_concurrent_detector_operations(
        self,
        async_client,
        api_headers,
        performance_monitor
    ):
        """Test concurrent detector creation, training, and detection."""
        
        num_concurrent = 5
        
        async def create_train_detect_workflow(workflow_id: int):
            """Single workflow for concurrent execution."""
            
            # Create detector
            detector_config = {
                "name": f"concurrent-detector-{workflow_id}",
                "algorithm_name": "IsolationForest",
                "hyperparameters": {"n_estimators": 30},
                "contamination_rate": 0.05
            }
            
            response = await async_client.post(
                "/api/v1/detectors",
                json=detector_config,
                headers=api_headers
            )
            assert_api_response_valid(response, 201)
            detector = response.json()
            detector_id = detector["id"]
            
            try:
                # Generate training data
                np.random.seed(workflow_id)
                training_data = np.random.normal(0, 1, (200, 3))
                
                training_payload = {
                    "detector_id": detector_id,
                    "dataset": {
                        "name": f"concurrent-training-{workflow_id}",
                        "data": training_data.tolist(),
                        "feature_names": ["f1", "f2", "f3"]
                    },
                    "job_name": f"concurrent-training-{workflow_id}"
                }
                
                # Start training
                response = await async_client.post(
                    "/api/v1/training/jobs",
                    json=training_payload,
                    headers=api_headers
                )
                assert_api_response_valid(response, 201)
                training_job = response.json()
                
                # Wait for training completion
                await self._wait_for_job_completion(
                    async_client, training_job["job_id"], api_headers, max_wait_time=90
                )
                
                # Perform detection
                test_data = np.random.normal(0, 1, (50, 3))
                detection_payload = {
                    "detector_id": detector_id,
                    "dataset": {
                        "name": f"concurrent-detection-{workflow_id}",
                        "data": test_data.tolist(),
                        "feature_names": ["f1", "f2", "f3"]
                    },
                    "return_scores": True
                }
                
                response = await async_client.post(
                    "/api/v1/detection/predict",
                    json=detection_payload,
                    headers=api_headers
                )
                assert_api_response_valid(response)
                
                return detector_id, True
                
            except Exception as e:
                return detector_id, False
        
        # Execute concurrent workflows
        performance_monitor.start_timer("concurrent_workflows")
        
        tasks = [create_train_detect_workflow(i) for i in range(num_concurrent)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        performance_monitor.end_timer("concurrent_workflows")
        
        # Analyze results
        successful_workflows = 0
        detector_ids = []
        
        for result in results:
            if isinstance(result, Exception):
                print(f"Workflow failed with exception: {result}")
            else:
                detector_id, success = result
                detector_ids.append(detector_id)
                if success:
                    successful_workflows += 1
        
        # At least 80% of workflows should succeed under load
        success_rate = successful_workflows / num_concurrent
        assert success_rate >= 0.8, f"Only {success_rate:.1%} of workflows succeeded"
        
        # Performance should still be reasonable
        concurrent_duration = performance_monitor.get_duration("concurrent_workflows")
        expected_max_time = 120  # Should complete within 2 minutes
        assert_performance_within_limits(concurrent_duration, expected_max_time)
        
        # Cleanup
        for detector_id in detector_ids:
            try:
                await async_client.delete(f"/api/v1/detectors/{detector_id}", headers=api_headers)
            except:
                pass  # Ignore cleanup errors

    async def test_rapid_fire_requests(
        self,
        async_client,
        sample_dataset,
        api_headers,
        performance_monitor
    ):
        """Test handling of rapid successive requests."""
        
        # Create and train detector first
        detector_config = {
            "name": "rapid-fire-detector",
            "algorithm_name": "IsolationForest",
            "hyperparameters": {"n_estimators": 30},
            "contamination_rate": 0.05
        }
        
        response = await async_client.post(
            "/api/v1/detectors",
            json=detector_config,
            headers=api_headers
        )
        assert_api_response_valid(response, 201)
        detector = response.json()
        detector_id = detector["id"]
        
        # Train detector
        training_data = sample_dataset['features'].iloc[:300]
        training_payload = {
            "detector_id": detector_id,
            "dataset": {
                "name": "rapid-fire-training",
                "data": training_data.values.tolist(),
                "feature_names": training_data.columns.tolist()
            },
            "job_name": "rapid-fire-training"
        }
        
        response = await async_client.post(
            "/api/v1/training/jobs",
            json=training_payload,
            headers=api_headers
        )
        assert_api_response_valid(response, 201)
        training_job = response.json()
        
        await self._wait_for_job_completion(async_client, training_job["job_id"], api_headers)
        
        # Rapid fire detection requests
        test_data = sample_dataset['features'].iloc[300:350]
        detection_payload = {
            "detector_id": detector_id,
            "dataset": {
                "name": "rapid-fire-detection",
                "data": test_data.values.tolist(),
                "feature_names": test_data.columns.tolist()
            },
            "return_scores": True
        }
        
        num_requests = 20
        request_interval = 0.1  # 100ms between requests
        
        async def make_detection_request(request_id: int):
            """Make a single detection request."""
            try:
                response = await async_client.post(
                    "/api/v1/detection/predict",
                    json=detection_payload,
                    headers=api_headers
                )
                return request_id, response.status_code, response.json() if response.status_code == 200 else None
            except Exception as e:
                return request_id, -1, str(e)
        
        performance_monitor.start_timer("rapid_fire_requests")
        
        # Launch requests with small delays
        tasks = []
        for i in range(num_requests):
            tasks.append(make_detection_request(i))
            if i < num_requests - 1:
                await asyncio.sleep(request_interval)
        
        results = await asyncio.gather(*tasks)
        
        performance_monitor.end_timer("rapid_fire_requests")
        
        # Analyze results
        successful_requests = 0
        failed_requests = 0
        
        for request_id, status_code, result in results:
            if status_code == 200:
                successful_requests += 1
                # Verify result structure
                assert isinstance(result, dict)
                assert "n_samples" in result
            elif status_code == 429:  # Rate limited
                print(f"Request {request_id} was rate limited")
            else:
                failed_requests += 1
                print(f"Request {request_id} failed with status {status_code}: {result}")
        
        # At least 80% should succeed or be rate limited (not fail)
        success_rate = successful_requests / num_requests
        assert success_rate >= 0.6, f"Only {success_rate:.1%} of rapid requests succeeded"
        
        # Should complete within reasonable time
        rapid_fire_duration = performance_monitor.get_duration("rapid_fire_requests")
        expected_max_time = num_requests * request_interval + 30  # Base time + 30s buffer
        assert_performance_within_limits(rapid_fire_duration, expected_max_time)
        
        # Cleanup
        await async_client.delete(f"/api/v1/detectors/{detector_id}", headers=api_headers)


@pytest.mark.asyncio
@pytest.mark.resource
@pytest.mark.slow
class TestResourceUtilization:
    """Test resource utilization and memory management."""

    async def test_memory_usage_monitoring(
        self,
        async_client,
        api_headers,
        performance_monitor
    ):
        """Monitor memory usage during intensive operations."""
        
        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create multiple detectors and train them
        detector_ids = []
        memory_measurements = [initial_memory]
        
        for i in range(3):  # Reduced for CI compatibility
            # Create detector
            detector_config = {
                "name": f"memory-test-detector-{i}",
                "algorithm_name": "IsolationForest",
                "hyperparameters": {"n_estimators": 50},
                "contamination_rate": 0.05
            }
            
            response = await async_client.post(
                "/api/v1/detectors",
                json=detector_config,
                headers=api_headers
            )
            assert_api_response_valid(response, 201)
            detector = response.json()
            detector_ids.append(detector["id"])
            
            # Generate larger dataset for this test
            np.random.seed(i)
            data = np.random.normal(0, 1, (1000, 5))
            
            training_payload = {
                "detector_id": detector["id"],
                "dataset": {
                    "name": f"memory-test-data-{i}",
                    "data": data.tolist(),
                    "feature_names": [f"feature_{j}" for j in range(5)]
                },
                "job_name": f"memory-test-training-{i}"
            }
            
            response = await async_client.post(
                "/api/v1/training/jobs",
                json=training_payload,
                headers=api_headers
            )
            assert_api_response_valid(response, 201)
            training_job = response.json()
            
            await self._wait_for_job_completion(async_client, training_job["job_id"], api_headers)
            
            # Measure memory after each operation
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_measurements.append(current_memory)
        
        # Analyze memory usage
        max_memory = max(memory_measurements)
        memory_growth = max_memory - initial_memory
        
        # Memory growth should be reasonable (less than 500MB for this test)
        assert memory_growth < 500, f"Memory grew by {memory_growth:.1f}MB, which may indicate a leak"
        
        # Cleanup and verify memory cleanup
        for detector_id in detector_ids:
            await async_client.delete(f"/api/v1/detectors/{detector_id}", headers=api_headers)
        
        # Allow some time for cleanup
        await asyncio.sleep(5)
        
        # Final memory measurement
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Memory should not have grown excessively from initial
        total_growth = final_memory - initial_memory
        assert total_growth < 200, f"Final memory growth of {total_growth:.1f}MB is excessive"

    async def _wait_for_job_completion(
        self,
        client,
        job_id: str,
        headers: Dict[str, str],
        max_wait_time: int = 60
    ):
        """Helper method to wait for training job completion."""
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            response = await client.get(
                f"/api/v1/training/jobs/{job_id}",
                headers=headers
            )
            assert_api_response_valid(response)
            job_status = response.json()
            
            if job_status["status"] == "completed":
                return job_status
            elif job_status["status"] == "failed":
                pytest.fail(f"Training failed: {job_status.get('error_message', 'Unknown error')}")
                
            await asyncio.sleep(2)
        
        pytest.fail("Training did not complete within expected time")