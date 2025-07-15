"""
Enhanced Performance and Load Testing

Comprehensive performance tests covering scalability, load handling,
resource utilization, and system stress testing under various conditions.
"""

import pytest
import asyncio
import time
import concurrent.futures
from typing import List, Dict, Any, Tuple
import statistics
import psutil
import numpy as np
import threading
from dataclasses import dataclass, field

from .conftest import (
    assert_performance_within_limits,
    assert_api_response_valid
)


@dataclass
class LoadTestMetrics:
    """Comprehensive metrics for load testing."""
    response_times: List[float] = field(default_factory=list)
    success_count: int = 0
    error_count: int = 0
    rate_limited_count: int = 0
    timeout_count: int = 0
    cpu_usage: List[float] = field(default_factory=list)
    memory_usage: List[float] = field(default_factory=list)
    error_details: List[Dict[str, Any]] = field(default_factory=list)
    
    @property
    def total_requests(self) -> int:
        return self.success_count + self.error_count + self.rate_limited_count + self.timeout_count
    
    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.success_count / self.total_requests
    
    @property
    def average_response_time(self) -> float:
        if not self.response_times:
            return 0.0
        return statistics.mean(self.response_times)
    
    @property
    def p95_response_time(self) -> float:
        if not self.response_times:
            return 0.0
        return np.percentile(self.response_times, 95)
    
    @property
    def p99_response_time(self) -> float:
        if not self.response_times:
            return 0.0
        return np.percentile(self.response_times, 99)
    
    @property
    def throughput_rps(self) -> float:
        """Requests per second."""
        if not self.response_times or len(self.response_times) < 2:
            return 0.0
        total_time = max(self.response_times) - min(self.response_times)
        if total_time <= 0:
            return 0.0
        return self.success_count / total_time


class SystemMonitor:
    """Enhanced system resource monitoring."""
    
    def __init__(self):
        self.monitoring = False
        self.metrics = LoadTestMetrics()
        self._monitor_thread = None
        self._start_time = None
    
    def start_monitoring(self):
        """Start comprehensive system monitoring."""
        self.monitoring = True
        self._start_time = time.time()
        self._monitor_thread = threading.Thread(target=self._monitor_resources)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop system monitoring."""
        self.monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
    
    def _monitor_resources(self):
        """Monitor CPU, memory, and system resources."""
        while self.monitoring:
            try:
                # CPU and memory monitoring
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory_info = psutil.virtual_memory()
                
                self.metrics.cpu_usage.append(cpu_percent)
                self.metrics.memory_usage.append(memory_info.percent)
                
                # Monitor disk I/O if needed
                # disk_io = psutil.disk_io_counters()
                
            except Exception:
                # Continue monitoring even if there are occasional errors
                pass
            
            time.sleep(1)
    
    def get_metrics(self) -> LoadTestMetrics:
        """Get comprehensive monitoring metrics."""
        return self.metrics
    
    def record_response(self, response_time: float, status_code: int, error: str = None):
        """Record individual response metrics."""
        self.metrics.response_times.append(response_time)
        
        if status_code == 200:
            self.metrics.success_count += 1
        elif status_code == 429:
            self.metrics.rate_limited_count += 1
        elif status_code == -1:  # Timeout
            self.metrics.timeout_count += 1
        else:
            self.metrics.error_count += 1
            if error:
                self.metrics.error_details.append({
                    "status_code": status_code,
                    "error": error,
                    "timestamp": time.time()
                })


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


@pytest.mark.asyncio
@pytest.mark.stress
@pytest.mark.slow
class TestStressTesting:
    """Comprehensive stress testing scenarios for system limits."""

    async def test_high_volume_training_stress(
        self,
        async_client,
        api_headers,
        performance_monitor
    ):
        """Test system behavior under high-volume concurrent training stress."""
        
        num_detectors = 8  # Reasonable number for CI environments
        detector_ids = []
        system_monitor = SystemMonitor()
        
        try:
            system_monitor.start_monitoring()
            performance_monitor.start_timer("stress_training")
            
            # Create detectors
            for i in range(num_detectors):
                detector_config = {
                    "name": f"stress-detector-{i}",
                    "algorithm_name": "IsolationForest",
                    "hyperparameters": {"n_estimators": 30},  # Reduced for speed
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
            
            # Launch concurrent training jobs with varied data
            training_tasks = []
            for i, detector_id in enumerate(detector_ids):
                np.random.seed(i)
                data_size = 300 + (i * 50)  # Varying sizes
                features = 3 + (i % 3)  # Varying feature counts
                data = np.random.normal(0, 1, (data_size, features))
                
                training_payload = {
                    "detector_id": detector_id,
                    "dataset": {
                        "name": f"stress-training-{i}",
                        "data": data.tolist(),
                        "feature_names": [f"f{j}" for j in range(features)]
                    },
                    "job_name": f"stress-training-{i}"
                }
                
                task = asyncio.create_task(
                    self._submit_and_monitor_training(
                        async_client, training_payload, api_headers, system_monitor
                    )
                )
                training_tasks.append(task)
            
            # Wait for all training to complete
            results = await asyncio.gather(*training_tasks, return_exceptions=True)
            
            performance_monitor.end_timer("stress_training")
            system_monitor.stop_monitoring()
            
            # Analyze stress test results
            successful_trainings = sum(1 for r in results if r is True)
            success_rate = successful_trainings / num_detectors
            
            # Under stress, we should achieve reasonable success rate
            assert success_rate >= 0.6, f"Only {success_rate:.1%} of training jobs succeeded under stress"
            
            # Performance should degrade gracefully
            stress_duration = performance_monitor.get_duration("stress_training")
            assert_performance_within_limits(stress_duration, 240.0)  # 4 minutes max
            
            # System resources should not be completely exhausted
            metrics = system_monitor.get_metrics()
            if metrics.cpu_usage:
                max_cpu = max(metrics.cpu_usage)
                assert max_cpu < 98, f"CPU usage peaked at {max_cpu}%"
                
            if metrics.memory_usage:
                max_memory = max(metrics.memory_usage)
                assert max_memory < 95, f"Memory usage peaked at {max_memory}%"
                
        finally:
            # Cleanup
            cleanup_tasks = []
            for detector_id in detector_ids:
                cleanup_tasks.append(
                    asyncio.create_task(
                        self._safe_delete_detector(async_client, detector_id, api_headers)
                    )
                )
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)

    async def test_sustained_load_endurance(
        self,
        async_client,
        sample_dataset,
        api_headers,
        performance_monitor
    ):
        """Test system endurance under sustained detection load."""
        
        # Create and train detector for sustained testing
        detector_config = {
            "name": "endurance-detector",
            "algorithm_name": "IsolationForest",
            "hyperparameters": {"n_estimators": 25},
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
            # Train detector
            training_data = sample_dataset['features'].iloc[:300]
            training_payload = {
                "detector_id": detector_id,
                "dataset": {
                    "name": "endurance-training",
                    "data": training_data.values.tolist(),
                    "feature_names": training_data.columns.tolist()
                },
                "job_name": "endurance-training"
            }
            
            response = await async_client.post(
                "/api/v1/training/jobs",
                json=training_payload,
                headers=api_headers
            )
            assert_api_response_valid(response, 201)
            training_job = response.json()
            
            await self._wait_for_job_completion(async_client, training_job["job_id"], api_headers)
            
            # Sustained load testing
            test_duration = 45  # 45 seconds of sustained load
            request_interval = 0.75  # One request every 750ms
            
            system_monitor = SystemMonitor()
            system_monitor.start_monitoring()
            performance_monitor.start_timer("sustained_load")
            
            end_time = time.time() + test_duration
            request_count = 0
            
            while time.time() < end_time:
                # Prepare varied detection requests
                sample_size = 15 + (request_count % 10)  # Varying sample sizes
                test_data = sample_dataset['features'].sample(n=sample_size, random_state=request_count)
                detection_payload = {
                    "detector_id": detector_id,
                    "dataset": {
                        "name": f"endurance-detection-{request_count}",
                        "data": test_data.values.tolist(),
                        "feature_names": test_data.columns.tolist()
                    }
                }
                
                # Make request and measure response time
                start_time = time.time()
                try:
                    response = await async_client.post(
                        "/api/v1/detection/predict",
                        json=detection_payload,
                        headers=api_headers
                    )
                    response_time = time.time() - start_time
                    system_monitor.record_response(response_time, response.status_code)
                    
                    if response.status_code == 200:
                        result = response.json()
                        assert result["n_samples"] == len(test_data)
                    
                except Exception as e:
                    response_time = time.time() - start_time
                    system_monitor.record_response(response_time, -1, str(e))
                
                request_count += 1
                await asyncio.sleep(request_interval)
            
            performance_monitor.end_timer("sustained_load")
            system_monitor.stop_monitoring()
            
            # Analyze endurance test results
            metrics = system_monitor.get_metrics()
            
            # Should maintain reasonable success rate under sustained load
            assert metrics.success_rate >= 0.75, f"Success rate dropped to {metrics.success_rate:.1%} under sustained load"
            
            # Response times should remain stable
            if len(metrics.response_times) >= 10:
                # Check for response time degradation over time
                first_third = metrics.response_times[:len(metrics.response_times)//3]
                last_third = metrics.response_times[-len(metrics.response_times)//3:]
                
                if first_third and last_third:
                    first_avg = statistics.mean(first_third)
                    last_avg = statistics.mean(last_third)
                    degradation = (last_avg - first_avg) / first_avg if first_avg > 0 else 0
                    
                    # Response time should not degrade by more than 100%
                    assert degradation < 1.0, f"Response time degraded by {degradation:.1%} during sustained load"
            
        finally:
            await self._safe_delete_detector(async_client, detector_id, api_headers)

    # Helper methods for stress testing
    async def _submit_and_monitor_training(
        self,
        client,
        training_payload: Dict[str, Any],
        headers: Dict[str, str],
        system_monitor: SystemMonitor
    ) -> bool:
        """Submit training job and monitor its completion."""
        start_time = time.time()
        
        try:
            response = await client.post(
                "/api/v1/training/jobs",
                json=training_payload,
                headers=headers
            )
            
            if response.status_code != 201:
                response_time = time.time() - start_time
                system_monitor.record_response(response_time, response.status_code)
                return False
            
            training_job = response.json()
            job_id = training_job["job_id"]
            
            # Wait for completion with timeout
            result = await self._wait_for_job_completion(
                client, job_id, headers, max_wait_time=90
            )
            
            response_time = time.time() - start_time
            if result and result.get("status") == "completed":
                system_monitor.record_response(response_time, 200)
                return True
            else:
                system_monitor.record_response(response_time, 500)
                return False
                
        except Exception as e:
            response_time = time.time() - start_time
            system_monitor.record_response(response_time, -1, str(e))
            return False

    async def _safe_delete_detector(
        self,
        client,
        detector_id: str,
        headers: Dict[str, str]
    ):
        """Safely delete detector, ignoring errors."""
        try:
            await client.delete(f"/api/v1/detectors/{detector_id}", headers=headers)
        except Exception:
            pass  # Ignore cleanup errors

    async def _wait_for_job_completion(
        self,
        client,
        job_id: str,
        headers: Dict[str, str],
        max_wait_time: int = 60
    ) -> Dict[str, Any]:
        """Wait for job completion with extended timeout handling."""
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            try:
                response = await client.get(
                    f"/api/v1/training/jobs/{job_id}",
                    headers=headers
                )
                
                if response.status_code != 200:
                    await asyncio.sleep(3)
                    continue
                
                job_status = response.json()
                
                if job_status["status"] in ["completed", "failed"]:
                    return job_status
                
            except Exception:
                # Continue waiting on temporary errors
                pass
                
            await asyncio.sleep(3)
        
        # Timeout reached - return None to indicate failure
        return {"status": "timeout", "error_message": f"Job {job_id} did not complete within {max_wait_time} seconds"}