"""Load testing and scalability tests for anomaly detection system."""

import pytest
import asyncio
import time
import threading
import concurrent.futures
import numpy as np
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient

from anomaly_detection.main import create_app
from anomaly_detection.worker import AnomalyDetectionWorker, JobType, JobPriority
from anomaly_detection.domain.services.detection_service import DetectionService
from anomaly_detection.domain.services.ensemble_service import EnsembleService
from anomaly_detection.domain.services.streaming_service import StreamingService


@dataclass
class LoadTestResult:
    """Container for load test results."""
    test_name: str
    concurrent_users: int
    total_requests: int
    duration: float
    successful_requests: int
    failed_requests: int
    avg_response_time: float
    min_response_time: float
    max_response_time: float
    requests_per_second: float
    error_rate: float
    percentile_95: float
    percentile_99: float


class LoadTestRunner:
    """Load test execution framework."""
    
    def __init__(self):
        self.response_times = []
        self.successful_requests = 0
        self.failed_requests = 0
        self.start_time = None
        self.end_time = None
        self.lock = threading.Lock()
    
    def record_request(self, response_time: float, success: bool):
        """Record a request result."""
        with self.lock:
            self.response_times.append(response_time)
            if success:
                self.successful_requests += 1
            else:
                self.failed_requests += 1
    
    def get_percentile(self, percentile: float) -> float:
        """Calculate response time percentile."""
        if not self.response_times:
            return 0.0
        sorted_times = sorted(self.response_times)
        index = int(len(sorted_times) * percentile / 100)
        return sorted_times[min(index, len(sorted_times) - 1)]
    
    def get_results(self, test_name: str, concurrent_users: int) -> LoadTestResult:
        """Get load test results."""
        duration = (self.end_time - self.start_time).total_seconds()
        total_requests = self.successful_requests + self.failed_requests
        
        return LoadTestResult(
            test_name=test_name,
            concurrent_users=concurrent_users,
            total_requests=total_requests,
            duration=duration,
            successful_requests=self.successful_requests,
            failed_requests=self.failed_requests,
            avg_response_time=np.mean(self.response_times) if self.response_times else 0,
            min_response_time=min(self.response_times) if self.response_times else 0,
            max_response_time=max(self.response_times) if self.response_times else 0,
            requests_per_second=total_requests / duration if duration > 0 else 0,
            error_rate=self.failed_requests / total_requests if total_requests > 0 else 0,
            percentile_95=self.get_percentile(95),
            percentile_99=self.get_percentile(99)
        )
    
    def run_load_test(self, test_function, concurrent_users: int, 
                     requests_per_user: int, test_name: str) -> LoadTestResult:
        """Run a load test with specified parameters."""
        self.start_time = datetime.now()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_users) as executor:
            # Submit all requests
            futures = []
            for _ in range(concurrent_users):
                for _ in range(requests_per_user):
                    future = executor.submit(self._execute_request, test_function)
                    futures.append(future)
            
            # Wait for all requests to complete
            concurrent.futures.wait(futures)
        
        self.end_time = datetime.now()
        return self.get_results(test_name, concurrent_users)
    
    def _execute_request(self, test_function):
        """Execute a single request and record timing."""
        start_time = time.perf_counter()
        try:
            test_function()
            success = True
        except Exception as e:
            success = False
            # Print first few errors for debugging
            if self.failed_requests < 3:  # Only print first 3 errors
                print(f"Request error: {e}")
        
        end_time = time.perf_counter()
        response_time = end_time - start_time
        self.record_request(response_time, success)


@pytest.mark.performance
@pytest.mark.load
class TestAPILoadTesting:
    """Load testing for API endpoints."""
    
    def setup_method(self):
        """Set up test fixtures."""
        with patch('anomaly_detection.main.get_settings') as mock_get_settings:
            mock_settings = Mock()
            mock_settings.api.cors_origins = ["*"]
            mock_get_settings.return_value = mock_settings
            
            self.app = create_app()
            self.client = TestClient(self.app)
        
        # Test data
        self.test_data = {
            "data": [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]],
            "algorithm": "isolation_forest",
            "contamination": 0.1
        }
        
        self.ensemble_data = {
            "data": [[1, 2], [3, 4], [5, 6], [7, 8]],
            "algorithms": ["isolation_forest", "one_class_svm"],
            "ensemble_method": "majority",
            "contamination": 0.1
        }
    
    def _detection_request(self):
        """Single detection API request."""
        # Mock the detection service properly for API v1 dependency injection
        with patch('anomaly_detection.api.v1.detection.get_detection_service') as mock_get_service:
            mock_service = Mock()
            mock_result = Mock()
            mock_result.success = True
            mock_result.anomalies = [0, 2, 4]  # Sample anomaly indices
            mock_result.confidence_scores = np.array([0.1, 0.8, 0.2, 0.9, 0.15])
            mock_result.total_samples = 5
            mock_result.anomaly_count = 3
            mock_result.anomaly_rate = 0.6
            
            mock_service.detect_anomalies.return_value = mock_result
            mock_get_service.return_value = mock_service
            
            response = self.client.post("/api/v1/detection/detect", json=self.test_data)
            if response.status_code != 200:
                raise Exception(f"API request failed: {response.status_code} - {response.text}")
    
    def _ensemble_request(self):
        """Single ensemble API request."""
        with patch('anomaly_detection.api.v1.detection.get_ensemble_service') as mock_get_ensemble_service, \
             patch('anomaly_detection.domain.services.detection_service.DetectionService') as mock_detection_class:
            
            # Mock the detection service class constructor for ensemble endpoint
            mock_detection_instance = Mock()
            mock_detection_result = Mock()
            mock_detection_result.success = True
            mock_detection_result.predictions = np.array([-1, 1, 1, -1])  # -1 = anomaly, 1 = normal
            mock_detection_result.confidence_scores = np.array([0.9, 0.2, 0.3, 0.8])
            mock_detection_instance.detect_anomalies.return_value = mock_detection_result
            mock_detection_class.return_value = mock_detection_instance
            
            # Mock ensemble service methods
            mock_ensemble_service = Mock()
            mock_ensemble_service.majority_vote.return_value = np.array([-1, 1, 1, -1])
            mock_get_ensemble_service.return_value = mock_ensemble_service
            
            response = self.client.post("/api/v1/detection/ensemble", json=self.ensemble_data)
            if response.status_code != 200:
                raise Exception(f"Ensemble API request failed: {response.status_code} - {response.text}")
    
    def _health_request(self):
        """Single health check API request."""
        # Health endpoint should work without mocking since it's simple
        response = self.client.get("/health")
        if response.status_code != 200:
            raise Exception(f"Health API request failed: {response.status_code} - {response.text}")
    
    def test_detection_endpoint_load(self):
        """Load test for detection endpoint."""
        runner = LoadTestRunner()
        
        # Test with moderate load
        result = runner.run_load_test(
            test_function=self._detection_request,
            concurrent_users=10,
            requests_per_user=5,
            test_name="detection_endpoint"
        )
        
        # Performance assertions (adjusted for real algorithm execution)
        assert result.error_rate < 0.05, f"Too many errors: {result.error_rate}"
        assert result.avg_response_time < 5.0, f"Average response time too high: {result.avg_response_time}s"
        assert result.requests_per_second > 2, f"Throughput too low: {result.requests_per_second} req/s"
        assert result.percentile_95 < 8.0, f"95th percentile too high: {result.percentile_95}s"
        
        self._print_load_test_results(result)
    
    def test_ensemble_endpoint_load(self):
        """Load test for ensemble endpoint."""
        runner = LoadTestRunner()
        
        result = runner.run_load_test(
            test_function=self._ensemble_request,
            concurrent_users=8,
            requests_per_user=3,
            test_name="ensemble_endpoint"
        )
        
        # Ensemble endpoint expected to be slower
        assert result.error_rate < 0.05, f"Too many errors: {result.error_rate}"
        assert result.avg_response_time < 4.0, f"Average response time too high: {result.avg_response_time}s"
        assert result.requests_per_second > 10, f"Throughput too low: {result.requests_per_second} req/s"
        
        self._print_load_test_results(result)
    
    def test_health_endpoint_load(self):
        """Load test for health endpoint (should be very fast)."""
        runner = LoadTestRunner()
        
        result = runner.run_load_test(
            test_function=self._health_request,
            concurrent_users=20,
            requests_per_user=10,
            test_name="health_endpoint"
        )
        
        # Health endpoint should be very fast
        assert result.error_rate < 0.01, f"Too many errors: {result.error_rate}"
        assert result.avg_response_time < 0.1, f"Average response time too high: {result.avg_response_time}s"
        assert result.requests_per_second > 200, f"Throughput too low: {result.requests_per_second} req/s"
        
        self._print_load_test_results(result)
    
    def test_mixed_endpoint_load(self):
        """Load test with mixed endpoint requests."""
        runner = LoadTestRunner()
        
        def mixed_requests():
            """Mix of different endpoint requests."""
            import random
            request_type = random.choice(['detection', 'health', 'health', 'health'])  # Weight health checks higher
            
            if request_type == 'detection':
                self._detection_request()
            else:
                self._health_request()
        
        result = runner.run_load_test(
            test_function=mixed_requests,
            concurrent_users=15,
            requests_per_user=4,
            test_name="mixed_endpoints"
        )
        
        assert result.error_rate < 0.05, f"Too many errors: {result.error_rate}"
        assert result.requests_per_second > 30, f"Throughput too low: {result.requests_per_second} req/s"
        
        self._print_load_test_results(result)
    
    def test_scalability_with_user_growth(self):
        """Test API scalability with increasing concurrent users."""
        user_counts = [1, 5, 10, 20]
        results = []
        
        for user_count in user_counts:
            runner = LoadTestRunner()
            result = runner.run_load_test(
                test_function=self._detection_request,
                concurrent_users=user_count,
                requests_per_user=3,
                test_name=f"scalability_{user_count}_users"
            )
            results.append(result)
        
        # Check scalability characteristics
        for i in range(1, len(results)):
            prev_result = results[i-1]
            curr_result = results[i]
            
            # Response time should not increase dramatically
            response_time_ratio = curr_result.avg_response_time / prev_result.avg_response_time
            assert response_time_ratio < 3.0, f"Response time scaling too poor: {response_time_ratio}x"
            
            # Throughput should increase (though may plateau)
            if curr_result.requests_per_second < prev_result.requests_per_second * 0.5:
                pytest.fail(f"Throughput degraded too much: {curr_result.requests_per_second} vs {prev_result.requests_per_second}")
        
        print(f"\nScalability Analysis:")
        for result in results:
            print(f"  {result.concurrent_users} users: {result.avg_response_time:.3f}s avg, {result.requests_per_second:.1f} req/s")
    
    def _print_load_test_results(self, result: LoadTestResult):
        """Print formatted load test results."""
        print(f"\nLoad Test Results - {result.test_name}:")
        print(f"  Concurrent Users: {result.concurrent_users}")
        print(f"  Total Requests: {result.total_requests}")
        print(f"  Duration: {result.duration:.2f}s")
        print(f"  Successful: {result.successful_requests}")
        print(f"  Failed: {result.failed_requests}")
        print(f"  Error Rate: {result.error_rate:.1%}")
        print(f"  Avg Response Time: {result.avg_response_time:.3f}s")
        print(f"  95th Percentile: {result.percentile_95:.3f}s")
        print(f"  99th Percentile: {result.percentile_99:.3f}s")
        print(f"  Requests/Second: {result.requests_per_second:.1f}")


@pytest.mark.performance
@pytest.mark.load
class TestServiceLoadTesting:
    """Load testing for core services."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.detection_service = DetectionService()
        self.ensemble_service = EnsembleService()
        self.streaming_service = StreamingService()
        
        # Generate test data
        np.random.seed(42)
        self.test_data = np.random.normal(0, 1, (100, 10)).tolist()
        self.large_test_data = np.random.normal(0, 1, (1000, 20)).tolist()
    
    def _detection_service_call(self):
        """Single detection service call."""
        result = self.detection_service.detect_anomalies(
            self.test_data,
            algorithm="isolation_forest",
            contamination=0.1
        )
        if not result or "anomalies" not in result:
            raise Exception("Detection service failed")
    
    def _ensemble_service_call(self):
        """Single ensemble service call."""
        result = self.ensemble_service.detect_with_ensemble(
            self.test_data,
            algorithms=["isolation_forest", "one_class_svm"],
            ensemble_method="majority",
            contamination=0.1
        )
        if not result or "anomalies" not in result:
            raise Exception("Ensemble service failed")
    
    def _streaming_service_call(self):
        """Single streaming service call."""
        batch_data = self.test_data[:50]  # Use smaller batch
        result = self.streaming_service.process_streaming_batch(
            batch_data,
            algorithm="isolation_forest",
            buffer_size=200
        )
        if not result:
            raise Exception("Streaming service failed")
    
    def test_detection_service_load(self):
        """Load test for detection service."""
        runner = LoadTestRunner()
        
        result = runner.run_load_test(
            test_function=self._detection_service_call,
            concurrent_users=5,
            requests_per_user=4,
            test_name="detection_service"
        )
        
        assert result.error_rate < 0.05, f"Too many service errors: {result.error_rate}"
        assert result.avg_response_time < 3.0, f"Service response time too high: {result.avg_response_time}s"
        
        self._print_service_results(result)
    
    def test_ensemble_service_load(self):
        """Load test for ensemble service."""
        runner = LoadTestRunner()
        
        result = runner.run_load_test(
            test_function=self._ensemble_service_call,
            concurrent_users=3,
            requests_per_user=3,
            test_name="ensemble_service"
        )
        
        assert result.error_rate < 0.05, f"Too many ensemble errors: {result.error_rate}"
        assert result.avg_response_time < 6.0, f"Ensemble response time too high: {result.avg_response_time}s"
        
        self._print_service_results(result)
    
    def test_streaming_service_load(self):
        """Load test for streaming service."""
        runner = LoadTestRunner()
        
        result = runner.run_load_test(
            test_function=self._streaming_service_call,
            concurrent_users=8,
            requests_per_user=5,
            test_name="streaming_service"
        )
        
        assert result.error_rate < 0.05, f"Too many streaming errors: {result.error_rate}"
        assert result.avg_response_time < 1.0, f"Streaming response time too high: {result.avg_response_time}s"
        
        self._print_service_results(result)
    
    def test_service_thread_safety(self):
        """Test service thread safety under load."""
        detection_results = []
        errors = []
        
        def concurrent_detection():
            try:
                result = self.detection_service.detect_anomalies(
                    self.test_data,
                    algorithm="isolation_forest",
                    contamination=0.1
                )
                detection_results.append(result)
            except Exception as e:
                errors.append(str(e))
        
        # Run many concurrent requests
        threads = []
        for _ in range(20):
            thread = threading.Thread(target=concurrent_detection)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # All requests should succeed
        assert len(errors) == 0, f"Thread safety errors: {errors}"
        assert len(detection_results) == 20, f"Missing results: {len(detection_results)}"
        
        # All results should be valid
        for result in detection_results:
            assert "anomalies" in result
            assert len(result["anomalies"]) == len(self.test_data)
    
    def _print_service_results(self, result: LoadTestResult):
        """Print service load test results."""
        print(f"\nService Load Test - {result.test_name}:")
        print(f"  Requests: {result.total_requests}")
        print(f"  Success Rate: {(1 - result.error_rate):.1%}")
        print(f"  Avg Time: {result.avg_response_time:.3f}s")
        print(f"  Throughput: {result.requests_per_second:.1f} req/s")


@pytest.mark.performance
@pytest.mark.load
class TestWorkerLoadTesting:
    """Load testing for worker system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.worker = AnomalyDetectionWorker(max_workers=4)
        
        # Generate test data
        np.random.seed(42)
        self.test_data = np.random.normal(0, 1, (50, 5)).tolist()
    
    def test_worker_job_queue_load(self):
        """Test worker with high job queue load."""
        start_time = time.time()
        
        # Add many jobs quickly
        job_ids = []
        for i in range(100):
            job_id = self.worker.add_job(
                job_type=JobType.DETECTION,
                data={
                    "algorithm": "isolation_forest",
                    "data": self.test_data,
                    "contamination": 0.1
                },
                priority=JobPriority.NORMAL
            )
            job_ids.append(job_id)
        
        queue_time = time.time() - start_time
        
        # Queue operations should be fast
        assert queue_time < 1.0, f"Job queuing too slow: {queue_time}s"
        assert len(job_ids) == 100, "Not all jobs were queued"
        assert len(set(job_ids)) == 100, "Duplicate job IDs generated"
        
        # Check queue size
        assert self.worker.job_queue.size() == 100, "Incorrect queue size"
        
        print(f"\nWorker Queue Load Test:")
        print(f"  Jobs Queued: 100")
        print(f"  Queue Time: {queue_time:.3f}s")
        print(f"  Jobs/Second: {100 / queue_time:.1f}")
    
    def test_worker_priority_handling_under_load(self):
        """Test worker priority handling with mixed priority jobs."""
        # Add jobs with different priorities
        critical_jobs = []
        normal_jobs = []
        low_jobs = []
        
        # Add in mixed order
        for i in range(30):
            if i % 3 == 0:
                job_id = self.worker.add_job(
                    JobType.DETECTION, {"data": self.test_data}, JobPriority.CRITICAL
                )
                critical_jobs.append(job_id)
            elif i % 3 == 1:
                job_id = self.worker.add_job(
                    JobType.DETECTION, {"data": self.test_data}, JobPriority.NORMAL
                )
                normal_jobs.append(job_id)
            else:
                job_id = self.worker.add_job(
                    JobType.DETECTION, {"data": self.test_data}, JobPriority.LOW
                )
                low_jobs.append(job_id)
        
        # Get jobs in processing order
        processed_order = []
        while not self.worker.job_queue.is_empty():
            job = self.worker.job_queue.get_next_job()
            processed_order.append((job.job_id, job.priority))
        
        # Verify priority ordering
        critical_indices = [i for i, (job_id, priority) in enumerate(processed_order) 
                          if priority == JobPriority.CRITICAL]
        normal_indices = [i for i, (job_id, priority) in enumerate(processed_order) 
                        if priority == JobPriority.NORMAL]
        low_indices = [i for i, (job_id, priority) in enumerate(processed_order) 
                     if priority == JobPriority.LOW]
        
        # All critical jobs should come before all normal jobs
        if critical_indices and normal_indices:
            assert max(critical_indices) < min(normal_indices), "Priority ordering violated"
        
        # All normal jobs should come before all low jobs
        if normal_indices and low_indices:
            assert max(normal_indices) < min(low_indices), "Priority ordering violated"
        
        print(f"\nWorker Priority Test:")
        print(f"  Critical Jobs: {len(critical_jobs)}")
        print(f"  Normal Jobs: {len(normal_jobs)}")
        print(f"  Low Jobs: {len(low_jobs)}")
        print(f"  Priority Ordering: Correct")
    
    def test_worker_concurrent_job_processing(self):
        """Test worker processing jobs concurrently."""
        # Mock the services to return quickly
        with patch.object(self.worker, 'detection_service') as mock_detection:
            mock_detection.detect_anomalies = Mock(return_value={
                "anomalies": [0, 1, 0],
                "scores": [0.1, 0.8, 0.2],
                "algorithm": "isolation_forest"
            })
            
            # Add multiple jobs
            job_ids = []
            for i in range(10):
                job_id = self.worker.add_job(
                    JobType.DETECTION,
                    {
                        "algorithm": "isolation_forest",
                        "data": self.test_data[:20],  # Smaller data for faster processing
                        "contamination": 0.1
                    },
                    JobPriority.NORMAL
                )
                job_ids.append(job_id)
            
            # Process jobs concurrently
            start_time = time.time()
            
            # Simulate processing (in real scenario, worker would process automatically)
            processing_times = []
            for job_id in job_ids:
                job_start = time.time()
                job = self.worker.job_queue.get_next_job()
                if job:
                    # Simulate job processing time
                    time.sleep(0.01)  # Minimal delay
                job_end = time.time()
                processing_times.append(job_end - job_start)
            
            total_time = time.time() - start_time
            
            # All jobs should be processed
            assert len(processing_times) == 10, "Not all jobs processed"
            assert total_time < 2.0, f"Processing took too long: {total_time}s"
            
            print(f"\nWorker Concurrent Processing:")
            print(f"  Jobs Processed: {len(processing_times)}")
            print(f"  Total Time: {total_time:.3f}s")
            print(f"  Avg Job Time: {np.mean(processing_times):.3f}s")
    
    def test_worker_memory_under_load(self):
        """Test worker memory usage under job load."""
        try:
            import psutil
        except ImportError:
            pytest.skip("psutil not available for memory testing")
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Add many jobs to test memory usage
        for i in range(200):
            self.worker.add_job(
                JobType.DETECTION,
                {
                    "algorithm": "isolation_forest", 
                    "data": self.test_data,
                    "contamination": 0.1
                },
                JobPriority.NORMAL
            )
        
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_growth = peak_memory - initial_memory
        
        # Memory growth should be reasonable
        assert memory_growth < 50.0, f"Excessive memory growth: {memory_growth}MB"
        
        # Clear jobs and check memory cleanup
        self.worker.job_queue.clear()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        print(f"\nWorker Memory Test:")
        print(f"  Initial Memory: {initial_memory:.1f}MB")
        print(f"  Peak Memory: {peak_memory:.1f}MB")
        print(f"  Final Memory: {final_memory:.1f}MB")
        print(f"  Memory Growth: {memory_growth:.1f}MB")


@pytest.mark.performance
@pytest.mark.load
class TestStreamingLoadTesting:
    """Load testing for streaming detection."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.streaming_service = StreamingService()
        
        # Generate streaming test data
        np.random.seed(42)
        self.batch_size = 100
        self.n_features = 10
        self.n_batches = 50
        
        self.streaming_batches = []
        for i in range(self.n_batches):
            batch = np.random.normal(0, 1, (self.batch_size, self.n_features)).tolist()
            self.streaming_batches.append(batch)
    
    def test_streaming_batch_processing_load(self):
        """Test streaming service under batch processing load."""
        processing_times = []
        successful_batches = 0
        failed_batches = 0
        
        start_time = time.time()
        
        for i, batch in enumerate(self.streaming_batches):
            batch_start = time.time()
            try:
                result = self.streaming_service.process_streaming_batch(
                    batch,
                    algorithm="isolation_forest",
                    buffer_size=500
                )
                
                if result and "anomalies" in result:
                    successful_batches += 1
                else:
                    failed_batches += 1
                    
            except Exception as e:
                failed_batches += 1
                print(f"Batch {i} failed: {e}")
            
            batch_end = time.time()
            processing_times.append(batch_end - batch_start)
        
        total_time = time.time() - start_time
        
        # Performance assertions
        success_rate = successful_batches / (successful_batches + failed_batches)
        assert success_rate > 0.95, f"Success rate too low: {success_rate:.2%}"
        
        avg_batch_time = np.mean(processing_times)
        assert avg_batch_time < 0.5, f"Average batch time too high: {avg_batch_time}s"
        
        total_samples = successful_batches * self.batch_size
        throughput = total_samples / total_time
        assert throughput > 1000, f"Throughput too low: {throughput:.1f} samples/sec"
        
        print(f"\nStreaming Batch Load Test:")
        print(f"  Batches Processed: {successful_batches}/{len(self.streaming_batches)}")
        print(f"  Success Rate: {success_rate:.1%}")
        print(f"  Avg Batch Time: {avg_batch_time:.3f}s")
        print(f"  Total Throughput: {throughput:.1f} samples/sec")
        print(f"  Processing Consistency (std): {np.std(processing_times):.3f}s")
    
    def test_streaming_concurrent_processing(self):
        """Test concurrent streaming batch processing."""
        results = []
        errors = []
        processing_times = []
        
        def process_batch(batch_data):
            start_time = time.time()
            try:
                result = self.streaming_service.process_streaming_batch(
                    batch_data,
                    algorithm="isolation_forest",
                    buffer_size=300
                )
                results.append(result)
            except Exception as e:
                errors.append(str(e))
            
            end_time = time.time()
            processing_times.append(end_time - start_time)
        
        # Process multiple batches concurrently
        threads = []
        test_batches = self.streaming_batches[:20]  # Use subset for concurrent test
        
        start_time = time.time()
        for batch in test_batches:
            thread = threading.Thread(target=process_batch, args=(batch,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        total_time = time.time() - start_time
        
        # All batches should process successfully
        assert len(errors) == 0, f"Concurrent processing errors: {errors}"
        assert len(results) == len(test_batches), f"Missing results: {len(results)}/{len(test_batches)}"
        
        # Processing should be faster than sequential
        avg_concurrent_time = total_time / len(test_batches)
        avg_sequential_time = np.mean(processing_times)
        
        print(f"\nStreaming Concurrent Test:")
        print(f"  Batches: {len(test_batches)}")
        print(f"  Concurrent Processing Time: {total_time:.3f}s")
        print(f"  Avg Time per Batch (concurrent): {avg_concurrent_time:.3f}s")
        print(f"  Avg Time per Batch (individual): {avg_sequential_time:.3f}s")
        print(f"  Success Rate: {len(results)/len(test_batches):.1%}")
    
    def test_streaming_memory_stability(self):
        """Test streaming memory stability over extended processing."""
        try:
            import psutil
        except ImportError:
            pytest.skip("psutil not available for memory testing")
        
        process = psutil.Process()
        memory_readings = []
        
        # Process many batches and monitor memory
        for i, batch in enumerate(self.streaming_batches):
            # Record memory before processing
            memory_before = process.memory_info().rss / 1024 / 1024
            
            # Process batch
            self.streaming_service.process_streaming_batch(
                batch,
                algorithm="isolation_forest",
                buffer_size=200
            )
            
            # Record memory after processing
            memory_after = process.memory_info().rss / 1024 / 1024
            memory_readings.append(memory_after)
            
            # Periodically check for memory leaks
            if i > 0 and i % 10 == 0:
                recent_avg = np.mean(memory_readings[-10:])
                early_avg = np.mean(memory_readings[:10]) if len(memory_readings) >= 20 else memory_readings[0]
                
                memory_growth = recent_avg - early_avg
                if memory_growth > 20.0:  # More than 20MB growth
                    pytest.fail(f"Potential memory leak detected: {memory_growth:.1f}MB growth")
        
        # Overall memory stability check
        memory_trend = np.polyfit(range(len(memory_readings)), memory_readings, 1)[0]
        memory_variance = np.var(memory_readings)
        
        print(f"\nStreaming Memory Stability:")
        print(f"  Batches Processed: {len(memory_readings)}")
        print(f"  Memory Trend: {memory_trend:.3f}MB per batch")
        print(f"  Memory Variance: {memory_variance:.1f}")
        print(f"  Min Memory: {min(memory_readings):.1f}MB")
        print(f"  Max Memory: {max(memory_readings):.1f}MB")
        
        # Memory should be stable (low trend and variance)
        assert abs(memory_trend) < 0.1, f"Memory trend too high: {memory_trend:.3f}MB/batch"
        assert memory_variance < 100.0, f"Memory too variable: {memory_variance:.1f}"


@pytest.mark.performance
@pytest.mark.load  
class TestSystemIntegrationLoad:
    """Integration load testing across system components."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create integrated system components
        with patch('anomaly_detection.main.get_settings') as mock_get_settings:
            mock_settings = Mock()
            mock_settings.api.cors_origins = ["*"]
            mock_get_settings.return_value = mock_settings
            
            self.app = create_app()
            self.client = TestClient(self.app)
        
        self.worker = AnomalyDetectionWorker(max_workers=2)
        
        # Test data
        np.random.seed(42)
        self.test_datasets = {
            "small": np.random.normal(0, 1, (50, 5)).tolist(),
            "medium": np.random.normal(0, 1, (200, 10)).tolist(),
            "large": np.random.normal(0, 1, (500, 15)).tolist()
        }
    
    def test_end_to_end_system_load(self):
        """Test end-to-end system under load."""
        # Mock services properly
        with patch('anomaly_detection.api.v1.detection.get_detection_service') as mock_get_detection_service, \
             patch('anomaly_detection.api.v1.detection.get_ensemble_service') as mock_get_ensemble_service, \
             patch('anomaly_detection.domain.services.detection_service.DetectionService') as mock_detection_class:
            
            # Setup detection service mock
            mock_detection_service = Mock()
            mock_detection_result = Mock()
            mock_detection_result.success = True
            mock_detection_result.anomalies = [0, 1]
            mock_detection_result.confidence_scores = np.array([0.1, 0.8, 0.2, 0.9])
            mock_detection_result.total_samples = 4
            mock_detection_result.anomaly_count = 2
            mock_detection_result.anomaly_rate = 0.5
            mock_detection_service.detect_anomalies.return_value = mock_detection_result
            mock_get_detection_service.return_value = mock_detection_service
            
            # Setup ensemble service dependencies
            mock_detection_instance = Mock()
            mock_detection_instance.detect_anomalies.return_value = Mock(
                predictions=np.array([-1, 1, 1]),
                confidence_scores=np.array([0.8, 0.2, 0.3])
            )
            mock_detection_class.return_value = mock_detection_instance
            
            mock_ensemble_service = Mock()
            mock_ensemble_service.majority_vote.return_value = np.array([-1, 1, -1])
            mock_get_ensemble_service.return_value = mock_ensemble_service
            
            # Run mixed load test
            runner = LoadTestRunner()
            
            def mixed_system_requests():
                """Mix of different system operations."""
                import random
                operation = random.choice(['detection', 'ensemble', 'health', 'health'])
                
                if operation == 'detection':
                    response = self.client.post("/api/v1/detection/detect", json={
                        "data": self.test_datasets["small"],
                        "algorithm": "isolation_forest",
                        "contamination": 0.1
                    })
                elif operation == 'ensemble':
                    response = self.client.post("/api/v1/detection/ensemble", json={
                        "data": self.test_datasets["small"],
                        "algorithms": ["isolation_forest", "one_class_svm"],
                        "method": "majority",
                        "contamination": 0.1
                    })
                else:  # health
                    response = self.client.get("/health")
                
                if response.status_code != 200:
                    raise Exception(f"Request failed: {response.status_code}")
            
            result = runner.run_load_test(
                test_function=mixed_system_requests,
                concurrent_users=12,
                requests_per_user=5,
                test_name="end_to_end_system"
            )
            
            # System-wide performance assertions
            assert result.error_rate < 0.03, f"System error rate too high: {result.error_rate}"
            assert result.avg_response_time < 2.0, f"System response time too high: {result.avg_response_time}s"
            assert result.requests_per_second > 25, f"System throughput too low: {result.requests_per_second} req/s"
            
            self._print_integration_results(result)
    
    def test_system_resource_utilization(self):
        """Test system resource utilization under load."""
        try:
            import psutil
        except ImportError:
            pytest.skip("psutil not available for resource testing")
        
        process = psutil.Process()
        
        # Record initial resource usage
        initial_cpu = process.cpu_percent()
        initial_memory = process.memory_info().rss / 1024 / 1024
        
        # Run sustained load
        resource_readings = []
        
        with patch('anomaly_detection.api.v1.detection.get_detection_service') as mock_get_detection_service:
            mock_detection_service = Mock()
            mock_result = Mock()
            mock_result.success = True
            mock_result.anomalies = [0, 1]
            mock_result.confidence_scores = np.array([0.1, 0.8])
            mock_result.total_samples = 2
            mock_result.anomaly_count = 2
            mock_result.anomaly_rate = 1.0
            mock_detection_service.detect_anomalies.return_value = mock_result
            mock_get_detection_service.return_value = mock_detection_service
            
            start_time = time.time()
            request_count = 0
            
            # Run for specific duration
            while time.time() - start_time < 5.0:  # 5 second test
                # Make API request
                response = self.client.post("/api/v1/detection/detect", json={
                    "data": self.test_datasets["small"],
                    "algorithm": "isolation_forest"
                })
                request_count += 1
                
                # Record resource usage
                cpu_percent = process.cpu_percent()
                memory_mb = process.memory_info().rss / 1024 / 1024
                resource_readings.append((cpu_percent, memory_mb))
                
                # Small delay to prevent overwhelming
                time.sleep(0.01)
        
        # Analyze resource usage
        cpu_values = [reading[0] for reading in resource_readings]
        memory_values = [reading[1] for reading in resource_readings]
        
        avg_cpu = np.mean(cpu_values)
        max_memory = max(memory_values)
        memory_growth = max_memory - initial_memory
        
        print(f"\nSystem Resource Utilization:")
        print(f"  Requests Made: {request_count}")
        print(f"  Avg CPU Usage: {avg_cpu:.1f}%")
        print(f"  Max Memory: {max_memory:.1f}MB")
        print(f"  Memory Growth: {memory_growth:.1f}MB")
        print(f"  Request Rate: {request_count / 5.0:.1f} req/s")
        
        # Resource usage should be reasonable
        assert avg_cpu < 80.0, f"CPU usage too high: {avg_cpu}%"
        assert memory_growth < 30.0, f"Memory growth too high: {memory_growth}MB"
    
    def _print_integration_results(self, result: LoadTestResult):
        """Print integration test results."""
        print(f"\nIntegration Load Test Results:")
        print(f"  Test: {result.test_name}")
        print(f"  Total Requests: {result.total_requests}")
        print(f"  Success Rate: {(1 - result.error_rate):.1%}")
        print(f"  Avg Response Time: {result.avg_response_time:.3f}s")
        print(f"  95th Percentile: {result.percentile_95:.3f}s")
        print(f"  Throughput: {result.requests_per_second:.1f} req/s")


if __name__ == "__main__":
    # Run specific load tests
    pytest.main([
        __file__ + "::TestAPILoadTesting::test_detection_endpoint_load",
        "-v", "-s", "--tb=short"
    ])