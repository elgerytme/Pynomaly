"""Advanced load testing scenarios for anomaly detection system."""

import pytest
import asyncio
import time
import threading
import concurrent.futures
import numpy as np
import psutil
import gc
from typing import List, Dict, Any, Tuple, Optional
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
class StressTestResult:
    """Container for stress test results."""
    test_name: str
    peak_concurrent_users: int
    ramp_up_duration: float
    sustained_duration: float
    ramp_down_duration: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    peak_response_time: float
    avg_response_time: float
    memory_peak_mb: float
    memory_growth_mb: float
    cpu_peak_percent: float
    error_rate: float
    throughput_peak: float
    throughput_avg: float


@dataclass 
class ScalabilityTestResult:
    """Container for scalability test results."""
    test_name: str
    data_sizes: List[int]
    response_times: List[float]
    throughput_rates: List[float]
    memory_usage: List[float]
    scaling_factor: float  # How response time scales with data size
    efficiency_score: float  # Overall efficiency rating


class AdvancedLoadTestRunner:
    """Advanced load test execution framework with resource monitoring."""
    
    def __init__(self):
        self.response_times = []
        self.successful_requests = 0
        self.failed_requests = 0
        self.start_time = None
        self.end_time = None
        self.lock = threading.Lock()
        
        # Resource monitoring
        self.process = psutil.Process()
        self.memory_readings = []
        self.cpu_readings = []
        self.monitoring_active = False
        self.monitor_thread = None
        
    def start_monitoring(self):
        """Start resource monitoring."""
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_resources, daemon=True)
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
    
    def _monitor_resources(self):
        """Monitor CPU and memory usage."""
        while self.monitoring_active:
            try:
                cpu_percent = self.process.cpu_percent()
                memory_mb = self.process.memory_info().rss / 1024 / 1024
                
                with self.lock:
                    self.cpu_readings.append(cpu_percent)
                    self.memory_readings.append(memory_mb)
                
                time.sleep(0.1)  # Sample every 100ms
            except:
                break
    
    def record_request(self, response_time: float, success: bool):
        """Record a request result."""
        with self.lock:
            self.response_times.append(response_time)
            if success:
                self.successful_requests += 1
            else:
                self.failed_requests += 1
    
    def run_stress_test(
        self, 
        test_function, 
        max_users: int,
        ramp_up_seconds: int,
        sustain_seconds: int,
        ramp_down_seconds: int,
        test_name: str
    ) -> StressTestResult:
        """Run a stress test with gradual user ramp-up and ramp-down."""
        self.start_time = datetime.now()
        initial_memory = self.process.memory_info().rss / 1024 / 1024
        
        # Start resource monitoring
        self.start_monitoring()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_users * 2) as executor:
            futures = []
            
            # Ramp-up phase
            ramp_start = time.time()
            for i in range(max_users):
                delay = (i / max_users) * ramp_up_seconds
                future = executor.submit(self._delayed_stress_requests, 
                                       test_function, delay, sustain_seconds + ramp_down_seconds)
                futures.append(future)
            
            # Wait for ramp-up to complete
            time.sleep(ramp_up_seconds)
            
            # Sustain phase (already running)
            time.sleep(sustain_seconds)
            
            # Ramp-down phase (requests naturally finish)
            time.sleep(ramp_down_seconds)
            
            # Wait for all requests to complete
            concurrent.futures.wait(futures, timeout=60)
        
        self.end_time = datetime.now()
        self.stop_monitoring()
        
        # Calculate results
        duration = (self.end_time - self.start_time).total_seconds()
        total_requests = self.successful_requests + self.failed_requests
        
        final_memory = self.process.memory_info().rss / 1024 / 1024
        memory_growth = final_memory - initial_memory
        
        return StressTestResult(
            test_name=test_name,
            peak_concurrent_users=max_users,
            ramp_up_duration=ramp_up_seconds,
            sustained_duration=sustain_seconds,
            ramp_down_duration=ramp_down_seconds,
            total_requests=total_requests,
            successful_requests=self.successful_requests,
            failed_requests=self.failed_requests,
            peak_response_time=max(self.response_times) if self.response_times else 0,
            avg_response_time=np.mean(self.response_times) if self.response_times else 0,
            memory_peak_mb=max(self.memory_readings) if self.memory_readings else initial_memory,
            memory_growth_mb=memory_growth,
            cpu_peak_percent=max(self.cpu_readings) if self.cpu_readings else 0,
            error_rate=self.failed_requests / total_requests if total_requests > 0 else 0,
            throughput_peak=max_users / min(self.response_times) if self.response_times else 0,
            throughput_avg=total_requests / duration if duration > 0 else 0
        )
    
    def _delayed_stress_requests(self, test_function, delay: float, duration: float):
        """Execute requests with initial delay for ramp-up."""
        time.sleep(delay)
        
        end_time = time.time() + duration
        while time.time() < end_time:
            start_time = time.perf_counter()
            try:
                test_function()
                success = True
            except Exception:
                success = False
            
            response_time = time.perf_counter() - start_time
            self.record_request(response_time, success)
            
            # Small delay between requests from same user
            time.sleep(0.05)


@pytest.mark.performance
@pytest.mark.advanced_load
class TestAdvancedLoadScenarios:
    """Advanced load testing scenarios."""
    
    def setup_method(self):
        """Set up test fixtures."""
        with patch('anomaly_detection.main.get_settings') as mock_get_settings:
            mock_settings = Mock()
            mock_settings.api.cors_origins = ["*"]
            mock_get_settings.return_value = mock_settings
            
            self.app = create_app()
            self.client = TestClient(self.app)
        
        # Test datasets of varying sizes
        np.random.seed(42)
        self.test_datasets = {
            "tiny": np.random.normal(0, 1, (10, 3)).tolist(),
            "small": np.random.normal(0, 1, (50, 5)).tolist(),
            "medium": np.random.normal(0, 1, (200, 10)).tolist(),
            "large": np.random.normal(0, 1, (1000, 20)).tolist(),
            "xlarge": np.random.normal(0, 1, (5000, 50)).tolist()
        }
        
        # Add some actual anomalies to datasets
        for dataset_name, dataset in self.test_datasets.items():
            if len(dataset) > 10:
                # Make last 10% of data anomalous by scaling up
                anomaly_count = len(dataset) // 10
                for i in range(-anomaly_count, 0):
                    for j in range(len(dataset[i])):
                        dataset[i][j] *= 3  # Make it anomalous
    
    def _detection_request_with_data(self, data: List[List[float]]):
        """Single detection API request with specific data."""
        response = self.client.post("/api/v1/detection/detect", json={
            "data": data,
            "algorithm": "isolation_forest",
            "contamination": 0.1
        })
        if response.status_code != 200:
            raise Exception(f"API request failed: {response.status_code} - {response.text}")
        return response.json()
    
    def _ensemble_request_with_data(self, data: List[List[float]]):
        """Single ensemble API request with specific data."""
        response = self.client.post("/api/v1/detection/ensemble", json={
            "data": data,
            "algorithms": ["isolation_forest", "one_class_svm"],
            "method": "majority",
            "contamination": 0.1
        })
        if response.status_code != 200:
            raise Exception(f"API request failed: {response.status_code} - {response.text}")
        return response.json()
    
    def test_stress_test_gradual_rampup(self):
        """Test system under gradual stress with ramp-up and ramp-down."""
        runner = AdvancedLoadTestRunner()
        
        def stress_request():
            return self._detection_request_with_data(self.test_datasets["small"])
        
        result = runner.run_stress_test(
            test_function=stress_request,
            max_users=20,
            ramp_up_seconds=10,
            sustain_seconds=30,
            ramp_down_seconds=10,
            test_name="gradual_rampup_stress"
        )
        
        # Stress test assertions
        assert result.error_rate < 0.10, f"Too many errors under stress: {result.error_rate}"
        assert result.avg_response_time < 10.0, f"Response time too high under stress: {result.avg_response_time}s"
        assert result.memory_growth_mb < 200, f"Memory growth too high: {result.memory_growth_mb}MB"
        assert result.cpu_peak_percent < 95, f"CPU usage too high: {result.cpu_peak_percent}%"
        
        self._print_stress_test_results(result)
    
    def test_data_size_scalability(self):
        """Test how performance scales with input data size."""
        datasets = ["tiny", "small", "medium", "large"]
        response_times = []
        throughput_rates = []
        memory_usage = []
        
        for dataset_name in datasets:
            data = self.test_datasets[dataset_name]
            
            # Measure single request performance
            gc.collect()  # Clean up before measurement
            initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            start_time = time.perf_counter()
            try:
                self._detection_request_with_data(data)
                success = True
            except Exception:
                success = False
            end_time = time.perf_counter()
            
            final_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            response_time = end_time - start_time
            throughput = 1 / response_time if response_time > 0 else 0
            memory_used = final_memory - initial_memory
            
            response_times.append(response_time)
            throughput_rates.append(throughput)
            memory_usage.append(memory_used)
            
            print(f"\nData size scalability - {dataset_name}:")
            print(f"  Samples: {len(data)}, Features: {len(data[0])}")
            print(f"  Response time: {response_time:.3f}s")
            print(f"  Throughput: {throughput:.2f} req/s")
            print(f"  Memory usage: {memory_used:.1f}MB")
        
        # Calculate scaling characteristics
        data_sizes = [len(self.test_datasets[name]) for name in datasets]
        
        # Linear regression to find scaling factor
        log_sizes = np.log(data_sizes)
        log_times = np.log(response_times)
        scaling_factor = np.polyfit(log_sizes, log_times, 1)[0]
        
        # Efficiency score (lower response time and memory per sample)
        samples_per_second = [len(self.test_datasets[datasets[i]]) / response_times[i] 
                            for i in range(len(datasets))]
        efficiency_score = np.mean(samples_per_second)
        
        result = ScalabilityTestResult(
            test_name="data_size_scalability",
            data_sizes=data_sizes,
            response_times=response_times,
            throughput_rates=throughput_rates,
            memory_usage=memory_usage,
            scaling_factor=scaling_factor,
            efficiency_score=efficiency_score
        )
        
        # Scalability assertions
        assert scaling_factor < 2.0, f"Algorithm scales poorly with data size: {scaling_factor}"
        assert all(t < 30.0 for t in response_times), f"Some response times too high: {max(response_times)}s"
        assert efficiency_score > 10, f"Processing efficiency too low: {efficiency_score} samples/s"
        
        print(f"\nScalability Analysis:")
        print(f"  Scaling factor: {scaling_factor:.2f} (< 1 = sublinear, ~1 = linear, > 1 = superlinear)")
        print(f"  Efficiency score: {efficiency_score:.1f} samples/s")
    
    def test_algorithm_comparison_load(self):
        """Compare performance of different algorithms under load."""
        algorithms = ["isolation_forest", "one_class_svm", "local_outlier_factor"]
        algorithm_results = {}
        
        for algorithm in algorithms:
            response_times = []
            error_count = 0
            
            # Test each algorithm with multiple requests
            for i in range(10):
                start_time = time.perf_counter()
                try:
                    response = self.client.post("/api/v1/detection/detect", json={
                        "data": self.test_datasets["medium"],
                        "algorithm": algorithm,
                        "contamination": 0.1
                    })
                    if response.status_code != 200:
                        error_count += 1
                except Exception:
                    error_count += 1
                
                response_time = time.perf_counter() - start_time
                response_times.append(response_time)
            
            algorithm_results[algorithm] = {
                "avg_response_time": np.mean(response_times),
                "std_response_time": np.std(response_times),
                "min_response_time": min(response_times),
                "max_response_time": max(response_times),
                "error_count": error_count,
                "error_rate": error_count / len(response_times),
                "throughput": len(response_times) / sum(response_times)
            }
        
        # Performance comparison analysis
        print(f"\nAlgorithm Performance Comparison:")
        fastest_algorithm = min(algorithm_results.keys(), 
                              key=lambda x: algorithm_results[x]["avg_response_time"])
        most_stable = min(algorithm_results.keys(),
                         key=lambda x: algorithm_results[x]["std_response_time"])
        
        for algorithm, results in algorithm_results.items():
            print(f"  {algorithm}:")
            print(f"    Avg Response: {results['avg_response_time']:.3f}s")
            print(f"    Std Dev: {results['std_response_time']:.3f}s")
            print(f"    Error Rate: {results['error_rate']:.1%}")
            print(f"    Throughput: {results['throughput']:.2f} req/s")
        
        print(f"\n  Fastest: {fastest_algorithm}")
        print(f"  Most Stable: {most_stable}")
        
        # Assertions for algorithm performance
        for algorithm, results in algorithm_results.items():
            assert results["error_rate"] < 0.20, f"{algorithm} error rate too high: {results['error_rate']}"
            assert results["avg_response_time"] < 15.0, f"{algorithm} too slow: {results['avg_response_time']}s"
    
    def test_memory_leak_detection(self):
        """Test for memory leaks during extended operation."""
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_readings = []
        
        # Run many requests and monitor memory
        for i in range(50):
            # Mix of different request types
            if i % 3 == 0:
                self._detection_request_with_data(self.test_datasets["small"])
            elif i % 3 == 1:
                self._detection_request_with_data(self.test_datasets["medium"])
            else:
                self._ensemble_request_with_data(self.test_datasets["small"])
            
            # Record memory every 10 requests
            if i % 10 == 0:
                current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                memory_readings.append(current_memory)
                
                # Force garbage collection periodically
                if i % 20 == 0:
                    gc.collect()
        
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_growth = final_memory - initial_memory
        
        # Calculate memory trend
        if len(memory_readings) > 2:
            memory_trend = np.polyfit(range(len(memory_readings)), memory_readings, 1)[0]
        else:
            memory_trend = 0
        
        print(f"\nMemory Leak Analysis:")
        print(f"  Initial Memory: {initial_memory:.1f}MB")
        print(f"  Final Memory: {final_memory:.1f}MB")
        print(f"  Memory Growth: {memory_growth:.1f}MB")
        print(f"  Memory Trend: {memory_trend:.3f}MB per 10 requests")
        print(f"  Memory Readings: {[f'{m:.1f}' for m in memory_readings]}")
        
        # Memory leak assertions
        assert memory_growth < 100, f"Excessive memory growth: {memory_growth}MB"
        assert abs(memory_trend) < 2.0, f"Memory leak detected: {memory_trend}MB per 10 requests"
    
    def test_concurrent_algorithm_execution(self):
        """Test concurrent execution of different algorithms."""
        algorithms = ["isolation_forest", "one_class_svm", "local_outlier_factor"]
        results = []
        errors = []
        
        def concurrent_request(algorithm):
            try:
                start_time = time.perf_counter()
                response = self.client.post("/api/v1/detection/detect", json={
                    "data": self.test_datasets["medium"],
                    "algorithm": algorithm,
                    "contamination": 0.1
                })
                end_time = time.perf_counter()
                
                if response.status_code == 200:
                    results.append({
                        "algorithm": algorithm,
                        "response_time": end_time - start_time,
                        "success": True
                    })
                else:
                    errors.append(f"{algorithm}: {response.status_code}")
            except Exception as e:
                errors.append(f"{algorithm}: {str(e)}")
        
        # Run multiple algorithms concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(algorithms) * 3) as executor:
            futures = []
            
            # Submit multiple requests for each algorithm
            for _ in range(5):  # 5 rounds of concurrent requests
                for algorithm in algorithms:
                    future = executor.submit(concurrent_request, algorithm)
                    futures.append(future)
            
            # Wait for all requests to complete
            concurrent.futures.wait(futures, timeout=120)
        
        # Analyze results
        total_requests = len(algorithms) * 5
        successful_requests = len(results)
        error_rate = len(errors) / total_requests
        
        algorithm_stats = {}
        for algorithm in algorithms:
            algo_results = [r for r in results if r["algorithm"] == algorithm]
            if algo_results:
                response_times = [r["response_time"] for r in algo_results]
                algorithm_stats[algorithm] = {
                    "count": len(algo_results),
                    "avg_time": np.mean(response_times),
                    "success_rate": len(algo_results) / 5  # 5 requests per algorithm
                }
        
        print(f"\nConcurrent Algorithm Execution:")
        print(f"  Total Requests: {total_requests}")
        print(f"  Successful: {successful_requests}")
        print(f"  Error Rate: {error_rate:.1%}")
        print(f"  Errors: {errors[:5]}")  # Show first 5 errors
        
        for algorithm, stats in algorithm_stats.items():
            print(f"  {algorithm}:")
            print(f"    Success Rate: {stats['success_rate']:.1%}")
            print(f"    Avg Response Time: {stats['avg_time']:.3f}s")
        
        # Concurrent execution assertions
        assert error_rate < 0.30, f"Too many concurrent execution errors: {error_rate}"
        assert successful_requests > total_requests * 0.7, f"Too many failed concurrent requests"
        
        # Each algorithm should work under concurrency
        for algorithm in algorithms:
            if algorithm in algorithm_stats:
                assert algorithm_stats[algorithm]["success_rate"] > 0.6, f"{algorithm} fails under concurrency"
    
    def test_burst_traffic_handling(self):
        """Test handling of sudden burst traffic."""
        burst_size = 30
        results = []
        errors = []
        
        def burst_request():
            try:
                start_time = time.perf_counter()
                response = self._detection_request_with_data(self.test_datasets["small"])
                end_time = time.perf_counter()
                
                results.append({
                    "response_time": end_time - start_time,
                    "success": True
                })
            except Exception as e:
                errors.append(str(e))
        
        # Create sudden burst of requests
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=burst_size) as executor:
            futures = [executor.submit(burst_request) for _ in range(burst_size)]
            concurrent.futures.wait(futures, timeout=60)
        
        end_time = time.time()
        burst_duration = end_time - start_time
        
        # Analyze burst handling
        successful_requests = len(results)
        error_rate = len(errors) / burst_size
        avg_response_time = np.mean([r["response_time"] for r in results]) if results else 0
        burst_throughput = successful_requests / burst_duration
        
        print(f"\nBurst Traffic Analysis:")
        print(f"  Burst Size: {burst_size} concurrent requests")
        print(f"  Burst Duration: {burst_duration:.2f}s")
        print(f"  Successful Requests: {successful_requests}/{burst_size}")
        print(f"  Error Rate: {error_rate:.1%}")
        print(f"  Avg Response Time: {avg_response_time:.3f}s")
        print(f"  Burst Throughput: {burst_throughput:.1f} req/s")
        
        # Burst handling assertions
        assert error_rate < 0.20, f"Poor burst handling - error rate: {error_rate}"
        assert successful_requests > burst_size * 0.8, f"Too many failures during burst"
        assert avg_response_time < 15.0, f"Response time too high during burst: {avg_response_time}s"
    
    def _print_stress_test_results(self, result: StressTestResult):
        """Print formatted stress test results."""
        print(f"\nStress Test Results - {result.test_name}:")
        print(f"  Peak Concurrent Users: {result.peak_concurrent_users}")
        print(f"  Ramp-up Duration: {result.ramp_up_duration}s")
        print(f"  Sustained Duration: {result.sustained_duration}s")
        print(f"  Total Requests: {result.total_requests}")
        print(f"  Success Rate: {(1 - result.error_rate):.1%}")
        print(f"  Avg Response Time: {result.avg_response_time:.3f}s")
        print(f"  Peak Response Time: {result.peak_response_time:.3f}s")
        print(f"  Memory Peak: {result.memory_peak_mb:.1f}MB")
        print(f"  Memory Growth: {result.memory_growth_mb:.1f}MB")
        print(f"  CPU Peak: {result.cpu_peak_percent:.1f}%")
        print(f"  Avg Throughput: {result.throughput_avg:.1f} req/s")


if __name__ == "__main__":
    # Run specific advanced load tests
    pytest.main([
        __file__ + "::TestAdvancedLoadScenarios::test_stress_test_gradual_rampup",
        "-v", "-s", "--tb=short"
    ])