"""System-wide performance tests for the anomaly detection platform."""

import pytest
import time
import numpy as np
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Dict, List, Any
import psutil
import requests
from fastapi.testclient import TestClient

from anomaly_detection.server import create_app
from anomaly_detection.domain.services.detection_service import DetectionService
from anomaly_detection.infrastructure.monitoring.metrics_collector import MetricsCollector


class TestSystemPerformance:
    """Performance tests for the complete system under various load conditions."""
    
    @pytest.fixture
    def api_client(self):
        """Create API test client for performance testing."""
        app = create_app()
        return TestClient(app)
    
    @pytest.fixture
    def large_test_dataset(self):
        """Generate large dataset for stress testing."""
        np.random.seed(42)
        # Create realistic financial transaction data
        normal_data = np.random.multivariate_normal(
            mean=[100, 5, 365],  # amount, frequency, account_age
            cov=[[2500, 10, 100], [10, 4, 20], [100, 20, 10000]],
            size=50000
        )
        
        # Add some anomalies
        anomaly_data = np.random.multivariate_normal(
            mean=[1000, 1, 30],  # high amount, low frequency, new account
            cov=[[10000, 5, 50], [5, 1, 10], [50, 10, 500]],
            size=2500
        )
        
        return np.vstack([normal_data, anomaly_data])
    
    @pytest.fixture
    def streaming_dataset(self):
        """Generate streaming dataset for real-time performance tests."""
        np.random.seed(42)
        # Generate data that arrives in batches
        batches = []
        for i in range(100):  # 100 batches
            batch_size = np.random.randint(50, 200)
            batch = np.random.randn(batch_size, 8)
            # Add timestamp simulation
            timestamps = [time.time() + i * 0.1 + j * 0.001 for j in range(batch_size)]
            batches.append((batch, timestamps))
        
        return batches
    
    def measure_api_performance(self, client: TestClient, endpoint: str, payload: Dict, num_requests: int = 10) -> Dict[str, Any]:
        """Measure API endpoint performance under load."""
        response_times = []
        success_count = 0
        error_count = 0
        status_codes = []
        
        for i in range(num_requests):
            start_time = time.perf_counter()
            
            try:
                response = client.post(endpoint, json=payload, timeout=60)
                end_time = time.perf_counter()
                
                response_time = (end_time - start_time) * 1000  # Convert to ms
                response_times.append(response_time)
                status_codes.append(response.status_code)
                
                if 200 <= response.status_code < 300:
                    success_count += 1
                else:
                    error_count += 1
                    
            except Exception as e:
                end_time = time.perf_counter()
                response_time = (end_time - start_time) * 1000
                response_times.append(response_time)
                error_count += 1
                status_codes.append(0)  # Timeout/connection error
        
        return {
            'avg_response_time_ms': np.mean(response_times) if response_times else 0,
            'median_response_time_ms': np.median(response_times) if response_times else 0,
            'p95_response_time_ms': np.percentile(response_times, 95) if response_times else 0,
            'p99_response_time_ms': np.percentile(response_times, 99) if response_times else 0,
            'min_response_time_ms': np.min(response_times) if response_times else 0,
            'max_response_time_ms': np.max(response_times) if response_times else 0,
            'success_rate': success_count / num_requests if num_requests > 0 else 0,
            'error_rate': error_count / num_requests if num_requests > 0 else 0,
            'total_requests': num_requests,
            'successful_requests': success_count,
            'failed_requests': error_count,
            'status_codes': status_codes
        }
    
    def test_api_endpoint_performance_under_load(self, api_client: TestClient, large_test_dataset):
        """Test API endpoint performance under sustained load."""
        print("\n=== API Endpoint Load Test ===")
        
        # Use smaller subset for API testing
        test_data = large_test_dataset[:1000].tolist()
        
        payload = {
            "data": test_data,
            "algorithm": "isolation_forest",
            "contamination": 0.1,
            "parameters": {"n_estimators": 50, "random_state": 42}
        }
        
        # Test detection endpoint under load
        print("Testing /api/v1/detect endpoint...")
        
        performance = self.measure_api_performance(
            client=api_client,
            endpoint="/api/v1/detect",
            payload=payload,
            num_requests=20
        )
        
        print(f"Performance Results:")
        print(f"  Average response time: {performance['avg_response_time_ms']:.0f}ms")
        print(f"  Median response time: {performance['median_response_time_ms']:.0f}ms")
        print(f"  95th percentile: {performance['p95_response_time_ms']:.0f}ms")
        print(f"  99th percentile: {performance['p99_response_time_ms']:.0f}ms")
        print(f"  Success rate: {performance['success_rate']:.1%}")
        print(f"  Error rate: {performance['error_rate']:.1%}")
        
        # Performance assertions
        assert performance['success_rate'] >= 0.8, f"Success rate too low: {performance['success_rate']:.1%}"
        assert performance['avg_response_time_ms'] < 30000, f"Average response time too high: {performance['avg_response_time_ms']:.0f}ms"
        assert performance['p95_response_time_ms'] < 60000, f"95th percentile too high: {performance['p95_response_time_ms']:.0f}ms"
    
    def test_concurrent_api_requests(self, api_client: TestClient):
        """Test API performance with concurrent requests."""
        print("\n=== Concurrent API Request Test ===")
        
        # Smaller dataset for concurrent testing
        np.random.seed(42)
        test_data = np.random.randn(500, 10).tolist()
        
        def make_request(request_id: int) -> Dict[str, Any]:
            """Make a single API request."""
            payload = {
                "data": test_data,
                "algorithm": "isolation_forest",
                "contamination": 0.1,
                "parameters": {"random_state": 42 + request_id}
            }
            
            start_time = time.perf_counter()
            
            try:
                response = api_client.post("/api/v1/detect", json=payload, timeout=45)
                end_time = time.perf_counter()
                
                return {
                    'request_id': request_id,
                    'success': 200 <= response.status_code < 300,
                    'status_code': response.status_code,
                    'response_time_ms': (end_time - start_time) * 1000,
                    'error': None
                }
                
            except Exception as e:
                end_time = time.perf_counter()
                return {
                    'request_id': request_id,
                    'success': False,
                    'status_code': 0,
                    'response_time_ms': (end_time - start_time) * 1000,
                    'error': str(e)
                }
        
        # Run concurrent requests
        num_concurrent = 8
        print(f"Running {num_concurrent} concurrent requests...")
        
        start_time = time.perf_counter()
        
        with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futures = [executor.submit(make_request, i) for i in range(num_concurrent)]
            results = [future.result() for future in futures]
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        # Analyze results
        successful_requests = [r for r in results if r['success']]
        failed_requests = [r for r in results if not r['success']]
        
        if successful_requests:
            avg_response_time = np.mean([r['response_time_ms'] for r in successful_requests])
            max_response_time = np.max([r['response_time_ms'] for r in successful_requests])
            min_response_time = np.min([r['response_time_ms'] for r in successful_requests])
        else:
            avg_response_time = max_response_time = min_response_time = 0
        
        success_rate = len(successful_requests) / len(results)
        throughput = len(results) / total_time
        
        print(f"Concurrent Performance Results:")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Success rate: {success_rate:.1%}")
        print(f"  Throughput: {throughput:.2f} requests/sec")
        print(f"  Average response time: {avg_response_time:.0f}ms")
        print(f"  Min/Max response time: {min_response_time:.0f}ms / {max_response_time:.0f}ms")
        
        if failed_requests:
            print(f"  Failed requests: {len(failed_requests)}")
            for req in failed_requests[:3]:  # Show first 3 failures
                print(f"    Request {req['request_id']}: {req.get('error', 'Unknown error')}")
        
        # Concurrent performance assertions
        assert success_rate >= 0.7, f"Concurrent success rate too low: {success_rate:.1%}"
        assert throughput > 0.1, f"Throughput too low: {throughput:.2f} requests/sec"
    
    def test_memory_usage_under_load(self, large_test_dataset):
        """Test memory usage patterns under sustained load."""
        print("\n=== Memory Usage Under Load Test ===")
        
        detection_service = DetectionService()
        process = psutil.Process()
        
        # Record initial memory
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_readings = [initial_memory]
        
        print(f"Initial memory usage: {initial_memory:.1f} MB")
        
        # Process dataset in chunks to simulate sustained load
        chunk_size = 5000
        num_chunks = min(10, len(large_test_dataset) // chunk_size)
        
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, len(large_test_dataset))
            chunk_data = large_test_dataset[start_idx:end_idx]
            
            print(f"Processing chunk {chunk_idx + 1}/{num_chunks} ({len(chunk_data)} samples)...")
            
            # Process chunk
            result = detection_service.detect_anomalies(
                data=chunk_data,
                algorithm='iforest',
                contamination=0.05,
                n_estimators=50
            )
            
            assert result.success, f"Detection failed on chunk {chunk_idx + 1}"
            
            # Record memory after processing
            current_memory = process.memory_info().rss / 1024 / 1024
            memory_readings.append(current_memory)
            
            print(f"  Memory after chunk: {current_memory:.1f} MB")
            print(f"  Detected anomalies: {result.anomaly_count}")
        
        # Analyze memory usage
        final_memory = memory_readings[-1]
        peak_memory = max(memory_readings)
        memory_growth = final_memory - initial_memory
        memory_variance = np.var(memory_readings)
        
        print(f"\nMemory Usage Analysis:")
        print(f"  Initial: {initial_memory:.1f} MB")
        print(f"  Final: {final_memory:.1f} MB")
        print(f"  Peak: {peak_memory:.1f} MB")
        print(f"  Growth: {memory_growth:.1f} MB")
        print(f"  Variance: {memory_variance:.1f}")
        
        # Memory usage assertions
        assert memory_growth < 1000, f"Excessive memory growth: {memory_growth:.1f} MB"
        assert peak_memory < initial_memory + 2000, f"Peak memory too high: {peak_memory:.1f} MB"
    
    def test_cpu_utilization_efficiency(self, large_test_dataset):
        """Test CPU utilization efficiency under load."""
        print("\n=== CPU Utilization Efficiency Test ===")
        
        detection_service = DetectionService()
        dataset = large_test_dataset[:10000]  # Use manageable size
        
        # Monitor CPU usage during detection
        cpu_readings = []
        
        def monitor_cpu():
            """Monitor CPU usage in background."""
            while True:
                try:
                    cpu_percent = psutil.cpu_percent(interval=0.1)
                    cpu_readings.append(cpu_percent)
                    if len(cpu_readings) > 300:  # Stop after 30 seconds of monitoring
                        break
                except:
                    break
        
        # Start CPU monitoring
        monitor_thread = threading.Thread(target=monitor_cpu, daemon=True)
        monitor_thread.start()
        
        # Run detection
        start_time = time.perf_counter()
        
        result = detection_service.detect_anomalies(
            data=dataset,
            algorithm='iforest',
            contamination=0.05,
            n_estimators=100,
            n_jobs=-1  # Use all available cores
        )
        
        end_time = time.perf_counter()
        detection_time = end_time - start_time
        
        # Wait a bit for CPU monitoring to complete
        time.sleep(1)
        
        # Analyze CPU usage
        if cpu_readings:
            avg_cpu = np.mean(cpu_readings)
            max_cpu = np.max(cpu_readings)
            cpu_variance = np.var(cpu_readings)
            
            print(f"CPU Utilization Analysis:")
            print(f"  Detection time: {detection_time:.2f}s")
            print(f"  Average CPU usage: {avg_cpu:.1f}%")
            print(f"  Peak CPU usage: {max_cpu:.1f}%")
            print(f"  CPU variance: {cpu_variance:.1f}")
            print(f"  Samples processed: {result.total_samples}")
            print(f"  Processing rate: {result.total_samples / detection_time:.0f} samples/sec")
            
            # CPU efficiency assertions
            assert avg_cpu > 10, f"CPU utilization too low: {avg_cpu:.1f}% (possible inefficiency)"
            assert avg_cpu < 90, f"CPU utilization too high: {avg_cpu:.1f}% (possible bottleneck)"
            assert result.success, "Detection should succeed under normal CPU load"
        
        else:
            print("Warning: Could not monitor CPU usage")
    
    def test_throughput_scalability(self):
        """Test throughput scalability with different dataset sizes."""
        print("\n=== Throughput Scalability Test ===")
        
        detection_service = DetectionService()
        
        # Test different dataset sizes
        sizes = [1000, 5000, 10000, 25000]
        throughput_results = {}
        
        for size in sizes:
            np.random.seed(42)
            dataset = np.random.randn(size, 15)
            
            print(f"Testing throughput with {size} samples...")
            
            start_time = time.perf_counter()
            
            result = detection_service.detect_anomalies(
                data=dataset,
                algorithm='iforest',
                contamination=0.1,
                n_estimators=50,
                random_state=42
            )
            
            end_time = time.perf_counter()
            
            if result.success:
                processing_time = end_time - start_time
                throughput = size / processing_time
                
                throughput_results[size] = {
                    'processing_time': processing_time,
                    'throughput': throughput,
                    'anomalies_detected': result.anomaly_count
                }
                
                print(f"  Processing time: {processing_time:.2f}s")
                print(f"  Throughput: {throughput:.0f} samples/sec")
                print(f"  Anomalies detected: {result.anomaly_count}")
        
        # Analyze scalability
        if len(throughput_results) >= 2:
            sizes_tested = sorted(throughput_results.keys())
            throughputs = [throughput_results[s]['throughput'] for s in sizes_tested]
            
            print(f"\nThroughput Scalability Analysis:")
            for i, size in enumerate(sizes_tested):
                print(f"  {size:,} samples: {throughputs[i]:.0f} samples/sec")
            
            # Check if throughput remains relatively stable (good scalability)
            throughput_variance = np.var(throughputs)
            avg_throughput = np.mean(throughputs)
            coefficient_of_variation = np.sqrt(throughput_variance) / avg_throughput
            
            print(f"  Average throughput: {avg_throughput:.0f} samples/sec")
            print(f"  Coefficient of variation: {coefficient_of_variation:.2f}")
            
            # Scalability assertions
            assert avg_throughput > 500, f"Average throughput too low: {avg_throughput:.0f} samples/sec"
            assert coefficient_of_variation < 0.5, f"Throughput too variable: {coefficient_of_variation:.2f}"
    
    def test_streaming_performance(self, streaming_dataset):
        """Test real-time streaming performance."""
        print("\n=== Streaming Performance Test ===")
        
        from anomaly_detection.domain.services.streaming_service import StreamingService
        
        streaming_service = StreamingService(window_size=200, update_frequency=50)
        
        # Process streaming batches
        total_samples = 0
        total_anomalies = 0
        processing_times = []
        
        start_time = time.perf_counter()
        
        for batch_idx, (batch_data, timestamps) in enumerate(streaming_dataset[:20]):  # Test first 20 batches
            batch_start = time.perf_counter()
            
            # Process batch
            batch_results = []
            for sample in batch_data:
                result = streaming_service.process_sample(sample)
                batch_results.append(result)
                if result.is_anomaly:
                    total_anomalies += 1
            
            batch_end = time.perf_counter()
            batch_time = batch_end - batch_start
            processing_times.append(batch_time)
            
            total_samples += len(batch_data)
            
            if (batch_idx + 1) % 5 == 0:  # Print every 5 batches
                batch_throughput = len(batch_data) / batch_time
                print(f"  Batch {batch_idx + 1}: {len(batch_data)} samples in {batch_time:.3f}s ({batch_throughput:.0f} samples/sec)")
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        # Analyze streaming performance
        avg_batch_time = np.mean(processing_times)
        max_batch_time = np.max(processing_times)
        overall_throughput = total_samples / total_time
        
        print(f"\nStreaming Performance Results:")
        print(f"  Total samples processed: {total_samples}")
        print(f"  Total processing time: {total_time:.2f}s")
        print(f"  Overall throughput: {overall_throughput:.0f} samples/sec")
        print(f"  Average batch time: {avg_batch_time:.3f}s")
        print(f"  Max batch time: {max_batch_time:.3f}s")
        print(f"  Total anomalies detected: {total_anomalies}")
        print(f"  Anomaly rate: {total_anomalies/total_samples:.1%}")
        
        # Streaming performance assertions
        assert overall_throughput > 1000, f"Streaming throughput too low: {overall_throughput:.0f} samples/sec"
        assert max_batch_time < 1.0, f"Max batch time too high: {max_batch_time:.3f}s"
        assert avg_batch_time < 0.5, f"Average batch time too high: {avg_batch_time:.3f}s"


if __name__ == "__main__":
    print("Anomaly Detection System Performance Test Suite")
    print("=" * 50)
    print("Testing system-wide performance under various load conditions")
    print("Includes API performance, memory usage, CPU utilization, and scalability")
    print()
    
    # Quick system check
    try:
        import psutil
        
        cpu_count = psutil.cpu_count()
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        print(f"System Resources:")
        print(f"  CPU cores: {cpu_count}")
        print(f"  Total memory: {memory_gb:.1f} GB")
        print(f"  Current CPU usage: {psutil.cpu_percent(interval=1):.1f}%")
        print(f"  Current memory usage: {psutil.virtual_memory().percent:.1f}%")
        print()
        print("✓ System monitoring capabilities available")
        print("Ready to run performance tests")
        
    except Exception as e:
        print(f"✗ System monitoring setup failed: {e}")
        print("Some performance tests may not provide detailed metrics")