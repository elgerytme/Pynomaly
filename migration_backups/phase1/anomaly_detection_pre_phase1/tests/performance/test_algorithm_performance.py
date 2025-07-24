"""Performance tests for anomaly detection algorithms."""

import pytest
import time
import numpy as np
import psutil
import threading
from typing import Dict, List, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
# from memory_profiler import profile  # Optional dependency
import gc

from anomaly_detection.domain.services.detection_service import DetectionService
from anomaly_detection.domain.services.ensemble_service import EnsembleService
from anomaly_detection.infrastructure.adapters.algorithms import SklearnAdapter


class TestAlgorithmPerformance:
    """Performance benchmarks for various algorithms."""
    
    @pytest.fixture
    def detection_service(self):
        """Create detection service for performance testing."""
        return DetectionService()
    
    @pytest.fixture
    def ensemble_service(self):
        """Create ensemble service for performance testing."""
        return EnsembleService()
    
    @pytest.fixture
    def small_dataset(self):
        """Generate small dataset (1K samples) for quick tests."""
        np.random.seed(42)
        return np.random.randn(1000, 10)
    
    @pytest.fixture
    def medium_dataset(self):
        """Generate medium dataset (10K samples) for standard tests."""
        np.random.seed(42)
        return np.random.randn(10000, 20)
    
    @pytest.fixture
    def large_dataset(self):
        """Generate large dataset (100K samples) for stress tests."""
        np.random.seed(42)
        return np.random.randn(100000, 50)
    
    @pytest.fixture
    def high_dimensional_dataset(self):
        """Generate high-dimensional dataset for dimension scalability tests."""
        np.random.seed(42)
        return np.random.randn(5000, 200)
    
    def measure_performance(self, func, *args, **kwargs) -> Dict[str, float]:
        """Measure execution time and memory usage of a function."""
        process = psutil.Process()
        
        # Get initial memory usage
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Force garbage collection before measurement
        gc.collect()
        
        start_time = time.perf_counter()
        cpu_start = time.process_time()
        
        try:
            result = func(*args, **kwargs)
            success = True
        except Exception as e:
            result = None
            success = False
            print(f"Function failed: {e}")
        
        end_time = time.perf_counter()
        cpu_end = time.process_time()
        
        # Get peak memory usage
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_delta = peak_memory - initial_memory
        
        return {
            'wall_time': end_time - start_time,
            'cpu_time': cpu_end - cpu_start,
            'memory_mb': memory_delta,
            'peak_memory_mb': peak_memory,
            'success': success,
            'result': result
        }
    
    def test_isolation_forest_scalability(self, detection_service, small_dataset, medium_dataset, large_dataset):
        """Test Isolation Forest performance across different dataset sizes."""
        print("\n=== Isolation Forest Scalability Test ===")
        
        datasets = [
            ("Small (1K)", small_dataset),
            ("Medium (10K)", medium_dataset),
            ("Large (100K)", large_dataset)
        ]
        
        results = {}
        
        for name, dataset in datasets:
            print(f"\nTesting {name} dataset ({dataset.shape[0]} samples, {dataset.shape[1]} features)")
            
            perf = self.measure_performance(
                detection_service.detect_anomalies,
                data=dataset,
                algorithm='iforest',
                contamination=0.1,
                n_estimators=100,
                random_state=42
            )
            
            results[name] = perf
            
            if perf['success']:
                samples_per_sec = dataset.shape[0] / perf['wall_time']
                print(f"  Wall time: {perf['wall_time']:.2f}s")
                print(f"  CPU time: {perf['cpu_time']:.2f}s")
                print(f"  Memory: {perf['memory_mb']:.1f} MB")
                print(f"  Throughput: {samples_per_sec:.0f} samples/sec")
                
                # Performance assertions
                assert perf['wall_time'] < 300  # Should complete within 5 minutes
                assert perf['memory_mb'] < 2000  # Should use less than 2GB
                assert samples_per_sec > 100  # Should process at least 100 samples/sec
        
        # Check scaling behavior
        small_time = results["Small (1K)"]['wall_time']
        medium_time = results["Medium (10K)"]['wall_time']
        
        # Time should scale roughly logarithmically (better than linear)
        scaling_factor = medium_time / small_time
        expected_max_scaling = 15  # 10x data should take at most 15x time
        
        print(f"\nScaling analysis: 10x data took {scaling_factor:.1f}x time")
        assert scaling_factor < expected_max_scaling, f"Poor scaling: {scaling_factor}x > {expected_max_scaling}x"
    
    def test_lof_performance_characteristics(self, detection_service, small_dataset, medium_dataset):
        """Test Local Outlier Factor performance characteristics."""
        print("\n=== LOF Performance Test ===")
        
        # LOF is quadratic, so we test smaller datasets
        datasets = [
            ("Small (1K)", small_dataset),
            ("Medium (5K)", medium_dataset[:5000])  # Limit to 5K for LOF
        ]
        
        for name, dataset in datasets:
            print(f"\nTesting LOF on {name} dataset")
            
            perf = self.measure_performance(
                detection_service.detect_anomalies,
                data=dataset,
                algorithm='lof',
                contamination=0.1,
                n_neighbors=20
            )
            
            if perf['success']:
                print(f"  Wall time: {perf['wall_time']:.2f}s")
                print(f"  Memory: {perf['memory_mb']:.1f} MB")
                
                # LOF should still be reasonably fast on small datasets
                if dataset.shape[0] <= 1000:
                    assert perf['wall_time'] < 30  # Small dataset should complete in 30s
                elif dataset.shape[0] <= 5000:
                    assert perf['wall_time'] < 180  # Medium dataset should complete in 3min
    
    def test_algorithm_comparison(self, detection_service, medium_dataset):
        """Compare performance of different algorithms on the same dataset."""
        print("\n=== Algorithm Performance Comparison ===")
        
        algorithms = [
            ('iforest', {'n_estimators': 100}),
            ('lof', {'n_neighbors': 20}),
            ('ocsvm', {'nu': 0.1}),
            ('elliptic', {})
        ]
        
        results = {}
        dataset = medium_dataset[:5000]  # Limit for LOF
        
        for algo_name, params in algorithms:
            print(f"\nTesting {algo_name}...")
            
            perf = self.measure_performance(
                detection_service.detect_anomalies,
                data=dataset,
                algorithm=algo_name,
                contamination=0.1,
                **params
            )
            
            results[algo_name] = perf
            
            if perf['success']:
                throughput = dataset.shape[0] / perf['wall_time']
                print(f"  Time: {perf['wall_time']:.2f}s")
                print(f"  Memory: {perf['memory_mb']:.1f} MB")
                print(f"  Throughput: {throughput:.0f} samples/sec")
        
        # Find fastest and most memory efficient
        successful_results = {k: v for k, v in results.items() if v['success']}
        
        if successful_results:
            fastest = min(successful_results.items(), key=lambda x: x[1]['wall_time'])
            most_memory_efficient = min(successful_results.items(), key=lambda x: x[1]['memory_mb'])
            
            print(f"\nFastest algorithm: {fastest[0]} ({fastest[1]['wall_time']:.2f}s)")
            print(f"Most memory efficient: {most_memory_efficient[0]} ({most_memory_efficient[1]['memory_mb']:.1f} MB)")
    
    def test_ensemble_performance_overhead(self, ensemble_service, medium_dataset):
        """Test performance overhead of ensemble methods."""
        print("\n=== Ensemble Performance Overhead Test ===")
        
        dataset = medium_dataset[:3000]  # Smaller dataset for ensemble testing
        
        # Test single algorithm baseline
        single_perf = self.measure_performance(
            DetectionService().detect_anomalies,
            data=dataset,
            algorithm='iforest',
            contamination=0.1
        )
        
        # Test ensemble methods
        ensemble_methods = ['majority', 'average', 'maximum']
        algorithms = ['iforest', 'lof']
        
        for method in ensemble_methods:
            print(f"\nTesting {method} ensemble...")
            
            perf = self.measure_performance(
                ensemble_service.detect_with_ensemble,
                data=dataset,
                algorithms=algorithms,
                method=method,
                contamination=0.1
            )
            
            if perf['success'] and single_perf['success']:
                overhead_ratio = perf['wall_time'] / single_perf['wall_time']
                memory_ratio = perf['memory_mb'] / max(single_perf['memory_mb'], 1)
                
                print(f"  Time: {perf['wall_time']:.2f}s (overhead: {overhead_ratio:.1f}x)")
                print(f"  Memory: {perf['memory_mb']:.1f} MB (ratio: {memory_ratio:.1f}x)")
                
                # Ensemble should not be more than 5x slower than single algorithm
                assert overhead_ratio < 5.0, f"Ensemble overhead too high: {overhead_ratio:.1f}x"
    
    def test_concurrent_detection_performance(self, detection_service, medium_dataset):
        """Test performance under concurrent load."""
        print("\n=== Concurrent Detection Performance Test ===")
        
        dataset = medium_dataset[:2000]  # Smaller dataset for concurrent testing
        num_workers = 4
        requests_per_worker = 3
        
        def detection_worker(worker_id: int) -> Dict[str, Any]:
            """Worker function for concurrent detection."""
            results = []
            
            for i in range(requests_per_worker):
                start_time = time.perf_counter()
                
                try:
                    result = detection_service.detect_anomalies(
                        data=dataset,
                        algorithm='iforest',
                        contamination=0.1,
                        random_state=42 + worker_id * 10 + i
                    )
                    
                    end_time = time.perf_counter()
                    
                    results.append({
                        'worker_id': worker_id,
                        'request_id': i,
                        'success': result.success,
                        'duration': end_time - start_time,
                        'anomaly_count': result.anomaly_count
                    })
                    
                except Exception as e:
                    results.append({
                        'worker_id': worker_id,
                        'request_id': i,
                        'success': False,
                        'error': str(e),
                        'duration': 0
                    })
            
            return results
        
        # Run concurrent detection
        start_time = time.perf_counter()
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_worker = {
                executor.submit(detection_worker, worker_id): worker_id
                for worker_id in range(num_workers)
            }
            
            all_results = []
            for future in as_completed(future_to_worker):
                worker_results = future.result()
                all_results.extend(worker_results)
        
        end_time = time.perf_counter()
        
        # Analyze results
        successful_requests = [r for r in all_results if r['success']]
        total_requests = len(all_results)
        success_rate = len(successful_requests) / total_requests
        
        if successful_requests:
            avg_duration = sum(r['duration'] for r in successful_requests) / len(successful_requests)
            max_duration = max(r['duration'] for r in successful_requests)
            min_duration = min(r['duration'] for r in successful_requests)
            
            total_duration = end_time - start_time
            effective_throughput = total_requests / total_duration
            
            print(f"Concurrent performance results:")
            print(f"  Total requests: {total_requests}")
            print(f"  Success rate: {success_rate:.1%}")
            print(f"  Average duration: {avg_duration:.2f}s")
            print(f"  Min/Max duration: {min_duration:.2f}s / {max_duration:.2f}s")
            print(f"  Total time: {total_duration:.2f}s")
            print(f"  Effective throughput: {effective_throughput:.1f} requests/sec")
            
            # Performance assertions
            assert success_rate >= 0.8  # At least 80% success rate
            assert avg_duration < 60  # Average request should complete within 1 minute
            assert effective_throughput > 0.1  # Should handle at least 0.1 requests/sec
    
    def test_memory_usage_patterns(self, detection_service, medium_dataset):
        """Test memory usage patterns and potential leaks."""
        print("\n=== Memory Usage Pattern Test ===")
        
        dataset = medium_dataset[:5000]
        process = psutil.Process()
        
        # Record initial memory
        initial_memory = process.memory_info().rss / 1024 / 1024
        memory_readings = [initial_memory]
        
        print(f"Initial memory: {initial_memory:.1f} MB")
        
        # Run multiple detection cycles
        for cycle in range(5):
            print(f"Cycle {cycle + 1}...")
            
            result = detection_service.detect_anomalies(
                data=dataset,
                algorithm='iforest',
                contamination=0.1,
                random_state=42 + cycle
            )
            
            # Force garbage collection
            gc.collect()
            
            current_memory = process.memory_info().rss / 1024 / 1024
            memory_readings.append(current_memory)
            
            print(f"  Memory after cycle: {current_memory:.1f} MB")
            assert result.success, f"Detection failed in cycle {cycle + 1}"
        
        # Analyze memory pattern
        final_memory = memory_readings[-1]
        memory_growth = final_memory - initial_memory
        max_memory = max(memory_readings)
        
        print(f"\nMemory analysis:")
        print(f"  Initial: {initial_memory:.1f} MB")
        print(f"  Final: {final_memory:.1f} MB")
        print(f"  Growth: {memory_growth:.1f} MB")
        print(f"  Peak: {max_memory:.1f} MB")
        
        # Memory should not grow excessively (potential leak detection)
        assert memory_growth < 500, f"Excessive memory growth: {memory_growth:.1f} MB"
        assert max_memory < initial_memory + 1000, f"Peak memory too high: {max_memory:.1f} MB"
    
    def test_high_dimensional_performance(self, detection_service, high_dimensional_dataset):
        """Test performance with high-dimensional data."""
        print("\n=== High-Dimensional Performance Test ===")
        
        print(f"Testing high-dimensional dataset: {high_dimensional_dataset.shape}")
        
        # Test algorithms that should handle high dimensions well
        algorithms = [
            ('iforest', {'n_estimators': 50}),  # Reduced estimators for speed
            ('ocsvm', {'kernel': 'linear'}),    # Linear kernel for high dimensions
        ]
        
        for algo_name, params in algorithms:
            print(f"\nTesting {algo_name} on high-dimensional data...")
            
            perf = self.measure_performance(
                detection_service.detect_anomalies,
                data=high_dimensional_dataset,
                algorithm=algo_name,
                contamination=0.1,
                **params
            )
            
            if perf['success']:
                samples_per_sec = high_dimensional_dataset.shape[0] / perf['wall_time']
                features_per_sec = high_dimensional_dataset.shape[1] * samples_per_sec
                
                print(f"  Time: {perf['wall_time']:.2f}s")
                print(f"  Memory: {perf['memory_mb']:.1f} MB")
                print(f"  Throughput: {samples_per_sec:.0f} samples/sec")
                print(f"  Feature throughput: {features_per_sec:.0f} features/sec")
                
                # Should handle high-dimensional data reasonably
                assert perf['wall_time'] < 300  # Complete within 5 minutes
                assert samples_per_sec > 10  # At least 10 samples/sec
    
    def test_batch_vs_single_performance(self, detection_service, medium_dataset):
        """Compare batch processing vs single sample performance."""
        print("\n=== Batch vs Single Sample Performance Test ===")
        
        dataset = medium_dataset[:1000]
        
        # Test batch processing
        batch_perf = self.measure_performance(
            detection_service.detect_anomalies,
            data=dataset,
            algorithm='iforest',
            contamination=0.1
        )
        
        # Test single sample processing
        single_times = []
        
        for i in range(min(100, len(dataset))):  # Test first 100 samples
            sample = dataset[i:i+1]  # Single sample as 2D array
            
            single_perf = self.measure_performance(
                detection_service.detect_anomalies,
                data=sample,
                algorithm='iforest',
                contamination=0.1
            )
            
            if single_perf['success']:
                single_times.append(single_perf['wall_time'])
        
        if batch_perf['success'] and single_times:
            avg_single_time = sum(single_times) / len(single_times)
            total_single_time = avg_single_time * len(dataset)
            
            batch_samples_per_sec = len(dataset) / batch_perf['wall_time']
            single_samples_per_sec = 1 / avg_single_time if avg_single_time > 0 else 0
            
            efficiency_ratio = batch_samples_per_sec / single_samples_per_sec if single_samples_per_sec > 0 else float('inf')
            
            print(f"Batch processing:")
            print(f"  Total time: {batch_perf['wall_time']:.2f}s")
            print(f"  Throughput: {batch_samples_per_sec:.0f} samples/sec")
            
            print(f"Single sample processing:")
            print(f"  Average time per sample: {avg_single_time:.4f}s")
            print(f"  Estimated total time: {total_single_time:.2f}s")
            print(f"  Throughput: {single_samples_per_sec:.0f} samples/sec")
            
            print(f"Batch efficiency: {efficiency_ratio:.1f}x faster")
            
            # Batch processing should be significantly more efficient
            assert efficiency_ratio > 5, f"Batch processing not efficient enough: {efficiency_ratio:.1f}x"


if __name__ == "__main__":
    print("Anomaly Detection Algorithm Performance Test Suite")
    print("=" * 55)
    print("Testing performance characteristics of detection algorithms")
    print("Measurements include wall time, CPU time, and memory usage")
    print()
    
    # Quick smoke test
    try:
        import numpy as np
        from anomaly_detection.domain.services.detection_service import DetectionService
        
        service = DetectionService()
        test_data = np.random.randn(100, 5)
        
        start = time.perf_counter()
        result = service.detect_anomalies(test_data, algorithm='iforest', contamination=0.1)
        end = time.perf_counter()
        
        print(f"✓ Quick performance test completed in {end-start:.2f}s")
        print(f"  Detected {result.anomaly_count} anomalies in {result.total_samples} samples")
        print("Ready to run comprehensive performance tests")
        
    except Exception as e:
        print(f"✗ Performance test setup failed: {e}")
        print("Some performance tests may not run properly")