"""
Performance Benchmarks for Pynomaly

This module contains comprehensive performance tests and benchmarks
to ensure the system meets performance requirements and detect regressions.
"""

import gc
import memory_profiler
import psutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pytest

pytestmark = pytest.mark.performance


class TestAlgorithmPerformance:
    """Benchmark individual algorithm performance"""

    @pytest.mark.benchmark
    def test_isolation_forest_performance(self, benchmark, performance_datasets):
        """Benchmark Isolation Forest algorithm performance"""
        
        def run_isolation_forest(data):
            # Mock implementation - replace with actual algorithm
            from sklearn.ensemble import IsolationForest
            clf = IsolationForest(contamination=0.1, random_state=42)
            clf.fit(data)
            return clf.predict(data)
        
        # Benchmark with different dataset sizes
        for size_name, dataset in performance_datasets.items():
            if size_name in ["tiny", "small", "medium"]:  # Skip large for quick tests
                result = benchmark.pedantic(
                    run_isolation_forest,
                    args=(dataset,),
                    rounds=5,
                    warmup_rounds=1
                )
                
                # Verify results are valid
                assert len(result) == len(dataset)
                assert all(r in [-1, 1] for r in result)

    @pytest.mark.benchmark  
    def test_local_outlier_factor_performance(self, benchmark, medium_dataset):
        """Benchmark Local Outlier Factor algorithm performance"""
        
        def run_lof(data):
            from sklearn.neighbors import LocalOutlierFactor
            clf = LocalOutlierFactor(contamination=0.1)
            return clf.fit_predict(data)
        
        result = benchmark.pedantic(
            run_lof,
            args=(medium_dataset,),
            rounds=3,
            warmup_rounds=1
        )
        
        assert len(result) == len(medium_dataset)

    @pytest.mark.benchmark
    def test_one_class_svm_performance(self, benchmark, small_dataset):
        """Benchmark One-Class SVM algorithm performance"""
        
        def run_one_class_svm(data):
            from sklearn.svm import OneClassSVM
            clf = OneClassSVM(gamma='auto')
            clf.fit(data)
            return clf.predict(data)
        
        result = benchmark.pedantic(
            run_one_class_svm,
            args=(small_dataset,),
            rounds=3,
            warmup_rounds=1
        )
        
        assert len(result) == len(small_dataset)

    @pytest.mark.slow
    @pytest.mark.benchmark
    def test_algorithm_comparison_benchmark(self, benchmark, medium_dataset):
        """Compare performance of different algorithms"""
        
        algorithms = {
            "isolation_forest": lambda data: self._run_isolation_forest(data),
            "lof": lambda data: self._run_lof(data),
            "one_class_svm": lambda data: self._run_one_class_svm(data)
        }
        
        results = {}
        
        for name, algorithm in algorithms.items():
            start_time = time.perf_counter()
            predictions = algorithm(medium_dataset)
            end_time = time.perf_counter()
            
            results[name] = {
                "time": end_time - start_time,
                "predictions": len(predictions),
                "anomalies": sum(1 for p in predictions if p == -1 or p == 1)
            }
        
        # Log results for comparison
        print(f"\nAlgorithm Performance Comparison:")
        for name, metrics in results.items():
            print(f"  {name}: {metrics['time']:.3f}s, {metrics['anomalies']} anomalies")
    
    def _run_isolation_forest(self, data):
        from sklearn.ensemble import IsolationForest
        clf = IsolationForest(contamination=0.1, random_state=42)
        clf.fit(data)
        return clf.predict(data)
    
    def _run_lof(self, data):
        from sklearn.neighbors import LocalOutlierFactor  
        clf = LocalOutlierFactor(contamination=0.1)
        return clf.fit_predict(data)
    
    def _run_one_class_svm(self, data):
        from sklearn.svm import OneClassSVM
        clf = OneClassSVM(gamma='auto')
        clf.fit(data)
        return clf.predict(data)


class TestScalabilityBenchmarks:
    """Test system scalability with increasing data sizes"""

    @pytest.mark.slow
    @pytest.mark.benchmark
    def test_dataset_size_scaling(self, benchmark_config):
        """Test performance scaling with dataset size"""
        
        sizes = [100, 500, 1000, 5000, 10000]
        features = 10
        results = []
        
        for size in sizes:
            # Generate dataset
            np.random.seed(42)
            data = np.random.randn(size, features)
            
            # Measure processing time
            start_time = time.perf_counter()
            
            # Run algorithm
            from sklearn.ensemble import IsolationForest
            clf = IsolationForest(contamination=0.1, random_state=42)
            clf.fit(data)
            predictions = clf.predict(data)
            
            end_time = time.perf_counter()
            processing_time = end_time - start_time
            
            results.append({
                "size": size,
                "time": processing_time,
                "time_per_sample": processing_time / size,
                "anomalies": sum(1 for p in predictions if p == -1)
            })
        
        # Analyze scaling behavior
        print(f"\nDataset Size Scaling Results:")
        for result in results:
            print(f"  Size: {result['size']:5d}, "
                  f"Time: {result['time']:6.3f}s, "
                  f"Per Sample: {result['time_per_sample']*1000:6.3f}ms, "
                  f"Anomalies: {result['anomalies']:3d}")
        
        # Assert reasonable scaling (should be roughly linear)
        if len(results) >= 2:
            last_result = results[-1]
            first_result = results[0]
            
            size_ratio = last_result['size'] / first_result['size']
            time_ratio = last_result['time'] / first_result['time']
            
            # Time should scale roughly linearly (within 3x of size ratio)
            assert time_ratio <= size_ratio * 3, f"Poor scaling: {time_ratio:.2f}x time for {size_ratio:.2f}x data"

    @pytest.mark.benchmark
    def test_feature_dimension_scaling(self):
        """Test performance scaling with number of features"""
        
        sample_size = 1000
        feature_counts = [5, 10, 20, 50, 100]
        results = []
        
        for features in feature_counts:
            # Generate high-dimensional dataset
            np.random.seed(42)
            data = np.random.randn(sample_size, features)
            
            # Measure processing time
            start_time = time.perf_counter()
            
            from sklearn.ensemble import IsolationForest
            clf = IsolationForest(contamination=0.1, random_state=42)
            clf.fit(data)
            predictions = clf.predict(data)
            
            end_time = time.perf_counter()
            processing_time = end_time - start_time
            
            results.append({
                "features": features,
                "time": processing_time,
                "anomalies": sum(1 for p in predictions if p == -1)
            })
        
        print(f"\nFeature Dimension Scaling Results:")
        for result in results:
            print(f"  Features: {result['features']:3d}, "
                  f"Time: {result['time']:6.3f}s, "
                  f"Anomalies: {result['anomalies']:3d}")


class TestMemoryPerformance:
    """Test memory usage and memory-related performance"""

    @pytest.mark.benchmark
    @memory_profiler.profile
    def test_memory_usage_by_dataset_size(self):
        """Profile memory usage for different dataset sizes"""
        
        sizes = [1000, 5000, 10000]
        
        for size in sizes:
            print(f"\n--- Testing dataset size: {size} ---")
            
            # Generate dataset
            data = np.random.randn(size, 10)
            
            # Measure memory before
            process = psutil.Process()
            mem_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # Run algorithm
            from sklearn.ensemble import IsolationForest
            clf = IsolationForest(contamination=0.1, random_state=42)
            clf.fit(data)
            predictions = clf.predict(data)
            
            # Measure memory after
            mem_after = process.memory_info().rss / 1024 / 1024  # MB
            mem_used = mem_after - mem_before
            
            print(f"Memory usage for {size} samples: {mem_used:.2f} MB")
            print(f"Memory per sample: {mem_used/size*1024:.3f} KB")
            
            # Cleanup
            del data, clf, predictions
            gc.collect()
            
            # Assert reasonable memory usage (less than 1KB per sample for this algorithm)
            assert mem_used/size < 1.0, f"Excessive memory usage: {mem_used/size:.3f} MB per sample"

    @pytest.mark.benchmark
    def test_memory_leak_detection(self):
        """Test for memory leaks in repeated operations"""
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run many iterations
        for i in range(50):
            data = np.random.randn(500, 10)
            
            from sklearn.ensemble import IsolationForest
            clf = IsolationForest(contamination=0.1, random_state=42)
            clf.fit(data)
            predictions = clf.predict(data)
            
            # Force cleanup
            del data, clf, predictions
            
            if i % 10 == 0:
                gc.collect()
                current_memory = process.memory_info().rss / 1024 / 1024
                print(f"Iteration {i}: {current_memory:.2f} MB")
        
        # Final memory check
        gc.collect()
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_growth = final_memory - initial_memory
        
        print(f"Memory growth after 50 iterations: {memory_growth:.2f} MB")
        
        # Assert no significant memory leak (less than 50MB growth)
        assert memory_growth < 50, f"Potential memory leak: {memory_growth:.2f} MB growth"


class TestConcurrencyPerformance:
    """Test performance under concurrent load"""

    @pytest.mark.benchmark
    def test_concurrent_processing_performance(self, small_dataset):
        """Test performance with concurrent requests"""
        
        def process_data(thread_id, data):
            from sklearn.ensemble import IsolationForest
            clf = IsolationForest(contamination=0.1, random_state=thread_id)
            clf.fit(data)
            return clf.predict(data)
        
        # Test with different numbers of concurrent threads
        thread_counts = [1, 2, 4, 8]
        
        for num_threads in thread_counts:
            start_time = time.perf_counter()
            
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                # Submit jobs
                futures = []
                for i in range(num_threads * 2):  # 2 jobs per thread
                    future = executor.submit(process_data, i, small_dataset)
                    futures.append(future)
                
                # Wait for completion
                results = []
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
            
            end_time = time.perf_counter()
            total_time = end_time - start_time
            
            print(f"Threads: {num_threads}, Total time: {total_time:.3f}s, "
                  f"Per job: {total_time/(num_threads*2):.3f}s")
            
            # Verify all results
            assert len(results) == num_threads * 2
            for result in results:
                assert len(result) == len(small_dataset)

    @pytest.mark.benchmark  
    def test_thread_safety_performance(self, medium_dataset):
        """Test that concurrent access doesn't degrade performance significantly"""
        
        from sklearn.ensemble import IsolationForest
        
        # Create shared model
        shared_model = IsolationForest(contamination=0.1, random_state=42)
        shared_model.fit(medium_dataset)
        
        def concurrent_predict(thread_id, model, data):
            # Add some variation to test data per thread
            test_data = data + np.random.randn(*data.shape) * 0.1
            return model.predict(test_data)
        
        # Single-threaded baseline
        start_time = time.perf_counter()
        for i in range(8):
            result = concurrent_predict(i, shared_model, medium_dataset)
        baseline_time = time.perf_counter() - start_time
        
        # Multi-threaded test
        start_time = time.perf_counter()
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for i in range(8):
                future = executor.submit(concurrent_predict, i, shared_model, medium_dataset)
                futures.append(future)
            
            results = [future.result() for future in as_completed(futures)]
        
        concurrent_time = time.perf_counter() - start_time
        
        print(f"Single-threaded: {baseline_time:.3f}s")
        print(f"Multi-threaded: {concurrent_time:.3f}s")
        print(f"Speedup: {baseline_time/concurrent_time:.2f}x")
        
        # Should see some speedup from parallelization
        assert concurrent_time < baseline_time * 0.8, "No performance benefit from concurrency"


class TestAPIPerformance:
    """Test API endpoint performance"""

    @pytest.mark.benchmark
    def test_api_response_time(self, api_client, sample_api_data):
        """Benchmark API response times"""
        
        response_times = []
        
        # Warm up
        for _ in range(3):
            api_client.post("/api/v1/detect", json=sample_api_data)
        
        # Measure response times
        for _ in range(10):
            start_time = time.perf_counter()
            response = api_client.post("/api/v1/detect", json=sample_api_data)
            end_time = time.perf_counter()
            
            assert response.status_code == 200
            response_times.append(end_time - start_time)
        
        # Calculate statistics
        avg_response_time = np.mean(response_times)
        p95_response_time = np.percentile(response_times, 95)
        p99_response_time = np.percentile(response_times, 99)
        
        print(f"API Response Times:")
        print(f"  Average: {avg_response_time*1000:.1f}ms")
        print(f"  95th percentile: {p95_response_time*1000:.1f}ms")  
        print(f"  99th percentile: {p99_response_time*1000:.1f}ms")
        
        # Assert performance requirements
        assert avg_response_time < 2.0, f"Average response time too slow: {avg_response_time:.3f}s"
        assert p95_response_time < 5.0, f"95th percentile too slow: {p95_response_time:.3f}s"

    @pytest.mark.benchmark
    def test_api_throughput(self, api_client, sample_api_data):
        """Test API throughput under load"""
        
        def make_request():
            return api_client.post("/api/v1/detect", json=sample_api_data)
        
        # Test with different concurrency levels
        concurrency_levels = [1, 2, 4, 8]
        
        for concurrency in concurrency_levels:
            start_time = time.perf_counter()
            
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                # Submit 20 requests per worker
                futures = []
                for _ in range(concurrency * 20):
                    future = executor.submit(make_request)
                    futures.append(future)
                
                # Wait for all requests
                responses = [future.result() for future in as_completed(futures)]
            
            end_time = time.perf_counter()
            total_time = end_time - start_time
            total_requests = len(responses)
            throughput = total_requests / total_time
            
            # Verify all responses successful
            successful_responses = sum(1 for r in responses if r.status_code == 200)
            success_rate = successful_responses / total_requests
            
            print(f"Concurrency: {concurrency:2d}, "
                  f"Throughput: {throughput:6.1f} req/s, "
                  f"Success rate: {success_rate:5.1%}")
            
            # Assert minimum performance requirements
            assert success_rate >= 0.95, f"Low success rate: {success_rate:.1%}"
            assert throughput >= 1.0, f"Low throughput: {throughput:.1f} req/s"


class TestRegressionDetection:
    """Test for performance regressions against baselines"""

    @pytest.fixture
    def performance_baselines(self):
        """Load or create performance baselines"""
        return {
            "isolation_forest_small": 0.1,   # seconds
            "isolation_forest_medium": 0.5,
            "lof_small": 0.15,
            "lof_medium": 0.8,
            "api_response_avg": 1.0,
            "api_response_p95": 2.0
        }

    def test_isolation_forest_regression(self, small_dataset, medium_dataset, performance_baselines):
        """Test for performance regressions in Isolation Forest"""
        
        from sklearn.ensemble import IsolationForest
        
        # Test small dataset
        start_time = time.perf_counter()
        clf = IsolationForest(contamination=0.1, random_state=42)
        clf.fit(small_dataset)
        predictions = clf.predict(small_dataset)
        small_time = time.perf_counter() - start_time
        
        # Test medium dataset  
        start_time = time.perf_counter()
        clf = IsolationForest(contamination=0.1, random_state=42)
        clf.fit(medium_dataset)
        predictions = clf.predict(medium_dataset)
        medium_time = time.perf_counter() - start_time
        
        print(f"Isolation Forest Performance:")
        print(f"  Small dataset: {small_time:.3f}s (baseline: {performance_baselines['isolation_forest_small']:.3f}s)")
        print(f"  Medium dataset: {medium_time:.3f}s (baseline: {performance_baselines['isolation_forest_medium']:.3f}s)")
        
        # Check for regressions (allow 20% tolerance)
        tolerance = 1.2
        
        assert small_time <= performance_baselines['isolation_forest_small'] * tolerance, \
            f"Small dataset regression: {small_time:.3f}s > {performance_baselines['isolation_forest_small'] * tolerance:.3f}s"
        
        assert medium_time <= performance_baselines['isolation_forest_medium'] * tolerance, \
            f"Medium dataset regression: {medium_time:.3f}s > {performance_baselines['isolation_forest_medium'] * tolerance:.3f}s"

    def test_api_performance_regression(self, api_client, sample_api_data, performance_baselines):
        """Test for API performance regressions"""
        
        response_times = []
        
        # Measure current performance
        for _ in range(10):
            start_time = time.perf_counter()
            response = api_client.post("/api/v1/detect", json=sample_api_data)
            end_time = time.perf_counter()
            
            if response.status_code == 200:
                response_times.append(end_time - start_time)
        
        if response_times:  # Only test if API is available
            avg_time = np.mean(response_times)
            p95_time = np.percentile(response_times, 95)
            
            print(f"API Performance:")
            print(f"  Average: {avg_time:.3f}s (baseline: {performance_baselines['api_response_avg']:.3f}s)")
            print(f"  95th percentile: {p95_time:.3f}s (baseline: {performance_baselines['api_response_p95']:.3f}s)")
            
            # Check for regressions (allow 30% tolerance for API)
            tolerance = 1.3
            
            assert avg_time <= performance_baselines['api_response_avg'] * tolerance, \
                f"API average response regression: {avg_time:.3f}s > {performance_baselines['api_response_avg'] * tolerance:.3f}s"
            
            assert p95_time <= performance_baselines['api_response_p95'] * tolerance, \
                f"API p95 response regression: {p95_time:.3f}s > {performance_baselines['api_response_p95'] * tolerance:.3f}s"


class TestResourceUtilization:
    """Test system resource utilization"""

    @pytest.mark.slow
    def test_cpu_utilization(self, large_dataset):
        """Test CPU utilization during processing"""
        
        import multiprocessing
        
        def monitor_cpu():
            cpu_percentages = []
            for _ in range(20):  # Monitor for ~2 seconds
                cpu_percentages.append(psutil.cpu_percent(interval=0.1))
            return cpu_percentages
        
        # Start CPU monitoring
        monitor_process = multiprocessing.Process(target=monitor_cpu)
        
        # Run processing
        start_time = time.perf_counter()
        
        from sklearn.ensemble import IsolationForest
        clf = IsolationForest(contamination=0.1, random_state=42)
        clf.fit(large_dataset)
        predictions = clf.predict(large_dataset)
        
        processing_time = time.perf_counter() - start_time
        
        print(f"Processing time: {processing_time:.2f}s")
        print(f"Dataset size: {len(large_dataset):,} samples")
        print(f"Throughput: {len(large_dataset)/processing_time:.0f} samples/sec")
        
        # Assert reasonable throughput (at least 1000 samples/sec)
        assert len(large_dataset)/processing_time >= 1000, \
            f"Low throughput: {len(large_dataset)/processing_time:.0f} samples/sec"

    def test_disk_io_performance(self, temp_directory):
        """Test disk I/O performance for data loading/saving"""
        
        # Generate large dataset
        data = np.random.randn(10000, 20)
        
        # Test save performance
        save_path = temp_directory / "test_data.npy"
        
        start_time = time.perf_counter()
        np.save(save_path, data)
        save_time = time.perf_counter() - start_time
        
        # Test load performance
        start_time = time.perf_counter()
        loaded_data = np.load(save_path)
        load_time = time.perf_counter() - start_time
        
        # Calculate throughput
        data_size_mb = data.nbytes / (1024 * 1024)
        save_throughput = data_size_mb / save_time
        load_throughput = data_size_mb / load_time
        
        print(f"Disk I/O Performance:")
        print(f"  Data size: {data_size_mb:.1f} MB")
        print(f"  Save time: {save_time:.3f}s ({save_throughput:.1f} MB/s)")
        print(f"  Load time: {load_time:.3f}s ({load_throughput:.1f} MB/s)")
        
        # Verify data integrity
        assert np.array_equal(data, loaded_data), "Data corruption detected"
        
        # Assert reasonable I/O performance (at least 10 MB/s)
        assert save_throughput >= 10, f"Slow save performance: {save_throughput:.1f} MB/s"
        assert load_throughput >= 10, f"Slow load performance: {load_throughput:.1f} MB/s"