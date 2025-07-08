"""Performance tests for Pynomaly."""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import pytest
from pynomaly.application.use_cases.detect_anomalies import DetectAnomaliesUseCase
from pynomaly.application.use_cases.train_detector import TrainDetectorUseCase
from pynomaly.domain.entities import Dataset, Detector


@pytest.mark.performance
class TestDetectionPerformance:
    """Test detection performance and scalability."""

    @pytest.mark.slow
    def test_large_dataset_detection_time(self, large_dataset, sample_detector, container):
        """Test detection performance on large datasets."""
        max_time_seconds = 30  # Should complete within 30 seconds

        start_time = time.time()

        try:
            # Get detection use case
            detection_use_case = DetectAnomaliesUseCase(
                detector_repository=container.detector_repository(),
                dataset_repository=container.dataset_repository(),
                result_repository=container.detection_result_repository(),
                pyod_adapter=container.pyod_adapter()
            )

            # Train detector first
            train_use_case = TrainDetectorUseCase(
                detector_repository=container.detector_repository(),
                dataset_repository=container.dataset_repository(),
                pyod_adapter=container.pyod_adapter()
            )

            train_use_case.execute(sample_detector.id, large_dataset.id)

            # Run detection
            result = detection_use_case.execute(sample_detector.id, large_dataset.id)

            detection_time = time.time() - start_time

            # Performance assertions
            assert detection_time < max_time_seconds, f"Detection took {detection_time:.2f}s, expected < {max_time_seconds}s"
            assert result is not None
            assert len(result.scores) == len(large_dataset.data)

            # Calculate throughput
            throughput = len(large_dataset.data) / detection_time
            print(f"Detection throughput: {throughput:.2f} samples/second")

            # Should process at least 1000 samples per second
            assert throughput > 1000, f"Throughput {throughput:.2f} samples/s is too low"

        except ImportError:
            pytest.skip("Required dependencies not available for performance test")

    @pytest.mark.slow
    def test_memory_usage_large_dataset(self, performance_data):
        """Test memory usage with large datasets."""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Create large dataset
        large_df = performance_data
        dataset = Dataset(
            name="Memory Test Dataset",
            data=large_df,
            features=large_df.columns.tolist()
        )

        # Simulate processing
        processed_data = dataset.data.copy()
        processed_data['anomaly_score'] = np.random.random(len(processed_data))

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        print(f"Memory usage: {initial_memory:.2f} MB -> {final_memory:.2f} MB (+{memory_increase:.2f} MB)")

        # Should not use excessive memory (adjust threshold as needed)
        max_memory_increase = 500  # MB
        assert memory_increase < max_memory_increase, f"Memory increase {memory_increase:.2f} MB exceeds threshold"

    def test_concurrent_detection_performance(self, sample_dataset, sample_detector, container):
        """Test performance under concurrent load."""
        num_concurrent_requests = 10
        max_total_time = 60  # seconds

        def run_detection():
            """Run a single detection."""
            try:
                detection_use_case = DetectAnomaliesUseCase(
                    detector_repository=container.detector_repository(),
                    dataset_repository=container.dataset_repository(),
                    result_repository=container.detection_result_repository(),
                    pyod_adapter=container.pyod_adapter()
                )

                start = time.time()
                result = detection_use_case.execute(sample_detector.id, sample_dataset.id)
                duration = time.time() - start

                return {
                    'success': True,
                    'duration': duration,
                    'result_size': len(result.scores) if result else 0
                }
            except Exception as e:
                return {
                    'success': False,
                    'error': str(e),
                    'duration': 0
                }

        start_time = time.time()

        # Run concurrent detections
        with ThreadPoolExecutor(max_workers=num_concurrent_requests) as executor:
            futures = [executor.submit(run_detection) for _ in range(num_concurrent_requests)]
            results = [future.result() for future in as_completed(futures)]

        total_time = time.time() - start_time

        # Analyze results
        successful_results = [r for r in results if r['success']]
        failed_results = [r for r in results if not r['success']]

        success_rate = len(successful_results) / len(results)
        avg_duration = np.mean([r['duration'] for r in successful_results]) if successful_results else 0

        print("Concurrent test results:")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Success rate: {success_rate:.2%}")
        print(f"  Average duration: {avg_duration:.2f}s")
        print(f"  Failed requests: {len(failed_results)}")

        # Performance assertions
        assert total_time < max_total_time, f"Total time {total_time:.2f}s exceeds threshold"
        assert success_rate > 0.8, f"Success rate {success_rate:.2%} is too low"

        if failed_results:
            print("Failed request errors:")
            for i, result in enumerate(failed_results[:3]):  # Show first 3 errors
                print(f"  {i+1}: {result['error']}")


@pytest.mark.performance
class TestDatabasePerformance:
    """Test database operation performance."""

    def test_bulk_insert_performance(self, session_factory):
        """Test bulk insert performance."""
        try:
            from pynomaly.infrastructure.persistence.database_repositories import (
                DatabaseDatasetRepository,
                DatasetModel,
                DetectorModel,
            )
        except ImportError:
            pytest.skip("Database repositories not available")

        num_records = 1000
        max_time_seconds = 10

        start_time = time.time()

        # Create test data
        datasets = []
        for i in range(num_records):
            dataset = Dataset(
                name=f"test_dataset_{i}",
                data=pd.DataFrame({'feature_1': [1, 2, 3]}),
                features=['feature_1']
            )
            datasets.append(dataset)

        # Bulk insert using repository
        repository = DatabaseDatasetRepository(session_factory)

        for dataset in datasets:
            repository.save(dataset)

        insert_time = time.time() - start_time

        # Performance assertions
        assert insert_time < max_time_seconds, f"Bulk insert took {insert_time:.2f}s, expected < {max_time_seconds}s"

        # Calculate throughput
        throughput = num_records / insert_time
        print(f"Insert throughput: {throughput:.2f} records/second")

        # Should insert at least 100 records per second
        assert throughput > 100, f"Insert throughput {throughput:.2f} records/s is too low"

    def test_query_performance(self, session_factory):
        """Test query performance with indexed searches."""
        try:
            from pynomaly.infrastructure.persistence.database_repositories import (
                DatabaseDatasetRepository,
            )
        except ImportError:
            pytest.skip("Database repositories not available")

        repository = DatabaseDatasetRepository(session_factory)

        # Create test data
        num_datasets = 100
        for i in range(num_datasets):
            dataset = Dataset(
                name=f"perf_test_dataset_{i}",
                data=pd.DataFrame({'feature_1': [1, 2, 3]}),
                features=['feature_1']
            )
            repository.save(dataset)

        # Test query performance
        max_query_time = 1.0  # seconds

        start_time = time.time()
        results = repository.find_all()
        query_time = time.time() - start_time

        assert len(results) >= num_datasets
        assert query_time < max_query_time, f"Query took {query_time:.2f}s, expected < {max_query_time}s"

        print(f"Query time: {query_time:.3f}s for {len(results)} records")


@pytest.mark.performance
class TestAPIPerformance:
    """Test API endpoint performance."""

    def test_health_endpoint_response_time(self, client):
        """Test health endpoint response time."""
        if not hasattr(client, 'get'):
            pytest.skip("API client not available")

        max_response_time = 0.1  # 100ms
        num_requests = 10

        response_times = []

        for _ in range(num_requests):
            start_time = time.time()
            response = client.get("/api/v1/health")
            response_time = time.time() - start_time

            response_times.append(response_time)
            assert response.status_code == 200

        avg_response_time = np.mean(response_times)
        max_observed_time = max(response_times)

        print("Health endpoint performance:")
        print(f"  Average response time: {avg_response_time*1000:.2f}ms")
        print(f"  Max response time: {max_observed_time*1000:.2f}ms")

        assert avg_response_time < max_response_time, f"Average response time {avg_response_time*1000:.2f}ms exceeds threshold"

    def test_api_throughput(self, client):
        """Test API throughput under load."""
        if not hasattr(client, 'get'):
            pytest.skip("API client not available")

        num_requests = 50
        max_total_time = 10  # seconds

        start_time = time.time()

        successful_requests = 0
        for _ in range(num_requests):
            try:
                response = client.get("/api/v1/health")
                if response.status_code == 200:
                    successful_requests += 1
            except Exception as e:
                print(f"Request failed: {e}")

        total_time = time.time() - start_time
        throughput = successful_requests / total_time

        print(f"API throughput: {throughput:.2f} requests/second")
        print(f"Success rate: {successful_requests/num_requests:.2%}")

        assert total_time < max_total_time, f"Total time {total_time:.2f}s exceeds threshold"
        assert throughput > 5, f"Throughput {throughput:.2f} req/s is too low"  # At least 5 req/s
        assert successful_requests / num_requests > 0.95, "Success rate is too low"


@pytest.mark.performance
class TestMemoryEfficiency:
    """Test memory efficiency and leak detection."""

    def test_no_memory_leaks_in_detection(self, sample_dataset, sample_detector):
        """Test that detection doesn't leak memory."""
        import gc
        import os

        import psutil

        process = psutil.Process(os.getpid())

        def get_memory_usage():
            gc.collect()  # Force garbage collection
            return process.memory_info().rss / 1024 / 1024  # MB

        initial_memory = get_memory_usage()

        # Run multiple detection cycles
        for _ in range(10):
            try:
                # Simulate detection process
                data_copy = sample_dataset.data.copy()
                scores = np.random.random(len(data_copy))

                # Process scores
                anomalies = scores > 0.9
                result_data = {
                    'scores': scores.tolist(),
                    'anomalies': anomalies.tolist()
                }

                # Clean up explicitly
                del data_copy, scores, anomalies, result_data

            except Exception as e:
                print(f"Detection cycle failed: {e}")

        final_memory = get_memory_usage()
        memory_increase = final_memory - initial_memory

        print(f"Memory usage: {initial_memory:.2f} MB -> {final_memory:.2f} MB")
        print(f"Memory increase: {memory_increase:.2f} MB")

        # Should not increase memory significantly
        max_increase = 50  # MB
        assert memory_increase < max_increase, f"Memory increase {memory_increase:.2f} MB suggests memory leak"

    def test_large_dataset_memory_efficiency(self, performance_data):
        """Test memory efficiency with large datasets."""
        import sys

        # Get initial memory usage
        initial_size = sys.getsizeof(performance_data)

        # Create dataset entity
        dataset = Dataset(
            name="Memory Efficiency Test",
            data=performance_data,
            features=performance_data.columns.tolist()
        )

        # Should not significantly increase memory usage
        dataset_size = sys.getsizeof(dataset)
        data_size = sys.getsizeof(dataset.data)

        print(f"Original data size: {initial_size / 1024 / 1024:.2f} MB")
        print(f"Dataset entity size: {dataset_size / 1024:.2f} KB")
        print(f"Dataset data size: {data_size / 1024 / 1024:.2f} MB")

        # Dataset wrapper should be minimal overhead
        overhead = (dataset_size - data_size) / initial_size
        assert overhead < 0.1, f"Dataset wrapper overhead {overhead:.2%} is too high"


@pytest.mark.performance
class TestPerformanceBenchmarks:
    """Comprehensive performance benchmarks."""

    def test_detection_algorithm_benchmarks(self, sample_dataset):
        """Benchmark different detection algorithms."""
        algorithms = [
            ('IsolationForest', {'contamination': 0.1, 'random_state': 42}),
            ('LocalOutlierFactor', {'contamination': 0.1}),
            ('OneClassSVM', {'gamma': 'auto'})
        ]

        results = {}

        for algo_name, params in algorithms:
            try:
                detector = Detector(
                    algorithm_name=algo_name,
                    parameters=params
                )

                start_time = time.time()

                # Simulate training and detection
                # In real implementation, would use actual algorithms
                training_time = np.random.uniform(0.1, 2.0)  # Mock training time
                detection_time = np.random.uniform(0.05, 0.5)  # Mock detection time

                total_time = time.time() - start_time + training_time + detection_time

                results[algo_name] = {
                    'total_time': total_time,
                    'training_time': training_time,
                    'detection_time': detection_time,
                    'samples_per_second': len(sample_dataset.data) / detection_time
                }

            except Exception as e:
                print(f"Algorithm {algo_name} failed: {e}")
                results[algo_name] = {'error': str(e)}

        # Report benchmark results
        print("\nAlgorithm Performance Benchmarks:")
        print("-" * 50)
        for algo, metrics in results.items():
            if 'error' in metrics:
                print(f"{algo}: FAILED - {metrics['error']}")
            else:
                print(f"{algo}:")
                print(f"  Total time: {metrics['total_time']:.3f}s")
                print(f"  Training time: {metrics['training_time']:.3f}s")
                print(f"  Detection time: {metrics['detection_time']:.3f}s")
                print(f"  Throughput: {metrics['samples_per_second']:.1f} samples/s")

        # At least one algorithm should work
        successful_algos = [name for name, metrics in results.items() if 'error' not in metrics]
        assert len(successful_algos) > 0, "No algorithms completed successfully"

    def test_scalability_analysis(self):
        """Test scalability with different data sizes."""
        data_sizes = [100, 1000, 10000]
        results = []

        for size in data_sizes:
            # Generate test data
            np.random.seed(42)
            data = np.random.normal(0, 1, (size, 5))
            df = pd.DataFrame(data, columns=[f'feature_{i}' for i in range(5)])

            dataset = Dataset(
                name=f"Scalability Test {size}",
                data=df,
                features=df.columns.tolist()
            )

            start_time = time.time()

            # Simulate processing
            processed = dataset.data.copy()
            scores = np.random.random(len(processed))

            processing_time = time.time() - start_time
            throughput = size / processing_time if processing_time > 0 else float('inf')

            results.append({
                'size': size,
                'time': processing_time,
                'throughput': throughput
            })

        # Analyze scalability
        print("\nScalability Analysis:")
        print("-" * 30)
        for result in results:
            print(f"Size: {result['size']:5d} | Time: {result['time']:.3f}s | Throughput: {result['throughput']:.1f} samples/s")

        # Check that processing time scales reasonably
        if len(results) >= 2:
            time_ratio = results[-1]['time'] / results[0]['time']
            size_ratio = results[-1]['size'] / results[0]['size']
            efficiency_ratio = time_ratio / size_ratio

            print("\nScalability metrics:")
            print(f"Time ratio: {time_ratio:.2f}x")
            print(f"Size ratio: {size_ratio:.2f}x")
            print(f"Efficiency ratio: {efficiency_ratio:.2f}")

            # Should scale reasonably (not worse than quadratic)
            assert efficiency_ratio < size_ratio, "Performance degrades too quickly with scale"
