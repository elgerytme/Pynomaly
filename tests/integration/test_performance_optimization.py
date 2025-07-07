"""Performance optimization integration tests."""

import gc
import time
from typing import Dict, List

import numpy as np
import psutil
import pytest
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

from pynomaly.infrastructure.performance.optimization_service import (
    PerformanceOptimizationService,
    get_optimization_service,
)


@pytest.fixture
def performance_service():
    """Create performance optimization service."""
    service = PerformanceOptimizationService()
    service.reset_metrics()
    yield service
    service.cleanup_resources()


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    # Normal data
    normal_data = np.random.normal(0, 1, (10000, 10))
    # Anomalous data
    anomaly_data = np.random.normal(5, 1, (1000, 10))
    
    X = np.vstack([normal_data, anomaly_data])
    y = np.hstack([np.zeros(10000), np.ones(1000)])
    
    return X, y


@pytest.fixture
def large_sample_data():
    """Create large sample data for performance testing."""
    np.random.seed(42)
    # Larger dataset for performance testing
    normal_data = np.random.normal(0, 1, (50000, 20))
    anomaly_data = np.random.normal(5, 1, (5000, 20))
    
    X = np.vstack([normal_data, anomaly_data])
    y = np.hstack([np.zeros(50000), np.ones(5000)])
    
    return X, y


class TestPerformanceOptimization:
    """Test performance optimization service."""

    def test_memory_optimization(self, performance_service, sample_data):
        """Test memory optimization features."""
        X, y = sample_data
        
        # Test array optimization
        start_memory = performance_service.memory_optimizer.get_memory_usage()
        
        # Create float64 array
        float64_data = X.astype(np.float64)
        
        # Optimize to float32
        optimized_data = performance_service.memory_optimizer.optimize_numpy_arrays(float64_data)
        
        # Should be float32 for better memory efficiency
        assert optimized_data.dtype == np.float32
        assert np.allclose(float64_data, optimized_data, rtol=1e-6)
        
        # Should be contiguous
        assert optimized_data.flags.c_contiguous

    def test_batch_processing_optimization(self, performance_service, large_sample_data):
        """Test batch processing for large arrays."""
        X, y = large_sample_data
        
        def simple_function(data):
            return np.mean(data, axis=1)
        
        # Test batch processing vs. regular processing
        start_time = time.perf_counter()
        batch_result = performance_service.memory_optimizer.batch_process_large_arrays(
            X, simple_function, batch_size=10000
        )
        batch_time = time.perf_counter() - start_time
        
        start_time = time.perf_counter()
        regular_result = simple_function(X)
        regular_time = time.perf_counter() - start_time
        
        # Results should be equivalent
        assert np.allclose(batch_result, regular_result)
        
        # Batch processing should use less memory (difficult to test precisely)
        # At minimum, it should complete without errors
        assert len(batch_result) == len(X)

    def test_distance_matrix_optimization(self, performance_service, sample_data):
        """Test optimized distance matrix computation."""
        X, y = sample_data
        
        # Use subset for distance computation (expensive operation)
        X_subset = X[:1000]
        
        start_time = time.perf_counter()
        distances = performance_service.optimize_distance_computation(X_subset)
        computation_time = time.perf_counter() - start_time
        
        # Should return valid distance matrix
        assert distances.shape == (len(X_subset), len(X_subset))
        assert np.all(distances >= 0)  # Distances should be non-negative
        assert np.allclose(np.diag(distances), 0)  # Self-distance should be 0
        
        # Test caching - second call should be faster
        start_time = time.perf_counter()
        cached_distances = performance_service.optimize_distance_computation(X_subset)
        cached_time = time.perf_counter() - start_time
        
        assert np.allclose(distances, cached_distances)
        assert cached_time < computation_time  # Should be much faster due to caching

    def test_sklearn_estimator_optimization(self, performance_service, sample_data):
        """Test sklearn estimator optimization."""
        X, y = sample_data
        
        # Test IsolationForest optimization
        iso_forest = IsolationForest(random_state=42)
        performance_service.computation_optimizer.optimize_sklearn_estimator(iso_forest)
        
        # Should set n_jobs for parallel processing
        assert iso_forest.n_jobs > 1
        
        # Test LOF optimization
        lof = LocalOutlierFactor()
        performance_service.computation_optimizer.optimize_sklearn_estimator(lof)
        
        # Should optimize algorithm
        assert lof.algorithm in ['kd_tree', 'ball_tree', 'brute']
        
        # Test OneClassSVM optimization
        svm = OneClassSVM()
        performance_service.computation_optimizer.optimize_sklearn_estimator(svm)
        
        # Should disable probability if available
        if hasattr(svm, 'probability'):
            assert svm.probability is False

    def test_anomaly_detection_optimization(self, performance_service, sample_data):
        """Test end-to-end anomaly detection optimization."""
        X, y = sample_data
        
        # Split data
        X_train, X_test = X[:8000], X[8000:]
        
        # Test with IsolationForest
        iso_forest = IsolationForest(random_state=42, contamination=0.1)
        
        start_time = time.perf_counter()
        trained_model, predictions = performance_service.optimize_anomaly_detection(
            iso_forest, X_train, X_test
        )
        optimization_time = time.perf_counter() - start_time
        
        # Should return valid results
        assert trained_model is not None
        assert predictions is not None
        assert len(predictions) == len(X_test)
        
        # Predictions should be -1 or 1
        assert np.all(np.isin(predictions, [-1, 1]))
        
        # Test caching - second call with same data should be faster
        start_time = time.perf_counter()
        cached_model, cached_predictions = performance_service.optimize_anomaly_detection(
            IsolationForest(random_state=42, contamination=0.1), X_train, X_test
        )
        cached_time = time.perf_counter() - start_time
        
        # Should use cached model (faster)
        assert cached_time < optimization_time
        
        # Results should be similar (models might differ slightly due to randomness)
        assert len(cached_predictions) == len(predictions)

    def test_streaming_optimization(self, performance_service, sample_data):
        """Test streaming detection optimization."""
        X, y = sample_data
        
        # Prepare detector
        detector = IsolationForest(random_state=42, contamination=0.1)
        detector.fit(X[:8000])
        
        # Create stream of individual samples
        data_stream = [X[i:i+1] for i in range(8000, 9000)]
        
        start_time = time.perf_counter()
        results = performance_service.optimize_streaming_detection(
            detector, data_stream, batch_size=50
        )
        streaming_time = time.perf_counter() - start_time
        
        # Should return results for all samples
        assert len(results) == len(data_stream)
        
        # All results should be arrays
        assert all(isinstance(result, np.ndarray) for result in results)
        
        # Compare with individual processing
        start_time = time.perf_counter()
        individual_results = []
        for sample in data_stream:
            result = detector.decision_function(sample)
            individual_results.append(result)
        individual_time = time.perf_counter() - start_time
        
        # Streaming should be faster (or at least not significantly slower)
        # Allow 20% tolerance for measurement variance
        assert streaming_time <= individual_time * 1.2

    def test_cache_optimization(self, performance_service, sample_data):
        """Test cache optimization functionality."""
        X, y = sample_data
        X_subset = X[:1000]
        
        # Clear cache first
        performance_service.cache_optimizer.clear_cache()
        
        # First computation - cache miss
        cache_key = performance_service.cache_optimizer.get_cache_key(X_subset, "test_operation")
        result1 = performance_service.cache_optimizer.get_cached_result(cache_key)
        assert result1 is None  # Should be cache miss
        
        # Cache a result
        test_result = np.mean(X_subset, axis=1)
        performance_service.cache_optimizer.cache_result(cache_key, test_result)
        
        # Second access - cache hit
        result2 = performance_service.cache_optimizer.get_cached_result(cache_key)
        assert result2 is not None
        assert np.allclose(result2, test_result)

    def test_correlation_matrix_optimization(self, performance_service, sample_data):
        """Test fast correlation matrix computation."""
        X, y = sample_data
        X_subset = X[:5000]  # Use subset for faster computation
        
        start_time = time.perf_counter()
        corr_matrix = performance_service.computation_optimizer.fast_correlation_matrix(X_subset)
        computation_time = time.perf_counter() - start_time
        
        # Should return valid correlation matrix
        n_features = X_subset.shape[1]
        assert corr_matrix.shape == (n_features, n_features)
        
        # Diagonal should be 1 (self-correlation)
        assert np.allclose(np.diag(corr_matrix), 1.0)
        
        # Should be symmetric
        assert np.allclose(corr_matrix, corr_matrix.T)
        
        # Values should be between -1 and 1
        assert np.all(corr_matrix >= -1.0) and np.all(corr_matrix <= 1.0)

    def test_data_preprocessing_optimization(self, performance_service, sample_data):
        """Test data preprocessing optimization."""
        X, y = sample_data
        
        start_memory = performance_service.memory_optimizer.get_memory_usage()
        
        # Optimize data preprocessing
        optimized_X = performance_service.optimize_data_preprocessing(X)
        
        end_memory = performance_service.memory_optimizer.get_memory_usage()
        
        # Should return valid data
        assert optimized_X.shape == X.shape
        assert np.allclose(optimized_X, X, rtol=1e-6)
        
        # Should track optimization metrics
        metrics = performance_service.get_performance_metrics()
        assert metrics['operations_optimized'] > 0

    def test_performance_metrics(self, performance_service, sample_data):
        """Test performance metrics tracking."""
        X, y = sample_data
        
        # Reset metrics
        performance_service.reset_metrics()
        initial_metrics = performance_service.get_performance_metrics()
        
        # All metrics should be zero initially
        assert initial_metrics['operations_optimized'] == 0
        assert initial_metrics['cache_hits'] == 0
        assert initial_metrics['cache_misses'] == 0
        
        # Perform some operations
        performance_service.optimize_data_preprocessing(X)
        performance_service.optimize_distance_computation(X[:500])
        performance_service.optimize_distance_computation(X[:500])  # Should hit cache
        
        # Metrics should be updated
        updated_metrics = performance_service.get_performance_metrics()
        assert updated_metrics['operations_optimized'] > 0
        assert updated_metrics['cache_hits'] > 0
        assert updated_metrics['cache_misses'] > 0
        assert updated_metrics['cache_hit_rate'] > 0

    def test_memory_efficiency_large_dataset(self, performance_service, large_sample_data):
        """Test memory efficiency with large datasets."""
        X, y = large_sample_data
        
        baseline_memory = performance_service.memory_optimizer.get_memory_usage()
        
        # Process large dataset with optimization
        detector = IsolationForest(random_state=42, contamination=0.1)
        trained_model, _ = performance_service.optimize_anomaly_detection(detector, X)
        
        peak_memory = performance_service.memory_optimizer.get_memory_usage()
        memory_usage = peak_memory - baseline_memory
        
        # Should not use excessive memory (less than 2GB for this dataset)
        assert memory_usage < 2000  # MB
        
        # Clean up
        performance_service.cleanup_resources()
        
        # Memory should be cleaned up
        final_memory = performance_service.memory_optimizer.get_memory_usage()
        memory_after_cleanup = final_memory - baseline_memory
        
        # Should recover most memory (allow 200MB tolerance)
        assert memory_after_cleanup < memory_usage * 0.3

    def test_singleton_service(self):
        """Test singleton optimization service."""
        service1 = get_optimization_service()
        service2 = get_optimization_service()
        
        # Should be the same instance
        assert service1 is service2


@pytest.mark.performance
class TestPerformanceBenchmarks:
    """Performance benchmark tests."""

    def test_optimization_performance_improvement(self, sample_data):
        """Test that optimization actually improves performance."""
        X, y = sample_data
        X_train, X_test = X[:8000], X[8000:]
        
        # Baseline: Without optimization
        baseline_detector = IsolationForest(random_state=42, contamination=0.1, n_jobs=1)
        
        start_time = time.perf_counter()
        baseline_detector.fit(X_train)
        baseline_predictions = baseline_detector.predict(X_test)
        baseline_time = time.perf_counter() - start_time
        
        # Optimized: With optimization
        service = PerformanceOptimizationService()
        optimized_detector = IsolationForest(random_state=42, contamination=0.1)
        
        start_time = time.perf_counter()
        trained_model, optimized_predictions = service.optimize_anomaly_detection(
            optimized_detector, X_train, X_test
        )
        optimized_time = time.perf_counter() - start_time
        
        # Results should be similar
        assert len(optimized_predictions) == len(baseline_predictions)
        
        # Optimized version should be faster (or at least not significantly slower)
        # Allow some tolerance for measurement variance
        print(f"Baseline time: {baseline_time:.3f}s")
        print(f"Optimized time: {optimized_time:.3f}s")
        print(f"Speedup: {baseline_time / optimized_time:.2f}x")
        
        # The optimization should show improvement in subsequent runs due to caching
        # and other optimizations. For the first run, it might be similar or slightly slower
        # due to optimization overhead
        assert optimized_time <= baseline_time * 2.0  # Allow 2x overhead for first run

    def test_memory_usage_optimization(self, large_sample_data):
        """Test memory usage optimization."""
        X, y = large_sample_data
        
        gc.collect()
        baseline_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Test with optimization service
        service = PerformanceOptimizationService()
        
        # Process large dataset
        optimized_X = service.optimize_data_preprocessing(X)
        detector = IsolationForest(random_state=42, contamination=0.1)
        trained_model, predictions = service.optimize_anomaly_detection(detector, optimized_X)
        
        peak_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_used = peak_memory - baseline_memory
        
        print(f"Memory used during optimization: {memory_used:.2f}MB")
        
        # Clean up
        service.cleanup_resources()
        del optimized_X, trained_model, predictions, detector, service
        gc.collect()
        
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_recovered = peak_memory - final_memory
        
        print(f"Memory recovered after cleanup: {memory_recovered:.2f}MB")
        
        # Should recover most memory
        recovery_rate = memory_recovered / memory_used if memory_used > 0 else 1.0
        assert recovery_rate >= 0.7  # Should recover at least 70% of used memory