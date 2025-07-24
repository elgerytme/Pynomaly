"""Validation tests for performance optimizations."""

import pytest
import time
import gc
import numpy as np
import psutil
from typing import List, Dict, Any
from unittest.mock import patch

from anomaly_detection.domain.services.detection_service import DetectionService
from anomaly_detection.domain.services.optimized_detection_service import (
    OptimizedDetectionService, 
    OptimizationConfig,
    get_optimized_detection_service
)


@pytest.mark.performance
@pytest.mark.optimization
class TestOptimizationValidation:
    """Test suite to validate performance optimizations."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Generate test datasets
        np.random.seed(42)
        
        self.small_dataset = np.random.normal(0, 1, (100, 5)).astype(np.float64)
        self.medium_dataset = np.random.normal(0, 1, (500, 10)).astype(np.float64)
        self.large_dataset = np.random.normal(0, 1, (2000, 15)).astype(np.float64)
        
        # Add anomalies to datasets
        for dataset in [self.small_dataset, self.medium_dataset, self.large_dataset]:
            anomaly_count = len(dataset) // 10
            for i in range(-anomaly_count, 0):
                dataset[i] *= 3  # Make anomalous
        
        # Services for comparison
        self.standard_service = DetectionService()
        
        # Optimized service with different configurations
        self.optimized_config = OptimizationConfig(
            enable_model_caching=True,
            enable_batch_processing=True,
            enable_parallel_processing=True,
            enable_memory_optimization=True,
            cache_size_limit=5,
            batch_size_threshold=200,
            max_workers=2,
            memory_limit_mb=300.0
        )
        self.optimized_service = OptimizedDetectionService(self.optimized_config)
        
    def teardown_method(self):
        """Clean up after tests."""
        if hasattr(self, 'optimized_service'):
            self.optimized_service.clear_cache()
    
    def test_performance_comparison_isolation_forest(self):
        """Compare performance between standard and optimized Isolation Forest."""
        algorithm = "iforest"
        contamination = 0.1
        
        # Test on medium dataset
        test_data = self.medium_dataset
        
        # Measure standard service performance
        start_time = time.perf_counter()
        standard_result = self.standard_service.detect_anomalies(
            test_data, algorithm, contamination
        )
        standard_time = time.perf_counter() - start_time
        
        # Measure optimized service performance
        start_time = time.perf_counter()
        optimized_result = self.optimized_service.detect_anomalies(
            test_data, algorithm, contamination
        )
        optimized_time = time.perf_counter() - start_time
        
        # Performance assertions
        assert optimized_result.success, "Optimized detection should succeed"
        assert standard_result.success, "Standard detection should succeed"
        
        # Results should be equivalent (within tolerance for randomness)
        assert optimized_result.total_samples == standard_result.total_samples
        assert abs(optimized_result.anomaly_rate - standard_result.anomaly_rate) < 0.1
        
        # Calculate performance improvement
        speedup = standard_time / optimized_time if optimized_time > 0 else 1.0
        
        print(f"\nPerformance Comparison - Isolation Forest:")
        print(f"  Dataset: {test_data.shape}")
        print(f"  Standard Time: {standard_time:.4f}s")
        print(f"  Optimized Time: {optimized_time:.4f}s")
        print(f"  Speedup: {speedup:.2f}x")
        print(f"  Standard Anomalies: {standard_result.anomaly_count}")
        print(f"  Optimized Anomalies: {optimized_result.anomaly_count}")
        
        # Optimized should be at least as fast (allowing for variation)
        assert optimized_time <= standard_time * 1.2, f"Optimized version slower: {speedup:.2f}x"
    
    def test_model_caching_effectiveness(self):
        """Test effectiveness of model caching."""
        algorithm = "iforest"
        contamination = 0.1
        test_data = self.small_dataset
        
        # First run - no cache
        start_time = time.perf_counter()
        result1 = self.optimized_service.detect_anomalies(test_data, algorithm, contamination)
        first_run_time = time.perf_counter() - start_time
        
        # Second run - should use cache
        start_time = time.perf_counter()
        result2 = self.optimized_service.detect_anomalies(test_data, algorithm, contamination)
        second_run_time = time.perf_counter() - start_time
        
        # Third run - should use cache
        start_time = time.perf_counter()
        result3 = self.optimized_service.detect_anomalies(test_data, algorithm, contamination)
        third_run_time = time.perf_counter() - start_time
        
        # Get cache statistics
        cache_stats = self.optimized_service.model_cache.get_stats()
        
        print(f"\nModel Caching Test:")
        print(f"  First run (no cache): {first_run_time:.4f}s")
        print(f"  Second run (cached): {second_run_time:.4f}s")
        print(f"  Third run (cached): {third_run_time:.4f}s")
        print(f"  Cache size: {cache_stats['size']}")
        print(f"  Cache hit rate: {cache_stats.get('hit_rate', 0):.2f}")
        
        # Results should be consistent
        assert result1.success and result2.success and result3.success
        assert result1.total_samples == result2.total_samples == result3.total_samples
        
        # Cache should be populated
        assert cache_stats['size'] > 0, "Cache should contain models"
        
        # Later runs should generally be faster (allowing for variation)
        avg_cached_time = (second_run_time + third_run_time) / 2
        speedup = first_run_time / avg_cached_time if avg_cached_time > 0 else 1.0
        
        print(f"  Caching speedup: {speedup:.2f}x")
    
    def test_batch_processing_optimization(self):
        """Test batch processing optimization for large datasets."""
        algorithm = "iforest"
        contamination = 0.1
        
        # Test with large dataset that should trigger batching
        test_data = self.large_dataset
        
        # Configure optimized service with batching
        batch_config = OptimizationConfig(
            enable_batch_processing=True,
            batch_size_threshold=500,  # Lower threshold to force batching
            enable_parallel_processing=True,
            max_workers=2
        )
        batch_service = OptimizedDetectionService(batch_config)
        
        # Configure service without batching
        no_batch_config = OptimizationConfig(
            enable_batch_processing=False,
            enable_parallel_processing=False
        )
        no_batch_service = OptimizedDetectionService(no_batch_config)
        
        # Measure batch processing performance
        start_time = time.perf_counter()
        batch_result = batch_service.detect_anomalies(test_data, algorithm, contamination)
        batch_time = time.perf_counter() - start_time
        
        # Measure non-batch performance
        start_time = time.perf_counter()
        no_batch_result = no_batch_service.detect_anomalies(test_data, algorithm, contamination)
        no_batch_time = time.perf_counter() - start_time
        
        print(f"\nBatch Processing Test:")
        print(f"  Dataset: {test_data.shape}")
        print(f"  Batch processing time: {batch_time:.4f}s")
        print(f"  No batch processing time: {no_batch_time:.4f}s")
        print(f"  Batch anomalies: {batch_result.anomaly_count}")
        print(f"  No batch anomalies: {no_batch_result.anomaly_count}")
        
        # Results should be consistent
        assert batch_result.success and no_batch_result.success
        assert batch_result.total_samples == no_batch_result.total_samples
        
        # Batch processing should be competitive (within 50% for small improvement)
        assert batch_time <= no_batch_time * 1.5, "Batch processing significantly slower"
        
        # Clean up
        batch_service.clear_cache()
        no_batch_service.clear_cache()
    
    def test_memory_optimization(self):
        """Test memory usage optimization."""
        algorithm = "iforest"
        contamination = 0.1
        
        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024
        
        # Configure memory-optimized service
        memory_config = OptimizationConfig(
            enable_memory_optimization=True,
            memory_limit_mb=200.0,
            cache_size_limit=3
        )
        memory_service = OptimizedDetectionService(memory_config)
        
        # Run multiple detections to test memory management
        memory_readings = []
        results = []
        
        for i in range(5):
            # Force garbage collection before measurement
            gc.collect()
            
            # Measure memory before detection
            memory_before = process.memory_info().rss / 1024 / 1024
            
            # Run detection
            result = memory_service.detect_anomalies(
                self.medium_dataset, algorithm, contamination
            )
            results.append(result)
            
            # Measure memory after detection
            memory_after = process.memory_info().rss / 1024 / 1024
            memory_readings.append(memory_after)
            
            print(f"  Run {i+1}: {memory_after:.1f}MB (+{memory_after - memory_before:.1f}MB)")
        
        final_memory = memory_readings[-1]
        memory_growth = final_memory - initial_memory
        max_memory = max(memory_readings)
        
        print(f"\nMemory Optimization Test:")
        print(f"  Initial memory: {initial_memory:.1f}MB")
        print(f"  Final memory: {final_memory:.1f}MB")
        print(f"  Memory growth: {memory_growth:.1f}MB")
        print(f"  Peak memory: {max_memory:.1f}MB")
        
        # All detections should succeed
        assert all(r.success for r in results), "All detections should succeed"
        
        # Memory growth should be reasonable (less than 50MB for this test)
        assert memory_growth < 50.0, f"Excessive memory growth: {memory_growth:.1f}MB"
        
        # Memory should not continuously grow (check for leaks)
        if len(memory_readings) >= 3:
            recent_trend = memory_readings[-1] - memory_readings[-3]
            assert recent_trend < 20.0, f"Potential memory leak: {recent_trend:.1f}MB growth"
        
        # Clean up
        memory_service.clear_cache()
        gc.collect()
    
    def test_parallel_batch_processing(self):
        """Test parallel batch processing performance."""
        algorithm = "iforest"
        contamination = 0.1
        
        # Create multiple data batches
        batches = [
            self.small_dataset,
            self.small_dataset * 1.1,  # Slight variation
            self.small_dataset * 0.9,
            self.small_dataset * 1.2
        ]
        
        # Configure parallel service
        parallel_config = OptimizationConfig(
            enable_parallel_processing=True,
            max_workers=2
        )
        parallel_service = OptimizedDetectionService(parallel_config)
        
        # Configure sequential service
        sequential_config = OptimizationConfig(
            enable_parallel_processing=False
        )
        sequential_service = OptimizedDetectionService(sequential_config)
        
        # Measure parallel batch processing
        start_time = time.perf_counter()
        parallel_results = parallel_service.detect_batch(batches, algorithm, contamination)
        parallel_time = time.perf_counter() - start_time
        
        # Measure sequential batch processing
        start_time = time.perf_counter()
        sequential_results = sequential_service.detect_batch(batches, algorithm, contamination)
        sequential_time = time.perf_counter() - start_time
        
        print(f"\nParallel Batch Processing Test:")
        print(f"  Batches: {len(batches)}")
        print(f"  Parallel time: {parallel_time:.4f}s")
        print(f"  Sequential time: {sequential_time:.4f}s")
        print(f"  Speedup: {sequential_time / parallel_time:.2f}x")
        
        # All results should be successful
        assert len(parallel_results) == len(batches)
        assert len(sequential_results) == len(batches)
        assert all(r.success for r in parallel_results)
        assert all(r.success for r in sequential_results)
        
        # Parallel should be competitive or better
        assert parallel_time <= sequential_time * 1.2, "Parallel processing significantly slower"
        
        # Clean up
        parallel_service.clear_cache()
        sequential_service.clear_cache()
    
    def test_algorithm_parameter_optimization(self):
        """Test automatic algorithm parameter optimization."""
        algorithm = "iforest"
        contamination = 0.1
        
        # Test with different data characteristics
        test_cases = [
            ("small_lowdim", np.random.normal(0, 1, (50, 3))),
            ("medium_highdim", np.random.normal(0, 1, (500, 20))),
            ("large_mediumdim", np.random.normal(0, 1, (2000, 10)))
        ]
        
        for case_name, test_data in test_cases:
            # Run optimized detection
            result = self.optimized_service.detect_anomalies(
                test_data, algorithm, contamination
            )
            
            print(f"\nParameter Optimization Test - {case_name}:")
            print(f"  Data shape: {test_data.shape}")
            print(f"  Success: {result.success}")
            print(f"  Anomalies detected: {result.anomaly_count}")
            print(f"  Anomaly rate: {result.anomaly_rate:.3f}")
            
            # All tests should succeed
            assert result.success, f"Detection failed for {case_name}"
            assert result.total_samples == len(test_data)
            
            # Anomaly rate should be reasonable
            assert 0.05 <= result.anomaly_rate <= 0.20, f"Unusual anomaly rate: {result.anomaly_rate}"
    
    def test_optimization_statistics_and_reporting(self):
        """Test optimization statistics collection and reporting."""
        # Run several detections to generate statistics
        algorithms = ["iforest", "lof"]
        datasets = [self.small_dataset, self.medium_dataset]
        
        for algorithm in algorithms:
            for dataset in datasets:
                result = self.optimized_service.detect_anomalies(
                    dataset, algorithm, contamination=0.1
                )
                assert result.success, f"Detection failed for {algorithm}"
        
        # Get optimization statistics
        stats = self.optimized_service.get_optimization_stats()
        
        print(f"\nOptimization Statistics:")
        print(f"  Algorithms tested: {list(stats['algorithm_performance'].keys())}")
        
        for algorithm, perf in stats['algorithm_performance'].items():
            print(f"  {algorithm}:")
            print(f"    Calls: {perf['call_count']}")
            print(f"    Avg time: {perf['avg_time']:.4f}s")
            print(f"    Avg throughput: {perf['avg_throughput']:.1f} samples/s")
        
        if 'cache_stats' in stats:
            cache = stats['cache_stats']
            print(f"  Cache:")
            print(f"    Size: {cache['size']}")
            print(f"    Hit rate: {cache.get('hit_rate', 0):.2f}")
        
        # Get performance report
        report = self.optimized_service.get_performance_report()
        
        # Validate statistics
        assert 'algorithm_statistics' in report
        assert 'optimization_config' in report
        
        # Should have statistics for tested algorithms
        for algorithm in algorithms:
            assert algorithm in stats['algorithm_performance']
            assert stats['algorithm_performance'][algorithm]['call_count'] > 0
    
    def test_cache_memory_limit_enforcement(self):
        """Test that cache respects memory limits."""
        # Configure service with very low memory limit
        low_memory_config = OptimizationConfig(
            enable_model_caching=True,
            memory_limit_mb=50.0,  # Very low limit
            cache_size_limit=10
        )
        
        service = OptimizedDetectionService(low_memory_config)
        
        # Generate many different parameter combinations to fill cache
        algorithm = "iforest"
        contamination_values = [0.05, 0.1, 0.15, 0.2, 0.25]
        n_estimators_values = [50, 100, 150]
        
        results = []
        for contamination in contamination_values:
            for n_estimators in n_estimators_values:
                result = service.detect_anomalies(
                    self.small_dataset, 
                    algorithm, 
                    contamination,
                    n_estimators=n_estimators
                )
                results.append(result)
        
        # Get cache stats
        cache_stats = service.model_cache.get_stats()
        
        print(f"\nCache Memory Limit Test:")
        print(f"  Combinations tested: {len(results)}")
        print(f"  Cache size: {cache_stats['size']}")
        print(f"  Cache max size: {cache_stats['max_size']}")
        
        # All detections should succeed
        assert all(r.success for r in results), "All detections should succeed"
        
        # Cache should not exceed memory limits (should have evicted some models)
        assert cache_stats['size'] <= cache_stats['max_size']
        
        # Clean up
        service.clear_cache()
    
    def test_optimization_recommendations(self):
        """Test optimization recommendation system."""
        # Create service and run some slow operations
        service = OptimizedDetectionService()
        
        # Run detection to generate performance data
        result = service.detect_anomalies(self.large_dataset, "iforest", 0.1)
        assert result.success
        
        # Get performance report with recommendations
        report = service.get_performance_report()
        recommendations = report.get('recommendations', [])
        
        print(f"\nOptimization Recommendations:")
        for i, rec in enumerate(recommendations):
            print(f"  {i+1}. {rec['type']}: {rec.get('suggestion', 'N/A')}")
        
        # Should have meaningful report structure
        assert 'optimization_config' in report
        assert 'algorithm_statistics' in report
        assert isinstance(recommendations, list)
        
        # Clean up
        service.clear_cache()


@pytest.mark.performance
@pytest.mark.optimization
class TestOptimizedServiceIntegration:
    """Integration tests for optimized service with realistic workflows."""
    
    def setup_method(self):
        """Set up integration test fixtures."""
        self.service = get_optimized_detection_service()
        
        # Generate realistic datasets
        np.random.seed(123)
        
        # Time series-like data
        self.timeseries_data = self._generate_timeseries_data(1000, 5)
        
        # High-dimensional data
        self.highdim_data = np.random.normal(0, 1, (300, 50)).astype(np.float64)
        
        # Mixed anomaly data
        self.mixed_data = self._generate_mixed_anomaly_data(500, 8)
    
    def _generate_timeseries_data(self, n_samples: int, n_features: int) -> np.ndarray:
        """Generate time series-like data with trends and seasonality."""
        t = np.linspace(0, 10, n_samples)
        data = np.zeros((n_samples, n_features))
        
        for i in range(n_features):
            # Base trend
            trend = 0.1 * t * (i + 1)
            # Seasonal component
            seasonal = np.sin(2 * np.pi * t * (i + 1) / 10) * (i + 1)
            # Noise
            noise = np.random.normal(0, 0.5, n_samples)
            
            data[:, i] = trend + seasonal + noise
        
        # Add some anomalies
        anomaly_indices = np.random.choice(n_samples, size=n_samples//20, replace=False)
        data[anomaly_indices] *= np.random.uniform(3, 5, size=(len(anomaly_indices), n_features))
        
        return data
    
    def _generate_mixed_anomaly_data(self, n_samples: int, n_features: int) -> np.ndarray:
        """Generate data with different types of anomalies."""
        data = np.random.normal(0, 1, (n_samples, n_features))
        
        # Point anomalies
        point_anomalies = np.random.choice(n_samples, size=n_samples//50, replace=False)
        data[point_anomalies] = np.random.uniform(-5, 5, size=(len(point_anomalies), n_features))
        
        # Cluster anomalies
        cluster_size = n_samples // 100
        cluster_start = np.random.randint(0, n_samples - cluster_size)
        data[cluster_start:cluster_start + cluster_size] += np.random.uniform(2, 4)
        
        return data
    
    def test_realistic_workflow_performance(self):
        """Test performance with realistic data science workflow."""
        algorithms = ["iforest", "lof"]
        datasets = {
            "timeseries": self.timeseries_data,
            "highdim": self.highdim_data,
            "mixed": self.mixed_data
        }
        
        workflow_results = {}
        
        for dataset_name, data in datasets.items():
            dataset_results = {}
            
            for algorithm in algorithms:
                # Measure total workflow time
                start_time = time.perf_counter()
                
                # Run detection multiple times (as in real workflow)
                results = []
                for contamination in [0.05, 0.1, 0.15]:
                    result = self.service.detect_anomalies(
                        data, algorithm, contamination
                    )
                    results.append(result)
                
                total_time = time.perf_counter() - start_time
                
                # Calculate metrics
                avg_anomaly_rate = np.mean([r.anomaly_rate for r in results])
                success_rate = sum(r.success for r in results) / len(results)
                
                dataset_results[algorithm] = {
                    "total_time": total_time,
                    "avg_time_per_run": total_time / len(results),
                    "success_rate": success_rate,
                    "avg_anomaly_rate": avg_anomaly_rate
                }
            
            workflow_results[dataset_name] = dataset_results
        
        # Print comprehensive results
        print(f"\nRealistic Workflow Performance:")
        for dataset_name, dataset_results in workflow_results.items():
            print(f"\n  Dataset: {dataset_name}")
            print(f"  Shape: {datasets[dataset_name].shape}")
            
            for algorithm, metrics in dataset_results.items():
                print(f"    {algorithm}:")
                print(f"      Total time: {metrics['total_time']:.4f}s")
                print(f"      Avg time per run: {metrics['avg_time_per_run']:.4f}s")
                print(f"      Success rate: {metrics['success_rate']:.1%}")
                print(f"      Avg anomaly rate: {metrics['avg_anomaly_rate']:.3f}")
        
        # Performance assertions
        for dataset_results in workflow_results.values():
            for metrics in dataset_results.values():
                assert metrics['success_rate'] == 1.0, "All runs should succeed"
                assert metrics['avg_time_per_run'] < 5.0, "Should be reasonably fast"
                assert 0.01 <= metrics['avg_anomaly_rate'] <= 0.30, "Reasonable anomaly rates"
    
    def test_stress_test_with_optimization(self):
        """Stress test the optimized service with high load."""
        algorithm = "iforest"
        contamination = 0.1
        
        # Run many detections in sequence
        n_iterations = 20
        dataset_sizes = [100, 300, 500, 300, 100]  # Varying sizes
        
        iteration_times = []
        memory_usage = []
        
        process = psutil.Process()
        
        for i in range(n_iterations):
            # Choose dataset size
            size = dataset_sizes[i % len(dataset_sizes)]
            test_data = np.random.normal(0, 1, (size, 10))
            
            # Measure iteration
            start_time = time.perf_counter()
            initial_memory = process.memory_info().rss / 1024 / 1024
            
            result = self.service.detect_anomalies(test_data, algorithm, contamination)
            
            end_time = time.perf_counter()
            final_memory = process.memory_info().rss / 1024 / 1024
            
            iteration_time = end_time - start_time
            memory_delta = final_memory - initial_memory
            
            iteration_times.append(iteration_time)
            memory_usage.append(final_memory)
            
            assert result.success, f"Iteration {i+1} failed"
            
            if i % 5 == 0:  # Log every 5 iterations
                print(f"  Iteration {i+1}: {iteration_time:.4f}s, {final_memory:.1f}MB")
        
        # Analyze stress test results
        avg_time = np.mean(iteration_times)
        max_time = max(iteration_times)
        time_stability = np.std(iteration_times) / avg_time
        
        memory_growth = memory_usage[-1] - memory_usage[0]
        max_memory = max(memory_usage)
        
        print(f"\nStress Test Results:")
        print(f"  Iterations: {n_iterations}")
        print(f"  Avg time: {avg_time:.4f}s")
        print(f"  Max time: {max_time:.4f}s")
        print(f"  Time stability (CV): {time_stability:.3f}")
        print(f"  Memory growth: {memory_growth:.1f}MB")
        print(f"  Peak memory: {max_memory:.1f}MB")
        
        # Performance stability assertions
        assert avg_time < 1.0, f"Average time too high: {avg_time:.4f}s"
        assert time_stability < 0.5, f"Performance too unstable: {time_stability:.3f}"
        assert memory_growth < 30.0, f"Excessive memory growth: {memory_growth:.1f}MB"
        
        # Get final optimization stats
        stats = self.service.get_optimization_stats()
        print(f"  Cache hits: {stats.get('cache_stats', {}).get('hit_rate', 0):.2f}")


if __name__ == "__main__":
    # Run optimization validation tests
    pytest.main([
        __file__ + "::TestOptimizationValidation::test_performance_comparison_isolation_forest",
        "-v", "-s", "--tb=short"
    ])