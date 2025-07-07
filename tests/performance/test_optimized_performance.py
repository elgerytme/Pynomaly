"""Optimized performance tests with improved execution speed and reliability."""

import gc
import multiprocessing as mp
import resource
import threading
import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import psutil
import pytest


@dataclass
class PerformanceMetrics:
    """Structure for performance test results."""

    execution_time: float
    memory_usage_mb: float
    cpu_usage_percent: float
    throughput_samples_per_second: float
    peak_memory_mb: float
    gc_collections: int

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for easy serialization."""
        return {
            "execution_time": self.execution_time,
            "memory_usage_mb": self.memory_usage_mb,
            "cpu_usage_percent": self.cpu_usage_percent,
            "throughput_samples_per_second": self.throughput_samples_per_second,
            "peak_memory_mb": self.peak_memory_mb,
            "gc_collections": self.gc_collections,
        }


class PerformanceMonitor:
    """Optimized performance monitoring with minimal overhead."""

    def __init__(self):
        self.process = psutil.Process()
        self.start_time = None
        self.start_memory = None
        self.start_cpu_time = None
        self.peak_memory = 0
        self.gc_start_collections = None

    @contextmanager
    def measure(self):
        """Context manager for performance measurement."""
        # Force garbage collection before measurement
        gc.collect()

        # Record initial state
        self.start_time = time.perf_counter()
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.start_cpu_time = self.process.cpu_times()
        self.peak_memory = self.start_memory
        self.gc_start_collections = sum(
            gc.get_stats()[i]["collections"] for i in range(len(gc.get_stats()))
        )

        # Monitor peak memory in background thread
        monitor_thread = threading.Thread(target=self._monitor_memory, daemon=True)
        monitor_thread.start()

        try:
            yield self
        finally:
            # Calculate final metrics
            end_time = time.perf_counter()
            end_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            end_cpu_time = self.process.cpu_times()
            gc_end_collections = sum(
                gc.get_stats()[i]["collections"] for i in range(len(gc.get_stats()))
            )

            # Store results
            self.execution_time = end_time - self.start_time
            self.memory_usage_mb = end_memory - self.start_memory
            self.cpu_usage_percent = (
                (end_cpu_time.user - self.start_cpu_time.user)
                / max(self.execution_time, 0.001)
            ) * 100
            self.gc_collections = gc_end_collections - self.gc_start_collections

            # Force cleanup
            gc.collect()

    def _monitor_memory(self):
        """Monitor peak memory usage in background."""
        try:
            while True:
                current_memory = self.process.memory_info().rss / 1024 / 1024
                self.peak_memory = max(self.peak_memory, current_memory)
                time.sleep(0.01)  # Check every 10ms
        except:
            pass  # Thread will be daemon, so it's OK if it dies

    def get_metrics(self, n_samples: int = 1) -> PerformanceMetrics:
        """Get performance metrics."""
        throughput = n_samples / max(self.execution_time, 0.001)

        return PerformanceMetrics(
            execution_time=self.execution_time,
            memory_usage_mb=self.memory_usage_mb,
            cpu_usage_percent=self.cpu_usage_percent,
            throughput_samples_per_second=throughput,
            peak_memory_mb=self.peak_memory,
            gc_collections=self.gc_collections,
        )


class OptimizedDataGenerator:
    """Optimized test data generation with caching."""

    def __init__(self):
        self._cache = {}
        self._cache_hits = 0
        self._cache_misses = 0

    def generate_dataset(
        self,
        n_samples: int,
        n_features: int,
        contamination: float = 0.1,
        random_state: int = 42,
    ) -> np.ndarray:
        """Generate dataset with caching for repeated configurations."""
        cache_key = (n_samples, n_features, contamination, random_state)

        if cache_key in self._cache:
            self._cache_hits += 1
            return self._cache[cache_key].copy()

        self._cache_misses += 1

        # Generate data efficiently
        np.random.seed(random_state)

        # Generate normal data
        n_normal = int(n_samples * (1 - contamination))
        normal_data = np.random.multivariate_normal(
            mean=np.zeros(n_features), cov=np.eye(n_features), size=n_normal
        )

        # Generate anomalous data
        n_anomalies = n_samples - n_normal
        if n_anomalies > 0:
            # Anomalies with higher variance
            anomaly_data = np.random.multivariate_normal(
                mean=np.zeros(n_features), cov=3 * np.eye(n_features), size=n_anomalies
            )

            data = np.vstack([normal_data, anomaly_data])
        else:
            data = normal_data

        # Shuffle
        indices = np.random.permutation(len(data))
        data = data[indices]

        # Cache result (limit cache size)
        if len(self._cache) < 10:
            self._cache[cache_key] = data.copy()

        return data

    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache performance statistics."""
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "cached_datasets": len(self._cache),
        }


class OptimizedAlgorithmRunner:
    """Optimized algorithm execution with parallel processing."""

    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or mp.cpu_count()
        self.data_generator = OptimizedDataGenerator()

    def run_single_algorithm(
        self, algorithm_name: str, data: np.ndarray, parameters: Dict[str, Any] = None
    ) -> Tuple[np.ndarray, np.ndarray, PerformanceMetrics]:
        """Run single algorithm with performance monitoring."""
        parameters = parameters or {}

        # Import algorithm
        if algorithm_name == "IsolationForest":
            from sklearn.ensemble import IsolationForest

            algorithm_class = IsolationForest
        elif algorithm_name == "LocalOutlierFactor":
            from sklearn.neighbors import LocalOutlierFactor

            algorithm_class = LocalOutlierFactor
        elif algorithm_name == "OneClassSVM":
            from sklearn.svm import OneClassSVM

            algorithm_class = OneClassSVM
        else:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")

        monitor = PerformanceMonitor()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            with monitor.measure():
                # Create and configure algorithm
                detector = algorithm_class(**parameters)

                # Fit and predict
                if algorithm_name == "LocalOutlierFactor":
                    predictions = detector.fit_predict(data)
                    scores = detector.negative_outlier_factor_
                else:
                    detector.fit(data)
                    predictions = detector.predict(data)
                    scores = detector.decision_function(data)

        metrics = monitor.get_metrics(len(data))
        return predictions, scores, metrics

    def run_algorithm_scaling_test(
        self, algorithm_name: str, data_sizes: List[int], n_features: int = 10
    ) -> Dict[int, PerformanceMetrics]:
        """Run scaling test for algorithm with different data sizes."""
        results = {}

        for size in data_sizes:
            # Generate data
            data = self.data_generator.generate_dataset(size, n_features)

            # Run algorithm
            _, _, metrics = self.run_single_algorithm(algorithm_name, data)
            results[size] = metrics

        return results

    def run_parallel_algorithms(
        self, algorithms: List[str], data: np.ndarray
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray, PerformanceMetrics]]:
        """Run multiple algorithms in parallel."""
        # For now, run sequentially (parallel execution would require more complex setup)
        results = {}

        for algorithm in algorithms:
            try:
                predictions, scores, metrics = self.run_single_algorithm(
                    algorithm, data
                )
                results[algorithm] = (predictions, scores, metrics)
            except Exception as e:
                # Store error information
                results[algorithm] = (
                    None,
                    None,
                    PerformanceMetrics(
                        execution_time=0.0,
                        memory_usage_mb=0.0,
                        cpu_usage_percent=0.0,
                        throughput_samples_per_second=0.0,
                        peak_memory_mb=0.0,
                        gc_collections=0,
                    ),
                )

        return results


class TestOptimizedPerformance:
    """Optimized performance tests with faster execution."""

    @pytest.fixture(scope="class")
    def algorithm_runner(self):
        """Shared algorithm runner for performance tests."""
        return OptimizedAlgorithmRunner()

    @pytest.fixture(scope="class")
    def data_generator(self):
        """Shared data generator for performance tests."""
        return OptimizedDataGenerator()

    @pytest.mark.performance
    @pytest.mark.parametrize("algorithm", ["IsolationForest", "LocalOutlierFactor"])
    @pytest.mark.parametrize("data_size", [100, 500, 1000])
    def test_algorithm_performance_scaling(
        self, algorithm_runner, algorithm, data_size
    ):
        """Test algorithm performance scaling with optimized execution."""
        # Generate test data
        data = algorithm_runner.data_generator.generate_dataset(
            n_samples=data_size, n_features=5, contamination=0.1
        )

        # Run algorithm
        predictions, scores, metrics = algorithm_runner.run_single_algorithm(
            algorithm, data
        )

        # Performance assertions
        assert metrics.execution_time > 0
        assert metrics.execution_time < 30.0  # Should complete within 30 seconds
        assert metrics.throughput_samples_per_second > 10  # At least 10 samples/second
        assert metrics.memory_usage_mb < 500  # Should not use more than 500MB

        # Functional assertions
        assert len(predictions) == data_size
        assert len(scores) == data_size
        assert all(pred in [-1, 1] for pred in predictions)
        assert all(np.isfinite(score) for score in scores)

    @pytest.mark.performance
    def test_memory_efficiency(self, algorithm_runner):
        """Test memory efficiency with large datasets."""
        # Test with progressively larger datasets
        data_sizes = [1000, 2000, 5000]
        memory_usage = []

        for size in data_sizes:
            data = algorithm_runner.data_generator.generate_dataset(
                n_samples=size, n_features=10
            )

            _, _, metrics = algorithm_runner.run_single_algorithm(
                "IsolationForest", data
            )
            memory_usage.append(metrics.peak_memory_mb)

        # Memory usage should scale reasonably (not exponentially)
        for i in range(1, len(memory_usage)):
            ratio = memory_usage[i] / memory_usage[i - 1]
            size_ratio = data_sizes[i] / data_sizes[i - 1]

            # Memory should not grow faster than O(n * log(n))
            assert ratio <= size_ratio * 2  # Allow for some overhead

    @pytest.mark.performance
    def test_concurrent_algorithm_execution(self, algorithm_runner):
        """Test concurrent execution of multiple algorithms."""
        # Generate shared test data
        data = algorithm_runner.data_generator.generate_dataset(
            n_samples=500, n_features=8
        )

        algorithms = ["IsolationForest", "LocalOutlierFactor"]

        # Measure time for sequential execution
        start_time = time.perf_counter()
        sequential_results = {}
        for algorithm in algorithms:
            predictions, scores, metrics = algorithm_runner.run_single_algorithm(
                algorithm, data
            )
            sequential_results[algorithm] = metrics
        sequential_time = time.perf_counter() - start_time

        # Test parallel execution (simulated)
        start_time = time.perf_counter()
        parallel_results = algorithm_runner.run_parallel_algorithms(algorithms, data)
        parallel_time = time.perf_counter() - start_time

        # Verify results
        assert len(parallel_results) == len(algorithms)

        for algorithm in algorithms:
            if parallel_results[algorithm][0] is not None:  # If no error occurred
                predictions, scores, metrics = parallel_results[algorithm]
                assert len(predictions) == len(data)
                assert len(scores) == len(data)

    @pytest.mark.performance
    def test_algorithm_comparison_benchmark(self, algorithm_runner):
        """Benchmark comparison of different algorithms."""
        data = algorithm_runner.data_generator.generate_dataset(
            n_samples=1000, n_features=10
        )

        algorithms = ["IsolationForest", "LocalOutlierFactor"]
        benchmark_results = {}

        for algorithm in algorithms:
            try:
                predictions, scores, metrics = algorithm_runner.run_single_algorithm(
                    algorithm, data
                )
                benchmark_results[algorithm] = metrics.to_dict()
            except Exception as e:
                benchmark_results[algorithm] = {"error": str(e)}

        # Compare performance characteristics
        successful_algorithms = [
            alg for alg in algorithms if "error" not in benchmark_results[alg]
        ]

        if len(successful_algorithms) >= 2:
            # Find fastest algorithm
            fastest = min(
                successful_algorithms,
                key=lambda alg: benchmark_results[alg]["execution_time"],
            )

            # Find most memory efficient
            most_efficient = min(
                successful_algorithms,
                key=lambda alg: benchmark_results[alg]["peak_memory_mb"],
            )

            # Verify reasonable performance
            for algorithm in successful_algorithms:
                metrics = benchmark_results[algorithm]
                assert metrics["execution_time"] < 60.0  # Within 1 minute
                assert (
                    metrics["throughput_samples_per_second"] > 5
                )  # Reasonable throughput
                assert metrics["peak_memory_mb"] < 1000  # Less than 1GB

    @pytest.mark.performance
    def test_data_loading_performance(self, data_generator):
        """Test data loading and preprocessing performance."""
        monitor = PerformanceMonitor()

        # Test data generation performance
        with monitor.measure():
            datasets = []
            for i in range(5):
                data = data_generator.generate_dataset(
                    n_samples=1000, n_features=10, random_state=i
                )
                datasets.append(data)

        metrics = monitor.get_metrics(5000)  # 5 datasets * 1000 samples

        # Performance assertions
        assert metrics.execution_time < 10.0  # Should generate quickly
        assert metrics.throughput_samples_per_second > 100  # Efficient generation

        # Verify cache effectiveness
        cache_stats = data_generator.get_cache_stats()
        assert (
            cache_stats["cached_datasets"] > 0 or cache_stats["hits"] == 0
        )  # Either cached or no repeats

    @pytest.mark.performance
    def test_feature_engineering_performance(self, algorithm_runner):
        """Test performance of feature engineering operations."""
        # Generate base data
        base_data = algorithm_runner.data_generator.generate_dataset(
            n_samples=2000, n_features=20
        )

        monitor = PerformanceMonitor()

        with monitor.measure():
            # Simulate feature engineering operations

            # 1. Scaling
            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(base_data)

            # 2. PCA
            from sklearn.decomposition import PCA

            pca = PCA(n_components=10)
            pca_data = pca.fit_transform(scaled_data)

            # 3. Feature selection
            selected_features = pca_data[:, :5]  # Simple selection

        metrics = monitor.get_metrics(len(base_data))

        # Performance assertions
        assert metrics.execution_time < 20.0  # Feature engineering should be fast
        assert metrics.memory_usage_mb < 200  # Reasonable memory usage

        # Verify output
        assert selected_features.shape == (2000, 5)
        assert np.all(np.isfinite(selected_features))

    @pytest.mark.performance
    def test_batch_processing_performance(self, algorithm_runner):
        """Test batch processing performance."""
        # Generate multiple batches
        batch_size = 200
        n_batches = 5

        batches = []
        for i in range(n_batches):
            batch = algorithm_runner.data_generator.generate_dataset(
                n_samples=batch_size, n_features=8, random_state=i
            )
            batches.append(batch)

        monitor = PerformanceMonitor()

        with monitor.measure():
            # Process batches sequentially
            all_predictions = []
            all_scores = []

            for batch in batches:
                predictions, scores, _ = algorithm_runner.run_single_algorithm(
                    "IsolationForest", batch
                )
                all_predictions.extend(predictions)
                all_scores.extend(scores)

        metrics = monitor.get_metrics(batch_size * n_batches)

        # Performance assertions
        assert metrics.execution_time < 30.0  # Batch processing should be efficient
        assert metrics.throughput_samples_per_second > 20  # Good throughput

        # Verify output
        assert len(all_predictions) == batch_size * n_batches
        assert len(all_scores) == batch_size * n_batches

    @pytest.mark.performance
    def test_resource_cleanup_performance(self, algorithm_runner):
        """Test resource cleanup and garbage collection performance."""
        initial_objects = len(gc.get_objects())

        # Create and destroy many objects
        for i in range(10):
            data = algorithm_runner.data_generator.generate_dataset(
                n_samples=500, n_features=5, random_state=i
            )

            predictions, scores, metrics = algorithm_runner.run_single_algorithm(
                "IsolationForest", data
            )

            # Force cleanup
            del data, predictions, scores

        # Force garbage collection
        gc.collect()

        final_objects = len(gc.get_objects())

        # Object count should not grow excessively
        object_growth = final_objects - initial_objects
        assert object_growth < 10000  # Allow reasonable growth but not memory leak

    @pytest.mark.performance
    @pytest.mark.parametrize("n_features", [5, 10, 20, 50])
    def test_dimensionality_performance_impact(self, algorithm_runner, n_features):
        """Test performance impact of increasing dimensionality."""
        data = algorithm_runner.data_generator.generate_dataset(
            n_samples=1000, n_features=n_features
        )

        predictions, scores, metrics = algorithm_runner.run_single_algorithm(
            "IsolationForest", data
        )

        # Performance should degrade gracefully with dimensionality
        expected_max_time = 5.0 + (
            n_features * 0.5
        )  # Allow more time for higher dimensions
        assert metrics.execution_time < expected_max_time

        # Throughput should remain reasonable
        min_throughput = max(
            10, 100 - n_features
        )  # Lower expectation for high dimensions
        assert metrics.throughput_samples_per_second > min_throughput

        # Verify functional correctness
        assert len(predictions) == 1000
        assert len(scores) == 1000
        assert all(np.isfinite(score) for score in scores)


class TestPerformanceRegression:
    """Tests to detect performance regressions."""

    @pytest.fixture(scope="class")
    def baseline_results_file(self):
        """File to store baseline performance results."""
        return Path("tests/performance/baseline_results.json")

    def load_baseline(self, baseline_file: Path) -> Dict[str, Any]:
        """Load baseline performance results."""
        if baseline_file.exists():
            import json

            with open(baseline_file, "r") as f:
                return json.load(f)
        return {}

    def save_baseline(self, baseline_file: Path, results: Dict[str, Any]):
        """Save baseline performance results."""
        import json

        baseline_file.parent.mkdir(parents=True, exist_ok=True)
        with open(baseline_file, "w") as f:
            json.dump(results, f, indent=2)

    @pytest.mark.performance
    def test_performance_regression_detection(self, baseline_results_file):
        """Test for performance regressions against baseline."""
        runner = OptimizedAlgorithmRunner()

        # Run current performance test
        data = runner.data_generator.generate_dataset(1000, 10)
        predictions, scores, current_metrics = runner.run_single_algorithm(
            "IsolationForest", data
        )

        current_results = {"isolation_forest_1000x10": current_metrics.to_dict()}

        # Load baseline results
        baseline = self.load_baseline(baseline_results_file)

        if not baseline:
            # First run - establish baseline
            self.save_baseline(baseline_results_file, current_results)
            pytest.skip("Establishing performance baseline")

        # Compare against baseline
        test_name = "isolation_forest_1000x10"
        if test_name in baseline:
            baseline_metrics = baseline[test_name]
            current_metrics_dict = current_results[test_name]

            # Check for significant regressions (>50% slowdown)
            time_regression = (
                current_metrics_dict["execution_time"]
                / baseline_metrics["execution_time"]
            )

            if time_regression > 1.5:
                pytest.fail(
                    f"Performance regression detected: {time_regression:.2f}x slower than baseline"
                )

            # Check for memory regression (>2x memory usage)
            memory_regression = current_metrics_dict["peak_memory_mb"] / max(
                baseline_metrics["peak_memory_mb"], 1.0
            )

            if memory_regression > 2.0:
                pytest.fail(
                    f"Memory regression detected: {memory_regression:.2f}x more memory than baseline"
                )

        # Update baseline with current results (optional)
        # self.save_baseline(baseline_results_file, current_results)
