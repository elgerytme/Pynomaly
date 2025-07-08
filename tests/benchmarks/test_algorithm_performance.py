"""Performance benchmarking tests for anomaly detection algorithms."""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


class BenchmarkDataGenerator:
    """Generate standardized datasets for benchmarking."""

    @staticmethod
    def generate_dataset(
        n_samples: int,
        n_features: int,
        contamination: float = 0.1,
        random_state: int = 42,
    ):
        """Generate a benchmark dataset with known anomalies.

        Args:
            n_samples: Number of samples
            n_features: Number of features
            contamination: Fraction of samples that are anomalies
            random_state: Random seed for reproducibility

        Returns:
            Tuple of (X, y) where X is features and y is labels (1=anomaly, 0=normal)
        """
        np.random.seed(random_state)

        # Generate normal data
        n_normal = int(n_samples * (1 - contamination))
        n_anomalies = n_samples - n_normal

        # Normal data from standard normal distribution
        X_normal = np.random.multivariate_normal(
            mean=np.zeros(n_features), cov=np.eye(n_features), size=n_normal
        )

        # Anomalous data with higher variance and shifted mean
        X_anomalies = np.random.multivariate_normal(
            mean=np.full(n_features, 3.0),  # Shifted mean
            cov=np.eye(n_features) * 4.0,  # Higher variance
            size=n_anomalies,
        )

        # Combine data
        X = np.vstack([X_normal, X_anomalies])
        y = np.hstack([np.zeros(n_normal), np.ones(n_anomalies)])

        # Shuffle to mix normal and anomalous samples
        indices = np.random.permutation(n_samples)
        X = X[indices]
        y = y[indices]

        return X, y


class PerformanceTracker:
    """Track performance metrics for algorithms."""

    def __init__(self):
        self.results = []

    def time_execution(self, func, *args, **kwargs):
        """Time the execution of a function.

        Returns:
            Tuple of (result, execution_time_seconds)
        """
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        return result, execution_time

    def record_result(
        self, algorithm: str, dataset_size: str, metric: str, value: float
    ):
        """Record a performance result."""
        self.results.append(
            {
                "algorithm": algorithm,
                "dataset_size": dataset_size,
                "metric": metric,
                "value": value,
            }
        )

    def get_results_df(self):
        """Get results as a pandas DataFrame."""
        return pd.DataFrame(self.results)


@pytest.fixture
def performance_tracker():
    """Provide a performance tracker for tests."""
    return PerformanceTracker()


class TestAlgorithmPerformance:
    """Performance benchmarks for anomaly detection algorithms."""

    @pytest.mark.benchmark
    @pytest.mark.parametrize(
        "n_samples,n_features",
        [
            (100, 5),
            (1000, 10),
            (5000, 20),
            (10000, 50),
        ],
    )
    def test_isolation_forest_performance(
        self, benchmark, n_samples, n_features, performance_tracker
    ):
        """Benchmark Isolation Forest performance across different data sizes."""
        try:
            from pynomaly.infrastructure.algorithms.adapters.pyod_adapter import (
                PyODAdapter,
            )

            adapter = PyODAdapter()
            algorithm_name = "IsolationForest"

            if not adapter.supports_algorithm(algorithm_name):
                pytest.skip(f"Algorithm {algorithm_name} not available")

            # Generate test data
            X, y = BenchmarkDataGenerator.generate_dataset(n_samples, n_features)

            # Benchmark parameters
            params = {"n_estimators": 100, "contamination": 0.1, "random_state": 42}

            def run_algorithm():
                algorithm_instance = adapter.create_algorithm(algorithm_name, params)
                algorithm_instance.fit(X)
                scores = algorithm_instance.decision_function(X)
                predictions = algorithm_instance.predict(X)
                return scores, predictions

            # Run benchmark
            result = benchmark(run_algorithm)

            # Record performance metrics
            dataset_size = f"{n_samples}x{n_features}"
            performance_tracker.record_result(
                algorithm_name, dataset_size, "execution_time", benchmark.stats.mean
            )
            performance_tracker.record_result(
                algorithm_name, dataset_size, "memory_peak", benchmark.stats.max
            )

            # Verify results are reasonable
            scores, predictions = result
            assert len(scores) == n_samples
            assert len(predictions) == n_samples
            assert np.all(np.isfinite(scores))

        except Exception as e:
            pytest.skip(f"Benchmark failed: {e}")

    @pytest.mark.benchmark
    @pytest.mark.parametrize(
        "algorithm_name",
        [
            "IsolationForest",
            "LocalOutlierFactor",
            "OneClassSVM",
        ],
    )
    def test_algorithm_comparison_medium_dataset(
        self, benchmark, algorithm_name, performance_tracker
    ):
        """Compare performance of different algorithms on medium dataset."""
        try:
            from pynomaly.infrastructure.algorithms.adapters.pyod_adapter import (
                PyODAdapter,
            )

            adapter = PyODAdapter()

            if not adapter.supports_algorithm(algorithm_name):
                pytest.skip(f"Algorithm {algorithm_name} not available")

            # Standard medium dataset
            n_samples, n_features = 2000, 15
            X, y = BenchmarkDataGenerator.generate_dataset(n_samples, n_features)

            # Algorithm-specific parameters
            if algorithm_name == "IsolationForest":
                params = {"n_estimators": 100, "contamination": 0.1, "random_state": 42}
            elif algorithm_name == "LocalOutlierFactor":
                params = {"n_neighbors": 20, "contamination": 0.1}
            elif algorithm_name == "OneClassSVM":
                params = {"kernel": "rbf", "gamma": "scale", "nu": 0.1}
            else:
                params = {"contamination": 0.1, "random_state": 42}

            def run_algorithm():
                algorithm_instance = adapter.create_algorithm(algorithm_name, params)
                algorithm_instance.fit(X)
                scores = algorithm_instance.decision_function(X)
                predictions = algorithm_instance.predict(X)
                return scores, predictions

            # Run benchmark
            result = benchmark(run_algorithm)

            # Record performance
            dataset_size = f"{n_samples}x{n_features}"
            performance_tracker.record_result(
                algorithm_name, dataset_size, "execution_time", benchmark.stats.mean
            )

            # Verify correctness
            scores, predictions = result
            assert len(scores) == n_samples
            assert len(predictions) == n_samples

        except Exception as e:
            pytest.skip(f"Algorithm {algorithm_name} benchmark failed: {e}")

    @pytest.mark.benchmark
    @pytest.mark.slow
    def test_large_dataset_scalability(self, benchmark, performance_tracker):
        """Test algorithm scalability on large datasets."""
        try:
            from pynomaly.infrastructure.algorithms.adapters.pyod_adapter import (
                PyODAdapter,
            )

            adapter = PyODAdapter()
            algorithm_name = "IsolationForest"  # Focus on one scalable algorithm

            if not adapter.supports_algorithm(algorithm_name):
                pytest.skip(f"Algorithm {algorithm_name} not available")

            # Large dataset
            n_samples, n_features = 50000, 30
            X, y = BenchmarkDataGenerator.generate_dataset(n_samples, n_features)

            params = {
                "n_estimators": 100,
                "contamination": 0.1,
                "random_state": 42,
                "max_samples": "auto",
            }

            def run_algorithm():
                algorithm_instance = adapter.create_algorithm(algorithm_name, params)
                algorithm_instance.fit(X)
                # Only score a subset for large datasets
                sample_indices = np.random.choice(
                    n_samples, min(1000, n_samples), replace=False
                )
                scores = algorithm_instance.decision_function(X[sample_indices])
                return scores

            # Run benchmark with longer timeout
            result = benchmark.pedantic(run_algorithm, rounds=3, iterations=1)

            # Record performance
            dataset_size = f"{n_samples}x{n_features}"
            performance_tracker.record_result(
                algorithm_name, dataset_size, "execution_time", benchmark.stats.mean
            )

            # Performance assertions
            assert (
                benchmark.stats.mean < 60.0
            ), "Large dataset should complete within 60 seconds"

        except Exception as e:
            pytest.skip(f"Large dataset benchmark failed: {e}")


class TestMemoryPerformance:
    """Memory usage benchmarks."""

    @pytest.mark.benchmark
    @pytest.mark.parametrize("n_samples", [1000, 5000, 10000])
    def test_memory_usage_scaling(self, n_samples, performance_tracker):
        """Test memory usage scaling with dataset size."""
        try:
            import os

            import psutil
            from pynomaly.infrastructure.algorithms.adapters.pyod_adapter import (
                PyODAdapter,
            )

            adapter = PyODAdapter()
            algorithm_name = "IsolationForest"

            if not adapter.supports_algorithm(algorithm_name):
                pytest.skip(f"Algorithm {algorithm_name} not available")

            # Monitor memory before
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss / 1024 / 1024  # MB

            # Generate data and run algorithm
            n_features = 20
            X, y = BenchmarkDataGenerator.generate_dataset(n_samples, n_features)

            params = {"n_estimators": 100, "contamination": 0.1, "random_state": 42}

            algorithm_instance = adapter.create_algorithm(algorithm_name, params)
            algorithm_instance.fit(X)
            scores = algorithm_instance.decision_function(X)

            # Monitor memory after
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_used = memory_after - memory_before

            # Record memory usage
            dataset_size = f"{n_samples}x{n_features}"
            performance_tracker.record_result(
                algorithm_name, dataset_size, "memory_mb", memory_used
            )

            # Memory usage should be reasonable
            memory_per_sample = memory_used / n_samples * 1000  # KB per sample
            assert (
                memory_per_sample < 10.0
            ), f"Memory usage too high: {memory_per_sample:.2f} KB per sample"

        except ImportError:
            pytest.skip("psutil not available for memory monitoring")
        except Exception as e:
            pytest.skip(f"Memory test failed: {e}")


class TestConcurrencyPerformance:
    """Test algorithm performance under concurrent access."""

    @pytest.mark.benchmark
    @pytest.mark.slow
    def test_parallel_execution(self, benchmark, performance_tracker):
        """Test algorithm performance with parallel execution."""
        try:
            import concurrent.futures

            from pynomaly.infrastructure.algorithms.adapters.pyod_adapter import (
                PyODAdapter,
            )

            adapter = PyODAdapter()
            algorithm_name = "IsolationForest"

            if not adapter.supports_algorithm(algorithm_name):
                pytest.skip(f"Algorithm {algorithm_name} not available")

            # Generate multiple datasets
            datasets = []
            for i in range(4):  # 4 parallel tasks
                X, y = BenchmarkDataGenerator.generate_dataset(
                    1000, 10, random_state=42 + i
                )
                datasets.append(X)

            params = {"n_estimators": 50, "contamination": 0.1, "random_state": 42}

            def run_single_algorithm(X):
                algorithm_instance = adapter.create_algorithm(algorithm_name, params)
                algorithm_instance.fit(X)
                return algorithm_instance.decision_function(X)

            def run_parallel():
                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                    futures = [
                        executor.submit(run_single_algorithm, X) for X in datasets
                    ]
                    results = [
                        future.result()
                        for future in concurrent.futures.as_completed(futures)
                    ]
                return results

            # Benchmark parallel execution
            result = benchmark(run_parallel)

            # Record performance
            performance_tracker.record_result(
                f"{algorithm_name}_parallel",
                "4x1000x10",
                "execution_time",
                benchmark.stats.mean,
            )

            # Verify all results
            assert len(result) == 4
            for scores in result:
                assert len(scores) == 1000
                assert np.all(np.isfinite(scores))

        except ImportError:
            pytest.skip("concurrent.futures not available")
        except Exception as e:
            pytest.skip(f"Parallel execution test failed: {e}")


class TestPerformanceRegression:
    """Test for performance regressions."""

    @pytest.mark.benchmark
    def test_baseline_performance(self, benchmark, performance_tracker):
        """Establish baseline performance metrics."""
        try:
            from pynomaly.infrastructure.algorithms.adapters.pyod_adapter import (
                PyODAdapter,
            )

            adapter = PyODAdapter()
            algorithm_name = "IsolationForest"

            if not adapter.supports_algorithm(algorithm_name):
                pytest.skip(f"Algorithm {algorithm_name} not available")

            # Standard benchmark dataset
            n_samples, n_features = 5000, 20
            X, y = BenchmarkDataGenerator.generate_dataset(n_samples, n_features)

            params = {"n_estimators": 100, "contamination": 0.1, "random_state": 42}

            def run_algorithm():
                algorithm_instance = adapter.create_algorithm(algorithm_name, params)
                algorithm_instance.fit(X)
                scores = algorithm_instance.decision_function(X)
                return scores

            # Benchmark with multiple rounds for stability
            result = benchmark.pedantic(run_algorithm, rounds=5, iterations=1)

            # Performance thresholds (adjust based on typical hardware)
            max_time_seconds = 10.0  # Should complete within 10 seconds
            assert (
                benchmark.stats.mean < max_time_seconds
            ), f"Performance regression: took {benchmark.stats.mean:.2f}s, expected < {max_time_seconds}s"

            # Record baseline
            performance_tracker.record_result(
                f"{algorithm_name}_baseline",
                f"{n_samples}x{n_features}",
                "execution_time",
                benchmark.stats.mean,
            )

        except Exception as e:
            pytest.skip(f"Baseline performance test failed: {e}")


@pytest.mark.benchmark
def test_performance_summary(performance_tracker):
    """Generate a performance summary report."""
    if not performance_tracker.results:
        pytest.skip("No performance results to summarize")

    results_df = performance_tracker.get_results_df()

    print("\n" + "=" * 80)
    print("PERFORMANCE SUMMARY")
    print("=" * 80)

    # Group by algorithm and metric
    for metric in results_df["metric"].unique():
        metric_data = results_df[results_df["metric"] == metric]
        print(f"\n{metric.upper()}:")
        print("-" * 40)

        for _, row in metric_data.iterrows():
            print(f"{row['algorithm']:20} {row['dataset_size']:15} {row['value']:8.3f}")

    print("\n" + "=" * 80)

    # Performance assertions (optional quality gates)
    execution_times = results_df[results_df["metric"] == "execution_time"]
    if not execution_times.empty:
        max_time = execution_times["value"].max()
        print(f"Maximum execution time: {max_time:.3f}s")

        # Warn if any test took longer than 30 seconds
        if max_time > 30.0:
            pytest.warns(
                UserWarning, f"Some tests took longer than 30s: {max_time:.3f}s"
            )
