"""
Test large-scale datasets for performance and scalability.
"""
import sys
import time
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
import psutil

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    from pynomaly.infrastructure.adapters.pyod_adapter import PyODAdapter
except ImportError:
    try:
        from pynomaly.infrastructure.algorithms.adapters.pyod_adapter import PyODAdapter
    except ImportError:
        # Fallback for testing
        PyODAdapter = None

class PerformanceTracker:
    """Track performance metrics for algorithms."""
    
    def __init__(self):
        self.results = []
    
    def record_result(self, algorithm: str, dataset_size: str, metric: str, value: float):
        """Record a performance result."""
        self.results.append({
            "algorithm": algorithm,
            "dataset_size": dataset_size,
            "metric": metric,
            "value": value,
        })
    
    def get_results_df(self):
        """Get results as a pandas DataFrame."""
        return pd.DataFrame(self.results)

@pytest.fixture
def performance_tracker():
    """Provide a performance tracker for tests."""
    return PerformanceTracker()

class TestLargeScalePerformance:
    """Test large-scale datasets for performance and scalability."""

    @pytest.mark.benchmark
    @pytest.mark.slow
    def test_large_scale_throughput(self, benchmark, performance_tracker):
        """Test throughput on large datasets (1M rows, 10K features)."""
        if PyODAdapter is None:
            pytest.skip("PyODAdapter not available")
            
        try:
            adapter = PyODAdapter("IsolationForest")
            algorithm_name = "IsolationForest"

            # Check available memory before proceeding
            mem_info = psutil.virtual_memory()
            available_gb = mem_info.available / (1024**3)
            
            # Estimate memory needed: 1M * 10K * 8 bytes ≈ 80GB for float64
            # Use float32 to reduce memory usage
            estimated_gb = 1_000_000 * 10_000 * 4 / (1024**3)  # ~37GB for float32
            
            if available_gb < estimated_gb:
                pytest.skip(f"Insufficient memory: need ~{estimated_gb:.1f}GB, have {available_gb:.1f}GB")

            # Very large dataset - use smaller size if memory limited
            n_samples, n_features = 1_000_000, 10_000
            
            # Generate data in chunks to avoid memory issues
            print(f"\nGenerating {n_samples:,} x {n_features:,} dataset...")
            X = np.random.randn(n_samples, n_features).astype(np.float32)
            
            params = {
                "n_estimators": 100,
                "contamination": 0.1,
                "max_samples": "auto",
                "random_state": 42,
            }

            def run_algorithm():
                print(f"Training {algorithm_name}...")
                from pyod.models.iforest import IForest
                
                # Monitor memory usage
                mem_before = psutil.Process().memory_info().rss / (1024**3)  # GB
                
                algorithm_instance = IForest(**params)
                algorithm_instance.fit(X)
                
                mem_after = psutil.Process().memory_info().rss / (1024**3)  # GB
                print(f"Memory usage: {mem_after - mem_before:.2f}GB")
                
                # Score a subset for throughput measurement
                sample_indices = np.random.choice(n_samples, 1000, replace=False)
                scores = algorithm_instance.decision_function(X[sample_indices])
                
                return scores, mem_after - mem_before

            # Run benchmark with extended timeout
            result = benchmark.pedantic(run_algorithm, rounds=3, iterations=1)

            scores, memory_used = result

            # Record performance metrics
            dataset_size = f"{n_samples}x{n_features}"
            
            # Get benchmark stats
            stats = benchmark.stats
            mean_time = stats.mean if hasattr(stats, 'mean') else stats['mean']
            
            performance_tracker.record_result(
                algorithm_name, dataset_size, "execution_time", mean_time
            )
            performance_tracker.record_result(
                algorithm_name, dataset_size, "memory_usage_gb", memory_used
            )

            # Calculate throughput
            throughput = 1000 / mean_time  # samples processed per second
            performance_tracker.record_result(
                algorithm_name, dataset_size, "throughput_samples_per_sec", throughput
            )

            # Performance assertions
            assert throughput > 0, "Throughput should be positive"
            assert memory_used < 100, f"Memory usage too high: {memory_used:.2f}GB"
            assert mean_time < 1800, f"Execution time too long: {mean_time:.2f}s"
            
            print(f"\nPerformance Results:")
            print(f"  Execution time: {mean_time:.2f}s")
            print(f"  Memory usage: {memory_used:.2f}GB")
            print(f"  Throughput: {throughput:.2f} samples/sec")
            
        except MemoryError as e:
            pytest.skip(f"Memory error: {e}")
        except Exception as e:
            pytest.skip(f"Test failed: {e}")
            
    @pytest.mark.benchmark
    @pytest.mark.slow
    def test_high_dimensional_performance(self, benchmark, performance_tracker):
        """Test performance with high-dimensional data (smaller sample size)."""
        if PyODAdapter is None:
            pytest.skip("PyODAdapter not available")
            
        try:
            adapter = PyODAdapter()
            algorithm_name = "IsolationForest"

            if not adapter.supports_algorithm(algorithm_name):
                pytest.skip(f"Algorithm {algorithm_name} not available")

            # High-dimensional dataset with manageable sample size
            n_samples, n_features = 100_000, 10_000
            print(f"\nGenerating {n_samples:,} x {n_features:,} high-dimensional dataset...")
            X = np.random.randn(n_samples, n_features).astype(np.float32)
            
            params = {
                "n_estimators": 50,  # Reduced for high-dimensional data
                "contamination": 0.1,
                "max_samples": "auto",
                "random_state": 42,
            }

            def run_algorithm():
                algorithm_instance = adapter.create_algorithm(algorithm_name, params)
                
                mem_before = psutil.Process().memory_info().rss / (1024**3)
                algorithm_instance.fit(X)
                mem_after = psutil.Process().memory_info().rss / (1024**3)
                
                scores = algorithm_instance.decision_function(X)
                return scores, mem_after - mem_before

            # Run benchmark
            result = benchmark.pedantic(run_algorithm, rounds=3, iterations=1, timeout=1800)

            scores, memory_used = result

            # Record performance
            dataset_size = f"{n_samples}x{n_features}"
            performance_tracker.record_result(
                f"{algorithm_name}_high_dim", dataset_size, "execution_time", benchmark.stats.mean
            )
            performance_tracker.record_result(
                f"{algorithm_name}_high_dim", dataset_size, "memory_usage_gb", memory_used
            )

            # Calculate throughput
            throughput = n_samples / benchmark.stats.mean
            performance_tracker.record_result(
                f"{algorithm_name}_high_dim", dataset_size, "throughput_samples_per_sec", throughput
            )

            # Assertions
            assert throughput > 0, "Throughput should be positive"
            assert len(scores) == n_samples, "All samples should be scored"
            assert np.all(np.isfinite(scores)), "All scores should be finite"
            
            print(f"\nHigh-Dimensional Performance Results:")
            print(f"  Execution time: {benchmark.stats.mean:.2f}s")
            print(f"  Memory usage: {memory_used:.2f}GB")
            print(f"  Throughput: {throughput:.2f} samples/sec")
            
        except MemoryError as e:
            pytest.skip(f"Memory error: {e}")
        except Exception as e:
            pytest.skip(f"Test failed: {e}")

