"""
Edge performance validation test to verify the testing framework works.
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
        # Mock for testing
        class PyODAdapter:
            def supports_algorithm(self, name):
                return name == "IsolationForest"
            
            def create_algorithm(self, name, params):
                class MockAlgorithm:
                    def fit(self, X):
                        time.sleep(0.1)  # Simulate processing time
                        return self
                    
                    def decision_function(self, X):
                        return np.random.random(len(X))
                
                return MockAlgorithm()

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

class TestEdgePerformanceValidation:
    """Validate edge performance testing framework."""
    
    @pytest.mark.benchmark
    def test_small_dataset_benchmark(self, benchmark, performance_tracker):
        """Test small dataset to validate framework."""
        adapter = PyODAdapter("IsolationForest")
        
        # Small dataset
        n_samples, n_features = 1000, 50
        X = np.random.randn(n_samples, n_features).astype(np.float32)
        
        params = {
            "n_estimators": 10,
            "contamination": 0.1,
            "random_state": 42,
        }
        
        def run_algorithm():
            # Create PyOD model directly since the adapter is now the detector
            from pyod.models.iforest import IForest
            
            mem_before = psutil.Process().memory_info().rss / (1024**2)  # MB
            algorithm_instance = IForest(**params)
            algorithm_instance.fit(X)
            mem_after = psutil.Process().memory_info().rss / (1024**2)  # MB
            
            scores = algorithm_instance.decision_function(X)
            return scores, mem_after - mem_before
        
        # Run benchmark
        result = benchmark(run_algorithm)
        scores, memory_used = result
        
        # Record performance
        dataset_size = f"{n_samples}x{n_features}"
        algorithm_name = "IsolationForest"
        
        # Get benchmark stats
        stats = benchmark.stats
        mean_time = stats.mean if hasattr(stats, 'mean') else stats['mean']
        
        performance_tracker.record_result(
            algorithm_name, dataset_size, "execution_time", mean_time
        )
        performance_tracker.record_result(
            algorithm_name, dataset_size, "memory_usage_mb", memory_used
        )
        
        # Calculate throughput
        throughput = n_samples / mean_time
        performance_tracker.record_result(
            algorithm_name, dataset_size, "throughput_samples_per_sec", throughput
        )
        
        # Assertions
        assert throughput > 0, "Throughput should be positive"
        assert len(scores) == n_samples, "All samples should be scored"
        assert np.all(np.isfinite(scores)), "All scores should be finite"
        
        print(f"\nSmall Dataset Performance Results:")
        print(f"  Execution time: {mean_time:.3f}s")
        print(f"  Memory usage: {memory_used:.2f}MB")
        print(f"  Throughput: {throughput:.2f} samples/sec")
        
    @pytest.mark.benchmark
    @pytest.mark.slow
    def test_medium_dataset_stress(self, benchmark, performance_tracker):
        """Test medium dataset for stress testing."""
        adapter = PyODAdapter("IsolationForest")
        
        # Medium dataset with high dimensions
        n_samples, n_features = 10_000, 1_000
        X = np.random.randn(n_samples, n_features).astype(np.float32)
        
        params = {
            "n_estimators": 50,
            "contamination": 0.1,
            "random_state": 42,
        }
        
        def run_algorithm():
            # Create PyOD model directly since the adapter is now the detector
            from pyod.models.iforest import IForest
            
            mem_before = psutil.Process().memory_info().rss / (1024**2)  # MB
            algorithm_instance = IForest(**params)
            algorithm_instance.fit(X)
            mem_after = psutil.Process().memory_info().rss / (1024**2)  # MB
            
            scores = algorithm_instance.decision_function(X)
            return scores, mem_after - mem_before
        
        # Run benchmark
        result = benchmark.pedantic(run_algorithm, rounds=3, iterations=1)
        scores, memory_used = result
        
        # Record performance
        dataset_size = f"{n_samples}x{n_features}"
        algorithm_name = "IsolationForest"
        
        # Get benchmark stats
        stats = benchmark.stats
        mean_time = stats.mean if hasattr(stats, 'mean') else stats['mean']
        
        performance_tracker.record_result(
            f"{algorithm_name}_medium", dataset_size, "execution_time", mean_time
        )
        performance_tracker.record_result(
            f"{algorithm_name}_medium", dataset_size, "memory_usage_mb", memory_used
        )
        
        # Calculate throughput
        throughput = n_samples / mean_time
        performance_tracker.record_result(
            f"{algorithm_name}_medium", dataset_size, "throughput_samples_per_sec", throughput
        )
        
        # Assertions
        assert throughput > 0, "Throughput should be positive"
        assert len(scores) == n_samples, "All samples should be scored"
        assert np.all(np.isfinite(scores)), "All scores should be finite"
        assert mean_time < 300, f"Execution time too long: {mean_time:.2f}s"
        
        print(f"\nMedium Dataset Performance Results:")
        print(f"  Execution time: {mean_time:.3f}s")
        print(f"  Memory usage: {memory_used:.2f}MB")
        print(f"  Throughput: {throughput:.2f} samples/sec")
