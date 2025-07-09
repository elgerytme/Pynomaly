"""
Simple performance tests with statistical baseline assertions.

This module contains performance tests that run in a containerized environment
with fixed CPU/RAM limits and compare against statistical baselines.
"""

import json
import os
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest
from scipy import stats


class PerformanceBaselineManager:
    """Manages performance baselines and statistical comparisons."""
    
    def __init__(self, baseline_path: str = "baselines.json"):
        self.baseline_path = Path(baseline_path)
        self.baselines = self._load_baselines()
        
    def _load_baselines(self) -> Dict[str, Any]:
        """Load baseline data from JSON file."""
        if self.baseline_path.exists():
            with open(self.baseline_path, 'r') as f:
                return json.load(f)
        return {}
    
    def get_baseline(self, test_name: str) -> Dict[str, Any]:
        """Get baseline data for a specific test."""
        return self.baselines.get("baselines", {}).get(test_name, {})
    
    def get_threshold(self, metric: str) -> float:
        """Get regression threshold for a metric."""
        thresholds = self.baselines.get("regression_thresholds", {})
        if metric == "median":
            return thresholds.get("max_median_increase_percent", 120) / 100
        elif metric == "mean":
            return thresholds.get("max_mean_increase_percent", 125) / 100
        return 1.2  # Default 20% increase
    
    def assert_performance_regression(self, test_name: str, current_stats: Dict[str, float]):
        """Assert that current performance doesn't exceed baseline thresholds."""
        baseline = self.get_baseline(test_name)
        if not baseline:
            pytest.skip(f"No baseline found for {test_name}")
            
        baseline_median = baseline.get("median_ms", 0)
        baseline_mean = baseline.get("mean_ms", 0)
        p_threshold = baseline.get("p_value_threshold", 0.05)
        
        current_median = current_stats.get("median", 0)
        current_mean = current_stats.get("mean", 0)
        
        # Statistical significance test
        if "samples" in current_stats and baseline_median > 0:
            baseline_samples = [baseline_median] * baseline.get("iterations", 10)
            current_samples = current_stats["samples"]
            
            # Perform t-test
            t_stat, p_value = stats.ttest_ind(current_samples, baseline_samples)
            
            # Assert statistical significance
            assert p_value < p_threshold or current_median <= baseline_median * self.get_threshold("median"), \
                f"Performance regression detected for {test_name}: " \
                f"median {current_median:.2f}ms > {baseline_median * self.get_threshold('median'):.2f}ms " \
                f"(p={p_value:.4f})"
        
        # Assert median performance (if baseline exists)
        if baseline_median > 0:
            assert current_median <= baseline_median * self.get_threshold("median"), \
                f"Median performance regression in {test_name}: " \
                f"{current_median:.2f}ms > {baseline_median * self.get_threshold('median'):.2f}ms"
        
        # Assert mean performance (if baseline exists)
        if baseline_mean > 0:
            assert current_mean <= baseline_mean * self.get_threshold("mean"), \
                f"Mean performance regression in {test_name}: " \
                f"{current_mean:.2f}ms > {baseline_mean * self.get_threshold('mean'):.2f}ms"


# Initialize baseline manager
baseline_manager = PerformanceBaselineManager()


@pytest.fixture
def sample_data_small():
    """Generate small sample dataset for testing."""
    np.random.seed(42)
    data = np.random.randn(100, 5)
    return pd.DataFrame(data, columns=[f"feature_{i}" for i in range(5)])


@pytest.fixture
def sample_data_medium():
    """Generate medium sample dataset for testing."""
    np.random.seed(42)
    data = np.random.randn(1000, 10)
    return pd.DataFrame(data, columns=[f"feature_{i}" for i in range(10)])


@pytest.fixture
def sample_data_large():
    """Generate large sample dataset for testing."""
    np.random.seed(42)
    data = np.random.randn(10000, 20)
    return pd.DataFrame(data, columns=[f"feature_{i}" for i in range(20)])


@pytest.mark.performance
def test_data_loading_performance(benchmark, sample_data_small):
    """Test data loading performance against baseline."""
    def load_data():
        # Simulate data loading
        return sample_data_small.copy()
    
    # Benchmark the function
    result = benchmark(load_data)
    
    # Get benchmark statistics
    stats_dict = {
        "median": benchmark.stats['median'] * 1000,  # Convert to ms
        "mean": benchmark.stats['mean'] * 1000,
        "std": benchmark.stats['stddev'] * 1000,
        "min": benchmark.stats['min'] * 1000,
        "max": benchmark.stats['max'] * 1000,
        "samples": [s * 1000 for s in benchmark.stats['data']]
    }
    
    # Assert against baseline
    baseline_manager.assert_performance_regression("data_loading", stats_dict)
    
    assert result is not None


@pytest.mark.performance
def test_anomaly_detection_small_performance(benchmark, sample_data_small):
    """Test anomaly detection performance on small dataset."""
    def detect_anomalies():
        # Simulate anomaly detection
        from sklearn.ensemble import IsolationForest
        detector = IsolationForest(random_state=42)
        return detector.fit_predict(sample_data_small)
    
    result = benchmark(detect_anomalies)
    
    stats_dict = {
        "median": benchmark.stats['median'] * 1000,
        "mean": benchmark.stats['mean'] * 1000,
        "std": benchmark.stats['stddev'] * 1000,
        "min": benchmark.stats['min'] * 1000,
        "max": benchmark.stats['max'] * 1000,
        "samples": [s * 1000 for s in benchmark.stats['data']]
    }
    
    baseline_manager.assert_performance_regression("anomaly_detection_small", stats_dict)
    
    assert len(result) == len(sample_data_small)


@pytest.mark.performance
def test_anomaly_detection_medium_performance(benchmark, sample_data_medium):
    """Test anomaly detection performance on medium dataset."""
    def detect_anomalies():
        from sklearn.ensemble import IsolationForest
        detector = IsolationForest(random_state=42)
        return detector.fit_predict(sample_data_medium)
    
    result = benchmark(detect_anomalies)
    
    stats_dict = {
        "median": benchmark.stats['median'] * 1000,
        "mean": benchmark.stats['mean'] * 1000,
        "std": benchmark.stats['stddev'] * 1000,
        "min": benchmark.stats['min'] * 1000,
        "max": benchmark.stats['max'] * 1000,
        "samples": [s * 1000 for s in benchmark.stats['data']]
    }
    
    baseline_manager.assert_performance_regression("anomaly_detection_medium", stats_dict)
    
    assert len(result) == len(sample_data_medium)


@pytest.mark.performance
def test_anomaly_detection_large_performance(benchmark, sample_data_large):
    """Test anomaly detection performance on large dataset."""
    def detect_anomalies():
        from sklearn.ensemble import IsolationForest
        detector = IsolationForest(random_state=42)
        return detector.fit_predict(sample_data_large)
    
    result = benchmark(detect_anomalies)
    
    stats_dict = {
        "median": benchmark.stats['median'] * 1000,
        "mean": benchmark.stats['mean'] * 1000,
        "std": benchmark.stats['stddev'] * 1000,
        "min": benchmark.stats['min'] * 1000,
        "max": benchmark.stats['max'] * 1000,
        "samples": [s * 1000 for s in benchmark.stats['data']]
    }
    
    baseline_manager.assert_performance_regression("anomaly_detection_large", stats_dict)
    
    assert len(result) == len(sample_data_large)


@pytest.mark.performance
def test_model_training_performance(benchmark, sample_data_medium):
    """Test model training performance."""
    def train_model():
        from sklearn.ensemble import IsolationForest
        detector = IsolationForest(random_state=42)
        return detector.fit(sample_data_medium)
    
    result = benchmark(train_model)
    
    stats_dict = {
        "median": benchmark.stats['median'] * 1000,
        "mean": benchmark.stats['mean'] * 1000,
        "std": benchmark.stats['stddev'] * 1000,
        "min": benchmark.stats['min'] * 1000,
        "max": benchmark.stats['max'] * 1000,
        "samples": [s * 1000 for s in benchmark.stats['data']]
    }
    
    baseline_manager.assert_performance_regression("model_training", stats_dict)
    
    assert result is not None


@pytest.mark.performance
def test_preprocessing_performance(benchmark, sample_data_medium):
    """Test data preprocessing performance."""
    def preprocess_data():
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        return scaler.fit_transform(sample_data_medium)
    
    result = benchmark(preprocess_data)
    
    stats_dict = {
        "median": benchmark.stats['median'] * 1000,
        "mean": benchmark.stats['mean'] * 1000,
        "std": benchmark.stats['stddev'] * 1000,
        "min": benchmark.stats['min'] * 1000,
        "max": benchmark.stats['max'] * 1000,
        "samples": [s * 1000 for s in benchmark.stats['data']]
    }
    
    baseline_manager.assert_performance_regression("preprocessing", stats_dict)
    
    assert result.shape == sample_data_medium.shape


@pytest.mark.performance
def test_ensemble_prediction_performance(benchmark, sample_data_medium):
    """Test ensemble prediction performance."""
    def ensemble_predict():
        from sklearn.ensemble import IsolationForest
        from sklearn.svm import OneClassSVM
        
        # Create ensemble of detectors
        detectors = [
            IsolationForest(random_state=42),
            OneClassSVM(nu=0.1)
        ]
        
        results = []
        for detector in detectors:
            detector.fit(sample_data_medium)
            results.append(detector.predict(sample_data_medium))
        
        # Ensemble prediction (majority vote)
        return np.mean(results, axis=0)
    
    result = benchmark(ensemble_predict)
    
    stats_dict = {
        "median": benchmark.stats['median'] * 1000,
        "mean": benchmark.stats['mean'] * 1000,
        "std": benchmark.stats['stddev'] * 1000,
        "min": benchmark.stats['min'] * 1000,
        "max": benchmark.stats['max'] * 1000,
        "samples": [s * 1000 for s in benchmark.stats['data']]
    }
    
    baseline_manager.assert_performance_regression("ensemble_prediction", stats_dict)
    
    assert len(result) == len(sample_data_medium)


@pytest.mark.performance
def test_api_response_simulation(benchmark):
    """Test API response time simulation."""
    def simulate_api_response():
        # Simulate API processing time
        time.sleep(0.01)  # 10ms simulation
        return {"status": "success", "anomalies": [1, 2, 3]}
    
    result = benchmark(simulate_api_response)
    
    stats_dict = {
        "median": benchmark.stats.median * 1000,
        "mean": benchmark.stats.mean * 1000,
        "std": benchmark.stats.stddev * 1000,
        "min": benchmark.stats.min * 1000,
        "max": benchmark.stats.max * 1000,
        "samples": [s * 1000 for s in benchmark.stats.stats]
    }
    
    baseline_manager.assert_performance_regression("api_response", stats_dict)
    
    assert result["status"] == "success"


if __name__ == "__main__":
    # Run performance tests
    pytest.main([__file__, "-v", "--benchmark-only"])
