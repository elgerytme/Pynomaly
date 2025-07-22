"""
Pytest configuration for anomaly detection package.
Provides fixtures for detection testing, benchmark datasets, and performance validation.
"""

import numpy as np
import pandas as pd
import pytest
from typing import Dict, Any, Tuple
import time
from pathlib import Path


@pytest.fixture
def synthetic_anomaly_dataset() -> Tuple[np.ndarray, np.ndarray]:
    """Standard synthetic anomaly dataset for testing."""
    np.random.seed(42)
    
    n_normal = 900
    n_anomalies = 100
    n_features = 10
    
    # Normal data
    normal_data = np.random.multivariate_normal(
        mean=np.zeros(n_features),
        cov=np.eye(n_features),
        size=n_normal
    )
    
    # Anomalous data
    anomaly_data = np.random.multivariate_normal(
        mean=np.ones(n_features) * 3,
        cov=np.eye(n_features) * 2,
        size=n_anomalies
    )
    
    X = np.vstack([normal_data, anomaly_data])
    y = np.hstack([np.ones(n_normal), -np.ones(n_anomalies)])
    
    # Shuffle
    indices = np.random.permutation(len(X))
    return X[indices], y[indices]


@pytest.fixture
def detection_config() -> Dict[str, Any]:
    """Standard detection configuration."""
    return {
        'contamination': 0.1,
        'algorithm_params': {
            'n_estimators': 100,
            'max_samples': 'auto',
            'random_state': 42
        },
        'performance_mode': 'balanced'
    }


@pytest.fixture
def performance_timer():
    """Timer for performance testing."""
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
            
        def start(self):
            self.start_time = time.perf_counter()
            
        def stop(self):
            self.end_time = time.perf_counter()
            
        @property
        def elapsed(self) -> float:
            return (self.end_time or time.perf_counter()) - (self.start_time or 0)
    
    return Timer()


def pytest_configure(config):
    """Configure pytest markers."""
    markers = [
        "accuracy: Algorithm accuracy tests",
        "performance: Performance benchmark tests", 
        "real_time: Real-time detection tests",
        "edge_cases: Edge case handling tests",
        "integration: Integration tests"
    ]
    
    for marker in markers:
        config.addinivalue_line("markers", marker)