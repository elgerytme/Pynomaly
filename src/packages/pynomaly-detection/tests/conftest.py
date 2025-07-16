"""Pytest configuration for pynomaly-detection tests."""

import pytest
import numpy as np


@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    # Add some outliers
    X[0:5] += 3
    return X


@pytest.fixture
def small_data():
    """Generate small dataset for quick tests."""
    np.random.seed(42)
    X = np.random.randn(20, 3)
    X[0:2] += 2
    return X


@pytest.fixture
def anomaly_detector():
    """Create a basic anomaly detector."""
    from pynomaly_detection import AnomalyDetector
    return AnomalyDetector()


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "slow: mark test as slow"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )