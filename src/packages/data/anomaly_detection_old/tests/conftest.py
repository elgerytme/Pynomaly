"""Pytest configuration and fixtures for pynomaly_detection tests."""

import pytest
import numpy as np
from typing import Generator, Any


@pytest.fixture
def random_seed() -> None:
    """Set random seed for reproducible tests."""
    np.random.seed(42)


@pytest.fixture
def small_dataset() -> np.ndarray:
    """Small dataset for quick tests."""
    np.random.seed(42)
    data = np.random.randn(100, 5)
    # Add some outliers
    data[:10] += 3
    return data


@pytest.fixture
def medium_dataset() -> np.ndarray:
    """Medium dataset for standard tests."""
    np.random.seed(42)
    data = np.random.randn(500, 10)
    # Add some outliers
    data[:50] += 2
    return data


@pytest.fixture
def large_dataset() -> np.ndarray:
    """Large dataset for performance tests."""
    np.random.seed(42)
    data = np.random.randn(2000, 15)
    # Add some outliers
    data[:200] += 2.5
    return data


@pytest.fixture
def normal_only_dataset() -> np.ndarray:
    """Dataset with only normal data."""
    np.random.seed(42)
    return np.random.randn(200, 8)


@pytest.fixture
def anomaly_only_dataset() -> np.ndarray:
    """Dataset with only anomalous data."""
    np.random.seed(42)
    return np.random.randn(100, 8) + 5


@pytest.fixture
def mixed_dataset() -> tuple[np.ndarray, np.ndarray]:
    """Mixed dataset with clear separation between normal and anomalous."""
    np.random.seed(42)
    normal = np.random.randn(400, 5)
    anomalous = np.random.randn(100, 5) + 4
    combined = np.vstack([normal, anomalous])
    labels = np.hstack([np.zeros(400), np.ones(100)])
    return combined, labels


@pytest.fixture
def high_dimensional_dataset() -> np.ndarray:
    """High-dimensional dataset for testing scalability."""
    np.random.seed(42)
    data = np.random.randn(300, 50)
    # Add some outliers
    data[:30] += 1.5
    return data


@pytest.fixture
def contamination_levels() -> list[float]:
    """Standard contamination levels for testing."""
    return [0.01, 0.05, 0.1, 0.2, 0.3]


@pytest.fixture
def performance_datasets() -> dict[str, np.ndarray]:
    """Datasets of different sizes for performance testing."""
    np.random.seed(42)
    
    return {
        "tiny": np.random.randn(50, 5),
        "small": np.random.randn(200, 5),
        "medium": np.random.randn(1000, 10),
        "large": np.random.randn(5000, 15),
        "xlarge": np.random.randn(10000, 20)
    }


@pytest.fixture
def sklearn_available() -> bool:
    """Check if sklearn is available for testing."""
    try:
        import sklearn
        return True
    except ImportError:
        return False


# Legacy fixtures for backward compatibility
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


def pytest_configure(config: Any) -> None:
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance tests"
    )
    config.addinivalue_line(
        "markers", "memory: marks tests as memory tests"
    )


def pytest_collection_modifyitems(config: Any, items: list[Any]) -> None:
    """Automatically mark slow tests."""
    for item in items:
        if "performance" in item.nodeid or "large_dataset" in item.nodeid:
            item.add_marker(pytest.mark.slow)
        
        if "test_integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
            
        if "test_performance" in item.nodeid:
            item.add_marker(pytest.mark.performance)
            
        if "memory" in item.nodeid:
            item.add_marker(pytest.mark.memory)


# Custom assertions for anomaly detection testing
def assert_valid_predictions(predictions: np.ndarray, data_length: int) -> None:
    """Assert that predictions are valid."""
    assert len(predictions) == data_length
    assert isinstance(predictions, np.ndarray)
    assert predictions.dtype == int
    assert all(p in [0, 1] for p in predictions)


def assert_reasonable_anomaly_count(predictions: np.ndarray, contamination: float, tolerance: float = 0.1) -> None:
    """Assert that anomaly count is reasonable given contamination level."""
    anomaly_count = np.sum(predictions)
    expected_count = int(contamination * len(predictions))
    tolerance_count = int(tolerance * len(predictions))
    
    assert abs(anomaly_count - expected_count) <= tolerance_count


# Register custom assertions
pytest.assert_valid_predictions = assert_valid_predictions
pytest.assert_reasonable_anomaly_count = assert_reasonable_anomaly_count