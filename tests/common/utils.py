"""Common test utilities to reduce duplication across test files.

This module contains frequently used helper functions, fixtures, and utilities
that are shared across multiple test files to avoid code duplication.
"""

import asyncio
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest

# Try to import domain entities - use try/except for optional imports
try:
    from pynomaly.domain.entities import Dataset, DetectionResult, Detector
    from pynomaly.domain.value_objects import AnomalyScore
    DOMAIN_ENTITIES_AVAILABLE = True
except ImportError:
    DOMAIN_ENTITIES_AVAILABLE = False


class TestDataGenerator:
    """Generate standardized test data for different testing scenarios."""

    def __init__(self, random_state: int = 42):
        """Initialize test data generator.

        Args:
            random_state: Random seed for reproducible data generation
        """
        self.random_state = random_state
        np.random.seed(random_state)

    def generate_simple_dataset(
        self, n_samples: int = 1000, n_features: int = 10, contamination: float = 0.1
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """Generate a simple dataset with known anomalies.

        Args:
            n_samples: Number of samples
            n_features: Number of features
            contamination: Fraction of samples that are anomalies

        Returns:
            Tuple of (DataFrame, labels) where labels indicate anomalies
        """
        n_normal = int(n_samples * (1 - contamination))
        n_anomalies = n_samples - n_normal

        # Generate normal data
        normal_data = np.random.multivariate_normal(
            mean=np.zeros(n_features), cov=np.eye(n_features), size=n_normal
        )

        # Generate anomalous data
        anomaly_data = np.random.multivariate_normal(
            mean=np.full(n_features, 3.0),  # Shifted mean
            cov=np.eye(n_features) * 2.0,  # Different covariance
            size=n_anomalies,
        )

        # Combine and shuffle
        X = np.vstack([normal_data, anomaly_data])
        y = np.hstack([np.zeros(n_normal), np.ones(n_anomalies)])

        # Shuffle
        indices = np.random.permutation(n_samples)
        X = X[indices]
        y = y[indices]

        # Create DataFrame
        columns = [f"feature_{i}" for i in range(n_features)]
        df = pd.DataFrame(X, columns=columns)

        return df, y

    def generate_time_series_dataset(
        self,
        n_timestamps: int = 1000,
        n_features: int = 5,
        anomaly_periods: list[tuple[int, int]] | None = None,
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """Generate time series data with temporal anomalies.

        Args:
            n_timestamps: Number of time points
            n_features: Number of features
            anomaly_periods: List of (start, end) tuples for anomaly periods

        Returns:
            Tuple of (DataFrame with timestamp, labels)
        """
        if anomaly_periods is None:
            anomaly_periods = [(200, 250), (700, 750)]

        # Generate base time series
        timestamps = pd.date_range(start="2023-01-01", periods=n_timestamps, freq="1H")

        # Generate normal seasonal pattern
        t = np.arange(n_timestamps)
        data = np.zeros((n_timestamps, n_features))

        for i in range(n_features):
            # Base trend
            trend = 0.001 * t * (i + 1)

            # Seasonal patterns
            daily_pattern = np.sin(2 * np.pi * t / 24) * (i + 1)
            weekly_pattern = np.sin(2 * np.pi * t / (24 * 7)) * 0.5 * (i + 1)

            # Noise
            noise = np.random.normal(0, 0.1, n_timestamps)

            data[:, i] = trend + daily_pattern + weekly_pattern + noise

        # Add anomalies
        labels = np.zeros(n_timestamps)
        for start, end in anomaly_periods:
            # Add spikes or drops
            for i in range(n_features):
                if np.random.random() > 0.5:
                    # Spike
                    data[start:end, i] += np.random.uniform(2, 5) * (i + 1)
                else:
                    # Drop
                    data[start:end, i] -= np.random.uniform(1, 3) * (i + 1)

            labels[start:end] = 1

        # Create DataFrame
        columns = [f"sensor_{i}" for i in range(n_features)]
        df = pd.DataFrame(data, columns=columns, index=timestamps)
        df["timestamp"] = timestamps

        return df, labels

    def generate_mixed_type_dataset(
        self, n_samples: int = 1000, contamination: float = 0.1
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """Generate dataset with mixed data types (numeric, categorical, text).

        Args:
            n_samples: Number of samples
            contamination: Fraction of anomalous samples

        Returns:
            Tuple of (DataFrame, labels)
        """
        n_normal = int(n_samples * (1 - contamination))
        n_anomalies = n_samples - n_normal

        # Numerical features
        normal_numerical = np.random.randn(n_normal, 3)
        anomaly_numerical = np.random.randn(n_anomalies, 3) * 3 + 2

        # Categorical features
        normal_categories = np.random.choice(["A", "B", "C"], size=n_normal, p=[0.5, 0.3, 0.2])
        anomaly_categories = np.random.choice(["D", "E"], size=n_anomalies)  # Different categories

        # Combine data
        X_num = np.vstack([normal_numerical, anomaly_numerical])
        X_cat = np.hstack([normal_categories, anomaly_categories])
        y = np.hstack([np.zeros(n_normal), np.ones(n_anomalies)])

        # Shuffle
        indices = np.random.permutation(n_samples)
        X_num = X_num[indices]
        X_cat = X_cat[indices]
        y = y[indices]

        # Create DataFrame
        df = pd.DataFrame(X_num, columns=["num_feature_1", "num_feature_2", "num_feature_3"])
        df["category"] = X_cat

        return df, y


class MockFactory:
    """Factory for creating common mock objects used in tests."""

    @staticmethod
    def create_mock_repository() -> Mock:
        """Create a mock repository with common methods."""
        repo = Mock()
        repo.get = AsyncMock()
        repo.save = AsyncMock()
        repo.update = AsyncMock()
        repo.delete = AsyncMock()
        repo.list = AsyncMock(return_value=[])
        return repo

    @staticmethod
    def create_mock_detector(detector_id: str = None) -> Mock:
        """Create a mock detector object."""
        detector = Mock()
        detector.id = detector_id or str(uuid4())
        detector.name = "Test Detector"
        detector.algorithm = "IsolationForest"
        detector.parameters = {"contamination": 0.1}
        detector.created_at = datetime.utcnow()
        return detector

    @staticmethod
    def create_mock_dataset(dataset_id: str = None, data: pd.DataFrame = None) -> Mock:
        """Create a mock dataset object."""
        dataset = Mock()
        dataset.id = dataset_id or str(uuid4())
        dataset.name = "Test Dataset"
        dataset.description = "Test dataset for testing"

        if data is not None:
            dataset.data = data
            dataset.features = list(data.columns)
        else:
            # Default small dataset
            default_data = pd.DataFrame({
                "feature_1": np.random.randn(100),
                "feature_2": np.random.randn(100),
                "feature_3": np.random.randn(100),
            })
            dataset.data = default_data
            dataset.features = ["feature_1", "feature_2", "feature_3"]

        dataset.created_at = datetime.utcnow()
        return dataset

    @staticmethod
    def create_mock_service() -> Mock:
        """Create a mock service with common async methods."""
        service = Mock()
        service.process = AsyncMock()
        service.validate = AsyncMock(return_value=True)
        service.configure = AsyncMock()
        return service


class TemporaryStorageManager:
    """Manager for temporary storage used in tests."""

    def __init__(self):
        self._temp_dirs = []

    def create_temp_directory(self) -> Path:
        """Create a temporary directory and track it for cleanup."""
        temp_dir = Path(tempfile.mkdtemp())
        self._temp_dirs.append(temp_dir)
        return temp_dir

    def cleanup_all(self):
        """Clean up all created temporary directories."""
        import shutil
        for temp_dir in self._temp_dirs:
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
        self._temp_dirs.clear()


class AsyncTestHelper:
    """Helper utilities for async testing."""

    @staticmethod
    def run_async(coro):
        """Run async coroutine in sync context."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)

    @staticmethod
    async def wait_for_condition(condition_func, timeout: float = 5.0, check_interval: float = 0.1):
        """Wait for a condition to become true with timeout."""
        start_time = asyncio.get_event_loop().time()
        while asyncio.get_event_loop().time() - start_time < timeout:
            if condition_func():
                return True
            await asyncio.sleep(check_interval)
        return False


class TestAssertions:
    """Common assertion helpers for tests."""

    @staticmethod
    def assert_dataframe_equal(df1: pd.DataFrame, df2: pd.DataFrame, check_dtype: bool = True):
        """Assert that two DataFrames are equal with detailed error messages."""
        try:
            pd.testing.assert_frame_equal(df1, df2, check_dtype=check_dtype)
        except AssertionError as e:
            raise AssertionError(f"DataFrames are not equal: {str(e)}")

    @staticmethod
    def assert_array_almost_equal(arr1: np.ndarray, arr2: np.ndarray, decimal: int = 7):
        """Assert that two arrays are almost equal."""
        np.testing.assert_array_almost_equal(arr1, arr2, decimal=decimal)

    @staticmethod
    def assert_score_in_range(score: float, min_val: float = 0.0, max_val: float = 1.0):
        """Assert that a score is within the expected range."""
        assert min_val <= score <= max_val, f"Score {score} not in range [{min_val}, {max_val}]"

    @staticmethod
    def assert_async_mock_called_with(mock: AsyncMock, *args, **kwargs):
        """Assert that an AsyncMock was called with specific arguments."""
        mock.assert_called_with(*args, **kwargs)


class ConfigurationHelper:
    """Helper for creating common test configurations."""

    @staticmethod
    def create_test_settings(**overrides) -> dict[str, Any]:
        """Create test settings with common defaults."""
        default_settings = {
            "debug": True,
            "environment": "test",
            "storage_path": "/tmp/pynomaly_test",
            "log_level": "DEBUG",
            "database_url": "sqlite:///:memory:",
            "auth_enabled": False,
            "cache_enabled": False,
            "monitoring_enabled": False,
        }
        default_settings.update(overrides)
        return default_settings

    @staticmethod
    def create_algorithm_config(algorithm_name: str = "IsolationForest", **params) -> dict[str, Any]:
        """Create algorithm configuration for testing."""
        default_params = {
            "contamination": 0.1,
            "random_state": 42,
            "n_estimators": 100,
        }
        default_params.update(params)

        return {
            "algorithm": algorithm_name,
            "parameters": default_params,
            "validation": {
                "enabled": True,
                "cross_validation": False,
            }
        }


# Commonly used test fixtures that can be imported
def pytest_configure():
    """Configure pytest with common fixtures."""
    pass


# Global instances for convenience
test_data_generator = TestDataGenerator()
mock_factory = MockFactory()
storage_manager = TemporaryStorageManager()
async_helper = AsyncTestHelper()
assertions = TestAssertions()
config_helper = ConfigurationHelper()


# Common pytest fixtures that can be imported
@pytest.fixture
def temp_storage():
    """Create temporary storage directory."""
    temp_dir = storage_manager.create_temp_directory()
    yield temp_dir
    # Cleanup happens automatically via storage_manager


@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    return test_data_generator.generate_simple_dataset()


@pytest.fixture
def time_series_data():
    """Generate time series data for testing."""
    return test_data_generator.generate_time_series_dataset()


@pytest.fixture
def mixed_data():
    """Generate mixed-type data for testing."""
    return test_data_generator.generate_mixed_type_dataset()


@pytest.fixture
def mock_repository():
    """Create a mock repository."""
    return mock_factory.create_mock_repository()


@pytest.fixture
def mock_detector():
    """Create a mock detector."""
    return mock_factory.create_mock_detector()


@pytest.fixture
def mock_dataset():
    """Create a mock dataset."""
    return mock_factory.create_mock_dataset()


@pytest.fixture
def test_config():
    """Create test configuration."""
    return config_helper.create_test_settings()


# Cleanup function for pytest
def pytest_unconfigure():
    """Cleanup function called when pytest finishes."""
    storage_manager.cleanup_all()
