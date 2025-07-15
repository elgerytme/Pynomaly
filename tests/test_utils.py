"""
Consolidated Test Utilities Library - Issue #106 Implementation

This module provides a centralized library of test utilities, mocks, and helpers
to standardize testing patterns across the entire test suite.
"""

from __future__ import annotations

import asyncio
import tempfile
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pandas as pd
import pytest

T = TypeVar("T")


class TestDataFactory:
    """Factory for creating consistent test data across all tests."""

    @staticmethod
    def create_sample_dataframe(
        n_samples: int = 100,
        n_features: int = 3,
        anomaly_ratio: float = 0.1,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Create deterministic sample dataset with configurable parameters."""
        np.random.seed(seed)

        n_anomalies = int(n_samples * anomaly_ratio)
        n_normal = n_samples - n_anomalies

        # Generate normal data
        normal_data = np.random.normal(0, 1, (n_normal, n_features))

        # Generate anomalous data
        anomalous_data = np.random.normal(3, 1, (n_anomalies, n_features))

        # Combine data
        data = np.vstack([normal_data, anomalous_data])

        # Create DataFrame
        columns = [f"feature_{i}" for i in range(n_features)]
        df = pd.DataFrame(data, columns=columns)

        # Add labels
        labels = [0] * n_normal + [1] * n_anomalies
        df["target"] = labels

        return df

    @staticmethod
    def create_time_series_data(
        n_samples: int = 100,
        n_features: int = 3,
        freq: str = "1H",
        start_date: str = "2024-01-01",
    ) -> pd.DataFrame:
        """Create time series dataset for temporal testing."""
        df = TestDataFactory.create_sample_dataframe(n_samples, n_features)
        df["timestamp"] = pd.date_range(start_date, periods=n_samples, freq=freq)
        return df

    @staticmethod
    def create_csv_file(
        data: pd.DataFrame, suffix: str = ".csv", delete: bool = False
    ) -> str:
        """Create temporary CSV file from DataFrame."""
        temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=delete)
        data.to_csv(temp_file.name, index=False)
        temp_file.close()
        return temp_file.name


class MockFactory:
    """Factory for creating standardized mocks across all tests."""

    @staticmethod
    def create_dataset_mock(
        name: str = "test_dataset", n_samples: int = 100, n_features: int = 3
    ) -> MagicMock:
        """Create standardized dataset mock."""
        mock = MagicMock()
        mock.id = f"{name}_id"
        mock.name = name
        mock.n_samples = n_samples
        mock.n_features = n_features
        mock.shape = (n_samples, n_features)
        mock.feature_names = [f"feature_{i}" for i in range(n_features)]
        mock.has_target = True
        mock.target_column = "target"
        mock.memory_usage = n_samples * n_features * 8  # bytes
        mock.created_at = MagicMock()
        mock.created_at.strftime.return_value = "2024-01-01 00:00"
        mock.to_dict.return_value = {
            "id": mock.id,
            "name": name,
            "n_samples": n_samples,
            "n_features": n_features,
        }
        return mock

    @staticmethod
    def create_detector_mock(
        name: str = "test_detector", algorithm: str = "IsolationForest"
    ) -> MagicMock:
        """Create standardized detector mock."""
        mock = MagicMock()
        mock.id = f"{name}_id"
        mock.name = name
        mock.algorithm_name = algorithm
        mock.is_fitted = True
        mock.parameters = {"contamination": 0.1, "random_state": 42}
        mock.contamination_rate = 0.1
        mock.created_at = MagicMock()
        mock.created_at.strftime.return_value = "2024-01-01 00:00"
        mock.to_dict.return_value = {
            "id": mock.id,
            "name": name,
            "algorithm": algorithm,
            "parameters": mock.parameters,
        }
        return mock

    @staticmethod
    def create_result_mock(
        detector_id: str = "test_detector_id",
        dataset_name: str = "test_dataset",
        n_samples: int = 100,
    ) -> MagicMock:
        """Create standardized detection result mock."""
        mock = MagicMock()
        mock.detector_id = detector_id
        mock.dataset_name = dataset_name
        mock.anomalies = []
        mock.scores = np.random.random(n_samples).tolist()
        mock.labels = [0] * (n_samples - 5) + [1] * 5  # 5 anomalies
        mock.threshold = 0.8
        mock.execution_time_ms = 100.0
        mock.metadata = {"algorithm": "IsolationForest"}
        return mock

    @staticmethod
    def create_repository_mock(async_mode: bool = False) -> MagicMock | AsyncMock:
        """Create standardized repository mock."""
        mock_class = AsyncMock if async_mode else MagicMock
        mock = mock_class()
        mock.save.return_value = True
        mock.find_by_id.return_value = None
        mock.find_by_name.return_value = None
        mock.find_all.return_value = []
        mock.list_all.return_value = []
        mock.delete.return_value = True
        mock.exists.return_value = False
        return mock

    @staticmethod
    def create_service_mock(service_type: str = "generic") -> MagicMock:
        """Create standardized service mock based on type."""
        mock = MagicMock()

        if service_type == "autonomous":
            mock.detect_autonomous.return_value = {
                "autonomous_detection_results": {
                    "success": True,
                    "data_profile": {
                        "samples": 100,
                        "features": 3,
                        "complexity_score": 0.75,
                        "recommended_contamination": 0.1,
                    },
                    "detection_results": {
                        "selected_algorithm": "IsolationForest",
                        "anomalies_found": 2,
                        "execution_time": 1.5,
                    },
                }
            }
        elif service_type == "export":
            mock.export_to_csv.return_value = True
            mock.export_to_json.return_value = True
            mock.get_available_formats.return_value = ["csv", "json", "excel"]
        elif service_type == "training":
            mock.start_training.return_value = {
                "job_id": "test-job",
                "status": "started",
            }
            mock.get_training_status.return_value = {
                "status": "running",
                "progress": 50,
            }

        return mock


class TestIsolation:
    """Utilities for test isolation and cleanup."""

    @staticmethod
    def reset_random_state(seed: int = 42) -> None:
        """Reset random state for deterministic testing."""
        np.random.seed(seed)

    @staticmethod
    def suppress_warnings() -> None:
        """Suppress common warnings during tests."""
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=UserWarning, module="pydantic.*")
        warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.*")

    @staticmethod
    def cleanup_temp_files(file_paths: list[str]) -> None:
        """Clean up temporary files safely."""
        for file_path in file_paths:
            try:
                Path(file_path).unlink(missing_ok=True)
            except (OSError, PermissionError):
                pass  # Ignore cleanup errors


class TestPerformance:
    """Performance testing utilities."""

    @staticmethod
    def time_function(func: Callable[..., T], *args, **kwargs) -> tuple[T, float]:
        """Time function execution and return result with elapsed time."""
        import time

        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_time = time.perf_counter() - start_time
        return result, elapsed_time

    @staticmethod
    def memory_usage() -> int:
        """Get current memory usage in bytes."""
        try:
            import os

            import psutil

            process = psutil.Process(os.getpid())
            return process.memory_info().rss
        except ImportError:
            return 0


class AsyncTestHelper:
    """Helper utilities for async testing."""

    @staticmethod
    def run_async(coro):
        """Run async coroutine in sync context."""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    @staticmethod
    def create_async_mock() -> AsyncMock:
        """Create async mock with common methods."""
        mock = AsyncMock()
        mock.save.return_value = True
        mock.find_by_id.return_value = None
        mock.delete.return_value = True
        return mock


class TestMarkers:
    """Test marker utilities and decorators."""

    @staticmethod
    def skip_if_no_dependency(dependency: str):
        """Skip test if dependency not available."""
        try:
            __import__(dependency)
            return pytest.mark.skipif(False, reason="")
        except ImportError:
            return pytest.mark.skipif(True, reason=f"{dependency} not available")

    @staticmethod
    def requires_torch():
        """Mark test as requiring PyTorch."""
        return TestMarkers.skip_if_no_dependency("torch")

    @staticmethod
    def requires_tensorflow():
        """Mark test as requiring TensorFlow."""
        return TestMarkers.skip_if_no_dependency("tensorflow")

    @staticmethod
    def requires_fastapi():
        """Mark test as requiring FastAPI."""
        return TestMarkers.skip_if_no_dependency("fastapi")


class ErrorSimulator:
    """Utilities for simulating various error conditions."""

    @staticmethod
    def file_error(error_type: str = "permission") -> Exception:
        """Create file system error."""
        error_map = {
            "permission": PermissionError("Permission denied"),
            "not_found": FileNotFoundError("File not found"),
            "disk_full": OSError("No space left on device"),
        }
        return error_map.get(error_type, OSError("Generic IO error"))

    @staticmethod
    def network_error(error_type: str = "connection") -> Exception:
        """Create network error."""
        error_map = {
            "connection": ConnectionError("Connection refused"),
            "timeout": TimeoutError("Request timed out"),
            "dns": OSError("Name resolution failed"),
        }
        return error_map.get(error_type, Exception("Generic network error"))


class RetryHelper:
    """Utilities for handling flaky tests and retry logic."""

    @staticmethod
    def retry_on_failure(
        func: Callable[..., T],
        max_attempts: int = 3,
        delay: float = 0.1,
        backoff: float = 2.0,
    ) -> T:
        """Retry function on failure with exponential backoff."""
        import time

        last_exception = None
        for attempt in range(max_attempts):
            try:
                return func()
            except Exception as e:
                last_exception = e
                if attempt == max_attempts - 1:
                    raise
                time.sleep(delay * (backoff**attempt))

        # This should never be reached, but satisfy type checker
        raise last_exception  # type: ignore


# Export commonly used utilities
__all__ = [
    "TestDataFactory",
    "MockFactory",
    "TestIsolation",
    "TestPerformance",
    "AsyncTestHelper",
    "TestMarkers",
    "ErrorSimulator",
    "RetryHelper",
]
