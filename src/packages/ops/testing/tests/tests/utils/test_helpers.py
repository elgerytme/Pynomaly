"""Standardized test utilities and helpers for common testing patterns."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Callable, TypeVar
from unittest.mock import MagicMock, AsyncMock
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil

T = TypeVar('T')


class TestDataGenerator:
    """Centralized test data generation with consistent patterns."""
    
    @staticmethod
    def sample_dataset(
        n_samples: int = 100,
        n_features: int = 3,
        contamination: float = 0.1,
        random_state: int = 42
    ) -> pd.DataFrame:
        """Generate consistent sample dataset for testing."""
        np.random.seed(random_state)
        
        n_anomalies = int(n_samples * contamination)
        n_normal = n_samples - n_anomalies
        
        # Generate normal data
        normal_data = np.random.normal(0, 1, (n_normal, n_features))
        
        # Generate anomalous data (shifted distribution)
        anomalous_data = np.random.normal(3, 1, (n_anomalies, n_features))
        
        # Combine data
        data = np.vstack([normal_data, anomalous_data])
        
        # Create DataFrame with consistent column names
        columns = [f"feature_{i}" for i in range(n_features)]
        return pd.DataFrame(data, columns=columns)
    
    @staticmethod
    def create_csv_file(data: pd.DataFrame, suffix: str = ".csv") -> str:
        """Create temporary CSV file from DataFrame."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False) as f:
            data.to_csv(f.name, index=False)
            return f.name
    
    @staticmethod
    def create_json_file(data: dict, suffix: str = ".json") -> str:
        """Create temporary JSON file from dictionary."""
        import json
        with tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False) as f:
            json.dump(data, f)
            return f.name


class MockFactory:
    """Factory for creating standardized mocks."""
    
    @staticmethod
    def create_detector_mock(
        algorithm_name: str = "IsolationForest",
        is_fitted: bool = True,
        **kwargs
    ) -> MagicMock:
        """Create standardized detector mock."""
        mock = MagicMock()
        mock.id = kwargs.get("id", "test-detector-id")
        mock.name = kwargs.get("name", "Test Detector")
        mock.algorithm_name = algorithm_name
        mock.is_fitted = is_fitted
        mock.parameters = kwargs.get("parameters", {"contamination": 0.1, "random_state": 42})
        return mock
    
    @staticmethod
    def create_repository_mock(async_repo: bool = False) -> MagicMock | AsyncMock:
        """Create standardized repository mock."""
        if async_repo:
            mock = AsyncMock()
        else:
            mock = MagicMock()
        
        mock.save.return_value = None
        mock.find_by_id.return_value = None
        mock.find_all.return_value = []
        mock.delete.return_value = None
        return mock
    
    @staticmethod
    def create_service_mock(async_service: bool = False) -> MagicMock | AsyncMock:
        """Create standardized service mock."""
        if async_service:
            mock = AsyncMock()
        else:
            mock = MagicMock()
        
        mock.process.return_value = {"status": "success"}
        return mock


class TestResourceManager:
    """Manages test resources and cleanup."""
    
    def __init__(self):
        self.temp_files: list[str] = []
        self.temp_dirs: list[str] = []
        self.resources: list[Any] = []
    
    def add_temp_file(self, file_path: str) -> str:
        """Add temporary file for cleanup."""
        self.temp_files.append(file_path)
        return file_path
    
    def add_temp_dir(self, dir_path: str) -> str:
        """Add temporary directory for cleanup."""
        self.temp_dirs.append(dir_path)
        return dir_path
    
    def add_resource(self, resource: Any) -> Any:
        """Add resource for cleanup."""
        self.resources.append(resource)
        return resource
    
    def cleanup(self):
        """Clean up all managed resources."""
        # Clean up files
        for file_path in self.temp_files:
            try:
                Path(file_path).unlink(missing_ok=True)
            except Exception:
                pass
        
        # Clean up directories
        for dir_path in self.temp_dirs:
            try:
                shutil.rmtree(dir_path, ignore_errors=True)
            except Exception:
                pass
        
        # Clean up resources
        for resource in reversed(self.resources):
            try:
                if hasattr(resource, "close"):
                    resource.close()
                elif hasattr(resource, "cleanup"):
                    resource.cleanup()
            except Exception:
                pass
        
        # Clear lists
        self.temp_files.clear()
        self.temp_dirs.clear()
        self.resources.clear()


class AsyncTestHelper:
    """Helper utilities for async testing."""
    
    @staticmethod
    def run_async(coro):
        """Run async function in sync test."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(coro)
    
    @staticmethod
    async def wait_for_condition(
        condition: Callable[[], bool],
        timeout: float = 5.0,
        interval: float = 0.1
    ) -> bool:
        """Wait for condition to become true."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if condition():
                return True
            await asyncio.sleep(interval)
        return False


class PerformanceTestHelper:
    """Helper utilities for performance testing."""
    
    @staticmethod
    def measure_execution_time(func: Callable[[], T]) -> tuple[T, float]:
        """Measure function execution time."""
        start_time = time.perf_counter()
        result = func()
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        return result, execution_time
    
    @staticmethod
    async def measure_async_execution_time(func: Callable) -> tuple[Any, float]:
        """Measure async function execution time."""
        start_time = time.perf_counter()
        result = await func()
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        return result, execution_time


class RetryHelper:
    """Helper for handling flaky tests."""
    
    @staticmethod
    def retry_on_failure(
        func: Callable[[], T],
        max_attempts: int = 3,
        delay: float = 0.1,
        backoff_multiplier: float = 2.0
    ) -> T:
        """Retry function on failure with exponential backoff."""
        last_exception = None
        
        for attempt in range(max_attempts):
            try:
                return func()
            except Exception as e:
                last_exception = e
                if attempt == max_attempts - 1:
                    break
                time.sleep(delay * (backoff_multiplier ** attempt))
        
        # Re-raise the last exception if all attempts failed
        raise last_exception


# Export all utilities
__all__ = [
    "TestDataGenerator",
    "MockFactory", 
    "TestResourceManager",
    "AsyncTestHelper",
    "PerformanceTestHelper",
    "RetryHelper",
]