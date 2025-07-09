"""Consolidated pytest configuration for stable testing."""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Generator
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pandas as pd
import pytest

# Add src to Python path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


# Core fixtures
@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()


@pytest.fixture(scope="function")
def temp_dir() -> Generator[Path, None, None]:
    """Create temporary directory for testing."""
    temp_path = Path(tempfile.mkdtemp())
    try:
        yield temp_path
    finally:
        shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture(scope="function")
def sample_data() -> pd.DataFrame:
    """Create deterministic sample dataset."""
    np.random.seed(42)  # Fixed seed for reproducibility
    n_samples = 100
    n_features = 3
    
    # Generate normal data
    normal_data = np.random.normal(0, 1, (n_samples - 10, n_features))
    
    # Generate anomalous data
    anomalous_data = np.random.normal(3, 1, (10, n_features))
    
    # Combine data
    data = np.vstack([normal_data, anomalous_data])
    
    return pd.DataFrame(data, columns=[f"feature_{i}" for i in range(n_features)])


@pytest.fixture(scope="function")
def sample_dataset(sample_data):
    """Create sample dataset entity."""
    try:
        from pynomaly.domain.entities import Dataset
        return Dataset(name="test_dataset", data=sample_data)
    except ImportError:
        # Return simple mock if entity not available
        mock_dataset = MagicMock()
        mock_dataset.name = "test_dataset"
        mock_dataset.data = sample_data
        return mock_dataset


@pytest.fixture(scope="function")
def mock_detector():
    """Create mock detector."""
    mock = MagicMock()
    mock.id = "test-detector-id"
    mock.name = "Test Detector"
    mock.algorithm_name = "IsolationForest"
    mock.is_fitted = True
    mock.parameters = {"contamination": 0.1, "random_state": 42}
    return mock


@pytest.fixture(scope="function")
def mock_async_repository():
    """Create mock async repository."""
    mock = AsyncMock()
    mock.save.return_value = None
    mock.find_by_id.return_value = None
    return mock


@pytest.fixture(scope="function")
def mock_sync_repository():
    """Create mock sync repository."""
    mock = MagicMock()
    mock.save.return_value = None
    mock.find_by_id.return_value = None
    return mock


# Test isolation
@pytest.fixture(autouse=True)
def isolate_tests():
    """Isolate tests by cleaning up state."""
    # Before test
    original_env = os.environ.copy()
    
    yield
    
    # After test - restore environment
    os.environ.clear()
    os.environ.update(original_env)
    
    # Reset random seed
    np.random.seed(None)


# Error handling
@pytest.fixture(scope="function")
def suppress_warnings():
    """Suppress common warnings during tests."""
    import warnings
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        warnings.simplefilter("ignore", DeprecationWarning)
        warnings.simplefilter("ignore", PendingDeprecationWarning)
        yield


# Test markers
def pytest_configure(config):
    """Configure pytest with essential markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "slow: Slow tests")
    config.addinivalue_line("markers", "flaky: Potentially flaky tests")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to handle flaky tests."""
    # Add timeout to potentially flaky tests
    for item in items:
        if "integration" in item.keywords or "slow" in item.keywords:
            item.add_marker(pytest.mark.timeout(60))
        elif "flaky" in item.keywords:
            item.add_marker(pytest.mark.timeout(30))
        else:
            item.add_marker(pytest.mark.timeout(10))


# Retry mechanism for flaky tests
@pytest.fixture(scope="function")
def retry_on_failure():
    """Provide retry mechanism for flaky operations."""
    def retry(func, max_attempts=3, delay=0.1):
        import time
        
        for attempt in range(max_attempts):
            try:
                return func()
            except Exception as e:
                if attempt == max_attempts - 1:
                    raise
                time.sleep(delay * (2 ** attempt))  # Exponential backoff
    
    return retry


# Resource management
@pytest.fixture(scope="function")
def resource_manager():
    """Manage test resources and cleanup."""
    resources = []
    
    def add_resource(resource):
        resources.append(resource)
        return resource
    
    yield add_resource
    
    # Cleanup resources
    for resource in reversed(resources):
        try:
            if hasattr(resource, 'close'):
                resource.close()
            elif hasattr(resource, 'cleanup'):
                resource.cleanup()
        except Exception:
            pass  # Ignore cleanup errors


# Performance testing
@pytest.fixture(scope="function")
def performance_timer():
    """Timer for performance testing."""
    import time
    
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
        
        def __enter__(self):
            self.start_time = time.perf_counter()
            return self
        
        def __exit__(self, *args):
            self.end_time = time.perf_counter()
        
        @property
        def elapsed(self):
            if self.start_time is None:
                return 0
            end = self.end_time if self.end_time is not None else time.perf_counter()
            return end - self.start_time
    
    return Timer


# Deterministic test data
@pytest.fixture(scope="session")
def deterministic_data():
    """Create deterministic test data for reproducible tests."""
    np.random.seed(42)
    return {
        'small': np.random.normal(0, 1, (50, 3)),
        'medium': np.random.normal(0, 1, (500, 5)),
        'large': np.random.normal(0, 1, (1000, 10))
    }


# Skip decorators for missing dependencies
def skip_if_no_torch():
    """Skip test if PyTorch not available."""
    try:
        import torch
        return pytest.mark.skipif(False, reason="")
    except ImportError:
        return pytest.mark.skipif(True, reason="PyTorch not available")


def skip_if_no_tensorflow():
    """Skip test if TensorFlow not available."""
    try:
        import tensorflow
        return pytest.mark.skipif(False, reason="")
    except ImportError:
        return pytest.mark.skipif(True, reason="TensorFlow not available")


def skip_if_no_fastapi():
    """Skip test if FastAPI not available."""
    try:
        import fastapi
        return pytest.mark.skipif(False, reason="")
    except ImportError:
        return pytest.mark.skipif(True, reason="FastAPI not available")


# Export commonly used fixtures and utilities
__all__ = [
    'event_loop',
    'temp_dir',
    'sample_data',
    'sample_dataset',
    'mock_detector',
    'mock_async_repository',
    'mock_sync_repository',
    'isolate_tests',
    'suppress_warnings',
    'retry_on_failure',
    'resource_manager',
    'performance_timer',
    'deterministic_data',
    'skip_if_no_torch',
    'skip_if_no_tensorflow',
    'skip_if_no_fastapi',
]