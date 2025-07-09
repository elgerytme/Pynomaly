"""Pytest configuration and fixtures for comprehensive testing."""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
import tempfile
from typing import AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pandas as pd
import pytest

# Basic imports that should be available
try:
    from pynomaly.domain.entities import Dataset, DetectionResult, Detector
    from pynomaly.domain.value_objects import AnomalyScore
except ImportError:
    # Create dummy classes if not available
    class Dataset:
        pass
    class DetectionResult:
        pass
    class Detector:
        pass
    class AnomalyScore:
        pass

# Optional imports for advanced features
try:
    from pynomaly.infrastructure.observability import setup_observability
    OBSERVABILITY_AVAILABLE = True
except ImportError:
    OBSERVABILITY_AVAILABLE = False
    def setup_observability(*args, **kwargs):
        return {}

try:
    from pynomaly.infrastructure.security.audit_logging import init_audit_logging
    AUDIT_LOGGING_AVAILABLE = True
except ImportError:
    AUDIT_LOGGING_AVAILABLE = False
    def init_audit_logging(*args, **kwargs):
        return None

# Import database fixtures if available
try:
    from .conftest_database import (
        test_async_database_repositories,
        test_container_with_database,
        test_database_manager,
        test_database_settings,
        test_database_url,
    )
except ImportError:
    # Database fixtures not available, create skip fixtures
    @pytest.fixture
    def test_async_database_repositories():
        pytest.skip("Database dependencies not available")

    @pytest.fixture
    def test_database_manager():
        pytest.skip("Database dependencies not available")


# Import app if available for API testing
try:
    from pynomaly.presentation.api.app import create_app
    APP_AVAILABLE = True
except ImportError:
    APP_AVAILABLE = False


# Import typer for CLI testing
try:
    import typer
    from typer.testing import CliRunner
    TYPER_AVAILABLE = True
except ImportError:
    TYPER_AVAILABLE = False


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_dataset():
    """Create a sample dataset for testing."""
    np.random.seed(42)
    data = np.random.randn(100, 5)
    return Dataset(data=data, name="test_dataset")


@pytest.fixture
def sample_detector():
    """Create a sample detector for testing."""
    return Detector(name="test_detector", algorithm="isolation_forest")


@pytest.fixture
def sample_detection_result():
    """Create a sample detection result for testing."""
    return DetectionResult(
        detector_id="test_detector",
        dataset_id="test_dataset",
        anomaly_scores=[0.1, 0.2, 0.9, 0.3],
        is_anomaly=[False, False, True, False]
    )


@pytest.fixture
def temp_directory():
    """Create a temporary directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_async_database():
    """Mock async database for testing."""
    return AsyncMock()


@pytest.fixture
def mock_detector_service():
    """Mock detector service for testing."""
    return MagicMock()


@pytest.fixture
def mock_dataset_service():
    """Mock dataset service for testing."""
    return MagicMock()


@pytest.fixture
def mock_detection_service():
    """Mock detection service for testing."""
    return MagicMock()


# Additional fixtures for testing
@pytest.fixture
def cli_runner():
    """CLI runner for testing Typer applications."""
    if TYPER_AVAILABLE:
        return CliRunner()
    else:
        pytest.skip("Typer not available")


@pytest.fixture
def test_client():
    """Test client for API testing."""
    if APP_AVAILABLE:
        from fastapi.testclient import TestClient
        app = create_app()
        return TestClient(app)
    else:
        pytest.skip("FastAPI app not available")


# Test data fixtures
@pytest.fixture
def anomaly_data():
    """Generate synthetic anomaly data for testing."""
    np.random.seed(42)
    normal_data = np.random.randn(900, 3)
    anomaly_data = np.random.randn(100, 3) + 3  # Shifted anomalies
    return np.vstack([normal_data, anomaly_data])


@pytest.fixture
def time_series_data():
    """Generate time series data for testing."""
    np.random.seed(42)
    time_points = np.linspace(0, 10, 1000)
    normal_pattern = np.sin(time_points) + 0.1 * np.random.randn(1000)
    # Add some anomalies
    anomaly_indices = [100, 200, 300, 400, 500]
    for idx in anomaly_indices:
        normal_pattern[idx] += 3 * np.random.randn()
    return time_points, normal_pattern


@pytest.fixture
def high_dimensional_data():
    """Generate high-dimensional data for testing."""
    np.random.seed(42)
    return np.random.randn(200, 50)


@pytest.fixture
def categorical_data():
    """Generate categorical data for testing."""
    np.random.seed(42)
    categories = ['A', 'B', 'C', 'D']
    return pd.DataFrame({
        'category': np.random.choice(categories, 100),
        'value': np.random.randn(100)
    })
