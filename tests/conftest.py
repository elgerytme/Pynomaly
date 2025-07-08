"""Pytest configuration and fixtures for testing."""

import os
import tempfile
from typing import Generator
from unittest.mock import MagicMock

import pandas as pd
import numpy as np
import pytest


# Import typer for CLI testing
try:
    import typer
    from typer.testing import CliRunner
    TYPER_AVAILABLE = True
except ImportError:
    TYPER_AVAILABLE = False


@pytest.fixture(scope="function")
def cli_runner():
    """Create a Typer CLI runner for testing CLI commands."""
    if not TYPER_AVAILABLE:
        pytest.skip("Typer not available")
    return CliRunner()


@pytest.fixture(scope="function")
def pyod_model_stub() -> MagicMock:
    """Provide a PyOD model stub for testing without actual fitting."""
    model_stub = MagicMock()
    model_stub.fit.return_value = None
    model_stub.predict.return_value = np.array([0, 1, 0, 1])
    model_stub.decision_function.return_value = np.array([0.1, 0.9, 0.2, 0.8])
    return model_stub


@pytest.fixture(scope="function")
def sample_data() -> pd.DataFrame:
    """Create sample dataset for testing."""
    np.random.seed(42)
    n_samples = 1000
    n_features = 5
    
    # Generate normal data
    normal_data = np.random.normal(0, 1, (n_samples - 50, n_features))
    
    # Generate anomalous data
    anomalous_data = np.random.normal(3, 1, (50, n_features))
    
    # Combine data
    data = np.vstack([normal_data, anomalous_data])
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=[f"feature_{i}" for i in range(n_features)])
    df["target"] = [0] * (n_samples - 50) + [1] * 50  # 0 = normal, 1 = anomaly
    
    return df


@pytest.fixture(scope="function")
def temp_dir() -> Generator[str, None, None]:
    """Create temporary directory for testing."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    import shutil
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture(scope="function")
def mock_model():
    """Create mock ML model."""
    mock = MagicMock()
    mock.fit.return_value = None
    mock.predict.return_value = np.array([0, 1, 0, 1])
    mock.decision_function.return_value = np.array([0.1, 0.9, 0.2, 0.8])
    return mock
