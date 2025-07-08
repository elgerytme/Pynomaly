"""Simplified pytest configuration and fixtures."""

from __future__ import annotations

import asyncio
import pandas as pd
import pytest
import numpy as np

# Simple settings for testing
@pytest.fixture
def test_settings():
    """Simple test settings."""
    return {
        "debug": True,
        "environment": "test",
        "storage_path": "/tmp/pynomaly_test",
        "log_level": "DEBUG",
    }

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    np.random.seed(42)
    n_samples = 1000
    n_features = 5

    # Normal data
    normal_data = np.random.randn(n_samples - 50, n_features)

    # Anomalies (5% contamination)
    anomalies = np.random.randn(50, n_features) * 3 + 5

    # Combine
    data = np.vstack([normal_data, anomalies])
    labels = np.array([0] * (n_samples - 50) + [1] * 50)

    # Create DataFrame
    df = pd.DataFrame(data, columns=[f"feature_{i}" for i in range(n_features)])
    df["label"] = labels

    return df
