"""Pytest configuration and fixtures."""

from __future__ import annotations

import asyncio
from typing import Generator
from uuid import uuid4

import pandas as pd
import pytest
from dependency_injector import providers

from pynomaly.domain.entities import Dataset, Detector
from pynomaly.infrastructure.config import Container, Settings

# Import dependency management
from .conftest_dependencies import (
    requires_dependency, 
    requires_dependencies, 
    requires_core_dependencies,
    get_dependency_report,
    print_dependency_status,
    AVAILABLE_DEPENDENCIES
)

# Import database fixtures if available
try:
    from .conftest_database import (
        test_database_url,
        test_database_settings, 
        test_database_manager,
        test_container_with_database,
        test_async_database_repositories
    )
except ImportError:
    # Database fixtures not available, create skip fixtures
    @pytest.fixture
    def test_async_database_repositories():
        pytest.skip("Database dependencies not available")
    
    @pytest.fixture  
    def test_database_manager():
        pytest.skip("Database dependencies not available")


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def settings() -> Settings:
    """Test settings."""
    return Settings(
        debug=True,
        environment="test",
        storage_path="/tmp/pynomaly_test",
        log_level="DEBUG"
    )


@pytest.fixture
def container(settings: Settings) -> Container:
    """Test DI container."""
    container = Container()
    container.config.override(providers.Singleton(lambda: settings))
    return container


@pytest.fixture
def sample_dataset() -> Dataset:
    """Sample dataset for testing."""
    # Generate synthetic data
    import numpy as np
    
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
    df = pd.DataFrame(
        data,
        columns=[f"feature_{i}" for i in range(n_features)]
    )
    df["label"] = labels
    
    return Dataset(
        name="Test Dataset",
        data=df.drop(columns=["label"]),
        target_column="label",
        metadata={"test": True}
    )


@pytest.fixture
def sample_detector() -> Detector:
    """Sample detector for testing."""
    return Detector(
        name="Test Detector",
        algorithm="IsolationForest",
        parameters={"contamination": 0.05, "random_state": 42}
    )


@pytest.fixture
def trained_detector(sample_detector: Detector, sample_dataset: Dataset, container: Container) -> Detector:
    """Trained detector for testing."""
    # Get adapter and train
    adapter = container.pyod_adapter()
    model = adapter.create_model(sample_detector.algorithm, sample_detector.parameters)
    adapter.fit(model, sample_dataset.data)
    
    # Update detector
    sample_detector.is_fitted = True
    sample_detector.fitted_model = model
    sample_detector.metadata["training_samples"] = len(sample_dataset.data)
    
    return sample_detector