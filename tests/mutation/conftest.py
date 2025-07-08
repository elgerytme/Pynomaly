"""Mutation testing fixtures that extend the root conftest.py."""

# Import specific fixtures from root conftest
# Mutation-specific fixtures
import numpy as np
import pandas as pd
import pytest

# Import specific fixtures instead of all
from ..conftest import container, sample_data, sample_dataset, temp_dir

try:
    from pynomaly.domain.entities import Dataset, Detector
    from pynomaly.domain.value_objects import ContaminationRate
    from pynomaly.infrastructure.adapters import PyODAdapter
    from pynomaly.infrastructure.repositories import InMemoryDatasetRepository

    MUTATION_DEPENDENCIES_AVAILABLE = True
except ImportError:
    MUTATION_DEPENDENCIES_AVAILABLE = False


@pytest.fixture
def mutation_sample_dataset() -> Dataset:
    """Create a sample dataset for mutation testing."""
    if not MUTATION_DEPENDENCIES_AVAILABLE:
        pytest.skip("Mutation testing dependencies not available")

    # Generate synthetic data with known anomalies
    np.random.seed(42)

    # Normal data
    normal_data = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], 80)

    # Anomalous data
    anomaly_data = np.random.multivariate_normal([3, 3], [[0.5, 0], [0, 0.5]], 20)

    # Combine data
    data = np.vstack([normal_data, anomaly_data])
    labels = np.hstack([np.zeros(80), np.ones(20)])

    df = pd.DataFrame(data, columns=["feature_1", "feature_2"])
    df["label"] = labels

    return Dataset(name="mutation_test_dataset", data=df, target_column="label")


@pytest.fixture
def basic_detector() -> Detector:
    """Create a basic detector for mutation testing."""
    if not MUTATION_DEPENDENCIES_AVAILABLE:
        pytest.skip("Mutation testing dependencies not available")

    return PyODAdapter(
        algorithm_name="IsolationForest",
        contamination_rate=ContaminationRate(0.2),
        n_estimators=10,
        random_state=42,
    )
