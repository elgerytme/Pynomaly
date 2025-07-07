"""Configuration for mutation testing."""

import numpy as np
import pandas as pd
import pytest

from pynomaly.domain.entities import Dataset, Detector
from pynomaly.domain.value_objects import ContaminationRate
from pynomaly.infrastructure.adapters import PyODAdapter
from pynomaly.infrastructure.repositories import InMemoryDatasetRepository


@pytest.fixture
def sample_dataset() -> Dataset:
    """Create a sample dataset for mutation testing."""
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
    return PyODAdapter(
        algorithm_name="IsolationForest",
        contamination_rate=ContaminationRate(0.2),
        n_estimators=10,
        random_state=42,
    )


@pytest.fixture
def dataset_repository() -> InMemoryDatasetRepository:
    """Create an in-memory dataset repository."""
    return InMemoryDatasetRepository()


@pytest.fixture
def fitted_detector(basic_detector: Detector, sample_dataset: Dataset) -> Detector:
    """Create a fitted detector for testing."""
    basic_detector.fit(sample_dataset)
    return basic_detector
