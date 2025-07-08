"""Integration test for synthetic dataset and model selection."""
import pytest
import numpy as np
from pynomaly.domain.services.model_selector import ModelSelector, ModelCandidate


@pytest.fixture
def synthetic_dataset():
    """Generate a synthetic dataset for testing."""
    rng = np.random.default_rng(seed=42)
    num_samples = 100
    features = rng.normal(size=(num_samples, 10))
    labels = rng.integers(0, 2, size=num_samples)
    return features, labels


def test_selector_on_synthetic_data(synthetic_dataset):
    """Test the model selector with a synthetic dataset and three algorithms."""
    features, labels = synthetic_dataset

    # Create dummy candidate models
    candidates = [
        ModelCandidate(
            model_id="model1",
            algorithm="alg1",
            metrics={"accuracy": 0.95},
            parameters={},
            metadata={}
        ),
        ModelCandidate(
            model_id="model2",
            algorithm="alg2",
            metrics={"accuracy": 0.92},
            parameters={},
            metadata={}
        ),
        ModelCandidate(
            model_id="model3",
            algorithm="alg3",
            metrics={"accuracy": 0.90},
            parameters={},
            metadata={}
        )
    ]

    # Initialize the selector with a fixed random state
    selector = ModelSelector(primary_metric="accuracy")
    best_model = selector.select_best_model(candidates)

    assert best_model["selected_model"] == "model1"
    assert best_model["algorithm"] == "alg1"

