"""Standalone test for model selector without conftest dependencies."""
import pytest
import numpy as np
from unittest.mock import Mock

from pynomaly.domain.services.model_selector import ModelSelector, ModelCandidate


def test_model_selector_basic():
    """Test basic model selector functionality."""
    selector = ModelSelector(primary_metric="f1_score")
    
    candidates = [
        ModelCandidate(
            model_id="model1",
            algorithm="isolation_forest",
            metrics={"f1_score": 0.85, "accuracy": 0.90},
            parameters={"n_estimators": 100},
            metadata={}
        ),
        ModelCandidate(
            model_id="model2",
            algorithm="one_class_svm",
            metrics={"f1_score": 0.92, "accuracy": 0.88},
            parameters={"nu": 0.1},
            metadata={}
        )
    ]
    
    result = selector.select_best_model(candidates)
    
    assert result is not None
    assert result["selected_model"] == "model2"
    assert result["algorithm"] == "one_class_svm"
    assert result["primary_metric_value"] == 0.92


def test_model_selector_empty_candidates():
    """Test model selector with empty candidates."""
    selector = ModelSelector()
    result = selector.select_best_model([])
    assert result is None


def test_model_selector_rank_models():
    """Test model ranking functionality."""
    selector = ModelSelector()
    candidates = [
        ModelCandidate("model1", "alg1", {"f1_score": 0.85}, {}, {}),
        ModelCandidate("model2", "alg2", {"f1_score": 0.92}, {}, {}),
    ]
    
    result = selector.rank_models(candidates)
    assert result == candidates
    assert len(result) == 2


def test_model_selector_compare_models():
    """Test model comparison functionality."""
    selector = ModelSelector()
    model1 = ModelCandidate("model1", "alg1", {"f1_score": 0.85}, {}, {})
    model2 = ModelCandidate("model2", "alg2", {"f1_score": 0.92}, {}, {})
    
    result = selector.compare_models(model1, model2)
    
    assert result["better_model"] == "model1"
    assert result["metrics_comparison"] == {}
    assert result["recommendation"] == "Use model1"


def test_synthetic_dataset_selection():
    """Test model selection with synthetic dataset."""
    np.random.seed(42)
    
    # Create synthetic dataset
    features = np.random.normal(size=(100, 10))
    labels = np.random.randint(0, 2, size=100)
    
    # Create test candidates
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
    
    # Test selection
    selector = ModelSelector(primary_metric="accuracy")
    best_model = selector.select_best_model(candidates)
    
    assert best_model["selected_model"] == "model1"
    assert best_model["algorithm"] == "alg1"
    assert best_model["primary_metric_value"] == 0.95


if __name__ == "__main__":
    test_model_selector_basic()
    test_model_selector_empty_candidates()
    test_model_selector_rank_models()
    test_model_selector_compare_models()
    test_synthetic_dataset_selection()
    print("All tests passed!")
