"""
Unit tests for ModelSelector with comprehensive coverage.
"""
import pytest
from unittest.mock import Mock, patch

from pynomaly.domain.services.model_selector import ModelSelector, ModelCandidate


class TestModelCandidate:
    """Test ModelCandidate dataclass."""
    
    def test_model_candidate_creation(self):
        """Test ModelCandidate creation."""
        candidate = ModelCandidate(
            model_id="test_model",
            algorithm="isolation_forest",
            metrics={"f1_score": 0.85, "accuracy": 0.90},
            parameters={"n_estimators": 100},
            metadata={"training_time": 10.5}
        )
        
        assert candidate.model_id == "test_model"
        assert candidate.algorithm == "isolation_forest"
        assert candidate.metrics["f1_score"] == 0.85
        assert candidate.parameters["n_estimators"] == 100
        assert candidate.metadata["training_time"] == 10.5


class TestModelSelector:
    """Test ModelSelector class."""
    
    def test_init_default_parameters(self):
        """Test ModelSelector initialization with default parameters."""
        selector = ModelSelector()
        
        assert selector.primary_metric == "f1_score"
        assert selector.secondary_metrics == ["accuracy", "precision", "recall"]
    
    def test_init_custom_parameters(self):
        """Test ModelSelector initialization with custom parameters."""
        selector = ModelSelector(
            primary_metric="accuracy",
            secondary_metrics=["precision", "recall"]
        )
        
        assert selector.primary_metric == "accuracy"
        assert selector.secondary_metrics == ["precision", "recall"]
    
    def test_select_best_model_empty_candidates(self):
        """Test select_best_model with empty candidates list."""
        selector = ModelSelector()
        result = selector.select_best_model([])
        
        assert result is None
    
    def test_select_best_model_with_model_candidates(self):
        """Test select_best_model with ModelCandidate objects."""
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
            ),
            ModelCandidate(
                model_id="model3",
                algorithm="local_outlier_factor",
                metrics={"f1_score": 0.78, "accuracy": 0.85},
                parameters={"n_neighbors": 20},
                metadata={}
            )
        ]
        
        result = selector.select_best_model(candidates)
        
        assert result is not None
        assert result["selected_model"] == "model2"
        assert result["algorithm"] == "one_class_svm"
        assert result["primary_metric_value"] == 0.92
        assert result["metrics"]["f1_score"] == 0.92
    
    def test_select_best_model_with_missing_primary_metric(self):
        """Test select_best_model when primary metric is missing."""
        selector = ModelSelector(primary_metric="precision")
        
        candidates = [
            ModelCandidate(
                model_id="model1",
                algorithm="isolation_forest",
                metrics={"f1_score": 0.85, "accuracy": 0.90},
                parameters={},
                metadata={}
            ),
            ModelCandidate(
                model_id="model2",
                algorithm="one_class_svm",
                metrics={"f1_score": 0.92, "accuracy": 0.88},
                parameters={},
                metadata={}
            )
        ]
        
        result = selector.select_best_model(candidates)
        
        # Should select based on 0.0 default value
        assert result is not None
        assert result["primary_metric_value"] == 0.0
    
    def test_select_best_model_with_performance_metrics_objects(self):
        """Test select_best_model with performance metrics objects."""
        selector = ModelSelector(primary_metric="accuracy")
        
        # Mock performance metrics objects
        mock_candidate1 = Mock()
        mock_candidate1.model_id = "model1"
        mock_candidate1.algorithm = "isolation_forest"
        mock_candidate1.metrics = {"accuracy": 0.85, "f1_score": 0.80}
        
        mock_candidate2 = Mock()
        mock_candidate2.model_id = "model2"
        mock_candidate2.algorithm = "one_class_svm"
        mock_candidate2.metrics = {"accuracy": 0.92, "f1_score": 0.88}
        
        candidates = [mock_candidate1, mock_candidate2]
        
        result = selector.select_best_model(candidates)
        
        assert result is not None
        assert result["selected_model"] == "model2"
        assert result["algorithm"] == "one_class_svm"
        assert result["primary_metric_value"] == 0.92
    
    def test_select_best_model_with_performance_metrics_no_model_id(self):
        """Test select_best_model with performance metrics without model_id."""
        selector = ModelSelector(primary_metric="accuracy")
        
        mock_candidate = Mock()
        mock_candidate.algorithm = "isolation_forest"
        mock_candidate.metrics = {"accuracy": 0.85}
        # No model_id attribute
        
        candidates = [mock_candidate]
        
        result = selector.select_best_model(candidates)
        
        assert result is not None
        assert result["selected_model"] is None
        assert result["algorithm"] == "isolation_forest"
        assert result["primary_metric_value"] == 0.85
    
    def test_select_best_model_with_performance_metrics_no_attributes(self):
        """Test select_best_model with performance metrics missing attributes."""
        selector = ModelSelector(primary_metric="accuracy")
        
        mock_candidate = Mock()
        # No attributes
        
        candidates = [mock_candidate]
        
        result = selector.select_best_model(candidates)
        
        assert result is not None
        assert result["selected_model"] is None
        assert result["algorithm"] == "unknown"
        assert result["metrics"] == {}
        assert result["primary_metric_value"] == 0.0
    
    def test_rank_models_returns_as_is(self):
        """Test rank_models returns candidates as-is (dummy implementation)."""
        selector = ModelSelector()
        
        candidates = [
            ModelCandidate("model1", "alg1", {"f1_score": 0.85}, {}, {}),
            ModelCandidate("model2", "alg2", {"f1_score": 0.92}, {}, {}),
        ]
        
        result = selector.rank_models(candidates)
        
        assert result == candidates
        assert len(result) == 2
        assert result[0].model_id == "model1"
        assert result[1].model_id == "model2"
    
    def test_compare_models_returns_fixed_result(self):
        """Test compare_models returns fixed result (dummy implementation)."""
        selector = ModelSelector()
        
        model1 = ModelCandidate("model1", "alg1", {"f1_score": 0.85}, {}, {})
        model2 = ModelCandidate("model2", "alg2", {"f1_score": 0.92}, {}, {})
        
        result = selector.compare_models(model1, model2)
        
        assert result["better_model"] == "model1"
        assert result["metrics_comparison"] == {}
        assert result["recommendation"] == "Use model1"
    
    def test_select_best_model_single_candidate(self):
        """Test select_best_model with single candidate."""
        selector = ModelSelector(primary_metric="f1_score")
        
        candidate = ModelCandidate(
            model_id="single_model",
            algorithm="isolation_forest",
            metrics={"f1_score": 0.85},
            parameters={},
            metadata={}
        )
        
        result = selector.select_best_model([candidate])
        
        assert result["selected_model"] == "single_model"
        assert result["algorithm"] == "isolation_forest"
        assert result["primary_metric_value"] == 0.85
    
    def test_select_best_model_identical_scores(self):
        """Test select_best_model with identical scores."""
        selector = ModelSelector(primary_metric="f1_score")
        
        candidates = [
            ModelCandidate("model1", "alg1", {"f1_score": 0.85}, {}, {}),
            ModelCandidate("model2", "alg2", {"f1_score": 0.85}, {}, {}),
        ]
        
        result = selector.select_best_model(candidates)
        
        # Should return the first one in case of ties
        assert result["selected_model"] == "model1"
        assert result["primary_metric_value"] == 0.85
    
    def test_select_best_model_none_candidates(self):
        """Test select_best_model with None as candidates."""
        selector = ModelSelector()
        result = selector.select_best_model(None)
        
        assert result is None
    
    def test_select_best_model_with_custom_secondary_metrics(self):
        """Test that secondary metrics are properly set but not used in dummy implementation."""
        selector = ModelSelector(
            primary_metric="f1_score",
            secondary_metrics=["custom_metric1", "custom_metric2"]
        )
        
        candidate = ModelCandidate(
            model_id="test_model",
            algorithm="test_alg",
            metrics={"f1_score": 0.85, "custom_metric1": 0.9},
            parameters={},
            metadata={}
        )
        
        result = selector.select_best_model([candidate])
        
        assert result["selected_model"] == "test_model"
        assert selector.secondary_metrics == ["custom_metric1", "custom_metric2"]
