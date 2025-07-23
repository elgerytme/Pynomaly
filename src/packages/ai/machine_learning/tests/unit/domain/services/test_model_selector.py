"""Comprehensive unit tests for ModelSelector domain service."""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from machine_learning.domain.services.model_selector import ModelSelector, ParetoOptimizer


class MockPerformanceMetrics:
    """Mock performance metrics object."""
    
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def to_dict(self):
        return {key: value for key, value in self.__dict__.items()}


class MockModelPerformanceMetrics:
    """Mock model performance metrics entity."""
    
    def __init__(self, model_id, metrics, cv_scores=None):
        self.model_id = model_id
        self.metrics = metrics
        self.cv_scores = cv_scores or {}


@pytest.fixture
def sample_objectives():
    """Sample objectives for Pareto optimization."""
    return [
        {"name": "f1_score", "direction": "maximize"},
        {"name": "precision", "direction": "maximize"},
        {"name": "execution_time", "direction": "minimize"}
    ]


@pytest.fixture
def sample_models_dict_metrics():
    """Sample models with dictionary metrics."""
    return [
        MockModelPerformanceMetrics(
            model_id="model_1",
            metrics={"f1_score": 0.85, "precision": 0.80, "recall": 0.90, "execution_time": 10.0}
        ),
        MockModelPerformanceMetrics(
            model_id="model_2", 
            metrics={"f1_score": 0.82, "precision": 0.88, "recall": 0.77, "execution_time": 5.0}
        ),
        MockModelPerformanceMetrics(
            model_id="model_3",
            metrics={"f1_score": 0.78, "precision": 0.75, "recall": 0.82, "execution_time": 15.0}
        )
    ]


@pytest.fixture
def sample_models_object_metrics():
    """Sample models with object metrics."""
    return [
        MockModelPerformanceMetrics(
            model_id="model_1",
            metrics=MockPerformanceMetrics(f1_score=0.85, precision=0.80, recall=0.90)
        ),
        MockModelPerformanceMetrics(
            model_id="model_2",
            metrics=MockPerformanceMetrics(f1_score=0.82, precision=0.88, recall=0.77)
        )
    ]


@pytest.fixture
def sample_models_with_cv_scores():
    """Sample models with cross-validation scores."""
    return [
        MockModelPerformanceMetrics(
            model_id="model_1",
            metrics={"f1_score": 0.85},
            cv_scores={"f1_score": [0.83, 0.86, 0.87, 0.84, 0.85]}
        ),
        MockModelPerformanceMetrics(
            model_id="model_2",
            metrics={"f1_score": 0.75},
            cv_scores={"f1_score": [0.73, 0.76, 0.77, 0.74, 0.75]}
        )
    ]


class TestParetoOptimizer:
    """Test cases for ParetoOptimizer."""

    def test_initialization(self, sample_objectives):
        """Test ParetoOptimizer initialization."""
        optimizer = ParetoOptimizer(sample_objectives)
        assert optimizer.objectives == sample_objectives

    def test_find_pareto_optimal_empty_list(self, sample_objectives):
        """Test find_pareto_optimal with empty model list."""
        optimizer = ParetoOptimizer(sample_objectives)
        result = optimizer.find_pareto_optimal([])
        assert result == []

    def test_find_pareto_optimal_single_model(self, sample_objectives, sample_models_dict_metrics):
        """Test find_pareto_optimal with single model."""
        optimizer = ParetoOptimizer(sample_objectives)
        result = optimizer.find_pareto_optimal([sample_models_dict_metrics[0]])
        
        assert len(result) == 1
        assert result[0]["model_id"] == "model_1"

    def test_find_pareto_optimal_multiple_models(self, sample_objectives, sample_models_dict_metrics):
        """Test find_pareto_optimal with multiple models."""
        optimizer = ParetoOptimizer(sample_objectives)
        result = optimizer.find_pareto_optimal(sample_models_dict_metrics)
        
        assert isinstance(result, list)
        assert len(result) >= 1
        assert all("model_id" in item for item in result)

    def test_find_pareto_optimal_maximize_objectives(self):
        """Test find_pareto_optimal with maximize objectives."""
        objectives = [
            {"name": "f1_score", "direction": "maximize"},
            {"name": "precision", "direction": "maximize"}
        ]
        optimizer = ParetoOptimizer(objectives)
        
        models = [
            MockModelPerformanceMetrics("model_1", {"f1_score": 0.9, "precision": 0.8}),
            MockModelPerformanceMetrics("model_2", {"f1_score": 0.8, "precision": 0.9}),
            MockModelPerformanceMetrics("model_3", {"f1_score": 0.7, "precision": 0.7})  # Dominated
        ]
        
        result = optimizer.find_pareto_optimal(models)
        pareto_ids = {item["model_id"] for item in result}
        
        # model_3 should be dominated
        assert "model_1" in pareto_ids
        assert "model_2" in pareto_ids
        assert len(pareto_ids) >= 2

    def test_find_pareto_optimal_minimize_objectives(self):
        """Test find_pareto_optimal with minimize objectives."""
        objectives = [
            {"name": "execution_time", "direction": "minimize"},
            {"name": "memory_usage", "direction": "minimize"}
        ]
        optimizer = ParetoOptimizer(objectives)
        
        models = [
            MockModelPerformanceMetrics("model_1", {"execution_time": 5.0, "memory_usage": 100}),
            MockModelPerformanceMetrics("model_2", {"execution_time": 10.0, "memory_usage": 50}),
            MockModelPerformanceMetrics("model_3", {"execution_time": 15.0, "memory_usage": 150})  # Dominated
        ]
        
        result = optimizer.find_pareto_optimal(models)
        pareto_ids = {item["model_id"] for item in result}
        
        # model_3 should be dominated
        assert "model_1" in pareto_ids
        assert "model_2" in pareto_ids
        assert "model_3" not in pareto_ids

    def test_find_pareto_optimal_mixed_objectives(self):
        """Test find_pareto_optimal with mixed maximize/minimize objectives."""
        objectives = [
            {"name": "f1_score", "direction": "maximize"},
            {"name": "execution_time", "direction": "minimize"}
        ]
        optimizer = ParetoOptimizer(objectives)
        
        models = [
            MockModelPerformanceMetrics("model_1", {"f1_score": 0.9, "execution_time": 10.0}),
            MockModelPerformanceMetrics("model_2", {"f1_score": 0.8, "execution_time": 5.0}),
            MockModelPerformanceMetrics("model_3", {"f1_score": 0.7, "execution_time": 15.0})  # Dominated
        ]
        
        result = optimizer.find_pareto_optimal(models)
        pareto_ids = {item["model_id"] for item in result}
        
        assert "model_1" in pareto_ids
        assert "model_2" in pareto_ids
        assert "model_3" not in pareto_ids

    def test_get_metric_value_object_metrics(self, sample_objectives):
        """Test _get_metric_value with object metrics."""
        optimizer = ParetoOptimizer(sample_objectives)
        model = MockModelPerformanceMetrics(
            "model_1", 
            MockPerformanceMetrics(f1_score=0.85, precision=0.80)
        )
        
        assert optimizer._get_metric_value(model, "f1_score") == 0.85
        assert optimizer._get_metric_value(model, "precision") == 0.80

    def test_get_metric_value_dict_metrics(self, sample_objectives):
        """Test _get_metric_value with dictionary metrics."""
        optimizer = ParetoOptimizer(sample_objectives)
        model = MockModelPerformanceMetrics(
            "model_1",
            {"f1_score": 0.85, "precision": 0.80}
        )
        
        assert optimizer._get_metric_value(model, "f1_score") == 0.85
        assert optimizer._get_metric_value(model, "precision") == 0.80

    def test_get_metric_value_missing_metric(self, sample_objectives):
        """Test _get_metric_value with missing metric."""
        optimizer = ParetoOptimizer(sample_objectives)
        model = MockModelPerformanceMetrics("model_1", {"f1_score": 0.85})
        
        assert optimizer._get_metric_value(model, "missing_metric") == 0.0

    def test_get_metric_value_no_metrics(self, sample_objectives):
        """Test _get_metric_value with no metrics."""
        optimizer = ParetoOptimizer(sample_objectives)
        model = MockModelPerformanceMetrics("model_1", None)
        
        assert optimizer._get_metric_value(model, "f1_score") == 0.0


class TestModelSelector:
    """Test cases for ModelSelector."""

    def test_initialization_default(self):
        """Test ModelSelector initialization with defaults."""
        selector = ModelSelector()
        
        assert selector.primary_metric == "f1_score"
        assert selector.secondary_metrics == []
        assert isinstance(selector.pareto_optimizer, ParetoOptimizer)
        assert len(selector.pareto_optimizer.objectives) == 1
        assert selector.pareto_optimizer.objectives[0]["name"] == "f1_score"

    def test_initialization_custom(self):
        """Test ModelSelector initialization with custom parameters."""
        secondary_metrics = ["precision", "recall"]
        selector = ModelSelector(
            primary_metric="accuracy", 
            secondary_metrics=secondary_metrics
        )
        
        assert selector.primary_metric == "accuracy"
        assert selector.secondary_metrics == secondary_metrics
        assert len(selector.pareto_optimizer.objectives) == 3

    @patch('machine_learning.domain.services.model_selector.MetricsCalculator')
    def test_rank_models_empty_list(self, mock_metrics_calc):
        """Test rank_models with empty model list."""
        selector = ModelSelector()
        result = selector.rank_models([])
        
        assert result == []
        mock_metrics_calc.compare_models.assert_not_called()

    @patch('machine_learning.domain.services.model_selector.MetricsCalculator')
    def test_rank_models_success(self, mock_metrics_calc, sample_models_dict_metrics):
        """Test rank_models with successful execution."""
        # Mock MetricsCalculator response
        mock_metrics_calc.compare_models.return_value = {
            "rankings": {
                "f1_score": [
                    {"model": "model_1", "score": 0.85, "rank": 1},
                    {"model": "model_2", "score": 0.82, "rank": 2},
                    {"model": "model_3", "score": 0.78, "rank": 3}
                ]
            }
        }
        
        selector = ModelSelector()
        result = selector.rank_models(sample_models_dict_metrics)
        
        assert isinstance(result, list)
        mock_metrics_calc.compare_models.assert_called_once()
        
        # Check that the call was made with correct arguments
        call_args = mock_metrics_calc.compare_models.call_args
        assert call_args[1]["primary_metric"] == "f1_score"

    def test_significant_difference_no_cv_scores(self, sample_models_dict_metrics):
        """Test significant_difference without cross-validation scores."""
        selector = ModelSelector()
        model1, model2 = sample_models_dict_metrics[0], sample_models_dict_metrics[1]
        
        result = selector.significant_difference(model1, model2)
        assert isinstance(result, bool)

    def test_significant_difference_with_cv_scores(self, sample_models_with_cv_scores):
        """Test significant_difference with cross-validation scores."""
        selector = ModelSelector()
        model1, model2 = sample_models_with_cv_scores[0], sample_models_with_cv_scores[1]
        
        with patch('machine_learning.domain.services.model_selector.ttest_ind') as mock_ttest:
            mock_ttest.return_value = (2.5, 0.02)  # Significant result
            
            result = selector.significant_difference(model1, model2)
            
            assert result is True
            mock_ttest.assert_called_once()

    def test_significant_difference_insufficient_cv_samples(self):
        """Test significant_difference with insufficient CV samples."""
        selector = ModelSelector()
        
        model1 = MockModelPerformanceMetrics(
            "model_1", {"f1_score": 0.85},
            cv_scores={"f1_score": [0.85]}  # Only 1 sample
        )
        model2 = MockModelPerformanceMetrics(
            "model_2", {"f1_score": 0.75},
            cv_scores={"f1_score": [0.75]}  # Only 1 sample
        )
        
        result = selector.significant_difference(model1, model2)
        assert isinstance(result, bool)

    def test_significant_difference_missing_metric(self):
        """Test significant_difference with missing primary metric."""
        selector = ModelSelector(primary_metric="accuracy")
        
        model1 = MockModelPerformanceMetrics("model_1", {"f1_score": 0.85})
        model2 = MockModelPerformanceMetrics("model_2", {"f1_score": 0.75})
        
        with pytest.raises(KeyError, match="Primary metric 'accuracy' not found"):
            selector.significant_difference(model1, model2)

    def test_significant_difference_zero_values_with_metric_present(self):
        """Test significant_difference with zero values but metric present."""
        selector = ModelSelector()
        
        model1 = MockModelPerformanceMetrics("model_1", {"f1_score": 0.0})
        model2 = MockModelPerformanceMetrics("model_2", {"f1_score": 0.0})
        
        result = selector.significant_difference(model1, model2)
        assert result is False

    def test_significant_difference_effect_size_calculation(self):
        """Test significant_difference effect size calculation."""
        selector = ModelSelector()
        
        # Models with large effect size
        model1 = MockModelPerformanceMetrics("model_1", {"f1_score": 0.90})
        model2 = MockModelPerformanceMetrics("model_2", {"f1_score": 0.70})
        
        result = selector.significant_difference(model1, model2)
        assert result is True

    def test_significant_difference_small_effect_size(self):
        """Test significant_difference with small effect size."""
        selector = ModelSelector()
        
        # Models with small effect size
        model1 = MockModelPerformanceMetrics("model_1", {"f1_score": 0.85})
        model2 = MockModelPerformanceMetrics("model_2", {"f1_score": 0.84})
        
        result = selector.significant_difference(model1, model2)
        assert result is False

    def test_select_best_model_empty_list(self):
        """Test select_best_model with empty model list."""
        selector = ModelSelector()
        result = selector.select_best_model([])
        
        assert result["decision"] == "No suitable models found"
        assert result["rationale"] == []

    @patch('machine_learning.domain.services.model_selector.MetricsCalculator')
    def test_select_best_model_success(self, mock_metrics_calc, sample_models_dict_metrics):
        """Test select_best_model with successful selection."""
        mock_metrics_calc.compare_models.return_value = {
            "rankings": {
                "f1_score": [
                    {"model": "model_1", "score": 0.85, "rank": 1},
                    {"model": "model_2", "score": 0.82, "rank": 2}
                ]
            }
        }
        
        selector = ModelSelector()
        
        with patch.object(selector, 'significant_difference', return_value=True):
            result = selector.select_best_model(sample_models_dict_metrics)
        
        assert "selected_model" in result
        assert "rationale" in result
        assert isinstance(result["rationale"], list)
        assert len(result["rationale"]) > 0

    @patch('machine_learning.domain.services.model_selector.MetricsCalculator')
    def test_select_best_model_no_significant_difference(self, mock_metrics_calc, sample_models_dict_metrics):
        """Test select_best_model with no significant differences."""
        mock_metrics_calc.compare_models.return_value = {
            "rankings": {
                "f1_score": [
                    {"model": "model_1", "score": 0.85, "rank": 1},
                    {"model": "model_2", "score": 0.82, "rank": 2}
                ]
            }
        }
        
        selector = ModelSelector()
        
        with patch.object(selector, 'significant_difference', return_value=False):
            result = selector.select_best_model(sample_models_dict_metrics)
        
        assert "selected_model" in result
        assert any("No significant difference detected" in rationale for rationale in result["rationale"])

    def test_convert_model_to_results_object_metrics(self):
        """Test _convert_model_to_results with object metrics."""
        selector = ModelSelector()
        
        mock_metrics = MockPerformanceMetrics(f1_score=0.85, precision=0.80, recall=0.90)
        model = MockModelPerformanceMetrics("model_1", mock_metrics)
        
        result = selector._convert_model_to_results(model)
        
        assert "f1_score" in result
        assert result["f1_score"]["value"] == 0.85
        assert "precision" in result
        assert result["precision"]["value"] == 0.80

    def test_convert_model_to_results_dict_metrics(self):
        """Test _convert_model_to_results with dictionary metrics."""
        selector = ModelSelector()
        
        model = MockModelPerformanceMetrics(
            "model_1", 
            {"f1_score": 0.85, "precision": 0.80, "execution_time": 10.5}
        )
        
        result = selector._convert_model_to_results(model)
        
        assert "f1_score" in result
        assert result["f1_score"]["value"] == 0.85
        assert "precision" in result
        assert result["precision"]["value"] == 0.80
        assert "execution_time" in result
        assert result["execution_time"]["value"] == 10.5

    def test_convert_model_to_results_filters_non_numeric(self):
        """Test _convert_model_to_results filters non-numeric values."""
        selector = ModelSelector()
        
        model = MockModelPerformanceMetrics(
            "model_1",
            {"f1_score": 0.85, "model_name": "RandomForest", "is_trained": True, "tags": ["tag1", "tag2"]}
        )
        
        result = selector._convert_model_to_results(model)
        
        assert "f1_score" in result
        assert "model_name" not in result
        assert "is_trained" not in result
        assert "tags" not in result

    def test_convert_model_to_results_no_metrics(self):
        """Test _convert_model_to_results with no metrics."""
        selector = ModelSelector()
        
        model = MockModelPerformanceMetrics("model_1", None)
        result = selector._convert_model_to_results(model)
        
        assert result == {}

    def test_get_metric_value_object_metrics(self):
        """Test _get_metric_value with object metrics."""
        selector = ModelSelector()
        
        model = MockModelPerformanceMetrics(
            "model_1",
            MockPerformanceMetrics(f1_score=0.85, precision=0.80)
        )
        
        assert selector._get_metric_value(model, "f1_score") == 0.85
        assert selector._get_metric_value(model, "precision") == 0.80
        assert selector._get_metric_value(model, "missing_metric") == 0.0

    def test_get_metric_value_dict_metrics(self):
        """Test _get_metric_value with dictionary metrics."""
        selector = ModelSelector()
        
        model = MockModelPerformanceMetrics(
            "model_1",
            {"f1_score": 0.85, "precision": 0.80}
        )
        
        assert selector._get_metric_value(model, "f1_score") == 0.85
        assert selector._get_metric_value(model, "precision") == 0.80
        assert selector._get_metric_value(model, "missing_metric") == 0.0

    def test_has_metric_object_metrics(self):
        """Test _has_metric with object metrics."""
        selector = ModelSelector()
        
        model = MockModelPerformanceMetrics(
            "model_1",
            MockPerformanceMetrics(f1_score=0.85, precision=0.80)
        )
        
        assert selector._has_metric(model, "f1_score") is True
        assert selector._has_metric(model, "precision") is True
        assert selector._has_metric(model, "missing_metric") is False

    def test_has_metric_dict_metrics(self):
        """Test _has_metric with dictionary metrics."""
        selector = ModelSelector()
        
        model = MockModelPerformanceMetrics(
            "model_1",
            {"f1_score": 0.85, "precision": 0.80}
        )
        
        assert selector._has_metric(model, "f1_score") is True
        assert selector._has_metric(model, "precision") is True
        assert selector._has_metric(model, "missing_metric") is False

    def test_has_metric_no_metrics(self):
        """Test _has_metric with no metrics."""
        selector = ModelSelector()
        
        model = MockModelPerformanceMetrics("model_1", None)
        
        assert selector._has_metric(model, "f1_score") is False

    def test_integration_full_workflow(self, sample_models_dict_metrics):
        """Test full workflow integration."""
        selector = ModelSelector(
            primary_metric="f1_score",
            secondary_metrics=["precision"]
        )
        
        with patch('machine_learning.domain.services.model_selector.MetricsCalculator') as mock_calc:
            mock_calc.compare_models.return_value = {
                "rankings": {
                    "f1_score": [
                        {"model": "model_1", "score": 0.85, "rank": 1},
                        {"model": "model_2", "score": 0.82, "rank": 2},
                        {"model": "model_3", "score": 0.78, "rank": 3}
                    ]
                }
            }
            
            # Test ranking
            rankings = selector.rank_models(sample_models_dict_metrics)
            assert len(rankings) > 0
            
            # Test selection
            selection = selector.select_best_model(sample_models_dict_metrics)
            assert "selected_model" in selection
            assert "rationale" in selection