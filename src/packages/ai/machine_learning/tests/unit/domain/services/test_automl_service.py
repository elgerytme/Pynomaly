"""Comprehensive unit tests for AutoMLService."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, AsyncMock, patch
from dataclasses import dataclass

from machine_learning.domain.services.automl_service import (
    AutoMLService,
    OptimizationConfig,
    OptimizationResult,
    OptimizationMetric,
    SearchStrategy,
    get_automl_service
)


@dataclass
class MockDataset:
    """Mock dataset for testing."""
    data: np.ndarray | pd.DataFrame
    

@dataclass
class MockPredictionResult:
    """Mock prediction result for testing."""
    total_samples: int
    anomaly_count: int
    execution_time: float
    contamination_rate: Mock
    anomalies: list
    metadata: dict


class MockAlgorithmConfig:
    """Mock algorithm configuration."""
    def __init__(self, algorithm, parameters=None, contamination=0.1, random_state=42):
        from machine_learning.domain.services.advanced_prediction_service import PredictionAlgorithm
        self.algorithm = algorithm
        self.parameters = parameters or {}
        self.contamination = contamination
        self.random_state = random_state


class MockPredictionAlgorithm:
    """Mock prediction algorithm enum."""
    ISOLATION_FOREST = "isolation_forest"
    LOCAL_OUTLIER_FACTOR = "local_outlier_factor"
    ONE_CLASS_SVM = "one_class_svm"
    DBSCAN = "dbscan"
    KNN = "knn"
    AUTO_ENCODER = "auto_encoder"
    
    def __init__(self, value):
        self.value = value


@pytest.fixture
def mock_dataset():
    """Create a mock dataset for testing."""
    data = np.random.randn(100, 10)
    return MockDataset(data=data)


@pytest.fixture
def mock_dataframe_dataset():
    """Create a mock DataFrame dataset for testing."""
    data = pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'feature3': np.random.randn(100)
    })
    return MockDataset(data=data)


@pytest.fixture
def mock_prediction_service():
    """Create a mock prediction service."""
    service = Mock()
    service.get_available_algorithms = AsyncMock(return_value=[
        MockPredictionAlgorithm.ISOLATION_FOREST,
        MockPredictionAlgorithm.LOCAL_OUTLIER_FACTOR,
        MockPredictionAlgorithm.ONE_CLASS_SVM
    ])
    
    # Mock prediction result
    result = MockPredictionResult(
        total_samples=100,
        anomaly_count=10,
        execution_time=1.5,
        contamination_rate=Mock(value=0.1),
        anomalies=[],
        metadata={
            'metrics': {
                'f1_score': 0.85,
                'precision': 0.80,
                'recall': 0.90,
                'auc_score': 0.88,
                'execution_time': 1.5,
                'memory_usage': 50.0,
                'anomaly_rate': 0.1
            }
        }
    )
    service.predict = AsyncMock(return_value=result)
    service.predict_ensemble = AsyncMock(return_value=result)
    
    return service


@pytest.fixture
def automl_service(mock_prediction_service):
    """Create AutoML service with mocked dependencies."""
    with patch('machine_learning.domain.services.automl_service.get_prediction_service', 
               return_value=mock_prediction_service):
        service = AutoMLService()
        return service


class TestAutoMLService:
    """Test cases for AutoMLService."""

    def test_initialization(self, automl_service):
        """Test AutoML service initialization."""
        assert automl_service.prediction_service is not None
        assert automl_service.optimization_history == []
        assert isinstance(automl_service.parameter_spaces, dict)
        
        # Check parameter spaces are initialized
        assert len(automl_service.parameter_spaces) > 0

    def test_parameter_spaces_initialization(self, automl_service):
        """Test parameter spaces are properly initialized."""
        spaces = automl_service.parameter_spaces
        
        # Check isolation forest parameters
        if MockPredictionAlgorithm.ISOLATION_FOREST in spaces:
            iso_params = spaces[MockPredictionAlgorithm.ISOLATION_FOREST]
            assert 'n_estimators' in iso_params
            assert 'contamination' in iso_params
            assert isinstance(iso_params['n_estimators'], list)

    @pytest.mark.asyncio
    async def test_optimize_prediction_default_config(self, automl_service, mock_dataset):
        """Test optimization with default configuration."""
        result = await automl_service.optimize_prediction(mock_dataset)
        
        assert isinstance(result, OptimizationResult)
        assert result.best_algorithm is not None
        assert result.best_config is not None
        assert result.best_score >= 0
        assert isinstance(result.best_metrics, dict)
        assert result.optimization_time > 0

    @pytest.mark.asyncio
    async def test_optimize_prediction_custom_config(self, automl_service, mock_dataset):
        """Test optimization with custom configuration."""
        config = OptimizationConfig(
            max_trials=5,
            search_strategy=SearchStrategy.GRID_SEARCH,
            primary_metric=OptimizationMetric.PRECISION,
            cv_folds=3
        )
        
        result = await automl_service.optimize_prediction(mock_dataset, config)
        
        assert isinstance(result, OptimizationResult)
        assert result.total_trials <= 5

    @pytest.mark.asyncio
    async def test_optimize_prediction_with_ground_truth(self, automl_service, mock_dataset):
        """Test optimization with ground truth labels."""
        ground_truth = np.random.choice([0, 1], size=100)
        
        result = await automl_service.optimize_prediction(mock_dataset, ground_truth=ground_truth)
        
        assert isinstance(result, OptimizationResult)
        assert result.best_score >= 0

    @pytest.mark.asyncio
    async def test_grid_search_optimization(self, automl_service, mock_dataset):
        """Test grid search optimization strategy."""
        config = OptimizationConfig(
            search_strategy=SearchStrategy.GRID_SEARCH,
            max_trials=10
        )
        
        result = await automl_service.optimize_prediction(mock_dataset, config)
        
        assert isinstance(result, OptimizationResult)
        assert len(result.trial_history) > 0
        assert result.total_trials > 0

    @pytest.mark.asyncio
    async def test_random_search_optimization(self, automl_service, mock_dataset):
        """Test random search optimization strategy."""
        config = OptimizationConfig(
            search_strategy=SearchStrategy.RANDOM_SEARCH,
            max_trials=10
        )
        
        result = await automl_service.optimize_prediction(mock_dataset, config)
        
        assert isinstance(result, OptimizationResult)
        assert len(result.trial_history) > 0

    @pytest.mark.asyncio
    async def test_bayesian_optimization_without_optuna(self, automl_service, mock_dataset):
        """Test Bayesian optimization fallback when Optuna not available."""
        config = OptimizationConfig(
            search_strategy=SearchStrategy.BAYESIAN_OPTIMIZATION,
            max_trials=5
        )
        
        with patch('machine_learning.domain.services.automl_service.OPTUNA_AVAILABLE', False):
            result = await automl_service.optimize_prediction(mock_dataset, config)
            
            assert isinstance(result, OptimizationResult)

    @pytest.mark.asyncio
    async def test_optimize_prediction_no_algorithms(self, automl_service, mock_dataset):
        """Test optimization when no algorithms are available."""
        automl_service.prediction_service.get_available_algorithms = AsyncMock(return_value=[])
        
        with pytest.raises(ValueError, match="No algorithms available for optimization"):
            await automl_service.optimize_prediction(mock_dataset)

    @pytest.mark.asyncio
    async def test_optimize_prediction_filtered_algorithms(self, automl_service, mock_dataset):
        """Test optimization with filtered algorithm list."""
        config = OptimizationConfig(
            algorithms_to_test=[MockPredictionAlgorithm.ISOLATION_FOREST],
            max_trials=5
        )
        
        result = await automl_service.optimize_prediction(mock_dataset, config)
        
        assert isinstance(result, OptimizationResult)

    @pytest.mark.asyncio
    async def test_optimize_prediction_with_ensemble(self, automl_service, mock_dataset):
        """Test optimization with ensemble methods enabled."""
        config = OptimizationConfig(
            ensemble_methods=True,
            max_ensemble_size=3,
            max_trials=10
        )
        
        result = await automl_service.optimize_prediction(mock_dataset, config)
        
        assert isinstance(result, OptimizationResult)
        # Ensemble optimization depends on having multiple algorithms

    @pytest.mark.asyncio 
    async def test_evaluate_configuration_success(self, automl_service, mock_dataset):
        """Test successful configuration evaluation."""
        config = MockAlgorithmConfig(MockPredictionAlgorithm.ISOLATION_FOREST)
        opt_config = OptimizationConfig()
        
        score, metrics = await automl_service._evaluate_configuration(
            mock_dataset, config, opt_config, None
        )
        
        assert score >= 0
        assert isinstance(metrics, dict)

    @pytest.mark.asyncio
    async def test_evaluate_configuration_with_different_metrics(self, automl_service, mock_dataset):
        """Test configuration evaluation with different primary metrics."""
        config = MockAlgorithmConfig(MockPredictionAlgorithm.ISOLATION_FOREST)
        
        # Test different primary metrics
        for metric in [OptimizationMetric.PRECISION, OptimizationMetric.RECALL, 
                      OptimizationMetric.AUC, OptimizationMetric.ANOMALY_RATE]:
            opt_config = OptimizationConfig(primary_metric=metric)
            score, metrics = await automl_service._evaluate_configuration(
                mock_dataset, config, opt_config, None
            )
            assert score >= 0

    @pytest.mark.asyncio
    async def test_evaluate_configuration_execution_time_metric(self, automl_service, mock_dataset):
        """Test configuration evaluation with execution time as primary metric."""
        config = MockAlgorithmConfig(MockPredictionAlgorithm.ISOLATION_FOREST)
        opt_config = OptimizationConfig(primary_metric=OptimizationMetric.EXECUTION_TIME)
        
        score, metrics = await automl_service._evaluate_configuration(
            mock_dataset, config, opt_config, None
        )
        
        assert 0 <= score <= 1  # Inverted time score should be between 0 and 1

    @pytest.mark.asyncio
    async def test_evaluate_configuration_failure(self, automl_service, mock_dataset):
        """Test configuration evaluation handles failures gracefully."""
        config = MockAlgorithmConfig(MockPredictionAlgorithm.ISOLATION_FOREST)
        opt_config = OptimizationConfig()
        
        # Mock service to raise exception
        automl_service.prediction_service.predict = AsyncMock(side_effect=Exception("Test error"))
        
        score, metrics = await automl_service._evaluate_configuration(
            mock_dataset, config, opt_config, None
        )
        
        assert score == 0.0
        assert metrics == {}

    @pytest.mark.asyncio
    async def test_auto_select_algorithm_quick_mode(self, automl_service, mock_dataset):
        """Test quick algorithm selection mode."""
        algorithm, config = await automl_service.auto_select_algorithm(
            mock_dataset, quick_mode=True
        )
        
        assert algorithm is not None
        assert config is not None

    @pytest.mark.asyncio
    async def test_auto_select_algorithm_full_mode(self, automl_service, mock_dataset):
        """Test full algorithm selection mode."""
        algorithm, config = await automl_service.auto_select_algorithm(
            mock_dataset, quick_mode=False
        )
        
        assert algorithm is not None
        assert config is not None

    @pytest.mark.asyncio
    async def test_quick_algorithm_selection_small_dataset(self, automl_service):
        """Test quick algorithm selection for small dataset."""
        small_data = np.random.randn(100, 5)
        dataset = MockDataset(data=small_data)
        
        algorithm, config = await automl_service._quick_algorithm_selection(dataset)
        
        assert algorithm is not None
        assert config is not None

    @pytest.mark.asyncio
    async def test_quick_algorithm_selection_large_dataset(self, automl_service):
        """Test quick algorithm selection for large dataset."""
        # Mock large dataset
        dataset = MockDataset(data=Mock())
        dataset.data.shape = (200000, 30)
        
        algorithm, config = await automl_service._quick_algorithm_selection(dataset)
        
        assert algorithm is not None

    @pytest.mark.asyncio
    async def test_quick_algorithm_selection_high_dimensional(self, automl_service):
        """Test quick algorithm selection for high-dimensional dataset."""
        dataset = MockDataset(data=Mock())
        dataset.data.shape = (5000, 100)
        
        algorithm, config = await automl_service._quick_algorithm_selection(dataset)
        
        assert algorithm is not None

    @pytest.mark.asyncio
    async def test_quick_algorithm_selection_dataframe(self, automl_service, mock_dataframe_dataset):
        """Test quick algorithm selection with DataFrame dataset."""
        algorithm, config = await automl_service._quick_algorithm_selection(mock_dataframe_dataset)
        
        assert algorithm is not None
        assert config is not None

    @pytest.mark.asyncio
    async def test_quick_algorithm_selection_no_data_attribute(self, automl_service):
        """Test quick algorithm selection when dataset has no data attribute."""
        dataset = Mock()
        del dataset.data  # Remove data attribute
        
        algorithm, config = await automl_service._quick_algorithm_selection(dataset)
        
        assert algorithm is not None
        assert config is not None

    @pytest.mark.asyncio
    async def test_get_optimization_recommendations_dataframe(self, automl_service, mock_dataframe_dataset):
        """Test getting optimization recommendations for DataFrame."""
        # Add missing values and high dimensionality
        mock_dataframe_dataset.data = pd.DataFrame({
            f'feature_{i}': np.random.randn(1000) if i < 70 else [np.nan] * 100 + list(np.random.randn(900))
            for i in range(80)
        })
        
        recommendations = await automl_service.get_optimization_recommendations(mock_dataframe_dataset)
        
        assert isinstance(recommendations, dict)
        assert "data_preprocessing" in recommendations
        assert "algorithm_suggestions" in recommendations
        assert "parameter_tuning" in recommendations
        assert "ensemble_methods" in recommendations
        assert "general_tips" in recommendations
        
        # Should recommend handling missing values
        assert any("missing values" in rec for rec in recommendations["data_preprocessing"])
        
        # Should recommend dimensionality reduction for high-dimensional data
        assert any("dimensionality reduction" in rec for rec in recommendations["data_preprocessing"])

    @pytest.mark.asyncio
    async def test_get_optimization_recommendations_with_results(self, automl_service, mock_dataset):
        """Test getting optimization recommendations with current results."""
        # Mock current results with high anomaly rate
        current_results = Mock()
        current_results.anomaly_count = 300
        current_results.total_samples = 1000
        current_results.execution_time = 120  # 2 minutes
        
        recommendations = await automl_service.get_optimization_recommendations(
            mock_dataset, current_results
        )
        
        # Should recommend reducing contamination for high anomaly rate
        assert any("reducing contamination" in rec for rec in recommendations["parameter_tuning"])
        
        # Should recommend reducing complexity for long execution time
        assert any("reducing model complexity" in rec for rec in recommendations["parameter_tuning"])

    @pytest.mark.asyncio
    async def test_get_optimization_recommendations_low_anomaly_rate(self, automl_service, mock_dataset):
        """Test optimization recommendations for low anomaly rate."""
        current_results = Mock()
        current_results.anomaly_count = 5
        current_results.total_samples = 1000
        current_results.execution_time = 30
        
        recommendations = await automl_service.get_optimization_recommendations(
            mock_dataset, current_results
        )
        
        # Should recommend increasing contamination for low anomaly rate
        assert any("increasing contamination" in rec for rec in recommendations["parameter_tuning"])

    @pytest.mark.asyncio
    async def test_get_optimization_recommendations_no_data(self, automl_service):
        """Test optimization recommendations when dataset has no data attribute."""
        dataset = Mock()
        del dataset.data
        
        recommendations = await automl_service.get_optimization_recommendations(dataset)
        
        assert isinstance(recommendations, dict)
        assert len(recommendations["general_tips"]) > 0

    def test_optimization_config_defaults(self):
        """Test OptimizationConfig default values."""
        config = OptimizationConfig()
        
        assert config.search_strategy == SearchStrategy.BAYESIAN_OPTIMIZATION
        assert config.max_trials == 100
        assert config.timeout_seconds is None
        assert config.primary_metric == OptimizationMetric.F1_SCORE
        assert config.secondary_metrics == [OptimizationMetric.AUC]
        assert config.cv_folds == 5
        assert config.algorithms_to_test is None
        assert config.ensemble_methods is True
        assert config.max_ensemble_size == 5
        assert config.random_state == 42

    def test_optimization_result_creation(self):
        """Test OptimizationResult creation."""
        result = OptimizationResult(
            best_algorithm=MockPredictionAlgorithm.ISOLATION_FOREST,
            best_config=MockAlgorithmConfig(MockPredictionAlgorithm.ISOLATION_FOREST),
            best_score=0.85,
            best_metrics={'f1_score': 0.85}
        )
        
        assert result.best_algorithm == MockPredictionAlgorithm.ISOLATION_FOREST
        assert result.best_score == 0.85
        assert result.best_metrics == {'f1_score': 0.85}
        assert result.trial_history == []
        assert result.total_trials == 0
        assert result.optimization_time == 0.0


class TestOptimizationMetric:
    """Test cases for OptimizationMetric enum."""

    def test_optimization_metric_values(self):
        """Test all OptimizationMetric enum values."""
        assert OptimizationMetric.F1_SCORE.value == "f1_score"
        assert OptimizationMetric.PRECISION.value == "precision"
        assert OptimizationMetric.RECALL.value == "recall"
        assert OptimizationMetric.AUC.value == "auc"
        assert OptimizationMetric.BALANCED_ACCURACY.value == "balanced_accuracy"
        assert OptimizationMetric.ANOMALY_RATE.value == "anomaly_rate"
        assert OptimizationMetric.EXECUTION_TIME.value == "execution_time"


class TestSearchStrategy:
    """Test cases for SearchStrategy enum."""

    def test_search_strategy_values(self):
        """Test all SearchStrategy enum values."""
        assert SearchStrategy.GRID_SEARCH.value == "grid_search"
        assert SearchStrategy.RANDOM_SEARCH.value == "random_search"
        assert SearchStrategy.BAYESIAN_OPTIMIZATION.value == "bayesian_optimization"
        assert SearchStrategy.GENETIC_ALGORITHM.value == "genetic_algorithm"
        assert SearchStrategy.MULTI_OBJECTIVE.value == "multi_objective"


class TestAutoMLServiceSingleton:
    """Test cases for AutoML service singleton."""

    def test_get_automl_service_singleton(self):
        """Test that get_automl_service returns singleton instance."""
        with patch('machine_learning.domain.services.automl_service.get_prediction_service'):
            service1 = get_automl_service()
            service2 = get_automl_service()
            
            assert service1 is service2

    def test_get_automl_service_creates_instance(self):
        """Test that get_automl_service creates instance when none exists."""
        import machine_learning.domain.services.automl_service as automl_module
        
        # Reset singleton
        automl_module._automl_service = None
        
        with patch('machine_learning.domain.services.automl_service.get_prediction_service'):
            service = get_automl_service()
            assert isinstance(service, AutoMLService)