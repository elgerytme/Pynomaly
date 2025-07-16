"""
Comprehensive tests for autonomous service.
Tests autonomous algorithm selection, configuration optimization, and intelligent workflows.
"""

import asyncio
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

import numpy as np
import pytest

from monorepo.application.services.autonomous_service import (
    AutonomousDetectionService as AutonomousService,
)
from monorepo.domain.entities.dataset import Dataset
from monorepo.domain.entities.detector import Detector
from monorepo.domain.entities.experiment import Experiment
from monorepo.domain.exceptions import DatasetError
from monorepo.domain.value_objects import PerformanceMetrics


class TestAutonomousService:
    """Test suite for AutonomousService application service."""

    @pytest.fixture
    def mock_dataset_repository(self):
        """Mock dataset repository."""
        mock_repo = AsyncMock()
        mock_repo.find_by_id.return_value = None
        mock_repo.save.return_value = None
        return mock_repo

    @pytest.fixture
    def mock_detector_repository(self):
        """Mock detector repository."""
        mock_repo = AsyncMock()
        mock_repo.find_by_id.return_value = None
        mock_repo.save.return_value = None
        return mock_repo

    @pytest.fixture
    def mock_experiment_repository(self):
        """Mock experiment repository."""
        mock_repo = AsyncMock()
        mock_repo.save.return_value = None
        mock_repo.find_by_dataset_id.return_value = []
        return mock_repo

    @pytest.fixture
    def mock_algorithm_registry(self):
        """Mock algorithm registry."""
        registry = Mock()
        registry.get_adapter.return_value = Mock()
        registry.list_algorithms.return_value = [
            "IsolationForest",
            "LocalOutlierFactor",
            "OneClassSVM",
            "EllipticEnvelope",
        ]
        registry.get_algorithm_metadata.return_value = {
            "complexity": "medium",
            "scalability": "high",
            "interpretability": "medium",
            "suitable_for": ["tabular", "mixed"],
        }
        return registry

    @pytest.fixture
    def mock_data_profiler(self):
        """Mock data profiler."""
        profiler = Mock()
        profiler.profile_dataset.return_value = {
            "shape": (1000, 10),
            "data_types": {"numeric": 8, "categorical": 2},
            "missing_values": 0.05,
            "outlier_ratio": 0.1,
            "correlation_matrix": np.eye(10),
            "feature_importance": np.random.random(10),
            "complexity_score": 0.7,
            "data_quality_score": 0.85,
        }
        return profiler

    @pytest.fixture
    def mock_hyperparameter_optimizer(self):
        """Mock hyperparameter optimizer."""
        optimizer = Mock()
        optimizer.optimize.return_value = {
            "best_params": {"n_estimators": 100, "contamination": 0.1},
            "best_score": 0.85,
            "optimization_history": [],
            "convergence_info": {"converged": True, "iterations": 50},
        }
        return optimizer

    @pytest.fixture
    def mock_performance_evaluator(self):
        """Mock performance evaluator."""
        evaluator = Mock()
        evaluator.evaluate.return_value = PerformanceMetrics(
            precision=0.85,
            recall=0.78,
            f1_score=0.814,
            accuracy=0.92,
            roc_auc=0.88,
        )
        return evaluator

    @pytest.fixture
    def sample_dataset(self):
        """Create sample dataset."""
        return Dataset(
            id=uuid4(),
            name="autonomous-test-dataset",
            file_path="/tmp/test.csv",
            features=["feature1", "feature2", "feature3", "feature4", "feature5"],
            feature_types={
                "feature1": "numeric",
                "feature2": "numeric",
                "feature3": "numeric",
                "feature4": "categorical",
                "feature5": "categorical",
            },
            target_column=None,
            data_shape=(1000, 5),
        )

    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        return np.random.randn(1000, 5)

    @pytest.fixture
    def autonomous_service(
        self,
        mock_dataset_repository,
        mock_detector_repository,
        mock_experiment_repository,
        mock_algorithm_registry,
        mock_data_profiler,
        mock_hyperparameter_optimizer,
        mock_performance_evaluator,
    ):
        """Create autonomous service with mocked dependencies."""
        return AutonomousService(
            dataset_repository=mock_dataset_repository,
            detector_repository=mock_detector_repository,
            experiment_repository=mock_experiment_repository,
            algorithm_registry=mock_algorithm_registry,
            data_profiler=mock_data_profiler,
            hyperparameter_optimizer=mock_hyperparameter_optimizer,
            performance_evaluator=mock_performance_evaluator,
        )

    @pytest.mark.asyncio
    async def test_analyze_dataset_characteristics(
        self,
        autonomous_service,
        sample_dataset,
        sample_data,
        mock_dataset_repository,
        mock_data_profiler,
    ):
        """Test dataset characteristics analysis."""
        # Setup
        dataset_id = sample_dataset.id
        mock_dataset_repository.find_by_id.return_value = sample_dataset

        # Execute
        characteristics = await autonomous_service.analyze_dataset_characteristics(
            dataset_id=dataset_id,
            data=sample_data,
        )

        # Verify
        assert characteristics is not None
        assert "shape" in characteristics
        assert "data_types" in characteristics
        assert "missing_values" in characteristics
        assert "outlier_ratio" in characteristics
        assert "complexity_score" in characteristics
        assert "data_quality_score" in characteristics

        # Verify data profiler was called
        mock_data_profiler.profile_dataset.assert_called_once_with(sample_data)

    @pytest.mark.asyncio
    async def test_recommend_algorithms(
        self,
        autonomous_service,
        sample_dataset,
        sample_data,
        mock_dataset_repository,
        mock_data_profiler,
        mock_algorithm_registry,
    ):
        """Test algorithm recommendation."""
        # Setup
        dataset_id = sample_dataset.id
        mock_dataset_repository.find_by_id.return_value = sample_dataset

        # Execute
        recommendations = await autonomous_service.recommend_algorithms(
            dataset_id=dataset_id,
            data=sample_data,
            max_recommendations=3,
        )

        # Verify
        assert recommendations is not None
        assert len(recommendations) <= 3
        assert len(recommendations) > 0

        for recommendation in recommendations:
            assert "algorithm_name" in recommendation
            assert "confidence_score" in recommendation
            assert "reasoning" in recommendation
            assert "expected_performance" in recommendation
            assert "computational_complexity" in recommendation

        # Verify data profiler was called
        mock_data_profiler.profile_dataset.assert_called_once_with(sample_data)

    @pytest.mark.asyncio
    async def test_recommend_algorithms_with_constraints(
        self,
        autonomous_service,
        sample_dataset,
        sample_data,
        mock_dataset_repository,
    ):
        """Test algorithm recommendation with constraints."""
        # Setup
        dataset_id = sample_dataset.id
        mock_dataset_repository.find_by_id.return_value = sample_dataset

        constraints = {
            "max_training_time": 300,  # 5 minutes
            "max_memory_usage": "2GB",
            "interpretability_required": True,
            "scalability_required": True,
        }

        # Execute
        recommendations = await autonomous_service.recommend_algorithms(
            dataset_id=dataset_id,
            data=sample_data,
            constraints=constraints,
        )

        # Verify
        assert recommendations is not None
        for recommendation in recommendations:
            assert recommendation["meets_constraints"] is True
            assert "constraint_analysis" in recommendation

    @pytest.mark.asyncio
    async def test_optimize_hyperparameters(
        self,
        autonomous_service,
        sample_dataset,
        sample_data,
        mock_dataset_repository,
        mock_hyperparameter_optimizer,
    ):
        """Test hyperparameter optimization."""
        # Setup
        dataset_id = sample_dataset.id
        algorithm_name = "IsolationForest"
        mock_dataset_repository.find_by_id.return_value = sample_dataset

        # Execute
        optimization_result = await autonomous_service.optimize_hyperparameters(
            dataset_id=dataset_id,
            data=sample_data,
            algorithm_name=algorithm_name,
        )

        # Verify
        assert optimization_result is not None
        assert "best_params" in optimization_result
        assert "best_score" in optimization_result
        assert "optimization_history" in optimization_result
        assert "convergence_info" in optimization_result

        # Verify optimizer was called
        mock_hyperparameter_optimizer.optimize.assert_called_once()

    @pytest.mark.asyncio
    async def test_optimize_hyperparameters_with_custom_search_space(
        self,
        autonomous_service,
        sample_dataset,
        sample_data,
        mock_dataset_repository,
        mock_hyperparameter_optimizer,
    ):
        """Test hyperparameter optimization with custom search space."""
        # Setup
        dataset_id = sample_dataset.id
        algorithm_name = "IsolationForest"
        custom_search_space = {
            "n_estimators": [50, 100, 200],
            "contamination": [0.05, 0.1, 0.15],
            "max_features": [0.8, 1.0],
        }
        mock_dataset_repository.find_by_id.return_value = sample_dataset

        # Execute
        optimization_result = await autonomous_service.optimize_hyperparameters(
            dataset_id=dataset_id,
            data=sample_data,
            algorithm_name=algorithm_name,
            search_space=custom_search_space,
        )

        # Verify
        assert optimization_result is not None
        assert optimization_result["search_space"] == custom_search_space

    @pytest.mark.asyncio
    async def test_create_autonomous_detector(
        self,
        autonomous_service,
        sample_dataset,
        sample_data,
        mock_dataset_repository,
        mock_detector_repository,
        mock_data_profiler,
        mock_hyperparameter_optimizer,
    ):
        """Test autonomous detector creation."""
        # Setup
        dataset_id = sample_dataset.id
        mock_dataset_repository.find_by_id.return_value = sample_dataset

        # Execute
        detector = await autonomous_service.create_autonomous_detector(
            dataset_id=dataset_id,
            data=sample_data,
            detector_name="autonomous-detector",
        )

        # Verify
        assert detector is not None
        assert isinstance(detector, Detector)
        assert detector.name == "autonomous-detector"
        assert detector.algorithm_name in [
            "IsolationForest",
            "LocalOutlierFactor",
            "OneClassSVM",
        ]
        assert detector.hyperparameters is not None
        assert detector.is_fitted is False  # Should be created but not fitted

        # Verify detector was saved
        mock_detector_repository.save.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_autonomous_detector_with_preferences(
        self,
        autonomous_service,
        sample_dataset,
        sample_data,
        mock_dataset_repository,
        mock_detector_repository,
    ):
        """Test autonomous detector creation with user preferences."""
        # Setup
        dataset_id = sample_dataset.id
        mock_dataset_repository.find_by_id.return_value = sample_dataset

        preferences = {
            "prefer_interpretability": True,
            "prefer_speed": False,
            "avoid_algorithms": ["OneClassSVM"],
            "max_complexity": "medium",
        }

        # Execute
        detector = await autonomous_service.create_autonomous_detector(
            dataset_id=dataset_id,
            data=sample_data,
            detector_name="preference-detector",
            preferences=preferences,
        )

        # Verify
        assert detector is not None
        assert detector.algorithm_name != "OneClassSVM"  # Should be avoided
        assert detector.metadata["preferences"] == preferences

    @pytest.mark.asyncio
    async def test_run_autonomous_experiment(
        self,
        autonomous_service,
        sample_dataset,
        sample_data,
        mock_dataset_repository,
        mock_experiment_repository,
        mock_performance_evaluator,
    ):
        """Test autonomous experiment execution."""
        # Setup
        dataset_id = sample_dataset.id
        mock_dataset_repository.find_by_id.return_value = sample_dataset

        # Execute
        experiment = await autonomous_service.run_autonomous_experiment(
            dataset_id=dataset_id,
            data=sample_data,
            experiment_name="autonomous-experiment",
        )

        # Verify
        assert experiment is not None
        assert isinstance(experiment, Experiment)
        assert experiment.name == "autonomous-experiment"
        assert experiment.dataset_id == dataset_id
        assert len(experiment.detectors) > 0
        assert experiment.results is not None

        # Verify experiment was saved
        mock_experiment_repository.save.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_autonomous_experiment_with_comparison(
        self,
        autonomous_service,
        sample_dataset,
        sample_data,
        mock_dataset_repository,
        mock_experiment_repository,
        mock_performance_evaluator,
    ):
        """Test autonomous experiment with algorithm comparison."""
        # Setup
        dataset_id = sample_dataset.id
        mock_dataset_repository.find_by_id.return_value = sample_dataset

        # Execute
        experiment = await autonomous_service.run_autonomous_experiment(
            dataset_id=dataset_id,
            data=sample_data,
            experiment_name="comparison-experiment",
            compare_algorithms=True,
            max_algorithms=3,
        )

        # Verify
        assert experiment is not None
        assert len(experiment.detectors) == 3
        assert experiment.comparison_results is not None
        assert "best_algorithm" in experiment.comparison_results
        assert "performance_ranking" in experiment.comparison_results
        assert "algorithm_comparison" in experiment.comparison_results

    @pytest.mark.asyncio
    async def test_optimize_ensemble_autonomously(
        self,
        autonomous_service,
        sample_dataset,
        sample_data,
        mock_dataset_repository,
        mock_performance_evaluator,
    ):
        """Test autonomous ensemble optimization."""
        # Setup
        dataset_id = sample_dataset.id
        mock_dataset_repository.find_by_id.return_value = sample_dataset

        # Execute
        ensemble_config = await autonomous_service.optimize_ensemble_autonomously(
            dataset_id=dataset_id,
            data=sample_data,
            ensemble_name="autonomous-ensemble",
        )

        # Verify
        assert ensemble_config is not None
        assert "ensemble_composition" in ensemble_config
        assert "aggregation_method" in ensemble_config
        assert "weights" in ensemble_config
        assert "expected_performance" in ensemble_config
        assert "diversity_metrics" in ensemble_config

    @pytest.mark.asyncio
    async def test_adaptive_threshold_selection(
        self,
        autonomous_service,
        sample_dataset,
        sample_data,
        mock_dataset_repository,
        mock_performance_evaluator,
    ):
        """Test adaptive threshold selection."""
        # Setup
        dataset_id = sample_dataset.id
        mock_dataset_repository.find_by_id.return_value = sample_dataset

        # Mock anomaly scores
        anomaly_scores = np.random.beta(2, 5, 1000)  # Skewed distribution

        # Execute
        threshold_config = await autonomous_service.select_adaptive_threshold(
            dataset_id=dataset_id,
            anomaly_scores=anomaly_scores,
        )

        # Verify
        assert threshold_config is not None
        assert "threshold_value" in threshold_config
        assert "threshold_method" in threshold_config
        assert "confidence_interval" in threshold_config
        assert "expected_precision" in threshold_config
        assert "expected_recall" in threshold_config

    @pytest.mark.asyncio
    async def test_autonomous_preprocessing_pipeline(
        self,
        autonomous_service,
        sample_dataset,
        sample_data,
        mock_dataset_repository,
        mock_data_profiler,
    ):
        """Test autonomous preprocessing pipeline creation."""
        # Setup
        dataset_id = sample_dataset.id
        mock_dataset_repository.find_by_id.return_value = sample_dataset

        # Execute
        preprocessing_config = await autonomous_service.create_preprocessing_pipeline(
            dataset_id=dataset_id,
            data=sample_data,
        )

        # Verify
        assert preprocessing_config is not None
        assert "pipeline_steps" in preprocessing_config
        assert "step_order" in preprocessing_config
        assert "parameters" in preprocessing_config
        assert "reasoning" in preprocessing_config

        # Verify common preprocessing steps
        steps = preprocessing_config["pipeline_steps"]
        assert any(step["name"] == "missing_value_imputation" for step in steps)
        assert any(step["name"] == "outlier_detection" for step in steps)
        assert any(step["name"] == "feature_scaling" for step in steps)

    @pytest.mark.asyncio
    async def test_autonomous_feature_engineering(
        self,
        autonomous_service,
        sample_dataset,
        sample_data,
        mock_dataset_repository,
        mock_data_profiler,
    ):
        """Test autonomous feature engineering."""
        # Setup
        dataset_id = sample_dataset.id
        mock_dataset_repository.find_by_id.return_value = sample_dataset

        # Execute
        feature_config = await autonomous_service.create_feature_engineering_pipeline(
            dataset_id=dataset_id,
            data=sample_data,
        )

        # Verify
        assert feature_config is not None
        assert "feature_transformations" in feature_config
        assert "feature_selection" in feature_config
        assert "new_features" in feature_config
        assert "feature_importance" in feature_config

    @pytest.mark.asyncio
    async def test_autonomous_model_selection_with_validation(
        self,
        autonomous_service,
        sample_dataset,
        sample_data,
        mock_dataset_repository,
        mock_performance_evaluator,
    ):
        """Test autonomous model selection with cross-validation."""
        # Setup
        dataset_id = sample_dataset.id
        mock_dataset_repository.find_by_id.return_value = sample_dataset

        # Execute
        selection_result = await autonomous_service.select_best_model_with_validation(
            dataset_id=dataset_id,
            data=sample_data,
            validation_method="k_fold",
            k_folds=5,
        )

        # Verify
        assert selection_result is not None
        assert "best_model" in selection_result
        assert "cv_results" in selection_result
        assert "validation_scores" in selection_result
        assert "confidence_intervals" in selection_result
        assert "model_stability" in selection_result

    @pytest.mark.asyncio
    async def test_autonomous_continuous_learning(
        self,
        autonomous_service,
        sample_dataset,
        sample_data,
        mock_dataset_repository,
        mock_detector_repository,
    ):
        """Test autonomous continuous learning setup."""
        # Setup
        dataset_id = sample_dataset.id
        existing_detector = Detector(
            id=uuid4(),
            name="existing-detector",
            algorithm_name="IsolationForest",
            hyperparameters={"n_estimators": 100},
            is_fitted=True,
        )
        mock_dataset_repository.find_by_id.return_value = sample_dataset
        mock_detector_repository.find_by_id.return_value = existing_detector

        # Execute
        learning_config = await autonomous_service.setup_continuous_learning(
            detector_id=existing_detector.id,
            new_data=sample_data,
        )

        # Verify
        assert learning_config is not None
        assert "learning_strategy" in learning_config
        assert "update_frequency" in learning_config
        assert "performance_monitoring" in learning_config
        assert "drift_detection" in learning_config
        assert "retraining_triggers" in learning_config

    @pytest.mark.asyncio
    async def test_autonomous_service_error_handling(
        self,
        autonomous_service,
        mock_dataset_repository,
    ):
        """Test autonomous service error handling."""
        # Test with non-existent dataset
        mock_dataset_repository.find_by_id.return_value = None

        with pytest.raises(DatasetError, match="Dataset not found"):
            await autonomous_service.analyze_dataset_characteristics(
                dataset_id=uuid4(),
                data=np.random.randn(100, 5),
            )

    @pytest.mark.asyncio
    async def test_autonomous_service_invalid_data(
        self,
        autonomous_service,
        sample_dataset,
        mock_dataset_repository,
    ):
        """Test autonomous service with invalid data."""
        # Setup
        dataset_id = sample_dataset.id
        mock_dataset_repository.find_by_id.return_value = sample_dataset

        # Test with None data
        with pytest.raises(ValueError, match="Data cannot be None"):
            await autonomous_service.analyze_dataset_characteristics(
                dataset_id=dataset_id,
                data=None,
            )

        # Test with empty data
        with pytest.raises(ValueError, match="Data cannot be empty"):
            await autonomous_service.analyze_dataset_characteristics(
                dataset_id=dataset_id,
                data=np.array([]),
            )

    @pytest.mark.asyncio
    async def test_autonomous_service_concurrent_operations(
        self,
        autonomous_service,
        sample_dataset,
        sample_data,
        mock_dataset_repository,
    ):
        """Test concurrent autonomous operations."""
        # Setup
        dataset_id = sample_dataset.id
        mock_dataset_repository.find_by_id.return_value = sample_dataset

        # Execute concurrent operations
        tasks = [
            autonomous_service.analyze_dataset_characteristics(dataset_id, sample_data),
            autonomous_service.recommend_algorithms(dataset_id, sample_data),
            autonomous_service.create_preprocessing_pipeline(dataset_id, sample_data),
        ]

        results = await asyncio.gather(*tasks)

        # Verify
        assert len(results) == 3
        characteristics, recommendations, preprocessing = results

        assert characteristics is not None
        assert recommendations is not None
        assert preprocessing is not None

    @pytest.mark.asyncio
    async def test_autonomous_service_performance_monitoring(
        self,
        autonomous_service,
        sample_dataset,
        sample_data,
        mock_dataset_repository,
    ):
        """Test autonomous service performance monitoring."""
        # Setup
        dataset_id = sample_dataset.id
        mock_dataset_repository.find_by_id.return_value = sample_dataset

        # Execute with performance monitoring
        with patch("time.time", side_effect=[0, 1, 2, 3]):  # Mock time progression
            result = await autonomous_service.analyze_dataset_characteristics(
                dataset_id=dataset_id,
                data=sample_data,
                monitor_performance=True,
            )

        # Verify
        assert result is not None
        assert "performance_metrics" in result
        assert "execution_time" in result["performance_metrics"]
        assert "memory_usage" in result["performance_metrics"]
