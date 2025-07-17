"""Tests for enhanced AutoML service."""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest

from monorepo.application.services.enhanced_automl_service import (
    EnhancedAutoMLConfig,
    EnhancedAutoMLResult,
    EnhancedAutoMLService,
)
from monorepo.domain.entities import Dataset


@pytest.fixture
def temp_storage():
    """Create temporary storage directory."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_repositories():
    """Mock repositories for testing."""
    detector_repo = AsyncMock()
    dataset_repo = AsyncMock()
    return detector_repo, dataset_repo


@pytest.fixture
def mock_adapter_registry():
    """Mock adapter registry."""
    registry = Mock()
    mock_adapter = Mock()
    mock_adapter.train.return_value = True
    mock_adapter.predict.return_value = (
        np.array([0, 1, 0, 1, 0]),  # predictions
        np.array([0.1, 0.9, 0.2, 0.8, 0.3]),  # scores
    )
    registry.get_adapter.return_value = mock_adapter
    return registry


@pytest.fixture
def sample_dataset():
    """Sample dataset for testing."""
    features = pd.DataFrame(
        {
            "feature1": np.random.normal(0, 1, 100),
            "feature2": np.random.normal(0, 1, 100),
            "feature3": np.random.normal(0, 1, 100),
        }
    )

    return Dataset(
        id=str(uuid4()),
        name="test_dataset",
        features=features,
        labels=None,
        metadata={},
    )


@pytest.fixture
def enhanced_config():
    """Enhanced AutoML configuration."""
    return EnhancedAutoMLConfig(
        max_optimization_time=60,
        n_trials=10,
        enable_meta_learning=True,
        enable_multi_objective=True,
        objectives=["auc", "training_time"],
        enable_parallel=False,  # Disable for testing
        random_state=42,
    )


@pytest.fixture
def enhanced_automl_service(
    mock_repositories, mock_adapter_registry, enhanced_config, temp_storage
):
    """Enhanced AutoML service for testing."""
    detector_repo, dataset_repo = mock_repositories

    service = EnhancedAutoMLService(
        detector_repository=detector_repo,
        dataset_repository=dataset_repo,
        adapter_registry=mock_adapter_registry,
        config=enhanced_config,
        storage_path=temp_storage,
    )

    return service


class TestEnhancedAutoMLConfig:
    """Test enhanced AutoML configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = EnhancedAutoMLConfig()

        assert config.max_optimization_time == 3600
        assert config.n_trials == 100
        assert config.cv_folds == 3
        assert config.random_state == 42
        assert config.enable_meta_learning is True
        assert config.enable_multi_objective is False
        assert config.enable_early_stopping is True
        assert config.enable_parallel is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = EnhancedAutoMLConfig(
            max_optimization_time=1800,
            n_trials=50,
            enable_meta_learning=False,
            enable_multi_objective=True,
            objectives=["auc", "precision", "training_time"],
            objective_weights={"auc": 0.5, "precision": 0.3, "training_time": 0.2},
        )

        assert config.max_optimization_time == 1800
        assert config.n_trials == 50
        assert config.enable_meta_learning is False
        assert config.enable_multi_objective is True
        assert len(config.objectives) == 3
        assert config.objective_weights["auc"] == 0.5


class TestEnhancedAutoMLService:
    """Test enhanced AutoML service."""

    def test_service_initialization(self, enhanced_automl_service):
        """Test service initialization."""
        service = enhanced_automl_service

        assert service.config is not None
        assert service.storage_path.exists()
        assert hasattr(service, "optimization_history")
        assert isinstance(service.optimization_history, list)

    def test_dataset_profile_conversion(self, enhanced_automl_service, sample_dataset):
        """Test dataset profile conversion to dictionary."""
        service = enhanced_automl_service

        # Mock profile dataset method
        with patch.object(service, "profile_dataset") as mock_profile:
            from monorepo.application.services.automl_service import DatasetProfile

            mock_profile.return_value = DatasetProfile(
                n_samples=100,
                n_features=3,
                contamination_estimate=0.1,
                feature_types={
                    "feature1": "numerical",
                    "feature2": "numerical",
                    "feature3": "numerical",
                },
                missing_values_ratio=0.0,
                categorical_features=[],
                numerical_features=["feature1", "feature2", "feature3"],
                time_series_features=[],
                sparsity_ratio=0.0,
                dimensionality_ratio=0.03,
                dataset_size_mb=0.001,
                has_temporal_structure=False,
                has_graph_structure=False,
            )

            profile = mock_profile.return_value
            profile_dict = service._convert_profile_to_dict(profile)

            assert isinstance(profile_dict, dict)
            assert "n_samples" in profile_dict
            assert "n_features" in profile_dict
            assert "contamination_estimate" in profile_dict
            assert profile_dict["n_samples"] == 100
            assert profile_dict["n_features"] == 3

    def test_synthetic_labels_creation(self, enhanced_automl_service):
        """Test synthetic labels creation for evaluation."""
        service = enhanced_automl_service

        X = np.random.normal(0, 1, (100, 3))
        contamination = 0.1

        y_true = service._create_synthetic_labels(X, contamination)

        assert len(y_true) == 100
        assert np.sum(y_true) == int(100 * contamination)
        assert np.all(np.isin(y_true, [0, 1]))

    def test_score_categorization(self, enhanced_automl_service):
        """Test performance score categorization."""
        service = enhanced_automl_service

        assert service._categorize_score(0.95) == "excellent"
        assert service._categorize_score(0.85) == "good"
        assert service._categorize_score(0.75) == "fair"
        assert service._categorize_score(0.65) == "poor"
        assert service._categorize_score(0.50) == "very_poor"

    def test_convergence_stability_calculation(self, enhanced_automl_service):
        """Test convergence stability calculation."""
        service = enhanced_automl_service

        # Stable convergence
        stable_history = [0.5, 0.6, 0.7, 0.75, 0.8, 0.82, 0.83, 0.84, 0.84, 0.85]
        stability = service._calculate_convergence_stability(stable_history)
        assert stability > 0.5

        # Unstable convergence
        unstable_history = [0.5, 0.3, 0.8, 0.2, 0.9, 0.1, 0.7, 0.4, 0.6, 0.5]
        stability = service._calculate_convergence_stability(unstable_history)
        assert stability < 0.5

        # Short history
        short_history = [0.5, 0.6]
        stability = service._calculate_convergence_stability(short_history)
        assert stability == 0.0

    @pytest.mark.asyncio
    async def test_basic_result_conversion(self, enhanced_automl_service):
        """Test conversion from basic to enhanced result."""
        service = enhanced_automl_service

        from monorepo.application.services.automl_service import AutoMLResult

        basic_result = AutoMLResult(
            best_algorithm="IsolationForest",
            best_params={"contamination": 0.1, "n_estimators": 100},
            best_score=0.85,
            optimization_time=120.0,
            trials_completed=50,
            algorithm_rankings=[("IsolationForest", 0.85), ("LOF", 0.80)],
        )

        enhanced_result = service._convert_basic_to_enhanced_result(basic_result)

        assert isinstance(enhanced_result, EnhancedAutoMLResult)
        assert enhanced_result.best_algorithm == "IsolationForest"
        assert enhanced_result.best_score == 0.85
        assert enhanced_result.optimization_strategy_used == "basic"
        assert enhanced_result.exploration_score == 0.5
        assert enhanced_result.exploitation_score == 0.5

    def test_advanced_ensemble_config_creation(self, enhanced_automl_service):
        """Test advanced ensemble configuration creation."""
        service = enhanced_automl_service

        # Create mock optimization results
        results = [
            EnhancedAutoMLResult(
                best_algorithm="IsolationForest",
                best_params={"contamination": 0.1},
                best_score=0.85,
                optimization_time=60.0,
                trials_completed=20,
                algorithm_rankings=[],
                convergence_stability=0.8,
            ),
            EnhancedAutoMLResult(
                best_algorithm="LOF",
                best_params={"contamination": 0.1},
                best_score=0.80,
                optimization_time=80.0,
                trials_completed=25,
                algorithm_rankings=[],
                convergence_stability=0.7,
            ),
            EnhancedAutoMLResult(
                best_algorithm="ECOD",
                best_params={"contamination": 0.1},
                best_score=0.75,
                optimization_time=40.0,
                trials_completed=15,
                algorithm_rankings=[],
                convergence_stability=0.9,
            ),
        ]

        ensemble_config = service._create_advanced_ensemble_config(results)

        assert ensemble_config["method"] == "advanced_weighted_voting"
        assert len(ensemble_config["algorithms"]) <= 3
        assert ensemble_config["voting_strategy"] == "soft"
        assert ensemble_config["dynamic_weighting"] is True
        assert ensemble_config["confidence_boosting"] is True

        # Check that weights sum to approximately 1
        total_weight = sum(alg["weight"] for alg in ensemble_config["algorithms"])
        assert abs(total_weight - 1.0) < 0.01

    def test_optimization_history_storage(self, enhanced_automl_service, temp_storage):
        """Test optimization history storage."""
        service = enhanced_automl_service

        result = EnhancedAutoMLResult(
            best_algorithm="IsolationForest",
            best_params={"contamination": 0.1},
            best_score=0.85,
            optimization_time=60.0,
            trials_completed=20,
            algorithm_rankings=[],
            optimization_strategy_used="bayesian",
        )

        dataset_profile = {
            "n_samples": 1000,
            "n_features": 50,
            "contamination_estimate": 0.1,
        }

        service._store_optimization_history(result, dataset_profile)

        # Check that history was stored in memory
        assert len(service.optimization_history) == 1
        assert service.optimization_history[0]["algorithm"] == "IsolationForest"

        # Check that history file was created
        history_file = temp_storage / "optimization_history.json"
        assert history_file.exists()

    def test_final_recommendations_generation(self, enhanced_automl_service):
        """Test final recommendations generation."""
        service = enhanced_automl_service

        # High performance result
        high_perf_result = EnhancedAutoMLResult(
            best_algorithm="IsolationForest",
            best_params={"contamination": 0.1},
            best_score=0.95,
            optimization_time=60.0,
            trials_completed=20,
            algorithm_rankings=[],
            convergence_stability=0.8,
            exploration_score=0.7,
            exploitation_score=0.8,
        )

        all_results = [high_perf_result]

        service._add_final_recommendations(high_perf_result, all_results)

        assert len(high_perf_result.optimization_recommendations) > 0
        assert "Excellent performance achieved" in " ".join(
            high_perf_result.optimization_recommendations
        )

        # Low performance result
        low_perf_result = EnhancedAutoMLResult(
            best_algorithm="LOF",
            best_params={"contamination": 0.1},
            best_score=0.60,
            optimization_time=60.0,
            trials_completed=20,
            algorithm_rankings=[],
            convergence_stability=0.3,
            exploration_score=0.2,
            exploitation_score=0.3,
        )

        service._add_final_recommendations(low_perf_result, [low_perf_result])

        assert len(low_perf_result.optimization_recommendations) > 0
        assert any(
            "more training data" in rec
            for rec in low_perf_result.optimization_recommendations
        )

    def test_optimization_insights_generation(self, enhanced_automl_service):
        """Test optimization insights generation."""
        service = enhanced_automl_service

        result = EnhancedAutoMLResult(
            best_algorithm="IsolationForest",
            best_params={"contamination": 0.1},
            best_score=0.85,
            optimization_time=120.0,
            trials_completed=50,
            algorithm_rankings=[("IsolationForest", 0.85), ("LOF", 0.80)],
            optimization_strategy_used="bayesian",
            exploration_score=0.7,
            exploitation_score=0.8,
            convergence_stability=0.75,
            training_time_breakdown={
                "optimization_time": 120.0,
                "average_trial_time": 2.4,
                "total_trials": 50,
            },
        )

        insights = service.get_optimization_insights(result)

        assert "performance_analysis" in insights
        assert "optimization_analysis" in insights
        assert "efficiency_analysis" in insights
        assert "recommendations" in insights
        assert "next_steps" in insights

        # Check performance analysis
        perf_analysis = insights["performance_analysis"]
        assert perf_analysis["best_score"] == 0.85
        assert perf_analysis["score_category"] == "good"
        assert len(perf_analysis["algorithm_rankings"]) == 2

        # Check optimization analysis
        opt_analysis = insights["optimization_analysis"]
        assert opt_analysis["strategy_used"] == "bayesian"
        assert opt_analysis["exploration_score"] == 0.7
        assert opt_analysis["exploitation_score"] == 0.8

        # Check efficiency analysis
        eff_analysis = insights["efficiency_analysis"]
        assert eff_analysis["total_optimization_time"] == 120.0
        assert eff_analysis["trials_completed"] == 50


@pytest.mark.skipif(
    not pytest.importorskip("optuna", minversion="3.0.0"), reason="Optuna not available"
)
class TestAdvancedOptimizationIntegration:
    """Test integration with advanced optimization features."""

    @pytest.mark.asyncio
    async def test_advanced_optimization_fallback(
        self, enhanced_automl_service, sample_dataset
    ):
        """Test fallback to basic optimization when advanced features unavailable."""
        service = enhanced_automl_service
        service.advanced_optimizer = None  # Simulate unavailable optimizer

        # Mock dataset repository
        service.dataset_repository.get.return_value = sample_dataset

        # Mock basic optimization method
        with patch.object(service, "optimize_hyperparameters") as mock_optimize:
            from monorepo.application.services.automl_service import AutoMLResult

            mock_basic_result = AutoMLResult(
                best_algorithm="IsolationForest",
                best_params={"contamination": 0.1},
                best_score=0.80,
                optimization_time=60.0,
                trials_completed=20,
                algorithm_rankings=[],
            )
            mock_optimize.return_value = mock_basic_result

            result = await service.advanced_optimize_hyperparameters(
                dataset_id=sample_dataset.id, algorithm="IsolationForest"
            )

            assert isinstance(result, EnhancedAutoMLResult)
            assert result.optimization_strategy_used == "basic"

    def test_objective_function_creation(self, enhanced_automl_service, sample_dataset):
        """Test advanced objective function creation."""
        service = enhanced_automl_service

        from monorepo.application.services.automl_service import (
            AlgorithmConfig,
            AlgorithmFamily,
        )

        config = AlgorithmConfig(
            name="IsolationForest",
            family=AlgorithmFamily.ISOLATION_BASED,
            adapter_type="sklearn",
            default_params={"contamination": 0.1},
            param_space={"contamination": {"type": "float", "low": 0.01, "high": 0.5}},
            complexity_score=0.5,
            training_time_factor=0.4,
            memory_factor=0.3,
        )

        objectives = ["auc", "training_time"]
        objective_func = service._create_advanced_objective_function(
            sample_dataset, config, objectives
        )

        # Test single objective
        params = {"contamination": 0.1}
        single_result = objective_func(params)
        assert isinstance(single_result, dict)
        assert "auc" in single_result
        assert "training_time" in single_result

        # Test with callbacks
        mock_callback = Mock(return_value=False)
        result_with_callback = objective_func(params, callbacks=[mock_callback])
        assert isinstance(result_with_callback, dict)


class TestEnhancedAutoMLResult:
    """Test enhanced AutoML result."""

    def test_enhanced_result_creation(self):
        """Test enhanced result creation."""
        result = EnhancedAutoMLResult(
            best_algorithm="IsolationForest",
            best_params={"contamination": 0.1},
            best_score=0.85,
            optimization_time=120.0,
            trials_completed=50,
            algorithm_rankings=[],
            optimization_strategy_used="bayesian",
            exploration_score=0.7,
            exploitation_score=0.8,
            convergence_stability=0.75,
        )

        assert result.best_algorithm == "IsolationForest"
        assert result.optimization_strategy_used == "bayesian"
        assert result.exploration_score == 0.7
        assert result.exploitation_score == 0.8
        assert result.convergence_stability == 0.75
        assert isinstance(result.optimization_recommendations, list)
        assert isinstance(result.next_steps, list)

    def test_enhanced_result_with_pareto_front(self):
        """Test enhanced result with Pareto front."""
        pareto_front = [
            {
                "parameters": {"contamination": 0.1, "n_estimators": 100},
                "objectives": {"auc": 0.85, "training_time": 0.7},
            },
            {
                "parameters": {"contamination": 0.15, "n_estimators": 50},
                "objectives": {"auc": 0.80, "training_time": 0.9},
            },
        ]

        result = EnhancedAutoMLResult(
            best_algorithm="IsolationForest",
            best_params={"contamination": 0.1},
            best_score=0.85,
            optimization_time=120.0,
            trials_completed=50,
            algorithm_rankings=[],
            pareto_front=pareto_front,
        )

        assert result.pareto_front is not None
        assert len(result.pareto_front) == 2
        assert "parameters" in result.pareto_front[0]
        assert "objectives" in result.pareto_front[0]
