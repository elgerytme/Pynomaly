"""Test suite for AutoML service."""

from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest

from pynomaly.application.services.automl_service import (
    AlgorithmConfig,
    AlgorithmFamily,
    AutoMLResult,
    AutoMLService,
    DatasetProfile,
    OptimizationObjective,
)
from pynomaly.domain.entities import Dataset, Detector
from pynomaly.domain.exceptions import AutoMLError


class TestAutoMLService:
    """Test suite for AutoML service functionality."""

    @pytest.fixture
    def mock_repositories(self):
        """Mock repositories for testing."""
        detector_repo = Mock()
        dataset_repo = Mock()
        adapter_registry = Mock()

        return detector_repo, dataset_repo, adapter_registry

    @pytest.fixture
    def automl_service(self, mock_repositories):
        """Create AutoML service with mocked dependencies."""
        detector_repo, dataset_repo, adapter_registry = mock_repositories

        service = AutoMLService(
            detector_repository=detector_repo,
            dataset_repository=dataset_repo,
            adapter_registry=adapter_registry,
            max_optimization_time=60,  # Short time for tests
            n_trials=10,
            cv_folds=2,
        )

        return service

    @pytest.fixture
    def sample_dataset(self):
        """Create a sample dataset for testing."""
        np.random.seed(42)

        # Create realistic data with clear patterns
        normal_data = np.random.multivariate_normal(
            mean=[0, 0, 0], cov=[[1, 0.2, 0.1], [0.2, 1, 0.3], [0.1, 0.3, 1]], size=300
        )

        # Add some anomalies
        anomalies = np.random.multivariate_normal(
            mean=[4, 4, 4], cov=[[0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]], size=30
        )

        all_data = np.vstack([normal_data, anomalies])
        df = pd.DataFrame(all_data, columns=["feature_1", "feature_2", "feature_3"])

        dataset = Dataset(name="Sample Dataset", data=df)
        # Features are automatically available through the features property

        return dataset

    @pytest.fixture
    def complex_dataset(self):
        """Create a more complex dataset for testing."""
        np.random.seed(42)

        # Numerical features
        numerical_data = np.random.randn(500, 8)

        # Categorical features
        categorical_data = np.random.choice(["A", "B", "C"], size=(500, 2))

        # Datetime features
        dates = pd.date_range("2023-01-01", periods=500, freq="D")

        # Create DataFrame
        df = pd.DataFrame(numerical_data, columns=[f"num_{i}" for i in range(8)])
        for i, col in enumerate(["cat_1", "cat_2"]):
            df[col] = categorical_data[:, i]
        df["timestamp"] = dates

        # Add some missing values
        df.loc[::10, "num_0"] = np.nan

        dataset = Dataset(name="Complex Dataset", data=df)
        # Features are automatically available through the features property

        return dataset

    def test_dataset_profile_creation(self, automl_service, sample_dataset):
        """Test dataset profiling functionality."""
        # Mock dataset repository
        automl_service.dataset_repository.get = AsyncMock(return_value=sample_dataset)

        # Test profiling
        import asyncio

        profile = asyncio.run(automl_service.profile_dataset("test_id"))

        assert isinstance(profile, DatasetProfile)
        assert profile.n_samples == 330  # 300 normal + 30 anomalies
        assert profile.n_features == 3
        assert len(profile.numerical_features) == 3
        assert len(profile.categorical_features) == 0
        assert 0.0 <= profile.contamination_estimate <= 0.5
        assert profile.complexity_score >= 0.0

    def test_complex_dataset_profiling(self, automl_service, complex_dataset):
        """Test profiling of complex dataset with mixed features."""
        automl_service.dataset_repository.get = AsyncMock(return_value=complex_dataset)

        import asyncio

        profile = asyncio.run(automl_service.profile_dataset("complex_id"))

        assert profile.n_samples == 500
        assert profile.n_features == 11  # 8 numerical + 2 categorical + 1 datetime
        assert len(profile.numerical_features) == 8
        assert len(profile.categorical_features) == 2
        assert len(profile.time_series_features) == 1
        assert profile.missing_values_ratio > 0  # We added missing values
        assert profile.has_temporal_structure is True

    def test_algorithm_recommendation(self, automl_service, sample_dataset):
        """Test algorithm recommendation based on dataset profile."""
        # Create profile directly
        automl_service.dataset_repository.get = AsyncMock(return_value=sample_dataset)

        import asyncio

        profile = asyncio.run(automl_service.profile_dataset("test_id"))

        # Test recommendation
        recommended = automl_service.recommend_algorithms(profile, max_algorithms=3)

        assert isinstance(recommended, list)
        assert len(recommended) <= 3
        assert all(alg in automl_service.algorithm_configs for alg in recommended)

    def test_algorithm_scoring(self, automl_service):
        """Test algorithm scoring mechanism."""
        # Create test profile
        profile = DatasetProfile(
            n_samples=1000,
            n_features=10,
            contamination_estimate=0.1,
            feature_types={"num_1": "numerical", "num_2": "numerical"},
            missing_values_ratio=0.0,
            categorical_features=[],
            numerical_features=["num_1", "num_2"],
            time_series_features=[],
            sparsity_ratio=0.0,
            dimensionality_ratio=0.01,
            dataset_size_mb=1.0,
            has_temporal_structure=False,
            has_graph_structure=False,
        )

        # Test scoring for different algorithms
        isolation_forest_config = automl_service.algorithm_configs["IsolationForest"]
        score = automl_service._calculate_algorithm_score(
            isolation_forest_config, profile
        )

        assert 0.0 <= score <= 2.0  # Score can be > 1 due to bonuses
        assert isinstance(score, float)

    def test_small_dataset_recommendations(self, automl_service):
        """Test recommendations for small datasets."""
        small_profile = DatasetProfile(
            n_samples=50,  # Small dataset
            n_features=5,
            contamination_estimate=0.1,
            feature_types={"num_1": "numerical"},
            missing_values_ratio=0.0,
            categorical_features=[],
            numerical_features=["num_1"],
            time_series_features=[],
            sparsity_ratio=0.0,
            dimensionality_ratio=0.1,
            dataset_size_mb=0.1,
            has_temporal_structure=False,
            has_graph_structure=False,
        )

        recommended = automl_service.recommend_algorithms(
            small_profile, max_algorithms=5
        )

        # Should prefer simpler algorithms for small datasets
        assert "LOF" in recommended or "KNN" in recommended

    def test_large_dataset_recommendations(self, automl_service):
        """Test recommendations for large datasets."""
        large_profile = DatasetProfile(
            n_samples=50000,  # Large dataset
            n_features=20,
            contamination_estimate=0.1,
            feature_types={"num_1": "numerical"},
            missing_values_ratio=0.0,
            categorical_features=[],
            numerical_features=["num_1"],
            time_series_features=[],
            sparsity_ratio=0.0,
            dimensionality_ratio=0.0004,
            dataset_size_mb=100.0,
            has_temporal_structure=False,
            has_graph_structure=False,
        )

        recommended = automl_service.recommend_algorithms(
            large_profile, max_algorithms=5
        )

        # Should prefer scalable algorithms for large datasets
        assert "IsolationForest" in recommended or "COPOD" in recommended

    @pytest.mark.asyncio
    async def test_hyperparameter_optimization_without_optuna(
        self, automl_service, sample_dataset
    ):
        """Test hyperparameter optimization when Optuna is not available."""
        automl_service.dataset_repository.get = AsyncMock(return_value=sample_dataset)

        # Mock Optuna as unavailable
        with patch(
            "pynomaly.application.services.automl_service.OPTUNA_AVAILABLE", False
        ):
            with pytest.raises(AutoMLError, match="Optuna is required"):
                await automl_service.optimize_hyperparameters(
                    "test_id", "IsolationForest", OptimizationObjective.AUC
                )

    @pytest.mark.asyncio
    async def test_hyperparameter_optimization_with_mocked_optuna(
        self, automl_service, sample_dataset
    ):
        """Test hyperparameter optimization with mocked Optuna."""
        automl_service.dataset_repository.get = AsyncMock(return_value=sample_dataset)

        # Mock Optuna study
        mock_study = Mock()
        mock_study.best_params = {"contamination": 0.1, "n_estimators": 100}
        mock_study.best_value = 0.85
        mock_study.trials = [Mock() for _ in range(10)]

        with patch(
            "pynomaly.application.services.automl_service.OPTUNA_AVAILABLE", True
        ):
            with patch(
                "pynomaly.application.services.automl_service.optuna"
            ) as mock_optuna:
                mock_optuna.create_study.return_value = mock_study

                result = await automl_service.optimize_hyperparameters(
                    "test_id", "IsolationForest", OptimizationObjective.AUC
                )

                assert isinstance(result, AutoMLResult)
                assert result.best_algorithm == "IsolationForest"
                assert result.best_score == 0.85
                assert result.trials_completed == 10

    @pytest.mark.asyncio
    async def test_auto_select_and_optimize(self, automl_service, sample_dataset):
        """Test complete AutoML pipeline."""
        automl_service.dataset_repository.get = AsyncMock(return_value=sample_dataset)

        # Mock individual optimization results
        mock_result_1 = AutoMLResult(
            best_algorithm="IsolationForest",
            best_params={"contamination": 0.1, "n_estimators": 100},
            best_score=0.85,
            optimization_time=30.0,
            trials_completed=10,
            algorithm_rankings=[("IsolationForest", 0.85)],
        )

        mock_result_2 = AutoMLResult(
            best_algorithm="LOF",
            best_params={"contamination": 0.1, "n_neighbors": 20},
            best_score=0.80,
            optimization_time=25.0,
            trials_completed=10,
            algorithm_rankings=[("LOF", 0.80)],
        )

        # Mock the optimize_hyperparameters method
        automl_service.optimize_hyperparameters = AsyncMock(
            side_effect=[mock_result_1, mock_result_2]
        )

        # Test complete AutoML
        result = await automl_service.auto_select_and_optimize(
            "test_id",
            objective=OptimizationObjective.AUC,
            max_algorithms=2,
            enable_ensemble=True,
        )

        assert isinstance(result, AutoMLResult)
        assert result.best_algorithm == "IsolationForest"  # Best score
        assert result.best_score == 0.85
        assert len(result.algorithm_rankings) == 2
        assert result.ensemble_config is not None  # Ensemble should be created

    @pytest.mark.asyncio
    async def test_create_optimized_detector(self, automl_service):
        """Test detector creation from AutoML results."""
        automl_result = AutoMLResult(
            best_algorithm="IsolationForest",
            best_params={"contamination": 0.1, "n_estimators": 100},
            best_score=0.85,
            optimization_time=30.0,
            trials_completed=10,
            algorithm_rankings=[("IsolationForest", 0.85)],
        )

        automl_service.detector_repository.save = AsyncMock()

        detector_id = await automl_service.create_optimized_detector(
            automl_result, "Custom Detector Name"
        )

        assert isinstance(detector_id, str)
        automl_service.detector_repository.save.assert_called_once()

    def test_ensemble_config_creation(self, automl_service):
        """Test ensemble configuration creation."""
        optimization_results = [
            AutoMLResult(
                best_algorithm="IsolationForest",
                best_params={},
                best_score=0.85,
                optimization_time=30.0,
                trials_completed=10,
                algorithm_rankings=[],
            ),
            AutoMLResult(
                best_algorithm="LOF",
                best_params={},
                best_score=0.80,
                optimization_time=25.0,
                trials_completed=10,
                algorithm_rankings=[],
            ),
            AutoMLResult(
                best_algorithm="OneClassSVM",
                best_params={},
                best_score=0.75,
                optimization_time=35.0,
                trials_completed=10,
                algorithm_rankings=[],
            ),
        ]

        ensemble_config = automl_service._create_ensemble_config(optimization_results)

        assert ensemble_config["method"] == "weighted_voting"
        assert len(ensemble_config["algorithms"]) == 3
        assert ensemble_config["voting_strategy"] == "soft"
        assert ensemble_config["normalize_scores"] is True

        # Check weights sum to 1 (approximately)
        weights = [alg["weight"] for alg in ensemble_config["algorithms"]]
        assert abs(sum(weights) - 1.0) < 1e-6

    def test_optimization_summary(self, automl_service):
        """Test optimization summary generation."""
        automl_result = AutoMLResult(
            best_algorithm="IsolationForest",
            best_params={"contamination": 0.1, "n_estimators": 100},
            best_score=0.85,
            optimization_time=30.0,
            trials_completed=10,
            algorithm_rankings=[("IsolationForest", 0.85), ("LOF", 0.80)],
            ensemble_config={"method": "weighted_voting"},
        )

        summary = automl_service.get_optimization_summary(automl_result)

        assert summary["best_algorithm"] == "IsolationForest"
        assert summary["best_score"] == 0.85
        assert summary["has_ensemble"] is True
        assert isinstance(summary["recommendations"], list)

    def test_low_score_recommendations(self, automl_service):
        """Test recommendations for low-performing models."""
        low_score_result = AutoMLResult(
            best_algorithm="IsolationForest",
            best_params={},
            best_score=0.6,  # Low score
            optimization_time=30.0,
            trials_completed=10,
            algorithm_rankings=[],
        )

        summary = automl_service.get_optimization_summary(low_score_result)

        assert any("more training data" in rec for rec in summary["recommendations"])

    @pytest.mark.asyncio
    async def test_dataset_not_found_error(self, automl_service):
        """Test error handling when dataset is not found."""
        automl_service.dataset_repository.get = AsyncMock(return_value=None)

        with pytest.raises(AutoMLError, match="Dataset .* not found"):
            await automl_service.profile_dataset("nonexistent_id")

    @pytest.mark.asyncio
    async def test_unsupported_algorithm_error(self, automl_service, sample_dataset):
        """Test error handling for unsupported algorithms."""
        automl_service.dataset_repository.get = AsyncMock(return_value=sample_dataset)

        with patch(
            "pynomaly.application.services.automl_service.OPTUNA_AVAILABLE", True
        ):
            with pytest.raises(AutoMLError, match="Algorithm .* not supported"):
                await automl_service.optimize_hyperparameters(
                    "test_id", "UnsupportedAlgorithm", OptimizationObjective.AUC
                )

    def test_algorithm_config_completeness(self, automl_service):
        """Test that all algorithm configs have required fields."""
        for name, config in automl_service.algorithm_configs.items():
            assert isinstance(config.name, str)
            assert isinstance(config.family, AlgorithmFamily)
            assert isinstance(config.adapter_type, str)
            assert isinstance(config.default_params, dict)
            assert isinstance(config.param_space, dict)
            assert 0.0 <= config.complexity_score <= 1.0
            assert config.training_time_factor > 0
            assert config.memory_factor > 0
            assert config.recommended_min_samples > 0
            assert config.recommended_max_samples > config.recommended_min_samples

    def test_dataset_profile_complexity_calculation(self):
        """Test dataset complexity score calculation."""
        # Simple dataset
        simple_profile = DatasetProfile(
            n_samples=100,
            n_features=3,
            contamination_estimate=0.1,
            feature_types={"num_1": "numerical"},
            missing_values_ratio=0.0,
            categorical_features=[],
            numerical_features=["num_1"],
            time_series_features=[],
            sparsity_ratio=0.0,
            dimensionality_ratio=0.03,
            dataset_size_mb=0.1,
            has_temporal_structure=False,
            has_graph_structure=False,
        )

        # Complex dataset
        complex_profile = DatasetProfile(
            n_samples=10000,
            n_features=500,
            contamination_estimate=0.1,
            feature_types={"num_1": "numerical"},
            missing_values_ratio=0.3,
            categorical_features=[],
            numerical_features=["num_1"],
            time_series_features=[],
            sparsity_ratio=0.5,
            dimensionality_ratio=0.05,
            dataset_size_mb=100.0,
            has_temporal_structure=False,
            has_graph_structure=False,
        )

        assert simple_profile.complexity_score < complex_profile.complexity_score
        assert 0.0 <= simple_profile.complexity_score <= 1.0
        assert 0.0 <= complex_profile.complexity_score <= 1.0

    @pytest.mark.integration
    def test_real_automl_workflow_simulation(self, automl_service, sample_dataset):
        """Test a realistic AutoML workflow with mocked components."""
        # This test simulates the complete workflow without external dependencies

        # Mock dataset repository
        automl_service.dataset_repository.get = AsyncMock(return_value=sample_dataset)

        # Mock adapter registry
        mock_adapter = Mock()
        mock_adapter.train.return_value = True
        mock_adapter.predict.return_value = (
            np.random.choice([0, 1], size=330),  # predictions
            np.random.rand(330),  # scores
        )
        automl_service.adapter_registry.get_adapter.return_value = mock_adapter

        import asyncio

        # Test the complete workflow
        async def run_workflow():
            # 1. Profile dataset
            profile = await automl_service.profile_dataset("test_id")
            assert profile.n_samples == 330

            # 2. Get recommendations
            recommendations = automl_service.recommend_algorithms(
                profile, max_algorithms=3
            )
            assert len(recommendations) > 0

            # 3. Test single algorithm optimization (mocked)
            with patch(
                "pynomaly.application.services.automl_service.OPTUNA_AVAILABLE", True
            ):
                with patch(
                    "pynomaly.application.services.automl_service.optuna"
                ) as mock_optuna:
                    mock_study = Mock()
                    mock_study.best_params = {"contamination": 0.1}
                    mock_study.best_value = 0.8
                    mock_study.trials = [Mock() for _ in range(5)]
                    mock_optuna.create_study.return_value = mock_study

                    result = await automl_service.optimize_hyperparameters(
                        "test_id", recommendations[0], OptimizationObjective.AUC
                    )

                    assert result.best_score == 0.8
                    assert result.best_algorithm == recommendations[0]

            return True

        success = asyncio.run(run_workflow())
        assert success
