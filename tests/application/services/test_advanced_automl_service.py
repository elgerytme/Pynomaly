"""Tests for Advanced AutoML Service."""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from pynomaly.application.dto.optimization_dto import (
    OptimizationConfigDTO,
    OptimizationObjectiveDTO,
    ResourceConstraintsDTO,
)
from pynomaly.application.services.advanced_automl_service import (
    AdvancedAutoMLService,
    OptimizationHistory,
    OptimizationObjective,
    ResourceConstraints,
)
from pynomaly.domain.entities import Dataset


@pytest.fixture
def sample_dataset():
    """Create a sample dataset for testing."""
    np.random.seed(42)
    data = np.random.rand(100, 5)
    return Dataset(
        name="test_dataset", data=data, features=["f1", "f2", "f3", "f4", "f5"]
    )


@pytest.fixture
def automl_service():
    """Create AutoML service for testing."""
    return AdvancedAutoMLService()


@pytest.fixture
def sample_objectives():
    """Create sample optimization objectives."""
    return [
        OptimizationObjective(
            name="accuracy",
            weight=0.6,
            direction="maximize",
            description="Detection accuracy",
        ),
        OptimizationObjective(
            name="speed", weight=0.4, direction="maximize", description="Training speed"
        ),
    ]


@pytest.fixture
def sample_constraints():
    """Create sample resource constraints."""
    return ResourceConstraints(
        max_time_seconds=60,
        max_trials=10,
        max_memory_mb=1024,
        max_cpu_cores=2,
        gpu_available=False,
        prefer_speed=True,
    )


class TestAdvancedAutoMLService:
    """Test cases for Advanced AutoML Service."""

    def test_initialization(self, tmp_path):
        """Test service initialization."""
        service = AdvancedAutoMLService(
            optimization_storage_path=tmp_path,
            enable_distributed=False,
            n_parallel_jobs=1,
        )

        assert service.storage_path == tmp_path
        assert service.enable_distributed is False
        assert service.n_parallel_jobs == 1
        assert len(service.optimization_history) == 0
        assert "IsolationForest" in service.algorithm_knowledge
        assert len(service.default_objectives) == 4

    def test_dataset_characteristics_analysis(self, automl_service, sample_dataset):
        """Test dataset characteristics analysis."""
        characteristics = automl_service._analyze_dataset_characteristics(
            sample_dataset
        )

        assert "n_samples" in characteristics
        assert "n_features" in characteristics
        assert "size_category" in characteristics
        assert "feature_types" in characteristics
        assert "data_distribution" in characteristics
        assert "sparsity" in characteristics
        assert "correlation_structure" in characteristics
        assert "outlier_characteristics" in characteristics

        assert characteristics["n_samples"] == 100
        assert characteristics["n_features"] == 5
        assert characteristics["size_category"] == "small"

    def test_categorize_size(self, automl_service):
        """Test dataset size categorization."""
        assert automl_service._categorize_size(500, 5) == "small"
        assert automl_service._categorize_size(5000, 50) == "medium"
        assert automl_service._categorize_size(50000, 500) == "large"
        assert automl_service._categorize_size(200000, 2000) == "very_large"

    def test_feature_type_analysis(self, automl_service):
        """Test feature type analysis."""
        # Test with integer-like data
        integer_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        types = automl_service._analyze_feature_types(integer_data)
        assert "integer_ratio" in types
        assert "continuous_ratio" in types
        assert types["integer_ratio"] > 0.5

    def test_distribution_analysis(self, automl_service, sample_dataset):
        """Test data distribution analysis."""
        distribution = automl_service._analyze_distribution(sample_dataset.data)

        assert "mean" in distribution
        assert "std" in distribution
        assert "skewness" in distribution
        assert "kurtosis" in distribution

        assert isinstance(distribution["mean"], float)
        assert isinstance(distribution["std"], float)

    def test_sparsity_calculation(self, automl_service):
        """Test sparsity calculation."""
        # Test with non-sparse data
        dense_data = np.ones((10, 5))
        sparsity = automl_service._calculate_sparsity(dense_data)
        assert sparsity == 0.0

        # Test with sparse data
        sparse_data = np.zeros((10, 5))
        sparse_data[0, 0] = 1
        sparsity = automl_service._calculate_sparsity(sparse_data)
        assert sparsity > 0.9

    def test_correlation_analysis(self, automl_service):
        """Test correlation structure analysis."""
        # Test with uncorrelated data
        uncorr_data = np.random.rand(100, 3)
        corr_stats = automl_service._analyze_correlations(uncorr_data)

        assert "max_correlation" in corr_stats
        assert "mean_correlation" in corr_stats
        assert "high_correlation_ratio" in corr_stats

        # Test with single feature
        single_data = np.random.rand(100, 1)
        single_corr = automl_service._analyze_correlations(single_data)
        assert single_corr["max_correlation"] == 0.0

    def test_outlier_pattern_analysis(self, automl_service, sample_dataset):
        """Test outlier pattern analysis."""
        outlier_stats = automl_service._analyze_outlier_patterns(sample_dataset.data)

        assert "mean_outlier_ratio" in outlier_stats
        assert "max_outlier_ratio" in outlier_stats
        assert outlier_stats["mean_outlier_ratio"] >= 0.0

    def test_dataset_similarity_calculation(self, automl_service):
        """Test dataset similarity calculation."""
        chars1 = {
            "n_samples": 1000,
            "n_features": 10,
            "data_distribution": {"std": 1.0},
        }
        chars2 = {
            "n_samples": 1200,
            "n_features": 12,
            "data_distribution": {"std": 1.1},
        }

        similarity = automl_service._calculate_dataset_similarity(chars1, chars2)
        assert 0.0 <= similarity <= 1.0

    def test_predict_optimal_parameters_no_history(self, automl_service):
        """Test parameter prediction with no history."""
        dataset_chars = {"n_samples": 100, "n_features": 5}
        params = automl_service._predict_optimal_parameters(
            dataset_chars, "IsolationForest"
        )
        assert params is None

    def test_predict_optimal_parameters_with_history(self, automl_service):
        """Test parameter prediction with optimization history."""
        # Add mock history
        history = OptimizationHistory(
            dataset_characteristics={
                "n_samples": 100,
                "n_features": 5,
                "size_category": "small",
                "data_distribution": {"std": 1.0},
            },
            algorithm_name="IsolationForest",
            best_parameters={"n_estimators": 150, "contamination": 0.1},
            performance_metrics={"accuracy": 0.85},
            optimization_time=120.0,
            resource_usage={"memory": 100.0},
        )
        automl_service.optimization_history.append(history)

        dataset_chars = {
            "n_samples": 95,
            "n_features": 5,
            "size_category": "small",
            "data_distribution": {"std": 0.9},
        }

        params = automl_service._predict_optimal_parameters(
            dataset_chars, "IsolationForest"
        )

        assert params is not None
        assert "n_estimators" in params
        assert "contamination" in params

    def test_generate_trial_parameters(self, automl_service):
        """Test trial parameter generation."""
        with patch("optuna.Trial") as mock_trial:
            # Mock trial methods
            mock_trial.suggest_int.return_value = 100
            mock_trial.suggest_float.return_value = 0.1
            mock_trial.suggest_categorical.return_value = "auto"

            # Test IsolationForest parameters
            params = automl_service._generate_trial_parameters(
                mock_trial, "IsolationForest", None
            )

            assert "n_estimators" in params
            assert "contamination" in params
            assert "random_state" in params

    def test_create_detector_from_params(self, automl_service):
        """Test detector creation from parameters."""
        params = {"n_estimators": 100, "contamination": 0.1}
        detector = automl_service._create_detector_from_params(
            "IsolationForest", params
        )

        assert detector is not None
        # Check that parameters were applied
        assert hasattr(detector, "algorithm_params")

    def test_generate_synthetic_anomalies(self, automl_service):
        """Test synthetic anomaly generation."""
        normal_data = np.random.rand(100, 5)
        anomalies = automl_service._generate_synthetic_anomalies(normal_data, 10)

        assert anomalies.shape == (10, 5)
        assert anomalies.dtype == normal_data.dtype

    def test_evaluate_accuracy(self, automl_service):
        """Test accuracy evaluation."""
        # Create test data with labels
        test_dataset = Dataset(
            name="test",
            data=np.random.rand(100, 5),
            features=["f1", "f2", "f3", "f4", "f5"],
            metadata={"labels": np.concatenate([np.zeros(90), np.ones(10)])},
        )

        scores = np.random.rand(100)
        accuracy = automl_service._evaluate_accuracy(scores, test_dataset)

        assert 0.0 <= accuracy <= 1.0

    def test_calculate_interpretability(self, automl_service):
        """Test interpretability calculation."""
        # Test different algorithms
        if_interp = automl_service._calculate_interpretability(
            "IsolationForest", {"n_estimators": 100}
        )
        lof_interp = automl_service._calculate_interpretability(
            "LocalOutlierFactor", {"n_neighbors": 20}
        )
        svm_interp = automl_service._calculate_interpretability(
            "OneClassSVM", {"kernel": "linear"}
        )

        assert 0.0 <= if_interp <= 1.0
        assert 0.0 <= lof_interp <= 1.0
        assert 0.0 <= svm_interp <= 1.0

        # IsolationForest should be more interpretable than OneClassSVM
        assert if_interp > svm_interp

    def test_get_default_parameters(self, automl_service):
        """Test default parameter retrieval."""
        if_defaults = automl_service._get_default_parameters("IsolationForest")
        lof_defaults = automl_service._get_default_parameters("LocalOutlierFactor")
        svm_defaults = automl_service._get_default_parameters("OneClassSVM")

        assert "n_estimators" in if_defaults
        assert "contamination" in if_defaults

        assert "n_neighbors" in lof_defaults
        assert "contamination" in lof_defaults

        assert "nu" in svm_defaults
        assert "gamma" in svm_defaults

    def test_generate_optimization_report(
        self, automl_service, sample_objectives, sample_constraints
    ):
        """Test optimization report generation."""
        with patch("optuna.Study") as mock_study:
            # Mock study with trials
            mock_trial = Mock()
            mock_trial.params = {"n_estimators": 100, "contamination": 0.1}
            mock_trial.values = [0.85, 0.7]
            mock_trial.number = 1

            mock_study.trials = [mock_trial]
            mock_study.best_trials = [mock_trial]

            dataset_chars = {"n_samples": 100, "n_features": 5}

            report = automl_service._generate_optimization_report(
                mock_study, sample_objectives, sample_constraints, 120.0, dataset_chars
            )

            assert "optimization_summary" in report
            assert "dataset_characteristics" in report
            assert "objectives" in report
            assert "constraints" in report
            assert "best_metrics" in report
            assert "best_parameters" in report
            assert "pareto_optimal_solutions" in report

    def test_save_and_load_optimization_history(self, automl_service, tmp_path):
        """Test optimization history persistence."""
        automl_service.storage_path = tmp_path

        # Add test history
        history = OptimizationHistory(
            dataset_characteristics={"n_samples": 100, "n_features": 5},
            algorithm_name="IsolationForest",
            best_parameters={"n_estimators": 100},
            performance_metrics={"accuracy": 0.85},
            optimization_time=120.0,
            resource_usage={"memory": 100.0},
        )
        automl_service.optimization_history.append(history)

        # Save history
        automl_service.save_optimization_history()

        # Clear and reload
        automl_service.optimization_history = []
        automl_service.load_optimization_history()

        assert len(automl_service.optimization_history) == 1
        loaded_history = automl_service.optimization_history[0]
        assert loaded_history.algorithm_name == "IsolationForest"
        assert loaded_history.best_parameters == {"n_estimators": 100}

    @pytest.mark.asyncio
    async def test_analyze_optimization_trends_no_history(self, automl_service):
        """Test trend analysis with no history."""
        trends = await automl_service.analyze_optimization_trends()
        assert "message" in trends
        assert "No optimization history available" in trends["message"]

    @pytest.mark.asyncio
    async def test_analyze_optimization_trends_with_history(self, automl_service):
        """Test trend analysis with optimization history."""
        # Add mock history with multiple entries
        for i in range(5):
            history = OptimizationHistory(
                dataset_characteristics={"n_samples": 100, "n_features": 5},
                algorithm_name="IsolationForest",
                best_parameters={"n_estimators": 100 + i * 10},
                performance_metrics={"accuracy": 0.8 + i * 0.01},
                optimization_time=120.0,
                resource_usage={"memory": 100.0},
            )
            automl_service.optimization_history.append(history)

        trends = await automl_service.analyze_optimization_trends()

        assert "algorithm_trends" in trends
        assert "total_optimizations" in trends
        assert "learning_insights" in trends
        assert "IsolationForest" in trends["algorithm_trends"]

        if_trends = trends["algorithm_trends"]["IsolationForest"]
        assert "total_optimizations" in if_trends
        assert "average_performance" in if_trends
        assert "performance_improvement" in if_trends

    def test_calculate_trend(self, automl_service):
        """Test trend calculation."""
        # Improving trend
        improving_values = [0.7, 0.75, 0.8, 0.85, 0.9]
        trend = automl_service._calculate_trend(improving_values)
        assert trend == "improving"

        # Declining trend
        declining_values = [0.9, 0.85, 0.8, 0.75, 0.7]
        trend = automl_service._calculate_trend(declining_values)
        assert trend == "declining"

        # Stable trend
        stable_values = [0.8, 0.81, 0.8, 0.79, 0.8]
        trend = automl_service._calculate_trend(stable_values)
        assert trend == "stable"

    def test_analyze_parameter_preferences(self, automl_service):
        """Test parameter preference analysis."""
        param_data = {
            "n_estimators": [100, 120, 110, 150, 130],
            "algorithm": ["auto", "ball_tree", "auto", "auto", "kd_tree"],
        }

        preferences = automl_service._analyze_parameter_preferences(param_data)

        assert "n_estimators" in preferences
        assert "algorithm" in preferences

        # Numeric parameter
        n_est_pref = preferences["n_estimators"]
        assert n_est_pref["type"] == "numeric"
        assert "mean" in n_est_pref
        assert "std" in n_est_pref

        # Categorical parameter
        algo_pref = preferences["algorithm"]
        assert algo_pref["type"] == "categorical"
        assert "most_common" in algo_pref

    def test_calculate_learning_rate(self, automl_service):
        """Test learning rate calculation."""
        # Improving performance
        improving_values = [0.7, 0.75, 0.8, 0.85]
        rate = automl_service._calculate_learning_rate(improving_values)
        assert rate > 0

        # Declining performance
        declining_values = [0.9, 0.85, 0.8, 0.75]
        rate = automl_service._calculate_learning_rate(declining_values)
        assert rate < 0

    def test_generate_learning_insights(self, automl_service):
        """Test learning insights generation."""
        # Create mock algorithm trends data
        algorithm_trends = {
            "IsolationForest": {
                "optimizations": [
                    Mock(performance_metrics={"accuracy": 0.7}),
                    Mock(performance_metrics={"accuracy": 0.75}),
                    Mock(performance_metrics={"accuracy": 0.8}),
                    Mock(performance_metrics={"accuracy": 0.82}),
                    Mock(performance_metrics={"accuracy": 0.85}),
                ]
            }
        }

        insights = automl_service._generate_learning_insights(algorithm_trends)

        assert isinstance(insights, list)
        assert len(insights) > 0

        # Should detect improvement
        assert any("Learning progress detected" in insight for insight in insights)


class TestOptimizationDTOs:
    """Test optimization DTOs."""

    def test_optimization_objective_dto(self):
        """Test OptimizationObjectiveDTO."""
        objective = OptimizationObjectiveDTO(
            name="accuracy",
            weight=0.6,
            direction="maximize",
            threshold=0.8,
            description="Detection accuracy",
        )

        assert objective.name == "accuracy"
        assert objective.weight == 0.6
        assert objective.direction == "maximize"
        assert objective.threshold == 0.8

    def test_resource_constraints_dto(self):
        """Test ResourceConstraintsDTO."""
        constraints = ResourceConstraintsDTO(
            max_time_seconds=3600,
            max_trials=100,
            max_memory_mb=4096,
            max_cpu_cores=4,
            gpu_available=True,
            prefer_speed=False,
        )

        assert constraints.max_time_seconds == 3600
        assert constraints.max_trials == 100
        assert constraints.gpu_available is True

    def test_optimization_config_dto(self):
        """Test OptimizationConfigDTO."""
        objectives = [
            OptimizationObjectiveDTO(name="accuracy", weight=0.6, direction="maximize")
        ]
        constraints = ResourceConstraintsDTO()

        config = OptimizationConfigDTO(
            algorithm_name="IsolationForest",
            objectives=objectives,
            constraints=constraints,
            enable_learning=True,
            enable_distributed=False,
            n_parallel_jobs=1,
        )

        assert config.algorithm_name == "IsolationForest"
        assert len(config.objectives) == 1
        assert config.enable_learning is True


@pytest.mark.integration
class TestAdvancedAutoMLIntegration:
    """Integration tests for Advanced AutoML."""

    def test_feature_flag_integration(self):
        """Test integration with feature flags."""
        from pynomaly.infrastructure.config.feature_flags import feature_flags

        assert feature_flags.is_enabled("advanced_automl")
        assert feature_flags.is_enabled("meta_learning")
        assert feature_flags.is_enabled("ensemble_optimization")

    @pytest.mark.skipif(
        not pytest.importorskip("optuna", reason="Optuna not available"),
        reason="Requires Optuna for full integration testing",
    )
    @pytest.mark.asyncio
    async def test_full_optimization_workflow(self, sample_dataset, tmp_path):
        """Test complete optimization workflow (requires Optuna)."""
        service = AdvancedAutoMLService(optimization_storage_path=tmp_path)

        objectives = [
            OptimizationObjective(name="accuracy", weight=1.0, direction="maximize")
        ]

        constraints = ResourceConstraints(
            max_time_seconds=30, max_trials=5, max_memory_mb=1024
        )

        try:
            detector, report = await service.optimize_detector_advanced(
                dataset=sample_dataset,
                algorithm_name="IsolationForest",
                objectives=objectives,
                constraints=constraints,
                enable_learning=True,
            )

            assert detector is not None
            assert "optimization_summary" in report
            assert "best_parameters" in report
            assert report["optimization_summary"]["total_trials"] <= 5

        except RuntimeError as e:
            if "Optuna is required" in str(e):
                pytest.skip("Optuna not available for integration test")
            else:
                raise
