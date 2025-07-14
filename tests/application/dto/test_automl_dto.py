"""
Comprehensive tests for AutoML DTOs.

This module tests all AutoML-related Data Transfer Objects to ensure proper validation,
serialization, and behavior across all use cases including algorithm recommendations,
optimization requests, ensemble configuration, and experiment tracking.
"""

import pytest
from pydantic import ValidationError

from pynomaly.application.dto.automl_dto import (
    AlgorithmRecommendationDTO,
    AlgorithmRecommendationRequestDTO,
    AutoMLProfileRequestDTO,
    AutoMLRequestDTO,
    AutoMLResponseDTO,
    AutoMLResultDTO,
    DatasetProfileDTO,
    EnsembleConfigDTO,
    ExperimentTrackingRequestDTO,
    ExperimentTrackingResponseDTO,
    HyperparameterOptimizationRequestDTO,
    HyperparameterOptimizationResponseDTO,
    HyperparameterSpaceDTO,
    OptimizationTrialDTO,
)


class TestAlgorithmRecommendationRequestDTO:
    """Test suite for AlgorithmRecommendationRequestDTO."""

    def test_basic_creation(self):
        """Test basic recommendation request creation."""
        request = AlgorithmRecommendationRequestDTO(dataset_id="dataset_123")

        assert request.dataset_id == "dataset_123"
        assert request.max_recommendations == 5  # Default
        assert request.performance_priority == 0.8  # Default
        assert request.speed_priority == 0.2  # Default
        assert request.include_experimental is False  # Default
        assert request.exclude_algorithms == []  # Default

    def test_complete_creation(self):
        """Test creation with all parameters."""
        exclude_algs = ["algorithm_a", "algorithm_b"]
        request = AlgorithmRecommendationRequestDTO(
            dataset_id="dataset_456",
            max_recommendations=10,
            performance_priority=0.9,
            speed_priority=0.1,
            include_experimental=True,
            exclude_algorithms=exclude_algs,
        )

        assert request.dataset_id == "dataset_456"
        assert request.max_recommendations == 10
        assert request.performance_priority == 0.9
        assert request.speed_priority == 0.1
        assert request.include_experimental is True
        assert request.exclude_algorithms == exclude_algs

    def test_max_recommendations_validation(self):
        """Test max_recommendations validation bounds."""
        # Test invalid values
        with pytest.raises(ValidationError):
            AlgorithmRecommendationRequestDTO(
                dataset_id="test", max_recommendations=0
            )

        with pytest.raises(ValidationError):
            AlgorithmRecommendationRequestDTO(
                dataset_id="test", max_recommendations=21
            )

        # Test valid values
        for max_recs in [1, 10, 20]:
            request = AlgorithmRecommendationRequestDTO(
                dataset_id="test", max_recommendations=max_recs
            )
            assert request.max_recommendations == max_recs

    def test_priority_validation(self):
        """Test priority field validation bounds."""
        # Test invalid performance priority
        with pytest.raises(ValidationError):
            AlgorithmRecommendationRequestDTO(
                dataset_id="test", performance_priority=-0.1
            )

        with pytest.raises(ValidationError):
            AlgorithmRecommendationRequestDTO(
                dataset_id="test", performance_priority=1.1
            )

        # Test invalid speed priority
        with pytest.raises(ValidationError):
            AlgorithmRecommendationRequestDTO(dataset_id="test", speed_priority=-0.1)

        with pytest.raises(ValidationError):
            AlgorithmRecommendationRequestDTO(dataset_id="test", speed_priority=1.1)

        # Test valid priority values
        for priority in [0.0, 0.5, 1.0]:
            request = AlgorithmRecommendationRequestDTO(
                dataset_id="test",
                performance_priority=priority,
                speed_priority=1.0 - priority,
            )
            assert request.performance_priority == priority
            assert request.speed_priority == 1.0 - priority


class TestDatasetProfileDTO:
    """Test suite for DatasetProfileDTO."""

    def test_basic_creation(self):
        """Test basic dataset profile creation."""
        feature_types = {"feature1": "numeric", "feature2": "categorical"}
        profile = DatasetProfileDTO(
            n_samples=1000,
            n_features=2,
            contamination_estimate=0.1,
            feature_types=feature_types,
            missing_values_ratio=0.05,
            sparsity_ratio=0.0,
            dimensionality_ratio=0.002,
            dataset_size_mb=5.2,
            complexity_score=0.3,
        )

        assert profile.n_samples == 1000
        assert profile.n_features == 2
        assert profile.contamination_estimate == 0.1
        assert profile.feature_types == feature_types
        assert profile.missing_values_ratio == 0.05
        assert profile.categorical_features == []  # Default
        assert profile.numerical_features == []  # Default
        assert profile.time_series_features == []  # Default
        assert profile.has_temporal_structure is False  # Default
        assert profile.has_graph_structure is False  # Default

    def test_complete_creation(self):
        """Test creation with all fields."""
        feature_types = {"f1": "numeric", "f2": "categorical", "f3": "text"}
        categorical_features = ["f2"]
        numerical_features = ["f1"]
        time_series_features = ["f3"]

        profile = DatasetProfileDTO(
            n_samples=5000,
            n_features=3,
            contamination_estimate=0.15,
            feature_types=feature_types,
            missing_values_ratio=0.1,
            categorical_features=categorical_features,
            numerical_features=numerical_features,
            time_series_features=time_series_features,
            sparsity_ratio=0.2,
            dimensionality_ratio=0.0006,
            dataset_size_mb=25.8,
            has_temporal_structure=True,
            has_graph_structure=False,
            complexity_score=0.7,
        )

        assert profile.categorical_features == categorical_features
        assert profile.numerical_features == numerical_features
        assert profile.time_series_features == time_series_features
        assert profile.has_temporal_structure is True
        assert profile.has_graph_structure is False
        assert profile.complexity_score == 0.7


class TestAlgorithmRecommendationDTO:
    """Test suite for AlgorithmRecommendationDTO."""

    def test_basic_creation(self):
        """Test basic algorithm recommendation creation."""
        params = {"n_estimators": 100, "contamination": 0.1}
        reasoning = ["High dimensional data", "Fast execution required"]

        rec = AlgorithmRecommendationDTO(
            algorithm_name="isolation_forest",
            score=0.85,
            family="tree_based",
            complexity_score=0.3,
            recommended_params=params,
            reasoning=reasoning,
        )

        assert rec.algorithm_name == "isolation_forest"
        assert rec.score == 0.85
        assert rec.family == "tree_based"
        assert rec.complexity_score == 0.3
        assert rec.recommended_params == params
        assert rec.reasoning == reasoning

    def test_empty_reasoning(self):
        """Test with empty reasoning list."""
        rec = AlgorithmRecommendationDTO(
            algorithm_name="one_class_svm",
            score=0.7,
            family="svm",
            complexity_score=0.8,
            recommended_params={},
        )

        assert rec.reasoning == []  # Default


class TestAutoMLRequestDTO:
    """Test suite for AutoMLRequestDTO."""

    def test_basic_creation(self):
        """Test basic AutoML request creation."""
        request = AutoMLRequestDTO(dataset_id="dataset_789")

        assert request.dataset_id == "dataset_789"
        assert request.objective == "auc"  # Default
        assert request.max_algorithms == 3  # Default
        assert request.max_optimization_time == 3600  # Default
        assert request.n_trials == 100  # Default
        assert request.enable_ensemble is True  # Default
        assert request.detector_name is None  # Default
        assert request.cross_validation_folds == 3  # Default
        assert request.random_state == 42  # Default

    def test_complete_creation(self):
        """Test creation with all parameters."""
        request = AutoMLRequestDTO(
            dataset_id="dataset_complete",
            objective="f1",
            max_algorithms=5,
            max_optimization_time=7200,
            n_trials=200,
            enable_ensemble=False,
            detector_name="custom_detector",
            cross_validation_folds=5,
            random_state=123,
        )

        assert request.objective == "f1"
        assert request.max_algorithms == 5
        assert request.max_optimization_time == 7200
        assert request.n_trials == 200
        assert request.enable_ensemble is False
        assert request.detector_name == "custom_detector"
        assert request.cross_validation_folds == 5
        assert request.random_state == 123

    def test_max_algorithms_validation(self):
        """Test max_algorithms validation bounds."""
        # Test invalid values
        with pytest.raises(ValidationError):
            AutoMLRequestDTO(dataset_id="test", max_algorithms=0)

        with pytest.raises(ValidationError):
            AutoMLRequestDTO(dataset_id="test", max_algorithms=11)

        # Test valid values
        for max_algs in [1, 5, 10]:
            request = AutoMLRequestDTO(dataset_id="test", max_algorithms=max_algs)
            assert request.max_algorithms == max_algs

    def test_optimization_time_validation(self):
        """Test max_optimization_time validation bounds."""
        # Test invalid values
        with pytest.raises(ValidationError):
            AutoMLRequestDTO(dataset_id="test", max_optimization_time=59)

        with pytest.raises(ValidationError):
            AutoMLRequestDTO(dataset_id="test", max_optimization_time=86401)

        # Test valid values
        for time_limit in [60, 3600, 86400]:
            request = AutoMLRequestDTO(dataset_id="test", max_optimization_time=time_limit)
            assert request.max_optimization_time == time_limit

    def test_n_trials_validation(self):
        """Test n_trials validation bounds."""
        # Test invalid values
        with pytest.raises(ValidationError):
            AutoMLRequestDTO(dataset_id="test", n_trials=9)

        with pytest.raises(ValidationError):
            AutoMLRequestDTO(dataset_id="test", n_trials=1001)

        # Test valid values
        for trials in [10, 100, 1000]:
            request = AutoMLRequestDTO(dataset_id="test", n_trials=trials)
            assert request.n_trials == trials

    def test_cross_validation_folds_validation(self):
        """Test cross_validation_folds validation bounds."""
        # Test invalid values
        with pytest.raises(ValidationError):
            AutoMLRequestDTO(dataset_id="test", cross_validation_folds=1)

        with pytest.raises(ValidationError):
            AutoMLRequestDTO(dataset_id="test", cross_validation_folds=11)

        # Test valid values
        for folds in [2, 5, 10]:
            request = AutoMLRequestDTO(dataset_id="test", cross_validation_folds=folds)
            assert request.cross_validation_folds == folds


class TestHyperparameterSpaceDTO:
    """Test suite for HyperparameterSpaceDTO."""

    def test_numerical_parameter_creation(self):
        """Test numerical parameter space creation."""
        param = HyperparameterSpaceDTO(
            parameter_name="n_estimators",
            parameter_type="int",
            low=10.0,
            high=200.0,
            description="Number of estimators",
        )

        assert param.parameter_name == "n_estimators"
        assert param.parameter_type == "int"
        assert param.low == 10.0
        assert param.high == 200.0
        assert param.choices is None  # Default
        assert param.log is False  # Default
        assert param.description == "Number of estimators"

    def test_categorical_parameter_creation(self):
        """Test categorical parameter space creation."""
        choices = ["auto", "sqrt", "log2"]
        param = HyperparameterSpaceDTO(
            parameter_name="max_features",
            parameter_type="categorical",
            choices=choices,
            description="Maximum features to consider",
        )

        assert param.parameter_name == "max_features"
        assert param.parameter_type == "categorical"
        assert param.low is None  # Default
        assert param.high is None  # Default
        assert param.choices == choices
        assert param.description == "Maximum features to consider"

    def test_log_scale_parameter(self):
        """Test parameter with log scale."""
        param = HyperparameterSpaceDTO(
            parameter_name="learning_rate",
            parameter_type="float",
            low=0.001,
            high=1.0,
            log=True,
            description="Learning rate with log scale",
        )

        assert param.log is True
        assert param.low == 0.001
        assert param.high == 1.0


class TestOptimizationTrialDTO:
    """Test suite for OptimizationTrialDTO."""

    def test_basic_creation(self):
        """Test basic optimization trial creation."""
        parameters = {"n_estimators": 150, "max_samples": "auto"}
        trial = OptimizationTrialDTO(
            trial_number=42,
            parameters=parameters,
            score=0.87,
            state="COMPLETE",
            duration=125.5,
            algorithm="isolation_forest",
        )

        assert trial.trial_number == 42
        assert trial.parameters == parameters
        assert trial.score == 0.87
        assert trial.state == "COMPLETE"
        assert trial.duration == 125.5
        assert trial.algorithm == "isolation_forest"

    def test_different_trial_states(self):
        """Test different trial states."""
        states = ["COMPLETE", "FAIL", "PRUNED"]

        for state in states:
            trial = OptimizationTrialDTO(
                trial_number=1,
                parameters={},
                score=0.5,
                state=state,
                duration=10.0,
                algorithm="test",
            )
            assert trial.state == state


class TestEnsembleConfigDTO:
    """Test suite for EnsembleConfigDTO."""

    def test_basic_creation(self):
        """Test basic ensemble configuration creation."""
        algorithms = [
            {"name": "isolation_forest", "params": {"n_estimators": 100}},
            {"name": "one_class_svm", "params": {"gamma": "auto"}},
        ]

        config = EnsembleConfigDTO(
            method="voting",
            algorithms=algorithms,
            voting_strategy="average",
        )

        assert config.method == "voting"
        assert config.algorithms == algorithms
        assert config.voting_strategy == "average"
        assert config.normalize_scores is True  # Default
        assert config.weights is None  # Default

    def test_complete_creation(self):
        """Test creation with all fields."""
        algorithms = [{"name": "algo1"}, {"name": "algo2"}]
        weights = [0.6, 0.4]

        config = EnsembleConfigDTO(
            method="weighted_voting",
            algorithms=algorithms,
            voting_strategy="weighted_average",
            normalize_scores=False,
            weights=weights,
        )

        assert config.normalize_scores is False
        assert config.weights == weights


class TestAutoMLResultDTO:
    """Test suite for AutoMLResultDTO."""

    def test_basic_creation(self):
        """Test basic AutoML result creation."""
        best_params = {"n_estimators": 150, "contamination": 0.1}
        rankings = [("isolation_forest", 0.89), ("one_class_svm", 0.85)]

        result = AutoMLResultDTO(
            best_algorithm="isolation_forest",
            best_params=best_params,
            best_score=0.89,
            optimization_time=1800.5,
            trials_completed=75,
            algorithm_rankings=rankings,
        )

        assert result.best_algorithm == "isolation_forest"
        assert result.best_params == best_params
        assert result.best_score == 0.89
        assert result.optimization_time == 1800.5
        assert result.trials_completed == 75
        assert result.algorithm_rankings == rankings
        assert result.ensemble_config is None  # Default
        assert result.cross_validation_scores is None  # Default
        assert result.feature_importance is None  # Default
        assert result.optimization_history is None  # Default

    def test_complete_creation(self):
        """Test creation with all fields."""
        best_params = {"n_estimators": 200}
        rankings = [("algo1", 0.9), ("algo2", 0.8)]

        ensemble_config = EnsembleConfigDTO(
            method="voting",
            algorithms=[{"name": "algo1"}, {"name": "algo2"}],
            voting_strategy="average",
        )

        cv_scores = [0.88, 0.91, 0.89]
        feature_importance = {"feature1": 0.3, "feature2": 0.7}

        history = [
            OptimizationTrialDTO(
                trial_number=1,
                parameters={"param": 1},
                score=0.8,
                state="COMPLETE",
                duration=10.0,
                algorithm="algo1",
            )
        ]

        result = AutoMLResultDTO(
            best_algorithm="algo1",
            best_params=best_params,
            best_score=0.9,
            optimization_time=3600.0,
            trials_completed=100,
            algorithm_rankings=rankings,
            ensemble_config=ensemble_config,
            cross_validation_scores=cv_scores,
            feature_importance=feature_importance,
            optimization_history=history,
        )

        assert result.ensemble_config == ensemble_config
        assert result.cross_validation_scores == cv_scores
        assert result.feature_importance == feature_importance
        assert result.optimization_history == history


class TestAutoMLResponseDTO:
    """Test suite for AutoMLResponseDTO."""

    def test_successful_response(self):
        """Test successful AutoML response."""
        dataset_profile = DatasetProfileDTO(
            n_samples=1000,
            n_features=10,
            contamination_estimate=0.1,
            feature_types={"f1": "numeric"},
            missing_values_ratio=0.0,
            sparsity_ratio=0.0,
            dimensionality_ratio=0.01,
            dataset_size_mb=1.0,
            complexity_score=0.5,
        )

        automl_result = AutoMLResultDTO(
            best_algorithm="isolation_forest",
            best_params={"n_estimators": 100},
            best_score=0.85,
            optimization_time=1200.0,
            trials_completed=50,
            algorithm_rankings=[("isolation_forest", 0.85)],
        )

        response = AutoMLResponseDTO(
            success=True,
            detector_id="detector_123",
            automl_result=automl_result,
            dataset_profile=dataset_profile,
            message="AutoML optimization completed successfully",
            execution_time=1250.5,
        )

        assert response.success is True
        assert response.detector_id == "detector_123"
        assert response.automl_result == automl_result
        assert response.dataset_profile == dataset_profile
        assert response.algorithm_recommendations is None  # Default
        assert response.optimization_summary is None  # Default
        assert response.message == "AutoML optimization completed successfully"
        assert response.error is None  # Default
        assert response.execution_time == 1250.5

    def test_failed_response(self):
        """Test failed AutoML response."""
        response = AutoMLResponseDTO(
            success=False,
            message="AutoML optimization failed",
            error="Dataset too small for optimization",
            execution_time=5.0,
        )

        assert response.success is False
        assert response.detector_id is None
        assert response.automl_result is None
        assert response.dataset_profile is None
        assert response.error == "Dataset too small for optimization"


class TestExperimentTrackingRequestDTO:
    """Test suite for ExperimentTrackingRequestDTO."""

    def test_basic_creation(self):
        """Test basic experiment tracking request creation."""
        parameters = {"n_estimators": 100, "contamination": 0.1}

        request = ExperimentTrackingRequestDTO(
            experiment_name="fraud_detection_v1",
            algorithm="isolation_forest",
            parameters=parameters,
            dataset_id="dataset_456",
        )

        assert request.experiment_name == "fraud_detection_v1"
        assert request.algorithm == "isolation_forest"
        assert request.parameters == parameters
        assert request.dataset_id == "dataset_456"
        assert request.tags == []  # Default
        assert request.metadata == {}  # Default

    def test_complete_creation(self):
        """Test creation with all fields."""
        parameters = {"n_estimators": 200}
        tags = ["production", "fraud", "v2"]
        metadata = {"version": "2.1", "owner": "data_team"}

        request = ExperimentTrackingRequestDTO(
            experiment_name="advanced_fraud_detection",
            algorithm="one_class_svm",
            parameters=parameters,
            dataset_id="dataset_789",
            tags=tags,
            metadata=metadata,
        )

        assert request.tags == tags
        assert request.metadata == metadata


class TestExperimentTrackingResponseDTO:
    """Test suite for ExperimentTrackingResponseDTO."""

    def test_successful_tracking_response(self):
        """Test successful experiment tracking response."""
        response = ExperimentTrackingResponseDTO(
            experiment_id="exp_12345",
            tracking_url="https://mlflow.example.com/experiments/12345",
            status="active",
            message="Experiment tracking started successfully",
        )

        assert response.experiment_id == "exp_12345"
        assert response.tracking_url == "https://mlflow.example.com/experiments/12345"
        assert response.status == "active"
        assert response.message == "Experiment tracking started successfully"
        assert response.error is None  # Default

    def test_failed_tracking_response(self):
        """Test failed experiment tracking response."""
        response = ExperimentTrackingResponseDTO(
            experiment_id="exp_failed",
            status="failed",
            message="Failed to start experiment tracking",
            error="MLflow server not available",
        )

        assert response.tracking_url is None
        assert response.status == "failed"
        assert response.error == "MLflow server not available"


class TestAutoMLIntegration:
    """Integration tests for AutoML DTOs."""

    def test_complete_automl_workflow(self):
        """Test complete AutoML workflow using multiple DTOs."""
        # Step 1: Profile dataset
        profile_request = AutoMLProfileRequestDTO(
            dataset_id="workflow_dataset",
            include_recommendations=True,
            max_recommendations=3,
        )

        # Step 2: AutoML request
        automl_request = AutoMLRequestDTO(
            dataset_id=profile_request.dataset_id,
            objective="f1",
            max_algorithms=3,
            n_trials=50,
        )

        # Step 3: Create mock results
        dataset_profile = DatasetProfileDTO(
            n_samples=5000,
            n_features=20,
            contamination_estimate=0.05,
            feature_types={f"feature_{i}": "numeric" for i in range(20)},
            missing_values_ratio=0.02,
            sparsity_ratio=0.1,
            dimensionality_ratio=0.004,
            dataset_size_mb=10.5,
            complexity_score=0.6,
        )

        automl_result = AutoMLResultDTO(
            best_algorithm="isolation_forest",
            best_params={"n_estimators": 150, "contamination": 0.05},
            best_score=0.91,
            optimization_time=2400.0,
            trials_completed=45,
            algorithm_rankings=[
                ("isolation_forest", 0.91),
                ("one_class_svm", 0.88),
                ("local_outlier_factor", 0.85),
            ],
        )

        # Step 4: AutoML response
        automl_response = AutoMLResponseDTO(
            success=True,
            detector_id="optimized_detector_789",
            automl_result=automl_result,
            dataset_profile=dataset_profile,
            message="AutoML optimization completed",
            execution_time=2450.0,
        )

        # Verify workflow consistency
        assert automl_response.dataset_profile.n_samples == 5000
        assert automl_response.automl_result.best_algorithm == "isolation_forest"
        assert len(automl_response.automl_result.algorithm_rankings) == 3

    def test_hyperparameter_optimization_workflow(self):
        """Test hyperparameter optimization workflow."""
        # Create parameter space
        param_spaces = {
            "n_estimators": HyperparameterSpaceDTO(
                parameter_name="n_estimators",
                parameter_type="int",
                low=50.0,
                high=300.0,
                description="Number of trees in forest",
            ),
            "contamination": HyperparameterSpaceDTO(
                parameter_name="contamination",
                parameter_type="float",
                low=0.01,
                high=0.2,
                log=True,
                description="Expected contamination rate",
            ),
        }

        # Optimization request
        opt_request = HyperparameterOptimizationRequestDTO(
            dataset_id="opt_dataset",
            algorithm="isolation_forest",
            custom_param_space=param_spaces,
            n_trials=30,
        )

        # Mock trials
        trials = [
            OptimizationTrialDTO(
                trial_number=i,
                parameters={"n_estimators": 100 + i, "contamination": 0.1},
                score=0.8 + (i * 0.01),
                state="COMPLETE",
                duration=15.0 + i,
                algorithm="isolation_forest",
            )
            for i in range(3)
        ]

        # Optimization response
        opt_response = HyperparameterOptimizationResponseDTO(
            success=True,
            best_params={"n_estimators": 102, "contamination": 0.1},
            best_score=0.82,
            optimization_time=150.0,
            trials_completed=3,
            optimization_history=trials,
            algorithm="isolation_forest",
            objective="auc",
            message="Optimization completed successfully",
        )

        # Verify optimization workflow
        assert opt_response.success is True
        assert opt_response.trials_completed == len(trials)
        assert abs(opt_response.optimization_history[2].score - 0.82) < 0.001  # Best trial (floating point tolerance)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
