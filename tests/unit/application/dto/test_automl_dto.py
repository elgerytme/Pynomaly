"""Tests for AutoML DTOs."""

import pytest
from pydantic import ValidationError

from pynomaly.application.dto.automl_dto import (
    AlgorithmRecommendationDTO,
    AlgorithmRecommendationRequestDTO,
    AutoMLProfileRequestDTO,
    AutoMLProfileResponseDTO,
    AutoMLRequestDTO,
    AutoMLResponseDTO,
    AutoMLResultDTO,
    DatasetProfileDTO,
    EnsembleConfigDTO,
    HyperparameterOptimizationRequestDTO,
    HyperparameterOptimizationResponseDTO,
    HyperparameterSpaceDTO,
    OptimizationTrialDTO,
)


class TestAlgorithmRecommendationRequestDTO:
    """Test suite for AlgorithmRecommendationRequestDTO."""

    def test_valid_creation(self):
        """Test creating a valid algorithm recommendation request."""
        dto = AlgorithmRecommendationRequestDTO(
            dataset_id="test_dataset",
            max_recommendations=5,
            performance_priority=0.8,
            speed_priority=0.2,
            include_experimental=True,
            exclude_algorithms=["Algorithm1", "Algorithm2"],
        )

        assert dto.dataset_id == "test_dataset"
        assert dto.max_recommendations == 5
        assert dto.performance_priority == 0.8
        assert dto.speed_priority == 0.2
        assert dto.include_experimental is True
        assert dto.exclude_algorithms == ["Algorithm1", "Algorithm2"]

    def test_default_values(self):
        """Test default values."""
        dto = AlgorithmRecommendationRequestDTO(dataset_id="test_dataset")

        assert dto.dataset_id == "test_dataset"
        assert dto.max_recommendations == 5
        assert dto.performance_priority == 0.8
        assert dto.speed_priority == 0.2
        assert dto.include_experimental is False
        assert dto.exclude_algorithms == []

    def test_validation_max_recommendations(self):
        """Test validation of max_recommendations."""
        # Valid range
        dto = AlgorithmRecommendationRequestDTO(
            dataset_id="test", max_recommendations=10
        )
        assert dto.max_recommendations == 10

        # Invalid: too small
        with pytest.raises(ValidationError):
            AlgorithmRecommendationRequestDTO(dataset_id="test", max_recommendations=0)

        # Invalid: too large
        with pytest.raises(ValidationError):
            AlgorithmRecommendationRequestDTO(dataset_id="test", max_recommendations=25)

    def test_validation_priority_values(self):
        """Test validation of priority values."""
        # Valid range
        dto = AlgorithmRecommendationRequestDTO(
            dataset_id="test", performance_priority=0.6, speed_priority=0.4
        )
        assert dto.performance_priority == 0.6
        assert dto.speed_priority == 0.4

        # Invalid: negative
        with pytest.raises(ValidationError):
            AlgorithmRecommendationRequestDTO(
                dataset_id="test", performance_priority=-0.1
            )

        # Invalid: greater than 1
        with pytest.raises(ValidationError):
            AlgorithmRecommendationRequestDTO(dataset_id="test", speed_priority=1.1)

    def test_required_fields(self):
        """Test required fields."""
        with pytest.raises(ValidationError):
            AlgorithmRecommendationRequestDTO()

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError):
            AlgorithmRecommendationRequestDTO(dataset_id="test", unknown_field="value")


class TestDatasetProfileDTO:
    """Test suite for DatasetProfileDTO."""

    def test_valid_creation(self):
        """Test creating a valid dataset profile."""
        dto = DatasetProfileDTO(
            n_samples=1000,
            n_features=50,
            contamination_estimate=0.1,
            feature_types={"feature1": "numeric", "feature2": "categorical"},
            missing_values_ratio=0.05,
            categorical_features=["feature2"],
            numerical_features=["feature1"],
            time_series_features=[],
            sparsity_ratio=0.1,
            dimensionality_ratio=0.05,
            dataset_size_mb=10.5,
            has_temporal_structure=True,
            has_graph_structure=False,
            complexity_score=0.3,
        )

        assert dto.n_samples == 1000
        assert dto.n_features == 50
        assert dto.contamination_estimate == 0.1
        assert dto.feature_types == {"feature1": "numeric", "feature2": "categorical"}
        assert dto.missing_values_ratio == 0.05
        assert dto.categorical_features == ["feature2"]
        assert dto.numerical_features == ["feature1"]
        assert dto.time_series_features == []
        assert dto.sparsity_ratio == 0.1
        assert dto.dimensionality_ratio == 0.05
        assert dto.dataset_size_mb == 10.5
        assert dto.has_temporal_structure is True
        assert dto.has_graph_structure is False
        assert dto.complexity_score == 0.3

    def test_default_values(self):
        """Test default values."""
        dto = DatasetProfileDTO(
            n_samples=1000,
            n_features=50,
            contamination_estimate=0.1,
            feature_types={"feature1": "numeric"},
            missing_values_ratio=0.05,
            sparsity_ratio=0.1,
            dimensionality_ratio=0.05,
            dataset_size_mb=10.5,
            complexity_score=0.3,
        )

        assert dto.categorical_features == []
        assert dto.numerical_features == []
        assert dto.time_series_features == []
        assert dto.has_temporal_structure is False
        assert dto.has_graph_structure is False

    def test_required_fields(self):
        """Test required fields."""
        with pytest.raises(ValidationError):
            DatasetProfileDTO()

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError):
            DatasetProfileDTO(
                n_samples=1000,
                n_features=50,
                contamination_estimate=0.1,
                feature_types={},
                missing_values_ratio=0.05,
                sparsity_ratio=0.1,
                dimensionality_ratio=0.05,
                dataset_size_mb=10.5,
                complexity_score=0.3,
                unknown_field="value",
            )


class TestAlgorithmRecommendationDTO:
    """Test suite for AlgorithmRecommendationDTO."""

    def test_valid_creation(self):
        """Test creating a valid algorithm recommendation."""
        dto = AlgorithmRecommendationDTO(
            algorithm_name="IsolationForest",
            score=0.85,
            family="isolation",
            complexity_score=0.3,
            recommended_params={"n_estimators": 100},
            reasoning=["Good for high-dimensional data"],
        )

        assert dto.algorithm_name == "IsolationForest"
        assert dto.score == 0.85
        assert dto.family == "isolation"
        assert dto.complexity_score == 0.3
        assert dto.recommended_params == {"n_estimators": 100}
        assert dto.reasoning == ["Good for high-dimensional data"]

    def test_default_values(self):
        """Test default values."""
        dto = AlgorithmRecommendationDTO(
            algorithm_name="IsolationForest",
            score=0.85,
            family="isolation",
            complexity_score=0.3,
            recommended_params={"n_estimators": 100},
        )

        assert dto.reasoning == []

    def test_required_fields(self):
        """Test required fields."""
        with pytest.raises(ValidationError):
            AlgorithmRecommendationDTO()

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError):
            AlgorithmRecommendationDTO(
                algorithm_name="IsolationForest",
                score=0.85,
                family="isolation",
                complexity_score=0.3,
                recommended_params={"n_estimators": 100},
                unknown_field="value",
            )


class TestAutoMLRequestDTO:
    """Test suite for AutoMLRequestDTO."""

    def test_valid_creation(self):
        """Test creating a valid AutoML request."""
        dto = AutoMLRequestDTO(
            dataset_id="test_dataset",
            objective="f1",
            max_algorithms=5,
            max_optimization_time=7200,
            n_trials=200,
            enable_ensemble=False,
            detector_name="custom_detector",
            cross_validation_folds=5,
            random_state=123,
        )

        assert dto.dataset_id == "test_dataset"
        assert dto.objective == "f1"
        assert dto.max_algorithms == 5
        assert dto.max_optimization_time == 7200
        assert dto.n_trials == 200
        assert dto.enable_ensemble is False
        assert dto.detector_name == "custom_detector"
        assert dto.cross_validation_folds == 5
        assert dto.random_state == 123

    def test_default_values(self):
        """Test default values."""
        dto = AutoMLRequestDTO(dataset_id="test_dataset")

        assert dto.dataset_id == "test_dataset"
        assert dto.objective == "auc"
        assert dto.max_algorithms == 3
        assert dto.max_optimization_time == 3600
        assert dto.n_trials == 100
        assert dto.enable_ensemble is True
        assert dto.detector_name is None
        assert dto.cross_validation_folds == 3
        assert dto.random_state == 42

    def test_validation_constraints(self):
        """Test validation constraints."""
        # Valid ranges
        dto = AutoMLRequestDTO(
            dataset_id="test",
            max_algorithms=1,
            max_optimization_time=60,
            n_trials=10,
            cross_validation_folds=2,
        )
        assert dto.max_algorithms == 1
        assert dto.max_optimization_time == 60
        assert dto.n_trials == 10
        assert dto.cross_validation_folds == 2

        # Invalid: max_algorithms too small
        with pytest.raises(ValidationError):
            AutoMLRequestDTO(dataset_id="test", max_algorithms=0)

        # Invalid: max_algorithms too large
        with pytest.raises(ValidationError):
            AutoMLRequestDTO(dataset_id="test", max_algorithms=15)

        # Invalid: max_optimization_time too small
        with pytest.raises(ValidationError):
            AutoMLRequestDTO(dataset_id="test", max_optimization_time=30)

        # Invalid: n_trials too small
        with pytest.raises(ValidationError):
            AutoMLRequestDTO(dataset_id="test", n_trials=5)

        # Invalid: cross_validation_folds too small
        with pytest.raises(ValidationError):
            AutoMLRequestDTO(dataset_id="test", cross_validation_folds=1)


class TestHyperparameterSpaceDTO:
    """Test suite for HyperparameterSpaceDTO."""

    def test_valid_creation_float(self):
        """Test creating a valid float hyperparameter space."""
        dto = HyperparameterSpaceDTO(
            parameter_name="learning_rate",
            parameter_type="float",
            low=0.01,
            high=1.0,
            log=True,
            description="Learning rate parameter",
        )

        assert dto.parameter_name == "learning_rate"
        assert dto.parameter_type == "float"
        assert dto.low == 0.01
        assert dto.high == 1.0
        assert dto.choices is None
        assert dto.log is True
        assert dto.description == "Learning rate parameter"

    def test_valid_creation_categorical(self):
        """Test creating a valid categorical hyperparameter space."""
        dto = HyperparameterSpaceDTO(
            parameter_name="algorithm",
            parameter_type="categorical",
            choices=["rf", "svm", "xgb"],
            description="Algorithm choice",
        )

        assert dto.parameter_name == "algorithm"
        assert dto.parameter_type == "categorical"
        assert dto.low is None
        assert dto.high is None
        assert dto.choices == ["rf", "svm", "xgb"]
        assert dto.log is False
        assert dto.description == "Algorithm choice"

    def test_default_values(self):
        """Test default values."""
        dto = HyperparameterSpaceDTO(
            parameter_name="test", parameter_type="float", description="Test parameter"
        )

        assert dto.low is None
        assert dto.high is None
        assert dto.choices is None
        assert dto.log is False

    def test_required_fields(self):
        """Test required fields."""
        with pytest.raises(ValidationError):
            HyperparameterSpaceDTO()


class TestOptimizationTrialDTO:
    """Test suite for OptimizationTrialDTO."""

    def test_valid_creation(self):
        """Test creating a valid optimization trial."""
        dto = OptimizationTrialDTO(
            trial_number=1,
            parameters={"n_estimators": 100, "max_depth": 5},
            score=0.85,
            state="COMPLETE",
            duration=45.5,
            algorithm="RandomForest",
        )

        assert dto.trial_number == 1
        assert dto.parameters == {"n_estimators": 100, "max_depth": 5}
        assert dto.score == 0.85
        assert dto.state == "COMPLETE"
        assert dto.duration == 45.5
        assert dto.algorithm == "RandomForest"

    def test_required_fields(self):
        """Test required fields."""
        with pytest.raises(ValidationError):
            OptimizationTrialDTO()

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError):
            OptimizationTrialDTO(
                trial_number=1,
                parameters={},
                score=0.85,
                state="COMPLETE",
                duration=45.5,
                algorithm="RandomForest",
                unknown_field="value",
            )


class TestEnsembleConfigDTO:
    """Test suite for EnsembleConfigDTO."""

    def test_valid_creation(self):
        """Test creating a valid ensemble config."""
        dto = EnsembleConfigDTO(
            method="voting",
            algorithms=[{"name": "RF", "params": {}}],
            voting_strategy="average",
            normalize_scores=False,
            weights=[0.6, 0.4],
        )

        assert dto.method == "voting"
        assert dto.algorithms == [{"name": "RF", "params": {}}]
        assert dto.voting_strategy == "average"
        assert dto.normalize_scores is False
        assert dto.weights == [0.6, 0.4]

    def test_default_values(self):
        """Test default values."""
        dto = EnsembleConfigDTO(
            method="voting",
            algorithms=[{"name": "RF", "params": {}}],
            voting_strategy="average",
        )

        assert dto.normalize_scores is True
        assert dto.weights is None

    def test_required_fields(self):
        """Test required fields."""
        with pytest.raises(ValidationError):
            EnsembleConfigDTO()


class TestAutoMLResultDTO:
    """Test suite for AutoMLResultDTO."""

    def test_valid_creation(self):
        """Test creating a valid AutoML result."""
        ensemble_config = EnsembleConfigDTO(
            method="voting",
            algorithms=[{"name": "RF", "params": {}}],
            voting_strategy="average",
        )

        trial = OptimizationTrialDTO(
            trial_number=1,
            parameters={"n_estimators": 100},
            score=0.85,
            state="COMPLETE",
            duration=45.5,
            algorithm="RandomForest",
        )

        dto = AutoMLResultDTO(
            best_algorithm="RandomForest",
            best_params={"n_estimators": 100},
            best_score=0.85,
            optimization_time=300.0,
            trials_completed=50,
            algorithm_rankings=[("RandomForest", 0.85), ("SVM", 0.80)],
            ensemble_config=ensemble_config,
            cross_validation_scores=[0.83, 0.85, 0.87],
            feature_importance={"feature1": 0.6, "feature2": 0.4},
            optimization_history=[trial],
        )

        assert dto.best_algorithm == "RandomForest"
        assert dto.best_params == {"n_estimators": 100}
        assert dto.best_score == 0.85
        assert dto.optimization_time == 300.0
        assert dto.trials_completed == 50
        assert dto.algorithm_rankings == [("RandomForest", 0.85), ("SVM", 0.80)]
        assert dto.ensemble_config == ensemble_config
        assert dto.cross_validation_scores == [0.83, 0.85, 0.87]
        assert dto.feature_importance == {"feature1": 0.6, "feature2": 0.4}
        assert dto.optimization_history == [trial]

    def test_default_values(self):
        """Test default values."""
        dto = AutoMLResultDTO(
            best_algorithm="RandomForest",
            best_params={"n_estimators": 100},
            best_score=0.85,
            optimization_time=300.0,
            trials_completed=50,
            algorithm_rankings=[("RandomForest", 0.85)],
        )

        assert dto.ensemble_config is None
        assert dto.cross_validation_scores is None
        assert dto.feature_importance is None
        assert dto.optimization_history is None

    def test_required_fields(self):
        """Test required fields."""
        with pytest.raises(ValidationError):
            AutoMLResultDTO()


class TestAutoMLResponseDTO:
    """Test suite for AutoMLResponseDTO."""

    def test_valid_creation(self):
        """Test creating a valid AutoML response."""
        automl_result = AutoMLResultDTO(
            best_algorithm="RandomForest",
            best_params={"n_estimators": 100},
            best_score=0.85,
            optimization_time=300.0,
            trials_completed=50,
            algorithm_rankings=[("RandomForest", 0.85)],
        )

        dataset_profile = DatasetProfileDTO(
            n_samples=1000,
            n_features=50,
            contamination_estimate=0.1,
            feature_types={"feature1": "numeric"},
            missing_values_ratio=0.05,
            sparsity_ratio=0.1,
            dimensionality_ratio=0.05,
            dataset_size_mb=10.5,
            complexity_score=0.3,
        )

        dto = AutoMLResponseDTO(
            success=True,
            detector_id="detector_123",
            automl_result=automl_result,
            dataset_profile=dataset_profile,
            algorithm_recommendations=[],
            optimization_summary={"total_time": 300.0},
            message="AutoML completed successfully",
            error=None,
            execution_time=305.0,
        )

        assert dto.success is True
        assert dto.detector_id == "detector_123"
        assert dto.automl_result == automl_result
        assert dto.dataset_profile == dataset_profile
        assert dto.algorithm_recommendations == []
        assert dto.optimization_summary == {"total_time": 300.0}
        assert dto.message == "AutoML completed successfully"
        assert dto.error is None
        assert dto.execution_time == 305.0

    def test_default_values(self):
        """Test default values."""
        dto = AutoMLResponseDTO(success=True, message="Success", execution_time=100.0)

        assert dto.detector_id is None
        assert dto.automl_result is None
        assert dto.dataset_profile is None
        assert dto.algorithm_recommendations is None
        assert dto.optimization_summary is None
        assert dto.error is None

    def test_required_fields(self):
        """Test required fields."""
        with pytest.raises(ValidationError):
            AutoMLResponseDTO()


class TestAutoMLProfileRequestDTO:
    """Test suite for AutoMLProfileRequestDTO."""

    def test_valid_creation(self):
        """Test creating a valid profile request."""
        dto = AutoMLProfileRequestDTO(
            dataset_id="test_dataset",
            include_recommendations=False,
            max_recommendations=8,
        )

        assert dto.dataset_id == "test_dataset"
        assert dto.include_recommendations is False
        assert dto.max_recommendations == 8

    def test_default_values(self):
        """Test default values."""
        dto = AutoMLProfileRequestDTO(dataset_id="test_dataset")

        assert dto.include_recommendations is True
        assert dto.max_recommendations == 5

    def test_validation_max_recommendations(self):
        """Test validation of max_recommendations."""
        # Valid range
        dto = AutoMLProfileRequestDTO(dataset_id="test", max_recommendations=3)
        assert dto.max_recommendations == 3

        # Invalid: too small
        with pytest.raises(ValidationError):
            AutoMLProfileRequestDTO(dataset_id="test", max_recommendations=0)

        # Invalid: too large
        with pytest.raises(ValidationError):
            AutoMLProfileRequestDTO(dataset_id="test", max_recommendations=15)

    def test_required_fields(self):
        """Test required fields."""
        with pytest.raises(ValidationError):
            AutoMLProfileRequestDTO()


class TestAutoMLProfileResponseDTO:
    """Test suite for AutoMLProfileResponseDTO."""

    def test_valid_creation(self):
        """Test creating a valid profile response."""
        dataset_profile = DatasetProfileDTO(
            n_samples=1000,
            n_features=50,
            contamination_estimate=0.1,
            feature_types={"feature1": "numeric"},
            missing_values_ratio=0.05,
            sparsity_ratio=0.1,
            dimensionality_ratio=0.05,
            dataset_size_mb=10.5,
            complexity_score=0.3,
        )

        recommendation = AlgorithmRecommendationDTO(
            algorithm_name="IsolationForest",
            score=0.85,
            family="isolation",
            complexity_score=0.3,
            recommended_params={"n_estimators": 100},
        )

        dto = AutoMLProfileResponseDTO(
            success=True,
            dataset_profile=dataset_profile,
            algorithm_recommendations=[recommendation],
            message="Profiling completed successfully",
            error=None,
            execution_time=15.0,
        )

        assert dto.success is True
        assert dto.dataset_profile == dataset_profile
        assert dto.algorithm_recommendations == [recommendation]
        assert dto.message == "Profiling completed successfully"
        assert dto.error is None
        assert dto.execution_time == 15.0

    def test_default_values(self):
        """Test default values."""
        dto = AutoMLProfileResponseDTO(
            success=True, message="Success", execution_time=15.0
        )

        assert dto.dataset_profile is None
        assert dto.algorithm_recommendations is None
        assert dto.error is None

    def test_required_fields(self):
        """Test required fields."""
        with pytest.raises(ValidationError):
            AutoMLProfileResponseDTO()


class TestHyperparameterOptimizationRequestDTO:
    """Test suite for HyperparameterOptimizationRequestDTO."""

    def test_valid_creation(self):
        """Test creating a valid hyperparameter optimization request."""
        param_space = HyperparameterSpaceDTO(
            parameter_name="n_estimators",
            parameter_type="int",
            low=10,
            high=1000,
            description="Number of estimators",
        )

        dto = HyperparameterOptimizationRequestDTO(
            dataset_id="test_dataset",
            algorithm="RandomForest",
            objective="f1",
            n_trials=200,
            timeout=7200,
            direction="minimize",
            custom_param_space={"n_estimators": param_space},
            cross_validation_folds=5,
            random_state=123,
        )

        assert dto.dataset_id == "test_dataset"
        assert dto.algorithm == "RandomForest"
        assert dto.objective == "f1"
        assert dto.n_trials == 200
        assert dto.timeout == 7200
        assert dto.direction == "minimize"
        assert dto.custom_param_space == {"n_estimators": param_space}
        assert dto.cross_validation_folds == 5
        assert dto.random_state == 123

    def test_default_values(self):
        """Test default values."""
        dto = HyperparameterOptimizationRequestDTO(
            dataset_id="test_dataset", algorithm="RandomForest"
        )

        assert dto.objective == "auc"
        assert dto.n_trials == 100
        assert dto.timeout == 3600
        assert dto.direction == "maximize"
        assert dto.custom_param_space is None
        assert dto.cross_validation_folds == 3
        assert dto.random_state == 42

    def test_validation_constraints(self):
        """Test validation constraints."""
        # Valid ranges
        dto = HyperparameterOptimizationRequestDTO(
            dataset_id="test",
            algorithm="RF",
            n_trials=10,
            timeout=60,
            cross_validation_folds=2,
        )
        assert dto.n_trials == 10
        assert dto.timeout == 60
        assert dto.cross_validation_folds == 2

        # Invalid: n_trials too small
        with pytest.raises(ValidationError):
            HyperparameterOptimizationRequestDTO(
                dataset_id="test", algorithm="RF", n_trials=5
            )

        # Invalid: timeout too small
        with pytest.raises(ValidationError):
            HyperparameterOptimizationRequestDTO(
                dataset_id="test", algorithm="RF", timeout=30
            )

    def test_required_fields(self):
        """Test required fields."""
        with pytest.raises(ValidationError):
            HyperparameterOptimizationRequestDTO()


class TestHyperparameterOptimizationResponseDTO:
    """Test suite for HyperparameterOptimizationResponseDTO."""

    def test_valid_creation(self):
        """Test creating a valid hyperparameter optimization response."""
        trial = OptimizationTrialDTO(
            trial_number=1,
            parameters={"n_estimators": 100},
            score=0.85,
            state="COMPLETE",
            duration=45.5,
            algorithm="RandomForest",
        )

        dto = HyperparameterOptimizationResponseDTO(
            success=True,
            best_params={"n_estimators": 100},
            best_score=0.85,
            optimization_time=300.0,
            trials_completed=50,
            optimization_history=[trial],
            algorithm="RandomForest",
            objective="auc",
            message="Optimization completed successfully",
            error=None,
        )

        assert dto.success is True
        assert dto.best_params == {"n_estimators": 100}
        assert dto.best_score == 0.85
        assert dto.optimization_time == 300.0
        assert dto.trials_completed == 50
        assert dto.optimization_history == [trial]
        assert dto.algorithm == "RandomForest"
        assert dto.objective == "auc"
        assert dto.message == "Optimization completed successfully"
        assert dto.error is None

    def test_default_values(self):
        """Test default values."""
        dto = HyperparameterOptimizationResponseDTO(
            success=True,
            optimization_time=300.0,
            algorithm="RandomForest",
            objective="auc",
            message="Success",
        )

        assert dto.best_params is None
        assert dto.best_score is None
        assert dto.trials_completed == 0
        assert dto.optimization_history is None
        assert dto.error is None

    def test_required_fields(self):
        """Test required fields."""
        with pytest.raises(ValidationError):
            HyperparameterOptimizationResponseDTO()


class TestAutoMLDTOIntegration:
    """Test integration scenarios for AutoML DTOs."""

    def test_complete_automl_workflow(self):
        """Test complete AutoML workflow with all DTOs."""
        # Create request
        request = AutoMLRequestDTO(
            dataset_id="test_dataset",
            objective="auc",
            max_algorithms=3,
            n_trials=100,
            enable_ensemble=True,
        )

        # Create dataset profile
        dataset_profile = DatasetProfileDTO(
            n_samples=1000,
            n_features=50,
            contamination_estimate=0.1,
            feature_types={"feature1": "numeric", "feature2": "categorical"},
            missing_values_ratio=0.05,
            sparsity_ratio=0.1,
            dimensionality_ratio=0.05,
            dataset_size_mb=10.5,
            complexity_score=0.3,
        )

        # Create algorithm recommendations
        recommendation = AlgorithmRecommendationDTO(
            algorithm_name="IsolationForest",
            score=0.85,
            family="isolation",
            complexity_score=0.3,
            recommended_params={"n_estimators": 100},
            reasoning=["Good for high-dimensional data"],
        )

        # Create optimization trial
        trial = OptimizationTrialDTO(
            trial_number=1,
            parameters={"n_estimators": 100},
            score=0.85,
            state="COMPLETE",
            duration=45.5,
            algorithm="IsolationForest",
        )

        # Create ensemble config
        ensemble_config = EnsembleConfigDTO(
            method="voting",
            algorithms=[{"name": "IsolationForest", "params": {"n_estimators": 100}}],
            voting_strategy="average",
            normalize_scores=True,
        )

        # Create AutoML result
        automl_result = AutoMLResultDTO(
            best_algorithm="IsolationForest",
            best_params={"n_estimators": 100},
            best_score=0.85,
            optimization_time=300.0,
            trials_completed=50,
            algorithm_rankings=[("IsolationForest", 0.85)],
            ensemble_config=ensemble_config,
            cross_validation_scores=[0.83, 0.85, 0.87],
            feature_importance={"feature1": 0.6, "feature2": 0.4},
            optimization_history=[trial],
        )

        # Create response
        response = AutoMLResponseDTO(
            success=True,
            detector_id="detector_123",
            automl_result=automl_result,
            dataset_profile=dataset_profile,
            algorithm_recommendations=[recommendation],
            optimization_summary={"total_time": 300.0},
            message="AutoML completed successfully",
            execution_time=305.0,
        )

        # Verify all components are properly integrated
        assert response.success is True
        assert response.automl_result.best_algorithm == "IsolationForest"
        assert response.dataset_profile.n_samples == 1000
        assert len(response.algorithm_recommendations) == 1
        assert response.algorithm_recommendations[0].algorithm_name == "IsolationForest"

    def test_profile_only_workflow(self):
        """Test profile-only workflow."""
        # Create profile request
        request = AutoMLProfileRequestDTO(
            dataset_id="test_dataset",
            include_recommendations=True,
            max_recommendations=5,
        )

        # Create dataset profile
        dataset_profile = DatasetProfileDTO(
            n_samples=1000,
            n_features=50,
            contamination_estimate=0.1,
            feature_types={"feature1": "numeric"},
            missing_values_ratio=0.05,
            sparsity_ratio=0.1,
            dimensionality_ratio=0.05,
            dataset_size_mb=10.5,
            complexity_score=0.3,
        )

        # Create recommendations
        recommendations = [
            AlgorithmRecommendationDTO(
                algorithm_name="IsolationForest",
                score=0.85,
                family="isolation",
                complexity_score=0.3,
                recommended_params={"n_estimators": 100},
            )
        ]

        # Create response
        response = AutoMLProfileResponseDTO(
            success=True,
            dataset_profile=dataset_profile,
            algorithm_recommendations=recommendations,
            message="Profiling completed successfully",
            execution_time=15.0,
        )

        # Verify profile workflow
        assert response.success is True
        assert response.dataset_profile.n_samples == 1000
        assert len(response.algorithm_recommendations) == 1
        assert response.algorithm_recommendations[0].algorithm_name == "IsolationForest"

    def test_hyperparameter_optimization_workflow(self):
        """Test hyperparameter optimization workflow."""
        # Create hyperparameter space
        param_space = HyperparameterSpaceDTO(
            parameter_name="n_estimators",
            parameter_type="int",
            low=10,
            high=1000,
            description="Number of estimators",
        )

        # Create request
        request = HyperparameterOptimizationRequestDTO(
            dataset_id="test_dataset",
            algorithm="RandomForest",
            objective="auc",
            n_trials=100,
            custom_param_space={"n_estimators": param_space},
        )

        # Create optimization trials
        trials = [
            OptimizationTrialDTO(
                trial_number=i,
                parameters={"n_estimators": 100 + i * 10},
                score=0.8 + i * 0.01,
                state="COMPLETE",
                duration=45.5,
                algorithm="RandomForest",
            )
            for i in range(5)
        ]

        # Create response
        response = HyperparameterOptimizationResponseDTO(
            success=True,
            best_params={"n_estimators": 140},
            best_score=0.84,
            optimization_time=300.0,
            trials_completed=5,
            optimization_history=trials,
            algorithm="RandomForest",
            objective="auc",
            message="Optimization completed successfully",
        )

        # Verify optimization workflow
        assert response.success is True
        assert response.best_params == {"n_estimators": 140}
        assert response.best_score == 0.84
        assert len(response.optimization_history) == 5
        assert response.algorithm == "RandomForest"

    def test_error_handling(self):
        """Test error handling in DTOs."""
        # Test failed AutoML response
        error_response = AutoMLResponseDTO(
            success=False,
            message="AutoML failed",
            error="Insufficient data",
            execution_time=10.0,
        )

        assert error_response.success is False
        assert error_response.error == "Insufficient data"
        assert error_response.detector_id is None
        assert error_response.automl_result is None

        # Test failed hyperparameter optimization
        error_opt_response = HyperparameterOptimizationResponseDTO(
            success=False,
            optimization_time=30.0,
            algorithm="RandomForest",
            objective="auc",
            message="Optimization failed",
            error="Algorithm not supported",
        )

        assert error_opt_response.success is False
        assert error_opt_response.error == "Algorithm not supported"
        assert error_opt_response.best_params is None
        assert error_opt_response.best_score is None

    def test_dto_serialization(self):
        """Test DTO serialization and deserialization."""
        # Create a complex DTO
        dto = AutoMLRequestDTO(
            dataset_id="test_dataset",
            objective="auc",
            max_algorithms=3,
            n_trials=100,
            enable_ensemble=True,
            detector_name="custom_detector",
        )

        # Serialize to dict
        dto_dict = dto.model_dump()

        assert dto_dict["dataset_id"] == "test_dataset"
        assert dto_dict["objective"] == "auc"
        assert dto_dict["max_algorithms"] == 3
        assert dto_dict["n_trials"] == 100
        assert dto_dict["enable_ensemble"] is True
        assert dto_dict["detector_name"] == "custom_detector"

        # Deserialize from dict
        dto_restored = AutoMLRequestDTO.model_validate(dto_dict)

        assert dto_restored.dataset_id == dto.dataset_id
        assert dto_restored.objective == dto.objective
        assert dto_restored.max_algorithms == dto.max_algorithms
        assert dto_restored.n_trials == dto.n_trials
        assert dto_restored.enable_ensemble == dto.enable_ensemble
        assert dto_restored.detector_name == dto.detector_name

    def test_nested_dto_validation(self):
        """Test nested DTO validation."""
        # Test valid nested structure
        ensemble_config = EnsembleConfigDTO(
            method="voting",
            algorithms=[{"name": "RF", "params": {}}],
            voting_strategy="average",
        )

        automl_result = AutoMLResultDTO(
            best_algorithm="RandomForest",
            best_params={"n_estimators": 100},
            best_score=0.85,
            optimization_time=300.0,
            trials_completed=50,
            algorithm_rankings=[("RandomForest", 0.85)],
            ensemble_config=ensemble_config,
        )

        assert automl_result.ensemble_config.method == "voting"
        assert automl_result.ensemble_config.voting_strategy == "average"

        # Test invalid nested structure
        with pytest.raises(ValidationError):
            AutoMLResultDTO(
                best_algorithm="RandomForest",
                best_params={"n_estimators": 100},
                best_score=0.85,
                optimization_time=300.0,
                trials_completed=50,
                algorithm_rankings=[("RandomForest", 0.85)],
                ensemble_config="invalid_config",  # Should be EnsembleConfigDTO
            )
