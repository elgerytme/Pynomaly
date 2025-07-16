"""Tests for Ensemble DTOs."""

from datetime import datetime

import pytest
from pydantic import ValidationError

from monorepo.application.dto.ensemble_dto import (
    DiversityMetricsDTO,
    EnsembleConfigurationDTO,
    EnsembleDetectionRequestDTO,
    EnsembleDetectionResponseDTO,
    EnsembleOptimizationRequestDTO,
    EnsembleOptimizationResponseDTO,
    EnsemblePerformanceDTO,
    EnsembleReportDTO,
    EnsembleStrategyDTO,
    MetaLearningKnowledgeDTO,
    create_default_ensemble_config,
    create_diversity_metrics_from_dict,
    create_ensemble_performance_from_dict,
)


class TestEnsembleStrategyDTO:
    """Test suite for EnsembleStrategyDTO."""

    def test_valid_creation(self):
        """Test creating a valid ensemble strategy DTO."""
        dto = EnsembleStrategyDTO(
            name="VotingStrategy",
            description="Simple voting ensemble strategy",
            requires_training=False,
            supports_weights=True,
            complexity="low",
            interpretability=0.8,
        )

        assert dto.name == "VotingStrategy"
        assert dto.description == "Simple voting ensemble strategy"
        assert dto.requires_training is False
        assert dto.supports_weights is True
        assert dto.complexity == "low"
        assert dto.interpretability == 0.8

    def test_default_values(self):
        """Test default values."""
        dto = EnsembleStrategyDTO(name="TestStrategy", description="Test description")

        assert dto.requires_training is False
        assert dto.supports_weights is True
        assert dto.complexity == "medium"
        assert dto.interpretability == 0.5

    def test_interpretability_validation(self):
        """Test interpretability validation."""
        # Valid range
        dto = EnsembleStrategyDTO(
            name="TestStrategy", description="Test description", interpretability=0.9
        )
        assert dto.interpretability == 0.9

        # Invalid: negative
        with pytest.raises(ValidationError):
            EnsembleStrategyDTO(
                name="TestStrategy",
                description="Test description",
                interpretability=-0.1,
            )

        # Invalid: greater than 1
        with pytest.raises(ValidationError):
            EnsembleStrategyDTO(
                name="TestStrategy",
                description="Test description",
                interpretability=1.1,
            )

    def test_required_fields(self):
        """Test required fields."""
        with pytest.raises(ValidationError):
            EnsembleStrategyDTO(name="TestStrategy")

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError):
            EnsembleStrategyDTO(
                name="TestStrategy",
                description="Test description",
                unknown_field="value",
            )


class TestDiversityMetricsDTO:
    """Test suite for DiversityMetricsDTO."""

    def test_valid_creation(self):
        """Test creating a valid diversity metrics DTO."""
        dto = DiversityMetricsDTO(
            disagreement_measure=0.6,
            double_fault_measure=0.2,
            q_statistic=0.3,
            correlation_coefficient=0.4,
            kappa_statistic=0.5,
            entropy_measure=0.7,
            overall_diversity=0.55,
        )

        assert dto.disagreement_measure == 0.6
        assert dto.double_fault_measure == 0.2
        assert dto.q_statistic == 0.3
        assert dto.correlation_coefficient == 0.4
        assert dto.kappa_statistic == 0.5
        assert dto.entropy_measure == 0.7
        assert dto.overall_diversity == 0.55

    def test_required_fields(self):
        """Test required fields."""
        with pytest.raises(ValidationError):
            DiversityMetricsDTO()

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError):
            DiversityMetricsDTO(
                disagreement_measure=0.6,
                double_fault_measure=0.2,
                q_statistic=0.3,
                correlation_coefficient=0.4,
                kappa_statistic=0.5,
                entropy_measure=0.7,
                overall_diversity=0.55,
                unknown_field="value",
            )


class TestMetaLearningKnowledgeDTO:
    """Test suite for MetaLearningKnowledgeDTO."""

    def test_valid_creation(self):
        """Test creating a valid meta-learning knowledge DTO."""
        characteristics = {"n_features": 10, "n_samples": 1000}
        performance = {"IsolationForest": 0.85, "OneClassSVM": 0.78}
        composition = ["IsolationForest", "LocalOutlierFactor"]
        weights = {"IsolationForest": 0.6, "LocalOutlierFactor": 0.4}
        diversity = {"min_diversity": 0.3, "target_diversity": 0.5}
        metrics = {"accuracy": 0.82, "f1_score": 0.79}

        dto = MetaLearningKnowledgeDTO(
            dataset_characteristics=characteristics,
            algorithm_performance=performance,
            ensemble_composition=composition,
            optimal_weights=weights,
            diversity_requirements=diversity,
            performance_metrics=metrics,
            confidence_score=0.9,
        )

        assert dto.dataset_characteristics == characteristics
        assert dto.algorithm_performance == performance
        assert dto.ensemble_composition == composition
        assert dto.optimal_weights == weights
        assert dto.diversity_requirements == diversity
        assert dto.performance_metrics == metrics
        assert dto.confidence_score == 0.9
        assert isinstance(dto.timestamp, datetime)

    def test_default_timestamp(self):
        """Test default timestamp."""
        dto = MetaLearningKnowledgeDTO(
            dataset_characteristics={},
            algorithm_performance={},
            ensemble_composition=[],
            optimal_weights={},
            diversity_requirements={},
            performance_metrics={},
            confidence_score=0.8,
        )

        assert isinstance(dto.timestamp, datetime)

    def test_required_fields(self):
        """Test required fields."""
        with pytest.raises(ValidationError):
            MetaLearningKnowledgeDTO()

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError):
            MetaLearningKnowledgeDTO(
                dataset_characteristics={},
                algorithm_performance={},
                ensemble_composition=[],
                optimal_weights={},
                diversity_requirements={},
                performance_metrics={},
                confidence_score=0.8,
                unknown_field="value",
            )


class TestEnsembleConfigurationDTO:
    """Test suite for EnsembleConfigurationDTO."""

    def test_valid_creation(self):
        """Test creating a valid ensemble configuration DTO."""
        algorithms = ["IsolationForest", "LocalOutlierFactor", "OneClassSVM"]

        dto = EnsembleConfigurationDTO(
            base_algorithms=algorithms,
            ensemble_strategy="weighted_voting",
            max_ensemble_size=4,
            min_diversity_threshold=0.4,
            weight_optimization=True,
            diversity_weighting=0.2,
            cross_validation_folds=5,
            meta_learning_enabled=True,
        )

        assert dto.base_algorithms == algorithms
        assert dto.ensemble_strategy == "weighted_voting"
        assert dto.max_ensemble_size == 4
        assert dto.min_diversity_threshold == 0.4
        assert dto.weight_optimization is True
        assert dto.diversity_weighting == 0.2
        assert dto.cross_validation_folds == 5
        assert dto.meta_learning_enabled is True

    def test_default_values(self):
        """Test default values."""
        dto = EnsembleConfigurationDTO(
            base_algorithms=["IsolationForest", "OneClassSVM"]
        )

        assert dto.ensemble_strategy == "voting"
        assert dto.max_ensemble_size == 5
        assert dto.min_diversity_threshold == 0.3
        assert dto.weight_optimization is True
        assert dto.diversity_weighting == 0.3
        assert dto.cross_validation_folds == 3
        assert dto.meta_learning_enabled is True

    def test_max_ensemble_size_validation(self):
        """Test max ensemble size validation."""
        # Valid range
        dto = EnsembleConfigurationDTO(
            base_algorithms=["IsolationForest", "OneClassSVM"], max_ensemble_size=3
        )
        assert dto.max_ensemble_size == 3

        # Invalid: too small
        with pytest.raises(ValidationError):
            EnsembleConfigurationDTO(
                base_algorithms=["IsolationForest", "OneClassSVM"], max_ensemble_size=1
            )

        # Invalid: too large
        with pytest.raises(ValidationError):
            EnsembleConfigurationDTO(
                base_algorithms=["IsolationForest", "OneClassSVM"], max_ensemble_size=11
            )

    def test_diversity_threshold_validation(self):
        """Test diversity threshold validation."""
        # Valid range
        dto = EnsembleConfigurationDTO(
            base_algorithms=["IsolationForest", "OneClassSVM"],
            min_diversity_threshold=0.5,
        )
        assert dto.min_diversity_threshold == 0.5

        # Invalid: negative
        with pytest.raises(ValidationError):
            EnsembleConfigurationDTO(
                base_algorithms=["IsolationForest", "OneClassSVM"],
                min_diversity_threshold=-0.1,
            )

        # Invalid: greater than 1
        with pytest.raises(ValidationError):
            EnsembleConfigurationDTO(
                base_algorithms=["IsolationForest", "OneClassSVM"],
                min_diversity_threshold=1.1,
            )

    def test_cv_folds_validation(self):
        """Test cross validation folds validation."""
        # Valid range
        dto = EnsembleConfigurationDTO(
            base_algorithms=["IsolationForest", "OneClassSVM"], cross_validation_folds=5
        )
        assert dto.cross_validation_folds == 5

        # Invalid: too small
        with pytest.raises(ValidationError):
            EnsembleConfigurationDTO(
                base_algorithms=["IsolationForest", "OneClassSVM"],
                cross_validation_folds=1,
            )

        # Invalid: too large
        with pytest.raises(ValidationError):
            EnsembleConfigurationDTO(
                base_algorithms=["IsolationForest", "OneClassSVM"],
                cross_validation_folds=11,
            )

    def test_required_fields(self):
        """Test required fields."""
        with pytest.raises(ValidationError):
            EnsembleConfigurationDTO()

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError):
            EnsembleConfigurationDTO(
                base_algorithms=["IsolationForest"], unknown_field="value"
            )


class TestEnsemblePerformanceDTO:
    """Test suite for EnsemblePerformanceDTO."""

    def test_valid_creation(self):
        """Test creating a valid ensemble performance DTO."""
        dto = EnsemblePerformanceDTO(
            weighted_individual_performance=0.75,
            diversity_score=0.6,
            estimated_ensemble_performance=0.85,
            performance_improvement=0.1,
            confidence_score=0.9,
        )

        assert dto.weighted_individual_performance == 0.75
        assert dto.diversity_score == 0.6
        assert dto.estimated_ensemble_performance == 0.85
        assert dto.performance_improvement == 0.1
        assert dto.confidence_score == 0.9

    def test_required_fields(self):
        """Test required fields."""
        with pytest.raises(ValidationError):
            EnsemblePerformanceDTO()

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError):
            EnsemblePerformanceDTO(
                weighted_individual_performance=0.75,
                diversity_score=0.6,
                estimated_ensemble_performance=0.85,
                performance_improvement=0.1,
                confidence_score=0.9,
                unknown_field="value",
            )


class TestEnsembleReportDTO:
    """Test suite for EnsembleReportDTO."""

    def test_valid_creation(self):
        """Test creating a valid ensemble report DTO."""
        diversity_metrics = DiversityMetricsDTO(
            disagreement_measure=0.6,
            double_fault_measure=0.2,
            q_statistic=0.3,
            correlation_coefficient=0.4,
            kappa_statistic=0.5,
            entropy_measure=0.7,
            overall_diversity=0.55,
        )

        config = EnsembleConfigurationDTO(
            base_algorithms=["IsolationForest", "OneClassSVM"]
        )

        performance = EnsemblePerformanceDTO(
            weighted_individual_performance=0.75,
            diversity_score=0.6,
            estimated_ensemble_performance=0.85,
            performance_improvement=0.1,
            confidence_score=0.9,
        )

        dto = EnsembleReportDTO(
            ensemble_summary={"size": 2, "strategy": "voting"},
            dataset_characteristics={"n_features": 10, "n_samples": 1000},
            individual_performance={
                "IsolationForest": {"accuracy": 0.8, "f1_score": 0.75},
                "OneClassSVM": {"accuracy": 0.78, "f1_score": 0.72},
            },
            diversity_analysis=diversity_metrics,
            ensemble_weights={"IsolationForest": 0.6, "OneClassSVM": 0.4},
            configuration=config,
            performance_summary=performance,
            recommendations=["Consider adding more diverse algorithms"],
            meta_learning_insights={"best_strategy": "voting"},
        )

        assert dto.ensemble_summary == {"size": 2, "strategy": "voting"}
        assert dto.dataset_characteristics == {"n_features": 10, "n_samples": 1000}
        assert "IsolationForest" in dto.individual_performance
        assert dto.diversity_analysis == diversity_metrics
        assert dto.ensemble_weights == {"IsolationForest": 0.6, "OneClassSVM": 0.4}
        assert dto.configuration == config
        assert dto.performance_summary == performance
        assert dto.recommendations == ["Consider adding more diverse algorithms"]
        assert dto.meta_learning_insights == {"best_strategy": "voting"}

    def test_required_fields(self):
        """Test required fields."""
        with pytest.raises(ValidationError):
            EnsembleReportDTO()

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        diversity_metrics = DiversityMetricsDTO(
            disagreement_measure=0.6,
            double_fault_measure=0.2,
            q_statistic=0.3,
            correlation_coefficient=0.4,
            kappa_statistic=0.5,
            entropy_measure=0.7,
            overall_diversity=0.55,
        )

        config = EnsembleConfigurationDTO(
            base_algorithms=["IsolationForest", "OneClassSVM"]
        )

        performance = EnsemblePerformanceDTO(
            weighted_individual_performance=0.75,
            diversity_score=0.6,
            estimated_ensemble_performance=0.85,
            performance_improvement=0.1,
            confidence_score=0.9,
        )

        with pytest.raises(ValidationError):
            EnsembleReportDTO(
                ensemble_summary={},
                dataset_characteristics={},
                individual_performance={},
                diversity_analysis=diversity_metrics,
                ensemble_weights={},
                configuration=config,
                performance_summary=performance,
                recommendations=[],
                meta_learning_insights={},
                unknown_field="value",
            )


class TestEnsembleDetectionRequestDTO:
    """Test suite for EnsembleDetectionRequestDTO."""

    def test_valid_creation_with_list_data(self):
        """Test creating a valid ensemble detection request with list data."""
        detector_ids = ["detector1", "detector2", "detector3"]
        data = [[1.0, 2.0, 3.0], [1.5, 2.5, 3.5], [2.0, 3.0, 4.0]]

        dto = EnsembleDetectionRequestDTO(
            detector_ids=detector_ids,
            data=data,
            voting_strategy="weighted_average",
            enable_dynamic_weighting=True,
            enable_uncertainty_estimation=True,
            confidence_threshold=0.9,
            consensus_threshold=0.7,
        )

        assert dto.detector_ids == detector_ids
        assert dto.data == data
        assert dto.voting_strategy == "weighted_average"
        assert dto.enable_dynamic_weighting is True
        assert dto.enable_uncertainty_estimation is True
        assert dto.confidence_threshold == 0.9
        assert dto.consensus_threshold == 0.7

    def test_valid_creation_with_dict_data(self):
        """Test creating a valid ensemble detection request with dict data."""
        detector_ids = ["detector1", "detector2"]
        data = [{"feature1": 1.0, "feature2": 2.0}, {"feature1": 1.5, "feature2": 2.5}]

        dto = EnsembleDetectionRequestDTO(detector_ids=detector_ids, data=data)

        assert dto.detector_ids == detector_ids
        assert dto.data == data

    def test_default_values(self):
        """Test default values."""
        dto = EnsembleDetectionRequestDTO(
            detector_ids=["detector1", "detector2"], data=[[1.0, 2.0], [1.5, 2.5]]
        )

        assert dto.voting_strategy == "dynamic_selection"
        assert dto.enable_dynamic_weighting is True
        assert dto.enable_uncertainty_estimation is True
        assert dto.enable_explanation is True
        assert dto.confidence_threshold == 0.8
        assert dto.consensus_threshold == 0.6
        assert dto.max_processing_time is None
        assert dto.enable_caching is True
        assert dto.return_individual_results is False

    def test_detector_ids_validation(self):
        """Test detector IDs validation."""
        # Valid: minimum 2 detectors
        dto = EnsembleDetectionRequestDTO(
            detector_ids=["detector1", "detector2"], data=[[1.0, 2.0]]
        )
        assert len(dto.detector_ids) == 2

        # Invalid: too few detectors
        with pytest.raises(ValidationError):
            EnsembleDetectionRequestDTO(detector_ids=["detector1"], data=[[1.0, 2.0]])

        # Invalid: too many detectors
        with pytest.raises(ValidationError):
            EnsembleDetectionRequestDTO(
                detector_ids=[f"detector{i}" for i in range(25)], data=[[1.0, 2.0]]
            )

    def test_voting_strategy_validation(self):
        """Test voting strategy validation."""
        # Valid strategy
        dto = EnsembleDetectionRequestDTO(
            detector_ids=["detector1", "detector2"],
            data=[[1.0, 2.0]],
            voting_strategy="simple_average",
        )
        assert dto.voting_strategy == "simple_average"

        # Invalid strategy
        with pytest.raises(ValidationError):
            EnsembleDetectionRequestDTO(
                detector_ids=["detector1", "detector2"],
                data=[[1.0, 2.0]],
                voting_strategy="invalid_strategy",
            )

    def test_data_validation_list_format(self):
        """Test data validation for list format."""
        # Valid list data
        dto = EnsembleDetectionRequestDTO(
            detector_ids=["detector1", "detector2"], data=[[1.0, 2.0], [1.5, 2.5]]
        )
        assert len(dto.data) == 2

        # Invalid: empty data
        with pytest.raises(ValidationError):
            EnsembleDetectionRequestDTO(
                detector_ids=["detector1", "detector2"], data=[]
            )

        # Invalid: inconsistent feature count
        with pytest.raises(ValidationError):
            EnsembleDetectionRequestDTO(
                detector_ids=["detector1", "detector2"],
                data=[[1.0, 2.0], [1.5, 2.5, 3.0]],
            )

    def test_data_validation_dict_format(self):
        """Test data validation for dict format."""
        # Valid dict data
        dto = EnsembleDetectionRequestDTO(
            detector_ids=["detector1", "detector2"],
            data=[
                {"feature1": 1.0, "feature2": 2.0},
                {"feature1": 1.5, "feature2": 2.5},
            ],
        )
        assert len(dto.data) == 2

        # Invalid: inconsistent keys
        with pytest.raises(ValidationError):
            EnsembleDetectionRequestDTO(
                detector_ids=["detector1", "detector2"],
                data=[
                    {"feature1": 1.0, "feature2": 2.0},
                    {"feature1": 1.5, "feature3": 2.5},
                ],
            )

    def test_threshold_validation(self):
        """Test threshold validation."""
        # Valid thresholds
        dto = EnsembleDetectionRequestDTO(
            detector_ids=["detector1", "detector2"],
            data=[[1.0, 2.0]],
            confidence_threshold=0.5,
            consensus_threshold=0.7,
        )
        assert dto.confidence_threshold == 0.5
        assert dto.consensus_threshold == 0.7

        # Invalid: negative threshold
        with pytest.raises(ValidationError):
            EnsembleDetectionRequestDTO(
                detector_ids=["detector1", "detector2"],
                data=[[1.0, 2.0]],
                confidence_threshold=-0.1,
            )

        # Invalid: threshold greater than 1
        with pytest.raises(ValidationError):
            EnsembleDetectionRequestDTO(
                detector_ids=["detector1", "detector2"],
                data=[[1.0, 2.0]],
                consensus_threshold=1.1,
            )

    def test_max_processing_time_validation(self):
        """Test max processing time validation."""
        # Valid: positive value
        dto = EnsembleDetectionRequestDTO(
            detector_ids=["detector1", "detector2"],
            data=[[1.0, 2.0]],
            max_processing_time=60.0,
        )
        assert dto.max_processing_time == 60.0

        # Invalid: negative value
        with pytest.raises(ValidationError):
            EnsembleDetectionRequestDTO(
                detector_ids=["detector1", "detector2"],
                data=[[1.0, 2.0]],
                max_processing_time=-10.0,
            )

        # Invalid: zero value
        with pytest.raises(ValidationError):
            EnsembleDetectionRequestDTO(
                detector_ids=["detector1", "detector2"],
                data=[[1.0, 2.0]],
                max_processing_time=0.0,
            )

    def test_required_fields(self):
        """Test required fields."""
        with pytest.raises(ValidationError):
            EnsembleDetectionRequestDTO(detector_ids=["detector1", "detector2"])

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError):
            EnsembleDetectionRequestDTO(
                detector_ids=["detector1", "detector2"],
                data=[[1.0, 2.0]],
                unknown_field="value",
            )


class TestEnsembleDetectionResponseDTO:
    """Test suite for EnsembleDetectionResponseDTO."""

    def test_valid_creation(self):
        """Test creating a valid ensemble detection response DTO."""
        dto = EnsembleDetectionResponseDTO(
            success=True,
            predictions=[0, 1, 0, 1],
            anomaly_scores=[0.2, 0.8, 0.3, 0.9],
            confidence_scores=[0.9, 0.8, 0.85, 0.95],
            uncertainty_scores=[0.1, 0.2, 0.15, 0.05],
            consensus_scores=[0.8, 0.9, 0.7, 0.95],
            individual_results={
                "detector1": [0.3, 0.7, 0.4, 0.8],
                "detector2": [0.1, 0.9, 0.2, 1.0],
            },
            detector_weights=[0.6, 0.4],
            voting_strategy_used="weighted_average",
            ensemble_metrics={"diversity": 0.7, "performance": 0.85},
            explanations=[
                {"feature_importance": {"feature1": 0.6, "feature2": 0.4}},
                {"feature_importance": {"feature1": 0.8, "feature2": 0.2}},
            ],
            processing_time=1.5,
            warnings=["Low confidence on sample 2"],
            error_message=None,
        )

        assert dto.success is True
        assert dto.predictions == [0, 1, 0, 1]
        assert dto.anomaly_scores == [0.2, 0.8, 0.3, 0.9]
        assert dto.confidence_scores == [0.9, 0.8, 0.85, 0.95]
        assert dto.uncertainty_scores == [0.1, 0.2, 0.15, 0.05]
        assert dto.consensus_scores == [0.8, 0.9, 0.7, 0.95]
        assert dto.individual_results == {
            "detector1": [0.3, 0.7, 0.4, 0.8],
            "detector2": [0.1, 0.9, 0.2, 1.0],
        }
        assert dto.detector_weights == [0.6, 0.4]
        assert dto.voting_strategy_used == "weighted_average"
        assert dto.ensemble_metrics == {"diversity": 0.7, "performance": 0.85}
        assert len(dto.explanations) == 2
        assert dto.processing_time == 1.5
        assert dto.warnings == ["Low confidence on sample 2"]
        assert dto.error_message is None

    def test_default_values(self):
        """Test default values."""
        dto = EnsembleDetectionResponseDTO(success=True)

        assert dto.success is True
        assert dto.predictions == []
        assert dto.anomaly_scores == []
        assert dto.confidence_scores == []
        assert dto.uncertainty_scores == []
        assert dto.consensus_scores == []
        assert dto.individual_results is None
        assert dto.detector_weights == []
        assert dto.voting_strategy_used == ""
        assert dto.ensemble_metrics == {}
        assert dto.explanations == []
        assert dto.processing_time == 0.0
        assert dto.warnings == []
        assert dto.error_message is None

    def test_failure_response(self):
        """Test failure response."""
        dto = EnsembleDetectionResponseDTO(
            success=False, error_message="Detection failed due to timeout"
        )

        assert dto.success is False
        assert dto.error_message == "Detection failed due to timeout"

    def test_required_fields(self):
        """Test required fields."""
        with pytest.raises(ValidationError):
            EnsembleDetectionResponseDTO()

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError):
            EnsembleDetectionResponseDTO(success=True, unknown_field="value")


class TestEnsembleOptimizationRequestDTO:
    """Test suite for EnsembleOptimizationRequestDTO."""

    def test_valid_creation(self):
        """Test creating a valid ensemble optimization request DTO."""
        detector_ids = ["detector1", "detector2", "detector3", "detector4"]
        strategies = ["weighted_average", "dynamic_selection"]

        dto = EnsembleOptimizationRequestDTO(
            detector_ids=detector_ids,
            validation_dataset_id="dataset123",
            optimization_objective="f1_score",
            target_voting_strategies=strategies,
            max_ensemble_size=3,
            min_diversity_threshold=0.4,
            enable_pruning=True,
            enable_weight_optimization=True,
            cross_validation_folds=5,
            optimization_timeout=300.0,
            random_state=42,
        )

        assert dto.detector_ids == detector_ids
        assert dto.validation_dataset_id == "dataset123"
        assert dto.optimization_objective == "f1_score"
        assert dto.target_voting_strategies == strategies
        assert dto.max_ensemble_size == 3
        assert dto.min_diversity_threshold == 0.4
        assert dto.enable_pruning is True
        assert dto.enable_weight_optimization is True
        assert dto.cross_validation_folds == 5
        assert dto.optimization_timeout == 300.0
        assert dto.random_state == 42

    def test_default_values(self):
        """Test default values."""
        dto = EnsembleOptimizationRequestDTO(
            detector_ids=["detector1", "detector2"], validation_dataset_id="dataset123"
        )

        assert dto.optimization_objective == "f1_score"
        assert dto.target_voting_strategies == ["dynamic_selection"]
        assert dto.max_ensemble_size == 5
        assert dto.min_diversity_threshold == 0.3
        assert dto.enable_pruning is True
        assert dto.enable_weight_optimization is True
        assert dto.cross_validation_folds == 5
        assert dto.optimization_timeout == 300.0
        assert dto.random_state == 42

    def test_detector_ids_validation(self):
        """Test detector IDs validation."""
        # Valid: minimum 2 detectors
        dto = EnsembleOptimizationRequestDTO(
            detector_ids=["detector1", "detector2"], validation_dataset_id="dataset123"
        )
        assert len(dto.detector_ids) == 2

        # Invalid: too few detectors
        with pytest.raises(ValidationError):
            EnsembleOptimizationRequestDTO(
                detector_ids=["detector1"], validation_dataset_id="dataset123"
            )

    def test_max_ensemble_size_validation(self):
        """Test max ensemble size validation."""
        # Valid range
        dto = EnsembleOptimizationRequestDTO(
            detector_ids=["detector1", "detector2"],
            validation_dataset_id="dataset123",
            max_ensemble_size=3,
        )
        assert dto.max_ensemble_size == 3

        # Invalid: too small
        with pytest.raises(ValidationError):
            EnsembleOptimizationRequestDTO(
                detector_ids=["detector1", "detector2"],
                validation_dataset_id="dataset123",
                max_ensemble_size=1,
            )

        # Invalid: too large
        with pytest.raises(ValidationError):
            EnsembleOptimizationRequestDTO(
                detector_ids=["detector1", "detector2"],
                validation_dataset_id="dataset123",
                max_ensemble_size=11,
            )

    def test_optimization_objective_validation(self):
        """Test optimization objective validation."""
        # Valid objective
        dto = EnsembleOptimizationRequestDTO(
            detector_ids=["detector1", "detector2"],
            validation_dataset_id="dataset123",
            optimization_objective="precision",
        )
        assert dto.optimization_objective == "precision"

        # Invalid objective
        with pytest.raises(ValidationError):
            EnsembleOptimizationRequestDTO(
                detector_ids=["detector1", "detector2"],
                validation_dataset_id="dataset123",
                optimization_objective="invalid_objective",
            )

    def test_target_voting_strategies_validation(self):
        """Test target voting strategies validation."""
        # Valid strategies
        dto = EnsembleOptimizationRequestDTO(
            detector_ids=["detector1", "detector2"],
            validation_dataset_id="dataset123",
            target_voting_strategies=["simple_average", "weighted_average"],
        )
        assert dto.target_voting_strategies == ["simple_average", "weighted_average"]

        # Invalid strategy
        with pytest.raises(ValidationError):
            EnsembleOptimizationRequestDTO(
                detector_ids=["detector1", "detector2"],
                validation_dataset_id="dataset123",
                target_voting_strategies=["invalid_strategy"],
            )

    def test_cross_validation_folds_validation(self):
        """Test cross validation folds validation."""
        # Valid range
        dto = EnsembleOptimizationRequestDTO(
            detector_ids=["detector1", "detector2"],
            validation_dataset_id="dataset123",
            cross_validation_folds=3,
        )
        assert dto.cross_validation_folds == 3

        # Invalid: too small
        with pytest.raises(ValidationError):
            EnsembleOptimizationRequestDTO(
                detector_ids=["detector1", "detector2"],
                validation_dataset_id="dataset123",
                cross_validation_folds=1,
            )

        # Invalid: too large
        with pytest.raises(ValidationError):
            EnsembleOptimizationRequestDTO(
                detector_ids=["detector1", "detector2"],
                validation_dataset_id="dataset123",
                cross_validation_folds=11,
            )

    def test_optimization_timeout_validation(self):
        """Test optimization timeout validation."""
        # Valid: positive value
        dto = EnsembleOptimizationRequestDTO(
            detector_ids=["detector1", "detector2"],
            validation_dataset_id="dataset123",
            optimization_timeout=600.0,
        )
        assert dto.optimization_timeout == 600.0

        # Invalid: negative value
        with pytest.raises(ValidationError):
            EnsembleOptimizationRequestDTO(
                detector_ids=["detector1", "detector2"],
                validation_dataset_id="dataset123",
                optimization_timeout=-10.0,
            )

        # Invalid: zero value
        with pytest.raises(ValidationError):
            EnsembleOptimizationRequestDTO(
                detector_ids=["detector1", "detector2"],
                validation_dataset_id="dataset123",
                optimization_timeout=0.0,
            )

    def test_required_fields(self):
        """Test required fields."""
        with pytest.raises(ValidationError):
            EnsembleOptimizationRequestDTO(detector_ids=["detector1", "detector2"])

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError):
            EnsembleOptimizationRequestDTO(
                detector_ids=["detector1", "detector2"],
                validation_dataset_id="dataset123",
                unknown_field="value",
            )


class TestEnsembleOptimizationResponseDTO:
    """Test suite for EnsembleOptimizationResponseDTO."""

    def test_valid_creation(self):
        """Test creating a valid ensemble optimization response DTO."""
        dto = EnsembleOptimizationResponseDTO(
            success=True,
            optimized_detector_ids=["detector1", "detector3"],
            optimal_voting_strategy="weighted_average",
            optimal_weights=[0.6, 0.4],
            ensemble_performance={"accuracy": 0.85, "f1_score": 0.82},
            diversity_metrics={"overall_diversity": 0.7, "disagreement": 0.6},
            optimization_history=[
                {"iteration": 1, "objective": 0.75},
                {"iteration": 2, "objective": 0.82},
            ],
            recommendations=["Consider adding more diverse algorithms"],
            optimization_time=45.5,
            error_message=None,
        )

        assert dto.success is True
        assert dto.optimized_detector_ids == ["detector1", "detector3"]
        assert dto.optimal_voting_strategy == "weighted_average"
        assert dto.optimal_weights == [0.6, 0.4]
        assert dto.ensemble_performance == {"accuracy": 0.85, "f1_score": 0.82}
        assert dto.diversity_metrics == {"overall_diversity": 0.7, "disagreement": 0.6}
        assert len(dto.optimization_history) == 2
        assert dto.recommendations == ["Consider adding more diverse algorithms"]
        assert dto.optimization_time == 45.5
        assert dto.error_message is None

    def test_default_values(self):
        """Test default values."""
        dto = EnsembleOptimizationResponseDTO(success=True)

        assert dto.success is True
        assert dto.optimized_detector_ids == []
        assert dto.optimal_voting_strategy == ""
        assert dto.optimal_weights == []
        assert dto.ensemble_performance == {}
        assert dto.diversity_metrics == {}
        assert dto.optimization_history == []
        assert dto.recommendations == []
        assert dto.optimization_time == 0.0
        assert dto.error_message is None

    def test_failure_response(self):
        """Test failure response."""
        dto = EnsembleOptimizationResponseDTO(
            success=False,
            error_message="Optimization failed due to insufficient diversity",
        )

        assert dto.success is False
        assert dto.error_message == "Optimization failed due to insufficient diversity"

    def test_required_fields(self):
        """Test required fields."""
        with pytest.raises(ValidationError):
            EnsembleOptimizationResponseDTO()

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError):
            EnsembleOptimizationResponseDTO(success=True, unknown_field="value")


class TestUtilityFunctions:
    """Test suite for utility functions."""

    def test_create_default_ensemble_config(self):
        """Test creating default ensemble configuration."""
        config = create_default_ensemble_config()

        assert isinstance(config, EnsembleConfigurationDTO)
        assert config.base_algorithms == [
            "IsolationForest",
            "LocalOutlierFactor",
            "OneClassSVM",
        ]
        assert config.ensemble_strategy == "voting"
        assert config.max_ensemble_size == 5
        assert config.min_diversity_threshold == 0.3
        assert config.weight_optimization is True
        assert config.diversity_weighting == 0.3
        assert config.cross_validation_folds == 3
        assert config.meta_learning_enabled is True

    def test_create_diversity_metrics_from_dict(self):
        """Test creating diversity metrics from dictionary."""
        data = {
            "disagreement_measure": 0.6,
            "double_fault_measure": 0.2,
            "q_statistic": 0.3,
            "correlation_coefficient": 0.4,
            "kappa_statistic": 0.5,
            "entropy_measure": 0.7,
            "overall_diversity": 0.55,
        }

        metrics = create_diversity_metrics_from_dict(data)

        assert isinstance(metrics, DiversityMetricsDTO)
        assert metrics.disagreement_measure == 0.6
        assert metrics.double_fault_measure == 0.2
        assert metrics.q_statistic == 0.3
        assert metrics.correlation_coefficient == 0.4
        assert metrics.kappa_statistic == 0.5
        assert metrics.entropy_measure == 0.7
        assert metrics.overall_diversity == 0.55

    def test_create_diversity_metrics_from_dict_missing_values(self):
        """Test creating diversity metrics with missing values."""
        data = {"disagreement_measure": 0.6, "overall_diversity": 0.55}

        metrics = create_diversity_metrics_from_dict(data)

        assert metrics.disagreement_measure == 0.6
        assert metrics.double_fault_measure == 0.0
        assert metrics.q_statistic == 0.0
        assert metrics.correlation_coefficient == 0.0
        assert metrics.kappa_statistic == 0.0
        assert metrics.entropy_measure == 0.0
        assert metrics.overall_diversity == 0.55

    def test_create_ensemble_performance_from_dict(self):
        """Test creating ensemble performance from dictionary."""
        data = {
            "weighted_individual_performance": 0.75,
            "diversity_score": 0.6,
            "estimated_ensemble_performance": 0.85,
            "performance_improvement": 0.1,
            "confidence_score": 0.9,
        }

        performance = create_ensemble_performance_from_dict(data)

        assert isinstance(performance, EnsemblePerformanceDTO)
        assert performance.weighted_individual_performance == 0.75
        assert performance.diversity_score == 0.6
        assert performance.estimated_ensemble_performance == 0.85
        assert performance.performance_improvement == 0.1
        assert performance.confidence_score == 0.9

    def test_create_ensemble_performance_from_dict_missing_values(self):
        """Test creating ensemble performance with missing values."""
        data = {"diversity_score": 0.6, "confidence_score": 0.9}

        performance = create_ensemble_performance_from_dict(data)

        assert performance.weighted_individual_performance == 0.0
        assert performance.diversity_score == 0.6
        assert performance.estimated_ensemble_performance == 0.0
        assert performance.performance_improvement == 0.0
        assert performance.confidence_score == 0.9


class TestEnsembleDTOIntegration:
    """Test integration scenarios for ensemble DTOs."""

    def test_complete_ensemble_workflow(self):
        """Test complete ensemble workflow."""
        # Create ensemble configuration
        config = EnsembleConfigurationDTO(
            base_algorithms=["IsolationForest", "LocalOutlierFactor", "OneClassSVM"],
            ensemble_strategy="weighted_voting",
            max_ensemble_size=3,
            min_diversity_threshold=0.4,
            weight_optimization=True,
        )

        # Create optimization request
        optimization_request = EnsembleOptimizationRequestDTO(
            detector_ids=["detector1", "detector2", "detector3"],
            validation_dataset_id="dataset123",
            optimization_objective="f1_score",
            max_ensemble_size=config.max_ensemble_size,
            min_diversity_threshold=config.min_diversity_threshold,
        )

        # Create optimization response
        optimization_response = EnsembleOptimizationResponseDTO(
            success=True,
            optimized_detector_ids=["detector1", "detector3"],
            optimal_voting_strategy="weighted_average",
            optimal_weights=[0.6, 0.4],
            ensemble_performance={"f1_score": 0.85},
            optimization_time=30.0,
        )

        # Create detection request
        detection_request = EnsembleDetectionRequestDTO(
            detector_ids=optimization_response.optimized_detector_ids,
            data=[[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]],
            voting_strategy=optimization_response.optimal_voting_strategy,
        )

        # Create detection response
        detection_response = EnsembleDetectionResponseDTO(
            success=True,
            predictions=[0, 1],
            anomaly_scores=[0.3, 0.8],
            confidence_scores=[0.9, 0.85],
            detector_weights=optimization_response.optimal_weights,
            voting_strategy_used=optimization_response.optimal_voting_strategy,
        )

        # Verify workflow consistency
        assert optimization_request.max_ensemble_size == config.max_ensemble_size
        assert (
            detection_request.detector_ids
            == optimization_response.optimized_detector_ids
        )
        assert (
            detection_request.voting_strategy
            == optimization_response.optimal_voting_strategy
        )
        assert (
            detection_response.detector_weights == optimization_response.optimal_weights
        )
        assert (
            detection_response.voting_strategy_used
            == optimization_response.optimal_voting_strategy
        )

    def test_ensemble_report_generation(self):
        """Test ensemble report generation."""
        # Create diversity metrics
        diversity_metrics = create_diversity_metrics_from_dict(
            {
                "disagreement_measure": 0.6,
                "double_fault_measure": 0.2,
                "q_statistic": 0.3,
                "correlation_coefficient": 0.4,
                "kappa_statistic": 0.5,
                "entropy_measure": 0.7,
                "overall_diversity": 0.55,
            }
        )

        # Create ensemble performance
        performance = create_ensemble_performance_from_dict(
            {
                "weighted_individual_performance": 0.75,
                "diversity_score": 0.6,
                "estimated_ensemble_performance": 0.85,
                "performance_improvement": 0.1,
                "confidence_score": 0.9,
            }
        )

        # Create configuration
        config = create_default_ensemble_config()

        # Create ensemble report
        report = EnsembleReportDTO(
            ensemble_summary={"size": 3, "strategy": "weighted_voting"},
            dataset_characteristics={"n_features": 10, "n_samples": 1000},
            individual_performance={
                "IsolationForest": {"accuracy": 0.8, "f1_score": 0.75},
                "LocalOutlierFactor": {"accuracy": 0.78, "f1_score": 0.72},
                "OneClassSVM": {"accuracy": 0.76, "f1_score": 0.70},
            },
            diversity_analysis=diversity_metrics,
            ensemble_weights={
                "IsolationForest": 0.4,
                "LocalOutlierFactor": 0.35,
                "OneClassSVM": 0.25,
            },
            configuration=config,
            performance_summary=performance,
            recommendations=[
                "Consider adding more diverse algorithms",
                "Increase diversity threshold",
            ],
            meta_learning_insights={
                "best_strategy": "weighted_voting",
                "optimal_size": 3,
            },
        )

        # Verify report completeness
        assert len(report.individual_performance) == 3
        assert report.diversity_analysis.overall_diversity == 0.55
        assert report.performance_summary.estimated_ensemble_performance == 0.85
        assert len(report.recommendations) == 2
        assert "best_strategy" in report.meta_learning_insights

    def test_dto_serialization(self):
        """Test DTO serialization and deserialization."""
        # Create ensemble configuration
        original_config = EnsembleConfigurationDTO(
            base_algorithms=["IsolationForest", "OneClassSVM"],
            ensemble_strategy="voting",
            max_ensemble_size=3,
            min_diversity_threshold=0.4,
            weight_optimization=True,
        )

        # Serialize to dict
        config_dict = original_config.model_dump()

        assert config_dict["base_algorithms"] == ["IsolationForest", "OneClassSVM"]
        assert config_dict["ensemble_strategy"] == "voting"
        assert config_dict["max_ensemble_size"] == 3
        assert config_dict["min_diversity_threshold"] == 0.4
        assert config_dict["weight_optimization"] is True

        # Deserialize from dict
        restored_config = EnsembleConfigurationDTO.model_validate(config_dict)

        assert restored_config.base_algorithms == original_config.base_algorithms
        assert restored_config.ensemble_strategy == original_config.ensemble_strategy
        assert restored_config.max_ensemble_size == original_config.max_ensemble_size
        assert (
            restored_config.min_diversity_threshold
            == original_config.min_diversity_threshold
        )
        assert (
            restored_config.weight_optimization == original_config.weight_optimization
        )

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Minimum ensemble size
        config = EnsembleConfigurationDTO(
            base_algorithms=["IsolationForest", "OneClassSVM"], max_ensemble_size=2
        )
        assert config.max_ensemble_size == 2

        # Zero diversity threshold
        config = EnsembleConfigurationDTO(
            base_algorithms=["IsolationForest", "OneClassSVM"],
            min_diversity_threshold=0.0,
        )
        assert config.min_diversity_threshold == 0.0

        # Maximum diversity threshold
        config = EnsembleConfigurationDTO(
            base_algorithms=["IsolationForest", "OneClassSVM"],
            min_diversity_threshold=1.0,
        )
        assert config.min_diversity_threshold == 1.0

        # Minimum CV folds
        config = EnsembleConfigurationDTO(
            base_algorithms=["IsolationForest", "OneClassSVM"], cross_validation_folds=2
        )
        assert config.cross_validation_folds == 2

        # Maximum CV folds
        config = EnsembleConfigurationDTO(
            base_algorithms=["IsolationForest", "OneClassSVM"],
            cross_validation_folds=10,
        )
        assert config.cross_validation_folds == 10

    def test_complex_data_structures(self):
        """Test complex data structures."""
        # Create meta-learning knowledge
        knowledge = MetaLearningKnowledgeDTO(
            dataset_characteristics={
                "n_features": 100,
                "n_samples": 10000,
                "feature_types": ["numerical", "categorical"],
                "class_imbalance": 0.1,
            },
            algorithm_performance={
                "IsolationForest": 0.85,
                "LocalOutlierFactor": 0.78,
                "OneClassSVM": 0.76,
                "EllipticEnvelope": 0.72,
            },
            ensemble_composition=[
                "IsolationForest",
                "LocalOutlierFactor",
                "OneClassSVM",
            ],
            optimal_weights={
                "IsolationForest": 0.4,
                "LocalOutlierFactor": 0.35,
                "OneClassSVM": 0.25,
            },
            diversity_requirements={"min_diversity": 0.3, "target_diversity": 0.5},
            performance_metrics={
                "accuracy": 0.82,
                "precision": 0.79,
                "recall": 0.85,
                "f1_score": 0.82,
            },
            confidence_score=0.9,
        )

        # Verify complex structure
        assert knowledge.dataset_characteristics["n_features"] == 100
        assert len(knowledge.algorithm_performance) == 4
        assert len(knowledge.ensemble_composition) == 3
        assert sum(knowledge.optimal_weights.values()) == 1.0
        assert knowledge.diversity_requirements["target_diversity"] == 0.5
        assert knowledge.performance_metrics["f1_score"] == 0.82
        assert knowledge.confidence_score == 0.9
        assert isinstance(knowledge.timestamp, datetime)
