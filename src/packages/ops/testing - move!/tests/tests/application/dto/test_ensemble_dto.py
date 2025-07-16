"""
Comprehensive tests for ensemble DTOs.

This module tests all ensemble-related Data Transfer Objects to ensure proper validation,
serialization, and behavior across all use cases including ensemble strategies, meta-learning,
diversity metrics, and ensemble detection.
"""

from datetime import datetime

import pytest
from pydantic import ValidationError

from monorepo.application.dto.ensemble_dto import (
    DiversityMetricsDTO,
    EnsembleConfigurationDTO,
    EnsembleDetectionRequestDTO,
    EnsembleDetectionResponseDTO,
    EnsembleDetectorOptimizationRequestDTO,
    EnsembleDetectorOptimizationResponseDTO,
    EnsemblePerformanceDTO,
    EnsembleStrategyDTO,
    MetaLearningKnowledgeDTO,
    create_default_ensemble_config,
    create_diversity_metrics_from_dict,
    create_ensemble_performance_from_dict,
)


class TestEnsembleStrategyDTO:
    """Test suite for EnsembleStrategyDTO."""

    def test_basic_creation(self):
        """Test basic ensemble strategy creation."""
        strategy = EnsembleStrategyDTO(
            name="voting", description="Simple majority voting strategy"
        )

        assert strategy.name == "voting"
        assert strategy.description == "Simple majority voting strategy"
        assert strategy.requires_training is False  # Default
        assert strategy.supports_weights is True  # Default
        assert strategy.complexity == "medium"  # Default
        assert strategy.interpretability == 0.5  # Default

    def test_complete_creation(self):
        """Test ensemble strategy creation with all fields."""
        strategy = EnsembleStrategyDTO(
            name="meta_learning",
            description="Advanced meta-learning ensemble strategy",
            requires_training=True,
            supports_weights=False,
            complexity="high",
            interpretability=0.3,
        )

        assert strategy.requires_training is True
        assert strategy.supports_weights is False
        assert strategy.complexity == "high"
        assert strategy.interpretability == 0.3

    def test_interpretability_validation(self):
        """Test interpretability validation bounds."""
        # Test invalid values
        with pytest.raises(ValidationError):
            EnsembleStrategyDTO(
                name="test",
                description="test",
                interpretability=-0.1,  # Below minimum
            )

        with pytest.raises(ValidationError):
            EnsembleStrategyDTO(
                name="test",
                description="test",
                interpretability=1.1,  # Above maximum
            )

        # Test valid boundary values
        for interp in [0.0, 0.5, 1.0]:
            strategy = EnsembleStrategyDTO(
                name="test", description="test", interpretability=interp
            )
            assert strategy.interpretability == interp

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError):
            EnsembleStrategyDTO(
                name="test",
                description="test",
                extra_field="not_allowed",  # type: ignore
            )


class TestDiversityMetricsDTO:
    """Test suite for DiversityMetricsDTO."""

    def test_basic_creation(self):
        """Test basic diversity metrics creation."""
        metrics = DiversityMetricsDTO(
            disagreement_measure=0.45,
            double_fault_measure=0.12,
            q_statistic=0.3,
            correlation_coefficient=0.25,
            kappa_statistic=0.4,
            entropy_measure=0.8,
            overall_diversity=0.5,
        )

        assert metrics.disagreement_measure == 0.45
        assert metrics.double_fault_measure == 0.12
        assert metrics.q_statistic == 0.3
        assert metrics.correlation_coefficient == 0.25
        assert metrics.kappa_statistic == 0.4
        assert metrics.entropy_measure == 0.8
        assert metrics.overall_diversity == 0.5

    def test_negative_values(self):
        """Test that negative values are allowed for some metrics."""
        # Some diversity metrics can be negative (like correlation)
        metrics = DiversityMetricsDTO(
            disagreement_measure=0.0,
            double_fault_measure=0.0,
            q_statistic=-0.5,  # Can be negative
            correlation_coefficient=-0.3,  # Can be negative
            kappa_statistic=-0.1,  # Can be negative
            entropy_measure=0.0,
            overall_diversity=0.0,
        )

        assert metrics.q_statistic == -0.5
        assert metrics.correlation_coefficient == -0.3
        assert metrics.kappa_statistic == -0.1

    def test_high_diversity_scenario(self):
        """Test scenario with high diversity metrics."""
        metrics = DiversityMetricsDTO(
            disagreement_measure=0.8,
            double_fault_measure=0.05,
            q_statistic=-0.2,  # Negative indicates diversity
            correlation_coefficient=0.1,  # Low correlation indicates diversity
            kappa_statistic=0.1,  # Low kappa indicates diversity
            entropy_measure=0.95,
            overall_diversity=0.85,
        )

        assert metrics.overall_diversity == 0.85
        assert metrics.entropy_measure == 0.95
        assert metrics.disagreement_measure == 0.8


class TestMetaLearningKnowledgeDTO:
    """Test suite for MetaLearningKnowledgeDTO."""

    def test_basic_creation(self):
        """Test basic meta-learning knowledge creation."""
        characteristics = {"n_samples": 1000, "n_features": 20, "contamination": 0.1}
        performance = {"isolation_forest": 0.85, "one_class_svm": 0.78}
        composition = ["isolation_forest", "local_outlier_factor"]
        weights = {"isolation_forest": 0.6, "local_outlier_factor": 0.4}
        diversity_req = {"min_disagreement": 0.3, "max_correlation": 0.7}
        perf_metrics = {"accuracy": 0.92, "f1_score": 0.88}

        knowledge = MetaLearningKnowledgeDTO(
            dataset_characteristics=characteristics,
            algorithm_performance=performance,
            ensemble_composition=composition,
            optimal_weights=weights,
            diversity_requirements=diversity_req,
            performance_metrics=perf_metrics,
            confidence_score=0.85,
        )

        assert knowledge.dataset_characteristics == characteristics
        assert knowledge.algorithm_performance == performance
        assert knowledge.ensemble_composition == composition
        assert knowledge.optimal_weights == weights
        assert knowledge.diversity_requirements == diversity_req
        assert knowledge.performance_metrics == perf_metrics
        assert knowledge.confidence_score == 0.85
        assert isinstance(knowledge.timestamp, datetime)

    def test_custom_timestamp(self):
        """Test with custom timestamp."""
        custom_time = datetime(2024, 1, 1, 12, 0, 0)

        knowledge = MetaLearningKnowledgeDTO(
            dataset_characteristics={},
            algorithm_performance={},
            ensemble_composition=[],
            optimal_weights={},
            diversity_requirements={},
            performance_metrics={},
            confidence_score=0.5,
            timestamp=custom_time,
        )

        assert knowledge.timestamp == custom_time

    def test_empty_collections(self):
        """Test with empty collections."""
        knowledge = MetaLearningKnowledgeDTO(
            dataset_characteristics={},
            algorithm_performance={},
            ensemble_composition=[],
            optimal_weights={},
            diversity_requirements={},
            performance_metrics={},
            confidence_score=0.0,
        )

        assert knowledge.ensemble_composition == []
        assert knowledge.optimal_weights == {}
        assert knowledge.confidence_score == 0.0


class TestEnsembleConfigurationDTO:
    """Test suite for EnsembleConfigurationDTO."""

    def test_basic_creation(self):
        """Test basic ensemble configuration creation."""
        algorithms = ["isolation_forest", "one_class_svm"]

        config = EnsembleConfigurationDTO(base_algorithms=algorithms)

        assert config.base_algorithms == algorithms
        assert config.ensemble_strategy == "voting"  # Default
        assert config.max_ensemble_size == 5  # Default
        assert config.min_diversity_threshold == 0.3  # Default
        assert config.weight_optimization is True  # Default
        assert config.diversity_weighting == 0.3  # Default
        assert config.cross_validation_folds == 3  # Default
        assert config.meta_learning_enabled is True  # Default

    def test_complete_creation(self):
        """Test ensemble configuration creation with all fields."""
        algorithms = ["isolation_forest", "one_class_svm", "local_outlier_factor"]

        config = EnsembleConfigurationDTO(
            base_algorithms=algorithms,
            ensemble_strategy="weighted_voting",
            max_ensemble_size=8,
            min_diversity_threshold=0.5,
            weight_optimization=False,
            diversity_weighting=0.7,
            cross_validation_folds=5,
            meta_learning_enabled=False,
        )

        assert config.ensemble_strategy == "weighted_voting"
        assert config.max_ensemble_size == 8
        assert config.min_diversity_threshold == 0.5
        assert config.weight_optimization is False
        assert config.diversity_weighting == 0.7
        assert config.cross_validation_folds == 5
        assert config.meta_learning_enabled is False

    def test_max_ensemble_size_validation(self):
        """Test max_ensemble_size validation bounds."""
        algorithms = ["algo1", "algo2"]

        # Test invalid values
        with pytest.raises(ValidationError):
            EnsembleConfigurationDTO(
                base_algorithms=algorithms,
                max_ensemble_size=1,  # Below minimum
            )

        with pytest.raises(ValidationError):
            EnsembleConfigurationDTO(
                base_algorithms=algorithms,
                max_ensemble_size=11,  # Above maximum
            )

        # Test valid boundary values
        for size in [2, 5, 10]:
            config = EnsembleConfigurationDTO(
                base_algorithms=algorithms, max_ensemble_size=size
            )
            assert config.max_ensemble_size == size

    def test_diversity_threshold_validation(self):
        """Test diversity threshold validation bounds."""
        algorithms = ["algo1", "algo2"]

        # Test valid boundary values
        for threshold in [0.0, 0.5, 1.0]:
            config = EnsembleConfigurationDTO(
                base_algorithms=algorithms, min_diversity_threshold=threshold
            )
            assert config.min_diversity_threshold == threshold

    def test_diversity_weighting_validation(self):
        """Test diversity weighting validation bounds."""
        algorithms = ["algo1", "algo2"]

        # Test valid boundary values
        for weighting in [0.0, 0.5, 1.0]:
            config = EnsembleConfigurationDTO(
                base_algorithms=algorithms, diversity_weighting=weighting
            )
            assert config.diversity_weighting == weighting

    def test_cross_validation_folds_validation(self):
        """Test cross validation folds validation bounds."""
        algorithms = ["algo1", "algo2"]

        # Test invalid values
        with pytest.raises(ValidationError):
            EnsembleConfigurationDTO(
                base_algorithms=algorithms,
                cross_validation_folds=1,  # Below minimum
            )

        with pytest.raises(ValidationError):
            EnsembleConfigurationDTO(
                base_algorithms=algorithms,
                cross_validation_folds=11,  # Above maximum
            )

        # Test valid boundary values
        for folds in [2, 5, 10]:
            config = EnsembleConfigurationDTO(
                base_algorithms=algorithms, cross_validation_folds=folds
            )
            assert config.cross_validation_folds == folds


class TestEnsembleDetectionRequestDTO:
    """Test suite for EnsembleDetectionRequestDTO."""

    def test_basic_creation(self):
        """Test basic ensemble detection request creation."""
        detector_ids = ["detector_1", "detector_2"]
        data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]

        request = EnsembleDetectionRequestDTO(detector_ids=detector_ids, data=data)

        assert request.detector_ids == detector_ids
        assert request.data == data
        assert request.voting_strategy == "dynamic_selection"  # Default
        assert request.enable_dynamic_weighting is True  # Default
        assert request.enable_uncertainty_estimation is True  # Default
        assert request.enable_explanation is True  # Default
        assert request.confidence_threshold == 0.8  # Default
        assert request.consensus_threshold == 0.6  # Default
        assert request.max_processing_time is None  # Default
        assert request.enable_caching is True  # Default
        assert request.return_individual_results is False  # Default

    def test_complete_creation(self):
        """Test ensemble detection request creation with all fields."""
        detector_ids = ["det1", "det2", "det3"]
        data = [[1.0, 2.0], [3.0, 4.0]]

        request = EnsembleDetectionRequestDTO(
            detector_ids=detector_ids,
            data=data,
            voting_strategy="weighted_average",
            enable_dynamic_weighting=False,
            enable_uncertainty_estimation=False,
            enable_explanation=False,
            confidence_threshold=0.9,
            consensus_threshold=0.7,
            max_processing_time=30.0,
            enable_caching=False,
            return_individual_results=True,
        )

        assert request.voting_strategy == "weighted_average"
        assert request.enable_dynamic_weighting is False
        assert request.enable_uncertainty_estimation is False
        assert request.enable_explanation is False
        assert request.confidence_threshold == 0.9
        assert request.consensus_threshold == 0.7
        assert request.max_processing_time == 30.0
        assert request.enable_caching is False
        assert request.return_individual_results is True

    def test_detector_ids_validation(self):
        """Test detector IDs validation."""
        data = [[1.0, 2.0]]

        # Test minimum detector count
        with pytest.raises(ValidationError):
            EnsembleDetectionRequestDTO(
                detector_ids=["single_detector"],  # Only one detector
                data=data,
            )

        # Test maximum detector count
        too_many_detectors = [f"detector_{i}" for i in range(21)]  # 21 detectors
        with pytest.raises(ValidationError):
            EnsembleDetectionRequestDTO(detector_ids=too_many_detectors, data=data)

        # Test valid counts
        for count in [2, 10, 20]:
            detector_ids = [f"detector_{i}" for i in range(count)]
            request = EnsembleDetectionRequestDTO(detector_ids=detector_ids, data=data)
            assert len(request.detector_ids) == count

    def test_voting_strategy_validation(self):
        """Test voting strategy validation."""
        detector_ids = ["det1", "det2"]
        data = [[1.0]]

        # Test invalid strategy
        with pytest.raises(ValidationError):
            EnsembleDetectionRequestDTO(
                detector_ids=detector_ids, data=data, voting_strategy="invalid_strategy"
            )

        # Test valid strategies
        valid_strategies = [
            "simple_average",
            "weighted_average",
            "bayesian_model_averaging",
            "rank_aggregation",
            "consensus_voting",
            "dynamic_selection",
            "uncertainty_weighted",
            "performance_weighted",
            "diversity_weighted",
            "adaptive_threshold",
            "robust_aggregation",
            "cascaded_voting",
        ]

        for strategy in valid_strategies:
            request = EnsembleDetectionRequestDTO(
                detector_ids=detector_ids, data=data, voting_strategy=strategy
            )
            assert request.voting_strategy == strategy

    def test_data_validation_list_of_lists(self):
        """Test data validation for list of lists format."""
        detector_ids = ["det1", "det2"]

        # Test empty data
        with pytest.raises(ValidationError):
            EnsembleDetectionRequestDTO(detector_ids=detector_ids, data=[])

        # Test valid list of lists
        valid_data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        request = EnsembleDetectionRequestDTO(
            detector_ids=detector_ids, data=valid_data
        )
        assert request.data == valid_data

        # Test inconsistent feature counts
        with pytest.raises(ValidationError):
            EnsembleDetectionRequestDTO(
                detector_ids=detector_ids,
                data=[[1.0, 2.0], [3.0, 4.0, 5.0]],  # Different feature counts
            )

    def test_data_validation_list_of_dicts(self):
        """Test data validation for list of dictionaries format."""
        detector_ids = ["det1", "det2"]

        # Test valid list of dicts
        valid_data = [
            {"feature1": 1.0, "feature2": 2.0},
            {"feature1": 3.0, "feature2": 4.0},
        ]
        request = EnsembleDetectionRequestDTO(
            detector_ids=detector_ids, data=valid_data
        )
        assert request.data == valid_data

        # Test inconsistent dictionary keys
        with pytest.raises(ValidationError):
            EnsembleDetectionRequestDTO(
                detector_ids=detector_ids,
                data=[
                    {"feature1": 1.0, "feature2": 2.0},
                    {"feature1": 3.0, "feature3": 4.0},  # Different keys
                ],
            )

    def test_threshold_validation(self):
        """Test threshold validation bounds."""
        detector_ids = ["det1", "det2"]
        data = [[1.0]]

        # Test valid threshold values
        for threshold in [0.0, 0.5, 1.0]:
            request = EnsembleDetectionRequestDTO(
                detector_ids=detector_ids,
                data=data,
                confidence_threshold=threshold,
                consensus_threshold=threshold,
            )
            assert request.confidence_threshold == threshold
            assert request.consensus_threshold == threshold

    def test_max_processing_time_validation(self):
        """Test max processing time validation."""
        detector_ids = ["det1", "det2"]
        data = [[1.0]]

        # Test invalid processing time
        with pytest.raises(ValidationError):
            EnsembleDetectionRequestDTO(
                detector_ids=detector_ids,
                data=data,
                max_processing_time=0.0,  # Must be positive
            )

        # Test valid processing time
        request = EnsembleDetectionRequestDTO(
            detector_ids=detector_ids, data=data, max_processing_time=60.0
        )
        assert request.max_processing_time == 60.0


class TestEnsembleDetectionResponseDTO:
    """Test suite for EnsembleDetectionResponseDTO."""

    def test_basic_creation(self):
        """Test basic ensemble detection response creation."""
        response = EnsembleDetectionResponseDTO(success=True)

        assert response.success is True
        assert response.predictions == []  # Default
        assert response.anomaly_scores == []  # Default
        assert response.confidence_scores == []  # Default
        assert response.uncertainty_scores == []  # Default
        assert response.consensus_scores == []  # Default
        assert response.individual_results is None  # Default
        assert response.detector_weights == []  # Default
        assert response.voting_strategy_used == ""  # Default
        assert response.ensemble_metrics == {}  # Default
        assert response.explanations == []  # Default
        assert response.processing_time == 0.0  # Default
        assert response.warnings == []  # Default
        assert response.error_message is None  # Default

    def test_complete_creation(self):
        """Test ensemble detection response creation with all fields."""
        predictions = [0, 1, 0, 1]
        anomaly_scores = [0.2, 0.8, 0.3, 0.9]
        confidence_scores = [0.9, 0.95, 0.85, 0.92]
        uncertainty_scores = [0.1, 0.05, 0.15, 0.08]
        consensus_scores = [0.8, 0.9, 0.7, 0.95]
        individual_results = {
            "det1": [0.1, 0.7, 0.2, 0.8],
            "det2": [0.3, 0.9, 0.4, 1.0],
        }
        detector_weights = [0.6, 0.4]
        ensemble_metrics = {"diversity": 0.7, "agreement": 0.85}
        explanations = [{"top_feature": "feature1"}, {"top_feature": "feature2"}]
        warnings = ["Low confidence detected"]

        response = EnsembleDetectionResponseDTO(
            success=True,
            predictions=predictions,
            anomaly_scores=anomaly_scores,
            confidence_scores=confidence_scores,
            uncertainty_scores=uncertainty_scores,
            consensus_scores=consensus_scores,
            individual_results=individual_results,
            detector_weights=detector_weights,
            voting_strategy_used="dynamic_selection",
            ensemble_metrics=ensemble_metrics,
            explanations=explanations,
            processing_time=5.2,
            warnings=warnings,
        )

        assert response.predictions == predictions
        assert response.anomaly_scores == anomaly_scores
        assert response.confidence_scores == confidence_scores
        assert response.uncertainty_scores == uncertainty_scores
        assert response.consensus_scores == consensus_scores
        assert response.individual_results == individual_results
        assert response.detector_weights == detector_weights
        assert response.voting_strategy_used == "dynamic_selection"
        assert response.ensemble_metrics == ensemble_metrics
        assert response.explanations == explanations
        assert response.processing_time == 5.2
        assert response.warnings == warnings

    def test_failed_response(self):
        """Test failed ensemble detection response."""
        response = EnsembleDetectionResponseDTO(
            success=False,
            error_message="Ensemble detection failed due to invalid input",
        )

        assert response.success is False
        assert (
            response.error_message == "Ensemble detection failed due to invalid input"
        )

    def test_response_with_individual_results(self):
        """Test response with individual detector results."""
        individual_results = {
            "isolation_forest": [0.1, 0.8, 0.3],
            "one_class_svm": [0.2, 0.9, 0.4],
            "local_outlier_factor": [0.15, 0.85, 0.35],
        }

        response = EnsembleDetectionResponseDTO(
            success=True, individual_results=individual_results
        )

        assert response.individual_results == individual_results
        assert len(response.individual_results) == 3


class TestEnsembleDetectorOptimizationRequestDTO:
    """Test suite for EnsembleDetectorOptimizationRequestDTO."""

    def test_basic_creation(self):
        """Test basic ensemble optimization request creation."""
        detector_ids = ["det1", "det2", "det3"]

        request = EnsembleDetectorOptimizationRequestDTO(
            detector_ids=detector_ids, validation_dataset_id="validation_dataset_123"
        )

        assert request.detector_ids == detector_ids
        assert request.validation_dataset_id == "validation_dataset_123"
        assert request.optimization_objective == "f1_score"  # Default
        assert request.target_voting_strategies == ["dynamic_selection"]  # Default
        assert request.max_ensemble_size == 5  # Default
        assert request.min_diversity_threshold == 0.3  # Default
        assert request.enable_pruning is True  # Default
        assert request.enable_weight_optimization is True  # Default
        assert request.cross_validation_folds == 5  # Default
        assert request.optimization_timeout == 300.0  # Default
        assert request.random_state == 42  # Default

    def test_complete_creation(self):
        """Test optimization request creation with all fields."""
        detector_ids = ["det1", "det2", "det3", "det4"]
        strategies = ["weighted_average", "consensus_voting"]

        request = EnsembleDetectorOptimizationRequestDTO(
            detector_ids=detector_ids,
            validation_dataset_id="validation_data",
            optimization_objective="auc_score",
            target_voting_strategies=strategies,
            max_ensemble_size=3,
            min_diversity_threshold=0.5,
            enable_pruning=False,
            enable_weight_optimization=False,
            cross_validation_folds=3,
            optimization_timeout=600.0,
            random_state=123,
        )

        assert request.optimization_objective == "auc_score"
        assert request.target_voting_strategies == strategies
        assert request.max_ensemble_size == 3
        assert request.min_diversity_threshold == 0.5
        assert request.enable_pruning is False
        assert request.enable_weight_optimization is False
        assert request.cross_validation_folds == 3
        assert request.optimization_timeout == 600.0
        assert request.random_state == 123

    def test_detector_ids_minimum_validation(self):
        """Test detector IDs minimum count validation."""
        # Test insufficient detectors
        with pytest.raises(ValidationError):
            EnsembleDetectorOptimizationRequestDTO(
                detector_ids=["single_detector"],
                validation_dataset_id="validation_data",
            )

        # Test valid minimum
        request = EnsembleDetectorOptimizationRequestDTO(
            detector_ids=["det1", "det2"], validation_dataset_id="validation_data"
        )
        assert len(request.detector_ids) == 2

    def test_optimization_objective_validation(self):
        """Test optimization objective validation."""
        detector_ids = ["det1", "det2"]

        # Test invalid objective
        with pytest.raises(ValidationError):
            EnsembleDetectorOptimizationRequestDTO(
                detector_ids=detector_ids,
                validation_dataset_id="validation_data",
                optimization_objective="invalid_objective",
            )

        # Test valid objectives
        valid_objectives = [
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "auc_score",
            "balanced_accuracy",
            "diversity",
            "stability",
            "efficiency",
        ]

        for objective in valid_objectives:
            request = EnsembleDetectorOptimizationRequestDTO(
                detector_ids=detector_ids,
                validation_dataset_id="validation_data",
                optimization_objective=objective,
            )
            assert request.optimization_objective == objective

    def test_target_voting_strategies_validation(self):
        """Test target voting strategies validation."""
        detector_ids = ["det1", "det2"]

        # Test invalid strategy
        with pytest.raises(ValidationError):
            EnsembleDetectorOptimizationRequestDTO(
                detector_ids=detector_ids,
                validation_dataset_id="validation_data",
                target_voting_strategies=["invalid_strategy"],
            )

        # Test valid strategies
        valid_strategies = [
            "simple_average",
            "weighted_average",
            "consensus_voting",
            "dynamic_selection",
        ]

        request = EnsembleDetectorOptimizationRequestDTO(
            detector_ids=detector_ids,
            validation_dataset_id="validation_data",
            target_voting_strategies=valid_strategies,
        )
        assert request.target_voting_strategies == valid_strategies

    def test_bounds_validation(self):
        """Test various parameter bounds validation."""
        detector_ids = ["det1", "det2"]

        # Test max_ensemble_size bounds
        with pytest.raises(ValidationError):
            EnsembleDetectorOptimizationRequestDTO(
                detector_ids=detector_ids,
                validation_dataset_id="validation_data",
                max_ensemble_size=1,  # Below minimum
            )

        with pytest.raises(ValidationError):
            EnsembleDetectorOptimizationRequestDTO(
                detector_ids=detector_ids,
                validation_dataset_id="validation_data",
                max_ensemble_size=11,  # Above maximum
            )

        # Test cross_validation_folds bounds
        with pytest.raises(ValidationError):
            EnsembleDetectorOptimizationRequestDTO(
                detector_ids=detector_ids,
                validation_dataset_id="validation_data",
                cross_validation_folds=1,  # Below minimum
            )

        # Test optimization_timeout bounds
        with pytest.raises(ValidationError):
            EnsembleDetectorOptimizationRequestDTO(
                detector_ids=detector_ids,
                validation_dataset_id="validation_data",
                optimization_timeout=0.0,  # Must be positive
            )


class TestUtilityFunctions:
    """Test suite for utility functions."""

    def test_create_default_ensemble_config(self):
        """Test default ensemble configuration creation."""
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
        """Test diversity metrics creation from dictionary."""
        data = {
            "disagreement_measure": 0.5,
            "double_fault_measure": 0.1,
            "q_statistic": 0.3,
            "correlation_coefficient": 0.2,
            "kappa_statistic": 0.4,
            "entropy_measure": 0.8,
            "overall_diversity": 0.6,
        }

        metrics = create_diversity_metrics_from_dict(data)

        assert isinstance(metrics, DiversityMetricsDTO)
        assert metrics.disagreement_measure == 0.5
        assert metrics.double_fault_measure == 0.1
        assert metrics.q_statistic == 0.3
        assert metrics.correlation_coefficient == 0.2
        assert metrics.kappa_statistic == 0.4
        assert metrics.entropy_measure == 0.8
        assert metrics.overall_diversity == 0.6

    def test_create_diversity_metrics_from_incomplete_dict(self):
        """Test diversity metrics creation from incomplete dictionary."""
        data = {"disagreement_measure": 0.5, "overall_diversity": 0.6}

        metrics = create_diversity_metrics_from_dict(data)

        assert metrics.disagreement_measure == 0.5
        assert metrics.overall_diversity == 0.6
        assert metrics.double_fault_measure == 0.0  # Default
        assert metrics.q_statistic == 0.0  # Default

    def test_create_ensemble_performance_from_dict(self):
        """Test ensemble performance creation from dictionary."""
        data = {
            "weighted_individual_performance": 0.85,
            "diversity_score": 0.7,
            "estimated_ensemble_performance": 0.92,
            "performance_improvement": 0.07,
            "confidence_score": 0.88,
        }

        performance = create_ensemble_performance_from_dict(data)

        assert isinstance(performance, EnsemblePerformanceDTO)
        assert performance.weighted_individual_performance == 0.85
        assert performance.diversity_score == 0.7
        assert performance.estimated_ensemble_performance == 0.92
        assert performance.performance_improvement == 0.07
        assert performance.confidence_score == 0.88

    def test_create_ensemble_performance_from_incomplete_dict(self):
        """Test ensemble performance creation from incomplete dictionary."""
        data = {"diversity_score": 0.7, "performance_improvement": 0.05}

        performance = create_ensemble_performance_from_dict(data)

        assert performance.diversity_score == 0.7
        assert performance.performance_improvement == 0.05
        assert performance.weighted_individual_performance == 0.0  # Default
        assert performance.estimated_ensemble_performance == 0.0  # Default


class TestEnsembleDTOIntegration:
    """Integration tests for ensemble DTOs."""

    def test_complete_ensemble_optimization_workflow(self):
        """Test complete ensemble optimization workflow."""
        # Step 1: Create optimization request
        detector_ids = [
            "isolation_forest",
            "one_class_svm",
            "local_outlier_factor",
            "elliptic_envelope",
        ]

        optimization_request = EnsembleDetectorOptimizationRequestDTO(
            detector_ids=detector_ids,
            validation_dataset_id="credit_card_fraud_validation",
            optimization_objective="f1_score",
            target_voting_strategies=["dynamic_selection", "weighted_average"],
            max_ensemble_size=3,
            min_diversity_threshold=0.4,
            enable_pruning=True,
            enable_weight_optimization=True,
            cross_validation_folds=5,
            optimization_timeout=600.0,
        )

        # Step 2: Create diversity metrics
        diversity_data = {
            "disagreement_measure": 0.6,
            "double_fault_measure": 0.08,
            "q_statistic": -0.15,
            "correlation_coefficient": 0.3,
            "kappa_statistic": 0.45,
            "entropy_measure": 0.85,
            "overall_diversity": 0.65,
        }
        diversity_metrics = create_diversity_metrics_from_dict(diversity_data)

        # Step 3: Create optimization response
        optimization_response = EnsembleDetectorOptimizationResponseDTO(
            success=True,
            optimized_detector_ids=[
                "isolation_forest",
                "local_outlier_factor",
                "one_class_svm",
            ],
            optimal_voting_strategy="dynamic_selection",
            optimal_weights=[0.4, 0.35, 0.25],
            ensemble_performance={"f1_score": 0.89, "precision": 0.91, "recall": 0.87},
            diversity_metrics={"overall_diversity": 0.65, "disagreement": 0.6},
            optimization_history=[
                {"iteration": 1, "f1_score": 0.82, "ensemble_size": 4},
                {"iteration": 2, "f1_score": 0.85, "ensemble_size": 3},
                {"iteration": 3, "f1_score": 0.89, "ensemble_size": 3},
            ],
            recommendations=[
                "Consider increasing diversity threshold",
                "Monitor for concept drift",
            ],
            optimization_time=487.3,
        )

        # Verify workflow consistency
        assert len(optimization_response.optimized_detector_ids) == 3
        assert optimization_response.optimal_voting_strategy == "dynamic_selection"
        assert len(optimization_response.optimal_weights) == 3
        assert optimization_response.ensemble_performance["f1_score"] == 0.89
        assert (
            optimization_response.optimization_time
            < optimization_request.optimization_timeout
        )

    def test_ensemble_detection_workflow(self):
        """Test ensemble detection workflow."""
        # Step 1: Create detection request
        detector_ids = [
            "optimized_detector_1",
            "optimized_detector_2",
            "optimized_detector_3",
        ]
        data = [[2.5, -1.0, 0.8, 1.2], [0.1, 0.3, -0.5, 2.1], [1.8, 0.9, 1.5, -0.3]]

        detection_request = EnsembleDetectionRequestDTO(
            detector_ids=detector_ids,
            data=data,
            voting_strategy="dynamic_selection",
            enable_dynamic_weighting=True,
            enable_uncertainty_estimation=True,
            enable_explanation=True,
            confidence_threshold=0.8,
            consensus_threshold=0.7,
            return_individual_results=True,
        )

        # Step 2: Create detection response
        detection_response = EnsembleDetectionResponseDTO(
            success=True,
            predictions=[0, 1, 0],
            anomaly_scores=[0.3, 0.85, 0.4],
            confidence_scores=[0.9, 0.92, 0.88],
            uncertainty_scores=[0.1, 0.08, 0.12],
            consensus_scores=[0.85, 0.95, 0.82],
            individual_results={
                "optimized_detector_1": [0.2, 0.8, 0.3],
                "optimized_detector_2": [0.35, 0.9, 0.45],
                "optimized_detector_3": [0.25, 0.85, 0.4],
            },
            detector_weights=[0.4, 0.35, 0.25],
            voting_strategy_used="dynamic_selection",
            ensemble_metrics={"diversity": 0.65, "agreement": 0.85, "stability": 0.9},
            explanations=[
                {"prediction": 0, "top_features": ["feature_1", "feature_3"]},
                {"prediction": 1, "top_features": ["feature_2", "feature_4"]},
                {"prediction": 0, "top_features": ["feature_1", "feature_2"]},
            ],
            processing_time=2.1,
        )

        # Verify detection workflow
        assert len(detection_response.predictions) == len(data)
        assert len(detection_response.anomaly_scores) == len(data)
        assert len(detection_response.confidence_scores) == len(data)
        assert len(detection_response.explanations) == len(data)
        assert len(detection_response.individual_results) == len(detector_ids)
        assert all(
            score >= detection_request.confidence_threshold
            for score in detection_response.confidence_scores
        )

    def test_meta_learning_knowledge_evolution(self):
        """Test meta-learning knowledge evolution."""
        # Initial knowledge
        initial_knowledge = MetaLearningKnowledgeDTO(
            dataset_characteristics={
                "n_samples": 1000,
                "n_features": 20,
                "contamination": 0.1,
            },
            algorithm_performance={"isolation_forest": 0.82, "one_class_svm": 0.78},
            ensemble_composition=["isolation_forest", "one_class_svm"],
            optimal_weights={"isolation_forest": 0.6, "one_class_svm": 0.4},
            diversity_requirements={"min_disagreement": 0.3},
            performance_metrics={"f1_score": 0.85, "precision": 0.88},
            confidence_score=0.75,
        )

        # Updated knowledge after more data
        updated_knowledge = MetaLearningKnowledgeDTO(
            dataset_characteristics={
                "n_samples": 2000,
                "n_features": 25,
                "contamination": 0.08,
            },
            algorithm_performance={
                "isolation_forest": 0.85,
                "one_class_svm": 0.81,
                "local_outlier_factor": 0.83,
            },
            ensemble_composition=[
                "isolation_forest",
                "local_outlier_factor",
                "one_class_svm",
            ],
            optimal_weights={
                "isolation_forest": 0.45,
                "local_outlier_factor": 0.35,
                "one_class_svm": 0.2,
            },
            diversity_requirements={"min_disagreement": 0.4, "max_correlation": 0.6},
            performance_metrics={"f1_score": 0.89, "precision": 0.92, "recall": 0.87},
            confidence_score=0.92,
        )

        # Verify knowledge evolution
        assert len(updated_knowledge.algorithm_performance) > len(
            initial_knowledge.algorithm_performance
        )
        assert len(updated_knowledge.ensemble_composition) > len(
            initial_knowledge.ensemble_composition
        )
        assert updated_knowledge.confidence_score > initial_knowledge.confidence_score
        assert (
            updated_knowledge.performance_metrics["f1_score"]
            > initial_knowledge.performance_metrics["f1_score"]
        )

    def test_ensemble_strategy_comparison(self):
        """Test comparison of different ensemble strategies."""
        strategies = [
            EnsembleStrategyDTO(
                name="simple_voting",
                description="Simple majority voting",
                requires_training=False,
                supports_weights=False,
                complexity="low",
                interpretability=0.9,
            ),
            EnsembleStrategyDTO(
                name="weighted_voting",
                description="Performance-weighted voting",
                requires_training=True,
                supports_weights=True,
                complexity="medium",
                interpretability=0.7,
            ),
            EnsembleStrategyDTO(
                name="meta_learning",
                description="Advanced meta-learning ensemble",
                requires_training=True,
                supports_weights=True,
                complexity="high",
                interpretability=0.4,
            ),
        ]

        # Verify strategy characteristics
        simple_strategy = strategies[0]
        weighted_strategy = strategies[1]
        meta_strategy = strategies[2]

        assert (
            simple_strategy.interpretability
            > weighted_strategy.interpretability
            > meta_strategy.interpretability
        )
        assert not simple_strategy.requires_training
        assert weighted_strategy.requires_training
        assert meta_strategy.requires_training
        assert not simple_strategy.supports_weights
        assert weighted_strategy.supports_weights
        assert meta_strategy.supports_weights


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
