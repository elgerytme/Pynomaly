"""Tests for Selection DTOs."""

from datetime import datetime
from uuid import uuid4

import pytest

from monorepo.application.dto.selection_dto import (
    AlgorithmBenchmarkDTO,
    AlgorithmComparisonDTO,
    AlgorithmPerformanceDTO,
    DatasetCharacteristicsDTO,
    LearningInsightsDTO,
    MetaLearningConfigDTO,
    OptimizationConstraintsDTO,
    PerformancePredictionDTO,
    SelectionExplanationDTO,
    SelectionHistoryDTO,
    SelectionRecommendationDTO,
    SelectionRequestDTO,
)


class TestDatasetCharacteristicsDTO:
    """Test suite for DatasetCharacteristicsDTO."""

    def test_valid_creation(self):
        """Test creating a valid dataset characteristics DTO."""
        dto = DatasetCharacteristicsDTO(
            n_samples=10000,
            n_features=50,
            n_numeric_features=40,
            n_categorical_features=10,
            feature_density=0.85,
            missing_value_ratio=0.05,
            outlier_ratio=0.02,
            mean_feature_correlation=0.3,
            feature_variance_ratio=2.5,
            data_dimensionality_ratio=0.005,
            skewness_mean=0.8,
            kurtosis_mean=2.5,
            class_imbalance=0.1,
            estimated_complexity="medium",
            data_type="tabular",
            domain="finance",
        )

        assert dto.n_samples == 10000
        assert dto.n_features == 50
        assert dto.n_numeric_features == 40
        assert dto.n_categorical_features == 10
        assert dto.feature_density == 0.85
        assert dto.missing_value_ratio == 0.05
        assert dto.outlier_ratio == 0.02
        assert dto.mean_feature_correlation == 0.3
        assert dto.feature_variance_ratio == 2.5
        assert dto.data_dimensionality_ratio == 0.005
        assert dto.skewness_mean == 0.8
        assert dto.kurtosis_mean == 2.5
        assert dto.class_imbalance == 0.1
        assert dto.estimated_complexity == "medium"
        assert dto.data_type == "tabular"
        assert dto.domain == "finance"

    def test_minimum_values(self):
        """Test minimum boundary values."""
        dto = DatasetCharacteristicsDTO(
            n_samples=0,
            n_features=0,
            n_numeric_features=0,
            n_categorical_features=0,
            feature_density=0.0,
            missing_value_ratio=0.0,
            outlier_ratio=0.0,
            mean_feature_correlation=0.0,
            feature_variance_ratio=0.0,
            data_dimensionality_ratio=0.0,
            skewness_mean=-2.0,
            kurtosis_mean=-1.0,
            class_imbalance=0.0,
        )

        assert dto.n_samples == 0
        assert dto.n_features == 0
        assert dto.n_numeric_features == 0
        assert dto.n_categorical_features == 0
        assert dto.feature_density == 0.0
        assert dto.missing_value_ratio == 0.0
        assert dto.outlier_ratio == 0.0
        assert dto.mean_feature_correlation == 0.0
        assert dto.feature_variance_ratio == 0.0
        assert dto.data_dimensionality_ratio == 0.0
        assert dto.skewness_mean == -2.0
        assert dto.kurtosis_mean == -1.0
        assert dto.class_imbalance == 0.0

    def test_maximum_values(self):
        """Test maximum boundary values."""
        dto = DatasetCharacteristicsDTO(
            n_samples=1000000,
            n_features=10000,
            n_numeric_features=8000,
            n_categorical_features=2000,
            feature_density=1.0,
            missing_value_ratio=1.0,
            outlier_ratio=1.0,
            mean_feature_correlation=1.0,
            feature_variance_ratio=100.0,
            data_dimensionality_ratio=10.0,
            skewness_mean=5.0,
            kurtosis_mean=20.0,
            class_imbalance=1.0,
        )

        assert dto.n_samples == 1000000
        assert dto.n_features == 10000
        assert dto.feature_density == 1.0
        assert dto.missing_value_ratio == 1.0
        assert dto.outlier_ratio == 1.0
        assert dto.mean_feature_correlation == 1.0
        assert dto.class_imbalance == 1.0

    def test_invalid_negative_samples(self):
        """Test validation for negative samples."""
        with pytest.raises(ValueError):
            DatasetCharacteristicsDTO(
                n_samples=-1,
                n_features=10,
                n_numeric_features=8,
                n_categorical_features=2,
                feature_density=0.8,
                missing_value_ratio=0.1,
                outlier_ratio=0.05,
                mean_feature_correlation=0.3,
                feature_variance_ratio=2.0,
                data_dimensionality_ratio=0.01,
                skewness_mean=0.5,
                kurtosis_mean=2.0,
                class_imbalance=0.2,
            )

    def test_invalid_feature_density(self):
        """Test validation for feature density out of range."""
        with pytest.raises(ValueError):
            DatasetCharacteristicsDTO(
                n_samples=1000,
                n_features=10,
                n_numeric_features=8,
                n_categorical_features=2,
                feature_density=1.5,
                missing_value_ratio=0.1,
                outlier_ratio=0.05,
                mean_feature_correlation=0.3,
                feature_variance_ratio=2.0,
                data_dimensionality_ratio=0.01,
                skewness_mean=0.5,
                kurtosis_mean=2.0,
                class_imbalance=0.2,
            )

    def test_high_dimensional_data(self):
        """Test high-dimensional data characteristics."""
        dto = DatasetCharacteristicsDTO(
            n_samples=100,
            n_features=5000,
            n_numeric_features=4800,
            n_categorical_features=200,
            feature_density=0.1,
            missing_value_ratio=0.0,
            outlier_ratio=0.01,
            mean_feature_correlation=0.05,
            feature_variance_ratio=50.0,
            data_dimensionality_ratio=50.0,
            skewness_mean=1.5,
            kurtosis_mean=5.0,
            class_imbalance=0.5,
            estimated_complexity="high",
            data_type="genomic",
        )

        assert dto.n_samples == 100
        assert dto.n_features == 5000
        assert dto.data_dimensionality_ratio == 50.0
        assert dto.estimated_complexity == "high"
        assert dto.data_type == "genomic"

    def test_optional_fields_none(self):
        """Test optional fields being None."""
        dto = DatasetCharacteristicsDTO(
            n_samples=1000,
            n_features=10,
            n_numeric_features=8,
            n_categorical_features=2,
            feature_density=0.8,
            missing_value_ratio=0.1,
            outlier_ratio=0.05,
            mean_feature_correlation=0.3,
            feature_variance_ratio=2.0,
            data_dimensionality_ratio=0.01,
            skewness_mean=0.5,
            kurtosis_mean=2.0,
            class_imbalance=0.2,
            estimated_complexity=None,
            data_type=None,
            domain=None,
        )

        assert dto.estimated_complexity is None
        assert dto.data_type is None
        assert dto.domain is None


class TestAlgorithmPerformanceDTO:
    """Test suite for AlgorithmPerformanceDTO."""

    def test_valid_creation(self):
        """Test creating a valid algorithm performance DTO."""
        dto = AlgorithmPerformanceDTO(
            primary_metric=0.85,
            secondary_metrics={"precision": 0.82, "recall": 0.88, "f1": 0.85},
            training_time_seconds=150.5,
            memory_usage_mb=512.0,
            prediction_time_ms=2.5,
            stability_score=0.92,
            interpretability_score=0.7,
            confidence_score=0.88,
            cv_mean=0.84,
            cv_std=0.03,
            cv_scores=[0.82, 0.85, 0.86, 0.83, 0.84],
        )

        assert dto.primary_metric == 0.85
        assert dto.secondary_metrics == {"precision": 0.82, "recall": 0.88, "f1": 0.85}
        assert dto.training_time_seconds == 150.5
        assert dto.memory_usage_mb == 512.0
        assert dto.prediction_time_ms == 2.5
        assert dto.stability_score == 0.92
        assert dto.interpretability_score == 0.7
        assert dto.confidence_score == 0.88
        assert dto.cv_mean == 0.84
        assert dto.cv_std == 0.03
        assert dto.cv_scores == [0.82, 0.85, 0.86, 0.83, 0.84]

    def test_minimum_values(self):
        """Test minimum boundary values."""
        dto = AlgorithmPerformanceDTO(
            primary_metric=0.0,
            training_time_seconds=0.0,
            memory_usage_mb=0.0,
            prediction_time_ms=0.0,
            stability_score=0.0,
            interpretability_score=0.0,
            confidence_score=0.0,
        )

        assert dto.primary_metric == 0.0
        assert dto.training_time_seconds == 0.0
        assert dto.memory_usage_mb == 0.0
        assert dto.prediction_time_ms == 0.0
        assert dto.stability_score == 0.0
        assert dto.interpretability_score == 0.0
        assert dto.confidence_score == 0.0

    def test_maximum_values(self):
        """Test maximum boundary values."""
        dto = AlgorithmPerformanceDTO(
            primary_metric=1.0,
            training_time_seconds=86400.0,
            memory_usage_mb=16384.0,
            prediction_time_ms=1000.0,
            stability_score=1.0,
            interpretability_score=1.0,
            confidence_score=1.0,
        )

        assert dto.primary_metric == 1.0
        assert dto.training_time_seconds == 86400.0
        assert dto.memory_usage_mb == 16384.0
        assert dto.prediction_time_ms == 1000.0
        assert dto.stability_score == 1.0
        assert dto.interpretability_score == 1.0
        assert dto.confidence_score == 1.0

    def test_invalid_primary_metric(self):
        """Test validation for primary metric out of range."""
        with pytest.raises(ValueError):
            AlgorithmPerformanceDTO(
                primary_metric=1.5, training_time_seconds=100.0, memory_usage_mb=256.0
            )

    def test_invalid_negative_time(self):
        """Test validation for negative training time."""
        with pytest.raises(ValueError):
            AlgorithmPerformanceDTO(
                primary_metric=0.8, training_time_seconds=-10.0, memory_usage_mb=256.0
            )

    def test_default_values(self):
        """Test default values."""
        dto = AlgorithmPerformanceDTO(
            primary_metric=0.8, training_time_seconds=100.0, memory_usage_mb=256.0
        )

        assert dto.prediction_time_ms == 0.0
        assert dto.stability_score == 0.0
        assert dto.interpretability_score == 0.0
        assert dto.confidence_score == 0.0
        assert dto.cv_mean is None
        assert dto.cv_std is None
        assert dto.cv_scores is None

    def test_empty_secondary_metrics(self):
        """Test empty secondary metrics."""
        dto = AlgorithmPerformanceDTO(
            primary_metric=0.8,
            secondary_metrics={},
            training_time_seconds=100.0,
            memory_usage_mb=256.0,
        )

        assert dto.secondary_metrics == {}

    def test_extensive_secondary_metrics(self):
        """Test extensive secondary metrics."""
        secondary_metrics = {
            "accuracy": 0.85,
            "precision": 0.82,
            "recall": 0.88,
            "f1": 0.85,
            "auc": 0.90,
            "mcc": 0.70,
            "specificity": 0.83,
            "sensitivity": 0.88,
            "npv": 0.91,
            "ppv": 0.82,
        }

        dto = AlgorithmPerformanceDTO(
            primary_metric=0.85,
            secondary_metrics=secondary_metrics,
            training_time_seconds=200.0,
            memory_usage_mb=512.0,
        )

        assert len(dto.secondary_metrics) == 10
        assert dto.secondary_metrics["accuracy"] == 0.85
        assert dto.secondary_metrics["auc"] == 0.90
        assert dto.secondary_metrics["mcc"] == 0.70


class TestOptimizationConstraintsDTO:
    """Test suite for OptimizationConstraintsDTO."""

    def test_valid_creation(self):
        """Test creating a valid optimization constraints DTO."""
        dto = OptimizationConstraintsDTO(
            max_training_time_seconds=3600.0,
            max_memory_mb=8192.0,
            max_prediction_time_ms=100.0,
            min_accuracy=0.8,
            min_interpretability=0.6,
            min_stability=0.7,
            require_online_prediction=True,
            require_batch_prediction=True,
            require_interpretability=True,
            available_libraries=["scikit-learn", "tensorflow", "pytorch"],
            gpu_available=True,
            distributed_computing=True,
        )

        assert dto.max_training_time_seconds == 3600.0
        assert dto.max_memory_mb == 8192.0
        assert dto.max_prediction_time_ms == 100.0
        assert dto.min_accuracy == 0.8
        assert dto.min_interpretability == 0.6
        assert dto.min_stability == 0.7
        assert dto.require_online_prediction is True
        assert dto.require_batch_prediction is True
        assert dto.require_interpretability is True
        assert dto.available_libraries == ["scikit-learn", "tensorflow", "pytorch"]
        assert dto.gpu_available is True
        assert dto.distributed_computing is True

    def test_default_values(self):
        """Test default values."""
        dto = OptimizationConstraintsDTO()

        assert dto.max_training_time_seconds is None
        assert dto.max_memory_mb is None
        assert dto.max_prediction_time_ms is None
        assert dto.min_accuracy is None
        assert dto.min_interpretability is None
        assert dto.min_stability is None
        assert dto.require_online_prediction is False
        assert dto.require_batch_prediction is True
        assert dto.require_interpretability is False
        assert dto.available_libraries is None
        assert dto.gpu_available is False
        assert dto.distributed_computing is False

    def test_invalid_negative_time(self):
        """Test validation for negative time constraint."""
        with pytest.raises(ValueError):
            OptimizationConstraintsDTO(max_training_time_seconds=-100.0)

    def test_invalid_accuracy_range(self):
        """Test validation for accuracy out of range."""
        with pytest.raises(ValueError):
            OptimizationConstraintsDTO(min_accuracy=1.5)

    def test_strict_constraints(self):
        """Test strict constraints configuration."""
        dto = OptimizationConstraintsDTO(
            max_training_time_seconds=60.0,
            max_memory_mb=512.0,
            max_prediction_time_ms=10.0,
            min_accuracy=0.95,
            min_interpretability=0.9,
            min_stability=0.95,
            require_online_prediction=True,
            require_batch_prediction=False,
            require_interpretability=True,
        )

        assert dto.max_training_time_seconds == 60.0
        assert dto.max_memory_mb == 512.0
        assert dto.max_prediction_time_ms == 10.0
        assert dto.min_accuracy == 0.95
        assert dto.min_interpretability == 0.9
        assert dto.min_stability == 0.95
        assert dto.require_online_prediction is True
        assert dto.require_batch_prediction is False
        assert dto.require_interpretability is True

    def test_relaxed_constraints(self):
        """Test relaxed constraints configuration."""
        dto = OptimizationConstraintsDTO(
            max_training_time_seconds=86400.0,
            max_memory_mb=32768.0,
            max_prediction_time_ms=1000.0,
            min_accuracy=0.5,
            min_interpretability=0.0,
            min_stability=0.0,
            require_online_prediction=False,
            require_batch_prediction=True,
            require_interpretability=False,
        )

        assert dto.max_training_time_seconds == 86400.0
        assert dto.max_memory_mb == 32768.0
        assert dto.max_prediction_time_ms == 1000.0
        assert dto.min_accuracy == 0.5
        assert dto.min_interpretability == 0.0
        assert dto.min_stability == 0.0

    def test_extensive_library_list(self):
        """Test extensive library list."""
        libraries = [
            "scikit-learn",
            "tensorflow",
            "pytorch",
            "xgboost",
            "lightgbm",
            "catboost",
            "h2o",
            "auto-sklearn",
            "optuna",
            "hyperopt",
        ]

        dto = OptimizationConstraintsDTO(
            available_libraries=libraries,
            gpu_available=True,
            distributed_computing=True,
        )

        assert len(dto.available_libraries) == 10
        assert "scikit-learn" in dto.available_libraries
        assert "tensorflow" in dto.available_libraries
        assert "pytorch" in dto.available_libraries
        assert "xgboost" in dto.available_libraries


class TestMetaLearningConfigDTO:
    """Test suite for MetaLearningConfigDTO."""

    def test_valid_creation(self):
        """Test creating a valid meta-learning config DTO."""
        dto = MetaLearningConfigDTO(
            enable_transfer_learning=True,
            similarity_threshold=0.8,
            min_historical_samples=10,
            meta_model_type="gradient_boosting",
            cross_validation_folds=10,
            learning_rate=0.001,
            forgetting_factor=0.9,
            ensemble_meta_models=True,
            adaptive_weighting=True,
        )

        assert dto.enable_transfer_learning is True
        assert dto.similarity_threshold == 0.8
        assert dto.min_historical_samples == 10
        assert dto.meta_model_type == "gradient_boosting"
        assert dto.cross_validation_folds == 10
        assert dto.learning_rate == 0.001
        assert dto.forgetting_factor == 0.9
        assert dto.ensemble_meta_models is True
        assert dto.adaptive_weighting is True

    def test_default_values(self):
        """Test default values."""
        dto = MetaLearningConfigDTO()

        assert dto.enable_transfer_learning is True
        assert dto.similarity_threshold == 0.7
        assert dto.min_historical_samples == 5
        assert dto.meta_model_type == "random_forest"
        assert dto.cross_validation_folds == 5
        assert dto.learning_rate == 0.01
        assert dto.forgetting_factor == 0.95
        assert dto.ensemble_meta_models is False
        assert dto.adaptive_weighting is True

    def test_invalid_similarity_threshold(self):
        """Test validation for similarity threshold out of range."""
        with pytest.raises(ValueError):
            MetaLearningConfigDTO(similarity_threshold=1.5)

    def test_invalid_cv_folds(self):
        """Test validation for CV folds out of range."""
        with pytest.raises(ValueError):
            MetaLearningConfigDTO(cross_validation_folds=1)

    def test_invalid_learning_rate(self):
        """Test validation for learning rate out of range."""
        with pytest.raises(ValueError):
            MetaLearningConfigDTO(learning_rate=0.0)

    def test_conservative_config(self):
        """Test conservative configuration."""
        dto = MetaLearningConfigDTO(
            enable_transfer_learning=False,
            similarity_threshold=0.9,
            min_historical_samples=20,
            meta_model_type="linear_regression",
            cross_validation_folds=3,
            learning_rate=0.001,
            forgetting_factor=1.0,
            ensemble_meta_models=False,
            adaptive_weighting=False,
        )

        assert dto.enable_transfer_learning is False
        assert dto.similarity_threshold == 0.9
        assert dto.min_historical_samples == 20
        assert dto.meta_model_type == "linear_regression"
        assert dto.cross_validation_folds == 3
        assert dto.learning_rate == 0.001
        assert dto.forgetting_factor == 1.0
        assert dto.ensemble_meta_models is False
        assert dto.adaptive_weighting is False

    def test_aggressive_config(self):
        """Test aggressive configuration."""
        dto = MetaLearningConfigDTO(
            enable_transfer_learning=True,
            similarity_threshold=0.5,
            min_historical_samples=1,
            meta_model_type="neural_network",
            cross_validation_folds=10,
            learning_rate=0.1,
            forgetting_factor=0.8,
            ensemble_meta_models=True,
            adaptive_weighting=True,
        )

        assert dto.enable_transfer_learning is True
        assert dto.similarity_threshold == 0.5
        assert dto.min_historical_samples == 1
        assert dto.meta_model_type == "neural_network"
        assert dto.cross_validation_folds == 10
        assert dto.learning_rate == 0.1
        assert dto.forgetting_factor == 0.8
        assert dto.ensemble_meta_models is True
        assert dto.adaptive_weighting is True


class TestSelectionRecommendationDTO:
    """Test suite for SelectionRecommendationDTO."""

    def test_valid_creation(self):
        """Test creating a valid selection recommendation DTO."""
        dataset_chars = DatasetCharacteristicsDTO(
            n_samples=1000,
            n_features=10,
            n_numeric_features=8,
            n_categorical_features=2,
            feature_density=0.8,
            missing_value_ratio=0.1,
            outlier_ratio=0.05,
            mean_feature_correlation=0.3,
            feature_variance_ratio=2.0,
            data_dimensionality_ratio=0.01,
            skewness_mean=0.5,
            kurtosis_mean=2.0,
            class_imbalance=0.2,
        )

        timestamp = datetime(2023, 1, 15, 10, 30, 0)
        recommendation_id = uuid4()

        dto = SelectionRecommendationDTO(
            recommended_algorithms=["random_forest", "gradient_boosting", "svm"],
            confidence_scores={
                "random_forest": 0.9,
                "gradient_boosting": 0.8,
                "svm": 0.7,
            },
            reasoning=[
                "High performance on similar datasets",
                "Good interpretability",
                "Robust to outliers",
            ],
            dataset_characteristics=dataset_chars,
            selection_context={"experiment_id": "exp_123", "user_id": "user_456"},
            timestamp=timestamp,
            recommendation_id=recommendation_id,
            predicted_performances={
                "random_forest": 0.85,
                "gradient_boosting": 0.82,
                "svm": 0.78,
            },
            uncertainty_estimates={
                "random_forest": 0.05,
                "gradient_boosting": 0.08,
                "svm": 0.12,
            },
        )

        assert dto.recommended_algorithms == [
            "random_forest",
            "gradient_boosting",
            "svm",
        ]
        assert dto.confidence_scores == {
            "random_forest": 0.9,
            "gradient_boosting": 0.8,
            "svm": 0.7,
        }
        assert len(dto.reasoning) == 3
        assert dto.dataset_characteristics == dataset_chars
        assert dto.selection_context == {
            "experiment_id": "exp_123",
            "user_id": "user_456",
        }
        assert dto.timestamp == timestamp
        assert dto.recommendation_id == recommendation_id
        assert dto.predicted_performances == {
            "random_forest": 0.85,
            "gradient_boosting": 0.82,
            "svm": 0.78,
        }
        assert dto.uncertainty_estimates == {
            "random_forest": 0.05,
            "gradient_boosting": 0.08,
            "svm": 0.12,
        }

    def test_minimal_creation(self):
        """Test creating with minimal required fields."""
        dataset_chars = DatasetCharacteristicsDTO(
            n_samples=100,
            n_features=5,
            n_numeric_features=5,
            n_categorical_features=0,
            feature_density=1.0,
            missing_value_ratio=0.0,
            outlier_ratio=0.0,
            mean_feature_correlation=0.2,
            feature_variance_ratio=1.0,
            data_dimensionality_ratio=0.05,
            skewness_mean=0.0,
            kurtosis_mean=0.0,
            class_imbalance=0.0,
        )

        timestamp = datetime.now()

        dto = SelectionRecommendationDTO(
            recommended_algorithms=["linear_regression"],
            confidence_scores={"linear_regression": 0.8},
            dataset_characteristics=dataset_chars,
            timestamp=timestamp,
        )

        assert dto.recommended_algorithms == ["linear_regression"]
        assert dto.confidence_scores == {"linear_regression": 0.8}
        assert dto.reasoning == []
        assert dto.selection_context == {}
        assert dto.recommendation_id is None
        assert dto.predicted_performances is None
        assert dto.uncertainty_estimates is None

    def test_single_algorithm_recommendation(self):
        """Test single algorithm recommendation."""
        dataset_chars = DatasetCharacteristicsDTO(
            n_samples=500,
            n_features=3,
            n_numeric_features=3,
            n_categorical_features=0,
            feature_density=1.0,
            missing_value_ratio=0.0,
            outlier_ratio=0.0,
            mean_feature_correlation=0.1,
            feature_variance_ratio=1.2,
            data_dimensionality_ratio=0.006,
            skewness_mean=0.2,
            kurtosis_mean=1.5,
            class_imbalance=0.1,
        )

        dto = SelectionRecommendationDTO(
            recommended_algorithms=["logistic_regression"],
            confidence_scores={"logistic_regression": 0.95},
            reasoning=[
                "Perfect fit for binary classification",
                "Linear separability detected",
            ],
            dataset_characteristics=dataset_chars,
            timestamp=datetime.now(),
        )

        assert len(dto.recommended_algorithms) == 1
        assert dto.recommended_algorithms[0] == "logistic_regression"
        assert dto.confidence_scores["logistic_regression"] == 0.95
        assert len(dto.reasoning) == 2

    def test_multiple_algorithms_recommendation(self):
        """Test multiple algorithms recommendation."""
        dataset_chars = DatasetCharacteristicsDTO(
            n_samples=10000,
            n_features=100,
            n_numeric_features=90,
            n_categorical_features=10,
            feature_density=0.7,
            missing_value_ratio=0.05,
            outlier_ratio=0.03,
            mean_feature_correlation=0.4,
            feature_variance_ratio=5.0,
            data_dimensionality_ratio=0.01,
            skewness_mean=1.0,
            kurtosis_mean=3.0,
            class_imbalance=0.3,
        )

        algorithms = [
            "random_forest",
            "xgboost",
            "lightgbm",
            "catboost",
            "neural_network",
        ]
        confidence_scores = {
            "random_forest": 0.9,
            "xgboost": 0.88,
            "lightgbm": 0.85,
            "catboost": 0.82,
            "neural_network": 0.75,
        }

        dto = SelectionRecommendationDTO(
            recommended_algorithms=algorithms,
            confidence_scores=confidence_scores,
            reasoning=[
                "Complex dataset requires ensemble methods",
                "High performance expected",
            ],
            dataset_characteristics=dataset_chars,
            timestamp=datetime.now(),
        )

        assert len(dto.recommended_algorithms) == 5
        assert len(dto.confidence_scores) == 5
        assert "random_forest" in dto.recommended_algorithms
        assert "neural_network" in dto.recommended_algorithms
        assert dto.confidence_scores["random_forest"] == 0.9
        assert dto.confidence_scores["neural_network"] == 0.75

    def test_comprehensive_context(self):
        """Test comprehensive selection context."""
        dataset_chars = DatasetCharacteristicsDTO(
            n_samples=2000,
            n_features=25,
            n_numeric_features=20,
            n_categorical_features=5,
            feature_density=0.9,
            missing_value_ratio=0.02,
            outlier_ratio=0.01,
            mean_feature_correlation=0.25,
            feature_variance_ratio=3.0,
            data_dimensionality_ratio=0.0125,
            skewness_mean=0.6,
            kurtosis_mean=2.2,
            class_imbalance=0.15,
        )

        context = {
            "experiment_id": "exp_789",
            "user_id": "user_123",
            "project_name": "fraud_detection",
            "deadline": "2023-02-01",
            "budget": "medium",
            "interpretability_required": True,
            "production_deployment": True,
            "real_time_scoring": False,
        }

        dto = SelectionRecommendationDTO(
            recommended_algorithms=["random_forest", "logistic_regression"],
            confidence_scores={"random_forest": 0.85, "logistic_regression": 0.8},
            reasoning=[
                "Balanced performance and interpretability",
                "Suitable for production",
            ],
            dataset_characteristics=dataset_chars,
            selection_context=context,
            timestamp=datetime.now(),
        )

        assert len(dto.selection_context) == 8
        assert dto.selection_context["experiment_id"] == "exp_789"
        assert dto.selection_context["interpretability_required"] is True
        assert dto.selection_context["real_time_scoring"] is False


class TestAlgorithmBenchmarkDTO:
    """Test suite for AlgorithmBenchmarkDTO."""

    def test_valid_creation(self):
        """Test creating a valid algorithm benchmark DTO."""
        benchmark_time = datetime(2023, 1, 15, 12, 0, 0)

        dto = AlgorithmBenchmarkDTO(
            algorithm_name="random_forest",
            algorithm_version="1.0.0",
            mean_score=0.85,
            std_score=0.03,
            cv_scores=[0.82, 0.85, 0.87, 0.84, 0.86],
            training_time_seconds=120.5,
            memory_usage_mb=512.0,
            prediction_time_ms=2.1,
            hyperparameters={
                "n_estimators": 100,
                "max_depth": 10,
                "min_samples_split": 2,
                "random_state": 42,
            },
            additional_metrics={"precision": 0.83, "recall": 0.87, "f1": 0.85},
            benchmark_timestamp=benchmark_time,
            environment_info={
                "python_version": "3.9.0",
                "sklearn_version": "1.0.0",
                "cpu_count": 8,
                "memory_gb": 16,
            },
        )

        assert dto.algorithm_name == "random_forest"
        assert dto.algorithm_version == "1.0.0"
        assert dto.mean_score == 0.85
        assert dto.std_score == 0.03
        assert dto.cv_scores == [0.82, 0.85, 0.87, 0.84, 0.86]
        assert dto.training_time_seconds == 120.5
        assert dto.memory_usage_mb == 512.0
        assert dto.prediction_time_ms == 2.1
        assert dto.hyperparameters["n_estimators"] == 100
        assert dto.additional_metrics["precision"] == 0.83
        assert dto.benchmark_timestamp == benchmark_time
        assert dto.environment_info["python_version"] == "3.9.0"

    def test_minimal_creation(self):
        """Test creating with minimal required fields."""
        dto = AlgorithmBenchmarkDTO(
            algorithm_name="svm",
            mean_score=0.75,
            std_score=0.05,
            cv_scores=[0.72, 0.74, 0.76, 0.75, 0.73],
            training_time_seconds=45.2,
            memory_usage_mb=256.0,
        )

        assert dto.algorithm_name == "svm"
        assert dto.algorithm_version is None
        assert dto.mean_score == 0.75
        assert dto.std_score == 0.05
        assert len(dto.cv_scores) == 5
        assert dto.training_time_seconds == 45.2
        assert dto.memory_usage_mb == 256.0
        assert dto.prediction_time_ms == 0.0
        assert dto.hyperparameters == {}
        assert dto.additional_metrics == {}
        assert dto.environment_info is None

    def test_invalid_negative_score(self):
        """Test validation for negative mean score."""
        with pytest.raises(ValueError):
            AlgorithmBenchmarkDTO(
                algorithm_name="test_algo",
                mean_score=-0.1,
                std_score=0.05,
                cv_scores=[0.1, 0.2, 0.3],
                training_time_seconds=100.0,
                memory_usage_mb=256.0,
            )

    def test_invalid_score_above_one(self):
        """Test validation for score above 1."""
        with pytest.raises(ValueError):
            AlgorithmBenchmarkDTO(
                algorithm_name="test_algo",
                mean_score=1.5,
                std_score=0.05,
                cv_scores=[0.1, 0.2, 0.3],
                training_time_seconds=100.0,
                memory_usage_mb=256.0,
            )

    def test_neural_network_benchmark(self):
        """Test neural network benchmark with extensive hyperparameters."""
        hyperparams = {
            "hidden_layers": [128, 64, 32],
            "activation": "relu",
            "optimizer": "adam",
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 100,
            "dropout_rate": 0.2,
            "l2_regularization": 0.01,
        }

        dto = AlgorithmBenchmarkDTO(
            algorithm_name="neural_network",
            algorithm_version="2.0.0",
            mean_score=0.88,
            std_score=0.02,
            cv_scores=[0.86, 0.88, 0.90, 0.87, 0.89],
            training_time_seconds=1800.0,
            memory_usage_mb=2048.0,
            prediction_time_ms=5.2,
            hyperparameters=hyperparams,
            additional_metrics={
                "train_loss": 0.15,
                "val_loss": 0.22,
                "train_accuracy": 0.92,
                "val_accuracy": 0.88,
            },
        )

        assert dto.algorithm_name == "neural_network"
        assert dto.mean_score == 0.88
        assert dto.training_time_seconds == 1800.0
        assert dto.hyperparameters["hidden_layers"] == [128, 64, 32]
        assert dto.hyperparameters["learning_rate"] == 0.001
        assert dto.additional_metrics["train_loss"] == 0.15
        assert dto.additional_metrics["val_accuracy"] == 0.88

    def test_high_variance_benchmark(self):
        """Test benchmark with high variance in CV scores."""
        dto = AlgorithmBenchmarkDTO(
            algorithm_name="unstable_algo",
            mean_score=0.7,
            std_score=0.15,
            cv_scores=[0.5, 0.6, 0.8, 0.9, 0.6],
            training_time_seconds=200.0,
            memory_usage_mb=1024.0,
        )

        assert dto.mean_score == 0.7
        assert dto.std_score == 0.15
        assert min(dto.cv_scores) == 0.5
        assert max(dto.cv_scores) == 0.9

    def test_comprehensive_environment_info(self):
        """Test comprehensive environment information."""
        env_info = {
            "python_version": "3.10.0",
            "sklearn_version": "1.1.0",
            "numpy_version": "1.21.0",
            "pandas_version": "1.3.0",
            "cpu_count": 16,
            "memory_gb": 32,
            "gpu_available": True,
            "gpu_type": "NVIDIA RTX 3080",
            "os": "Ubuntu 20.04",
            "benchmark_date": "2023-01-15",
        }

        dto = AlgorithmBenchmarkDTO(
            algorithm_name="xgboost",
            mean_score=0.82,
            std_score=0.04,
            cv_scores=[0.78, 0.81, 0.85, 0.83, 0.79],
            training_time_seconds=300.0,
            memory_usage_mb=1536.0,
            environment_info=env_info,
        )

        assert len(dto.environment_info) == 10
        assert dto.environment_info["python_version"] == "3.10.0"
        assert dto.environment_info["gpu_available"] is True
        assert dto.environment_info["gpu_type"] == "NVIDIA RTX 3080"


class TestSelectionHistoryDTO:
    """Test suite for SelectionHistoryDTO."""

    def test_valid_creation(self):
        """Test creating a valid selection history DTO."""
        dataset_chars = DatasetCharacteristicsDTO(
            n_samples=1000,
            n_features=10,
            n_numeric_features=8,
            n_categorical_features=2,
            feature_density=0.8,
            missing_value_ratio=0.1,
            outlier_ratio=0.05,
            mean_feature_correlation=0.3,
            feature_variance_ratio=2.0,
            data_dimensionality_ratio=0.01,
            skewness_mean=0.5,
            kurtosis_mean=2.0,
            class_imbalance=0.2,
        )

        performance = AlgorithmPerformanceDTO(
            primary_metric=0.85,
            training_time_seconds=120.0,
            memory_usage_mb=512.0,
            stability_score=0.9,
            interpretability_score=0.8,
            confidence_score=0.85,
        )

        constraints = OptimizationConstraintsDTO(
            max_training_time_seconds=300.0,
            min_accuracy=0.8,
            require_interpretability=True,
        )

        timestamp = datetime(2023, 1, 15, 10, 0, 0)

        dto = SelectionHistoryDTO(
            dataset_characteristics=dataset_chars,
            selected_algorithm="random_forest",
            performance=performance,
            selection_context={"experiment_id": "exp_123", "user_id": "user_456"},
            constraints_used=constraints,
            timestamp=timestamp,
            dataset_hash="abc123def456",
            user_feedback={"satisfaction": 8, "would_use_again": True},
            was_successful=True,
            lessons_learned=[
                "Feature engineering improved performance",
                "Hyperparameter tuning was crucial",
            ],
        )

        assert dto.dataset_characteristics == dataset_chars
        assert dto.selected_algorithm == "random_forest"
        assert dto.performance == performance
        assert dto.selection_context == {
            "experiment_id": "exp_123",
            "user_id": "user_456",
        }
        assert dto.constraints_used == constraints
        assert dto.timestamp == timestamp
        assert dto.dataset_hash == "abc123def456"
        assert dto.user_feedback == {"satisfaction": 8, "would_use_again": True}
        assert dto.was_successful is True
        assert len(dto.lessons_learned) == 2

    def test_minimal_creation(self):
        """Test creating with minimal required fields."""
        dataset_chars = DatasetCharacteristicsDTO(
            n_samples=100,
            n_features=5,
            n_numeric_features=5,
            n_categorical_features=0,
            feature_density=1.0,
            missing_value_ratio=0.0,
            outlier_ratio=0.0,
            mean_feature_correlation=0.2,
            feature_variance_ratio=1.0,
            data_dimensionality_ratio=0.05,
            skewness_mean=0.0,
            kurtosis_mean=0.0,
            class_imbalance=0.0,
        )

        performance = AlgorithmPerformanceDTO(
            primary_metric=0.7, training_time_seconds=30.0, memory_usage_mb=128.0
        )

        dto = SelectionHistoryDTO(
            dataset_characteristics=dataset_chars,
            selected_algorithm="linear_regression",
            performance=performance,
            timestamp=datetime.now(),
            dataset_hash="simple_hash",
        )

        assert dto.dataset_characteristics == dataset_chars
        assert dto.selected_algorithm == "linear_regression"
        assert dto.performance == performance
        assert dto.selection_context == {}
        assert dto.constraints_used is None
        assert dto.user_feedback is None
        assert dto.was_successful is True
        assert dto.lessons_learned is None

    def test_failed_selection(self):
        """Test failed selection history."""
        dataset_chars = DatasetCharacteristicsDTO(
            n_samples=50,
            n_features=100,
            n_numeric_features=100,
            n_categorical_features=0,
            feature_density=0.1,
            missing_value_ratio=0.5,
            outlier_ratio=0.2,
            mean_feature_correlation=0.9,
            feature_variance_ratio=100.0,
            data_dimensionality_ratio=2.0,
            skewness_mean=5.0,
            kurtosis_mean=25.0,
            class_imbalance=0.9,
        )

        performance = AlgorithmPerformanceDTO(
            primary_metric=0.3,
            training_time_seconds=1800.0,
            memory_usage_mb=8192.0,
            stability_score=0.2,
            interpretability_score=0.1,
            confidence_score=0.3,
        )

        dto = SelectionHistoryDTO(
            dataset_characteristics=dataset_chars,
            selected_algorithm="complex_neural_network",
            performance=performance,
            timestamp=datetime.now(),
            dataset_hash="challenging_hash",
            user_feedback={"satisfaction": 2, "would_use_again": False},
            was_successful=False,
            lessons_learned=[
                "Dataset too small for complex model",
                "High dimensionality caused overfitting",
                "Need better feature selection",
            ],
        )

        assert dto.was_successful is False
        assert dto.performance.primary_metric == 0.3
        assert dto.user_feedback["satisfaction"] == 2
        assert len(dto.lessons_learned) == 3
        assert "Dataset too small for complex model" in dto.lessons_learned

    def test_comprehensive_feedback(self):
        """Test comprehensive user feedback."""
        dataset_chars = DatasetCharacteristicsDTO(
            n_samples=5000,
            n_features=50,
            n_numeric_features=40,
            n_categorical_features=10,
            feature_density=0.95,
            missing_value_ratio=0.01,
            outlier_ratio=0.02,
            mean_feature_correlation=0.2,
            feature_variance_ratio=3.0,
            data_dimensionality_ratio=0.01,
            skewness_mean=0.3,
            kurtosis_mean=1.8,
            class_imbalance=0.1,
        )

        performance = AlgorithmPerformanceDTO(
            primary_metric=0.92,
            training_time_seconds=600.0,
            memory_usage_mb=1024.0,
            stability_score=0.95,
            interpretability_score=0.85,
            confidence_score=0.9,
        )

        feedback = {
            "satisfaction": 9,
            "would_use_again": True,
            "ease_of_use": 8,
            "performance_met_expectations": True,
            "training_time_acceptable": True,
            "interpretability_sufficient": True,
            "additional_comments": "Excellent performance on this dataset",
        }

        dto = SelectionHistoryDTO(
            dataset_characteristics=dataset_chars,
            selected_algorithm="gradient_boosting",
            performance=performance,
            timestamp=datetime.now(),
            dataset_hash="success_hash",
            user_feedback=feedback,
            was_successful=True,
            lessons_learned=["Gradient boosting works well on this type of data"],
        )

        assert dto.user_feedback["satisfaction"] == 9
        assert dto.user_feedback["performance_met_expectations"] is True
        assert (
            dto.user_feedback["additional_comments"]
            == "Excellent performance on this dataset"
        )
        assert len(dto.user_feedback) == 7


class TestLearningInsightsDTO:
    """Test suite for LearningInsightsDTO."""

    def test_valid_creation(self):
        """Test creating a valid learning insights DTO."""
        generated_at = datetime(2023, 1, 15, 10, 0, 0)

        dto = LearningInsightsDTO(
            total_selections=100,
            unique_algorithms=15,
            unique_datasets=50,
            algorithm_performance_stats={
                "random_forest": {"mean": 0.85, "std": 0.05, "count": 25},
                "xgboost": {"mean": 0.82, "std": 0.07, "count": 20},
                "svm": {"mean": 0.78, "std": 0.06, "count": 15},
            },
            dataset_type_preferences={
                "tabular": ["random_forest", "xgboost", "lightgbm"],
                "text": ["naive_bayes", "svm", "neural_network"],
                "image": ["cnn", "resnet", "vgg"],
            },
            performance_trends={
                "random_forest": [0.82, 0.84, 0.85, 0.86, 0.85],
                "xgboost": [0.78, 0.80, 0.82, 0.83, 0.82],
                "neural_network": [0.70, 0.75, 0.80, 0.82, 0.85],
            },
            feature_importance_insights={
                "n_samples": 0.25,
                "n_features": 0.20,
                "feature_density": 0.15,
                "missing_value_ratio": 0.10,
                "class_imbalance": 0.30,
            },
            meta_model_accuracy=0.78,
            recommendation_confidence=0.82,
            generated_at=generated_at,
            analysis_period_days=90,
        )

        assert dto.total_selections == 100
        assert dto.unique_algorithms == 15
        assert dto.unique_datasets == 50
        assert len(dto.algorithm_performance_stats) == 3
        assert dto.algorithm_performance_stats["random_forest"]["mean"] == 0.85
        assert len(dto.dataset_type_preferences) == 3
        assert "random_forest" in dto.dataset_type_preferences["tabular"]
        assert len(dto.performance_trends["random_forest"]) == 5
        assert dto.feature_importance_insights["class_imbalance"] == 0.30
        assert dto.meta_model_accuracy == 0.78
        assert dto.recommendation_confidence == 0.82
        assert dto.generated_at == generated_at
        assert dto.analysis_period_days == 90

    def test_minimal_creation(self):
        """Test creating with minimal required fields."""
        dto = LearningInsightsDTO(total_selections=10, generated_at=datetime.now())

        assert dto.total_selections == 10
        assert dto.unique_algorithms is None
        assert dto.unique_datasets is None
        assert dto.algorithm_performance_stats == {}
        assert dto.dataset_type_preferences == {}
        assert dto.performance_trends == {}
        assert dto.feature_importance_insights == {}
        assert dto.meta_model_accuracy is None
        assert dto.recommendation_confidence == 0.0
        assert dto.analysis_period_days is None

    def test_extensive_algorithm_stats(self):
        """Test extensive algorithm performance statistics."""
        stats = {
            "random_forest": {
                "mean": 0.85,
                "std": 0.05,
                "count": 25,
                "min": 0.75,
                "max": 0.95,
            },
            "xgboost": {
                "mean": 0.82,
                "std": 0.07,
                "count": 20,
                "min": 0.70,
                "max": 0.92,
            },
            "lightgbm": {
                "mean": 0.83,
                "std": 0.06,
                "count": 18,
                "min": 0.72,
                "max": 0.90,
            },
            "catboost": {
                "mean": 0.81,
                "std": 0.08,
                "count": 15,
                "min": 0.68,
                "max": 0.89,
            },
            "svm": {"mean": 0.78, "std": 0.06, "count": 12, "min": 0.68, "max": 0.85},
            "neural_network": {
                "mean": 0.80,
                "std": 0.10,
                "count": 10,
                "min": 0.60,
                "max": 0.95,
            },
        }

        dto = LearningInsightsDTO(
            total_selections=100,
            unique_algorithms=6,
            algorithm_performance_stats=stats,
            generated_at=datetime.now(),
        )

        assert len(dto.algorithm_performance_stats) == 6
        assert dto.algorithm_performance_stats["random_forest"]["mean"] == 0.85
        assert dto.algorithm_performance_stats["neural_network"]["std"] == 0.10
        assert dto.algorithm_performance_stats["catboost"]["count"] == 15

    def test_domain_specific_preferences(self):
        """Test domain-specific algorithm preferences."""
        preferences = {
            "finance": ["random_forest", "xgboost", "logistic_regression"],
            "healthcare": ["svm", "neural_network", "naive_bayes"],
            "marketing": ["clustering", "decision_tree", "random_forest"],
            "manufacturing": ["time_series", "regression", "anomaly_detection"],
            "retail": ["recommendation_systems", "clustering", "classification"],
        }

        dto = LearningInsightsDTO(
            total_selections=200,
            dataset_type_preferences=preferences,
            generated_at=datetime.now(),
        )

        assert len(dto.dataset_type_preferences) == 5
        assert "random_forest" in dto.dataset_type_preferences["finance"]
        assert "svm" in dto.dataset_type_preferences["healthcare"]
        assert "clustering" in dto.dataset_type_preferences["marketing"]
        assert "time_series" in dto.dataset_type_preferences["manufacturing"]

    def test_performance_trends_analysis(self):
        """Test performance trends over time."""
        trends = {
            "random_forest": [0.80, 0.82, 0.83, 0.85, 0.84, 0.86, 0.85],
            "xgboost": [0.75, 0.77, 0.80, 0.82, 0.81, 0.83, 0.82],
            "neural_network": [0.65, 0.70, 0.75, 0.78, 0.80, 0.82, 0.85],
            "svm": [0.78, 0.77, 0.79, 0.78, 0.77, 0.79, 0.78],
        }

        dto = LearningInsightsDTO(
            total_selections=150, performance_trends=trends, generated_at=datetime.now()
        )

        assert len(dto.performance_trends) == 4
        assert len(dto.performance_trends["random_forest"]) == 7
        assert max(dto.performance_trends["random_forest"]) == 0.86
        assert min(dto.performance_trends["neural_network"]) == 0.65
        assert dto.performance_trends["svm"][-1] == 0.78

    def test_feature_importance_insights(self):
        """Test feature importance insights."""
        insights = {
            "n_samples": 0.25,
            "n_features": 0.20,
            "feature_density": 0.15,
            "missing_value_ratio": 0.10,
            "outlier_ratio": 0.05,
            "mean_feature_correlation": 0.08,
            "feature_variance_ratio": 0.07,
            "data_dimensionality_ratio": 0.05,
            "class_imbalance": 0.30,
        }

        dto = LearningInsightsDTO(
            total_selections=75,
            feature_importance_insights=insights,
            generated_at=datetime.now(),
        )

        assert len(dto.feature_importance_insights) == 9
        assert dto.feature_importance_insights["class_imbalance"] == 0.30
        assert dto.feature_importance_insights["n_samples"] == 0.25
        assert dto.feature_importance_insights["n_features"] == 0.20
        assert sum(dto.feature_importance_insights.values()) == 1.25

    def test_high_confidence_insights(self):
        """Test insights with high confidence."""
        dto = LearningInsightsDTO(
            total_selections=500,
            unique_algorithms=20,
            unique_datasets=200,
            meta_model_accuracy=0.92,
            recommendation_confidence=0.95,
            generated_at=datetime.now(),
            analysis_period_days=365,
        )

        assert dto.total_selections == 500
        assert dto.unique_algorithms == 20
        assert dto.unique_datasets == 200
        assert dto.meta_model_accuracy == 0.92
        assert dto.recommendation_confidence == 0.95
        assert dto.analysis_period_days == 365

    def test_low_confidence_insights(self):
        """Test insights with low confidence."""
        dto = LearningInsightsDTO(
            total_selections=5,
            unique_algorithms=3,
            unique_datasets=4,
            meta_model_accuracy=0.45,
            recommendation_confidence=0.35,
            generated_at=datetime.now(),
            analysis_period_days=7,
        )

        assert dto.total_selections == 5
        assert dto.unique_algorithms == 3
        assert dto.unique_datasets == 4
        assert dto.meta_model_accuracy == 0.45
        assert dto.recommendation_confidence == 0.35
        assert dto.analysis_period_days == 7


class TestAlgorithmComparisonDTO:
    """Test suite for AlgorithmComparisonDTO."""

    def test_valid_creation(self):
        """Test creating a valid algorithm comparison DTO."""
        dto = AlgorithmComparisonDTO(
            algorithm_a="random_forest",
            algorithm_b="xgboost",
            performance_difference=0.03,
            statistical_significance=0.02,
            time_difference_seconds=-30.0,
            memory_difference_mb=100.0,
            interpretability_comparison="Random Forest is more interpretable",
            stability_comparison="Both algorithms show similar stability",
            recommended_choice="random_forest",
            recommendation_reasoning=[
                "Slightly better performance",
                "More interpretable",
                "Faster training time",
            ],
        )

        assert dto.algorithm_a == "random_forest"
        assert dto.algorithm_b == "xgboost"
        assert dto.performance_difference == 0.03
        assert dto.statistical_significance == 0.02
        assert dto.time_difference_seconds == -30.0
        assert dto.memory_difference_mb == 100.0
        assert dto.interpretability_comparison == "Random Forest is more interpretable"
        assert dto.stability_comparison == "Both algorithms show similar stability"
        assert dto.recommended_choice == "random_forest"
        assert len(dto.recommendation_reasoning) == 3

    def test_minimal_creation(self):
        """Test creating with minimal required fields."""
        dto = AlgorithmComparisonDTO(
            algorithm_a="svm",
            algorithm_b="logistic_regression",
            performance_difference=-0.05,
            recommended_choice="logistic_regression",
        )

        assert dto.algorithm_a == "svm"
        assert dto.algorithm_b == "logistic_regression"
        assert dto.performance_difference == -0.05
        assert dto.statistical_significance is None
        assert dto.time_difference_seconds == 0.0
        assert dto.memory_difference_mb == 0.0
        assert dto.interpretability_comparison is None
        assert dto.stability_comparison is None
        assert dto.recommended_choice == "logistic_regression"
        assert dto.recommendation_reasoning == []

    def test_significant_performance_difference(self):
        """Test comparison with significant performance difference."""
        dto = AlgorithmComparisonDTO(
            algorithm_a="neural_network",
            algorithm_b="decision_tree",
            performance_difference=0.15,
            statistical_significance=0.001,
            time_difference_seconds=1500.0,
            memory_difference_mb=1536.0,
            interpretability_comparison="Decision Tree is much more interpretable",
            stability_comparison="Neural Network shows lower stability",
            recommended_choice="neural_network",
            recommendation_reasoning=[
                "Significantly better performance",
                "Performance gain justifies complexity",
                "Suitable for complex patterns",
            ],
        )

        assert dto.performance_difference == 0.15
        assert dto.statistical_significance == 0.001
        assert dto.time_difference_seconds == 1500.0
        assert dto.memory_difference_mb == 1536.0
        assert dto.recommended_choice == "neural_network"
        assert "Significantly better performance" in dto.recommendation_reasoning

    def test_close_performance_comparison(self):
        """Test comparison with very close performance."""
        dto = AlgorithmComparisonDTO(
            algorithm_a="random_forest",
            algorithm_b="extra_trees",
            performance_difference=0.002,
            statistical_significance=0.45,
            time_difference_seconds=5.0,
            memory_difference_mb=-20.0,
            interpretability_comparison="Both have similar interpretability",
            stability_comparison="Extra Trees slightly more stable",
            recommended_choice="extra_trees",
            recommendation_reasoning=[
                "Marginally better performance",
                "Lower memory usage",
                "Slightly more stable",
            ],
        )

        assert dto.performance_difference == 0.002
        assert dto.statistical_significance == 0.45
        assert dto.time_difference_seconds == 5.0
        assert dto.memory_difference_mb == -20.0
        assert dto.recommended_choice == "extra_trees"
        assert "Lower memory usage" in dto.recommendation_reasoning

    def test_trade_off_comparison(self):
        """Test comparison highlighting trade-offs."""
        dto = AlgorithmComparisonDTO(
            algorithm_a="complex_ensemble",
            algorithm_b="simple_linear",
            performance_difference=0.08,
            statistical_significance=0.01,
            time_difference_seconds=3600.0,
            memory_difference_mb=2048.0,
            interpretability_comparison="Simple Linear is much more interpretable",
            stability_comparison="Complex Ensemble shows higher variance",
            recommended_choice="simple_linear",
            recommendation_reasoning=[
                "Interpretability is crucial for this use case",
                "Faster inference required",
                "Performance difference not worth complexity",
            ],
        )

        assert dto.performance_difference == 0.08
        assert dto.time_difference_seconds == 3600.0
        assert dto.memory_difference_mb == 2048.0
        assert dto.recommended_choice == "simple_linear"
        assert "Interpretability is crucial" in dto.recommendation_reasoning
        assert (
            "Performance difference not worth complexity"
            in dto.recommendation_reasoning
        )

    def test_detailed_reasoning(self):
        """Test comparison with detailed reasoning."""
        reasoning = [
            "Algorithm A shows 5% better accuracy",
            "Algorithm B trains 3x faster",
            "Algorithm A requires 2x more memory",
            "Algorithm B is more suitable for production",
            "Algorithm A has better handling of missing values",
            "Algorithm B provides better feature importance scores",
        ]

        dto = AlgorithmComparisonDTO(
            algorithm_a="gradient_boosting",
            algorithm_b="random_forest",
            performance_difference=0.05,
            statistical_significance=0.03,
            time_difference_seconds=200.0,
            memory_difference_mb=512.0,
            interpretability_comparison="Random Forest provides better interpretability",
            stability_comparison="Gradient Boosting shows slightly higher variance",
            recommended_choice="random_forest",
            recommendation_reasoning=reasoning,
        )

        assert len(dto.recommendation_reasoning) == 6
        assert "Algorithm A shows 5% better accuracy" in dto.recommendation_reasoning
        assert (
            "Algorithm B is more suitable for production"
            in dto.recommendation_reasoning
        )
        assert dto.recommended_choice == "random_forest"


class TestPerformancePredictionDTO:
    """Test suite for PerformancePredictionDTO."""

    def test_valid_creation(self):
        """Test creating a valid performance prediction DTO."""
        prediction_time = datetime(2023, 1, 15, 10, 30, 0)

        dto = PerformancePredictionDTO(
            algorithm="random_forest",
            predicted_performance=0.85,
            confidence_interval=(0.82, 0.88),
            prediction_confidence=0.9,
            uncertainty_sources=["Limited historical data", "Dataset complexity"],
            similar_datasets_count=25,
            historical_performance_range=(0.75, 0.92),
            prediction_timestamp=prediction_time,
            model_version="v2.1.0",
        )

        assert dto.algorithm == "random_forest"
        assert dto.predicted_performance == 0.85
        assert dto.confidence_interval == (0.82, 0.88)
        assert dto.prediction_confidence == 0.9
        assert dto.uncertainty_sources == [
            "Limited historical data",
            "Dataset complexity",
        ]
        assert dto.similar_datasets_count == 25
        assert dto.historical_performance_range == (0.75, 0.92)
        assert dto.prediction_timestamp == prediction_time
        assert dto.model_version == "v2.1.0"

    def test_minimal_creation(self):
        """Test creating with minimal required fields."""
        dto = PerformancePredictionDTO(
            algorithm="svm",
            predicted_performance=0.78,
            confidence_interval=(0.73, 0.83),
            prediction_confidence=0.7,
        )

        assert dto.algorithm == "svm"
        assert dto.predicted_performance == 0.78
        assert dto.confidence_interval == (0.73, 0.83)
        assert dto.prediction_confidence == 0.7
        assert dto.uncertainty_sources == []
        assert dto.similar_datasets_count == 0
        assert dto.historical_performance_range is None
        assert dto.model_version is None

    def test_high_confidence_prediction(self):
        """Test high confidence prediction."""
        dto = PerformancePredictionDTO(
            algorithm="logistic_regression",
            predicted_performance=0.82,
            confidence_interval=(0.81, 0.83),
            prediction_confidence=0.95,
            uncertainty_sources=[],
            similar_datasets_count=100,
            historical_performance_range=(0.80, 0.84),
            model_version="v1.0.0",
        )

        assert dto.predicted_performance == 0.82
        assert dto.confidence_interval == (0.81, 0.83)
        assert dto.prediction_confidence == 0.95
        assert dto.uncertainty_sources == []
        assert dto.similar_datasets_count == 100
        assert dto.historical_performance_range == (0.80, 0.84)

    def test_low_confidence_prediction(self):
        """Test low confidence prediction."""
        uncertainty_sources = [
            "Very limited historical data",
            "Novel dataset characteristics",
            "High feature dimensionality",
            "Unusual class distribution",
            "Potential data quality issues",
        ]

        dto = PerformancePredictionDTO(
            algorithm="neural_network",
            predicted_performance=0.65,
            confidence_interval=(0.45, 0.85),
            prediction_confidence=0.3,
            uncertainty_sources=uncertainty_sources,
            similar_datasets_count=2,
            historical_performance_range=(0.40, 0.90),
        )

        assert dto.predicted_performance == 0.65
        assert dto.confidence_interval == (0.45, 0.85)
        assert dto.prediction_confidence == 0.3
        assert len(dto.uncertainty_sources) == 5
        assert dto.similar_datasets_count == 2
        assert dto.historical_performance_range == (0.40, 0.90)

    def test_invalid_performance_range(self):
        """Test validation for performance out of range."""
        with pytest.raises(ValueError):
            PerformancePredictionDTO(
                algorithm="test_algo",
                predicted_performance=1.5,
                confidence_interval=(0.8, 0.9),
                prediction_confidence=0.8,
            )

    def test_invalid_confidence_range(self):
        """Test validation for confidence out of range."""
        with pytest.raises(ValueError):
            PerformancePredictionDTO(
                algorithm="test_algo",
                predicted_performance=0.8,
                confidence_interval=(0.7, 0.9),
                prediction_confidence=1.2,
            )

    def test_narrow_confidence_interval(self):
        """Test narrow confidence interval."""
        dto = PerformancePredictionDTO(
            algorithm="stable_algorithm",
            predicted_performance=0.88,
            confidence_interval=(0.87, 0.89),
            prediction_confidence=0.92,
            similar_datasets_count=50,
            historical_performance_range=(0.85, 0.91),
        )

        assert dto.confidence_interval == (0.87, 0.89)
        interval_width = dto.confidence_interval[1] - dto.confidence_interval[0]
        assert interval_width == 0.02

    def test_wide_confidence_interval(self):
        """Test wide confidence interval."""
        dto = PerformancePredictionDTO(
            algorithm="unstable_algorithm",
            predicted_performance=0.70,
            confidence_interval=(0.50, 0.90),
            prediction_confidence=0.4,
            similar_datasets_count=3,
            historical_performance_range=(0.45, 0.95),
        )

        assert dto.confidence_interval == (0.50, 0.90)
        interval_width = dto.confidence_interval[1] - dto.confidence_interval[0]
        assert interval_width == 0.40


class TestSelectionExplanationDTO:
    """Test suite for SelectionExplanationDTO."""

    def test_valid_creation(self):
        """Test creating a valid selection explanation DTO."""
        dto = SelectionExplanationDTO(
            primary_reason="Random Forest showed the best performance on similar datasets",
            supporting_reasons=[
                "High accuracy on tabular data",
                "Good interpretability",
                "Robust to outliers",
                "Efficient training time",
            ],
            evidence_sources=[
                "Historical performance data",
                "Cross-validation results",
                "Meta-learning model predictions",
            ],
            confidence_factors={
                "historical_similarity": 0.85,
                "dataset_characteristics": 0.78,
                "meta_model_confidence": 0.82,
            },
            alternative_algorithms=["gradient_boosting", "extra_trees", "svm"],
            why_not_alternatives={
                "gradient_boosting": "Slightly lower predicted performance",
                "extra_trees": "Less interpretable than Random Forest",
                "svm": "Poor performance on high-dimensional data",
            },
            assumptions=[
                "Data quality is consistent",
                "Feature distribution remains stable",
                "Target variable definition unchanged",
            ],
            limitations=[
                "Prediction based on limited historical data",
                "May not generalize to very different datasets",
            ],
            recommendations=[
                "Monitor performance on validation set",
                "Consider hyperparameter tuning",
                "Evaluate interpretability requirements",
            ],
        )

        assert (
            dto.primary_reason
            == "Random Forest showed the best performance on similar datasets"
        )
        assert len(dto.supporting_reasons) == 4
        assert len(dto.evidence_sources) == 3
        assert len(dto.confidence_factors) == 3
        assert len(dto.alternative_algorithms) == 3
        assert len(dto.why_not_alternatives) == 3
        assert len(dto.assumptions) == 3
        assert len(dto.limitations) == 2
        assert len(dto.recommendations) == 3

    def test_minimal_creation(self):
        """Test creating with minimal required fields."""
        dto = SelectionExplanationDTO(
            primary_reason="Algorithm performed well in initial tests"
        )

        assert dto.primary_reason == "Algorithm performed well in initial tests"
        assert dto.supporting_reasons == []
        assert dto.evidence_sources == []
        assert dto.confidence_factors == {}
        assert dto.alternative_algorithms == []
        assert dto.why_not_alternatives == {}
        assert dto.assumptions == []
        assert dto.limitations == []
        assert dto.recommendations == []

    def test_detailed_confidence_factors(self):
        """Test detailed confidence factors."""
        confidence_factors = {
            "historical_similarity": 0.85,
            "dataset_size_match": 0.78,
            "feature_type_similarity": 0.82,
            "domain_relevance": 0.90,
            "meta_model_confidence": 0.75,
            "cross_validation_stability": 0.88,
            "expert_validation": 0.92,
            "computational_feasibility": 0.95,
        }

        dto = SelectionExplanationDTO(
            primary_reason="Comprehensive analysis supports this choice",
            confidence_factors=confidence_factors,
        )

        assert len(dto.confidence_factors) == 8
        assert dto.confidence_factors["historical_similarity"] == 0.85
        assert dto.confidence_factors["expert_validation"] == 0.92
        assert dto.confidence_factors["computational_feasibility"] == 0.95

    def test_comprehensive_alternatives_analysis(self):
        """Test comprehensive alternatives analysis."""
        alternatives = [
            "gradient_boosting",
            "extra_trees",
            "svm",
            "neural_network",
            "naive_bayes",
            "decision_tree",
            "knn",
            "linear_regression",
        ]

        why_not = {
            "gradient_boosting": "Requires more hyperparameter tuning",
            "extra_trees": "Similar performance but less interpretable",
            "svm": "Poor scalability with large datasets",
            "neural_network": "Requires more computational resources",
            "naive_bayes": "Strong independence assumptions not met",
            "decision_tree": "Prone to overfitting on this dataset",
            "knn": "Computationally expensive for prediction",
            "linear_regression": "Dataset shows non-linear patterns",
        }

        dto = SelectionExplanationDTO(
            primary_reason="Random Forest balances performance and interpretability",
            alternative_algorithms=alternatives,
            why_not_alternatives=why_not,
        )

        assert len(dto.alternative_algorithms) == 8
        assert len(dto.why_not_alternatives) == 8
        assert "gradient_boosting" in dto.alternative_algorithms
        assert "neural_network" in dto.alternative_algorithms
        assert dto.why_not_alternatives["svm"] == "Poor scalability with large datasets"
        assert (
            dto.why_not_alternatives["neural_network"]
            == "Requires more computational resources"
        )

    def test_risk_assessment_explanation(self):
        """Test explanation with risk assessment."""
        dto = SelectionExplanationDTO(
            primary_reason="Conservative choice based on risk assessment",
            supporting_reasons=[
                "Well-established algorithm with proven track record",
                "Lower risk of unexpected failures",
                "Good baseline performance guaranteed",
            ],
            assumptions=[
                "Risk tolerance is low",
                "Reliability preferred over maximum performance",
                "Deployment timeline is tight",
            ],
            limitations=[
                "May not achieve absolute best performance",
                "Less innovative than cutting-edge methods",
                "May require more feature engineering",
            ],
            recommendations=[
                "Consider upgrading to more advanced methods later",
                "Monitor performance degradation over time",
                "Plan for iterative improvement",
            ],
        )

        assert "Conservative choice based on risk assessment" in dto.primary_reason
        assert "Risk tolerance is low" in dto.assumptions
        assert "May not achieve absolute best performance" in dto.limitations
        assert (
            "Consider upgrading to more advanced methods later" in dto.recommendations
        )

    def test_performance_focused_explanation(self):
        """Test performance-focused explanation."""
        dto = SelectionExplanationDTO(
            primary_reason="Achieves highest predicted performance on this dataset",
            supporting_reasons=[
                "Outperforms alternatives by 5% on similar data",
                "Excellent handling of feature interactions",
                "Robust to noise and outliers",
                "Scales well with dataset size",
            ],
            evidence_sources=[
                "Benchmark studies on similar datasets",
                "Cross-validation performance",
                "Meta-learning predictions",
                "Algorithm complexity analysis",
            ],
            confidence_factors={
                "performance_prediction": 0.92,
                "dataset_similarity": 0.88,
                "validation_consistency": 0.85,
            },
            assumptions=[
                "Performance is the primary objective",
                "Computational resources are available",
                "Interpretability is secondary",
            ],
        )

        assert "highest predicted performance" in dto.primary_reason
        assert "Outperforms alternatives by 5%" in dto.supporting_reasons
        assert "Performance is the primary objective" in dto.assumptions
        assert dto.confidence_factors["performance_prediction"] == 0.92


class TestSelectionRequestDTO:
    """Test suite for SelectionRequestDTO."""

    def test_valid_creation(self):
        """Test creating a valid selection request DTO."""
        dataset_id = uuid4()

        dataset_chars = DatasetCharacteristicsDTO(
            n_samples=1000,
            n_features=10,
            n_numeric_features=8,
            n_categorical_features=2,
            feature_density=0.8,
            missing_value_ratio=0.1,
            outlier_ratio=0.05,
            mean_feature_correlation=0.3,
            feature_variance_ratio=2.0,
            data_dimensionality_ratio=0.01,
            skewness_mean=0.5,
            kurtosis_mean=2.0,
            class_imbalance=0.2,
        )

        constraints = OptimizationConstraintsDTO(
            max_training_time_seconds=300.0,
            max_memory_mb=1024.0,
            min_accuracy=0.8,
            require_interpretability=True,
        )

        dto = SelectionRequestDTO(
            dataset_id=dataset_id,
            dataset_characteristics=dataset_chars,
            optimization_goal="balanced",
            constraints=constraints,
            require_explanation=True,
            include_alternatives=True,
            use_meta_learning=True,
            user_expertise="intermediate",
            use_case="fraud_detection",
            deployment_environment="production",
        )

        assert dto.dataset_id == dataset_id
        assert dto.dataset_characteristics == dataset_chars
        assert dto.optimization_goal == "balanced"
        assert dto.constraints == constraints
        assert dto.require_explanation is True
        assert dto.include_alternatives is True
        assert dto.use_meta_learning is True
        assert dto.user_expertise == "intermediate"
        assert dto.use_case == "fraud_detection"
        assert dto.deployment_environment == "production"

    def test_minimal_creation(self):
        """Test creating with minimal required fields."""
        dto = SelectionRequestDTO()

        assert dto.dataset_id is None
        assert dto.dataset_characteristics is None
        assert dto.optimization_goal == "performance"
        assert dto.constraints is None
        assert dto.require_explanation is True
        assert dto.include_alternatives is True
        assert dto.use_meta_learning is True
        assert dto.user_expertise is None
        assert dto.use_case is None
        assert dto.deployment_environment is None

    def test_performance_optimization_goal(self):
        """Test performance optimization goal."""
        dto = SelectionRequestDTO(
            optimization_goal="performance",
            constraints=None,
            require_explanation=False,
            include_alternatives=False,
            use_meta_learning=True,
        )

        assert dto.optimization_goal == "performance"
        assert dto.constraints is None
        assert dto.require_explanation is False
        assert dto.include_alternatives is False
        assert dto.use_meta_learning is True

    def test_speed_optimization_goal(self):
        """Test speed optimization goal."""
        constraints = OptimizationConstraintsDTO(
            max_training_time_seconds=60.0,
            max_prediction_time_ms=10.0,
            require_online_prediction=True,
        )

        dto = SelectionRequestDTO(
            optimization_goal="speed",
            constraints=constraints,
            deployment_environment="real_time",
        )

        assert dto.optimization_goal == "speed"
        assert dto.constraints.max_training_time_seconds == 60.0
        assert dto.constraints.max_prediction_time_ms == 10.0
        assert dto.constraints.require_online_prediction is True
        assert dto.deployment_environment == "real_time"

    def test_interpretability_optimization_goal(self):
        """Test interpretability optimization goal."""
        constraints = OptimizationConstraintsDTO(
            min_interpretability=0.8, require_interpretability=True
        )

        dto = SelectionRequestDTO(
            optimization_goal="interpretability",
            constraints=constraints,
            user_expertise="beginner",
            use_case="medical_diagnosis",
        )

        assert dto.optimization_goal == "interpretability"
        assert dto.constraints.min_interpretability == 0.8
        assert dto.constraints.require_interpretability is True
        assert dto.user_expertise == "beginner"
        assert dto.use_case == "medical_diagnosis"

    def test_different_user_expertise_levels(self):
        """Test different user expertise levels."""
        expertise_levels = ["beginner", "intermediate", "expert"]

        for level in expertise_levels:
            dto = SelectionRequestDTO(
                user_expertise=level,
                require_explanation=True if level == "beginner" else False,
            )
            assert dto.user_expertise == level

    def test_various_use_cases(self):
        """Test various use cases."""
        use_cases = [
            "fraud_detection",
            "medical_diagnosis",
            "marketing_optimization",
            "predictive_maintenance",
            "risk_assessment",
            "recommendation_system",
        ]

        for use_case in use_cases:
            dto = SelectionRequestDTO(use_case=use_case, optimization_goal="balanced")
            assert dto.use_case == use_case

    def test_deployment_environments(self):
        """Test different deployment environments."""
        environments = [
            "development",
            "testing",
            "staging",
            "production",
            "real_time",
            "batch",
            "edge",
            "cloud",
        ]

        for env in environments:
            dto = SelectionRequestDTO(deployment_environment=env)
            assert dto.deployment_environment == env

    def test_comprehensive_request(self):
        """Test comprehensive request with all fields."""
        dataset_id = uuid4()

        dataset_chars = DatasetCharacteristicsDTO(
            n_samples=50000,
            n_features=200,
            n_numeric_features=150,
            n_categorical_features=50,
            feature_density=0.6,
            missing_value_ratio=0.08,
            outlier_ratio=0.03,
            mean_feature_correlation=0.4,
            feature_variance_ratio=8.0,
            data_dimensionality_ratio=0.004,
            skewness_mean=1.2,
            kurtosis_mean=4.5,
            class_imbalance=0.25,
            estimated_complexity="high",
            data_type="mixed",
            domain="finance",
        )

        constraints = OptimizationConstraintsDTO(
            max_training_time_seconds=1800.0,
            max_memory_mb=4096.0,
            max_prediction_time_ms=50.0,
            min_accuracy=0.85,
            min_interpretability=0.6,
            min_stability=0.8,
            require_online_prediction=True,
            require_batch_prediction=True,
            require_interpretability=True,
            available_libraries=["scikit-learn", "xgboost", "lightgbm"],
            gpu_available=True,
            distributed_computing=False,
        )

        dto = SelectionRequestDTO(
            dataset_id=dataset_id,
            dataset_characteristics=dataset_chars,
            optimization_goal="balanced",
            constraints=constraints,
            require_explanation=True,
            include_alternatives=True,
            use_meta_learning=True,
            user_expertise="expert",
            use_case="risk_assessment",
            deployment_environment="production",
        )

        assert dto.dataset_id == dataset_id
        assert dto.dataset_characteristics.n_samples == 50000
        assert dto.dataset_characteristics.estimated_complexity == "high"
        assert dto.constraints.max_training_time_seconds == 1800.0
        assert dto.constraints.min_accuracy == 0.85
        assert dto.optimization_goal == "balanced"
        assert dto.user_expertise == "expert"
        assert dto.use_case == "risk_assessment"
        assert dto.deployment_environment == "production"
