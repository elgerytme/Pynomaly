"""Comprehensive tests for Data Transfer Objects (DTOs)."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest
from pydantic import ValidationError

from pynomaly.application.dto import (
    AnomalyDTO,
    CreateDetectorDTO,
    CreateExperimentDTO,
    DatasetDTO,
    DetectionRequestDTO,
    DetectionResultDTO,
    DetectorResponseDTO,
)
from pynomaly.domain.entities import Anomaly, Dataset, DetectionResult, Detector
from pynomaly.domain.value_objects import (
    AnomalyScore,
    ConfidenceInterval,
    ContaminationRate,
)


class TestCreateDetectorDTO:
    """Test CreateDetectorDTO validation and conversion."""

    def test_valid_dto_creation(self):
        """Test creating a valid CreateDetectorDTO."""
        dto = CreateDetectorDTO(
            name="test_detector",
            algorithm_name="IsolationForest",
            contamination_rate=0.1,
            parameters={"n_estimators": 100, "random_state": 42},
        )

        assert dto.name == "test_detector"
        assert dto.algorithm_name == "IsolationForest"
        assert dto.contamination_rate == 0.1
        assert dto.parameters["n_estimators"] == 100

    def test_contamination_validation(self):
        """Test contamination rate validation."""
        # Valid contamination rates
        CreateDetectorDTO(name="test", algorithm_name="test", contamination_rate=0.05)
        CreateDetectorDTO(name="test", algorithm_name="test", contamination_rate=0.5)

        # Invalid contamination rates
        with pytest.raises(ValidationError):
            CreateDetectorDTO(
                name="test", algorithm_name="test", contamination_rate=-0.1
            )

        with pytest.raises(ValidationError):
            CreateDetectorDTO(
                name="test", algorithm_name="test", contamination_rate=1.5
            )

    def test_name_validation(self):
        """Test name validation."""
        # Valid names
        CreateDetectorDTO(
            name="valid_name", algorithm_name="test", contamination_rate=0.1
        )
        CreateDetectorDTO(
            name="Valid Name 123", algorithm_name="test", contamination_rate=0.1
        )

        # Invalid names
        with pytest.raises(ValidationError):
            CreateDetectorDTO(name="", algorithm_name="test", contamination_rate=0.1)

        with pytest.raises(ValidationError):
            CreateDetectorDTO(name="   ", algorithm_name="test", contamination_rate=0.1)

    def test_algorithm_validation(self):
        """Test algorithm validation."""
        valid_algorithms = [
            "IsolationForest",
            "LocalOutlierFactor",
            "OneClassSVM",
            "LOF",
            "KNN",
            "ABOD",
        ]

        for algo in valid_algorithms:
            CreateDetectorDTO(name="test", algorithm_name=algo, contamination_rate=0.1)

        # Test with empty algorithm name should raise validation error
        with pytest.raises(ValidationError):
            CreateDetectorDTO(name="test", algorithm_name="", contamination_rate=0.1)

    def test_parameters_optional(self):
        """Test that parameters are optional."""
        dto = CreateDetectorDTO(
            name="test", algorithm_name="IsolationForest", contamination_rate=0.1
        )
        assert dto.parameters == {}

    def test_json_serialization(self):
        """Test JSON serialization of DTO."""
        dto = CreateDetectorDTO(
            name="test_detector",
            algorithm_name="IsolationForest",
            contamination_rate=0.1,
            parameters={"n_estimators": 100},
        )

        # Test that it can be serialized
        json_data = dto.model_dump()

        assert json_data["name"] == "test_detector"
        assert json_data["algorithm_name"] == "IsolationForest"
        assert json_data["contamination_rate"] == 0.1
        assert json_data["parameters"] == {"n_estimators": 100}


class TestDetectorResponseDTO:
    """Test DetectorResponseDTO serialization."""

    def test_valid_response_dto(self):
        """Test creating a valid DetectorResponseDTO."""
        from uuid import uuid4

        dto = DetectorResponseDTO(
            id=uuid4(),
            name="test_detector",
            algorithm_name="IsolationForest",
            contamination_rate=0.1,
            is_fitted=True,
            created_at=datetime.utcnow(),
            parameters={"n_estimators": 100},
        )

        assert dto.name == "test_detector"
        assert dto.algorithm_name == "IsolationForest"
        assert dto.contamination_rate == 0.1
        assert dto.is_fitted is True
        assert dto.parameters == {"n_estimators": 100}
        assert dto.status == "active"
        assert dto.version == "1.0.0"

    def test_serialization(self):
        """Test JSON serialization."""
        from uuid import uuid4

        dto = DetectorResponseDTO(
            id=uuid4(),
            name="test_detector",
            algorithm_name="IsolationForest",
            contamination_rate=0.1,
            is_fitted=False,
            created_at=datetime.utcnow(),
        )

        serialized = dto.model_dump()

        assert "id" in serialized
        assert "name" in serialized
        assert "algorithm_name" in serialized
        assert "contamination_rate" in serialized
        assert "created_at" in serialized
        assert "is_fitted" in serialized
        assert "status" in serialized
        assert "version" in serialized


class TestDatasetDTO:
    """Test DatasetDTO validation and conversion."""

    def test_valid_dataset_dto(self):
        """Test creating a valid DatasetDTO."""
        features = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        targets = [0, 1, 0]

        dto = DatasetDTO(name="test_dataset", features=features, targets=targets)

        assert dto.name == "test_dataset"
        assert dto.features == features
        assert dto.targets == targets

    def test_dataset_without_targets(self):
        """Test dataset DTO without targets."""
        features = [[1.0, 2.0], [3.0, 4.0]]

        dto = DatasetDTO(name="test_dataset", features=features)

        assert dto.name == "test_dataset"
        assert dto.features == features
        assert dto.targets is None

    def test_feature_validation(self):
        """Test feature data validation."""
        # Valid features
        DatasetDTO(name="test", features=[[1.0, 2.0], [3.0, 4.0]])

        # Invalid features - empty
        with pytest.raises(ValidationError):
            DatasetDTO(name="test", features=[])

        # Invalid features - inconsistent dimensions
        with pytest.raises(ValidationError):
            DatasetDTO(name="test", features=[[1.0, 2.0], [3.0]])

    def test_targets_validation(self):
        """Test targets validation."""
        features = [[1.0, 2.0], [3.0, 4.0]]

        # Valid targets
        DatasetDTO(name="test", features=features, targets=[0, 1])

        # Invalid targets - length mismatch
        with pytest.raises(ValidationError):
            DatasetDTO(name="test", features=features, targets=[0])

        # Invalid targets - not binary
        with pytest.raises(ValidationError):
            DatasetDTO(name="test", features=features, targets=[0, 2])

    def test_to_domain_entity(self):
        """Test conversion to domain entity."""
        features = [[1.0, 2.0], [3.0, 4.0]]
        targets = [0, 1]

        dto = DatasetDTO(name="test_dataset", features=features, targets=targets)
        dataset = dto.to_domain_entity()

        assert isinstance(dataset, Dataset)
        assert dataset.name == "test_dataset"
        assert dataset.features.shape == (2, 2)
        assert np.array_equal(dataset.targets, np.array([0, 1]))


class TestDetectionRequestDTO:
    """Test DetectionRequestDTO validation."""

    def test_valid_request(self):
        """Test valid detection request."""
        dto = DetectionRequestDTO(
            detector_id="detector123",
            dataset_name="test_data",
            features=[[1.0, 2.0], [3.0, 4.0]],
        )

        assert dto.detector_id == "detector123"
        assert dto.dataset_name == "test_data"
        assert dto.features == [[1.0, 2.0], [3.0, 4.0]]

    def test_optional_parameters(self):
        """Test optional parameters."""
        dto = DetectionRequestDTO(
            detector_id="detector123",
            dataset_name="test_data",
            features=[[1.0, 2.0]],
            preprocess=True,
            return_confidence=True,
            threshold_override=0.8,
        )

        assert dto.preprocess is True
        assert dto.return_confidence is True
        assert dto.threshold_override == 0.8

    def test_validation(self):
        """Test request validation."""
        # Invalid detector_id
        with pytest.raises(ValidationError):
            DetectionRequestDTO(detector_id="", dataset_name="test", features=[[1.0]])

        # Invalid threshold
        with pytest.raises(ValidationError):
            DetectionRequestDTO(
                detector_id="test",
                dataset_name="test",
                features=[[1.0]],
                threshold_override=1.5,
            )


class TestDetectionResultDTO:
    """Test DetectionResultDTO serialization."""

    def test_from_domain_entity(self):
        """Test creating DTO from domain entity."""
        # Create domain entities
        detector = Detector(
            name="test_detector",
            algorithm="isolation_forest",
            contamination=ContaminationRate(0.1),
        )

        features = np.random.random((100, 5))
        dataset = Dataset(name="test_dataset", features=features)

        scores = np.random.random(100)
        anomalies = [
            Anomaly(score=AnomalyScore(0.95), index=0),
            Anomaly(score=AnomalyScore(0.87), index=10),
        ]

        result = DetectionResult(
            detector=detector, dataset=dataset, anomalies=anomalies, scores=scores
        )

        dto = DetectionResultDTO.from_domain_entity(result)

        assert dto.detector_id == detector.id
        assert dto.dataset_name == "test_dataset"
        assert len(dto.anomalies) == 2
        assert len(dto.scores) == 100
        assert dto.total_samples == 100
        assert dto.anomaly_count == 2

    def test_summary_statistics(self):
        """Test summary statistics calculation."""
        detector = Detector(
            name="test", algorithm="test", contamination=ContaminationRate(0.1)
        )

        features = np.random.random((50, 3))
        dataset = Dataset(name="test", features=features)
        scores = np.array([0.1, 0.2, 0.9, 0.8, 0.3] + [0.1] * 45)  # 2 high scores

        anomalies = [
            Anomaly(score=AnomalyScore(0.9), index=2),
            Anomaly(score=AnomalyScore(0.8), index=3),
        ]

        result = DetectionResult(
            detector=detector, dataset=dataset, anomalies=anomalies, scores=scores
        )

        dto = DetectionResultDTO.from_domain_entity(result)

        assert dto.total_samples == 50
        assert dto.anomaly_count == 2
        assert dto.contamination_rate == 0.04  # 2/50
        assert dto.max_score == 0.9
        assert dto.min_score == 0.1
        assert abs(dto.mean_score - np.mean(scores)) < 1e-10


class TestAnomalyDTO:
    """Test AnomalyDTO serialization."""

    def test_from_domain_entity(self):
        """Test creating DTO from anomaly entity."""
        features = np.array([1.0, 2.0, 3.0])
        confidence = ConfidenceInterval(lower=0.8, upper=0.95, confidence_level=0.95)
        explanation = {"feature_0": 0.3, "feature_1": 0.7}

        anomaly = Anomaly(
            score=AnomalyScore(0.92),
            index=42,
            features=features,
            confidence=confidence,
            explanation=explanation,
        )

        dto = AnomalyDTO.from_domain_entity(anomaly)

        assert dto.score == 0.92
        assert dto.index == 42
        assert dto.features == [1.0, 2.0, 3.0]
        assert dto.confidence_lower == 0.8
        assert dto.confidence_upper == 0.95
        assert dto.explanation == explanation

    def test_without_optional_fields(self):
        """Test DTO creation without optional fields."""
        anomaly = Anomaly(score=AnomalyScore(0.85), index=10)
        dto = AnomalyDTO.from_domain_entity(anomaly)

        assert dto.score == 0.85
        assert dto.index == 10
        assert dto.features is None
        assert dto.confidence_lower is None
        assert dto.confidence_upper is None
        assert dto.explanation is None


class TestCreateExperimentDTO:
    """Test CreateExperimentDTO validation."""

    def test_valid_experiment_creation(self):
        """Test creating a valid experiment DTO."""
        config = {
            "algorithms": ["isolation_forest", "lof"],
            "contamination": 0.1,
            "dataset": "fraud_detection",
        }

        dto = CreateExperimentDTO(
            name="fraud_experiment",
            description="Fraud detection experiment",
            config=config,
        )

        assert dto.name == "fraud_experiment"
        assert dto.description == "Fraud detection experiment"
        assert dto.config == config

    def test_optional_description(self):
        """Test experiment creation without description."""
        dto = CreateExperimentDTO(name="test_experiment", config={"test": "value"})

        assert dto.name == "test_experiment"
        assert dto.description is None
        assert dto.config == {"test": "value"}

    def test_name_validation(self):
        """Test experiment name validation."""
        # Valid names
        CreateExperimentDTO(name="valid_name", config={})
        CreateExperimentDTO(name="Valid Name 123", config={})

        # Invalid names
        with pytest.raises(ValidationError):
            CreateExperimentDTO(name="", config={})

        with pytest.raises(ValidationError):
            CreateExperimentDTO(name="   ", config={})


class TestBatchDetectionRequestDTO:
    """Test BatchDetectionRequestDTO validation."""

    def test_valid_batch_request(self):
        """Test valid batch detection request."""
        datasets = [
            {"name": "dataset1", "features": [[1.0, 2.0]]},
            {"name": "dataset2", "features": [[3.0, 4.0]]},
        ]

        dto = BatchDetectionRequestDTO(detector_id="detector123", datasets=datasets)

        assert dto.detector_id == "detector123"
        assert len(dto.datasets) == 2

    def test_processing_options(self):
        """Test batch processing options."""
        datasets = [{"name": "test", "features": [[1.0]]}]

        dto = BatchDetectionRequestDTO(
            detector_id="detector123", datasets=datasets, parallel=True, max_workers=4
        )

        assert dto.parallel is True
        assert dto.max_workers == 4

    def test_validation(self):
        """Test batch request validation."""
        # Empty datasets
        with pytest.raises(ValidationError):
            BatchDetectionRequestDTO(detector_id="test", datasets=[])

        # Invalid max_workers
        with pytest.raises(ValidationError):
            BatchDetectionRequestDTO(
                detector_id="test",
                datasets=[{"name": "test", "features": [[1.0]]}],
                max_workers=0,
            )


class TestEnsembleRequestDTO:
    """Test EnsembleRequestDTO validation."""

    def test_valid_ensemble_request(self):
        """Test valid ensemble request."""
        dto = EnsembleRequestDTO(
            detector_ids=["det1", "det2", "det3"],
            dataset_name="test_data",
            features=[[1.0, 2.0]],
            aggregation_method="mean",
        )

        assert len(dto.detector_ids) == 3
        assert dto.aggregation_method == "mean"

    def test_weighted_ensemble(self):
        """Test weighted ensemble request."""
        dto = EnsembleRequestDTO(
            detector_ids=["det1", "det2"],
            dataset_name="test_data",
            features=[[1.0, 2.0]],
            aggregation_method="weighted",
            weights=[0.7, 0.3],
        )

        assert dto.aggregation_method == "weighted"
        assert dto.weights == [0.7, 0.3]

    def test_validation(self):
        """Test ensemble request validation."""
        # Too few detectors
        with pytest.raises(ValidationError):
            EnsembleRequestDTO(
                detector_ids=["det1"],
                dataset_name="test",
                features=[[1.0]],
                aggregation_method="mean",
            )

        # Invalid aggregation method
        with pytest.raises(ValidationError):
            EnsembleRequestDTO(
                detector_ids=["det1", "det2"],
                dataset_name="test",
                features=[[1.0]],
                aggregation_method="invalid",
            )

        # Weight count mismatch
        with pytest.raises(ValidationError):
            EnsembleRequestDTO(
                detector_ids=["det1", "det2"],
                dataset_name="test",
                features=[[1.0]],
                aggregation_method="weighted",
                weights=[0.5],  # Should be 2 weights
            )


class TestModelMetricsDTO:
    """Test ModelMetricsDTO calculations."""

    def test_classification_metrics(self):
        """Test classification metrics calculation."""
        y_true = [0, 0, 1, 1, 0, 1, 0, 0, 1, 1]
        y_pred = [0, 0, 1, 0, 0, 1, 1, 0, 1, 1]
        y_scores = [0.1, 0.2, 0.9, 0.3, 0.1, 0.8, 0.6, 0.2, 0.9, 0.85]

        dto = ModelMetricsDTO.from_predictions(
            y_true=y_true, y_pred=y_pred, y_scores=y_scores
        )

        assert 0 <= dto.accuracy <= 1
        assert 0 <= dto.precision <= 1
        assert 0 <= dto.recall <= 1
        assert 0 <= dto.f1_score <= 1
        assert 0 <= dto.auc_roc <= 1

    def test_without_ground_truth(self):
        """Test metrics calculation without ground truth."""
        y_scores = [0.1, 0.2, 0.9, 0.3, 0.8]

        dto = ModelMetricsDTO.from_scores_only(y_scores=y_scores)

        # Should only have score-based metrics
        assert dto.accuracy is None
        assert dto.precision is None
        assert dto.recall is None
        assert dto.f1_score is None
        assert dto.auc_roc is None

        # Should have score statistics
        assert dto.mean_score is not None
        assert dto.std_score is not None
        assert dto.min_score is not None
        assert dto.max_score is not None


class TestValidationErrorDTO:
    """Test ValidationErrorDTO for API error responses."""

    def test_single_error(self):
        """Test single validation error."""
        dto = ValidationErrorDTO(
            field="contamination", message="Value must be between 0 and 0.5", value=1.2
        )

        assert dto.field == "contamination"
        assert dto.message == "Value must be between 0 and 0.5"
        assert dto.value == 1.2

    def test_multiple_errors(self):
        """Test multiple validation errors."""
        errors = [
            ValidationErrorDTO(field="name", message="Name cannot be empty", value=""),
            ValidationErrorDTO(
                field="algorithm", message="Unknown algorithm", value="invalid"
            ),
        ]

        assert len(errors) == 2
        assert errors[0].field == "name"
        assert errors[1].field == "algorithm"

    def test_error_serialization(self):
        """Test error serialization for API responses."""
        dto = ValidationErrorDTO(
            field="features", message="Features cannot be empty", value=[]
        )

        serialized = dto.dict()

        assert "field" in serialized
        assert "message" in serialized
        assert "value" in serialized
