"""Tests for Detection DTOs."""

from datetime import datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from monorepo.application.dto.detection_dto import (
    AnomalyDTO,
    DetectionRequestDTO,
    DetectionResultDTO,
    DetectionSummaryDTO,
    ExplanationRequestDTO,
    ExplanationResultDTO,
    TrainingRequestDTO,
    TrainingResultDTO,
)


class TestDetectionRequestDTO:
    """Test suite for DetectionRequestDTO."""

    def test_valid_creation_with_dataset_id(self):
        """Test creating a valid detection request with dataset_id."""
        detector_id = uuid4()
        dataset_id = uuid4()

        dto = DetectionRequestDTO(
            detector_id=detector_id,
            dataset_id=dataset_id,
            threshold=0.6,
            return_scores=True,
            return_explanations=True,
            validate_features=False,
            save_results=False,
        )

        assert dto.detector_id == detector_id
        assert dto.dataset_id == dataset_id
        assert dto.data is None
        assert dto.threshold == 0.6
        assert dto.return_scores is True
        assert dto.return_explanations is True
        assert dto.validate_features is False
        assert dto.save_results is False

    def test_valid_creation_with_data(self):
        """Test creating a valid detection request with inline data."""
        detector_id = uuid4()
        data = [
            {"feature1": 1.0, "feature2": 2.0},
            {"feature1": 1.5, "feature2": 2.5},
            {"feature1": 10.0, "feature2": 20.0},  # Potential anomaly
        ]

        dto = DetectionRequestDTO(
            detector_id=detector_id,
            data=data,
            threshold=0.8,
            return_scores=False,
            return_explanations=False,
        )

        assert dto.detector_id == detector_id
        assert dto.dataset_id is None
        assert dto.data == data
        assert dto.threshold == 0.8
        assert dto.return_scores is False
        assert dto.return_explanations is False

    def test_default_values(self):
        """Test default values."""
        detector_id = uuid4()
        dataset_id = uuid4()

        dto = DetectionRequestDTO(detector_id=detector_id, dataset_id=dataset_id)

        assert dto.detector_id == detector_id
        assert dto.dataset_id == dataset_id
        assert dto.data is None
        assert dto.threshold is None
        assert dto.return_scores is True
        assert dto.return_explanations is False
        assert dto.validate_features is True
        assert dto.save_results is True

    def test_threshold_validation(self):
        """Test threshold validation."""
        detector_id = uuid4()
        dataset_id = uuid4()

        # Valid range
        dto = DetectionRequestDTO(
            detector_id=detector_id, dataset_id=dataset_id, threshold=0.5
        )
        assert dto.threshold == 0.5

        # Invalid: negative
        with pytest.raises(ValidationError):
            DetectionRequestDTO(
                detector_id=detector_id, dataset_id=dataset_id, threshold=-0.1
            )

        # Invalid: greater than 1
        with pytest.raises(ValidationError):
            DetectionRequestDTO(
                detector_id=detector_id, dataset_id=dataset_id, threshold=1.1
            )

    def test_validation_either_dataset_or_data(self):
        """Test validation that either dataset_id or data must be provided."""
        detector_id = uuid4()

        # Invalid: neither dataset_id nor data
        with pytest.raises(
            ValidationError, match="Either dataset_id or data must be provided"
        ):
            DetectionRequestDTO(detector_id=detector_id)

        # Invalid: both dataset_id and data
        with pytest.raises(
            ValidationError, match="Provide either dataset_id or data, not both"
        ):
            DetectionRequestDTO(
                detector_id=detector_id, dataset_id=uuid4(), data=[{"feature": 1.0}]
            )

    def test_required_fields(self):
        """Test required fields."""
        with pytest.raises(ValidationError):
            DetectionRequestDTO(dataset_id=uuid4())

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError):
            DetectionRequestDTO(
                detector_id=uuid4(), dataset_id=uuid4(), unknown_field="value"
            )

    def test_json_schema_example(self):
        """Test that JSON schema example is valid."""
        schema = DetectionRequestDTO.model_json_schema()
        example = schema.get("example")

        assert example is not None
        assert "detector_id" in example
        assert "dataset_id" in example
        assert example["threshold"] == 0.5
        assert example["return_scores"] is True
        assert example["return_explanations"] is False


class TestTrainingRequestDTO:
    """Test suite for TrainingRequestDTO."""

    def test_valid_creation(self):
        """Test creating a valid training request."""
        detector_id = uuid4()
        dataset_id = uuid4()
        parameters = {"n_estimators": 100, "contamination": 0.1}

        dto = TrainingRequestDTO(
            detector_id=detector_id,
            dataset_id=dataset_id,
            validation_split=0.3,
            cross_validation=True,
            save_model=False,
            parameters=parameters,
        )

        assert dto.detector_id == detector_id
        assert dto.dataset_id == dataset_id
        assert dto.validation_split == 0.3
        assert dto.cross_validation is True
        assert dto.save_model is False
        assert dto.parameters == parameters

    def test_default_values(self):
        """Test default values."""
        detector_id = uuid4()
        dataset_id = uuid4()

        dto = TrainingRequestDTO(detector_id=detector_id, dataset_id=dataset_id)

        assert dto.detector_id == detector_id
        assert dto.dataset_id == dataset_id
        assert dto.validation_split is None
        assert dto.cross_validation is False
        assert dto.save_model is True
        assert dto.parameters == {}

    def test_validation_split_validation(self):
        """Test validation split validation."""
        detector_id = uuid4()
        dataset_id = uuid4()

        # Valid range
        dto = TrainingRequestDTO(
            detector_id=detector_id, dataset_id=dataset_id, validation_split=0.2
        )
        assert dto.validation_split == 0.2

        # Invalid: negative
        with pytest.raises(ValidationError):
            TrainingRequestDTO(
                detector_id=detector_id, dataset_id=dataset_id, validation_split=-0.1
            )

        # Invalid: greater than 0.5
        with pytest.raises(ValidationError):
            TrainingRequestDTO(
                detector_id=detector_id, dataset_id=dataset_id, validation_split=0.6
            )

    def test_required_fields(self):
        """Test required fields."""
        with pytest.raises(ValidationError):
            TrainingRequestDTO(detector_id=uuid4())

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError):
            TrainingRequestDTO(
                detector_id=uuid4(), dataset_id=uuid4(), unknown_field="value"
            )

    def test_json_schema_example(self):
        """Test that JSON schema example is valid."""
        schema = TrainingRequestDTO.model_json_schema()
        example = schema.get("example")

        assert example is not None
        assert "detector_id" in example
        assert "dataset_id" in example
        assert example["validation_split"] == 0.2
        assert example["cross_validation"] is False
        assert example["save_model"] is True
        assert "parameters" in example


class TestAnomalyDTO:
    """Test suite for AnomalyDTO."""

    def test_valid_creation(self):
        """Test creating a valid anomaly DTO."""
        anomaly_id = uuid4()
        timestamp = datetime.now()
        data_point = {"feature1": 10.0, "feature2": -5.0}
        metadata = {"source": "sensor_1", "batch": "batch_001"}

        dto = AnomalyDTO(
            id=anomaly_id,
            score=0.85,
            detector_name="IsolationForest",
            timestamp=timestamp,
            data_point=data_point,
            metadata=metadata,
            explanation="Feature1 value is unusually high",
            severity="high",
            confidence_lower=0.8,
            confidence_upper=0.9,
        )

        assert dto.id == anomaly_id
        assert dto.score == 0.85
        assert dto.detector_name == "IsolationForest"
        assert dto.timestamp == timestamp
        assert dto.data_point == data_point
        assert dto.metadata == metadata
        assert dto.explanation == "Feature1 value is unusually high"
        assert dto.severity == "high"
        assert dto.confidence_lower == 0.8
        assert dto.confidence_upper == 0.9

    def test_default_values(self):
        """Test default values."""
        anomaly_id = uuid4()
        timestamp = datetime.now()
        data_point = {"feature1": 1.0}

        dto = AnomalyDTO(
            id=anomaly_id,
            score=0.6,
            detector_name="OneClassSVM",
            timestamp=timestamp,
            data_point=data_point,
        )

        assert dto.metadata == {}
        assert dto.explanation is None
        assert dto.severity == "medium"
        assert dto.confidence_lower is None
        assert dto.confidence_upper is None

    def test_score_validation(self):
        """Test score validation."""
        anomaly_id = uuid4()
        timestamp = datetime.now()
        data_point = {"feature1": 1.0}

        # Valid range
        dto = AnomalyDTO(
            id=anomaly_id,
            score=0.5,
            detector_name="Test",
            timestamp=timestamp,
            data_point=data_point,
        )
        assert dto.score == 0.5

        # Invalid: negative
        with pytest.raises(ValidationError):
            AnomalyDTO(
                id=anomaly_id,
                score=-0.1,
                detector_name="Test",
                timestamp=timestamp,
                data_point=data_point,
            )

        # Invalid: greater than 1
        with pytest.raises(ValidationError):
            AnomalyDTO(
                id=anomaly_id,
                score=1.1,
                detector_name="Test",
                timestamp=timestamp,
                data_point=data_point,
            )

    def test_severity_levels(self):
        """Test different severity levels."""
        anomaly_id = uuid4()
        timestamp = datetime.now()
        data_point = {"feature1": 1.0}

        severity_levels = ["low", "medium", "high", "critical"]

        for severity in severity_levels:
            dto = AnomalyDTO(
                id=anomaly_id,
                score=0.5,
                detector_name="Test",
                timestamp=timestamp,
                data_point=data_point,
                severity=severity,
            )
            assert dto.severity == severity

    def test_required_fields(self):
        """Test required fields."""
        with pytest.raises(ValidationError):
            AnomalyDTO(score=0.5)

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError):
            AnomalyDTO(
                id=uuid4(),
                score=0.5,
                detector_name="Test",
                timestamp=datetime.now(),
                data_point={"feature1": 1.0},
                unknown_field="value",
            )


class TestDetectionResultDTO:
    """Test suite for DetectionResultDTO."""

    def test_valid_creation(self):
        """Test creating a valid detection result."""
        result_id = uuid4()
        detector_id = uuid4()
        dataset_id = uuid4()
        timestamp = datetime.now()

        anomaly1 = AnomalyDTO(
            id=uuid4(),
            score=0.9,
            detector_name="IsolationForest",
            timestamp=timestamp,
            data_point={"feature1": 10.0},
        )

        anomaly2 = AnomalyDTO(
            id=uuid4(),
            score=0.8,
            detector_name="IsolationForest",
            timestamp=timestamp,
            data_point={"feature1": 8.0},
        )

        dto = DetectionResultDTO(
            id=result_id,
            detector_id=detector_id,
            dataset_id=dataset_id,
            timestamp=timestamp,
            n_samples=1000,
            n_anomalies=2,
            anomaly_rate=0.002,
            threshold=0.7,
            execution_time_ms=1500.0,
            metadata={"version": "1.0"},
            anomalies=[anomaly1, anomaly2],
            predictions=[0, 0, 1, 0, 1],
            scores=[0.1, 0.2, 0.9, 0.3, 0.8],
            score_statistics={"mean": 0.46, "std": 0.35},
            has_confidence_intervals=True,
            quality_warnings=["Low confidence in some predictions"],
        )

        assert dto.id == result_id
        assert dto.detector_id == detector_id
        assert dto.dataset_id == dataset_id
        assert dto.timestamp == timestamp
        assert dto.n_samples == 1000
        assert dto.n_anomalies == 2
        assert dto.anomaly_rate == 0.002
        assert dto.threshold == 0.7
        assert dto.execution_time_ms == 1500.0
        assert dto.metadata == {"version": "1.0"}
        assert len(dto.anomalies) == 2
        assert dto.predictions == [0, 0, 1, 0, 1]
        assert dto.scores == [0.1, 0.2, 0.9, 0.3, 0.8]
        assert dto.score_statistics == {"mean": 0.46, "std": 0.35}
        assert dto.has_confidence_intervals is True
        assert dto.quality_warnings == ["Low confidence in some predictions"]

    def test_default_values(self):
        """Test default values."""
        result_id = uuid4()
        detector_id = uuid4()
        dataset_id = uuid4()
        timestamp = datetime.now()

        dto = DetectionResultDTO(
            id=result_id,
            detector_id=detector_id,
            dataset_id=dataset_id,
            timestamp=timestamp,
            n_samples=100,
            n_anomalies=5,
            anomaly_rate=0.05,
            threshold=0.5,
        )

        assert dto.execution_time_ms is None
        assert dto.metadata == {}
        assert dto.anomalies == []
        assert dto.predictions is None
        assert dto.scores is None
        assert dto.score_statistics == {}
        assert dto.has_confidence_intervals is False
        assert dto.quality_warnings == []

    def test_anomaly_rate_validation(self):
        """Test anomaly rate validation."""
        result_id = uuid4()
        detector_id = uuid4()
        dataset_id = uuid4()
        timestamp = datetime.now()

        # Valid range
        dto = DetectionResultDTO(
            id=result_id,
            detector_id=detector_id,
            dataset_id=dataset_id,
            timestamp=timestamp,
            n_samples=100,
            n_anomalies=5,
            anomaly_rate=0.05,
            threshold=0.5,
        )
        assert dto.anomaly_rate == 0.05

        # Invalid: negative
        with pytest.raises(ValidationError):
            DetectionResultDTO(
                id=result_id,
                detector_id=detector_id,
                dataset_id=dataset_id,
                timestamp=timestamp,
                n_samples=100,
                n_anomalies=5,
                anomaly_rate=-0.1,
                threshold=0.5,
            )

        # Invalid: greater than 1
        with pytest.raises(ValidationError):
            DetectionResultDTO(
                id=result_id,
                detector_id=detector_id,
                dataset_id=dataset_id,
                timestamp=timestamp,
                n_samples=100,
                n_anomalies=5,
                anomaly_rate=1.1,
                threshold=0.5,
            )

    def test_consistency_checks(self):
        """Test consistency between n_samples, n_anomalies, and anomaly_rate."""
        result_id = uuid4()
        detector_id = uuid4()
        dataset_id = uuid4()
        timestamp = datetime.now()

        # Consistent values
        dto = DetectionResultDTO(
            id=result_id,
            detector_id=detector_id,
            dataset_id=dataset_id,
            timestamp=timestamp,
            n_samples=1000,
            n_anomalies=50,
            anomaly_rate=0.05,
            threshold=0.5,
        )

        # Manual check of consistency
        expected_rate = dto.n_anomalies / dto.n_samples
        assert abs(dto.anomaly_rate - expected_rate) < 0.001

    def test_required_fields(self):
        """Test required fields."""
        with pytest.raises(ValidationError):
            DetectionResultDTO(detector_id=uuid4(), dataset_id=uuid4())

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError):
            DetectionResultDTO(
                id=uuid4(),
                detector_id=uuid4(),
                dataset_id=uuid4(),
                timestamp=datetime.now(),
                n_samples=100,
                n_anomalies=5,
                anomaly_rate=0.05,
                threshold=0.5,
                unknown_field="value",
            )

    def test_json_schema_example(self):
        """Test that JSON schema example is valid."""
        schema = DetectionResultDTO.model_json_schema()
        example = schema.get("example")

        assert example is not None
        assert "id" in example
        assert "detector_id" in example
        assert "dataset_id" in example
        assert example["n_samples"] == 1000
        assert example["n_anomalies"] == 45
        assert example["anomaly_rate"] == 0.045
        assert "score_statistics" in example


class TestTrainingResultDTO:
    """Test suite for TrainingResultDTO."""

    def test_valid_creation(self):
        """Test creating a valid training result."""
        detector_id = uuid4()
        dataset_id = uuid4()
        timestamp = datetime.now()

        training_metrics = {"accuracy": 0.95, "precision": 0.88}
        validation_metrics = {"accuracy": 0.92, "precision": 0.85}
        dataset_summary = {"n_samples": 1000, "n_features": 10}
        parameters_used = {"n_estimators": 100, "contamination": 0.1}

        dto = TrainingResultDTO(
            detector_id=detector_id,
            dataset_id=dataset_id,
            timestamp=timestamp,
            training_time_ms=5000.0,
            model_path="/models/detector_123.pkl",
            training_warnings=["Low sample count"],
            training_metrics=training_metrics,
            validation_metrics=validation_metrics,
            dataset_summary=dataset_summary,
            parameters_used=parameters_used,
        )

        assert dto.detector_id == detector_id
        assert dto.dataset_id == dataset_id
        assert dto.timestamp == timestamp
        assert dto.training_time_ms == 5000.0
        assert dto.model_path == "/models/detector_123.pkl"
        assert dto.training_warnings == ["Low sample count"]
        assert dto.training_metrics == training_metrics
        assert dto.validation_metrics == validation_metrics
        assert dto.dataset_summary == dataset_summary
        assert dto.parameters_used == parameters_used

    def test_default_values(self):
        """Test default values."""
        detector_id = uuid4()
        dataset_id = uuid4()
        timestamp = datetime.now()

        dto = TrainingResultDTO(
            detector_id=detector_id,
            dataset_id=dataset_id,
            timestamp=timestamp,
            training_time_ms=2000.0,
        )

        assert dto.model_path is None
        assert dto.training_warnings == []
        assert dto.training_metrics == {}
        assert dto.validation_metrics is None
        assert dto.dataset_summary == {}
        assert dto.parameters_used == {}

    def test_required_fields(self):
        """Test required fields."""
        with pytest.raises(ValidationError):
            TrainingResultDTO(detector_id=uuid4(), dataset_id=uuid4())

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError):
            TrainingResultDTO(
                detector_id=uuid4(),
                dataset_id=uuid4(),
                timestamp=datetime.now(),
                training_time_ms=1000.0,
                unknown_field="value",
            )


class TestExplanationRequestDTO:
    """Test suite for ExplanationRequestDTO."""

    def test_valid_creation(self):
        """Test creating a valid explanation request."""
        detector_id = uuid4()
        instance = {"feature1": 10.0, "feature2": -5.0}
        feature_names = ["feature1", "feature2"]

        dto = ExplanationRequestDTO(
            detector_id=detector_id,
            instance=instance,
            method="lime",
            feature_names=feature_names,
            n_features=5,
        )

        assert dto.detector_id == detector_id
        assert dto.instance == instance
        assert dto.method == "lime"
        assert dto.feature_names == feature_names
        assert dto.n_features == 5

    def test_default_values(self):
        """Test default values."""
        detector_id = uuid4()
        instance = {"feature1": 1.0}

        dto = ExplanationRequestDTO(detector_id=detector_id, instance=instance)

        assert dto.method == "shap"
        assert dto.feature_names is None
        assert dto.n_features == 10

    def test_method_validation(self):
        """Test method validation."""
        detector_id = uuid4()
        instance = {"feature1": 1.0}

        # Valid methods
        valid_methods = ["shap", "lime"]
        for method in valid_methods:
            dto = ExplanationRequestDTO(
                detector_id=detector_id, instance=instance, method=method
            )
            assert dto.method == method

        # Invalid method
        with pytest.raises(ValidationError):
            ExplanationRequestDTO(
                detector_id=detector_id, instance=instance, method="invalid_method"
            )

    def test_n_features_validation(self):
        """Test n_features validation."""
        detector_id = uuid4()
        instance = {"feature1": 1.0}

        # Valid range
        dto = ExplanationRequestDTO(
            detector_id=detector_id, instance=instance, n_features=25
        )
        assert dto.n_features == 25

        # Invalid: too small
        with pytest.raises(ValidationError):
            ExplanationRequestDTO(
                detector_id=detector_id, instance=instance, n_features=0
            )

        # Invalid: too large
        with pytest.raises(ValidationError):
            ExplanationRequestDTO(
                detector_id=detector_id, instance=instance, n_features=100
            )

    def test_required_fields(self):
        """Test required fields."""
        with pytest.raises(ValidationError):
            ExplanationRequestDTO(detector_id=uuid4())

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError):
            ExplanationRequestDTO(
                detector_id=uuid4(), instance={"feature1": 1.0}, unknown_field="value"
            )


class TestExplanationResultDTO:
    """Test suite for ExplanationResultDTO."""

    def test_valid_creation(self):
        """Test creating a valid explanation result."""
        feature_importance = {"feature1": 0.6, "feature2": 0.4}
        visualization_data = {"plot_type": "bar", "data": [1, 2, 3]}

        dto = ExplanationResultDTO(
            method_used="shap",
            prediction=0.85,
            confidence=0.92,
            feature_importance=feature_importance,
            explanation_text="Feature1 contributes most to the anomaly score",
            visualization_data=visualization_data,
        )

        assert dto.method_used == "shap"
        assert dto.prediction == 0.85
        assert dto.confidence == 0.92
        assert dto.feature_importance == feature_importance
        assert dto.explanation_text == "Feature1 contributes most to the anomaly score"
        assert dto.visualization_data == visualization_data

    def test_default_values(self):
        """Test default values."""
        dto = ExplanationResultDTO(method_used="lime", prediction=0.7)

        assert dto.confidence is None
        assert dto.feature_importance == {}
        assert dto.explanation_text is None
        assert dto.visualization_data is None

    def test_required_fields(self):
        """Test required fields."""
        with pytest.raises(ValidationError):
            ExplanationResultDTO(method_used="shap")

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError):
            ExplanationResultDTO(
                method_used="shap", prediction=0.8, unknown_field="value"
            )


class TestDetectionSummaryDTO:
    """Test suite for DetectionSummaryDTO."""

    def test_valid_creation(self):
        """Test creating a valid detection summary."""
        top_algorithms = [
            {"name": "IsolationForest", "count": 100},
            {"name": "OneClassSVM", "count": 50},
        ]
        performance_metrics = {"avg_accuracy": 0.92, "avg_precision": 0.88}

        dto = DetectionSummaryDTO(
            total_detections=1000,
            recent_detections=50,
            average_anomaly_rate=0.05,
            most_active_detector="detector_123",
            top_algorithms=top_algorithms,
            performance_metrics=performance_metrics,
        )

        assert dto.total_detections == 1000
        assert dto.recent_detections == 50
        assert dto.average_anomaly_rate == 0.05
        assert dto.most_active_detector == "detector_123"
        assert dto.top_algorithms == top_algorithms
        assert dto.performance_metrics == performance_metrics

    def test_default_values(self):
        """Test default values."""
        dto = DetectionSummaryDTO(
            total_detections=100, recent_detections=10, average_anomaly_rate=0.02
        )

        assert dto.most_active_detector is None
        assert dto.top_algorithms == []
        assert dto.performance_metrics == {}

    def test_required_fields(self):
        """Test required fields."""
        with pytest.raises(ValidationError):
            DetectionSummaryDTO(total_detections=100)

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError):
            DetectionSummaryDTO(
                total_detections=100,
                recent_detections=10,
                average_anomaly_rate=0.02,
                unknown_field="value",
            )


class TestDetectionDTOIntegration:
    """Test integration scenarios for detection DTOs."""

    def test_complete_detection_workflow(self):
        """Test complete detection workflow."""
        # Create detection request
        detector_id = uuid4()
        dataset_id = uuid4()

        detection_request = DetectionRequestDTO(
            detector_id=detector_id,
            dataset_id=dataset_id,
            threshold=0.7,
            return_scores=True,
            return_explanations=True,
        )

        # Create detection result
        result_id = uuid4()
        timestamp = datetime.now()

        anomaly1 = AnomalyDTO(
            id=uuid4(),
            score=0.9,
            detector_name="IsolationForest",
            timestamp=timestamp,
            data_point={"feature1": 10.0, "feature2": -5.0},
            severity="high",
            explanation="Feature1 value is extremely high",
        )

        anomaly2 = AnomalyDTO(
            id=uuid4(),
            score=0.8,
            detector_name="IsolationForest",
            timestamp=timestamp,
            data_point={"feature1": 8.0, "feature2": -3.0},
            severity="medium",
        )

        detection_result = DetectionResultDTO(
            id=result_id,
            detector_id=detector_id,
            dataset_id=dataset_id,
            timestamp=timestamp,
            n_samples=1000,
            n_anomalies=2,
            anomaly_rate=0.002,
            threshold=0.7,
            execution_time_ms=1500.0,
            anomalies=[anomaly1, anomaly2],
            predictions=[0] * 998 + [1, 1],
            scores=[0.1] * 998 + [0.9, 0.8],
            score_statistics={"min": 0.1, "max": 0.9, "mean": 0.12, "std": 0.08},
        )

        # Verify workflow
        assert detection_result.detector_id == detection_request.detector_id
        assert detection_result.dataset_id == detection_request.dataset_id
        assert detection_result.threshold == detection_request.threshold
        assert len(detection_result.anomalies) == 2
        assert detection_result.n_anomalies == 2
        assert detection_result.anomaly_rate == 0.002
        assert len(detection_result.predictions) == 1000
        assert len(detection_result.scores) == 1000

    def test_training_workflow(self):
        """Test training workflow."""
        detector_id = uuid4()
        dataset_id = uuid4()

        # Create training request
        training_request = TrainingRequestDTO(
            detector_id=detector_id,
            dataset_id=dataset_id,
            validation_split=0.2,
            cross_validation=True,
            save_model=True,
            parameters={"n_estimators": 100, "contamination": 0.1},
        )

        # Create training result
        timestamp = datetime.now()

        training_result = TrainingResultDTO(
            detector_id=detector_id,
            dataset_id=dataset_id,
            timestamp=timestamp,
            training_time_ms=10000.0,
            model_path="/models/detector_123.pkl",
            training_warnings=[],
            training_metrics={
                "training_accuracy": 0.95,
                "training_precision": 0.88,
                "training_recall": 0.92,
            },
            validation_metrics={
                "validation_accuracy": 0.92,
                "validation_precision": 0.85,
                "validation_recall": 0.89,
            },
            dataset_summary={
                "n_samples": 8000,
                "n_features": 15,
                "n_training_samples": 6400,
                "n_validation_samples": 1600,
            },
            parameters_used=training_request.parameters,
        )

        # Verify workflow
        assert training_result.detector_id == training_request.detector_id
        assert training_result.dataset_id == training_request.dataset_id
        assert training_result.parameters_used == training_request.parameters
        assert training_result.model_path is not None
        assert training_result.training_metrics["training_accuracy"] == 0.95
        assert training_result.validation_metrics["validation_accuracy"] == 0.92

    def test_explanation_workflow(self):
        """Test explanation workflow."""
        detector_id = uuid4()
        instance = {"feature1": 10.0, "feature2": -5.0, "feature3": 2.0}

        # Create explanation request
        explanation_request = ExplanationRequestDTO(
            detector_id=detector_id,
            instance=instance,
            method="shap",
            feature_names=["feature1", "feature2", "feature3"],
            n_features=3,
        )

        # Create explanation result
        explanation_result = ExplanationResultDTO(
            method_used="shap",
            prediction=0.85,
            confidence=0.92,
            feature_importance={"feature1": 0.6, "feature2": 0.3, "feature3": 0.1},
            explanation_text="Feature1 (value: 10.0) contributes most to the high anomaly score",
            visualization_data={
                "plot_type": "waterfall",
                "features": ["feature1", "feature2", "feature3"],
                "values": [0.6, 0.3, 0.1],
                "base_value": 0.1,
            },
        )

        # Verify workflow
        assert explanation_result.method_used == explanation_request.method
        assert explanation_result.prediction == 0.85
        assert len(explanation_result.feature_importance) == 3
        assert explanation_result.feature_importance["feature1"] == 0.6
        assert explanation_result.explanation_text is not None
        assert explanation_result.visualization_data["plot_type"] == "waterfall"

    def test_detection_summary_workflow(self):
        """Test detection summary workflow."""
        # Create detection summary
        summary = DetectionSummaryDTO(
            total_detections=5000,
            recent_detections=150,
            average_anomaly_rate=0.03,
            most_active_detector="isolation_forest_v2",
            top_algorithms=[
                {"name": "IsolationForest", "count": 2000, "success_rate": 0.95},
                {"name": "OneClassSVM", "count": 1500, "success_rate": 0.88},
                {"name": "LocalOutlierFactor", "count": 1000, "success_rate": 0.82},
                {"name": "EllipticEnvelope", "count": 500, "success_rate": 0.79},
            ],
            performance_metrics={
                "avg_accuracy": 0.91,
                "avg_precision": 0.86,
                "avg_recall": 0.84,
                "avg_f1_score": 0.85,
                "avg_execution_time_ms": 1200.0,
            },
        )

        # Verify summary
        assert summary.total_detections == 5000
        assert summary.recent_detections == 150
        assert summary.average_anomaly_rate == 0.03
        assert summary.most_active_detector == "isolation_forest_v2"
        assert len(summary.top_algorithms) == 4
        assert summary.top_algorithms[0]["name"] == "IsolationForest"
        assert summary.performance_metrics["avg_accuracy"] == 0.91

    def test_dto_serialization(self):
        """Test DTO serialization and deserialization."""
        # Create detection request
        detector_id = uuid4()
        dataset_id = uuid4()

        original_request = DetectionRequestDTO(
            detector_id=detector_id,
            dataset_id=dataset_id,
            threshold=0.6,
            return_scores=True,
            return_explanations=False,
        )

        # Serialize to dict
        request_dict = original_request.model_dump()

        assert request_dict["detector_id"] == str(detector_id)
        assert request_dict["dataset_id"] == str(dataset_id)
        assert request_dict["threshold"] == 0.6
        assert request_dict["return_scores"] is True
        assert request_dict["return_explanations"] is False

        # Deserialize from dict
        restored_request = DetectionRequestDTO.model_validate(request_dict)

        assert restored_request.detector_id == original_request.detector_id
        assert restored_request.dataset_id == original_request.dataset_id
        assert restored_request.threshold == original_request.threshold
        assert restored_request.return_scores == original_request.return_scores
        assert (
            restored_request.return_explanations == original_request.return_explanations
        )

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Zero anomalies result
        result_id = uuid4()
        detector_id = uuid4()
        dataset_id = uuid4()
        timestamp = datetime.now()

        zero_anomalies_result = DetectionResultDTO(
            id=result_id,
            detector_id=detector_id,
            dataset_id=dataset_id,
            timestamp=timestamp,
            n_samples=1000,
            n_anomalies=0,
            anomaly_rate=0.0,
            threshold=0.9,
            predictions=[0] * 1000,
            scores=[0.1] * 1000,
        )

        assert zero_anomalies_result.n_anomalies == 0
        assert zero_anomalies_result.anomaly_rate == 0.0
        assert len(zero_anomalies_result.anomalies) == 0
        assert all(p == 0 for p in zero_anomalies_result.predictions)

        # High anomaly rate result
        high_anomaly_result = DetectionResultDTO(
            id=result_id,
            detector_id=detector_id,
            dataset_id=dataset_id,
            timestamp=timestamp,
            n_samples=100,
            n_anomalies=50,
            anomaly_rate=0.5,
            threshold=0.3,
            predictions=[1] * 50 + [0] * 50,
            scores=[0.8] * 50 + [0.2] * 50,
        )

        assert high_anomaly_result.anomaly_rate == 0.5
        assert sum(high_anomaly_result.predictions) == 50

        # Perfect anomaly scores
        perfect_anomaly = AnomalyDTO(
            id=uuid4(),
            score=1.0,
            detector_name="Test",
            timestamp=timestamp,
            data_point={"feature1": 100.0},
            severity="critical",
            confidence_lower=0.99,
            confidence_upper=1.0,
        )

        assert perfect_anomaly.score == 1.0
        assert perfect_anomaly.severity == "critical"
        assert perfect_anomaly.confidence_lower == 0.99
        assert perfect_anomaly.confidence_upper == 1.0

    def test_error_handling_scenarios(self):
        """Test error handling scenarios."""
        # Training with warnings
        detector_id = uuid4()
        dataset_id = uuid4()
        timestamp = datetime.now()

        training_with_warnings = TrainingResultDTO(
            detector_id=detector_id,
            dataset_id=dataset_id,
            timestamp=timestamp,
            training_time_ms=15000.0,
            training_warnings=[
                "Dataset size is small (n=50)",
                "High feature correlation detected",
                "Possible overfitting due to high contamination rate",
            ],
            training_metrics={
                "training_accuracy": 0.99,  # Suspiciously high
                "training_precision": 0.98,
            },
            validation_metrics={
                "validation_accuracy": 0.75,  # Much lower
                "validation_precision": 0.70,
            },
        )

        assert len(training_with_warnings.training_warnings) == 3
        assert "Dataset size is small" in training_with_warnings.training_warnings[0]
        assert training_with_warnings.training_metrics["training_accuracy"] == 0.99
        assert training_with_warnings.validation_metrics["validation_accuracy"] == 0.75

        # Detection with quality warnings
        detection_with_warnings = DetectionResultDTO(
            id=uuid4(),
            detector_id=detector_id,
            dataset_id=dataset_id,
            timestamp=timestamp,
            n_samples=100,
            n_anomalies=30,
            anomaly_rate=0.3,
            threshold=0.5,
            has_confidence_intervals=False,
            quality_warnings=[
                "Low confidence predictions for 15% of samples",
                "Feature drift detected compared to training data",
                "Unusual score distribution",
            ],
        )

        assert len(detection_with_warnings.quality_warnings) == 3
        assert (
            "Low confidence predictions" in detection_with_warnings.quality_warnings[0]
        )
        assert detection_with_warnings.has_confidence_intervals is False
