"""
Comprehensive tests for detection DTOs.

This module tests all detection-related Data Transfer Objects to ensure proper validation,
serialization, and behavior across all use cases including detection requests, training,
anomaly representation, and result handling.
"""

import json
from datetime import datetime
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from monorepo.application.dto.detection_dto import (
    AnomalyDTO,
    ConfidenceInterval,
    DetectionRequestDTO,
    DetectionResultDTO,
    DetectionSummaryDTO,
    ExplanationRequestDTO,
    ExplanationResultDTO,
    TrainingRequestDTO,
    TrainingResultDTO,
)


class TestConfidenceInterval:
    """Test suite for ConfidenceInterval DTO."""

    def test_basic_creation(self):
        """Test basic confidence interval creation."""
        ci = ConfidenceInterval(lower=0.1, upper=0.9)

        assert ci.lower == 0.1
        assert ci.upper == 0.9
        assert ci.confidence_level == 0.95  # Default
        assert ci.method == "bootstrap"  # Default

    def test_complete_creation(self):
        """Test creation with all parameters."""
        ci = ConfidenceInterval(
            lower=0.05,
            upper=0.95,
            confidence_level=0.99,
            method="percentile",
        )

        assert ci.lower == 0.05
        assert ci.upper == 0.95
        assert ci.confidence_level == 0.99
        assert ci.method == "percentile"

    def test_confidence_level_validation(self):
        """Test confidence level validation bounds."""
        # Test invalid values
        with pytest.raises(ValidationError):
            ConfidenceInterval(lower=0.1, upper=0.9, confidence_level=-0.1)

        with pytest.raises(ValidationError):
            ConfidenceInterval(lower=0.1, upper=0.9, confidence_level=1.1)

        # Test valid boundary values
        for level in [0.0, 0.5, 0.95, 0.99, 1.0]:
            ci = ConfidenceInterval(lower=0.1, upper=0.9, confidence_level=level)
            assert ci.confidence_level == level

    def test_json_schema_example(self):
        """Test JSON schema example validation."""
        example = ConfidenceInterval.model_config["json_schema_extra"]["example"]

        ci = ConfidenceInterval(**example)
        assert ci.lower == example["lower"]
        assert ci.upper == example["upper"]
        assert ci.confidence_level == example["confidence_level"]
        assert ci.method == example["method"]

    def test_invalid_interval_bounds(self):
        """Test validation with invalid interval bounds."""
        # Test where lower > upper
        with pytest.raises(ValidationError, match=""):
            # Note: The DTO itself doesn't validate bounds - this would be business logic
            ConfidenceInterval(lower=0.9, upper=0.1)

    def test_serialization(self):
        """Test serialization to dict."""
        ci = ConfidenceInterval(
            lower=0.2, upper=0.8, confidence_level=0.9, method="bayesian"
        )

        data = ci.model_dump()
        expected = {
            "lower": 0.2,
            "upper": 0.8,
            "confidence_level": 0.9,
            "method": "bayesian",
        }

        assert data == expected


class TestDetectionRequestDTO:
    """Test suite for DetectionRequestDTO."""

    def test_basic_creation_with_dataset(self):
        """Test basic creation with dataset_id."""
        detector_id = uuid4()
        dataset_id = uuid4()

        request = DetectionRequestDTO(detector_id=detector_id, dataset_id=dataset_id)

        assert request.detector_id == detector_id
        assert request.dataset_id == dataset_id
        assert request.data is None
        assert request.threshold is None
        assert request.return_scores is True  # Default
        assert request.return_explanations is False  # Default
        assert request.validate_features is True  # Default
        assert request.save_results is True  # Default

    def test_creation_with_inline_data(self):
        """Test creation with inline data instead of dataset_id."""
        detector_id = uuid4()
        data = [
            {"feature1": 1.0, "feature2": 2.0},
            {"feature1": 1.5, "feature2": 2.5},
        ]

        request = DetectionRequestDTO(detector_id=detector_id, data=data)

        assert request.detector_id == detector_id
        assert request.dataset_id is None
        assert request.data == data

    def test_complete_creation(self):
        """Test creation with all parameters."""
        detector_id = uuid4()
        dataset_id = uuid4()

        request = DetectionRequestDTO(
            detector_id=detector_id,
            dataset_id=dataset_id,
            threshold=0.75,
            return_scores=False,
            return_explanations=True,
            validate_features=False,
            save_results=False,
        )

        assert request.threshold == 0.75
        assert request.return_scores is False
        assert request.return_explanations is True
        assert request.validate_features is False
        assert request.save_results is False

    def test_threshold_validation(self):
        """Test threshold validation bounds."""
        detector_id = uuid4()
        dataset_id = uuid4()

        # Test invalid thresholds
        with pytest.raises(ValidationError):
            DetectionRequestDTO(
                detector_id=detector_id, dataset_id=dataset_id, threshold=-0.1
            )

        with pytest.raises(ValidationError):
            DetectionRequestDTO(
                detector_id=detector_id, dataset_id=dataset_id, threshold=1.1
            )

        # Test valid thresholds
        for threshold in [0.0, 0.5, 1.0]:
            request = DetectionRequestDTO(
                detector_id=detector_id, dataset_id=dataset_id, threshold=threshold
            )
            assert request.threshold == threshold

    def test_post_init_validation_no_data_source(self):
        """Test post-init validation when neither dataset_id nor data is provided."""
        detector_id = uuid4()

        with pytest.raises(
            ValueError, match="Either dataset_id or data must be provided"
        ):
            DetectionRequestDTO(detector_id=detector_id)

    def test_post_init_validation_both_data_sources(self):
        """Test post-init validation when both dataset_id and data are provided."""
        detector_id = uuid4()
        dataset_id = uuid4()
        data = [{"feature1": 1.0}]

        with pytest.raises(
            ValueError, match="Provide either dataset_id or data, not both"
        ):
            DetectionRequestDTO(
                detector_id=detector_id, dataset_id=dataset_id, data=data
            )

    def test_json_schema_example(self):
        """Test JSON schema example validation."""
        example = DetectionRequestDTO.model_config["json_schema_extra"]["example"]

        request = DetectionRequestDTO(
            detector_id=UUID(example["detector_id"]),
            dataset_id=UUID(example["dataset_id"]),
            threshold=example["threshold"],
            return_scores=example["return_scores"],
            return_explanations=example["return_explanations"],
            validate_features=example["validate_features"],
            save_results=example["save_results"],
        )

        assert request.detector_id == UUID(example["detector_id"])
        assert request.dataset_id == UUID(example["dataset_id"])


class TestTrainingRequestDTO:
    """Test suite for TrainingRequestDTO."""

    def test_basic_creation(self):
        """Test basic training request creation."""
        detector_id = uuid4()
        dataset_id = uuid4()

        request = TrainingRequestDTO(detector_id=detector_id, dataset_id=dataset_id)

        assert request.detector_id == detector_id
        assert request.dataset_id == dataset_id
        assert request.validation_split is None
        assert request.cross_validation is False
        assert request.save_model is True
        assert request.parameters == {}

    def test_complete_creation(self):
        """Test creation with all parameters."""
        detector_id = uuid4()
        dataset_id = uuid4()
        parameters = {"n_estimators": 100, "max_samples": "auto", "contamination": 0.1}

        request = TrainingRequestDTO(
            detector_id=detector_id,
            dataset_id=dataset_id,
            validation_split=0.3,
            cross_validation=True,
            save_model=False,
            parameters=parameters,
        )

        assert request.validation_split == 0.3
        assert request.cross_validation is True
        assert request.save_model is False
        assert request.parameters == parameters

    def test_validation_split_bounds(self):
        """Test validation split bounds validation."""
        detector_id = uuid4()
        dataset_id = uuid4()

        # Test invalid values
        with pytest.raises(ValidationError):
            TrainingRequestDTO(
                detector_id=detector_id, dataset_id=dataset_id, validation_split=-0.1
            )

        with pytest.raises(ValidationError):
            TrainingRequestDTO(
                detector_id=detector_id, dataset_id=dataset_id, validation_split=0.6
            )

        # Test valid values
        for split in [0.0, 0.2, 0.5]:
            request = TrainingRequestDTO(
                detector_id=detector_id, dataset_id=dataset_id, validation_split=split
            )
            assert request.validation_split == split

    def test_json_schema_example(self):
        """Test JSON schema example validation."""
        example = TrainingRequestDTO.model_config["json_schema_extra"]["example"]

        request = TrainingRequestDTO(
            detector_id=UUID(example["detector_id"]),
            dataset_id=UUID(example["dataset_id"]),
            validation_split=example["validation_split"],
            cross_validation=example["cross_validation"],
            save_model=example["save_model"],
            parameters=example["parameters"],
        )

        assert request.detector_id == UUID(example["detector_id"])
        assert request.parameters == example["parameters"]


class TestAnomalyDTO:
    """Test suite for AnomalyDTO."""

    def test_basic_creation(self):
        """Test basic anomaly DTO creation."""
        anomaly_id = uuid4()
        timestamp = datetime.utcnow()
        data_point = {"feature1": 5.0, "feature2": -2.3, "feature3": "category_a"}

        anomaly = AnomalyDTO(
            id=anomaly_id,
            score=0.85,
            detector_name="isolation_forest",
            timestamp=timestamp,
            data_point=data_point,
        )

        assert anomaly.id == anomaly_id
        assert anomaly.score == 0.85
        assert anomaly.detector_name == "isolation_forest"
        assert anomaly.timestamp == timestamp
        assert anomaly.data_point == data_point
        assert anomaly.metadata == {}
        assert anomaly.explanation is None
        assert anomaly.severity == "medium"  # Default
        assert anomaly.confidence_lower is None
        assert anomaly.confidence_upper is None

    def test_complete_creation(self):
        """Test creation with all fields."""
        anomaly_id = uuid4()
        timestamp = datetime.utcnow()
        data_point = {"value": 100.5}
        metadata = {"algorithm_version": "2.1", "preprocessing": "standard"}

        anomaly = AnomalyDTO(
            id=anomaly_id,
            score=0.95,
            detector_name="one_class_svm",
            timestamp=timestamp,
            data_point=data_point,
            metadata=metadata,
            explanation="Value significantly exceeds normal range",
            severity="high",
            confidence_lower=0.92,
            confidence_upper=0.98,
        )

        assert anomaly.metadata == metadata
        assert anomaly.explanation == "Value significantly exceeds normal range"
        assert anomaly.severity == "high"
        assert anomaly.confidence_lower == 0.92
        assert anomaly.confidence_upper == 0.98

    def test_score_validation(self):
        """Test score validation bounds."""
        anomaly_id = uuid4()
        timestamp = datetime.utcnow()
        data_point = {"value": 1.0}

        # Test invalid scores
        with pytest.raises(ValidationError):
            AnomalyDTO(
                id=anomaly_id,
                score=-0.1,
                detector_name="test",
                timestamp=timestamp,
                data_point=data_point,
            )

        with pytest.raises(ValidationError):
            AnomalyDTO(
                id=anomaly_id,
                score=1.1,
                detector_name="test",
                timestamp=timestamp,
                data_point=data_point,
            )

        # Test valid scores
        for score in [0.0, 0.5, 1.0]:
            anomaly = AnomalyDTO(
                id=anomaly_id,
                score=score,
                detector_name="test",
                timestamp=timestamp,
                data_point=data_point,
            )
            assert anomaly.score == score

    def test_severity_levels(self):
        """Test different severity levels."""
        anomaly_id = uuid4()
        timestamp = datetime.utcnow()
        data_point = {"value": 1.0}

        severity_levels = ["low", "medium", "high", "critical"]

        for severity in severity_levels:
            anomaly = AnomalyDTO(
                id=anomaly_id,
                score=0.5,
                detector_name="test",
                timestamp=timestamp,
                data_point=data_point,
                severity=severity,
            )
            assert anomaly.severity == severity


class TestDetectionResultDTO:
    """Test suite for DetectionResultDTO."""

    def test_basic_creation(self):
        """Test basic detection result creation."""
        result_id = uuid4()
        detector_id = uuid4()
        dataset_id = uuid4()
        timestamp = datetime.utcnow()

        result = DetectionResultDTO(
            id=result_id,
            detector_id=detector_id,
            dataset_id=dataset_id,
            timestamp=timestamp,
            n_samples=1000,
            n_anomalies=50,
            anomaly_rate=0.05,
            threshold=0.6,
        )

        assert result.id == result_id
        assert result.detector_id == detector_id
        assert result.dataset_id == dataset_id
        assert result.timestamp == timestamp
        assert result.n_samples == 1000
        assert result.n_anomalies == 50
        assert result.anomaly_rate == 0.05
        assert result.threshold == 0.6
        assert result.execution_time_ms is None
        assert result.metadata == {}
        assert result.anomalies == []
        assert result.predictions is None
        assert result.scores is None
        assert result.score_statistics == {}
        assert result.has_confidence_intervals is False
        assert result.quality_warnings == []

    def test_complete_creation(self):
        """Test creation with all fields."""
        result_id = uuid4()
        detector_id = uuid4()
        dataset_id = uuid4()
        timestamp = datetime.utcnow()

        # Create some anomalies
        anomaly1 = AnomalyDTO(
            id=uuid4(),
            score=0.85,
            detector_name="test_detector",
            timestamp=timestamp,
            data_point={"value": 5.0},
        )
        anomaly2 = AnomalyDTO(
            id=uuid4(),
            score=0.92,
            detector_name="test_detector",
            timestamp=timestamp,
            data_point={"value": 8.0},
        )

        metadata = {"algorithm": "isolation_forest", "model_version": "1.2"}
        predictions = [0, 1, 0, 1, 0]  # 0=normal, 1=anomaly
        scores = [0.2, 0.85, 0.3, 0.92, 0.1]
        score_stats = {
            "min": 0.1,
            "max": 0.92,
            "mean": 0.474,
            "median": 0.3,
            "std": 0.377,
        }
        warnings = ["Low confidence for 2 predictions", "Missing feature normalization"]

        result = DetectionResultDTO(
            id=result_id,
            detector_id=detector_id,
            dataset_id=dataset_id,
            timestamp=timestamp,
            n_samples=5,
            n_anomalies=2,
            anomaly_rate=0.4,
            threshold=0.6,
            execution_time_ms=1500.5,
            metadata=metadata,
            anomalies=[anomaly1, anomaly2],
            predictions=predictions,
            scores=scores,
            score_statistics=score_stats,
            has_confidence_intervals=True,
            quality_warnings=warnings,
        )

        assert result.execution_time_ms == 1500.5
        assert result.metadata == metadata
        assert len(result.anomalies) == 2
        assert result.predictions == predictions
        assert result.scores == scores
        assert result.score_statistics == score_stats
        assert result.has_confidence_intervals is True
        assert result.quality_warnings == warnings

    def test_anomaly_rate_validation(self):
        """Test anomaly rate validation bounds."""
        result_id = uuid4()
        detector_id = uuid4()
        dataset_id = uuid4()
        timestamp = datetime.utcnow()

        # Test invalid rates
        with pytest.raises(ValidationError):
            DetectionResultDTO(
                id=result_id,
                detector_id=detector_id,
                dataset_id=dataset_id,
                timestamp=timestamp,
                n_samples=100,
                n_anomalies=10,
                anomaly_rate=-0.1,  # Invalid
                threshold=0.5,
            )

        with pytest.raises(ValidationError):
            DetectionResultDTO(
                id=result_id,
                detector_id=detector_id,
                dataset_id=dataset_id,
                timestamp=timestamp,
                n_samples=100,
                n_anomalies=10,
                anomaly_rate=1.1,  # Invalid
                threshold=0.5,
            )

        # Test valid rates
        for rate in [0.0, 0.5, 1.0]:
            result = DetectionResultDTO(
                id=result_id,
                detector_id=detector_id,
                dataset_id=dataset_id,
                timestamp=timestamp,
                n_samples=100,
                n_anomalies=int(100 * rate),
                anomaly_rate=rate,
                threshold=0.5,
            )
            assert result.anomaly_rate == rate

    def test_json_schema_example(self):
        """Test JSON schema example validation."""
        example = DetectionResultDTO.model_config["json_schema_extra"]["example"]

        result = DetectionResultDTO(
            id=UUID(example["id"]),
            detector_id=UUID(example["detector_id"]),
            dataset_id=UUID(example["dataset_id"]),
            timestamp=datetime.fromisoformat(example["timestamp"]),
            n_samples=example["n_samples"],
            n_anomalies=example["n_anomalies"],
            anomaly_rate=example["anomaly_rate"],
            threshold=example["threshold"],
            execution_time_ms=example["execution_time_ms"],
            anomalies=example["anomalies"],
            score_statistics=example["score_statistics"],
        )

        assert result.n_samples == example["n_samples"]
        assert result.score_statistics == example["score_statistics"]


class TestTrainingResultDTO:
    """Test suite for TrainingResultDTO."""

    def test_basic_creation(self):
        """Test basic training result creation."""
        detector_id = uuid4()
        dataset_id = uuid4()
        timestamp = datetime.utcnow()

        result = TrainingResultDTO(
            detector_id=detector_id,
            dataset_id=dataset_id,
            timestamp=timestamp,
            training_time_ms=5000.0,
        )

        assert result.detector_id == detector_id
        assert result.dataset_id == dataset_id
        assert result.timestamp == timestamp
        assert result.training_time_ms == 5000.0
        assert result.model_path is None
        assert result.training_warnings == []
        assert result.training_metrics == {}
        assert result.validation_metrics is None
        assert result.dataset_summary == {}
        assert result.parameters_used == {}

    def test_complete_creation(self):
        """Test creation with all fields."""
        detector_id = uuid4()
        dataset_id = uuid4()
        timestamp = datetime.utcnow()

        training_metrics = {"accuracy": 0.95, "f1_score": 0.87}
        validation_metrics = {"val_accuracy": 0.92, "val_f1_score": 0.85}
        dataset_summary = {"n_samples": 1000, "n_features": 20}
        parameters = {"n_estimators": 100, "contamination": 0.1}
        warnings = ["Low validation score", "Consider more training data"]

        result = TrainingResultDTO(
            detector_id=detector_id,
            dataset_id=dataset_id,
            timestamp=timestamp,
            training_time_ms=8500.5,
            model_path="/models/detector_123.pkl",
            training_warnings=warnings,
            training_metrics=training_metrics,
            validation_metrics=validation_metrics,
            dataset_summary=dataset_summary,
            parameters_used=parameters,
        )

        assert result.model_path == "/models/detector_123.pkl"
        assert result.training_warnings == warnings
        assert result.training_metrics == training_metrics
        assert result.validation_metrics == validation_metrics
        assert result.dataset_summary == dataset_summary
        assert result.parameters_used == parameters


class TestExplanationRequestDTO:
    """Test suite for ExplanationRequestDTO."""

    def test_basic_creation(self):
        """Test basic explanation request creation."""
        detector_id = uuid4()
        instance = {"feature1": 1.5, "feature2": "category_a", "feature3": 0.8}

        request = ExplanationRequestDTO(detector_id=detector_id, instance=instance)

        assert request.detector_id == detector_id
        assert request.instance == instance
        assert request.method == "shap"  # Default
        assert request.feature_names is None
        assert request.n_features == 10  # Default

    def test_complete_creation(self):
        """Test creation with all parameters."""
        detector_id = uuid4()
        instance = {"x1": 2.0, "x2": 3.0, "x3": 1.0}
        feature_names = ["x1", "x2", "x3"]

        request = ExplanationRequestDTO(
            detector_id=detector_id,
            instance=instance,
            method="lime",
            feature_names=feature_names,
            n_features=3,
        )

        assert request.method == "lime"
        assert request.feature_names == feature_names
        assert request.n_features == 3

    def test_method_validation(self):
        """Test method validation pattern."""
        detector_id = uuid4()
        instance = {"value": 1.0}

        # Test valid methods
        for method in ["shap", "lime"]:
            request = ExplanationRequestDTO(
                detector_id=detector_id, instance=instance, method=method
            )
            assert request.method == method

        # Test invalid method
        with pytest.raises(ValidationError):
            ExplanationRequestDTO(
                detector_id=detector_id, instance=instance, method="invalid_method"
            )

    def test_n_features_validation(self):
        """Test n_features validation bounds."""
        detector_id = uuid4()
        instance = {"value": 1.0}

        # Test invalid values
        with pytest.raises(ValidationError):
            ExplanationRequestDTO(
                detector_id=detector_id, instance=instance, n_features=0
            )

        with pytest.raises(ValidationError):
            ExplanationRequestDTO(
                detector_id=detector_id, instance=instance, n_features=51
            )

        # Test valid values
        for n_features in [1, 10, 50]:
            request = ExplanationRequestDTO(
                detector_id=detector_id, instance=instance, n_features=n_features
            )
            assert request.n_features == n_features


class TestExplanationResultDTO:
    """Test suite for ExplanationResultDTO."""

    def test_basic_creation(self):
        """Test basic explanation result creation."""
        result = ExplanationResultDTO(method_used="shap", prediction=0.85)

        assert result.method_used == "shap"
        assert result.prediction == 0.85
        assert result.confidence is None
        assert result.feature_importance == {}
        assert result.explanation_text is None
        assert result.visualization_data is None

    def test_complete_creation(self):
        """Test creation with all fields."""
        feature_importance = {"feature1": 0.3, "feature2": -0.15, "feature3": 0.25}
        visualization_data = {"plot_type": "bar", "values": [0.3, -0.15, 0.25]}

        result = ExplanationResultDTO(
            method_used="lime",
            prediction=0.75,
            confidence=0.92,
            feature_importance=feature_importance,
            explanation_text="Feature1 contributed most to anomaly detection",
            visualization_data=visualization_data,
        )

        assert result.method_used == "lime"
        assert result.prediction == 0.75
        assert result.confidence == 0.92
        assert result.feature_importance == feature_importance
        assert (
            result.explanation_text == "Feature1 contributed most to anomaly detection"
        )
        assert result.visualization_data == visualization_data


class TestDetectionSummaryDTO:
    """Test suite for DetectionSummaryDTO."""

    def test_basic_creation(self):
        """Test basic detection summary creation."""
        summary = DetectionSummaryDTO(
            total_detections=1000, recent_detections=50, average_anomaly_rate=0.05
        )

        assert summary.total_detections == 1000
        assert summary.recent_detections == 50
        assert summary.average_anomaly_rate == 0.05
        assert summary.most_active_detector is None
        assert summary.top_algorithms == []
        assert summary.performance_metrics == {}

    def test_complete_creation(self):
        """Test creation with all fields."""
        top_algorithms = [
            {"name": "isolation_forest", "usage_count": 500},
            {"name": "one_class_svm", "usage_count": 300},
        ]
        performance_metrics = {
            "avg_execution_time_ms": 150.5,
            "avg_accuracy": 0.92,
            "total_processing_time_hours": 25.5,
        }

        summary = DetectionSummaryDTO(
            total_detections=2000,
            recent_detections=75,
            average_anomaly_rate=0.08,
            most_active_detector="fraud_detector_v2",
            top_algorithms=top_algorithms,
            performance_metrics=performance_metrics,
        )

        assert summary.most_active_detector == "fraud_detector_v2"
        assert summary.top_algorithms == top_algorithms
        assert summary.performance_metrics == performance_metrics


class TestDetectionDTOIntegration:
    """Integration tests for detection DTOs."""

    def test_complete_detection_workflow(self):
        """Test complete detection workflow using multiple DTOs."""
        # Step 1: Training request
        detector_id = uuid4()
        dataset_id = uuid4()

        training_request = TrainingRequestDTO(
            detector_id=detector_id,
            dataset_id=dataset_id,
            validation_split=0.2,
            parameters={"n_estimators": 100},
        )

        # Step 2: Training result
        training_result = TrainingResultDTO(
            detector_id=detector_id,
            dataset_id=dataset_id,
            timestamp=datetime.utcnow(),
            training_time_ms=5000.0,
            training_metrics={"accuracy": 0.95},
            parameters_used=training_request.parameters,
        )

        # Step 3: Detection request
        detection_request = DetectionRequestDTO(
            detector_id=detector_id, dataset_id=dataset_id, threshold=0.6
        )

        # Step 4: Create anomalies
        anomaly1 = AnomalyDTO(
            id=uuid4(),
            score=0.85,
            detector_name="test_detector",
            timestamp=datetime.utcnow(),
            data_point={"value": 5.0},
        )

        # Step 5: Detection result
        detection_result = DetectionResultDTO(
            id=uuid4(),
            detector_id=detector_id,
            dataset_id=dataset_id,
            timestamp=datetime.utcnow(),
            n_samples=100,
            n_anomalies=5,
            anomaly_rate=0.05,
            threshold=detection_request.threshold,
            anomalies=[anomaly1],
        )

        # Step 6: Explanation request for an anomaly
        explanation_request = ExplanationRequestDTO(
            detector_id=detector_id, instance=anomaly1.data_point
        )

        # Step 7: Explanation result
        explanation_result = ExplanationResultDTO(
            method_used="shap",
            prediction=anomaly1.score,
            feature_importance={"value": 0.8},
        )

        # Verify workflow consistency
        assert training_result.detector_id == training_request.detector_id
        assert detection_result.detector_id == detection_request.detector_id
        assert detection_result.threshold == detection_request.threshold
        assert explanation_result.prediction == anomaly1.score

    def test_serialization_roundtrip(self):
        """Test serialization roundtrip for complex DTOs."""
        # Create a complex detection result
        detector_id = uuid4()
        dataset_id = uuid4()
        result_id = uuid4()
        timestamp = datetime.utcnow().replace(microsecond=0)

        anomaly = AnomalyDTO(
            id=uuid4(),
            score=0.92,
            detector_name="isolation_forest",
            timestamp=timestamp,
            data_point={"feature1": 5.0, "feature2": "outlier"},
            severity="high",
        )

        original = DetectionResultDTO(
            id=result_id,
            detector_id=detector_id,
            dataset_id=dataset_id,
            timestamp=timestamp,
            n_samples=1000,
            n_anomalies=25,
            anomaly_rate=0.025,
            threshold=0.7,
            execution_time_ms=1250.5,
            anomalies=[anomaly],
            predictions=[0, 1, 0],
            scores=[0.3, 0.92, 0.15],
            score_statistics={"mean": 0.46, "std": 0.39},
        )

        # Serialize to JSON
        json_data = original.model_dump(mode="json")
        json_str = json.dumps(json_data, default=str)

        # Parse back from JSON
        parsed_data = json.loads(json_str)

        # Reconstruct with proper type conversions
        reconstructed = DetectionResultDTO(
            **{
                **parsed_data,
                "id": UUID(parsed_data["id"]),
                "detector_id": UUID(parsed_data["detector_id"]),
                "dataset_id": UUID(parsed_data["dataset_id"]),
                "timestamp": datetime.fromisoformat(parsed_data["timestamp"]),
                "anomalies": [
                    AnomalyDTO(
                        **{
                            **a,
                            "id": UUID(a["id"]),
                            "timestamp": datetime.fromisoformat(a["timestamp"]),
                        }
                    )
                    for a in parsed_data["anomalies"]
                ],
            }
        )

        assert reconstructed.id == original.id
        assert reconstructed.n_anomalies == original.n_anomalies
        assert len(reconstructed.anomalies) == len(original.anomalies)
        assert reconstructed.anomalies[0].score == original.anomalies[0].score


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
