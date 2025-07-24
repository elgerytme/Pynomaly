"""Tests for data models and validation."""

import pytest
from datetime import datetime
from pydantic import ValidationError as PydanticValidationError

from anomaly_detection_sdk.models import (
    AlgorithmType,
    AnomalyData,
    DetectionResult,
    ModelInfo,
    StreamingConfig,
    ExplanationResult,
    HealthStatus,
    BatchProcessingRequest,
    TrainingRequest,
    TrainingResult
)


class TestAlgorithmType:
    """Test the AlgorithmType enum."""

    def test_all_algorithm_types(self):
        """Test all algorithm type values."""
        assert AlgorithmType.ISOLATION_FOREST == "isolation_forest"
        assert AlgorithmType.LOCAL_OUTLIER_FACTOR == "local_outlier_factor"
        assert AlgorithmType.ONE_CLASS_SVM == "one_class_svm"
        assert AlgorithmType.ELLIPTIC_ENVELOPE == "elliptic_envelope"
        assert AlgorithmType.AUTOENCODER == "autoencoder"
        assert AlgorithmType.ENSEMBLE == "ensemble"

    def test_algorithm_type_validation(self):
        """Test algorithm type validation in models."""
        # Valid algorithm
        config = StreamingConfig(algorithm=AlgorithmType.ISOLATION_FOREST)
        assert config.algorithm == AlgorithmType.ISOLATION_FOREST

        # Invalid algorithm string should raise error
        with pytest.raises(PydanticValidationError):
            StreamingConfig(algorithm="invalid_algorithm")


class TestAnomalyData:
    """Test the AnomalyData model."""

    def test_anomaly_data_creation(self):
        """Test creating AnomalyData with required fields."""
        anomaly = AnomalyData(
            index=5,
            score=0.85,
            data_point=[1.5, 2.7, 3.2]
        )
        
        assert anomaly.index == 5
        assert anomaly.score == 0.85
        assert anomaly.data_point == [1.5, 2.7, 3.2]
        assert anomaly.confidence is None
        assert anomaly.timestamp is None

    def test_anomaly_data_with_optional_fields(self):
        """Test creating AnomalyData with optional fields."""
        timestamp = datetime.now()
        anomaly = AnomalyData(
            index=3,
            score=0.92,
            data_point=[10.0, 20.0],
            confidence=0.88,
            timestamp=timestamp
        )
        
        assert anomaly.confidence == 0.88
        assert anomaly.timestamp == timestamp

    def test_anomaly_data_validation(self):
        """Test AnomalyData validation."""
        # Missing required fields
        with pytest.raises(PydanticValidationError):
            AnomalyData()

        # Invalid data types
        with pytest.raises(PydanticValidationError):
            AnomalyData(index="not_an_int", score=0.5, data_point=[1, 2])

        with pytest.raises(PydanticValidationError):
            AnomalyData(index=1, score="not_a_float", data_point=[1, 2])


class TestDetectionResult:
    """Test the DetectionResult model."""

    def test_detection_result_creation(self):
        """Test creating DetectionResult with required fields."""
        anomalies = [
            AnomalyData(index=1, score=0.8, data_point=[1, 2]),
            AnomalyData(index=3, score=0.9, data_point=[3, 4]),
        ]
        
        result = DetectionResult(
            anomalies=anomalies,
            total_points=10,
            anomaly_count=2,
            algorithm_used=AlgorithmType.ISOLATION_FOREST,
            execution_time=0.15
        )
        
        assert len(result.anomalies) == 2
        assert result.total_points == 10
        assert result.anomaly_count == 2
        assert result.algorithm_used == AlgorithmType.ISOLATION_FOREST
        assert result.execution_time == 0.15
        assert result.model_version is None
        assert result.metadata == {}

    def test_detection_result_with_optional_fields(self):
        """Test DetectionResult with optional fields."""
        result = DetectionResult(
            anomalies=[],
            total_points=5,
            anomaly_count=0,
            algorithm_used=AlgorithmType.LOCAL_OUTLIER_FACTOR,
            execution_time=0.05,
            model_version="v1.2.3",
            metadata={"param1": "value1", "param2": 42}
        )
        
        assert result.model_version == "v1.2.3"
        assert result.metadata["param1"] == "value1"
        assert result.metadata["param2"] == 42

    def test_detection_result_validation(self):
        """Test DetectionResult validation."""
        # Invalid anomaly count (should match length of anomalies list)
        anomalies = [AnomalyData(index=1, score=0.8, data_point=[1, 2])]
        
        # This should work (anomaly_count matches)
        result = DetectionResult(
            anomalies=anomalies,
            total_points=5,
            anomaly_count=1,
            algorithm_used=AlgorithmType.ISOLATION_FOREST,
            execution_time=0.1
        )
        assert result.anomaly_count == 1

        # Negative values should be invalid
        with pytest.raises(PydanticValidationError):
            DetectionResult(
                anomalies=[],
                total_points=-1,
                anomaly_count=0,
                algorithm_used=AlgorithmType.ISOLATION_FOREST,
                execution_time=0.1
            )


class TestModelInfo:
    """Test the ModelInfo model."""

    def test_model_info_creation(self):
        """Test creating ModelInfo."""
        created_at = datetime.now()
        model = ModelInfo(
            model_id="model-123",
            algorithm=AlgorithmType.ISOLATION_FOREST,
            created_at=created_at,
            training_data_size=1000,
            performance_metrics={"accuracy": 0.95, "f1_score": 0.92},
            hyperparameters={"n_estimators": 100, "contamination": 0.1},
            version="1.0",
            status="trained"
        )
        
        assert model.model_id == "model-123"
        assert model.algorithm == AlgorithmType.ISOLATION_FOREST
        assert model.created_at == created_at
        assert model.training_data_size == 1000
        assert model.performance_metrics["accuracy"] == 0.95
        assert model.hyperparameters["n_estimators"] == 100
        assert model.version == "1.0"
        assert model.status == "trained"

    def test_model_info_validation(self):
        """Test ModelInfo validation."""
        # Empty model_id should be invalid
        with pytest.raises(PydanticValidationError):
            ModelInfo(
                model_id="",
                algorithm=AlgorithmType.ISOLATION_FOREST,
                created_at=datetime.now(),
                training_data_size=100,
                performance_metrics={},
                hyperparameters={},
                version="1.0",
                status="trained"
            )


class TestStreamingConfig:
    """Test the StreamingConfig model."""

    def test_streaming_config_defaults(self):
        """Test StreamingConfig with default values."""
        config = StreamingConfig()
        
        assert config.buffer_size == 100
        assert config.detection_threshold == 0.5
        assert config.batch_size == 10
        assert config.algorithm == AlgorithmType.ISOLATION_FOREST
        assert config.auto_retrain is False

    def test_streaming_config_custom_values(self):
        """Test StreamingConfig with custom values."""
        config = StreamingConfig(
            buffer_size=50,
            detection_threshold=0.8,
            batch_size=5,
            algorithm=AlgorithmType.LOCAL_OUTLIER_FACTOR,
            auto_retrain=True
        )
        
        assert config.buffer_size == 50
        assert config.detection_threshold == 0.8
        assert config.batch_size == 5
        assert config.algorithm == AlgorithmType.LOCAL_OUTLIER_FACTOR
        assert config.auto_retrain is True

    def test_streaming_config_validation(self):
        """Test StreamingConfig validation."""
        # Negative buffer_size should be invalid
        with pytest.raises(PydanticValidationError):
            StreamingConfig(buffer_size=-1)

        # Detection threshold outside [0, 1] should be invalid
        with pytest.raises(PydanticValidationError):
            StreamingConfig(detection_threshold=1.5)

        with pytest.raises(PydanticValidationError):
            StreamingConfig(detection_threshold=-0.1)


class TestExplanationResult:
    """Test the ExplanationResult model."""

    def test_explanation_result_creation(self):
        """Test creating ExplanationResult."""
        explanation = ExplanationResult(
            anomaly_index=2,
            feature_importance={"feature_0": 0.8, "feature_1": 0.2},
            shap_values=[0.3, 0.1],
            lime_explanation={"type": "tabular", "values": [0.4, 0.2]},
            explanation_text="High value in feature_0 indicates anomaly",
            confidence=0.95
        )
        
        assert explanation.anomaly_index == 2
        assert explanation.feature_importance["feature_0"] == 0.8
        assert explanation.shap_values == [0.3, 0.1]
        assert explanation.lime_explanation["type"] == "tabular"
        assert "feature_0" in explanation.explanation_text
        assert explanation.confidence == 0.95

    def test_explanation_result_minimal(self):
        """Test ExplanationResult with minimal required fields."""
        explanation = ExplanationResult(
            anomaly_index=1,
            feature_importance={"f1": 0.9},
            explanation_text="Anomaly detected",
            confidence=0.8
        )
        
        assert explanation.shap_values is None
        assert explanation.lime_explanation is None


class TestHealthStatus:
    """Test the HealthStatus model."""

    def test_health_status_creation(self):
        """Test creating HealthStatus."""
        timestamp = datetime.now()
        health = HealthStatus(
            status="healthy",
            timestamp=timestamp,
            version="1.0.0",
            uptime=3600.5,
            components={"database": "healthy", "cache": "healthy"},
            metrics={"requests_per_second": 100, "memory_usage": "50%"}
        )
        
        assert health.status == "healthy"
        assert health.timestamp == timestamp
        assert health.version == "1.0.0"
        assert health.uptime == 3600.5
        assert health.components["database"] == "healthy"
        assert health.metrics["requests_per_second"] == 100

    def test_health_status_defaults(self):
        """Test HealthStatus with default values."""
        health = HealthStatus(
            status="healthy",
            timestamp=datetime.now(),
            version="1.0.0",
            uptime=100.0
        )
        
        assert health.components == {}
        assert health.metrics == {}


class TestBatchProcessingRequest:
    """Test the BatchProcessingRequest model."""

    def test_batch_request_creation(self):
        """Test creating BatchProcessingRequest."""
        data = [[1, 2], [3, 4], [5, 6]]
        request = BatchProcessingRequest(
            data=data,
            algorithm=AlgorithmType.ISOLATION_FOREST,
            parameters={"contamination": 0.1},
            return_explanations=True
        )
        
        assert request.data == data
        assert request.algorithm == AlgorithmType.ISOLATION_FOREST
        assert request.parameters["contamination"] == 0.1
        assert request.return_explanations is True

    def test_batch_request_defaults(self):
        """Test BatchProcessingRequest with default values."""
        data = [[1, 2], [3, 4]]
        request = BatchProcessingRequest(data=data)
        
        assert request.algorithm == AlgorithmType.ISOLATION_FOREST
        assert request.parameters == {}
        assert request.return_explanations is False

    def test_batch_request_validation(self):
        """Test BatchProcessingRequest validation."""
        # Empty data should be invalid
        with pytest.raises(PydanticValidationError):
            BatchProcessingRequest(data=[])


class TestTrainingRequest:
    """Test the TrainingRequest model."""

    def test_training_request_creation(self):
        """Test creating TrainingRequest."""
        data = [[1, 2], [3, 4], [5, 6]]
        request = TrainingRequest(
            data=data,
            algorithm=AlgorithmType.LOCAL_OUTLIER_FACTOR,
            hyperparameters={"n_neighbors": 20},
            validation_split=0.3,
            model_name="test-model"
        )
        
        assert request.data == data
        assert request.algorithm == AlgorithmType.LOCAL_OUTLIER_FACTOR
        assert request.hyperparameters["n_neighbors"] == 20
        assert request.validation_split == 0.3
        assert request.model_name == "test-model"

    def test_training_request_defaults(self):
        """Test TrainingRequest with default values."""
        data = [[1, 2], [3, 4]]
        request = TrainingRequest(
            data=data,
            algorithm=AlgorithmType.ISOLATION_FOREST
        )
        
        assert request.hyperparameters == {}
        assert request.validation_split == 0.2
        assert request.model_name is None

    def test_training_request_validation(self):
        """Test TrainingRequest validation."""
        # Invalid validation_split
        with pytest.raises(PydanticValidationError):
            TrainingRequest(
                data=[[1, 2]],
                algorithm=AlgorithmType.ISOLATION_FOREST,
                validation_split=1.5  # Should be between 0 and 1
            )


class TestTrainingResult:
    """Test the TrainingResult model."""

    def test_training_result_creation(self):
        """Test creating TrainingResult."""
        model_info = ModelInfo(
            model_id="trained-model-123",
            algorithm=AlgorithmType.ISOLATION_FOREST,
            created_at=datetime.now(),
            training_data_size=500,
            performance_metrics={"accuracy": 0.9},
            hyperparameters={"n_estimators": 100},
            version="1.0",
            status="trained"
        )
        
        result = TrainingResult(
            model_id="trained-model-123",
            training_time=45.2,
            performance_metrics={"accuracy": 0.9, "precision": 0.85},
            validation_metrics={"accuracy": 0.88, "recall": 0.82},
            model_info=model_info
        )
        
        assert result.model_id == "trained-model-123"
        assert result.training_time == 45.2
        assert result.performance_metrics["accuracy"] == 0.9
        assert result.validation_metrics["accuracy"] == 0.88
        assert result.model_info.model_id == "trained-model-123"


class TestModelSerialization:
    """Test model serialization and deserialization."""

    def test_anomaly_data_json_serialization(self):
        """Test AnomalyData JSON serialization."""
        anomaly = AnomalyData(
            index=1,
            score=0.8,
            data_point=[1.5, 2.7],
            confidence=0.9
        )
        
        # Serialize to dict
        data_dict = anomaly.model_dump()
        assert data_dict["index"] == 1
        assert data_dict["score"] == 0.8
        assert data_dict["data_point"] == [1.5, 2.7]
        assert data_dict["confidence"] == 0.9

        # Deserialize from dict
        new_anomaly = AnomalyData(**data_dict)
        assert new_anomaly.index == anomaly.index
        assert new_anomaly.score == anomaly.score
        assert new_anomaly.data_point == anomaly.data_point

    def test_detection_result_json_serialization(self):
        """Test DetectionResult JSON serialization."""
        anomalies = [AnomalyData(index=1, score=0.8, data_point=[1, 2])]
        result = DetectionResult(
            anomalies=anomalies,
            total_points=5,
            anomaly_count=1,
            algorithm_used=AlgorithmType.ISOLATION_FOREST,
            execution_time=0.1,
            metadata={"test": True}
        )
        
        # Serialize to dict
        data_dict = result.model_dump()
        assert data_dict["total_points"] == 5
        assert data_dict["algorithm_used"] == "isolation_forest"
        assert len(data_dict["anomalies"]) == 1

        # Deserialize from dict
        new_result = DetectionResult(**data_dict)
        assert new_result.total_points == result.total_points
        assert new_result.algorithm_used == result.algorithm_used

    def test_model_dump_with_exclude(self):
        """Test model serialization with field exclusion."""
        anomaly = AnomalyData(
            index=1,
            score=0.8,
            data_point=[1, 2],
            confidence=0.9
        )
        
        # Exclude confidence field
        data_dict = anomaly.model_dump(exclude={"confidence"})
        assert "confidence" not in data_dict
        assert "index" in data_dict
        assert "score" in data_dict

    def test_model_validation_on_creation(self):
        """Test that model validation occurs on creation."""
        # Valid timestamp string should be parsed
        anomaly_dict = {
            "index": 1,
            "score": 0.8,
            "data_point": [1, 2],
            "timestamp": "2023-01-01T00:00:00Z"
        }
        
        anomaly = AnomalyData(**anomaly_dict)
        assert isinstance(anomaly.timestamp, datetime)

        # Invalid timestamp should raise error
        anomaly_dict["timestamp"] = "invalid-timestamp"
        with pytest.raises(PydanticValidationError):
            AnomalyData(**anomaly_dict)