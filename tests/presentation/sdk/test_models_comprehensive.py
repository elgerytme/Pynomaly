"""
Comprehensive test suite for Pynomaly SDK data models.

This module provides extensive testing coverage for SDK data models,
including validation, serialization, and edge case handling.
"""

import json
from datetime import datetime

import pytest
from pydantic import ValidationError

from pynomaly.presentation.sdk.models import (
    DatasetInfo,
    DetectionRequest,
    DetectionResponse,
    ErrorResponse,
    HealthStatus,
    ModelInfo,
    ModelMetrics,
    PaginatedResponse,
    TrainingRequest,
    TrainingResponse,
)


class TestDetectionRequest:
    """Test DetectionRequest model."""

    def test_detection_request_valid_data(self):
        """Test DetectionRequest with valid data."""
        request = DetectionRequest(
            data=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            algorithm="isolation_forest",
            parameters={
                "contamination": 0.1,
                "n_estimators": 100
            }
        )

        assert request.data == [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        assert request.algorithm == "isolation_forest"
        assert request.parameters["contamination"] == 0.1
        assert request.parameters["n_estimators"] == 100

    def test_detection_request_minimal_data(self):
        """Test DetectionRequest with minimal required data."""
        request = DetectionRequest(
            data=[[1, 2], [3, 4]],
            algorithm="one_class_svm"
        )

        assert request.data == [[1, 2], [3, 4]]
        assert request.algorithm == "one_class_svm"
        assert request.parameters is None

    def test_detection_request_empty_data_validation(self):
        """Test DetectionRequest validation with empty data."""
        with pytest.raises(ValidationError) as exc_info:
            DetectionRequest(
                data=[],
                algorithm="isolation_forest"
            )

        assert "Data cannot be empty" in str(exc_info.value)

    def test_detection_request_invalid_algorithm(self):
        """Test DetectionRequest validation with invalid algorithm."""
        with pytest.raises(ValidationError) as exc_info:
            DetectionRequest(
                data=[[1, 2], [3, 4]],
                algorithm="invalid_algorithm"
            )

        assert "Invalid algorithm" in str(exc_info.value)

    def test_detection_request_serialization(self):
        """Test DetectionRequest serialization."""
        request = DetectionRequest(
            data=[[1, 2], [3, 4]],
            algorithm="isolation_forest",
            parameters={"contamination": 0.1}
        )

        serialized = request.to_dict()
        assert isinstance(serialized, dict)
        assert serialized["data"] == [[1, 2], [3, 4]]
        assert serialized["algorithm"] == "isolation_forest"
        assert serialized["parameters"]["contamination"] == 0.1

    def test_detection_request_json_serialization(self):
        """Test DetectionRequest JSON serialization."""
        request = DetectionRequest(
            data=[[1, 2], [3, 4]],
            algorithm="isolation_forest"
        )

        json_str = request.to_json()
        assert isinstance(json_str, str)

        # Parse back to verify
        parsed = json.loads(json_str)
        assert parsed["data"] == [[1, 2], [3, 4]]
        assert parsed["algorithm"] == "isolation_forest"

    def test_detection_request_from_dict(self):
        """Test DetectionRequest creation from dictionary."""
        data_dict = {
            "data": [[1, 2], [3, 4]],
            "algorithm": "isolation_forest",
            "parameters": {"contamination": 0.1}
        }

        request = DetectionRequest.from_dict(data_dict)
        assert request.data == [[1, 2], [3, 4]]
        assert request.algorithm == "isolation_forest"
        assert request.parameters["contamination"] == 0.1


class TestDetectionResponse:
    """Test DetectionResponse model."""

    def test_detection_response_valid_data(self):
        """Test DetectionResponse with valid data."""
        response = DetectionResponse(
            anomaly_scores=[0.1, 0.9, 0.2, 0.8],
            anomaly_labels=[0, 1, 0, 1],
            execution_time=0.123,
            model_info={"name": "isolation_forest", "version": "1.0"},
            metadata={"n_samples": 4, "n_features": 3}
        )

        assert response.anomaly_scores == [0.1, 0.9, 0.2, 0.8]
        assert response.anomaly_labels == [0, 1, 0, 1]
        assert response.execution_time == 0.123
        assert response.model_info["name"] == "isolation_forest"
        assert response.metadata["n_samples"] == 4

    def test_detection_response_score_label_mismatch(self):
        """Test DetectionResponse validation with mismatched scores and labels."""
        with pytest.raises(ValidationError) as exc_info:
            DetectionResponse(
                anomaly_scores=[0.1, 0.9, 0.2],
                anomaly_labels=[0, 1],  # Different length
                execution_time=0.123
            )

        assert "Scores and labels must have the same length" in str(exc_info.value)

    def test_detection_response_invalid_execution_time(self):
        """Test DetectionResponse validation with invalid execution time."""
        with pytest.raises(ValidationError) as exc_info:
            DetectionResponse(
                anomaly_scores=[0.1, 0.9],
                anomaly_labels=[0, 1],
                execution_time=-0.1  # Negative time
            )

        assert "Execution time must be positive" in str(exc_info.value)

    def test_detection_response_anomaly_count(self):
        """Test DetectionResponse anomaly count calculation."""
        response = DetectionResponse(
            anomaly_scores=[0.1, 0.9, 0.2, 0.8],
            anomaly_labels=[0, 1, 0, 1],
            execution_time=0.123
        )

        assert response.anomaly_count == 2
        assert response.normal_count == 2
        assert response.anomaly_rate == 0.5

    def test_detection_response_serialization(self):
        """Test DetectionResponse serialization."""
        response = DetectionResponse(
            anomaly_scores=[0.1, 0.9],
            anomaly_labels=[0, 1],
            execution_time=0.123,
            model_info={"name": "test_model"}
        )

        serialized = response.to_dict()
        assert isinstance(serialized, dict)
        assert serialized["anomaly_scores"] == [0.1, 0.9]
        assert serialized["anomaly_labels"] == [0, 1]
        assert serialized["execution_time"] == 0.123
        assert serialized["model_info"]["name"] == "test_model"


class TestTrainingRequest:
    """Test TrainingRequest model."""

    def test_training_request_valid_data(self):
        """Test TrainingRequest with valid data."""
        request = TrainingRequest(
            data=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            algorithm="isolation_forest",
            hyperparameters={
                "n_estimators": 100,
                "contamination": 0.1,
                "max_samples": "auto"
            },
            validation_split=0.2,
            cross_validation=True
        )

        assert request.data == [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        assert request.algorithm == "isolation_forest"
        assert request.hyperparameters["n_estimators"] == 100
        assert request.validation_split == 0.2
        assert request.cross_validation is True

    def test_training_request_minimal_data(self):
        """Test TrainingRequest with minimal required data."""
        request = TrainingRequest(
            data=[[1, 2], [3, 4]],
            algorithm="one_class_svm"
        )

        assert request.data == [[1, 2], [3, 4]]
        assert request.algorithm == "one_class_svm"
        assert request.hyperparameters is None
        assert request.validation_split is None
        assert request.cross_validation is False

    def test_training_request_invalid_validation_split(self):
        """Test TrainingRequest validation with invalid validation split."""
        with pytest.raises(ValidationError) as exc_info:
            TrainingRequest(
                data=[[1, 2], [3, 4]],
                algorithm="isolation_forest",
                validation_split=1.5  # Invalid split
            )

        assert "Validation split must be between 0 and 1" in str(exc_info.value)

    def test_training_request_insufficient_data(self):
        """Test TrainingRequest validation with insufficient data."""
        with pytest.raises(ValidationError) as exc_info:
            TrainingRequest(
                data=[[1, 2]],  # Only one sample
                algorithm="isolation_forest"
            )

        assert "Insufficient data for training" in str(exc_info.value)

    def test_training_request_serialization(self):
        """Test TrainingRequest serialization."""
        request = TrainingRequest(
            data=[[1, 2], [3, 4]],
            algorithm="isolation_forest",
            hyperparameters={"n_estimators": 50}
        )

        serialized = request.to_dict()
        assert isinstance(serialized, dict)
        assert serialized["data"] == [[1, 2], [3, 4]]
        assert serialized["algorithm"] == "isolation_forest"
        assert serialized["hyperparameters"]["n_estimators"] == 50


class TestTrainingResponse:
    """Test TrainingResponse model."""

    def test_training_response_valid_data(self):
        """Test TrainingResponse with valid data."""
        response = TrainingResponse(
            job_id="train-123",
            status="started",
            model_id="model-456",
            estimated_duration=300,
            progress=0,
            metrics={"accuracy": 0.95},
            created_at=datetime.now()
        )

        assert response.job_id == "train-123"
        assert response.status == "started"
        assert response.model_id == "model-456"
        assert response.estimated_duration == 300
        assert response.progress == 0
        assert response.metrics["accuracy"] == 0.95

    def test_training_response_completed_status(self):
        """Test TrainingResponse with completed status."""
        response = TrainingResponse(
            job_id="train-123",
            status="completed",
            model_id="model-456",
            progress=100,
            metrics={"accuracy": 0.95, "f1_score": 0.92},
            training_time=250
        )

        assert response.status == "completed"
        assert response.progress == 100
        assert response.is_completed
        assert response.training_time == 250

    def test_training_response_failed_status(self):
        """Test TrainingResponse with failed status."""
        response = TrainingResponse(
            job_id="train-123",
            status="failed",
            error_message="Training failed due to insufficient memory",
            progress=45
        )

        assert response.status == "failed"
        assert response.is_failed
        assert response.error_message == "Training failed due to insufficient memory"
        assert response.progress == 45

    def test_training_response_invalid_progress(self):
        """Test TrainingResponse validation with invalid progress."""
        with pytest.raises(ValidationError) as exc_info:
            TrainingResponse(
                job_id="train-123",
                status="running",
                progress=150  # Invalid progress
            )

        assert "Progress must be between 0 and 100" in str(exc_info.value)

    def test_training_response_serialization(self):
        """Test TrainingResponse serialization."""
        response = TrainingResponse(
            job_id="train-123",
            status="running",
            progress=50,
            metrics={"current_loss": 0.5}
        )

        serialized = response.to_dict()
        assert isinstance(serialized, dict)
        assert serialized["job_id"] == "train-123"
        assert serialized["status"] == "running"
        assert serialized["progress"] == 50
        assert serialized["metrics"]["current_loss"] == 0.5


class TestDatasetInfo:
    """Test DatasetInfo model."""

    def test_dataset_info_valid_data(self):
        """Test DatasetInfo with valid data."""
        info = DatasetInfo(
            dataset_id="dataset-123",
            name="test_dataset",
            description="A test dataset for anomaly detection",
            size=1000,
            features=5,
            file_format="csv",
            upload_date=datetime.now(),
            tags=["test", "anomaly", "sample"]
        )

        assert info.dataset_id == "dataset-123"
        assert info.name == "test_dataset"
        assert info.description == "A test dataset for anomaly detection"
        assert info.size == 1000
        assert info.features == 5
        assert info.file_format == "csv"
        assert "test" in info.tags

    def test_dataset_info_minimal_data(self):
        """Test DatasetInfo with minimal required data."""
        info = DatasetInfo(
            dataset_id="dataset-123",
            name="minimal_dataset",
            size=100,
            features=3
        )

        assert info.dataset_id == "dataset-123"
        assert info.name == "minimal_dataset"
        assert info.size == 100
        assert info.features == 3
        assert info.description is None
        assert info.tags is None

    def test_dataset_info_invalid_size(self):
        """Test DatasetInfo validation with invalid size."""
        with pytest.raises(ValidationError) as exc_info:
            DatasetInfo(
                dataset_id="dataset-123",
                name="test_dataset",
                size=0,  # Invalid size
                features=3
            )

        assert "Size must be positive" in str(exc_info.value)

    def test_dataset_info_invalid_features(self):
        """Test DatasetInfo validation with invalid features."""
        with pytest.raises(ValidationError) as exc_info:
            DatasetInfo(
                dataset_id="dataset-123",
                name="test_dataset",
                size=100,
                features=0  # Invalid features
            )

        assert "Features must be positive" in str(exc_info.value)

    def test_dataset_info_serialization(self):
        """Test DatasetInfo serialization."""
        info = DatasetInfo(
            dataset_id="dataset-123",
            name="test_dataset",
            size=100,
            features=3,
            tags=["test"]
        )

        serialized = info.to_dict()
        assert isinstance(serialized, dict)
        assert serialized["dataset_id"] == "dataset-123"
        assert serialized["name"] == "test_dataset"
        assert serialized["size"] == 100
        assert serialized["features"] == 3
        assert serialized["tags"] == ["test"]


class TestModelInfo:
    """Test ModelInfo model."""

    def test_model_info_valid_data(self):
        """Test ModelInfo with valid data."""
        info = ModelInfo(
            model_id="model-123",
            name="test_model",
            algorithm="isolation_forest",
            hyperparameters={"n_estimators": 100, "contamination": 0.1},
            metrics={"accuracy": 0.95, "precision": 0.93},
            created_at=datetime.now(),
            version="1.0",
            status="active"
        )

        assert info.model_id == "model-123"
        assert info.name == "test_model"
        assert info.algorithm == "isolation_forest"
        assert info.hyperparameters["n_estimators"] == 100
        assert info.metrics["accuracy"] == 0.95
        assert info.version == "1.0"
        assert info.status == "active"

    def test_model_info_minimal_data(self):
        """Test ModelInfo with minimal required data."""
        info = ModelInfo(
            model_id="model-123",
            name="minimal_model",
            algorithm="one_class_svm"
        )

        assert info.model_id == "model-123"
        assert info.name == "minimal_model"
        assert info.algorithm == "one_class_svm"
        assert info.hyperparameters is None
        assert info.metrics is None

    def test_model_info_invalid_algorithm(self):
        """Test ModelInfo validation with invalid algorithm."""
        with pytest.raises(ValidationError) as exc_info:
            ModelInfo(
                model_id="model-123",
                name="test_model",
                algorithm="invalid_algorithm"
            )

        assert "Invalid algorithm" in str(exc_info.value)

    def test_model_info_serialization(self):
        """Test ModelInfo serialization."""
        info = ModelInfo(
            model_id="model-123",
            name="test_model",
            algorithm="isolation_forest",
            metrics={"accuracy": 0.95}
        )

        serialized = info.to_dict()
        assert isinstance(serialized, dict)
        assert serialized["model_id"] == "model-123"
        assert serialized["name"] == "test_model"
        assert serialized["algorithm"] == "isolation_forest"
        assert serialized["metrics"]["accuracy"] == 0.95


class TestHealthStatus:
    """Test HealthStatus model."""

    def test_health_status_healthy(self):
        """Test HealthStatus with healthy status."""
        status = HealthStatus(
            status="healthy",
            version="1.0.0",
            uptime=3600,
            timestamp=datetime.now(),
            services={
                "database": {"status": "healthy", "response_time": 0.05},
                "cache": {"status": "healthy", "response_time": 0.01}
            }
        )

        assert status.status == "healthy"
        assert status.version == "1.0.0"
        assert status.uptime == 3600
        assert status.is_healthy
        assert status.services["database"]["status"] == "healthy"

    def test_health_status_unhealthy(self):
        """Test HealthStatus with unhealthy status."""
        status = HealthStatus(
            status="unhealthy",
            version="1.0.0",
            timestamp=datetime.now(),
            services={
                "database": {"status": "unhealthy", "error": "Connection failed"}
            }
        )

        assert status.status == "unhealthy"
        assert not status.is_healthy
        assert status.services["database"]["status"] == "unhealthy"

    def test_health_status_invalid_status(self):
        """Test HealthStatus validation with invalid status."""
        with pytest.raises(ValidationError) as exc_info:
            HealthStatus(
                status="invalid_status",
                version="1.0.0"
            )

        assert "Invalid status" in str(exc_info.value)

    def test_health_status_serialization(self):
        """Test HealthStatus serialization."""
        status = HealthStatus(
            status="healthy",
            version="1.0.0",
            uptime=3600
        )

        serialized = status.to_dict()
        assert isinstance(serialized, dict)
        assert serialized["status"] == "healthy"
        assert serialized["version"] == "1.0.0"
        assert serialized["uptime"] == 3600


class TestErrorResponse:
    """Test ErrorResponse model."""

    def test_error_response_valid_data(self):
        """Test ErrorResponse with valid data."""
        error = ErrorResponse(
            error_code="VALIDATION_ERROR",
            message="Invalid input data",
            details={"field": "data", "reason": "empty"},
            timestamp=datetime.now()
        )

        assert error.error_code == "VALIDATION_ERROR"
        assert error.message == "Invalid input data"
        assert error.details["field"] == "data"
        assert error.details["reason"] == "empty"

    def test_error_response_minimal_data(self):
        """Test ErrorResponse with minimal required data."""
        error = ErrorResponse(
            error_code="GENERIC_ERROR",
            message="An error occurred"
        )

        assert error.error_code == "GENERIC_ERROR"
        assert error.message == "An error occurred"
        assert error.details is None

    def test_error_response_serialization(self):
        """Test ErrorResponse serialization."""
        error = ErrorResponse(
            error_code="TEST_ERROR",
            message="Test error message",
            details={"test": "value"}
        )

        serialized = error.to_dict()
        assert isinstance(serialized, dict)
        assert serialized["error_code"] == "TEST_ERROR"
        assert serialized["message"] == "Test error message"
        assert serialized["details"]["test"] == "value"


class TestPaginatedResponse:
    """Test PaginatedResponse model."""

    def test_paginated_response_valid_data(self):
        """Test PaginatedResponse with valid data."""
        items = [{"id": 1, "name": "item1"}, {"id": 2, "name": "item2"}]
        response = PaginatedResponse(
            items=items,
            total=100,
            page=1,
            per_page=10,
            has_next=True,
            has_prev=False
        )

        assert response.items == items
        assert response.total == 100
        assert response.page == 1
        assert response.per_page == 10
        assert response.has_next is True
        assert response.has_prev is False
        assert response.total_pages == 10

    def test_paginated_response_edge_cases(self):
        """Test PaginatedResponse edge cases."""
        # Last page
        response = PaginatedResponse(
            items=[{"id": 1}],
            total=21,
            page=3,
            per_page=10,
            has_next=False,
            has_prev=True
        )

        assert response.total_pages == 3
        assert response.has_next is False
        assert response.has_prev is True

    def test_paginated_response_invalid_page(self):
        """Test PaginatedResponse validation with invalid page."""
        with pytest.raises(ValidationError) as exc_info:
            PaginatedResponse(
                items=[],
                total=100,
                page=0,  # Invalid page
                per_page=10
            )

        assert "Page must be positive" in str(exc_info.value)

    def test_paginated_response_serialization(self):
        """Test PaginatedResponse serialization."""
        response = PaginatedResponse(
            items=[{"id": 1, "name": "item1"}],
            total=1,
            page=1,
            per_page=10
        )

        serialized = response.to_dict()
        assert isinstance(serialized, dict)
        assert serialized["items"] == [{"id": 1, "name": "item1"}]
        assert serialized["total"] == 1
        assert serialized["page"] == 1
        assert serialized["per_page"] == 10
        assert serialized["total_pages"] == 1


class TestModelMetrics:
    """Test ModelMetrics model."""

    def test_model_metrics_valid_data(self):
        """Test ModelMetrics with valid data."""
        metrics = ModelMetrics(
            model_id="model-123",
            accuracy=0.95,
            precision=0.93,
            recall=0.91,
            f1_score=0.92,
            roc_auc=0.94,
            confusion_matrix=[[450, 50], [30, 470]],
            feature_importance={"feature1": 0.4, "feature2": 0.3}
        )

        assert metrics.model_id == "model-123"
        assert metrics.accuracy == 0.95
        assert metrics.precision == 0.93
        assert metrics.recall == 0.91
        assert metrics.f1_score == 0.92
        assert metrics.roc_auc == 0.94
        assert metrics.confusion_matrix == [[450, 50], [30, 470]]
        assert metrics.feature_importance["feature1"] == 0.4

    def test_model_metrics_invalid_accuracy(self):
        """Test ModelMetrics validation with invalid accuracy."""
        with pytest.raises(ValidationError) as exc_info:
            ModelMetrics(
                model_id="model-123",
                accuracy=1.5  # Invalid accuracy
            )

        assert "Accuracy must be between 0 and 1" in str(exc_info.value)

    def test_model_metrics_serialization(self):
        """Test ModelMetrics serialization."""
        metrics = ModelMetrics(
            model_id="model-123",
            accuracy=0.95,
            precision=0.93
        )

        serialized = metrics.to_dict()
        assert isinstance(serialized, dict)
        assert serialized["model_id"] == "model-123"
        assert serialized["accuracy"] == 0.95
        assert serialized["precision"] == 0.93


class TestComplexModelValidation:
    """Test complex model validation scenarios."""

    def test_nested_model_validation(self):
        """Test validation of nested models."""
        response = DetectionResponse(
            anomaly_scores=[0.1, 0.9],
            anomaly_labels=[0, 1],
            execution_time=0.123,
            model_info=ModelInfo(
                model_id="model-123",
                name="nested_model",
                algorithm="isolation_forest"
            )
        )

        assert isinstance(response.model_info, ModelInfo)
        assert response.model_info.model_id == "model-123"

    def test_model_serialization_with_datetime(self):
        """Test model serialization with datetime fields."""
        now = datetime.now()
        info = DatasetInfo(
            dataset_id="dataset-123",
            name="datetime_dataset",
            size=100,
            features=3,
            upload_date=now
        )

        serialized = info.to_dict()
        assert isinstance(serialized, dict)
        assert isinstance(serialized["upload_date"], str)  # Should be ISO format

    def test_model_deserialization_from_json(self):
        """Test model deserialization from JSON."""
        json_data = {
            "anomaly_scores": [0.1, 0.9],
            "anomaly_labels": [0, 1],
            "execution_time": 0.123,
            "model_info": {"name": "test_model", "version": "1.0"}
        }

        response = DetectionResponse.from_dict(json_data)
        assert response.anomaly_scores == [0.1, 0.9]
        assert response.anomaly_labels == [0, 1]
        assert response.execution_time == 0.123
        assert response.model_info["name"] == "test_model"

    def test_model_partial_update(self):
        """Test model partial update functionality."""
        info = ModelInfo(
            model_id="model-123",
            name="original_model",
            algorithm="isolation_forest"
        )

        # Update with new data
        updated_info = info.copy(update={
            "name": "updated_model",
            "version": "2.0"
        })

        assert updated_info.model_id == "model-123"  # Unchanged
        assert updated_info.name == "updated_model"  # Updated
        assert updated_info.algorithm == "isolation_forest"  # Unchanged
        assert updated_info.version == "2.0"  # New field

    def test_model_validation_with_custom_validators(self):
        """Test model validation with custom validators."""
        # Test custom validation for data consistency
        with pytest.raises(ValidationError) as exc_info:
            DetectionResponse(
                anomaly_scores=[0.1, 0.9, 0.2],
                anomaly_labels=[0, 1],  # Length mismatch
                execution_time=0.123
            )

        assert "length mismatch" in str(exc_info.value).lower()


# Test fixtures
@pytest.fixture
def sample_detection_request():
    """Create a sample detection request."""
    return DetectionRequest(
        data=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        algorithm="isolation_forest",
        parameters={"contamination": 0.1}
    )


@pytest.fixture
def sample_detection_response():
    """Create a sample detection response."""
    return DetectionResponse(
        anomaly_scores=[0.1, 0.9, 0.2],
        anomaly_labels=[0, 1, 0],
        execution_time=0.123,
        model_info={"name": "test_model", "version": "1.0"}
    )


@pytest.fixture
def sample_training_request():
    """Create a sample training request."""
    return TrainingRequest(
        data=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        algorithm="isolation_forest",
        hyperparameters={"n_estimators": 100}
    )


@pytest.fixture
def sample_training_response():
    """Create a sample training response."""
    return TrainingResponse(
        job_id="train-123",
        status="started",
        model_id="model-456",
        estimated_duration=300,
        progress=0
    )


@pytest.fixture
def sample_dataset_info():
    """Create a sample dataset info."""
    return DatasetInfo(
        dataset_id="dataset-123",
        name="sample_dataset",
        size=1000,
        features=5,
        file_format="csv"
    )


@pytest.fixture
def sample_model_info():
    """Create a sample model info."""
    return ModelInfo(
        model_id="model-123",
        name="sample_model",
        algorithm="isolation_forest",
        hyperparameters={"n_estimators": 100},
        metrics={"accuracy": 0.95}
    )
