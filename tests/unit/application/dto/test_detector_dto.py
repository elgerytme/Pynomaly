"""Tests for Detector DTOs."""

from datetime import datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from pynomaly.application.dto.detector_dto import (
    CreateDetectorDTO,
    DetectionRequestDTO,
    DetectorDTO,
    DetectorResponseDTO,
    UpdateDetectorDTO,
)


class TestDetectorDTO:
    """Test suite for DetectorDTO."""

    def test_valid_creation(self):
        """Test creating a valid detector DTO."""
        detector_id = uuid4()
        created_at = datetime.now()
        trained_at = datetime.now()
        parameters = {"n_estimators": 100, "max_features": "auto"}
        metadata = {"version": "1.0", "created_by": "user123"}

        dto = DetectorDTO(
            id=detector_id,
            name="fraud_detector",
            algorithm_name="IsolationForest",
            contamination_rate=0.1,
            is_fitted=True,
            created_at=created_at,
            trained_at=trained_at,
            parameters=parameters,
            metadata=metadata,
            requires_fitting=True,
            supports_streaming=False,
            supports_multivariate=True,
            time_complexity="O(n log n)",
            space_complexity="O(n)",
        )

        assert dto.id == detector_id
        assert dto.name == "fraud_detector"
        assert dto.algorithm_name == "IsolationForest"
        assert dto.contamination_rate == 0.1
        assert dto.is_fitted is True
        assert dto.created_at == created_at
        assert dto.trained_at == trained_at
        assert dto.parameters == parameters
        assert dto.metadata == metadata
        assert dto.requires_fitting is True
        assert dto.supports_streaming is False
        assert dto.supports_multivariate is True
        assert dto.time_complexity == "O(n log n)"
        assert dto.space_complexity == "O(n)"

    def test_default_values(self):
        """Test default values."""
        detector_id = uuid4()
        created_at = datetime.now()

        dto = DetectorDTO(
            id=detector_id,
            name="test_detector",
            algorithm_name="OneClassSVM",
            contamination_rate=0.05,
            is_fitted=False,
            created_at=created_at,
        )

        assert dto.trained_at is None
        assert dto.parameters == {}
        assert dto.metadata == {}
        assert dto.requires_fitting is True
        assert dto.supports_streaming is False
        assert dto.supports_multivariate is True
        assert dto.time_complexity is None
        assert dto.space_complexity is None

    def test_contamination_rate_validation(self):
        """Test contamination rate validation."""
        detector_id = uuid4()
        created_at = datetime.now()

        # Valid range
        dto = DetectorDTO(
            id=detector_id,
            name="test_detector",
            algorithm_name="IsolationForest",
            contamination_rate=0.2,
            is_fitted=False,
            created_at=created_at,
        )
        assert dto.contamination_rate == 0.2

        # Invalid: negative
        with pytest.raises(ValidationError):
            DetectorDTO(
                id=detector_id,
                name="test_detector",
                algorithm_name="IsolationForest",
                contamination_rate=-0.1,
                is_fitted=False,
                created_at=created_at,
            )

        # Invalid: greater than 1
        with pytest.raises(ValidationError):
            DetectorDTO(
                id=detector_id,
                name="test_detector",
                algorithm_name="IsolationForest",
                contamination_rate=1.1,
                is_fitted=False,
                created_at=created_at,
            )

    def test_fitted_unfitted_states(self):
        """Test fitted and unfitted states."""
        detector_id = uuid4()
        created_at = datetime.now()

        # Unfitted detector
        unfitted_dto = DetectorDTO(
            id=detector_id,
            name="unfitted_detector",
            algorithm_name="IsolationForest",
            contamination_rate=0.1,
            is_fitted=False,
            created_at=created_at,
            trained_at=None,
        )

        assert unfitted_dto.is_fitted is False
        assert unfitted_dto.trained_at is None

        # Fitted detector
        trained_at = datetime.now()
        fitted_dto = DetectorDTO(
            id=detector_id,
            name="fitted_detector",
            algorithm_name="IsolationForest",
            contamination_rate=0.1,
            is_fitted=True,
            created_at=created_at,
            trained_at=trained_at,
        )

        assert fitted_dto.is_fitted is True
        assert fitted_dto.trained_at == trained_at

    def test_algorithm_capabilities(self):
        """Test different algorithm capabilities."""
        detector_id = uuid4()
        created_at = datetime.now()

        # Streaming algorithm
        streaming_dto = DetectorDTO(
            id=detector_id,
            name="streaming_detector",
            algorithm_name="StreamingIsolationForest",
            contamination_rate=0.1,
            is_fitted=False,
            created_at=created_at,
            requires_fitting=False,
            supports_streaming=True,
            supports_multivariate=True,
        )

        assert streaming_dto.requires_fitting is False
        assert streaming_dto.supports_streaming is True
        assert streaming_dto.supports_multivariate is True

        # Univariate algorithm
        univariate_dto = DetectorDTO(
            id=detector_id,
            name="univariate_detector",
            algorithm_name="ZScore",
            contamination_rate=0.1,
            is_fitted=False,
            created_at=created_at,
            requires_fitting=False,
            supports_streaming=True,
            supports_multivariate=False,
        )

        assert univariate_dto.requires_fitting is False
        assert univariate_dto.supports_streaming is True
        assert univariate_dto.supports_multivariate is False

    def test_required_fields(self):
        """Test required fields."""
        with pytest.raises(ValidationError):
            DetectorDTO(name="test")

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError):
            DetectorDTO(
                id=uuid4(),
                name="test_detector",
                algorithm_name="IsolationForest",
                contamination_rate=0.1,
                is_fitted=False,
                created_at=datetime.now(),
                unknown_field="value",
            )


class TestCreateDetectorDTO:
    """Test suite for CreateDetectorDTO."""

    def test_valid_creation(self):
        """Test creating a valid create detector DTO."""
        parameters = {"n_estimators": 100, "max_features": "auto"}
        metadata = {"description": "Fraud detection model", "version": "1.0"}

        dto = CreateDetectorDTO(
            name="fraud_detector",
            algorithm_name="IsolationForest",
            contamination_rate=0.15,
            parameters=parameters,
            metadata=metadata,
        )

        assert dto.name == "fraud_detector"
        assert dto.algorithm_name == "IsolationForest"
        assert dto.contamination_rate == 0.15
        assert dto.parameters == parameters
        assert dto.metadata == metadata

    def test_default_values(self):
        """Test default values."""
        dto = CreateDetectorDTO(name="basic_detector", algorithm_name="OneClassSVM")

        assert dto.name == "basic_detector"
        assert dto.algorithm_name == "OneClassSVM"
        assert dto.contamination_rate == 0.1
        assert dto.parameters == {}
        assert dto.metadata == {}

    def test_name_validation(self):
        """Test name validation."""
        # Valid name
        dto = CreateDetectorDTO(name="valid_name", algorithm_name="IsolationForest")
        assert dto.name == "valid_name"

        # Invalid: empty name
        with pytest.raises(ValidationError):
            CreateDetectorDTO(name="", algorithm_name="IsolationForest")

        # Invalid: too long name
        with pytest.raises(ValidationError):
            CreateDetectorDTO(name="a" * 101, algorithm_name="IsolationForest")

        # Invalid: whitespace only
        with pytest.raises(ValidationError):
            CreateDetectorDTO(name="   ", algorithm_name="IsolationForest")

    def test_name_trimming(self):
        """Test name trimming."""
        dto = CreateDetectorDTO(
            name="  trimmed_name  ", algorithm_name="IsolationForest"
        )
        assert dto.name == "trimmed_name"

    def test_algorithm_name_validation(self):
        """Test algorithm name validation."""
        # Valid algorithm name
        dto = CreateDetectorDTO(name="test_detector", algorithm_name="IsolationForest")
        assert dto.algorithm_name == "IsolationForest"

        # Invalid: empty algorithm name
        with pytest.raises(ValidationError):
            CreateDetectorDTO(name="test_detector", algorithm_name="")

    def test_contamination_rate_validation(self):
        """Test contamination rate validation."""
        # Valid range
        dto = CreateDetectorDTO(
            name="test_detector",
            algorithm_name="IsolationForest",
            contamination_rate=0.25,
        )
        assert dto.contamination_rate == 0.25

        # Invalid: negative
        with pytest.raises(ValidationError):
            CreateDetectorDTO(
                name="test_detector",
                algorithm_name="IsolationForest",
                contamination_rate=-0.1,
            )

        # Invalid: greater than 1
        with pytest.raises(ValidationError):
            CreateDetectorDTO(
                name="test_detector",
                algorithm_name="IsolationForest",
                contamination_rate=1.5,
            )

    def test_parameters_handling(self):
        """Test parameters handling."""
        # Complex parameters
        complex_params = {
            "n_estimators": 200,
            "max_features": "sqrt",
            "contamination": 0.1,
            "random_state": 42,
            "bootstrap": True,
            "n_jobs": -1,
            "verbose": 0,
        }

        dto = CreateDetectorDTO(
            name="complex_detector",
            algorithm_name="IsolationForest",
            parameters=complex_params,
        )

        assert dto.parameters == complex_params
        assert dto.parameters["n_estimators"] == 200
        assert dto.parameters["max_features"] == "sqrt"
        assert dto.parameters["bootstrap"] is True

    def test_metadata_handling(self):
        """Test metadata handling."""
        metadata = {
            "description": "Production fraud detector",
            "version": "2.1.0",
            "created_by": "ml_team",
            "dataset_version": "v3.2",
            "training_date": "2024-01-15",
            "tags": ["fraud", "production", "validated"],
        }

        dto = CreateDetectorDTO(
            name="production_detector",
            algorithm_name="IsolationForest",
            metadata=metadata,
        )

        assert dto.metadata == metadata
        assert dto.metadata["description"] == "Production fraud detector"
        assert dto.metadata["version"] == "2.1.0"
        assert "fraud" in dto.metadata["tags"]

    def test_required_fields(self):
        """Test required fields."""
        with pytest.raises(ValidationError):
            CreateDetectorDTO(name="test")

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError):
            CreateDetectorDTO(
                name="test_detector",
                algorithm_name="IsolationForest",
                unknown_field="value",
            )


class TestUpdateDetectorDTO:
    """Test suite for UpdateDetectorDTO."""

    def test_valid_creation(self):
        """Test creating a valid update detector DTO."""
        parameters = {"n_estimators": 150, "max_features": "log2"}
        metadata = {"updated_by": "admin", "last_modified": "2024-01-20"}

        dto = UpdateDetectorDTO(
            name="updated_detector",
            contamination_rate=0.08,
            parameters=parameters,
            metadata=metadata,
        )

        assert dto.name == "updated_detector"
        assert dto.contamination_rate == 0.08
        assert dto.parameters == parameters
        assert dto.metadata == metadata

    def test_default_values(self):
        """Test default values."""
        dto = UpdateDetectorDTO()

        assert dto.name is None
        assert dto.contamination_rate is None
        assert dto.parameters is None
        assert dto.metadata is None

    def test_partial_updates(self):
        """Test partial updates."""
        # Update only name
        dto_name = UpdateDetectorDTO(name="new_name")
        assert dto_name.name == "new_name"
        assert dto_name.contamination_rate is None

        # Update only contamination rate
        dto_contamination = UpdateDetectorDTO(contamination_rate=0.12)
        assert dto_contamination.contamination_rate == 0.12
        assert dto_contamination.name is None

        # Update only parameters
        dto_params = UpdateDetectorDTO(parameters={"n_estimators": 200})
        assert dto_params.parameters == {"n_estimators": 200}
        assert dto_params.name is None

        # Update only metadata
        dto_metadata = UpdateDetectorDTO(metadata={"version": "2.0"})
        assert dto_metadata.metadata == {"version": "2.0"}
        assert dto_metadata.name is None

    def test_name_validation(self):
        """Test name validation."""
        # Valid name
        dto = UpdateDetectorDTO(name="valid_update")
        assert dto.name == "valid_update"

        # Invalid: empty name
        with pytest.raises(ValidationError):
            UpdateDetectorDTO(name="")

        # Invalid: too long name
        with pytest.raises(ValidationError):
            UpdateDetectorDTO(name="a" * 101)

    def test_contamination_rate_validation(self):
        """Test contamination rate validation."""
        # Valid range
        dto = UpdateDetectorDTO(contamination_rate=0.3)
        assert dto.contamination_rate == 0.3

        # Invalid: negative
        with pytest.raises(ValidationError):
            UpdateDetectorDTO(contamination_rate=-0.1)

        # Invalid: greater than 1
        with pytest.raises(ValidationError):
            UpdateDetectorDTO(contamination_rate=1.2)

    def test_parameters_update(self):
        """Test parameters update."""
        # Update with new parameters
        new_params = {"n_estimators": 300, "max_depth": 10, "min_samples_split": 5}

        dto = UpdateDetectorDTO(parameters=new_params)
        assert dto.parameters == new_params

        # Update with empty parameters
        dto_empty = UpdateDetectorDTO(parameters={})
        assert dto_empty.parameters == {}

    def test_metadata_update(self):
        """Test metadata update."""
        # Update with new metadata
        new_metadata = {
            "last_updated": "2024-01-20",
            "updated_by": "admin",
            "change_reason": "Performance optimization",
        }

        dto = UpdateDetectorDTO(metadata=new_metadata)
        assert dto.metadata == new_metadata

        # Update with empty metadata
        dto_empty = UpdateDetectorDTO(metadata={})
        assert dto_empty.metadata == {}

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError):
            UpdateDetectorDTO(name="test_detector", unknown_field="value")


class TestDetectorResponseDTO:
    """Test suite for DetectorResponseDTO."""

    def test_valid_creation(self):
        """Test creating a valid detector response DTO."""
        detector_id = uuid4()
        created_at = datetime.now()
        trained_at = datetime.now()
        parameters = {"n_estimators": 100}
        metadata = {"version": "1.0"}

        dto = DetectorResponseDTO(
            id=detector_id,
            name="response_detector",
            algorithm_name="IsolationForest",
            contamination_rate=0.1,
            is_fitted=True,
            created_at=created_at,
            trained_at=trained_at,
            parameters=parameters,
            metadata=metadata,
            status="active",
            version="2.0.0",
        )

        assert dto.id == detector_id
        assert dto.name == "response_detector"
        assert dto.algorithm_name == "IsolationForest"
        assert dto.contamination_rate == 0.1
        assert dto.is_fitted is True
        assert dto.created_at == created_at
        assert dto.trained_at == trained_at
        assert dto.parameters == parameters
        assert dto.metadata == metadata
        assert dto.status == "active"
        assert dto.version == "2.0.0"

    def test_default_values(self):
        """Test default values."""
        detector_id = uuid4()
        created_at = datetime.now()

        dto = DetectorResponseDTO(
            id=detector_id,
            name="basic_response",
            algorithm_name="OneClassSVM",
            contamination_rate=0.1,
            is_fitted=False,
            created_at=created_at,
        )

        assert dto.trained_at is None
        assert dto.parameters == {}
        assert dto.metadata == {}
        assert dto.status == "active"
        assert dto.version == "1.0.0"

    def test_status_values(self):
        """Test different status values."""
        detector_id = uuid4()
        created_at = datetime.now()

        statuses = ["active", "training", "inactive", "error", "deprecated"]

        for status in statuses:
            dto = DetectorResponseDTO(
                id=detector_id,
                name="test_detector",
                algorithm_name="IsolationForest",
                contamination_rate=0.1,
                is_fitted=False,
                created_at=created_at,
                status=status,
            )
            assert dto.status == status

    def test_version_values(self):
        """Test different version values."""
        detector_id = uuid4()
        created_at = datetime.now()

        versions = ["1.0.0", "2.1.5", "3.0.0-beta", "1.2.3-alpha.1"]

        for version in versions:
            dto = DetectorResponseDTO(
                id=detector_id,
                name="test_detector",
                algorithm_name="IsolationForest",
                contamination_rate=0.1,
                is_fitted=False,
                created_at=created_at,
                version=version,
            )
            assert dto.version == version

    def test_required_fields(self):
        """Test required fields."""
        with pytest.raises(ValidationError):
            DetectorResponseDTO(name="test")

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError):
            DetectorResponseDTO(
                id=uuid4(),
                name="test_detector",
                algorithm_name="IsolationForest",
                contamination_rate=0.1,
                is_fitted=False,
                created_at=datetime.now(),
                unknown_field="value",
            )


class TestDetectionRequestDTO:
    """Test suite for DetectionRequestDTO."""

    def test_valid_creation(self):
        """Test creating a valid detection request DTO."""
        detector_id = uuid4()
        data = [[1.0, 2.0, 3.0], [1.5, 2.5, 3.5], [10.0, 20.0, 30.0]]

        dto = DetectionRequestDTO(
            detector_id=detector_id,
            data=data,
            return_scores=True,
            return_feature_importance=True,
            threshold=0.8,
        )

        assert dto.detector_id == detector_id
        assert dto.data == data
        assert dto.return_scores is True
        assert dto.return_feature_importance is True
        assert dto.threshold == 0.8

    def test_default_values(self):
        """Test default values."""
        detector_id = uuid4()
        data = [[1.0, 2.0]]

        dto = DetectionRequestDTO(detector_id=detector_id, data=data)

        assert dto.detector_id == detector_id
        assert dto.data == data
        assert dto.return_scores is True
        assert dto.return_feature_importance is False
        assert dto.threshold is None

    def test_data_formats(self):
        """Test different data formats."""
        detector_id = uuid4()

        # Single sample
        single_sample = [[1.0, 2.0, 3.0]]
        dto_single = DetectionRequestDTO(detector_id=detector_id, data=single_sample)
        assert dto_single.data == single_sample

        # Multiple samples
        multiple_samples = [[1.0, 2.0, 3.0], [1.1, 2.1, 3.1], [1.2, 2.2, 3.2]]
        dto_multiple = DetectionRequestDTO(
            detector_id=detector_id, data=multiple_samples
        )
        assert dto_multiple.data == multiple_samples

        # High-dimensional data
        high_dim = [[i * 0.1 for i in range(100)] for _ in range(10)]
        dto_high_dim = DetectionRequestDTO(detector_id=detector_id, data=high_dim)
        assert len(dto_high_dim.data) == 10
        assert len(dto_high_dim.data[0]) == 100

    def test_empty_data(self):
        """Test empty data handling."""
        detector_id = uuid4()

        # Empty data array
        dto_empty = DetectionRequestDTO(detector_id=detector_id, data=[])
        assert dto_empty.data == []

    def test_required_fields(self):
        """Test required fields."""
        with pytest.raises(ValidationError):
            DetectionRequestDTO(detector_id=uuid4())

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError):
            DetectionRequestDTO(
                detector_id=uuid4(), data=[[1.0, 2.0]], unknown_field="value"
            )


class TestDetectorDTOIntegration:
    """Test integration scenarios for detector DTOs."""

    def test_detector_lifecycle(self):
        """Test complete detector lifecycle."""
        # Create detector
        create_dto = CreateDetectorDTO(
            name="lifecycle_detector",
            algorithm_name="IsolationForest",
            contamination_rate=0.1,
            parameters={"n_estimators": 100, "random_state": 42},
            metadata={"created_by": "test_user", "version": "1.0"},
        )

        # Simulate created detector
        detector_id = uuid4()
        created_at = datetime.now()

        detector_dto = DetectorDTO(
            id=detector_id,
            name=create_dto.name,
            algorithm_name=create_dto.algorithm_name,
            contamination_rate=create_dto.contamination_rate,
            is_fitted=False,
            created_at=created_at,
            parameters=create_dto.parameters,
            metadata=create_dto.metadata,
        )

        # Update detector
        update_dto = UpdateDetectorDTO(
            contamination_rate=0.08,
            parameters={"n_estimators": 150},
            metadata={"updated_by": "admin", "version": "1.1"},
        )

        # Simulate updated detector
        updated_detector = DetectorDTO(
            id=detector_id,
            name=detector_dto.name,
            algorithm_name=detector_dto.algorithm_name,
            contamination_rate=update_dto.contamination_rate,
            is_fitted=detector_dto.is_fitted,
            created_at=created_at,
            parameters=update_dto.parameters,
            metadata=update_dto.metadata,
        )

        # Create response
        response_dto = DetectorResponseDTO(
            id=updated_detector.id,
            name=updated_detector.name,
            algorithm_name=updated_detector.algorithm_name,
            contamination_rate=updated_detector.contamination_rate,
            is_fitted=updated_detector.is_fitted,
            created_at=updated_detector.created_at,
            parameters=updated_detector.parameters,
            metadata=updated_detector.metadata,
            status="active",
            version="1.1.0",
        )

        # Verify lifecycle
        assert response_dto.name == create_dto.name
        assert response_dto.algorithm_name == create_dto.algorithm_name
        assert response_dto.contamination_rate == update_dto.contamination_rate
        assert response_dto.parameters == update_dto.parameters
        assert response_dto.metadata == update_dto.metadata
        assert response_dto.status == "active"
        assert response_dto.version == "1.1.0"

    def test_detector_training_workflow(self):
        """Test detector training workflow."""
        # Create unfitted detector
        detector_id = uuid4()
        created_at = datetime.now()

        unfitted_detector = DetectorDTO(
            id=detector_id,
            name="training_detector",
            algorithm_name="IsolationForest",
            contamination_rate=0.1,
            is_fitted=False,
            created_at=created_at,
            parameters={"n_estimators": 100},
        )

        # Simulate training completion
        trained_at = datetime.now()

        fitted_detector = DetectorDTO(
            id=detector_id,
            name=unfitted_detector.name,
            algorithm_name=unfitted_detector.algorithm_name,
            contamination_rate=unfitted_detector.contamination_rate,
            is_fitted=True,
            created_at=created_at,
            trained_at=trained_at,
            parameters=unfitted_detector.parameters,
            metadata={"training_completed": "2024-01-15"},
        )

        # Create detection request
        data = [[1.0, 2.0, 3.0], [1.1, 2.1, 3.1], [10.0, 20.0, 30.0]]

        detection_request = DetectionRequestDTO(
            detector_id=detector_id,
            data=data,
            return_scores=True,
            return_feature_importance=True,
            threshold=0.7,
        )

        # Verify training workflow
        assert unfitted_detector.is_fitted is False
        assert unfitted_detector.trained_at is None
        assert fitted_detector.is_fitted is True
        assert fitted_detector.trained_at is not None
        assert detection_request.detector_id == detector_id
        assert len(detection_request.data) == 3

    def test_detector_response_variations(self):
        """Test different detector response variations."""
        detector_id = uuid4()
        created_at = datetime.now()

        # Active detector
        active_detector = DetectorResponseDTO(
            id=detector_id,
            name="active_detector",
            algorithm_name="IsolationForest",
            contamination_rate=0.1,
            is_fitted=True,
            created_at=created_at,
            status="active",
            version="1.0.0",
        )

        # Training detector
        training_detector = DetectorResponseDTO(
            id=detector_id,
            name="training_detector",
            algorithm_name="IsolationForest",
            contamination_rate=0.1,
            is_fitted=False,
            created_at=created_at,
            status="training",
            version="1.0.0",
        )

        # Error detector
        error_detector = DetectorResponseDTO(
            id=detector_id,
            name="error_detector",
            algorithm_name="IsolationForest",
            contamination_rate=0.1,
            is_fitted=False,
            created_at=created_at,
            status="error",
            version="1.0.0",
            metadata={"error_message": "Training failed"},
        )

        # Verify different statuses
        assert active_detector.status == "active"
        assert active_detector.is_fitted is True
        assert training_detector.status == "training"
        assert training_detector.is_fitted is False
        assert error_detector.status == "error"
        assert error_detector.metadata["error_message"] == "Training failed"

    def test_dto_serialization(self):
        """Test DTO serialization and deserialization."""
        # Create detector DTO
        detector_id = uuid4()
        created_at = datetime.now()

        original_dto = DetectorDTO(
            id=detector_id,
            name="serialization_test",
            algorithm_name="IsolationForest",
            contamination_rate=0.1,
            is_fitted=True,
            created_at=created_at,
            parameters={"n_estimators": 100},
            metadata={"version": "1.0"},
        )

        # Serialize to dict
        dto_dict = original_dto.model_dump()

        assert dto_dict["id"] == str(detector_id)
        assert dto_dict["name"] == "serialization_test"
        assert dto_dict["algorithm_name"] == "IsolationForest"
        assert dto_dict["contamination_rate"] == 0.1
        assert dto_dict["is_fitted"] is True
        assert dto_dict["parameters"] == {"n_estimators": 100}
        assert dto_dict["metadata"] == {"version": "1.0"}

        # Deserialize from dict
        restored_dto = DetectorDTO.model_validate(dto_dict)

        assert restored_dto.id == original_dto.id
        assert restored_dto.name == original_dto.name
        assert restored_dto.algorithm_name == original_dto.algorithm_name
        assert restored_dto.contamination_rate == original_dto.contamination_rate
        assert restored_dto.is_fitted == original_dto.is_fitted
        assert restored_dto.parameters == original_dto.parameters
        assert restored_dto.metadata == original_dto.metadata

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Zero contamination rate
        zero_contamination = CreateDetectorDTO(
            name="zero_contamination",
            algorithm_name="IsolationForest",
            contamination_rate=0.0,
        )
        assert zero_contamination.contamination_rate == 0.0

        # Maximum contamination rate
        max_contamination = CreateDetectorDTO(
            name="max_contamination",
            algorithm_name="IsolationForest",
            contamination_rate=1.0,
        )
        assert max_contamination.contamination_rate == 1.0

        # Single character name
        single_char = CreateDetectorDTO(name="a", algorithm_name="IsolationForest")
        assert single_char.name == "a"

        # Maximum length name
        max_length_name = CreateDetectorDTO(
            name="a" * 100, algorithm_name="IsolationForest"
        )
        assert len(max_length_name.name) == 100

        # Empty detection data
        empty_data = DetectionRequestDTO(detector_id=uuid4(), data=[])
        assert empty_data.data == []

        # Single sample detection
        single_sample = DetectionRequestDTO(detector_id=uuid4(), data=[[1.0]])
        assert len(single_sample.data) == 1
        assert len(single_sample.data[0]) == 1

    def test_algorithm_specific_configurations(self):
        """Test algorithm-specific configurations."""
        # IsolationForest configuration
        isolation_forest = CreateDetectorDTO(
            name="isolation_forest_detector",
            algorithm_name="IsolationForest",
            parameters={
                "n_estimators": 100,
                "max_samples": "auto",
                "contamination": 0.1,
                "max_features": 1.0,
                "bootstrap": False,
                "n_jobs": -1,
                "random_state": 42,
                "verbose": 0,
            },
        )

        assert isolation_forest.algorithm_name == "IsolationForest"
        assert isolation_forest.parameters["n_estimators"] == 100
        assert isolation_forest.parameters["max_samples"] == "auto"
        assert isolation_forest.parameters["bootstrap"] is False

        # OneClassSVM configuration
        one_class_svm = CreateDetectorDTO(
            name="one_class_svm_detector",
            algorithm_name="OneClassSVM",
            parameters={
                "kernel": "rbf",
                "degree": 3,
                "gamma": "scale",
                "coef0": 0.0,
                "tol": 1e-3,
                "nu": 0.5,
                "shrinking": True,
                "cache_size": 200,
                "verbose": False,
                "max_iter": -1,
            },
        )

        assert one_class_svm.algorithm_name == "OneClassSVM"
        assert one_class_svm.parameters["kernel"] == "rbf"
        assert one_class_svm.parameters["gamma"] == "scale"
        assert one_class_svm.parameters["shrinking"] is True

        # LocalOutlierFactor configuration
        lof = CreateDetectorDTO(
            name="lof_detector",
            algorithm_name="LocalOutlierFactor",
            parameters={
                "n_neighbors": 20,
                "algorithm": "auto",
                "leaf_size": 30,
                "metric": "minkowski",
                "p": 2,
                "metric_params": None,
                "contamination": 0.1,
                "novelty": True,
                "n_jobs": None,
            },
        )

        assert lof.algorithm_name == "LocalOutlierFactor"
        assert lof.parameters["n_neighbors"] == 20
        assert lof.parameters["algorithm"] == "auto"
        assert lof.parameters["novelty"] is True
