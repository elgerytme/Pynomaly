"""
Comprehensive tests for detector DTOs.

This module tests all detector-related Data Transfer Objects to ensure proper validation,
serialization, and behavior across all use cases including detector creation, updates,
and detection requests.
"""

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

    def test_basic_creation(self):
        """Test basic detector DTO creation."""
        detector_id = uuid4()
        created_at = datetime.utcnow()

        dto = DetectorDTO(
            id=detector_id,
            name="test_detector",
            algorithm_name="isolation_forest",
            contamination_rate=0.1,
            is_fitted=False,
            created_at=created_at,
        )

        assert dto.id == detector_id
        assert dto.name == "test_detector"
        assert dto.algorithm_name == "isolation_forest"
        assert dto.contamination_rate == 0.1
        assert dto.is_fitted is False
        assert dto.created_at == created_at
        assert dto.trained_at is None  # Default
        assert dto.parameters == {}  # Default
        assert dto.metadata == {}  # Default
        assert dto.requires_fitting is True  # Default
        assert dto.supports_streaming is False  # Default
        assert dto.supports_multivariate is True  # Default
        assert dto.time_complexity is None  # Default
        assert dto.space_complexity is None  # Default

    def test_complete_creation(self):
        """Test detector DTO creation with all fields."""
        detector_id = uuid4()
        created_at = datetime.utcnow()
        trained_at = datetime.utcnow()
        parameters = {"n_estimators": 100, "max_samples": "auto"}
        metadata = {"version": "1.0", "author": "data_team"}

        dto = DetectorDTO(
            id=detector_id,
            name="advanced_detector",
            algorithm_name="one_class_svm",
            contamination_rate=0.05,
            is_fitted=True,
            created_at=created_at,
            trained_at=trained_at,
            parameters=parameters,
            metadata=metadata,
            requires_fitting=False,
            supports_streaming=True,
            supports_multivariate=False,
            time_complexity="O(n²)",
            space_complexity="O(n)",
        )

        assert dto.trained_at == trained_at
        assert dto.parameters == parameters
        assert dto.metadata == metadata
        assert dto.requires_fitting is False
        assert dto.supports_streaming is True
        assert dto.supports_multivariate is False
        assert dto.time_complexity == "O(n²)"
        assert dto.space_complexity == "O(n)"

    def test_contamination_rate_validation(self):
        """Test contamination rate validation bounds."""
        detector_id = uuid4()
        created_at = datetime.utcnow()

        # Test invalid values
        with pytest.raises(ValidationError):
            DetectorDTO(
                id=detector_id,
                name="test",
                algorithm_name="test",
                contamination_rate=-0.1,  # Invalid
                is_fitted=False,
                created_at=created_at,
            )

        with pytest.raises(ValidationError):
            DetectorDTO(
                id=detector_id,
                name="test",
                algorithm_name="test",
                contamination_rate=1.1,  # Invalid
                is_fitted=False,
                created_at=created_at,
            )

        # Test valid boundary values
        for rate in [0.0, 0.5, 1.0]:
            dto = DetectorDTO(
                id=detector_id,
                name="test",
                algorithm_name="test",
                contamination_rate=rate,
                is_fitted=False,
                created_at=created_at,
            )
            assert dto.contamination_rate == rate

    def test_fitted_detector_with_training_time(self):
        """Test fitted detector with training timestamp."""
        detector_id = uuid4()
        created_at = datetime.utcnow()
        trained_at = datetime.utcnow()

        dto = DetectorDTO(
            id=detector_id,
            name="fitted_detector",
            algorithm_name="isolation_forest",
            contamination_rate=0.1,
            is_fitted=True,
            created_at=created_at,
            trained_at=trained_at,
        )

        assert dto.is_fitted is True
        assert dto.trained_at == trained_at
        assert dto.trained_at >= dto.created_at


class TestCreateDetectorDTO:
    """Test suite for CreateDetectorDTO."""

    def test_minimal_creation(self):
        """Test creation with minimal required fields."""
        dto = CreateDetectorDTO(
            name="minimal_detector",
            algorithm_name="isolation_forest",
        )

        assert dto.name == "minimal_detector"
        assert dto.algorithm_name == "isolation_forest"
        assert dto.contamination_rate == 0.1  # Default
        assert dto.parameters == {}  # Default
        assert dto.metadata == {}  # Default

    def test_complete_creation(self):
        """Test creation with all fields."""
        parameters = {"n_estimators": 200, "contamination": 0.05}
        metadata = {"description": "Custom detector for fraud detection"}

        dto = CreateDetectorDTO(
            name="complete_detector",
            algorithm_name="one_class_svm",
            contamination_rate=0.15,
            parameters=parameters,
            metadata=metadata,
        )

        assert dto.name == "complete_detector"
        assert dto.algorithm_name == "one_class_svm"
        assert dto.contamination_rate == 0.15
        assert dto.parameters == parameters
        assert dto.metadata == metadata

    def test_name_validation(self):
        """Test name field validation."""
        # Test minimum length
        with pytest.raises(ValidationError):
            CreateDetectorDTO(name="", algorithm_name="test")

        # Test maximum length
        long_name = "x" * 101
        with pytest.raises(ValidationError):
            CreateDetectorDTO(name=long_name, algorithm_name="test")

        # Test valid name lengths
        valid_names = ["a", "x" * 100, "detector_123"]
        for name in valid_names:
            dto = CreateDetectorDTO(name=name, algorithm_name="test")
            assert dto.name == name

    def test_name_whitespace_validation(self):
        """Test name whitespace validation."""
        # Test empty name (handled by Pydantic min_length)
        with pytest.raises(ValidationError):
            CreateDetectorDTO(name="", algorithm_name="test")

        # Test whitespace-only names
        whitespace_names = ["   ", "\t\n", "  \t  "]
        for name in whitespace_names:
            with pytest.raises(ValueError, match="Name cannot be empty or whitespace only"):
                CreateDetectorDTO(name=name, algorithm_name="test")

        # Test name with leading/trailing whitespace gets trimmed
        dto = CreateDetectorDTO(name="  detector_name  ", algorithm_name="test")
        assert dto.name == "detector_name"

    def test_algorithm_name_validation(self):
        """Test algorithm name validation."""
        # Test minimum length
        with pytest.raises(ValidationError):
            CreateDetectorDTO(name="test", algorithm_name="")

        # Test valid algorithm names
        valid_algorithms = ["isolation_forest", "one_class_svm", "local_outlier_factor"]
        for algorithm in valid_algorithms:
            dto = CreateDetectorDTO(name="test", algorithm_name=algorithm)
            assert dto.algorithm_name == algorithm

    def test_contamination_rate_validation(self):
        """Test contamination rate validation bounds."""
        # Test invalid values
        with pytest.raises(ValidationError):
            CreateDetectorDTO(
                name="test",
                algorithm_name="test",
                contamination_rate=-0.1
            )

        with pytest.raises(ValidationError):
            CreateDetectorDTO(
                name="test",
                algorithm_name="test",
                contamination_rate=1.1
            )

        # Test valid values
        for rate in [0.0, 0.1, 0.5, 1.0]:
            dto = CreateDetectorDTO(
                name="test",
                algorithm_name="test",
                contamination_rate=rate
            )
            assert dto.contamination_rate == rate

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError):
            CreateDetectorDTO(
                name="test",
                algorithm_name="test",
                extra_field="not_allowed"  # type: ignore
            )


class TestUpdateDetectorDTO:
    """Test suite for UpdateDetectorDTO."""

    def test_empty_update(self):
        """Test update DTO with no fields to update."""
        dto = UpdateDetectorDTO()

        assert dto.name is None
        assert dto.contamination_rate is None
        assert dto.parameters is None
        assert dto.metadata is None

    def test_partial_update(self):
        """Test update DTO with some fields."""
        dto = UpdateDetectorDTO(
            name="updated_name",
            contamination_rate=0.2,
        )

        assert dto.name == "updated_name"
        assert dto.contamination_rate == 0.2
        assert dto.parameters is None
        assert dto.metadata is None

    def test_complete_update(self):
        """Test update DTO with all fields."""
        parameters = {"updated_param": "value"}
        metadata = {"updated": True}

        dto = UpdateDetectorDTO(
            name="fully_updated",
            contamination_rate=0.3,
            parameters=parameters,
            metadata=metadata,
        )

        assert dto.name == "fully_updated"
        assert dto.contamination_rate == 0.3
        assert dto.parameters == parameters
        assert dto.metadata == metadata

    def test_name_validation(self):
        """Test name validation in update DTO."""
        # Test minimum length
        with pytest.raises(ValidationError):
            UpdateDetectorDTO(name="")

        # Test maximum length
        long_name = "x" * 101
        with pytest.raises(ValidationError):
            UpdateDetectorDTO(name=long_name)

        # Test valid names
        valid_names = ["a", "updated_detector", "x" * 100]
        for name in valid_names:
            dto = UpdateDetectorDTO(name=name)
            assert dto.name == name

    def test_contamination_rate_validation(self):
        """Test contamination rate validation in update DTO."""
        # Test invalid values
        with pytest.raises(ValidationError):
            UpdateDetectorDTO(contamination_rate=-0.1)

        with pytest.raises(ValidationError):
            UpdateDetectorDTO(contamination_rate=1.1)

        # Test valid values
        for rate in [0.0, 0.25, 1.0]:
            dto = UpdateDetectorDTO(contamination_rate=rate)
            assert dto.contamination_rate == rate


class TestDetectorResponseDTO:
    """Test suite for DetectorResponseDTO."""

    def test_basic_creation(self):
        """Test basic detector response creation."""
        detector_id = uuid4()
        created_at = datetime.utcnow()

        dto = DetectorResponseDTO(
            id=detector_id,
            name="response_detector",
            algorithm_name="isolation_forest",
            contamination_rate=0.1,
            is_fitted=True,
            created_at=created_at,
        )

        assert dto.id == detector_id
        assert dto.name == "response_detector"
        assert dto.algorithm_name == "isolation_forest"
        assert dto.contamination_rate == 0.1
        assert dto.is_fitted is True
        assert dto.created_at == created_at
        assert dto.trained_at is None  # Default
        assert dto.parameters == {}  # Default
        assert dto.metadata == {}  # Default
        assert dto.status == "active"  # Default
        assert dto.version == "1.0.0"  # Default

    def test_complete_creation(self):
        """Test detector response creation with all fields."""
        detector_id = uuid4()
        created_at = datetime.utcnow()
        trained_at = datetime.utcnow()
        parameters = {"n_estimators": 150}
        metadata = {"model_path": "/models/detector.pkl"}

        dto = DetectorResponseDTO(
            id=detector_id,
            name="complete_response",
            algorithm_name="one_class_svm",
            contamination_rate=0.05,
            is_fitted=True,
            created_at=created_at,
            trained_at=trained_at,
            parameters=parameters,
            metadata=metadata,
            status="trained",
            version="2.1.0",
        )

        assert dto.trained_at == trained_at
        assert dto.parameters == parameters
        assert dto.metadata == metadata
        assert dto.status == "trained"
        assert dto.version == "2.1.0"

    def test_different_statuses(self):
        """Test different detector statuses."""
        detector_id = uuid4()
        created_at = datetime.utcnow()

        statuses = ["active", "inactive", "training", "failed", "deprecated"]

        for status in statuses:
            dto = DetectorResponseDTO(
                id=detector_id,
                name="test",
                algorithm_name="test",
                contamination_rate=0.1,
                is_fitted=False,
                created_at=created_at,
                status=status,
            )
            assert dto.status == status

    def test_version_formats(self):
        """Test different version formats."""
        detector_id = uuid4()
        created_at = datetime.utcnow()

        versions = ["1.0.0", "2.1.3", "0.1.0-beta", "3.0.0-rc.1"]

        for version in versions:
            dto = DetectorResponseDTO(
                id=detector_id,
                name="test",
                algorithm_name="test",
                contamination_rate=0.1,
                is_fitted=False,
                created_at=created_at,
                version=version,
            )
            assert dto.version == version


class TestDetectionRequestDTO:
    """Test suite for DetectionRequestDTO."""

    def test_basic_creation(self):
        """Test basic detection request creation."""
        detector_id = uuid4()
        data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]

        dto = DetectionRequestDTO(
            detector_id=detector_id,
            data=data,
        )

        assert dto.detector_id == detector_id
        assert dto.data == data
        assert dto.return_scores is True  # Default
        assert dto.return_feature_importance is False  # Default
        assert dto.threshold is None  # Default

    def test_complete_creation(self):
        """Test detection request creation with all fields."""
        detector_id = uuid4()
        data = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]

        dto = DetectionRequestDTO(
            detector_id=detector_id,
            data=data,
            return_scores=False,
            return_feature_importance=True,
            threshold=0.75,
        )

        assert dto.detector_id == detector_id
        assert dto.data == data
        assert dto.return_scores is False
        assert dto.return_feature_importance is True
        assert dto.threshold == 0.75

    def test_single_sample_data(self):
        """Test detection request with single sample."""
        detector_id = uuid4()
        data = [[1.0, 2.0, 3.0, 4.0]]  # Single sample

        dto = DetectionRequestDTO(
            detector_id=detector_id,
            data=data,
        )

        assert len(dto.data) == 1
        assert len(dto.data[0]) == 4

    def test_multiple_samples_data(self):
        """Test detection request with multiple samples."""
        detector_id = uuid4()
        data = [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
        ]

        dto = DetectionRequestDTO(
            detector_id=detector_id,
            data=data,
        )

        assert len(dto.data) == 4
        assert all(len(sample) == 2 for sample in dto.data)

    def test_empty_data(self):
        """Test detection request with empty data."""
        detector_id = uuid4()
        data = []  # Empty data

        dto = DetectionRequestDTO(
            detector_id=detector_id,
            data=data,
        )

        assert dto.data == []

    def test_different_feature_dimensions(self):
        """Test detection request with different feature dimensions."""
        detector_id = uuid4()

        # Test 1D features
        data_1d = [[1.0], [2.0], [3.0]]
        dto_1d = DetectionRequestDTO(detector_id=detector_id, data=data_1d)
        assert all(len(sample) == 1 for sample in dto_1d.data)

        # Test high-dimensional features
        data_high_dim = [[float(i) for i in range(100)] for _ in range(5)]
        dto_high_dim = DetectionRequestDTO(detector_id=detector_id, data=data_high_dim)
        assert all(len(sample) == 100 for sample in dto_high_dim.data)

    def test_threshold_values(self):
        """Test different threshold values."""
        detector_id = uuid4()
        data = [[1.0, 2.0]]

        # Test various threshold values
        thresholds = [0.0, 0.25, 0.5, 0.75, 1.0]

        for threshold in thresholds:
            dto = DetectionRequestDTO(
                detector_id=detector_id,
                data=data,
                threshold=threshold,
            )
            assert dto.threshold == threshold

    def test_boolean_flags(self):
        """Test different combinations of boolean flags."""
        detector_id = uuid4()
        data = [[1.0, 2.0]]

        flag_combinations = [
            (True, True),
            (True, False),
            (False, True),
            (False, False),
        ]

        for return_scores, return_importance in flag_combinations:
            dto = DetectionRequestDTO(
                detector_id=detector_id,
                data=data,
                return_scores=return_scores,
                return_feature_importance=return_importance,
            )
            assert dto.return_scores == return_scores
            assert dto.return_feature_importance == return_importance


class TestDetectorDTOIntegration:
    """Integration tests for detector DTOs."""

    def test_detector_lifecycle_workflow(self):
        """Test complete detector lifecycle using multiple DTOs."""
        # Step 1: Create detector
        create_request = CreateDetectorDTO(
            name="lifecycle_detector",
            algorithm_name="isolation_forest",
            contamination_rate=0.1,
            parameters={"n_estimators": 100, "random_state": 42},
            metadata={"project": "fraud_detection", "version": "1.0"},
        )

        # Step 2: Simulate created detector
        detector_id = uuid4()
        created_at = datetime.utcnow()

        detector = DetectorDTO(
            id=detector_id,
            name=create_request.name,
            algorithm_name=create_request.algorithm_name,
            contamination_rate=create_request.contamination_rate,
            is_fitted=False,
            created_at=created_at,
            parameters=create_request.parameters,
            metadata=create_request.metadata,
        )

        # Step 3: Update detector
        update_request = UpdateDetectorDTO(
            name="updated_lifecycle_detector",
            contamination_rate=0.15,
            metadata={"project": "fraud_detection", "version": "1.1", "updated": True},
        )

        # Step 4: Simulate updated detector
        updated_detector = DetectorDTO(
            id=detector.id,
            name=update_request.name or detector.name,
            algorithm_name=detector.algorithm_name,
            contamination_rate=update_request.contamination_rate or detector.contamination_rate,
            is_fitted=True,  # Assume it got trained
            created_at=detector.created_at,
            trained_at=datetime.utcnow(),
            parameters=update_request.parameters or detector.parameters,
            metadata=update_request.metadata or detector.metadata,
        )

        # Step 5: Create response DTO
        response = DetectorResponseDTO(
            id=updated_detector.id,
            name=updated_detector.name,
            algorithm_name=updated_detector.algorithm_name,
            contamination_rate=updated_detector.contamination_rate,
            is_fitted=updated_detector.is_fitted,
            created_at=updated_detector.created_at,
            trained_at=updated_detector.trained_at,
            parameters=updated_detector.parameters,
            metadata=updated_detector.metadata,
            status="active",
            version="1.0.0",
        )

        # Step 6: Create detection request
        detection_data = [
            [1.5, 2.5, 3.5],
            [4.5, 5.5, 6.5],
            [7.5, 8.5, 9.5],
        ]

        detection_request = DetectionRequestDTO(
            detector_id=updated_detector.id,
            data=detection_data,
            return_scores=True,
            return_feature_importance=True,
            threshold=0.6,
        )

        # Verify workflow consistency
        assert response.name == "updated_lifecycle_detector"
        assert response.contamination_rate == 0.15
        assert response.is_fitted is True
        assert response.metadata["version"] == "1.1"
        assert response.metadata["updated"] is True
        assert detection_request.detector_id == response.id
        assert len(detection_request.data) == 3

    def test_detector_parameter_evolution(self):
        """Test detector parameter evolution through updates."""
        # Initial creation
        initial_params = {
            "n_estimators": 100,
            "max_samples": "auto",
            "contamination": 0.1,
            "random_state": 42,
        }

        create_dto = CreateDetectorDTO(
            name="param_evolution_detector",
            algorithm_name="isolation_forest",
            parameters=initial_params,
        )

        # First update - modify some parameters
        update_params_1 = {
            "n_estimators": 150,  # Changed
            "max_samples": "auto",  # Same
            "contamination": 0.15,  # Changed
            "random_state": 42,  # Same
            "new_param": "added",  # Added
        }

        update_dto_1 = UpdateDetectorDTO(
            parameters=update_params_1,
        )

        # Second update - further modifications
        update_params_2 = {
            "n_estimators": 200,  # Changed again
            "max_samples": 0.8,  # Changed
            "contamination": 0.1,  # Reverted
            "random_state": 123,  # Changed
            "new_param": "modified",  # Modified
            "another_param": True,  # Added
        }

        update_dto_2 = UpdateDetectorDTO(
            parameters=update_params_2,
        )

        # Verify parameter evolution
        assert create_dto.parameters["n_estimators"] == 100
        assert update_dto_1.parameters["n_estimators"] == 150
        assert update_dto_2.parameters["n_estimators"] == 200

        assert "new_param" not in create_dto.parameters
        assert update_dto_1.parameters["new_param"] == "added"
        assert update_dto_2.parameters["new_param"] == "modified"

        assert "another_param" not in create_dto.parameters
        assert "another_param" not in update_dto_1.parameters
        assert update_dto_2.parameters["another_param"] is True

    def test_detection_request_patterns(self):
        """Test various detection request patterns."""
        detector_id = uuid4()

        # Pattern 1: Real-time single sample detection
        single_sample_request = DetectionRequestDTO(
            detector_id=detector_id,
            data=[[5.2, 3.1, 1.4, 0.2]],
            return_scores=True,
            threshold=0.5,
        )

        # Pattern 2: Batch processing
        batch_data = [[float(i), float(i+1), float(i+2)] for i in range(100)]
        batch_request = DetectionRequestDTO(
            detector_id=detector_id,
            data=batch_data,
            return_scores=False,  # Don't need scores for batch
            return_feature_importance=False,  # Skip for performance
        )

        # Pattern 3: Detailed analysis request
        analysis_data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        analysis_request = DetectionRequestDTO(
            detector_id=detector_id,
            data=analysis_data,
            return_scores=True,
            return_feature_importance=True,
            threshold=0.75,
        )

        # Pattern 4: High-dimensional data
        high_dim_data = [[float(j) for j in range(50)] for i in range(10)]
        high_dim_request = DetectionRequestDTO(
            detector_id=detector_id,
            data=high_dim_data,
            return_scores=True,
        )

        # Verify different patterns
        assert len(single_sample_request.data) == 1
        assert len(batch_request.data) == 100
        assert analysis_request.return_feature_importance is True
        assert len(high_dim_request.data[0]) == 50

    def test_detector_validation_edge_cases(self):
        """Test detector validation edge cases."""
        # Edge case 1: Minimum contamination rate
        min_contamination_dto = CreateDetectorDTO(
            name="min_contamination",
            algorithm_name="test",
            contamination_rate=0.0,
        )
        assert min_contamination_dto.contamination_rate == 0.0

        # Edge case 2: Maximum contamination rate
        max_contamination_dto = CreateDetectorDTO(
            name="max_contamination",
            algorithm_name="test",
            contamination_rate=1.0,
        )
        assert max_contamination_dto.contamination_rate == 1.0

        # Edge case 3: Single character name
        single_char_dto = CreateDetectorDTO(
            name="x",
            algorithm_name="test",
        )
        assert single_char_dto.name == "x"

        # Edge case 4: Maximum length name
        max_length_name = "x" * 100
        max_name_dto = CreateDetectorDTO(
            name=max_length_name,
            algorithm_name="test",
        )
        assert len(max_name_dto.name) == 100

        # Edge case 5: Empty data detection request
        detector_id = uuid4()
        empty_data_request = DetectionRequestDTO(
            detector_id=detector_id,
            data=[],
        )
        assert empty_data_request.data == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
