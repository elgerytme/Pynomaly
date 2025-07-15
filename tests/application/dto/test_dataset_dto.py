"""
Comprehensive tests for dataset DTOs.

This module tests all dataset-related Data Transfer Objects to ensure proper validation,
serialization, and behavior across all use cases including creation, quality reporting,
responses, and upload operations.
"""

import json
from datetime import datetime
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from pynomaly.application.dto.dataset_dto import (
    CreateDatasetDTO,
    DataQualityReportDTO,
    DatasetDTO,
    DatasetResponseDTO,
    DatasetUploadResponseDTO,
)


class TestDatasetDTO:
    """Test suite for DatasetDTO."""

    def test_basic_creation(self):
        """Test basic dataset DTO creation."""
        dataset_id = uuid4()
        created_at = datetime.utcnow()
        feature_names = ["feature1", "feature2", "feature3"]

        dto = DatasetDTO(
            id=dataset_id,
            name="test_dataset",
            shape=(1000, 3),
            n_samples=1000,
            n_features=3,
            feature_names=feature_names,
            has_target=True,
            target_column="target",
            created_at=created_at,
            memory_usage_mb=2.5,
            numeric_features=2,
            categorical_features=1,
        )

        assert dto.id == dataset_id
        assert dto.name == "test_dataset"
        assert dto.shape == (1000, 3)
        assert dto.n_samples == 1000
        assert dto.n_features == 3
        assert dto.feature_names == feature_names
        assert dto.has_target is True
        assert dto.target_column == "target"
        assert dto.created_at == created_at
        assert dto.memory_usage_mb == 2.5
        assert dto.numeric_features == 2
        assert dto.categorical_features == 1
        assert dto.metadata == {}
        assert dto.description is None

    def test_creation_with_metadata(self):
        """Test dataset DTO creation with metadata and description."""
        dataset_id = uuid4()
        metadata = {"source": "kaggle", "version": "1.0", "tags": ["classification"]}

        dto = DatasetDTO(
            id=dataset_id,
            name="test_dataset",
            shape=(500, 10),
            n_samples=500,
            n_features=10,
            feature_names=[f"feature_{i}" for i in range(10)],
            has_target=False,
            created_at=datetime.utcnow(),
            memory_usage_mb=1.2,
            numeric_features=8,
            categorical_features=2,
            metadata=metadata,
            description="Test dataset for validation",
        )

        assert dto.metadata == metadata
        assert dto.description == "Test dataset for validation"
        assert dto.has_target is False
        assert dto.target_column is None

    def test_json_schema_example(self):
        """Test that the JSON schema example is valid."""
        example = DatasetDTO.model_config["json_schema_extra"]["example"]

        # Validate that example creates a valid instance
        dto = DatasetDTO(
            id=UUID(example["id"]),
            name=example["name"],
            shape=tuple(example["shape"]),
            n_samples=example["n_samples"],
            n_features=example["n_features"],
            feature_names=example["feature_names"],
            has_target=example["has_target"],
            target_column=example["target_column"],
            created_at=datetime.fromisoformat(example["created_at"]),
            memory_usage_mb=example["memory_usage_mb"],
            numeric_features=example["numeric_features"],
            categorical_features=example["categorical_features"],
        )

        assert dto.id == UUID(example["id"])
        assert dto.name == example["name"]

    def test_invalid_uuid(self):
        """Test validation with invalid UUID."""
        with pytest.raises(ValidationError):
            DatasetDTO(
                id="invalid-uuid",  # type: ignore
                name="test",
                shape=(100, 5),
                n_samples=100,
                n_features=5,
                feature_names=["f1", "f2", "f3", "f4", "f5"],
                has_target=True,
                created_at=datetime.utcnow(),
                memory_usage_mb=1.0,
                numeric_features=5,
                categorical_features=0,
            )

    def test_negative_values_validation(self):
        """Test validation with negative values."""
        with pytest.raises(ValidationError):
            DatasetDTO(
                id=uuid4(),
                name="test",
                shape=(-100, 5),  # Invalid shape
                n_samples=100,
                n_features=5,
                feature_names=["f1", "f2", "f3", "f4", "f5"],
                has_target=True,
                created_at=datetime.utcnow(),
                memory_usage_mb=1.0,
                numeric_features=5,
                categorical_features=0,
            )

    def test_feature_count_mismatch(self):
        """Test validation when feature names don't match n_features."""
        # This should still be valid - feature_names list length
        # might not always match n_features in real scenarios
        dto = DatasetDTO(
            id=uuid4(),
            name="test",
            shape=(100, 5),
            n_samples=100,
            n_features=5,
            feature_names=["f1", "f2", "f3"],  # Only 3 names for 5 features
            has_target=True,
            created_at=datetime.utcnow(),
            memory_usage_mb=1.0,
            numeric_features=5,
            categorical_features=0,
        )

        assert len(dto.feature_names) == 3
        assert dto.n_features == 5

    def test_feature_type_counts(self):
        """Test validation of feature type counts."""
        # Test that numeric + categorical can be less than or equal to n_features
        dto = DatasetDTO(
            id=uuid4(),
            name="test",
            shape=(100, 10),
            n_samples=100,
            n_features=10,
            feature_names=[f"f{i}" for i in range(10)],
            has_target=True,
            created_at=datetime.utcnow(),
            memory_usage_mb=1.0,
            numeric_features=7,
            categorical_features=2,  # 7 + 2 = 9, leaving 1 for other types
        )

        assert dto.numeric_features + dto.categorical_features <= dto.n_features


class TestCreateDatasetDTO:
    """Test suite for CreateDatasetDTO."""

    def test_minimal_creation(self):
        """Test creation with minimal required fields."""
        dto = CreateDatasetDTO(name="minimal_dataset")

        assert dto.name == "minimal_dataset"
        assert dto.description is None
        assert dto.target_column is None
        assert dto.metadata == {}
        assert dto.file_path is None
        assert dto.file_format is None
        assert dto.delimiter == ","
        assert dto.encoding == "utf-8"
        assert dto.parse_dates is None
        assert dto.dtype is None

    def test_complete_creation(self):
        """Test creation with all fields."""
        metadata = {"project": "anomaly_detection", "owner": "data_team"}
        dtype = {"col1": "float64", "col2": "int32"}

        dto = CreateDatasetDTO(
            name="complete_dataset",
            description="A complete dataset for testing",
            target_column="anomaly",
            metadata=metadata,
            file_path="/data/test.csv",
            file_format="csv",
            delimiter=";",
            encoding="latin-1",
            parse_dates=["timestamp", "created_at"],
            dtype=dtype,
        )

        assert dto.name == "complete_dataset"
        assert dto.description == "A complete dataset for testing"
        assert dto.target_column == "anomaly"
        assert dto.metadata == metadata
        assert dto.file_path == "/data/test.csv"
        assert dto.file_format == "csv"
        assert dto.delimiter == ";"
        assert dto.encoding == "latin-1"
        assert dto.parse_dates == ["timestamp", "created_at"]
        assert dto.dtype == dtype

    def test_name_validation(self):
        """Test name field validation."""
        # Test minimum length
        with pytest.raises(ValidationError):
            CreateDatasetDTO(name="")

        # Test maximum length
        long_name = "x" * 101
        with pytest.raises(ValidationError):
            CreateDatasetDTO(name=long_name)

        # Test valid names
        valid_names = ["a", "x" * 100, "dataset_123", "My Dataset"]
        for name in valid_names:
            dto = CreateDatasetDTO(name=name)
            assert dto.name == name

    def test_description_validation(self):
        """Test description field validation."""
        # Test maximum length
        long_description = "x" * 501
        with pytest.raises(ValidationError):
            CreateDatasetDTO(name="test", description=long_description)

        # Test valid description
        valid_description = "x" * 500
        dto = CreateDatasetDTO(name="test", description=valid_description)
        assert dto.description == valid_description

    def test_file_format_validation(self):
        """Test file format validation."""
        # Test valid formats
        valid_formats = ["csv", "parquet", "json", "excel"]
        for format_type in valid_formats:
            dto = CreateDatasetDTO(name="test", file_format=format_type)
            assert dto.file_format == format_type

        # Test invalid format
        with pytest.raises(ValidationError):
            CreateDatasetDTO(name="test", file_format="invalid_format")

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError):
            CreateDatasetDTO(
                name="test",
                extra_field="not_allowed",  # type: ignore
            )

    def test_nested_metadata(self):
        """Test with complex nested metadata."""
        nested_metadata = {
            "source": {
                "type": "database",
                "connection": {"host": "localhost", "port": 5432},
                "query": "SELECT * FROM anomalies",
            },
            "preprocessing": {
                "steps": ["normalize", "handle_missing"],
                "parameters": {"normalization": "z-score"},
            },
        }

        dto = CreateDatasetDTO(name="test", metadata=nested_metadata)
        assert dto.metadata == nested_metadata


class TestDataQualityReportDTO:
    """Test suite for DataQualityReportDTO."""

    def test_basic_creation(self):
        """Test basic quality report creation."""
        dto = DataQualityReportDTO(
            quality_score=0.85,
            n_missing_values=10,
            n_duplicates=5,
            n_outliers=15,
        )

        assert dto.quality_score == 0.85
        assert dto.n_missing_values == 10
        assert dto.n_duplicates == 5
        assert dto.n_outliers == 15
        assert dto.missing_columns == []
        assert dto.constant_columns == []
        assert dto.high_cardinality_columns == []
        assert dto.highly_correlated_features == []
        assert dto.recommendations == []

    def test_complete_creation(self):
        """Test creation with all fields."""
        missing_columns = ["col1", "col2"]
        constant_columns = ["const_col"]
        high_cardinality_columns = ["user_id", "session_id"]
        correlations = [("feature1", "feature2", 0.95), ("feature3", "feature4", 0.87)]
        recommendations = [
            "Remove constant columns",
            "Handle missing values in col1 and col2",
            "Consider feature selection for highly correlated features",
        ]

        dto = DataQualityReportDTO(
            quality_score=0.72,
            n_missing_values=100,
            n_duplicates=25,
            n_outliers=50,
            missing_columns=missing_columns,
            constant_columns=constant_columns,
            high_cardinality_columns=high_cardinality_columns,
            highly_correlated_features=correlations,
            recommendations=recommendations,
        )

        assert dto.quality_score == 0.72
        assert dto.missing_columns == missing_columns
        assert dto.constant_columns == constant_columns
        assert dto.high_cardinality_columns == high_cardinality_columns
        assert dto.highly_correlated_features == correlations
        assert dto.recommendations == recommendations

    def test_quality_score_validation(self):
        """Test quality score validation bounds."""
        # Test minimum bound
        with pytest.raises(ValidationError):
            DataQualityReportDTO(
                quality_score=-0.1,
                n_missing_values=0,
                n_duplicates=0,
                n_outliers=0,
            )

        # Test maximum bound
        with pytest.raises(ValidationError):
            DataQualityReportDTO(
                quality_score=1.1,
                n_missing_values=0,
                n_duplicates=0,
                n_outliers=0,
            )

        # Test valid bounds
        for score in [0.0, 0.5, 1.0]:
            dto = DataQualityReportDTO(
                quality_score=score,
                n_missing_values=0,
                n_duplicates=0,
                n_outliers=0,
            )
            assert dto.quality_score == score

    def test_negative_counts_validation(self):
        """Test validation of negative counts."""
        # All count fields should be non-negative
        with pytest.raises(ValidationError):
            DataQualityReportDTO(
                quality_score=0.5,
                n_missing_values=-1,
                n_duplicates=0,
                n_outliers=0,
            )

    def test_correlation_tuple_structure(self):
        """Test highly correlated features tuple structure."""
        # Valid correlation tuples
        valid_correlations = [
            ("feat1", "feat2", 0.95),
            ("feat3", "feat4", -0.87),
            ("feat5", "feat6", 1.0),
        ]

        dto = DataQualityReportDTO(
            quality_score=0.8,
            n_missing_values=0,
            n_duplicates=0,
            n_outliers=0,
            highly_correlated_features=valid_correlations,
        )

        assert dto.highly_correlated_features == valid_correlations


class TestDatasetResponseDTO:
    """Test suite for DatasetResponseDTO."""

    def test_basic_creation(self):
        """Test basic response DTO creation."""
        dataset_id = uuid4()
        created_at = datetime.utcnow()

        dto = DatasetResponseDTO(
            id=dataset_id,
            name="response_dataset",
            shape=(1000, 20),
            n_samples=1000,
            n_features=20,
            feature_names=[f"feature_{i}" for i in range(20)],
            has_target=True,
            target_column="target",
            created_at=created_at,
            memory_usage_mb=5.2,
            numeric_features=18,
            categorical_features=2,
        )

        assert dto.id == dataset_id
        assert dto.name == "response_dataset"
        assert dto.status == "ready"  # Default value
        assert dto.quality_score is None  # Default value
        assert dto.description is None  # Default value

    def test_creation_with_optional_fields(self):
        """Test creation with all optional fields."""
        dataset_id = uuid4()
        metadata = {"processing_version": "2.1", "last_updated": "2024-01-01"}

        dto = DatasetResponseDTO(
            id=dataset_id,
            name="complete_response",
            description="Complete dataset response",
            shape=(500, 15),
            n_samples=500,
            n_features=15,
            feature_names=[f"col_{i}" for i in range(15)],
            has_target=False,
            created_at=datetime.utcnow(),
            memory_usage_mb=3.8,
            numeric_features=12,
            categorical_features=3,
            quality_score=0.91,
            status="processing",
            metadata=metadata,
        )

        assert dto.description == "Complete dataset response"
        assert dto.quality_score == 0.91
        assert dto.status == "processing"
        assert dto.metadata == metadata

    def test_json_schema_example_validation(self):
        """Test the JSON schema example is valid."""
        example = DatasetResponseDTO.model_config["json_schema_extra"]["example"]

        dto = DatasetResponseDTO(
            id=UUID(example["id"]),
            name=example["name"],
            description=example["description"],
            shape=tuple(example["shape"]),
            n_samples=example["n_samples"],
            n_features=example["n_features"],
            feature_names=example["feature_names"],
            has_target=example["has_target"],
            target_column=example["target_column"],
            created_at=datetime.fromisoformat(example["created_at"]),
            memory_usage_mb=example["memory_usage_mb"],
            numeric_features=example["numeric_features"],
            categorical_features=example["categorical_features"],
            quality_score=example["quality_score"],
            status=example["status"],
        )

        assert dto.id == UUID(example["id"])
        assert dto.quality_score == example["quality_score"]

    def test_serialization_roundtrip(self):
        """Test JSON serialization roundtrip."""
        dataset_id = uuid4()
        original = DatasetResponseDTO(
            id=dataset_id,
            name="serialization_test",
            shape=(100, 5),
            n_samples=100,
            n_features=5,
            feature_names=["a", "b", "c", "d", "e"],
            has_target=True,
            target_column="target",
            created_at=datetime.utcnow().replace(
                microsecond=0
            ),  # Remove microseconds for comparison
            memory_usage_mb=1.5,
            numeric_features=4,
            categorical_features=1,
            quality_score=0.88,
        )

        # Convert to JSON and back
        json_data = original.model_dump(mode="json")
        json_str = json.dumps(json_data, default=str)
        parsed_data = json.loads(json_str)

        # Create new instance from parsed data
        reconstructed = DatasetResponseDTO(
            **{
                **parsed_data,
                "id": UUID(parsed_data["id"]),
                "created_at": datetime.fromisoformat(parsed_data["created_at"]),
                "shape": tuple(parsed_data["shape"]),
            }
        )

        assert reconstructed.id == original.id
        assert reconstructed.name == original.name
        assert reconstructed.created_at == original.created_at


class TestDatasetUploadResponseDTO:
    """Test suite for DatasetUploadResponseDTO."""

    def test_basic_creation(self):
        """Test basic upload response creation."""
        upload_id = "upload_123"
        upload_url = "https://storage.example.com/upload/123"

        dto = DatasetUploadResponseDTO(
            id=upload_id,
            name="upload_dataset",
            status="processing",
            file_size=1024 * 1024,  # 1MB
            upload_url=upload_url,
        )

        assert dto.id == upload_id
        assert dto.name == "upload_dataset"
        assert dto.status == "processing"
        assert dto.file_size == 1024 * 1024
        assert dto.upload_url == upload_url
        assert dto.created_at is None  # Default
        assert dto.progress == 0.0  # Default
        assert dto.error_message is None  # Default

    def test_complete_creation(self):
        """Test creation with all fields."""
        created_at = datetime.utcnow()

        dto = DatasetUploadResponseDTO(
            id="upload_456",
            name="complete_upload",
            status="completed",
            file_size=5 * 1024 * 1024,  # 5MB
            upload_url="https://storage.example.com/upload/456",
            created_at=created_at,
            progress=100.0,
            error_message=None,
        )

        assert dto.id == "upload_456"
        assert dto.status == "completed"
        assert dto.created_at == created_at
        assert dto.progress == 100.0

    def test_failed_upload_scenario(self):
        """Test failed upload scenario."""
        dto = DatasetUploadResponseDTO(
            id="upload_failed",
            name="failed_dataset",
            status="failed",
            file_size=0,
            upload_url="https://storage.example.com/upload/failed",
            progress=45.0,
            error_message="File format not supported",
        )

        assert dto.status == "failed"
        assert dto.progress == 45.0
        assert dto.error_message == "File format not supported"

    def test_progress_values(self):
        """Test various progress values."""
        progress_values = [0.0, 25.5, 50.0, 75.8, 100.0]

        for progress in progress_values:
            dto = DatasetUploadResponseDTO(
                id="test",
                name="test",
                status="processing",
                file_size=1000,
                upload_url="http://test.com",
                progress=progress,
            )
            assert dto.progress == progress

    def test_file_size_values(self):
        """Test various file size values."""
        # Test different file sizes
        file_sizes = [0, 1024, 1024 * 1024, 100 * 1024 * 1024]  # 0B, 1KB, 1MB, 100MB

        for size in file_sizes:
            dto = DatasetUploadResponseDTO(
                id="test",
                name="test",
                status="processing",
                file_size=size,
                upload_url="http://test.com",
            )
            assert dto.file_size == size


class TestDatasetDTOIntegration:
    """Integration tests for dataset DTOs."""

    def test_dto_workflow(self):
        """Test typical DTO workflow from creation to response."""
        # Step 1: Create dataset request
        create_dto = CreateDatasetDTO(
            name="workflow_dataset",
            description="Dataset for workflow testing",
            target_column="anomaly",
            file_path="/data/workflow.csv",
            file_format="csv",
        )

        # Step 2: Upload response
        upload_dto = DatasetUploadResponseDTO(
            id="upload_workflow",
            name=create_dto.name,
            status="processing",
            file_size=2 * 1024 * 1024,
            upload_url="https://storage.example.com/upload/workflow",
            progress=100.0,
        )

        # Step 3: Quality report
        quality_dto = DataQualityReportDTO(
            quality_score=0.89,
            n_missing_values=5,
            n_duplicates=2,
            n_outliers=10,
            recommendations=["Handle 5 missing values", "Remove 2 duplicate rows"],
        )

        # Step 4: Final dataset response
        dataset_id = uuid4()
        response_dto = DatasetResponseDTO(
            id=dataset_id,
            name=create_dto.name,
            description=create_dto.description,
            shape=(1000, 25),
            n_samples=1000,
            n_features=25,
            feature_names=[f"feature_{i}" for i in range(25)],
            has_target=True,
            target_column=create_dto.target_column,
            created_at=datetime.utcnow(),
            memory_usage_mb=4.2,
            numeric_features=20,
            categorical_features=5,
            quality_score=quality_dto.quality_score,
            status="ready",
        )

        # Verify workflow consistency
        assert upload_dto.name == create_dto.name
        assert response_dto.name == create_dto.name
        assert response_dto.description == create_dto.description
        assert response_dto.target_column == create_dto.target_column
        assert response_dto.quality_score == quality_dto.quality_score

    def test_edge_cases_and_boundary_conditions(self):
        """Test edge cases and boundary conditions."""
        # Minimum valid dataset
        min_dataset = DatasetDTO(
            id=uuid4(),
            name="min",
            shape=(1, 1),
            n_samples=1,
            n_features=1,
            feature_names=["single_feature"],
            has_target=False,
            created_at=datetime.utcnow(),
            memory_usage_mb=0.001,
            numeric_features=1,
            categorical_features=0,
        )

        assert min_dataset.n_samples == 1
        assert min_dataset.n_features == 1

        # Large dataset
        large_dataset = DatasetDTO(
            id=uuid4(),
            name="large",
            shape=(1000000, 500),
            n_samples=1000000,
            n_features=500,
            feature_names=[f"feature_{i}" for i in range(500)],
            has_target=True,
            target_column="target",
            created_at=datetime.utcnow(),
            memory_usage_mb=2048.5,
            numeric_features=450,
            categorical_features=50,
        )

        assert large_dataset.n_samples == 1000000
        assert large_dataset.memory_usage_mb == 2048.5

        # Perfect quality score
        perfect_quality = DataQualityReportDTO(
            quality_score=1.0,
            n_missing_values=0,
            n_duplicates=0,
            n_outliers=0,
        )

        assert perfect_quality.quality_score == 1.0

        # Zero quality score
        zero_quality = DataQualityReportDTO(
            quality_score=0.0,
            n_missing_values=1000,
            n_duplicates=500,
            n_outliers=200,
        )

        assert zero_quality.quality_score == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
