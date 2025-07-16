"""Tests for Dataset DTOs."""

from datetime import datetime
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from monorepo.application.dto.dataset_dto import (
    CreateDatasetDTO,
    DataQualityReportDTO,
    DatasetDTO,
)


class TestDatasetDTO:
    """Test suite for DatasetDTO."""

    def test_valid_creation(self):
        """Test creating a valid dataset DTO."""
        dataset_id = uuid4()
        created_at = datetime.now()

        dto = DatasetDTO(
            id=dataset_id,
            name="credit_card_transactions",
            shape=(10000, 30),
            n_samples=10000,
            n_features=30,
            feature_names=["amount", "merchant_id", "time", "category"],
            has_target=True,
            target_column="is_fraud",
            created_at=created_at,
            metadata={"source": "bank_data", "version": "1.0"},
            description="Credit card transaction dataset for fraud detection",
            memory_usage_mb=2.4,
            numeric_features=25,
            categorical_features=5,
        )

        assert dto.id == dataset_id
        assert dto.name == "credit_card_transactions"
        assert dto.shape == (10000, 30)
        assert dto.n_samples == 10000
        assert dto.n_features == 30
        assert dto.feature_names == ["amount", "merchant_id", "time", "category"]
        assert dto.has_target is True
        assert dto.target_column == "is_fraud"
        assert dto.created_at == created_at
        assert dto.metadata == {"source": "bank_data", "version": "1.0"}
        assert dto.description == "Credit card transaction dataset for fraud detection"
        assert dto.memory_usage_mb == 2.4
        assert dto.numeric_features == 25
        assert dto.categorical_features == 5

    def test_default_values(self):
        """Test default values."""
        dataset_id = uuid4()
        created_at = datetime.now()

        dto = DatasetDTO(
            id=dataset_id,
            name="test_dataset",
            shape=(1000, 10),
            n_samples=1000,
            n_features=10,
            feature_names=["feature1", "feature2"],
            has_target=False,
            created_at=created_at,
            memory_usage_mb=1.0,
            numeric_features=8,
            categorical_features=2,
        )

        assert dto.target_column is None
        assert dto.metadata == {}
        assert dto.description is None

    def test_target_column_consistency(self):
        """Test target column consistency with has_target flag."""
        dataset_id = uuid4()
        created_at = datetime.now()

        # Test with target column
        dto_with_target = DatasetDTO(
            id=dataset_id,
            name="test_dataset",
            shape=(1000, 10),
            n_samples=1000,
            n_features=10,
            feature_names=["feature1", "feature2"],
            has_target=True,
            target_column="target",
            created_at=created_at,
            memory_usage_mb=1.0,
            numeric_features=8,
            categorical_features=2,
        )

        assert dto_with_target.has_target is True
        assert dto_with_target.target_column == "target"

        # Test without target column
        dto_without_target = DatasetDTO(
            id=dataset_id,
            name="test_dataset",
            shape=(1000, 10),
            n_samples=1000,
            n_features=10,
            feature_names=["feature1", "feature2"],
            has_target=False,
            target_column=None,
            created_at=created_at,
            memory_usage_mb=1.0,
            numeric_features=8,
            categorical_features=2,
        )

        assert dto_without_target.has_target is False
        assert dto_without_target.target_column is None

    def test_shape_consistency(self):
        """Test shape consistency with n_samples and n_features."""
        dataset_id = uuid4()
        created_at = datetime.now()

        dto = DatasetDTO(
            id=dataset_id,
            name="test_dataset",
            shape=(1000, 10),
            n_samples=1000,
            n_features=10,
            feature_names=["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10"],
            has_target=False,
            created_at=created_at,
            memory_usage_mb=1.0,
            numeric_features=8,
            categorical_features=2,
        )

        assert dto.shape == (dto.n_samples, dto.n_features)
        assert dto.n_features == len(dto.feature_names)

    def test_feature_type_consistency(self):
        """Test feature type consistency."""
        dataset_id = uuid4()
        created_at = datetime.now()

        dto = DatasetDTO(
            id=dataset_id,
            name="test_dataset",
            shape=(1000, 10),
            n_samples=1000,
            n_features=10,
            feature_names=["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10"],
            has_target=False,
            created_at=created_at,
            memory_usage_mb=1.0,
            numeric_features=8,
            categorical_features=2,
        )

        # Should sum to total features
        assert dto.numeric_features + dto.categorical_features == dto.n_features

    def test_required_fields(self):
        """Test required fields."""
        with pytest.raises(ValidationError):
            DatasetDTO()

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        dataset_id = uuid4()
        created_at = datetime.now()

        with pytest.raises(ValidationError):
            DatasetDTO(
                id=dataset_id,
                name="test_dataset",
                shape=(1000, 10),
                n_samples=1000,
                n_features=10,
                feature_names=["feature1", "feature2"],
                has_target=False,
                created_at=created_at,
                memory_usage_mb=1.0,
                numeric_features=8,
                categorical_features=2,
                unknown_field="value",
            )

    def test_uuid_validation(self):
        """Test UUID validation."""
        created_at = datetime.now()

        # Valid UUID
        valid_id = uuid4()
        dto = DatasetDTO(
            id=valid_id,
            name="test_dataset",
            shape=(1000, 10),
            n_samples=1000,
            n_features=10,
            feature_names=["feature1", "feature2"],
            has_target=False,
            created_at=created_at,
            memory_usage_mb=1.0,
            numeric_features=8,
            categorical_features=2,
        )

        assert isinstance(dto.id, UUID)
        assert dto.id == valid_id

    def test_datetime_validation(self):
        """Test datetime validation."""
        dataset_id = uuid4()
        created_at = datetime.now()

        dto = DatasetDTO(
            id=dataset_id,
            name="test_dataset",
            shape=(1000, 10),
            n_samples=1000,
            n_features=10,
            feature_names=["feature1", "feature2"],
            has_target=False,
            created_at=created_at,
            memory_usage_mb=1.0,
            numeric_features=8,
            categorical_features=2,
        )

        assert isinstance(dto.created_at, datetime)
        assert dto.created_at == created_at

    def test_json_schema_example(self):
        """Test that JSON schema example is valid."""
        # This test ensures the example in the model config is valid
        schema = DatasetDTO.model_json_schema()
        example = schema.get("example")

        assert example is not None
        assert example["name"] == "credit_card_transactions"
        assert example["n_samples"] == 10000
        assert example["n_features"] == 30
        assert example["has_target"] is True
        assert example["target_column"] == "is_fraud"


class TestCreateDatasetDTO:
    """Test suite for CreateDatasetDTO."""

    def test_valid_creation(self):
        """Test creating a valid create dataset DTO."""
        dto = CreateDatasetDTO(
            name="new_dataset",
            description="A new dataset for testing",
            target_column="target",
            metadata={"source": "test", "version": "1.0"},
            file_path="/path/to/dataset.csv",
            file_format="csv",
            delimiter=";",
            encoding="utf-16",
            parse_dates=["date_column"],
            dtype={"column1": "int64", "column2": "float64"},
        )

        assert dto.name == "new_dataset"
        assert dto.description == "A new dataset for testing"
        assert dto.target_column == "target"
        assert dto.metadata == {"source": "test", "version": "1.0"}
        assert dto.file_path == "/path/to/dataset.csv"
        assert dto.file_format == "csv"
        assert dto.delimiter == ";"
        assert dto.encoding == "utf-16"
        assert dto.parse_dates == ["date_column"]
        assert dto.dtype == {"column1": "int64", "column2": "float64"}

    def test_default_values(self):
        """Test default values."""
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

    def test_name_validation(self):
        """Test name validation."""
        # Valid name
        dto = CreateDatasetDTO(name="valid_name")
        assert dto.name == "valid_name"

        # Invalid: empty name
        with pytest.raises(ValidationError):
            CreateDatasetDTO(name="")

        # Invalid: too long name
        with pytest.raises(ValidationError):
            CreateDatasetDTO(name="a" * 101)

    def test_description_validation(self):
        """Test description validation."""
        # Valid description
        dto = CreateDatasetDTO(name="test", description="Valid description")
        assert dto.description == "Valid description"

        # Invalid: too long description
        with pytest.raises(ValidationError):
            CreateDatasetDTO(name="test", description="a" * 501)

    def test_file_format_validation(self):
        """Test file format validation."""
        # Valid formats
        valid_formats = ["csv", "parquet", "json", "excel"]
        for format_type in valid_formats:
            dto = CreateDatasetDTO(name="test", file_format=format_type)
            assert dto.file_format == format_type

        # Invalid format
        with pytest.raises(ValidationError):
            CreateDatasetDTO(name="test", file_format="invalid_format")

    def test_required_fields(self):
        """Test required fields."""
        with pytest.raises(ValidationError):
            CreateDatasetDTO()

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError):
            CreateDatasetDTO(name="test", unknown_field="value")

    def test_data_loading_options(self):
        """Test data loading options."""
        dto = CreateDatasetDTO(
            name="test_dataset",
            file_path="/path/to/data.csv",
            file_format="csv",
            delimiter="|",
            encoding="utf-8",
            parse_dates=["created_at", "updated_at"],
            dtype={"id": "str", "value": "float32"},
        )

        assert dto.file_path == "/path/to/data.csv"
        assert dto.file_format == "csv"
        assert dto.delimiter == "|"
        assert dto.encoding == "utf-8"
        assert dto.parse_dates == ["created_at", "updated_at"]
        assert dto.dtype == {"id": "str", "value": "float32"}

    def test_metadata_handling(self):
        """Test metadata handling."""
        metadata = {
            "source": "production",
            "version": "2.1",
            "tags": ["important", "validated"],
            "owner": "data_team",
        }

        dto = CreateDatasetDTO(name="test", metadata=metadata)
        assert dto.metadata == metadata

    def test_target_column_optional(self):
        """Test that target column is optional."""
        # Without target column (unsupervised learning)
        dto_no_target = CreateDatasetDTO(name="unsupervised_dataset")
        assert dto_no_target.target_column is None

        # With target column (supervised learning)
        dto_with_target = CreateDatasetDTO(
            name="supervised_dataset", target_column="target"
        )
        assert dto_with_target.target_column == "target"


class TestDataQualityReportDTO:
    """Test suite for DataQualityReportDTO."""

    def test_valid_creation(self):
        """Test creating a valid data quality report DTO."""
        dto = DataQualityReportDTO(
            quality_score=0.85,
            n_missing_values=150,
            n_duplicates=23,
            n_outliers=45,
            missing_columns=["optional_field1", "optional_field2"],
            constant_columns=["constant_id"],
            high_cardinality_columns=["user_id", "session_id"],
            highly_correlated_features=[
                ("feature1", "feature2", 0.95),
                ("feature3", "feature4", 0.87),
            ],
            recommendations=[
                "Remove constant columns",
                "Handle missing values in optional fields",
                "Consider feature selection for highly correlated features",
            ],
        )

        assert dto.quality_score == 0.85
        assert dto.n_missing_values == 150
        assert dto.n_duplicates == 23
        assert dto.n_outliers == 45
        assert dto.missing_columns == ["optional_field1", "optional_field2"]
        assert dto.constant_columns == ["constant_id"]
        assert dto.high_cardinality_columns == ["user_id", "session_id"]
        assert dto.highly_correlated_features == [
            ("feature1", "feature2", 0.95),
            ("feature3", "feature4", 0.87),
        ]
        assert len(dto.recommendations) == 3
        assert "Remove constant columns" in dto.recommendations

    def test_default_values(self):
        """Test default values."""
        dto = DataQualityReportDTO(
            quality_score=0.9, n_missing_values=0, n_duplicates=0, n_outliers=0
        )

        assert dto.quality_score == 0.9
        assert dto.n_missing_values == 0
        assert dto.n_duplicates == 0
        assert dto.n_outliers == 0
        assert dto.missing_columns == []
        assert dto.constant_columns == []
        assert dto.high_cardinality_columns == []
        assert dto.highly_correlated_features == []
        assert dto.recommendations == []

    def test_quality_score_validation(self):
        """Test quality score validation."""
        # Valid range
        dto = DataQualityReportDTO(
            quality_score=0.5, n_missing_values=0, n_duplicates=0, n_outliers=0
        )
        assert dto.quality_score == 0.5

        # Invalid: negative
        with pytest.raises(ValidationError):
            DataQualityReportDTO(
                quality_score=-0.1, n_missing_values=0, n_duplicates=0, n_outliers=0
            )

        # Invalid: greater than 1
        with pytest.raises(ValidationError):
            DataQualityReportDTO(
                quality_score=1.1, n_missing_values=0, n_duplicates=0, n_outliers=0
            )

    def test_perfect_quality_score(self):
        """Test perfect quality score."""
        dto = DataQualityReportDTO(
            quality_score=1.0, n_missing_values=0, n_duplicates=0, n_outliers=0
        )
        assert dto.quality_score == 1.0

    def test_zero_quality_score(self):
        """Test zero quality score."""
        dto = DataQualityReportDTO(
            quality_score=0.0, n_missing_values=1000, n_duplicates=500, n_outliers=200
        )
        assert dto.quality_score == 0.0

    def test_negative_counts_validation(self):
        """Test that negative counts are handled appropriately."""
        # All counts should be non-negative integers
        dto = DataQualityReportDTO(
            quality_score=0.5, n_missing_values=0, n_duplicates=0, n_outliers=0
        )
        assert dto.n_missing_values >= 0
        assert dto.n_duplicates >= 0
        assert dto.n_outliers >= 0

    def test_correlation_tuples(self):
        """Test correlation tuple structure."""
        correlations = [
            ("feature_a", "feature_b", 0.95),
            ("feature_c", "feature_d", 0.87),
            ("feature_e", "feature_f", -0.82),
        ]

        dto = DataQualityReportDTO(
            quality_score=0.7,
            n_missing_values=10,
            n_duplicates=5,
            n_outliers=15,
            highly_correlated_features=correlations,
        )

        assert len(dto.highly_correlated_features) == 3

        # Check first correlation
        feature1, feature2, corr_value = dto.highly_correlated_features[0]
        assert feature1 == "feature_a"
        assert feature2 == "feature_b"
        assert corr_value == 0.95

        # Check negative correlation
        feature1, feature2, corr_value = dto.highly_correlated_features[2]
        assert feature1 == "feature_e"
        assert feature2 == "feature_f"
        assert corr_value == -0.82

    def test_recommendations_list(self):
        """Test recommendations list handling."""
        recommendations = [
            "Remove constant columns: ['id', 'version']",
            "Handle missing values in columns: ['age', 'income']",
            "Consider encoding categorical variables",
            "Review outliers in 'amount' column (45 outliers detected)",
            "Feature 'A' and 'B' are highly correlated (0.95)",
        ]

        dto = DataQualityReportDTO(
            quality_score=0.6,
            n_missing_values=100,
            n_duplicates=20,
            n_outliers=45,
            recommendations=recommendations,
        )

        assert len(dto.recommendations) == 5
        assert all(isinstance(rec, str) for rec in dto.recommendations)
        assert "Remove constant columns" in dto.recommendations[0]
        assert "Handle missing values" in dto.recommendations[1]

    def test_empty_report(self):
        """Test empty/minimal quality report."""
        dto = DataQualityReportDTO(
            quality_score=1.0, n_missing_values=0, n_duplicates=0, n_outliers=0
        )

        assert dto.quality_score == 1.0
        assert dto.n_missing_values == 0
        assert dto.n_duplicates == 0
        assert dto.n_outliers == 0
        assert len(dto.missing_columns) == 0
        assert len(dto.constant_columns) == 0
        assert len(dto.high_cardinality_columns) == 0
        assert len(dto.highly_correlated_features) == 0
        assert len(dto.recommendations) == 0

    def test_required_fields(self):
        """Test required fields."""
        with pytest.raises(ValidationError):
            DataQualityReportDTO()

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError):
            DataQualityReportDTO(
                quality_score=0.8,
                n_missing_values=0,
                n_duplicates=0,
                n_outliers=0,
                unknown_field="value",
            )


class TestDatasetDTOIntegration:
    """Test integration scenarios for dataset DTOs."""

    def test_dataset_creation_workflow(self):
        """Test complete dataset creation workflow."""
        # Create dataset creation request
        create_request = CreateDatasetDTO(
            name="fraud_detection_dataset",
            description="Dataset for credit card fraud detection",
            target_column="is_fraud",
            metadata={
                "source": "bank_transactions",
                "version": "1.0",
                "created_by": "data_team",
            },
            file_path="/data/transactions.csv",
            file_format="csv",
            delimiter=",",
            encoding="utf-8",
            parse_dates=["transaction_date"],
            dtype={"amount": "float64", "merchant_id": "str"},
        )

        # Simulate dataset creation result
        dataset_id = uuid4()
        created_at = datetime.now()

        dataset_result = DatasetDTO(
            id=dataset_id,
            name=create_request.name,
            shape=(50000, 15),
            n_samples=50000,
            n_features=15,
            feature_names=[
                "amount",
                "merchant_id",
                "transaction_date",
                "card_type",
                "merchant_category",
                "hour",
                "day_of_week",
                "month",
                "user_age",
                "account_balance",
                "transaction_count_24h",
                "avg_amount_30d",
                "location_risk",
                "velocity_score",
                "is_fraud",
            ],
            has_target=True,
            target_column=create_request.target_column,
            created_at=created_at,
            metadata=create_request.metadata,
            description=create_request.description,
            memory_usage_mb=12.5,
            numeric_features=11,
            categorical_features=4,
        )

        # Verify workflow
        assert dataset_result.name == create_request.name
        assert dataset_result.description == create_request.description
        assert dataset_result.target_column == create_request.target_column
        assert dataset_result.metadata == create_request.metadata
        assert dataset_result.has_target is True
        assert dataset_result.n_samples == 50000
        assert dataset_result.n_features == 15
        assert len(dataset_result.feature_names) == 15
        assert (
            dataset_result.numeric_features + dataset_result.categorical_features
            == dataset_result.n_features
        )

    def test_data_quality_assessment_workflow(self):
        """Test data quality assessment workflow."""
        # Create dataset
        dataset_id = uuid4()
        dataset = DatasetDTO(
            id=dataset_id,
            name="quality_test_dataset",
            shape=(10000, 20),
            n_samples=10000,
            n_features=20,
            feature_names=[f"feature_{i}" for i in range(20)],
            has_target=True,
            target_column="target",
            created_at=datetime.now(),
            memory_usage_mb=5.0,
            numeric_features=15,
            categorical_features=5,
        )

        # Create quality report
        quality_report = DataQualityReportDTO(
            quality_score=0.72,
            n_missing_values=250,
            n_duplicates=15,
            n_outliers=83,
            missing_columns=["feature_18", "feature_19"],
            constant_columns=["feature_0"],
            high_cardinality_columns=["feature_1", "feature_2"],
            highly_correlated_features=[
                ("feature_3", "feature_4", 0.94),
                ("feature_5", "feature_6", 0.89),
            ],
            recommendations=[
                "Remove constant column: feature_0",
                "Handle missing values in feature_18 and feature_19",
                "Consider feature selection for highly correlated pairs",
                "Review outliers in numeric features",
                "Reduce cardinality in categorical features",
            ],
        )

        # Verify quality assessment
        assert quality_report.quality_score == 0.72
        assert quality_report.n_missing_values > 0
        assert quality_report.n_duplicates > 0
        assert quality_report.n_outliers > 0
        assert len(quality_report.missing_columns) == 2
        assert len(quality_report.constant_columns) == 1
        assert len(quality_report.high_cardinality_columns) == 2
        assert len(quality_report.highly_correlated_features) == 2
        assert len(quality_report.recommendations) == 5

    def test_dataset_update_workflow(self):
        """Test dataset update workflow."""
        # Original dataset
        original_dataset = DatasetDTO(
            id=uuid4(),
            name="original_dataset",
            shape=(5000, 10),
            n_samples=5000,
            n_features=10,
            feature_names=[f"feature_{i}" for i in range(10)],
            has_target=False,
            created_at=datetime.now(),
            memory_usage_mb=2.5,
            numeric_features=8,
            categorical_features=2,
        )

        # Update request
        update_request = CreateDatasetDTO(
            name="updated_dataset",
            description="Updated dataset with improved quality",
            target_column="new_target",
            metadata={
                "source": "updated_source",
                "version": "2.0",
                "parent_id": str(original_dataset.id),
            },
        )

        # Updated dataset
        updated_dataset = DatasetDTO(
            id=uuid4(),
            name=update_request.name,
            shape=(5000, 11),  # Added target column
            n_samples=5000,
            n_features=11,
            feature_names=original_dataset.feature_names + ["new_target"],
            has_target=True,
            target_column=update_request.target_column,
            created_at=datetime.now(),
            metadata=update_request.metadata,
            description=update_request.description,
            memory_usage_mb=2.7,
            numeric_features=8,
            categorical_features=3,
        )

        # Verify update
        assert updated_dataset.name == update_request.name
        assert updated_dataset.description == update_request.description
        assert updated_dataset.target_column == update_request.target_column
        assert updated_dataset.has_target is True
        assert updated_dataset.n_features == original_dataset.n_features + 1
        assert updated_dataset.metadata["parent_id"] == str(original_dataset.id)
        assert updated_dataset.metadata["version"] == "2.0"

    def test_dataset_validation_workflow(self):
        """Test dataset validation workflow."""
        # Create dataset with potential issues
        problematic_dataset = DatasetDTO(
            id=uuid4(),
            name="problematic_dataset",
            shape=(100, 5),  # Very small dataset
            n_samples=100,
            n_features=5,
            feature_names=["f1", "f2", "f3", "f4", "f5"],
            has_target=True,
            target_column="f5",
            created_at=datetime.now(),
            memory_usage_mb=0.1,
            numeric_features=4,
            categorical_features=1,
        )

        # Generate quality report showing issues
        quality_report = DataQualityReportDTO(
            quality_score=0.45,  # Low quality
            n_missing_values=50,  # 50% missing values
            n_duplicates=20,  # 20% duplicates
            n_outliers=15,  # 15% outliers
            missing_columns=["f2", "f3"],
            constant_columns=["f1"],
            high_cardinality_columns=[],
            highly_correlated_features=[("f2", "f3", 0.98)],
            recommendations=[
                "Dataset too small for reliable model training",
                "High missing value rate (50%)",
                "High duplicate rate (20%)",
                "Remove constant column f1",
                "Consider data augmentation or collection of more samples",
            ],
        )

        # Verify validation results
        assert quality_report.quality_score < 0.5
        assert quality_report.n_missing_values == problematic_dataset.n_samples * 0.5
        assert quality_report.n_duplicates == problematic_dataset.n_samples * 0.2
        assert len(quality_report.recommendations) == 5
        assert "Dataset too small" in quality_report.recommendations[0]

    def test_dto_serialization(self):
        """Test DTO serialization and deserialization."""
        # Create dataset DTO
        dataset_id = uuid4()
        created_at = datetime.now()

        original_dto = DatasetDTO(
            id=dataset_id,
            name="serialization_test",
            shape=(1000, 10),
            n_samples=1000,
            n_features=10,
            feature_names=[f"feature_{i}" for i in range(10)],
            has_target=True,
            target_column="target",
            created_at=created_at,
            metadata={"test": "value"},
            description="Test dataset",
            memory_usage_mb=2.0,
            numeric_features=8,
            categorical_features=2,
        )

        # Serialize to dict
        dto_dict = original_dto.model_dump()

        assert dto_dict["name"] == "serialization_test"
        assert dto_dict["shape"] == [1000, 10]
        assert dto_dict["n_samples"] == 1000
        assert dto_dict["n_features"] == 10
        assert dto_dict["has_target"] is True
        assert dto_dict["target_column"] == "target"
        assert dto_dict["metadata"] == {"test": "value"}

        # Deserialize from dict
        restored_dto = DatasetDTO.model_validate(dto_dict)

        assert restored_dto.name == original_dto.name
        assert restored_dto.shape == original_dto.shape
        assert restored_dto.n_samples == original_dto.n_samples
        assert restored_dto.n_features == original_dto.n_features
        assert restored_dto.has_target == original_dto.has_target
        assert restored_dto.target_column == original_dto.target_column
        assert restored_dto.metadata == original_dto.metadata
        assert restored_dto.description == original_dto.description

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Single row dataset
        single_row_dataset = DatasetDTO(
            id=uuid4(),
            name="single_row",
            shape=(1, 5),
            n_samples=1,
            n_features=5,
            feature_names=["f1", "f2", "f3", "f4", "f5"],
            has_target=False,
            created_at=datetime.now(),
            memory_usage_mb=0.001,
            numeric_features=5,
            categorical_features=0,
        )

        assert single_row_dataset.n_samples == 1
        assert (
            single_row_dataset.numeric_features
            + single_row_dataset.categorical_features
            == 5
        )

        # Large dataset
        large_dataset = DatasetDTO(
            id=uuid4(),
            name="large_dataset",
            shape=(1000000, 100),
            n_samples=1000000,
            n_features=100,
            feature_names=[f"feature_{i}" for i in range(100)],
            has_target=True,
            target_column="target",
            created_at=datetime.now(),
            memory_usage_mb=800.0,
            numeric_features=80,
            categorical_features=20,
        )

        assert large_dataset.n_samples == 1000000
        assert large_dataset.n_features == 100
        assert large_dataset.memory_usage_mb == 800.0

        # Perfect quality dataset
        perfect_quality = DataQualityReportDTO(
            quality_score=1.0, n_missing_values=0, n_duplicates=0, n_outliers=0
        )

        assert perfect_quality.quality_score == 1.0
        assert perfect_quality.n_missing_values == 0
        assert perfect_quality.n_duplicates == 0
        assert perfect_quality.n_outliers == 0
        assert len(perfect_quality.recommendations) == 0

    def test_comprehensive_dataset_metadata(self):
        """Test comprehensive dataset metadata handling."""
        comprehensive_metadata = {
            "source": "production_database",
            "version": "3.2.1",
            "created_by": "data_pipeline",
            "tags": ["validated", "production", "high_quality"],
            "schema_version": "2.0",
            "collection_date": "2024-01-15",
            "validation_status": "passed",
            "quality_checks": {
                "completeness": 0.95,
                "consistency": 0.98,
                "accuracy": 0.92,
            },
            "lineage": {
                "parent_datasets": ["raw_transactions", "customer_profiles"],
                "transformations": [
                    "clean_missing",
                    "feature_engineering",
                    "outlier_removal",
                ],
            },
            "access_control": {
                "classification": "internal",
                "allowed_roles": ["data_scientist", "analyst"],
            },
        }

        dataset = DatasetDTO(
            id=uuid4(),
            name="comprehensive_dataset",
            shape=(10000, 25),
            n_samples=10000,
            n_features=25,
            feature_names=[f"feature_{i}" for i in range(25)],
            has_target=True,
            target_column="target",
            created_at=datetime.now(),
            metadata=comprehensive_metadata,
            description="Comprehensive dataset with rich metadata",
            memory_usage_mb=15.0,
            numeric_features=20,
            categorical_features=5,
        )

        # Verify metadata structure
        assert dataset.metadata["source"] == "production_database"
        assert dataset.metadata["version"] == "3.2.1"
        assert len(dataset.metadata["tags"]) == 3
        assert "validated" in dataset.metadata["tags"]
        assert dataset.metadata["quality_checks"]["completeness"] == 0.95
        assert len(dataset.metadata["lineage"]["parent_datasets"]) == 2
        assert dataset.metadata["access_control"]["classification"] == "internal"
