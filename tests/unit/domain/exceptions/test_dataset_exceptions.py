"""Tests for dataset domain exceptions."""

import pytest

from pynomaly.domain.exceptions.dataset_exceptions import (
    DataTypeError,
    DataValidationError,
    DatasetError,
    FeatureMismatchError,
    InsufficientDataError,
)
from pynomaly.domain.exceptions.base import DomainError, PynamolyError, ValidationError


class TestDatasetError:
    """Test suite for DatasetError base class."""

    def test_inheritance(self):
        """Test DatasetError inheritance."""
        error = DatasetError("Dataset error")
        assert isinstance(error, DomainError)
        assert isinstance(error, PynamolyError)
        assert isinstance(error, DatasetError)

    def test_basic_functionality(self):
        """Test basic functionality."""
        error = DatasetError("Dataset error", details={"key": "value"})
        assert error.message == "Dataset error"
        assert error.details == {"key": "value"}

    def test_with_context(self):
        """Test with_context method."""
        error = DatasetError("Dataset error")
        error.with_context(context="test")
        assert error.details["context"] == "test"

    def test_string_representation(self):
        """Test string representation."""
        error = DatasetError("Dataset error")
        assert str(error) == "Dataset error"

    def test_exception_raising(self):
        """Test that error can be raised."""
        with pytest.raises(DatasetError, match="Dataset error"):
            raise DatasetError("Dataset error")


class TestDataValidationError:
    """Test suite for DataValidationError."""

    def test_multiple_inheritance(self):
        """Test DataValidationError multiple inheritance."""
        error = DataValidationError("Validation failed")
        assert isinstance(error, ValidationError)
        assert isinstance(error, DatasetError)
        assert isinstance(error, DomainError)
        assert isinstance(error, PynamolyError)

    def test_basic_creation(self):
        """Test basic data validation error creation."""
        error = DataValidationError("Validation failed")
        assert error.message == "Validation failed"
        assert error.details == {}

    def test_creation_with_dataset_name(self):
        """Test creation with dataset name."""
        error = DataValidationError("Invalid data", dataset_name="test_dataset")
        expected_message = "Validation failed for dataset 'test_dataset': Invalid data"
        assert error.message == expected_message
        assert error.details["dataset_name"] == "test_dataset"

    def test_creation_with_additional_details(self):
        """Test creation with additional details."""
        error = DataValidationError(
            "Invalid data",
            dataset_name="test_dataset",
            field="age",
            value=-5,
            min_value=0
        )
        expected_message = "Validation failed for dataset 'test_dataset': Invalid data"
        assert error.message == expected_message
        assert error.details["dataset_name"] == "test_dataset"
        assert error.details["field"] == "age"
        assert error.details["value"] == -5
        assert error.details["min_value"] == 0

    def test_creation_without_dataset_name(self):
        """Test creation without dataset name."""
        error = DataValidationError("Invalid data", field="age")
        assert error.message == "Invalid data"
        assert error.details["field"] == "age"
        assert "dataset_name" not in error.details

    def test_none_dataset_name(self):
        """Test handling of None dataset name."""
        error = DataValidationError("Invalid data", dataset_name=None)
        assert error.message == "Invalid data"
        assert error.details["dataset_name"] is None

    def test_empty_dataset_name(self):
        """Test handling of empty dataset name."""
        error = DataValidationError("Invalid data", dataset_name="")
        assert error.message == "Invalid data"
        assert error.details["dataset_name"] == ""

    def test_string_representation(self):
        """Test string representation."""
        error = DataValidationError("Invalid data", dataset_name="test_dataset")
        result = str(error)
        assert "Validation failed for dataset 'test_dataset': Invalid data" in result
        assert "dataset_name=test_dataset" in result

    def test_exception_raising(self):
        """Test that error can be raised."""
        with pytest.raises(DataValidationError):
            raise DataValidationError("Validation failed")


class TestInsufficientDataError:
    """Test suite for InsufficientDataError."""

    def test_inheritance(self):
        """Test InsufficientDataError inheritance."""
        error = InsufficientDataError("test_dataset", 10, 100)
        assert isinstance(error, DatasetError)
        assert isinstance(error, DomainError)
        assert isinstance(error, PynamolyError)

    def test_basic_creation(self):
        """Test basic insufficient data error creation."""
        error = InsufficientDataError("test_dataset", 10, 100)
        expected_message = "Dataset 'test_dataset' has insufficient data: 10 samples, but 100 required"
        assert error.message == expected_message
        assert error.details["dataset_name"] == "test_dataset"
        assert error.details["n_samples"] == 10
        assert error.details["min_required"] == 100
        assert error.details["operation"] is None

    def test_creation_with_operation(self):
        """Test creation with operation."""
        error = InsufficientDataError("test_dataset", 10, 100, operation="training")
        expected_message = "Dataset 'test_dataset' has insufficient data: 10 samples, but 100 required for training"
        assert error.message == expected_message
        assert error.details["dataset_name"] == "test_dataset"
        assert error.details["n_samples"] == 10
        assert error.details["min_required"] == 100
        assert error.details["operation"] == "training"

    def test_creation_with_additional_details(self):
        """Test creation with additional details."""
        error = InsufficientDataError(
            "test_dataset",
            10,
            100,
            operation="training",
            model_type="RandomForest",
            expected_accuracy=0.95
        )
        expected_message = "Dataset 'test_dataset' has insufficient data: 10 samples, but 100 required for training"
        assert error.message == expected_message
        assert error.details["dataset_name"] == "test_dataset"
        assert error.details["n_samples"] == 10
        assert error.details["min_required"] == 100
        assert error.details["operation"] == "training"
        assert error.details["model_type"] == "RandomForest"
        assert error.details["expected_accuracy"] == 0.95

    def test_zero_samples(self):
        """Test with zero samples."""
        error = InsufficientDataError("empty_dataset", 0, 1)
        expected_message = "Dataset 'empty_dataset' has insufficient data: 0 samples, but 1 required"
        assert error.message == expected_message
        assert error.details["n_samples"] == 0
        assert error.details["min_required"] == 1

    def test_large_numbers(self):
        """Test with large numbers."""
        error = InsufficientDataError("large_dataset", 1000000, 10000000)
        expected_message = "Dataset 'large_dataset' has insufficient data: 1000000 samples, but 10000000 required"
        assert error.message == expected_message
        assert error.details["n_samples"] == 1000000
        assert error.details["min_required"] == 10000000

    def test_string_representation(self):
        """Test string representation."""
        error = InsufficientDataError("test_dataset", 10, 100, operation="training")
        result = str(error)
        assert "Dataset 'test_dataset' has insufficient data: 10 samples, but 100 required for training" in result
        assert "dataset_name=test_dataset" in result
        assert "n_samples=10" in result
        assert "min_required=100" in result
        assert "operation=training" in result

    def test_exception_raising(self):
        """Test that error can be raised."""
        with pytest.raises(InsufficientDataError):
            raise InsufficientDataError("test_dataset", 10, 100)


class TestDataTypeError:
    """Test suite for DataTypeError."""

    def test_inheritance(self):
        """Test DataTypeError inheritance."""
        error = DataTypeError("Type error")
        assert isinstance(error, DatasetError)
        assert isinstance(error, DomainError)
        assert isinstance(error, PynamolyError)

    def test_basic_creation(self):
        """Test basic data type error creation."""
        error = DataTypeError("Invalid data type")
        assert error.message == "Invalid data type"
        assert error.details == {}

    def test_creation_with_feature(self):
        """Test creation with feature."""
        error = DataTypeError("Invalid data type", feature="age")
        assert error.message == "Invalid data type"
        assert error.details["feature"] == "age"

    def test_creation_with_expected_type(self):
        """Test creation with expected type."""
        error = DataTypeError("Invalid data type", expected_type="int")
        assert error.message == "Invalid data type"
        assert error.details["expected_type"] == "int"

    def test_creation_with_actual_type(self):
        """Test creation with actual type."""
        error = DataTypeError("Invalid data type", actual_type="string")
        assert error.message == "Invalid data type"
        assert error.details["actual_type"] == "string"

    def test_creation_with_all_parameters(self):
        """Test creation with all parameters."""
        error = DataTypeError(
            "Invalid data type",
            feature="age",
            expected_type="int",
            actual_type="string"
        )
        assert error.message == "Invalid data type"
        assert error.details["feature"] == "age"
        assert error.details["expected_type"] == "int"
        assert error.details["actual_type"] == "string"

    def test_creation_with_additional_details(self):
        """Test creation with additional details."""
        error = DataTypeError(
            "Invalid data type",
            feature="age",
            expected_type="int",
            actual_type="string",
            dataset_name="test_dataset",
            row_index=42
        )
        assert error.message == "Invalid data type"
        assert error.details["feature"] == "age"
        assert error.details["expected_type"] == "int"
        assert error.details["actual_type"] == "string"
        assert error.details["dataset_name"] == "test_dataset"
        assert error.details["row_index"] == 42

    def test_none_values(self):
        """Test handling of None values."""
        error = DataTypeError(
            "Invalid data type",
            feature=None,
            expected_type=None,
            actual_type=None
        )
        assert error.message == "Invalid data type"
        assert "feature" not in error.details
        assert "expected_type" not in error.details
        assert "actual_type" not in error.details

    def test_empty_string_values(self):
        """Test handling of empty string values."""
        error = DataTypeError(
            "Invalid data type",
            feature="",
            expected_type="",
            actual_type=""
        )
        assert error.message == "Invalid data type"
        assert error.details["feature"] == ""
        assert error.details["expected_type"] == ""
        assert error.details["actual_type"] == ""

    def test_string_representation(self):
        """Test string representation."""
        error = DataTypeError(
            "Invalid data type",
            feature="age",
            expected_type="int",
            actual_type="string"
        )
        result = str(error)
        assert "Invalid data type" in result
        assert "feature=age" in result
        assert "expected_type=int" in result
        assert "actual_type=string" in result

    def test_exception_raising(self):
        """Test that error can be raised."""
        with pytest.raises(DataTypeError):
            raise DataTypeError("Invalid data type")


class TestFeatureMismatchError:
    """Test suite for FeatureMismatchError."""

    def test_inheritance(self):
        """Test FeatureMismatchError inheritance."""
        error = FeatureMismatchError("Feature mismatch")
        assert isinstance(error, DatasetError)
        assert isinstance(error, DomainError)
        assert isinstance(error, PynamolyError)

    def test_basic_creation(self):
        """Test basic feature mismatch error creation."""
        error = FeatureMismatchError("Feature mismatch")
        assert error.message == "Feature mismatch"
        assert error.details == {}

    def test_creation_with_expected_features(self):
        """Test creation with expected features."""
        expected_features = ["age", "income", "score"]
        error = FeatureMismatchError("Feature mismatch", expected_features=expected_features)
        assert error.message == "Feature mismatch"
        assert error.details["expected_features"] == expected_features

    def test_creation_with_actual_features(self):
        """Test creation with actual features."""
        actual_features = ["age", "income"]
        error = FeatureMismatchError("Feature mismatch", actual_features=actual_features)
        assert error.message == "Feature mismatch"
        assert error.details["actual_features"] == actual_features

    def test_creation_with_missing_features(self):
        """Test creation with missing features."""
        missing_features = ["score"]
        error = FeatureMismatchError("Feature mismatch", missing_features=missing_features)
        assert error.message == "Feature mismatch"
        assert error.details["missing_features"] == missing_features

    def test_creation_with_extra_features(self):
        """Test creation with extra features."""
        extra_features = ["bonus"]
        error = FeatureMismatchError("Feature mismatch", extra_features=extra_features)
        assert error.message == "Feature mismatch"
        assert error.details["extra_features"] == extra_features

    def test_creation_with_all_parameters(self):
        """Test creation with all parameters."""
        expected_features = ["age", "income", "score"]
        actual_features = ["age", "income", "bonus"]
        missing_features = ["score"]
        extra_features = ["bonus"]
        
        error = FeatureMismatchError(
            "Feature mismatch",
            expected_features=expected_features,
            actual_features=actual_features,
            missing_features=missing_features,
            extra_features=extra_features
        )
        assert error.message == "Feature mismatch"
        assert error.details["expected_features"] == expected_features
        assert error.details["actual_features"] == actual_features
        assert error.details["missing_features"] == missing_features
        assert error.details["extra_features"] == extra_features

    def test_creation_with_additional_details(self):
        """Test creation with additional details."""
        expected_features = ["age", "income"]
        error = FeatureMismatchError(
            "Feature mismatch",
            expected_features=expected_features,
            dataset_name="test_dataset",
            model_name="test_model",
            operation="prediction"
        )
        assert error.message == "Feature mismatch"
        assert error.details["expected_features"] == expected_features
        assert error.details["dataset_name"] == "test_dataset"
        assert error.details["model_name"] == "test_model"
        assert error.details["operation"] == "prediction"

    def test_none_values(self):
        """Test handling of None values."""
        error = FeatureMismatchError(
            "Feature mismatch",
            expected_features=None,
            actual_features=None,
            missing_features=None,
            extra_features=None
        )
        assert error.message == "Feature mismatch"
        assert "expected_features" not in error.details
        assert "actual_features" not in error.details
        assert "missing_features" not in error.details
        assert "extra_features" not in error.details

    def test_empty_list_values(self):
        """Test handling of empty list values."""
        error = FeatureMismatchError(
            "Feature mismatch",
            expected_features=[],
            actual_features=[],
            missing_features=[],
            extra_features=[]
        )
        assert error.message == "Feature mismatch"
        assert error.details["expected_features"] == []
        assert error.details["actual_features"] == []
        assert error.details["missing_features"] == []
        assert error.details["extra_features"] == []

    def test_string_representation(self):
        """Test string representation."""
        expected_features = ["age", "income"]
        missing_features = ["score"]
        error = FeatureMismatchError(
            "Feature mismatch",
            expected_features=expected_features,
            missing_features=missing_features
        )
        result = str(error)
        assert "Feature mismatch" in result
        assert "expected_features=['age', 'income']" in result
        assert "missing_features=['score']" in result

    def test_exception_raising(self):
        """Test that error can be raised."""
        with pytest.raises(FeatureMismatchError):
            raise FeatureMismatchError("Feature mismatch")


class TestDatasetExceptionIntegration:
    """Test integration scenarios for dataset exceptions."""

    def test_exception_hierarchy(self):
        """Test dataset exception hierarchy relationships."""
        # DatasetError inherits from DomainError
        dataset_error = DatasetError("Dataset error")
        assert isinstance(dataset_error, DomainError)
        assert isinstance(dataset_error, PynamolyError)
        
        # DataValidationError inherits from both ValidationError and DatasetError
        validation_error = DataValidationError("Validation error")
        assert isinstance(validation_error, ValidationError)
        assert isinstance(validation_error, DatasetError)
        assert isinstance(validation_error, DomainError)
        assert isinstance(validation_error, PynamolyError)
        
        # Other dataset errors inherit from DatasetError
        insufficient_data_error = InsufficientDataError("test", 10, 100)
        assert isinstance(insufficient_data_error, DatasetError)
        assert isinstance(insufficient_data_error, DomainError)
        assert isinstance(insufficient_data_error, PynamolyError)

    def test_exception_chaining(self):
        """Test exception chaining with dataset errors."""
        original_error = ValueError("Original error")
        dataset_error = DatasetError("Dataset error occurred", cause=original_error)
        
        assert dataset_error.cause == original_error
        assert "Caused by: ValueError: Original error" in str(dataset_error)

    def test_context_building(self):
        """Test building context across dataset exception handling."""
        error = DataValidationError("Invalid data")
        error.with_context(user_id="123", timestamp="2023-01-01")
        error.with_context(batch_id="batch_001")
        
        assert error.details["user_id"] == "123"
        assert error.details["timestamp"] == "2023-01-01"
        assert error.details["batch_id"] == "batch_001"

    def test_complex_error_scenario(self):
        """Test complex error scenario with multiple details."""
        error = FeatureMismatchError(
            "Feature mismatch detected",
            expected_features=["age", "income", "score"],
            actual_features=["age", "income", "bonus"],
            missing_features=["score"],
            extra_features=["bonus"]
        )
        error.with_context(
            dataset_name="production_data",
            model_version="v1.2.3",
            operation="batch_prediction"
        )
        
        result = str(error)
        assert "Feature mismatch detected" in result
        assert "expected_features=['age', 'income', 'score']" in result
        assert "missing_features=['score']" in result
        assert "dataset_name=production_data" in result
        assert "model_version=v1.2.3" in result

    def test_insufficient_data_with_context(self):
        """Test insufficient data error with additional context."""
        error = InsufficientDataError(
            "training_data",
            50,
            1000,
            operation="model_training"
        )
        error.with_context(
            model_type="IsolationForest",
            required_for="anomaly_detection",
            timestamp="2023-01-01T12:00:00Z"
        )
        
        result = str(error)
        assert "Dataset 'training_data' has insufficient data: 50 samples, but 1000 required for model_training" in result
        assert "model_type=IsolationForest" in result
        assert "required_for=anomaly_detection" in result

    def test_nested_exception_handling(self):
        """Test nested exception handling with dataset errors."""
        try:
            try:
                raise ValueError("Data parsing failed")
            except ValueError as e:
                raise DataValidationError("Validation failed", cause=e)
        except DataValidationError as e:
            assert e.cause is not None
            assert isinstance(e.cause, ValueError)
            assert str(e.cause) == "Data parsing failed"

    def test_exception_in_try_catch(self):
        """Test dataset exceptions in try-catch blocks."""
        with pytest.raises(DataTypeError) as exc_info:
            raise DataTypeError(
                "Invalid data type",
                feature="age",
                expected_type="int",
                actual_type="string"
            )
        
        error = exc_info.value
        assert error.message == "Invalid data type"
        assert error.details["feature"] == "age"
        assert error.details["expected_type"] == "int"
        assert error.details["actual_type"] == "string"

    def test_multiple_inheritance_behavior(self):
        """Test multiple inheritance behavior of DataValidationError."""
        error = DataValidationError("Validation failed", dataset_name="test")
        
        # Should have ValidationError behavior
        assert hasattr(error, 'details')
        assert hasattr(error, 'with_context')
        
        # Should have DatasetError behavior
        assert isinstance(error, DatasetError)
        
        # Should properly handle method resolution
        error.with_context(additional_info="test")
        assert error.details["additional_info"] == "test"

    def test_error_serialization_compatibility(self):
        """Test dataset error compatibility with serialization."""
        error = InsufficientDataError(
            "test_dataset",
            10,
            100,
            operation="training"
        )
        error.with_context(model_type="RandomForest")
        
        # Should be able to extract serializable data
        error_data = {
            "message": error.message,
            "details": error.details,
            "type": type(error).__name__
        }
        
        assert "Dataset 'test_dataset' has insufficient data" in error_data["message"]
        assert error_data["details"]["dataset_name"] == "test_dataset"
        assert error_data["details"]["n_samples"] == 10
        assert error_data["details"]["min_required"] == 100
        assert error_data["details"]["operation"] == "training"
        assert error_data["details"]["model_type"] == "RandomForest"
        assert error_data["type"] == "InsufficientDataError"

    def test_error_message_construction(self):
        """Test error message construction in various scenarios."""
        # Test InsufficientDataError message construction
        error1 = InsufficientDataError("dataset1", 5, 10)
        assert "Dataset 'dataset1' has insufficient data: 5 samples, but 10 required" in error1.message
        
        error2 = InsufficientDataError("dataset2", 100, 1000, operation="validation")
        assert "Dataset 'dataset2' has insufficient data: 100 samples, but 1000 required for validation" in error2.message
        
        # Test DataValidationError message construction
        error3 = DataValidationError("Invalid format")
        assert error3.message == "Invalid format"
        
        error4 = DataValidationError("Invalid format", dataset_name="test_data")
        assert error4.message == "Validation failed for dataset 'test_data': Invalid format"

    def test_parameter_validation(self):
        """Test parameter validation in dataset exceptions."""
        # Test that required parameters are properly handled
        error = InsufficientDataError("test", 0, 1)
        assert error.details["dataset_name"] == "test"
        assert error.details["n_samples"] == 0
        assert error.details["min_required"] == 1
        
        # Test that optional parameters are handled correctly
        error_with_operation = InsufficientDataError("test", 0, 1, operation="test_op")
        assert error_with_operation.details["operation"] == "test_op"
        
        error_without_operation = InsufficientDataError("test", 0, 1)
        assert error_without_operation.details["operation"] is None