"""Tests for detector domain exceptions."""

import pytest

from pynomaly.domain.exceptions.detector_exceptions import (
    DetectorConfigurationError,
    DetectorError,
    DetectorNotFittedError,
    FittingError,
    InvalidAlgorithmError,
)
from pynomaly.domain.exceptions.base import (
    ConfigurationError,
    DomainError,
    NotFittedError,
    PynamolyError,
)


class TestDetectorError:
    """Test suite for DetectorError base class."""

    def test_inheritance(self):
        """Test DetectorError inheritance."""
        error = DetectorError("Detector error")
        assert isinstance(error, DomainError)
        assert isinstance(error, PynamolyError)
        assert isinstance(error, DetectorError)

    def test_basic_functionality(self):
        """Test basic functionality."""
        error = DetectorError("Detector error", details={"key": "value"})
        assert error.message == "Detector error"
        assert error.details == {"key": "value"}

    def test_with_context(self):
        """Test with_context method."""
        error = DetectorError("Detector error")
        error.with_context(context="test")
        assert error.details["context"] == "test"

    def test_string_representation(self):
        """Test string representation."""
        error = DetectorError("Detector error")
        assert str(error) == "Detector error"

    def test_exception_raising(self):
        """Test that error can be raised."""
        with pytest.raises(DetectorError, match="Detector error"):
            raise DetectorError("Detector error")


class TestDetectorNotFittedError:
    """Test suite for DetectorNotFittedError."""

    def test_multiple_inheritance(self):
        """Test DetectorNotFittedError multiple inheritance."""
        error = DetectorNotFittedError("IsolationForest")
        assert isinstance(error, NotFittedError)
        assert isinstance(error, DetectorError)
        assert isinstance(error, DomainError)
        assert isinstance(error, PynamolyError)

    def test_default_creation(self):
        """Test creation with default operation."""
        error = DetectorNotFittedError("IsolationForest")
        expected_message = "Detector 'IsolationForest' must be fitted before calling detect()"
        assert error.message == expected_message
        assert error.details["detector_name"] == "IsolationForest"
        assert error.details["operation"] == "detect"

    def test_creation_with_custom_operation(self):
        """Test creation with custom operation."""
        error = DetectorNotFittedError("IsolationForest", operation="predict")
        expected_message = "Detector 'IsolationForest' must be fitted before calling predict()"
        assert error.message == expected_message
        assert error.details["detector_name"] == "IsolationForest"
        assert error.details["operation"] == "predict"

    def test_creation_with_additional_details(self):
        """Test creation with additional details."""
        error = DetectorNotFittedError(
            "IsolationForest",
            operation="predict",
            model_version="v1.0",
            dataset_name="test_data"
        )
        expected_message = "Detector 'IsolationForest' must be fitted before calling predict()"
        assert error.message == expected_message
        assert error.details["detector_name"] == "IsolationForest"
        assert error.details["operation"] == "predict"
        assert error.details["model_version"] == "v1.0"
        assert error.details["dataset_name"] == "test_data"

    def test_empty_detector_name(self):
        """Test with empty detector name."""
        error = DetectorNotFittedError("")
        expected_message = "Detector '' must be fitted before calling detect()"
        assert error.message == expected_message
        assert error.details["detector_name"] == ""

    def test_empty_operation(self):
        """Test with empty operation."""
        error = DetectorNotFittedError("IsolationForest", operation="")
        expected_message = "Detector 'IsolationForest' must be fitted before calling ()"
        assert error.message == expected_message
        assert error.details["operation"] == ""

    def test_string_representation(self):
        """Test string representation."""
        error = DetectorNotFittedError("IsolationForest", operation="predict")
        result = str(error)
        assert "Detector 'IsolationForest' must be fitted before calling predict()" in result
        assert "detector_name=IsolationForest" in result
        assert "operation=predict" in result

    def test_exception_raising(self):
        """Test that error can be raised."""
        with pytest.raises(DetectorNotFittedError):
            raise DetectorNotFittedError("IsolationForest")


class TestDetectorConfigurationError:
    """Test suite for DetectorConfigurationError."""

    def test_multiple_inheritance(self):
        """Test DetectorConfigurationError multiple inheritance."""
        error = DetectorConfigurationError("IsolationForest", "Invalid parameter")
        assert isinstance(error, ConfigurationError)
        assert isinstance(error, DetectorError)
        assert isinstance(error, DomainError)
        assert isinstance(error, PynamolyError)

    def test_basic_creation(self):
        """Test basic detector configuration error creation."""
        error = DetectorConfigurationError("IsolationForest", "Invalid parameter")
        expected_message = "Configuration error in detector 'IsolationForest': Invalid parameter"
        assert error.message == expected_message
        assert error.details["detector_name"] == "IsolationForest"

    def test_creation_with_additional_details(self):
        """Test creation with additional details."""
        error = DetectorConfigurationError(
            "IsolationForest",
            "Invalid parameter",
            parameter="n_estimators",
            expected="positive integer",
            actual=-10
        )
        expected_message = "Configuration error in detector 'IsolationForest': Invalid parameter"
        assert error.message == expected_message
        assert error.details["detector_name"] == "IsolationForest"
        assert error.details["parameter"] == "n_estimators"
        assert error.details["expected"] == "positive integer"
        assert error.details["actual"] == -10

    def test_empty_detector_name(self):
        """Test with empty detector name."""
        error = DetectorConfigurationError("", "Invalid parameter")
        expected_message = "Configuration error in detector '': Invalid parameter"
        assert error.message == expected_message
        assert error.details["detector_name"] == ""

    def test_empty_message(self):
        """Test with empty message."""
        error = DetectorConfigurationError("IsolationForest", "")
        expected_message = "Configuration error in detector 'IsolationForest': "
        assert error.message == expected_message

    def test_string_representation(self):
        """Test string representation."""
        error = DetectorConfigurationError(
            "IsolationForest",
            "Invalid parameter",
            parameter="n_estimators"
        )
        result = str(error)
        assert "Configuration error in detector 'IsolationForest': Invalid parameter" in result
        assert "detector_name=IsolationForest" in result
        assert "parameter=n_estimators" in result

    def test_exception_raising(self):
        """Test that error can be raised."""
        with pytest.raises(DetectorConfigurationError):
            raise DetectorConfigurationError("IsolationForest", "Invalid parameter")


class TestInvalidAlgorithmError:
    """Test suite for InvalidAlgorithmError."""

    def test_inheritance(self):
        """Test InvalidAlgorithmError inheritance."""
        error = InvalidAlgorithmError("InvalidAlgorithm")
        assert isinstance(error, DetectorError)
        assert isinstance(error, DomainError)
        assert isinstance(error, PynamolyError)

    def test_basic_creation(self):
        """Test basic invalid algorithm error creation."""
        error = InvalidAlgorithmError("InvalidAlgorithm")
        expected_message = "Algorithm 'InvalidAlgorithm' is not supported"
        assert error.message == expected_message
        assert error.details["algorithm_name"] == "InvalidAlgorithm"

    def test_creation_with_available_algorithms(self):
        """Test creation with available algorithms."""
        available_algorithms = ["IsolationForest", "OneClassSVM", "LocalOutlierFactor"]
        error = InvalidAlgorithmError("InvalidAlgorithm", available_algorithms=available_algorithms)
        expected_message = "Algorithm 'InvalidAlgorithm' is not supported. Available algorithms: IsolationForest, OneClassSVM, LocalOutlierFactor"
        assert error.message == expected_message
        assert error.details["algorithm_name"] == "InvalidAlgorithm"
        assert error.details["available_algorithms"] == available_algorithms

    def test_creation_with_empty_available_algorithms(self):
        """Test creation with empty available algorithms."""
        error = InvalidAlgorithmError("InvalidAlgorithm", available_algorithms=[])
        expected_message = "Algorithm 'InvalidAlgorithm' is not supported. Available algorithms: "
        assert error.message == expected_message
        assert error.details["algorithm_name"] == "InvalidAlgorithm"
        assert error.details["available_algorithms"] == []

    def test_creation_with_additional_details(self):
        """Test creation with additional details."""
        available_algorithms = ["IsolationForest", "OneClassSVM"]
        error = InvalidAlgorithmError(
            "InvalidAlgorithm",
            available_algorithms=available_algorithms,
            category="anomaly_detection",
            version="1.0"
        )
        expected_message = "Algorithm 'InvalidAlgorithm' is not supported. Available algorithms: IsolationForest, OneClassSVM"
        assert error.message == expected_message
        assert error.details["algorithm_name"] == "InvalidAlgorithm"
        assert error.details["available_algorithms"] == available_algorithms
        assert error.details["category"] == "anomaly_detection"
        assert error.details["version"] == "1.0"

    def test_none_available_algorithms(self):
        """Test with None available algorithms."""
        error = InvalidAlgorithmError("InvalidAlgorithm", available_algorithms=None)
        expected_message = "Algorithm 'InvalidAlgorithm' is not supported"
        assert error.message == expected_message
        assert error.details["algorithm_name"] == "InvalidAlgorithm"
        assert "available_algorithms" not in error.details

    def test_empty_algorithm_name(self):
        """Test with empty algorithm name."""
        error = InvalidAlgorithmError("")
        expected_message = "Algorithm '' is not supported"
        assert error.message == expected_message
        assert error.details["algorithm_name"] == ""

    def test_single_available_algorithm(self):
        """Test with single available algorithm."""
        error = InvalidAlgorithmError("InvalidAlgorithm", available_algorithms=["IsolationForest"])
        expected_message = "Algorithm 'InvalidAlgorithm' is not supported. Available algorithms: IsolationForest"
        assert error.message == expected_message
        assert error.details["available_algorithms"] == ["IsolationForest"]

    def test_string_representation(self):
        """Test string representation."""
        available_algorithms = ["IsolationForest", "OneClassSVM"]
        error = InvalidAlgorithmError("InvalidAlgorithm", available_algorithms=available_algorithms)
        result = str(error)
        assert "Algorithm 'InvalidAlgorithm' is not supported" in result
        assert "available_algorithms=['IsolationForest', 'OneClassSVM']" in result

    def test_exception_raising(self):
        """Test that error can be raised."""
        with pytest.raises(InvalidAlgorithmError):
            raise InvalidAlgorithmError("InvalidAlgorithm")


class TestFittingError:
    """Test suite for FittingError."""

    def test_inheritance(self):
        """Test FittingError inheritance."""
        error = FittingError("IsolationForest", "Insufficient data")
        assert isinstance(error, DetectorError)
        assert isinstance(error, DomainError)
        assert isinstance(error, PynamolyError)

    def test_basic_creation(self):
        """Test basic fitting error creation."""
        error = FittingError("IsolationForest", "Insufficient data")
        expected_message = "Failed to fit detector 'IsolationForest': Insufficient data"
        assert error.message == expected_message
        assert error.details["detector_name"] == "IsolationForest"
        assert error.details["reason"] == "Insufficient data"

    def test_creation_with_dataset_name(self):
        """Test creation with dataset name."""
        error = FittingError("IsolationForest", "Insufficient data", dataset_name="training_data")
        expected_message = "Failed to fit detector 'IsolationForest': Insufficient data"
        assert error.message == expected_message
        assert error.details["detector_name"] == "IsolationForest"
        assert error.details["reason"] == "Insufficient data"
        assert error.details["dataset_name"] == "training_data"

    def test_creation_with_additional_details(self):
        """Test creation with additional details."""
        error = FittingError(
            "IsolationForest",
            "Insufficient data",
            dataset_name="training_data",
            n_samples=50,
            min_required=1000,
            contamination=0.1
        )
        expected_message = "Failed to fit detector 'IsolationForest': Insufficient data"
        assert error.message == expected_message
        assert error.details["detector_name"] == "IsolationForest"
        assert error.details["reason"] == "Insufficient data"
        assert error.details["dataset_name"] == "training_data"
        assert error.details["n_samples"] == 50
        assert error.details["min_required"] == 1000
        assert error.details["contamination"] == 0.1

    def test_creation_without_dataset_name(self):
        """Test creation without dataset name."""
        error = FittingError("IsolationForest", "Insufficient data", dataset_name=None)
        expected_message = "Failed to fit detector 'IsolationForest': Insufficient data"
        assert error.message == expected_message
        assert error.details["detector_name"] == "IsolationForest"
        assert error.details["reason"] == "Insufficient data"
        assert "dataset_name" not in error.details

    def test_empty_detector_name(self):
        """Test with empty detector name."""
        error = FittingError("", "Insufficient data")
        expected_message = "Failed to fit detector '': Insufficient data"
        assert error.message == expected_message
        assert error.details["detector_name"] == ""

    def test_empty_reason(self):
        """Test with empty reason."""
        error = FittingError("IsolationForest", "")
        expected_message = "Failed to fit detector 'IsolationForest': "
        assert error.message == expected_message
        assert error.details["reason"] == ""

    def test_empty_dataset_name(self):
        """Test with empty dataset name."""
        error = FittingError("IsolationForest", "Insufficient data", dataset_name="")
        expected_message = "Failed to fit detector 'IsolationForest': Insufficient data"
        assert error.message == expected_message
        assert error.details["dataset_name"] == ""

    def test_string_representation(self):
        """Test string representation."""
        error = FittingError(
            "IsolationForest",
            "Insufficient data",
            dataset_name="training_data"
        )
        result = str(error)
        assert "Failed to fit detector 'IsolationForest': Insufficient data" in result
        assert "detector_name=IsolationForest" in result
        assert "reason=Insufficient data" in result
        assert "dataset_name=training_data" in result

    def test_exception_raising(self):
        """Test that error can be raised."""
        with pytest.raises(FittingError):
            raise FittingError("IsolationForest", "Insufficient data")


class TestDetectorExceptionIntegration:
    """Test integration scenarios for detector exceptions."""

    def test_exception_hierarchy(self):
        """Test detector exception hierarchy relationships."""
        # DetectorError inherits from DomainError
        detector_error = DetectorError("Detector error")
        assert isinstance(detector_error, DomainError)
        assert isinstance(detector_error, PynamolyError)
        
        # DetectorNotFittedError inherits from both NotFittedError and DetectorError
        not_fitted_error = DetectorNotFittedError("IsolationForest")
        assert isinstance(not_fitted_error, NotFittedError)
        assert isinstance(not_fitted_error, DetectorError)
        assert isinstance(not_fitted_error, DomainError)
        assert isinstance(not_fitted_error, PynamolyError)
        
        # DetectorConfigurationError inherits from both ConfigurationError and DetectorError
        config_error = DetectorConfigurationError("IsolationForest", "Invalid parameter")
        assert isinstance(config_error, ConfigurationError)
        assert isinstance(config_error, DetectorError)
        assert isinstance(config_error, DomainError)
        assert isinstance(config_error, PynamolyError)
        
        # Other detector errors inherit from DetectorError
        invalid_algorithm_error = InvalidAlgorithmError("InvalidAlgorithm")
        assert isinstance(invalid_algorithm_error, DetectorError)
        assert isinstance(invalid_algorithm_error, DomainError)
        assert isinstance(invalid_algorithm_error, PynamolyError)
        
        fitting_error = FittingError("IsolationForest", "Failed")
        assert isinstance(fitting_error, DetectorError)
        assert isinstance(fitting_error, DomainError)
        assert isinstance(fitting_error, PynamolyError)

    def test_exception_chaining(self):
        """Test exception chaining with detector errors."""
        original_error = ValueError("Original error")
        detector_error = DetectorError("Detector error occurred", cause=original_error)
        
        assert detector_error.cause == original_error
        assert "Caused by: ValueError: Original error" in str(detector_error)

    def test_context_building(self):
        """Test building context across detector exception handling."""
        error = DetectorNotFittedError("IsolationForest")
        error.with_context(user_id="123", timestamp="2023-01-01")
        error.with_context(model_version="v1.0")
        
        assert error.details["user_id"] == "123"
        assert error.details["timestamp"] == "2023-01-01"
        assert error.details["model_version"] == "v1.0"

    def test_complex_error_scenario(self):
        """Test complex error scenario with multiple details."""
        error = FittingError(
            "IsolationForest",
            "Insufficient data for training",
            dataset_name="training_data"
        )
        error.with_context(
            n_samples=50,
            min_required=1000,
            contamination=0.1,
            operation="batch_training"
        )
        
        result = str(error)
        assert "Failed to fit detector 'IsolationForest': Insufficient data for training" in result
        assert "dataset_name=training_data" in result
        assert "n_samples=50" in result
        assert "min_required=1000" in result
        assert "contamination=0.1" in result

    def test_invalid_algorithm_with_context(self):
        """Test invalid algorithm error with additional context."""
        available_algorithms = ["IsolationForest", "OneClassSVM", "LocalOutlierFactor"]
        error = InvalidAlgorithmError(
            "UnsupportedAlgorithm",
            available_algorithms=available_algorithms
        )
        error.with_context(
            requested_by="user123",
            category="anomaly_detection",
            version="1.0",
            timestamp="2023-01-01T12:00:00Z"
        )
        
        result = str(error)
        assert "Algorithm 'UnsupportedAlgorithm' is not supported" in result
        assert "available_algorithms=['IsolationForest', 'OneClassSVM', 'LocalOutlierFactor']" in result
        assert "requested_by=user123" in result
        assert "category=anomaly_detection" in result

    def test_nested_exception_handling(self):
        """Test nested exception handling with detector errors."""
        try:
            try:
                raise ValueError("Parameter validation failed")
            except ValueError as e:
                raise DetectorConfigurationError("IsolationForest", "Invalid configuration", cause=e)
        except DetectorConfigurationError as e:
            assert e.cause is not None
            assert isinstance(e.cause, ValueError)
            assert str(e.cause) == "Parameter validation failed"

    def test_exception_in_try_catch(self):
        """Test detector exceptions in try-catch blocks."""
        with pytest.raises(DetectorNotFittedError) as exc_info:
            raise DetectorNotFittedError("IsolationForest", operation="predict")
        
        error = exc_info.value
        assert "Detector 'IsolationForest' must be fitted before calling predict()" in error.message
        assert error.details["detector_name"] == "IsolationForest"
        assert error.details["operation"] == "predict"

    def test_multiple_inheritance_behavior(self):
        """Test multiple inheritance behavior of detector errors."""
        # Test DetectorNotFittedError multiple inheritance
        error1 = DetectorNotFittedError("IsolationForest")
        
        # Should have NotFittedError behavior
        assert hasattr(error1, 'details')
        assert hasattr(error1, 'with_context')
        
        # Should have DetectorError behavior
        assert isinstance(error1, DetectorError)
        
        # Test DetectorConfigurationError multiple inheritance
        error2 = DetectorConfigurationError("IsolationForest", "Invalid parameter")
        
        # Should have ConfigurationError behavior
        assert isinstance(error2, ConfigurationError)
        
        # Should have DetectorError behavior
        assert isinstance(error2, DetectorError)
        
        # Should properly handle method resolution
        error2.with_context(additional_info="test")
        assert error2.details["additional_info"] == "test"

    def test_error_serialization_compatibility(self):
        """Test detector error compatibility with serialization."""
        error = FittingError(
            "IsolationForest",
            "Insufficient data",
            dataset_name="training_data"
        )
        error.with_context(n_samples=50, min_required=1000)
        
        # Should be able to extract serializable data
        error_data = {
            "message": error.message,
            "details": error.details,
            "type": type(error).__name__
        }
        
        assert "Failed to fit detector 'IsolationForest': Insufficient data" in error_data["message"]
        assert error_data["details"]["detector_name"] == "IsolationForest"
        assert error_data["details"]["reason"] == "Insufficient data"
        assert error_data["details"]["dataset_name"] == "training_data"
        assert error_data["details"]["n_samples"] == 50
        assert error_data["details"]["min_required"] == 1000
        assert error_data["type"] == "FittingError"

    def test_error_message_construction(self):
        """Test error message construction in various scenarios."""
        # Test DetectorNotFittedError message construction
        error1 = DetectorNotFittedError("IsolationForest")
        assert "Detector 'IsolationForest' must be fitted before calling detect()" in error1.message
        
        error2 = DetectorNotFittedError("IsolationForest", operation="predict")
        assert "Detector 'IsolationForest' must be fitted before calling predict()" in error2.message
        
        # Test DetectorConfigurationError message construction
        error3 = DetectorConfigurationError("IsolationForest", "Invalid parameter")
        assert "Configuration error in detector 'IsolationForest': Invalid parameter" in error3.message
        
        # Test InvalidAlgorithmError message construction
        error4 = InvalidAlgorithmError("InvalidAlgorithm")
        assert "Algorithm 'InvalidAlgorithm' is not supported" in error4.message
        
        error5 = InvalidAlgorithmError("InvalidAlgorithm", available_algorithms=["IsolationForest"])
        assert "Algorithm 'InvalidAlgorithm' is not supported. Available algorithms: IsolationForest" in error5.message
        
        # Test FittingError message construction
        error6 = FittingError("IsolationForest", "Insufficient data")
        assert "Failed to fit detector 'IsolationForest': Insufficient data" in error6.message

    def test_parameter_validation(self):
        """Test parameter validation in detector exceptions."""
        # Test DetectorNotFittedError required parameters
        error1 = DetectorNotFittedError("IsolationForest")
        assert error1.details["detector_name"] == "IsolationForest"
        assert error1.details["operation"] == "detect"
        
        # Test DetectorConfigurationError required parameters
        error2 = DetectorConfigurationError("IsolationForest", "Invalid parameter")
        assert error2.details["detector_name"] == "IsolationForest"
        
        # Test InvalidAlgorithmError required parameters
        error3 = InvalidAlgorithmError("InvalidAlgorithm")
        assert error3.details["algorithm_name"] == "InvalidAlgorithm"
        
        # Test FittingError required parameters
        error4 = FittingError("IsolationForest", "Failed")
        assert error4.details["detector_name"] == "IsolationForest"
        assert error4.details["reason"] == "Failed"
        
        # Test optional parameters
        error5 = FittingError("IsolationForest", "Failed", dataset_name="test")
        assert error5.details["dataset_name"] == "test"
        
        error6 = FittingError("IsolationForest", "Failed", dataset_name=None)
        assert "dataset_name" not in error6.details

    def test_detector_workflow_errors(self):
        """Test detector errors in typical workflow scenarios."""
        # Test workflow: trying to use unfitted detector
        with pytest.raises(DetectorNotFittedError) as exc_info:
            raise DetectorNotFittedError("IsolationForest", operation="detect")
        
        unfitted_error = exc_info.value
        assert "must be fitted before calling detect()" in unfitted_error.message
        
        # Test workflow: invalid configuration during setup
        config_error = DetectorConfigurationError(
            "IsolationForest",
            "Invalid contamination parameter",
            parameter="contamination",
            expected="float between 0 and 1",
            actual=1.5
        )
        assert "Configuration error in detector 'IsolationForest'" in config_error.message
        
        # Test workflow: fitting fails
        fitting_error = FittingError(
            "IsolationForest",
            "Insufficient data for training",
            dataset_name="small_dataset"
        )
        assert "Failed to fit detector 'IsolationForest'" in fitting_error.message
        
        # Test workflow: invalid algorithm selection
        algorithm_error = InvalidAlgorithmError(
            "NonExistentAlgorithm",
            available_algorithms=["IsolationForest", "OneClassSVM"]
        )
        assert "Algorithm 'NonExistentAlgorithm' is not supported" in algorithm_error.message

    def test_error_context_propagation(self):
        """Test error context propagation through detector operations."""
        # Create initial error with context
        error = DetectorConfigurationError("IsolationForest", "Invalid parameter")
        error.with_context(
            user_id="user123",
            operation="model_training",
            timestamp="2023-01-01T12:00:00Z"
        )
        
        # Add additional context
        error.with_context(
            parameter="n_estimators",
            expected_type="int",
            actual_value=-10
        )
        
        # Verify all context is preserved
        assert error.details["user_id"] == "user123"
        assert error.details["operation"] == "model_training"
        assert error.details["timestamp"] == "2023-01-01T12:00:00Z"
        assert error.details["parameter"] == "n_estimators"
        assert error.details["expected_type"] == "int"
        assert error.details["actual_value"] == -10
        
        # Test that context is properly formatted in string representation
        result = str(error)
        assert "user_id=user123" in result
        assert "parameter=n_estimators" in result
        assert "actual_value=-10" in result