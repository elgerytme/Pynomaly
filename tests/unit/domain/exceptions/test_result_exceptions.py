"""Tests for result domain exceptions."""

import pytest

from pynomaly.domain.exceptions.base import DomainError, PynamolyError
from pynomaly.domain.exceptions.result_exceptions import (
    InconsistentResultError,
    ResultError,
    ScoreCalculationError,
    ThresholdError,
)


class TestResultError:
    """Test suite for ResultError base class."""

    def test_inheritance(self):
        """Test ResultError inheritance."""
        error = ResultError("Result error")
        assert isinstance(error, DomainError)
        assert isinstance(error, PynamolyError)
        assert isinstance(error, ResultError)

    def test_basic_functionality(self):
        """Test basic functionality."""
        error = ResultError("Result error", details={"key": "value"})
        assert error.message == "Result error"
        assert error.details == {"key": "value"}

    def test_with_context(self):
        """Test with_context method."""
        error = ResultError("Result error")
        error.with_context(context="test")
        assert error.details["context"] == "test"

    def test_string_representation(self):
        """Test string representation."""
        error = ResultError("Result error")
        assert str(error) == "Result error"

    def test_exception_raising(self):
        """Test that error can be raised."""
        with pytest.raises(ResultError, match="Result error"):
            raise ResultError("Result error")


class TestScoreCalculationError:
    """Test suite for ScoreCalculationError."""

    def test_inheritance(self):
        """Test ScoreCalculationError inheritance."""
        error = ScoreCalculationError("Score calculation failed")
        assert isinstance(error, ResultError)
        assert isinstance(error, DomainError)
        assert isinstance(error, PynamolyError)

    def test_basic_creation(self):
        """Test basic score calculation error creation."""
        error = ScoreCalculationError("Score calculation failed")
        assert error.message == "Score calculation failed"
        assert error.details == {}

    def test_creation_with_detector_name(self):
        """Test creation with detector name."""
        error = ScoreCalculationError(
            "Score calculation failed", detector_name="IsolationForest"
        )
        assert error.message == "Score calculation failed"
        assert error.details["detector_name"] == "IsolationForest"

    def test_creation_with_dataset_name(self):
        """Test creation with dataset name."""
        error = ScoreCalculationError(
            "Score calculation failed", dataset_name="test_dataset"
        )
        assert error.message == "Score calculation failed"
        assert error.details["dataset_name"] == "test_dataset"

    def test_creation_with_both_names(self):
        """Test creation with both detector and dataset names."""
        error = ScoreCalculationError(
            "Score calculation failed",
            detector_name="IsolationForest",
            dataset_name="test_dataset",
        )
        assert error.message == "Score calculation failed"
        assert error.details["detector_name"] == "IsolationForest"
        assert error.details["dataset_name"] == "test_dataset"

    def test_creation_with_additional_details(self):
        """Test creation with additional details."""
        error = ScoreCalculationError(
            "Score calculation failed",
            detector_name="IsolationForest",
            dataset_name="test_dataset",
            n_samples=1000,
            n_features=50,
            contamination=0.1,
        )
        assert error.message == "Score calculation failed"
        assert error.details["detector_name"] == "IsolationForest"
        assert error.details["dataset_name"] == "test_dataset"
        assert error.details["n_samples"] == 1000
        assert error.details["n_features"] == 50
        assert error.details["contamination"] == 0.1

    def test_none_values(self):
        """Test handling of None values."""
        error = ScoreCalculationError(
            "Score calculation failed", detector_name=None, dataset_name=None
        )
        assert error.message == "Score calculation failed"
        assert "detector_name" not in error.details
        assert "dataset_name" not in error.details

    def test_empty_string_values(self):
        """Test handling of empty string values."""
        error = ScoreCalculationError(
            "Score calculation failed", detector_name="", dataset_name=""
        )
        assert error.message == "Score calculation failed"
        assert error.details["detector_name"] == ""
        assert error.details["dataset_name"] == ""

    def test_string_representation(self):
        """Test string representation."""
        error = ScoreCalculationError(
            "Score calculation failed",
            detector_name="IsolationForest",
            dataset_name="test_dataset",
        )
        result = str(error)
        assert "Score calculation failed" in result
        assert "detector_name=IsolationForest" in result
        assert "dataset_name=test_dataset" in result

    def test_exception_raising(self):
        """Test that error can be raised."""
        with pytest.raises(ScoreCalculationError):
            raise ScoreCalculationError("Score calculation failed")


class TestThresholdError:
    """Test suite for ThresholdError."""

    def test_inheritance(self):
        """Test ThresholdError inheritance."""
        error = ThresholdError("Threshold calculation failed")
        assert isinstance(error, ResultError)
        assert isinstance(error, DomainError)
        assert isinstance(error, PynamolyError)

    def test_basic_creation(self):
        """Test basic threshold error creation."""
        error = ThresholdError("Threshold calculation failed")
        assert error.message == "Threshold calculation failed"
        assert error.details == {}

    def test_creation_with_threshold_value(self):
        """Test creation with threshold value."""
        error = ThresholdError("Invalid threshold", threshold_value=0.5)
        assert error.message == "Invalid threshold"
        assert error.details["threshold_value"] == 0.5

    def test_creation_with_contamination_rate(self):
        """Test creation with contamination rate."""
        error = ThresholdError("Invalid contamination", contamination_rate=0.1)
        assert error.message == "Invalid contamination"
        assert error.details["contamination_rate"] == 0.1

    def test_creation_with_n_samples(self):
        """Test creation with n_samples."""
        error = ThresholdError("Insufficient samples", n_samples=100)
        assert error.message == "Insufficient samples"
        assert error.details["n_samples"] == 100

    def test_creation_with_all_parameters(self):
        """Test creation with all parameters."""
        error = ThresholdError(
            "Threshold calculation failed",
            threshold_value=0.5,
            contamination_rate=0.1,
            n_samples=1000,
        )
        assert error.message == "Threshold calculation failed"
        assert error.details["threshold_value"] == 0.5
        assert error.details["contamination_rate"] == 0.1
        assert error.details["n_samples"] == 1000

    def test_creation_with_additional_details(self):
        """Test creation with additional details."""
        error = ThresholdError(
            "Threshold calculation failed",
            threshold_value=0.5,
            contamination_rate=0.1,
            n_samples=1000,
            method="quantile",
            detector_name="IsolationForest",
        )
        assert error.message == "Threshold calculation failed"
        assert error.details["threshold_value"] == 0.5
        assert error.details["contamination_rate"] == 0.1
        assert error.details["n_samples"] == 1000
        assert error.details["method"] == "quantile"
        assert error.details["detector_name"] == "IsolationForest"

    def test_none_values(self):
        """Test handling of None values."""
        error = ThresholdError(
            "Threshold calculation failed",
            threshold_value=None,
            contamination_rate=None,
            n_samples=None,
        )
        assert error.message == "Threshold calculation failed"
        assert "threshold_value" not in error.details
        assert "contamination_rate" not in error.details
        assert "n_samples" not in error.details

    def test_zero_values(self):
        """Test handling of zero values."""
        error = ThresholdError(
            "Threshold calculation failed",
            threshold_value=0.0,
            contamination_rate=0.0,
            n_samples=0,
        )
        assert error.message == "Threshold calculation failed"
        assert error.details["threshold_value"] == 0.0
        assert error.details["contamination_rate"] == 0.0
        assert error.details["n_samples"] == 0

    def test_negative_values(self):
        """Test handling of negative values."""
        error = ThresholdError(
            "Invalid threshold",
            threshold_value=-0.5,
            contamination_rate=-0.1,
            n_samples=-100,
        )
        assert error.message == "Invalid threshold"
        assert error.details["threshold_value"] == -0.5
        assert error.details["contamination_rate"] == -0.1
        assert error.details["n_samples"] == -100

    def test_string_representation(self):
        """Test string representation."""
        error = ThresholdError(
            "Threshold calculation failed",
            threshold_value=0.5,
            contamination_rate=0.1,
            n_samples=1000,
        )
        result = str(error)
        assert "Threshold calculation failed" in result
        assert "threshold_value=0.5" in result
        assert "contamination_rate=0.1" in result
        assert "n_samples=1000" in result

    def test_exception_raising(self):
        """Test that error can be raised."""
        with pytest.raises(ThresholdError):
            raise ThresholdError("Threshold calculation failed")


class TestInconsistentResultError:
    """Test suite for InconsistentResultError."""

    def test_inheritance(self):
        """Test InconsistentResultError inheritance."""
        error = InconsistentResultError("Inconsistent result")
        assert isinstance(error, ResultError)
        assert isinstance(error, DomainError)
        assert isinstance(error, PynamolyError)

    def test_basic_creation(self):
        """Test basic inconsistent result error creation."""
        error = InconsistentResultError("Inconsistent result")
        assert error.message == "Inconsistent result"
        assert error.details == {}

    def test_creation_with_n_scores(self):
        """Test creation with n_scores."""
        error = InconsistentResultError("Inconsistent result", n_scores=1000)
        assert error.message == "Inconsistent result"
        assert error.details["n_scores"] == 1000

    def test_creation_with_n_labels(self):
        """Test creation with n_labels."""
        error = InconsistentResultError("Inconsistent result", n_labels=800)
        assert error.message == "Inconsistent result"
        assert error.details["n_labels"] == 800

    def test_creation_with_n_anomalies_expected(self):
        """Test creation with n_anomalies_expected."""
        error = InconsistentResultError("Inconsistent result", n_anomalies_expected=100)
        assert error.message == "Inconsistent result"
        assert error.details["n_anomalies_expected"] == 100

    def test_creation_with_n_anomalies_actual(self):
        """Test creation with n_anomalies_actual."""
        error = InconsistentResultError("Inconsistent result", n_anomalies_actual=120)
        assert error.message == "Inconsistent result"
        assert error.details["n_anomalies_actual"] == 120

    def test_creation_with_all_parameters(self):
        """Test creation with all parameters."""
        error = InconsistentResultError(
            "Inconsistent result",
            n_scores=1000,
            n_labels=800,
            n_anomalies_expected=100,
            n_anomalies_actual=120,
        )
        assert error.message == "Inconsistent result"
        assert error.details["n_scores"] == 1000
        assert error.details["n_labels"] == 800
        assert error.details["n_anomalies_expected"] == 100
        assert error.details["n_anomalies_actual"] == 120

    def test_creation_with_additional_details(self):
        """Test creation with additional details."""
        error = InconsistentResultError(
            "Inconsistent result",
            n_scores=1000,
            n_labels=800,
            n_anomalies_expected=100,
            n_anomalies_actual=120,
            detector_name="IsolationForest",
            dataset_name="test_dataset",
            contamination=0.1,
        )
        assert error.message == "Inconsistent result"
        assert error.details["n_scores"] == 1000
        assert error.details["n_labels"] == 800
        assert error.details["n_anomalies_expected"] == 100
        assert error.details["n_anomalies_actual"] == 120
        assert error.details["detector_name"] == "IsolationForest"
        assert error.details["dataset_name"] == "test_dataset"
        assert error.details["contamination"] == 0.1

    def test_none_values(self):
        """Test handling of None values."""
        error = InconsistentResultError(
            "Inconsistent result",
            n_scores=None,
            n_labels=None,
            n_anomalies_expected=None,
            n_anomalies_actual=None,
        )
        assert error.message == "Inconsistent result"
        assert "n_scores" not in error.details
        assert "n_labels" not in error.details
        assert "n_anomalies_expected" not in error.details
        assert "n_anomalies_actual" not in error.details

    def test_zero_values(self):
        """Test handling of zero values."""
        error = InconsistentResultError(
            "Inconsistent result",
            n_scores=0,
            n_labels=0,
            n_anomalies_expected=0,
            n_anomalies_actual=0,
        )
        assert error.message == "Inconsistent result"
        assert error.details["n_scores"] == 0
        assert error.details["n_labels"] == 0
        assert error.details["n_anomalies_expected"] == 0
        assert error.details["n_anomalies_actual"] == 0

    def test_negative_values(self):
        """Test handling of negative values."""
        error = InconsistentResultError(
            "Inconsistent result",
            n_scores=-1,
            n_labels=-1,
            n_anomalies_expected=-1,
            n_anomalies_actual=-1,
        )
        assert error.message == "Inconsistent result"
        assert error.details["n_scores"] == -1
        assert error.details["n_labels"] == -1
        assert error.details["n_anomalies_expected"] == -1
        assert error.details["n_anomalies_actual"] == -1

    def test_string_representation(self):
        """Test string representation."""
        error = InconsistentResultError(
            "Inconsistent result",
            n_scores=1000,
            n_labels=800,
            n_anomalies_expected=100,
            n_anomalies_actual=120,
        )
        result = str(error)
        assert "Inconsistent result" in result
        assert "n_scores=1000" in result
        assert "n_labels=800" in result
        assert "n_anomalies_expected=100" in result
        assert "n_anomalies_actual=120" in result

    def test_exception_raising(self):
        """Test that error can be raised."""
        with pytest.raises(InconsistentResultError):
            raise InconsistentResultError("Inconsistent result")


class TestResultExceptionIntegration:
    """Test integration scenarios for result exceptions."""

    def test_exception_hierarchy(self):
        """Test result exception hierarchy relationships."""
        # ResultError inherits from DomainError
        result_error = ResultError("Result error")
        assert isinstance(result_error, DomainError)
        assert isinstance(result_error, PynamolyError)

        # Other result errors inherit from ResultError
        score_error = ScoreCalculationError("Score calculation failed")
        assert isinstance(score_error, ResultError)
        assert isinstance(score_error, DomainError)
        assert isinstance(score_error, PynamolyError)

        threshold_error = ThresholdError("Threshold calculation failed")
        assert isinstance(threshold_error, ResultError)
        assert isinstance(threshold_error, DomainError)
        assert isinstance(threshold_error, PynamolyError)

        inconsistent_error = InconsistentResultError("Inconsistent result")
        assert isinstance(inconsistent_error, ResultError)
        assert isinstance(inconsistent_error, DomainError)
        assert isinstance(inconsistent_error, PynamolyError)

    def test_exception_chaining(self):
        """Test exception chaining with result errors."""
        original_error = ValueError("Original error")
        result_error = ResultError("Result error occurred", cause=original_error)

        assert result_error.cause == original_error
        assert "Caused by: ValueError: Original error" in str(result_error)

    def test_context_building(self):
        """Test building context across result exception handling."""
        error = ScoreCalculationError("Score calculation failed")
        error.with_context(user_id="123", timestamp="2023-01-01")
        error.with_context(operation="batch_scoring")

        assert error.details["user_id"] == "123"
        assert error.details["timestamp"] == "2023-01-01"
        assert error.details["operation"] == "batch_scoring"

    def test_complex_error_scenario(self):
        """Test complex error scenario with multiple details."""
        error = InconsistentResultError(
            "Detection results are inconsistent",
            n_scores=1000,
            n_labels=800,
            n_anomalies_expected=100,
            n_anomalies_actual=120,
        )
        error.with_context(
            detector_name="IsolationForest",
            dataset_name="production_data",
            contamination=0.1,
            threshold_method="quantile",
        )

        result = str(error)
        assert "Detection results are inconsistent" in result
        assert "n_scores=1000" in result
        assert "n_labels=800" in result
        assert "n_anomalies_expected=100" in result
        assert "n_anomalies_actual=120" in result
        assert "detector_name=IsolationForest" in result
        assert "dataset_name=production_data" in result
        assert "contamination=0.1" in result

    def test_threshold_error_with_context(self):
        """Test threshold error with additional context."""
        error = ThresholdError(
            "Threshold calculation failed",
            threshold_value=0.5,
            contamination_rate=0.1,
            n_samples=1000,
        )
        error.with_context(
            method="quantile",
            detector_name="IsolationForest",
            dataset_name="validation_data",
            timestamp="2023-01-01T12:00:00Z",
        )

        result = str(error)
        assert "Threshold calculation failed" in result
        assert "threshold_value=0.5" in result
        assert "contamination_rate=0.1" in result
        assert "n_samples=1000" in result
        assert "method=quantile" in result
        assert "detector_name=IsolationForest" in result

    def test_nested_exception_handling(self):
        """Test nested exception handling with result errors."""
        try:
            try:
                raise ValueError("Mathematical error")
            except ValueError as e:
                raise ScoreCalculationError("Score calculation failed", cause=e)
        except ScoreCalculationError as e:
            assert e.cause is not None
            assert isinstance(e.cause, ValueError)
            assert str(e.cause) == "Mathematical error"

    def test_exception_in_try_catch(self):
        """Test result exceptions in try-catch blocks."""
        with pytest.raises(InconsistentResultError) as exc_info:
            raise InconsistentResultError(
                "Inconsistent result", n_scores=1000, n_labels=800
            )

        error = exc_info.value
        assert error.message == "Inconsistent result"
        assert error.details["n_scores"] == 1000
        assert error.details["n_labels"] == 800

    def test_error_serialization_compatibility(self):
        """Test result error compatibility with serialization."""
        error = InconsistentResultError(
            "Inconsistent result",
            n_scores=1000,
            n_labels=800,
            n_anomalies_expected=100,
            n_anomalies_actual=120,
        )
        error.with_context(detector_name="IsolationForest", dataset_name="test_data")

        # Should be able to extract serializable data
        error_data = {
            "message": error.message,
            "details": error.details,
            "type": type(error).__name__,
        }

        assert error_data["message"] == "Inconsistent result"
        assert error_data["details"]["n_scores"] == 1000
        assert error_data["details"]["n_labels"] == 800
        assert error_data["details"]["n_anomalies_expected"] == 100
        assert error_data["details"]["n_anomalies_actual"] == 120
        assert error_data["details"]["detector_name"] == "IsolationForest"
        assert error_data["details"]["dataset_name"] == "test_data"
        assert error_data["type"] == "InconsistentResultError"

    def test_parameter_validation(self):
        """Test parameter validation in result exceptions."""
        # Test ScoreCalculationError required parameters
        error1 = ScoreCalculationError("Score calculation failed")
        assert error1.message == "Score calculation failed"

        # Test optional parameters
        error2 = ScoreCalculationError(
            "Score calculation failed", detector_name="IsolationForest"
        )
        assert error2.details["detector_name"] == "IsolationForest"

        error3 = ScoreCalculationError("Score calculation failed", detector_name=None)
        assert "detector_name" not in error3.details

        # Test ThresholdError required parameters
        error4 = ThresholdError("Threshold calculation failed")
        assert error4.message == "Threshold calculation failed"

        # Test optional parameters
        error5 = ThresholdError("Threshold calculation failed", threshold_value=0.5)
        assert error5.details["threshold_value"] == 0.5

        error6 = ThresholdError("Threshold calculation failed", threshold_value=None)
        assert "threshold_value" not in error6.details

    def test_result_workflow_errors(self):
        """Test result errors in typical workflow scenarios."""
        # Test workflow: score calculation fails
        with pytest.raises(ScoreCalculationError) as exc_info:
            raise ScoreCalculationError(
                "Score calculation failed",
                detector_name="IsolationForest",
                dataset_name="test_data",
            )

        score_error = exc_info.value
        assert "Score calculation failed" in score_error.message
        assert score_error.details["detector_name"] == "IsolationForest"
        assert score_error.details["dataset_name"] == "test_data"

        # Test workflow: threshold calculation fails
        threshold_error = ThresholdError(
            "Invalid threshold value", threshold_value=-0.5, contamination_rate=0.1
        )
        assert "Invalid threshold value" in threshold_error.message
        assert threshold_error.details["threshold_value"] == -0.5
        assert threshold_error.details["contamination_rate"] == 0.1

        # Test workflow: inconsistent results detected
        inconsistent_error = InconsistentResultError(
            "Score and label count mismatch", n_scores=1000, n_labels=800
        )
        assert "Score and label count mismatch" in inconsistent_error.message
        assert inconsistent_error.details["n_scores"] == 1000
        assert inconsistent_error.details["n_labels"] == 800

    def test_error_context_propagation(self):
        """Test error context propagation through result operations."""
        # Create initial error with context
        error = ScoreCalculationError("Score calculation failed")
        error.with_context(
            user_id="user123",
            operation="batch_scoring",
            timestamp="2023-01-01T12:00:00Z",
        )

        # Add additional context
        error.with_context(
            detector_name="IsolationForest",
            dataset_name="production_data",
            n_samples=10000,
        )

        # Verify all context is preserved
        assert error.details["user_id"] == "user123"
        assert error.details["operation"] == "batch_scoring"
        assert error.details["timestamp"] == "2023-01-01T12:00:00Z"
        assert error.details["detector_name"] == "IsolationForest"
        assert error.details["dataset_name"] == "production_data"
        assert error.details["n_samples"] == 10000

        # Test that context is properly formatted in string representation
        result = str(error)
        assert "user_id=user123" in result
        assert "detector_name=IsolationForest" in result
        assert "dataset_name=production_data" in result
        assert "n_samples=10000" in result

    def test_numeric_parameter_handling(self):
        """Test numeric parameter handling in result exceptions."""
        # Test ThresholdError with different numeric types
        error1 = ThresholdError(
            "Threshold error",
            threshold_value=0.5,
            contamination_rate=0.1,
            n_samples=1000,
        )
        assert isinstance(error1.details["threshold_value"], float)
        assert isinstance(error1.details["contamination_rate"], float)
        assert isinstance(error1.details["n_samples"], int)

        # Test InconsistentResultError with different numeric types
        error2 = InconsistentResultError(
            "Inconsistent result",
            n_scores=1000,
            n_labels=800,
            n_anomalies_expected=100,
            n_anomalies_actual=120,
        )
        assert isinstance(error2.details["n_scores"], int)
        assert isinstance(error2.details["n_labels"], int)
        assert isinstance(error2.details["n_anomalies_expected"], int)
        assert isinstance(error2.details["n_anomalies_actual"], int)

        # Test with edge cases
        error3 = ThresholdError(
            "Edge case error",
            threshold_value=float("inf"),
            contamination_rate=0.0,
            n_samples=1,
        )
        assert error3.details["threshold_value"] == float("inf")
        assert error3.details["contamination_rate"] == 0.0
        assert error3.details["n_samples"] == 1

    def test_all_result_error_types_coverage(self):
        """Test that all result error types are properly covered."""
        # Test all result error types
        result_error = ResultError("Generic result error")
        score_error = ScoreCalculationError("Score calculation failed")
        threshold_error = ThresholdError("Threshold calculation failed")
        inconsistent_error = InconsistentResultError("Inconsistent result")

        # Verify inheritance
        assert isinstance(result_error, ResultError)
        assert isinstance(score_error, ResultError)
        assert isinstance(threshold_error, ResultError)
        assert isinstance(inconsistent_error, ResultError)

        # Verify all inherit from DomainError
        assert isinstance(result_error, DomainError)
        assert isinstance(score_error, DomainError)
        assert isinstance(threshold_error, DomainError)
        assert isinstance(inconsistent_error, DomainError)

    def test_error_message_formatting(self):
        """Test error message formatting in various scenarios."""
        # Test with detailed context
        error = InconsistentResultError(
            "Score and label arrays have different lengths", n_scores=1000, n_labels=800
        )
        error.with_context(
            detector_name="IsolationForest",
            dataset_name="production_data",
            operation="batch_detection",
        )

        result = str(error)
        assert "Score and label arrays have different lengths" in result
        assert "Details:" in result
        assert "n_scores=1000" in result
        assert "n_labels=800" in result
        assert "detector_name=IsolationForest" in result
        assert "dataset_name=production_data" in result
        assert "operation=batch_detection" in result

    def test_edge_cases_and_boundaries(self):
        """Test edge cases and boundary conditions."""
        # Test with extreme values
        error1 = ThresholdError(
            "Extreme threshold",
            threshold_value=float("inf"),
            contamination_rate=1.0,
            n_samples=1,
        )
        assert error1.details["threshold_value"] == float("inf")
        assert error1.details["contamination_rate"] == 1.0
        assert error1.details["n_samples"] == 1

        # Test with negative values
        error2 = InconsistentResultError(
            "Negative counts",
            n_scores=-1,
            n_labels=-1,
            n_anomalies_expected=-1,
            n_anomalies_actual=-1,
        )
        assert error2.details["n_scores"] == -1
        assert error2.details["n_labels"] == -1
        assert error2.details["n_anomalies_expected"] == -1
        assert error2.details["n_anomalies_actual"] == -1

        # Test with very large values
        error3 = InconsistentResultError(
            "Large values",
            n_scores=1000000,
            n_labels=1000000,
            n_anomalies_expected=100000,
            n_anomalies_actual=100000,
        )
        assert error3.details["n_scores"] == 1000000
        assert error3.details["n_labels"] == 1000000
        assert error3.details["n_anomalies_expected"] == 100000
        assert error3.details["n_anomalies_actual"] == 100000
