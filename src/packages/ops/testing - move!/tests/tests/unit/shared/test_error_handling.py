"""Tests for error handling utilities."""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from monorepo.shared.error_handling import (
    ConfigurationError,
    PerformanceError,
    ValidationError,
    create_user_friendly_message,
    handle_cli_errors,
    safe_import,
    validate_algorithm_name,
    validate_contamination_rate,
    validate_data_format,
    validate_data_shape,
    validate_file_exists,
)


class TestPynomaliError:
    """Test suite for PynomaliError base class."""

    def test_basic_error_creation(self):
        """Test basic error creation."""
        error = PynomaliError("Test error message")
        assert str(error) == "Test error message"
        assert error.message == "Test error message"
        assert error.details == {}

    def test_error_with_details(self):
        """Test error creation with details."""
        details = {"key": "value", "number": 42}
        error = PynomaliError("Test error", details=details)
        assert error.message == "Test error"
        assert error.details == details

    def test_error_to_dict(self):
        """Test error to_dict conversion."""
        details = {"field": "invalid"}
        error = PynomaliError("Validation failed", details=details)

        result = error.to_dict()
        expected = {
            "error_type": "PynomaliError",
            "message": "Validation failed",
            "details": details,
        }
        assert result == expected

    def test_error_inheritance(self):
        """Test error inheritance."""
        error = PynomaliError("Base error")
        assert isinstance(error, Exception)
        assert isinstance(error, PynomaliError)


class TestSpecificErrors:
    """Test suite for specific error types."""

    def test_validation_error(self):
        """Test ValidationError."""
        error = ValidationError("Invalid value")
        assert isinstance(error, PynomaliError)
        assert str(error) == "Invalid value"

    def test_configuration_error(self):
        """Test ConfigurationError."""
        error = ConfigurationError("Invalid config")
        assert isinstance(error, PynomaliError)
        assert str(error) == "Invalid config"

    def test_resource_not_found_error(self):
        """Test ResourceNotFoundError."""
        error = ResourceNotFoundError("File not found")
        assert isinstance(error, PynomaliError)
        assert str(error) == "File not found"

    def test_performance_error(self):
        """Test PerformanceError."""
        error = PerformanceError("Too slow")
        assert isinstance(error, PynomaliError)
        assert str(error) == "Too slow"

    def test_error_to_dict_inheritance(self):
        """Test to_dict works for inherited errors."""
        error = ValidationError("Invalid input", details={"field": "age"})
        result = error.to_dict()

        assert result["error_type"] == "ValidationError"
        assert result["message"] == "Invalid input"
        assert result["details"]["field"] == "age"


class TestValidateFileExists:
    """Test suite for validate_file_exists function."""

    def test_valid_file_string_path(self):
        """Test validation with valid file using string path."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmp:
            tmp.write("test content")
            tmp.flush()

            result = validate_file_exists(tmp.name)
            assert isinstance(result, Path)
            assert result.exists()
            assert result.is_file()

    def test_valid_file_path_object(self):
        """Test validation with valid file using Path object."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmp:
            tmp.write("test content")
            tmp.flush()

            result = validate_file_exists(Path(tmp.name))
            assert isinstance(result, Path)
            assert result.exists()

    def test_nonexistent_file(self):
        """Test validation with nonexistent file."""
        with pytest.raises(ResourceNotFoundError, match="File not found"):
            validate_file_exists("/nonexistent/file.txt")

    def test_directory_instead_of_file(self):
        """Test validation when path is directory."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            with pytest.raises(ResourceNotFoundError, match="Path is not a file"):
                validate_file_exists(tmp_dir)

    def test_unreadable_file(self):
        """Test validation with unreadable file."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmp:
            tmp.write("test content")
            tmp.flush()

            # Make file unreadable
            Path(tmp.name).chmod(0o000)

            try:
                with pytest.raises(ResourceNotFoundError, match="not readable"):
                    validate_file_exists(tmp.name)
            finally:
                # Restore permissions for cleanup
                Path(tmp.name).chmod(0o644)

    def test_error_details(self):
        """Test error details are included."""
        nonexistent_path = "/nonexistent/file.txt"

        with pytest.raises(ResourceNotFoundError) as exc_info:
            validate_file_exists(nonexistent_path)

        error = exc_info.value
        assert "file_path" in error.details
        assert "absolute_path" in error.details
        assert error.details["file_path"] == nonexistent_path


class TestValidateDataFormat:
    """Test suite for validate_data_format function."""

    def test_csv_format(self):
        """Test CSV format detection."""
        path = Path("test.csv")
        result = validate_data_format(path)
        assert result == "csv"

    def test_json_format(self):
        """Test JSON format detection."""
        path = Path("test.json")
        result = validate_data_format(path)
        assert result == "json"

    def test_parquet_format(self):
        """Test Parquet format detection."""
        path = Path("test.parquet")
        result = validate_data_format(path)
        assert result == "parquet"

    def test_excel_formats(self):
        """Test Excel format detection."""
        path_xlsx = Path("test.xlsx")
        result_xlsx = validate_data_format(path_xlsx)
        assert result_xlsx == "excel"

        path_xls = Path("test.xls")
        result_xls = validate_data_format(path_xls)
        assert result_xls == "excel"

    def test_case_insensitive(self):
        """Test case insensitive format detection."""
        path = Path("test.CSV")
        result = validate_data_format(path)
        assert result == "csv"

    def test_unsupported_format(self):
        """Test unsupported format raises error."""
        path = Path("test.txt")

        with pytest.raises(ValidationError, match="Unsupported file format"):
            validate_data_format(path)

    def test_error_details_for_unsupported(self):
        """Test error details for unsupported format."""
        path = Path("test.txt")

        with pytest.raises(ValidationError) as exc_info:
            validate_data_format(path)

        error = exc_info.value
        assert "file_path" in error.details
        assert "detected_format" in error.details
        assert "supported_formats" in error.details
        assert error.details["detected_format"] == ".txt"


class TestValidateContaminationRate:
    """Test suite for validate_contamination_rate function."""

    def test_valid_float_rate(self):
        """Test valid float contamination rate."""
        result = validate_contamination_rate(0.1)
        assert result == 0.1

    def test_valid_int_rate(self):
        """Test valid integer contamination rate."""
        result = validate_contamination_rate(1)
        assert result == 1.0
        assert isinstance(result, float)

    def test_boundary_values(self):
        """Test boundary values."""
        # Just above 0
        result = validate_contamination_rate(0.001)
        assert result == 0.001

        # Just below 1
        result = validate_contamination_rate(0.999)
        assert result == 0.999

    def test_invalid_type(self):
        """Test invalid type raises error."""
        with pytest.raises(ValidationError, match="must be a number"):
            validate_contamination_rate("0.1")

    def test_invalid_range_zero(self):
        """Test zero contamination rate raises error."""
        with pytest.raises(ValidationError, match="must be between 0.0 and 1.0"):
            validate_contamination_rate(0.0)

    def test_invalid_range_one(self):
        """Test one contamination rate raises error."""
        with pytest.raises(ValidationError, match="must be between 0.0 and 1.0"):
            validate_contamination_rate(1.0)

    def test_invalid_range_negative(self):
        """Test negative contamination rate raises error."""
        with pytest.raises(ValidationError, match="must be between 0.0 and 1.0"):
            validate_contamination_rate(-0.1)

    def test_invalid_range_above_one(self):
        """Test contamination rate above 1 raises error."""
        with pytest.raises(ValidationError, match="must be between 0.0 and 1.0"):
            validate_contamination_rate(1.1)

    def test_error_details(self):
        """Test error details are included."""
        with pytest.raises(ValidationError) as exc_info:
            validate_contamination_rate(1.5)

        error = exc_info.value
        assert "value" in error.details
        assert "valid_range" in error.details
        assert error.details["value"] == 1.5


class TestValidateAlgorithmName:
    """Test suite for validate_algorithm_name function."""

    def test_valid_algorithm_exact_match(self):
        """Test valid algorithm with exact match."""
        available = ["IsolationForest", "LocalOutlierFactor", "OneClassSVM"]
        result = validate_algorithm_name("IsolationForest", available)
        assert result == "IsolationForest"

    def test_valid_algorithm_case_insensitive(self):
        """Test valid algorithm with case insensitive match."""
        available = ["IsolationForest", "LocalOutlierFactor"]
        result = validate_algorithm_name("isolationforest", available)
        assert result == "IsolationForest"

    def test_invalid_type(self):
        """Test non-string algorithm name raises error."""
        available = ["IsolationForest"]

        with pytest.raises(ValidationError, match="must be a string"):
            validate_algorithm_name(123, available)

    def test_unknown_algorithm(self):
        """Test unknown algorithm raises error."""
        available = ["IsolationForest", "LocalOutlierFactor"]

        with pytest.raises(ValidationError, match="Unknown algorithm"):
            validate_algorithm_name("UnknownAlgorithm", available)

    def test_error_details_for_unknown(self):
        """Test error details for unknown algorithm."""
        available = ["IsolationForest", "LocalOutlierFactor"]

        with pytest.raises(ValidationError) as exc_info:
            validate_algorithm_name("UnknownAlgorithm", available)

        error = exc_info.value
        assert "requested_algorithm" in error.details
        assert "available_algorithms" in error.details
        assert "suggestion" in error.details
        assert error.details["requested_algorithm"] == "UnknownAlgorithm"

    def test_empty_available_algorithms(self):
        """Test with empty available algorithms list."""
        with pytest.raises(ValidationError, match="Unknown algorithm"):
            validate_algorithm_name("AnyAlgorithm", [])


class TestValidateDataShape:
    """Test suite for validate_data_shape function."""

    def test_valid_data_shape(self):
        """Test valid data shape."""
        mock_data = Mock()
        mock_data.shape = (100, 5)

        # Should not raise any exception
        validate_data_shape(mock_data)

    def test_custom_minimum_requirements(self):
        """Test with custom minimum requirements."""
        mock_data = Mock()
        mock_data.shape = (50, 3)

        # Should not raise with custom minimums
        validate_data_shape(mock_data, min_samples=10, min_features=2)

    def test_insufficient_samples(self):
        """Test insufficient samples raises error."""
        mock_data = Mock()
        mock_data.shape = (5, 10)

        with pytest.raises(ValidationError, match="Insufficient samples"):
            validate_data_shape(mock_data, min_samples=10)

    def test_insufficient_features(self):
        """Test insufficient features raises error."""
        mock_data = Mock()
        mock_data.shape = (100, 1)

        with pytest.raises(ValidationError, match="Insufficient features"):
            validate_data_shape(mock_data, min_features=2)

    def test_no_shape_attribute(self):
        """Test data without shape attribute raises error."""
        mock_data = Mock()
        del mock_data.shape

        with pytest.raises(ValidationError, match="must have a 'shape' attribute"):
            validate_data_shape(mock_data)

    def test_error_details(self):
        """Test error details are included."""
        mock_data = Mock()
        mock_data.shape = (5, 2)

        with pytest.raises(ValidationError) as exc_info:
            validate_data_shape(mock_data, min_samples=10)

        error = exc_info.value
        assert "current_samples" in error.details
        assert "minimum_required" in error.details
        assert "data_shape" in error.details
        assert error.details["current_samples"] == 5
        assert error.details["minimum_required"] == 10


class TestHandleCliErrors:
    """Test suite for handle_cli_errors decorator."""

    def test_successful_function_call(self):
        """Test decorator with successful function call."""

        @handle_cli_errors
        def successful_func():
            return "success"

        result = successful_func()
        assert result == "success"

    def test_pynomali_error_handling(self):
        """Test decorator handles PynomaliError."""

        @handle_cli_errors
        def failing_func():
            raise ValidationError("Test error", details={"field": "value"})

        with patch("builtins.print") as mock_print:
            result = failing_func()
            assert result is False
            mock_print.assert_called()

    def test_unexpected_error_handling(self):
        """Test decorator handles unexpected errors."""

        @handle_cli_errors
        def failing_func():
            raise ValueError("Unexpected error")

        with patch("builtins.print") as mock_print:
            result = failing_func()
            assert result is False
            mock_print.assert_called()

    def test_decorator_preserves_function_attributes(self):
        """Test decorator preserves function attributes."""

        @handle_cli_errors
        def test_func():
            """Test function."""
            pass

        assert test_func.__name__ == "test_func"
        assert test_func.__doc__ == "Test function."


class TestHandleApiErrors:
    """Test suite for handle_api_errors decorator."""

    def test_successful_sync_function(self):
        """Test decorator with successful sync function."""

        @handle_api_errors
        def successful_func():
            return "success"

        result = successful_func()
        assert result == "success"

    def test_successful_async_function(self):
        """Test decorator with successful async function."""

        @handle_api_errors
        async def successful_async_func():
            return "async_success"

        result = asyncio.run(successful_async_func())
        assert result == "async_success"

    def test_pynomali_error_passthrough(self):
        """Test decorator passes through PynomaliError."""

        @handle_api_errors
        def failing_func():
            raise ValidationError("Test error")

        with pytest.raises(ValidationError):
            failing_func()

    def test_unexpected_error_wrapping(self):
        """Test decorator wraps unexpected errors."""

        @handle_api_errors
        def failing_func():
            raise ValueError("Unexpected error")

        with pytest.raises(PynomaliError, match="Internal server error"):
            failing_func()

    def test_async_error_handling(self):
        """Test decorator handles async errors."""

        @handle_api_errors
        async def failing_async_func():
            raise ValueError("Async error")

        with pytest.raises(PynomaliError, match="Internal server error"):
            asyncio.run(failing_async_func())


class TestSafeImport:
    """Test suite for safe_import function."""

    def test_successful_import(self):
        """Test successful module import."""
        result = safe_import("os")
        import os

        assert result == os

    def test_failed_import_default_message(self):
        """Test failed import with default message."""
        with pytest.raises(
            ConfigurationError, match="Required dependency.*not installed"
        ):
            safe_import("nonexistent_module")

    def test_failed_import_custom_message(self):
        """Test failed import with custom message."""
        custom_message = "Custom error message"

        with pytest.raises(ConfigurationError, match=custom_message):
            safe_import("nonexistent_module", custom_message)

    def test_error_details(self):
        """Test error details are included."""
        with pytest.raises(ConfigurationError) as exc_info:
            safe_import("nonexistent_module")

        error = exc_info.value
        assert "module_name" in error.details
        assert "import_error" in error.details
        assert "suggestion" in error.details
        assert error.details["module_name"] == "nonexistent_module"


class TestCreateUserFriendlyMessage:
    """Test suite for create_user_friendly_message function."""

    def test_pynomali_error_message(self):
        """Test message for PynomaliError."""
        error = PynomaliError("Custom error message")
        result = create_user_friendly_message(error)
        assert result == "Custom error message"

    def test_file_not_found_error(self):
        """Test message for FileNotFoundError."""
        error = FileNotFoundError("File not found")
        result = create_user_friendly_message(error)
        assert "file could not be found" in result
        assert "check the file path" in result

    def test_permission_error(self):
        """Test message for PermissionError."""
        error = PermissionError("Permission denied")
        result = create_user_friendly_message(error)
        assert "Permission denied" in result
        assert "read access" in result

    def test_memory_error(self):
        """Test message for MemoryError."""
        error = MemoryError("Out of memory")
        result = create_user_friendly_message(error)
        assert "Not enough memory" in result
        assert "smaller dataset" in result

    def test_import_error(self):
        """Test message for ImportError."""
        error = ImportError("Module not found")
        result = create_user_friendly_message(error)
        assert "not installed" in result
        assert "pip install" in result

    def test_unknown_error_type(self):
        """Test message for unknown error type."""
        error = RuntimeError("Runtime error")
        result = create_user_friendly_message(error)
        assert "RuntimeError" in result
        assert "Runtime error" in result

    def test_generic_error_types(self):
        """Test messages for generic error types."""
        test_cases = [
            (ValueError("Invalid value"), "Invalid data value"),
            (KeyError("Missing key"), "Required data field is missing"),
            (TypeError("Type error"), "Data type mismatch"),
            (ConnectionError("Connection failed"), "Network connection failed"),
            (TimeoutError("Timeout"), "Operation timed out"),
        ]

        for error, expected_text in test_cases:
            result = create_user_friendly_message(error)
            assert expected_text in result


class TestIntegrationScenarios:
    """Test integration scenarios."""

    def test_file_validation_workflow(self):
        """Test complete file validation workflow."""
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
            tmp.write("col1,col2\n1,2\n3,4\n")
            tmp.flush()

            # Test complete workflow
            path = validate_file_exists(tmp.name)
            format_type = validate_data_format(path)

            assert isinstance(path, Path)
            assert format_type == "csv"

    def test_error_chain_handling(self):
        """Test error chain handling."""

        def process_data():
            try:
                validate_contamination_rate(1.5)
            except ValidationError as e:
                raise ConfigurationError("Configuration failed") from e

        with pytest.raises(ConfigurationError):
            process_data()

    def test_decorator_combination(self):
        """Test combining decorators."""

        @handle_cli_errors
        @handle_api_errors
        def combined_func():
            return "success"

        # Should work normally
        result = combined_func()
        assert result == "success"
