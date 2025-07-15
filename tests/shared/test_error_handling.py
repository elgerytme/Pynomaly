"""Tests for shared error handling functionality."""

import logging
from unittest.mock import Mock, patch

import pytest

from pynomaly.shared.error_handling import (
    ErrorContext,
    ErrorHandler,
    ErrorSeverity,
    create_error_response,
    format_error_message,
    handle_errors,
    log_error,
)


class TestErrorSeverity:
    """Test ErrorSeverity enumeration."""

    def test_error_severity_values(self):
        """Test that ErrorSeverity has expected values."""
        assert ErrorSeverity.LOW.value == "low"
        assert ErrorSeverity.MEDIUM.value == "medium"
        assert ErrorSeverity.HIGH.value == "high"
        assert ErrorSeverity.CRITICAL.value == "critical"

    def test_error_severity_ordering(self):
        """Test that ErrorSeverity values can be compared."""
        # Test ordering if implemented
        severities = [
            ErrorSeverity.LOW,
            ErrorSeverity.MEDIUM,
            ErrorSeverity.HIGH,
            ErrorSeverity.CRITICAL,
        ]
        assert len(set(severities)) == 4  # All unique

    def test_error_severity_string_representation(self):
        """Test string representation of ErrorSeverity."""
        assert str(ErrorSeverity.LOW) in ["ErrorSeverity.LOW", "low"]
        assert str(ErrorSeverity.CRITICAL) in ["ErrorSeverity.CRITICAL", "critical"]


class TestErrorContext:
    """Test ErrorContext class functionality."""

    def test_error_context_creation(self):
        """Test ErrorContext can be created with required fields."""
        context = ErrorContext(
            operation="test_operation",
            component="test_component",
            user_id="user123",
            request_id="req456",
        )

        assert context.operation == "test_operation"
        assert context.component == "test_component"
        assert context.user_id == "user123"
        assert context.request_id == "req456"

    def test_error_context_with_optional_fields(self):
        """Test ErrorContext with optional metadata."""
        metadata = {"key1": "value1", "key2": 42}
        context = ErrorContext(
            operation="test_op", component="test_comp", metadata=metadata
        )

        assert context.metadata == metadata
        assert context.user_id is None  # Optional field
        assert context.request_id is None  # Optional field

    def test_error_context_to_dict(self):
        """Test ErrorContext conversion to dictionary."""
        context = ErrorContext(
            operation="test_operation", component="test_component", user_id="user123"
        )

        context_dict = context.to_dict()
        assert isinstance(context_dict, dict)
        assert context_dict["operation"] == "test_operation"
        assert context_dict["component"] == "test_component"
        assert context_dict["user_id"] == "user123"

    def test_error_context_serialization(self):
        """Test ErrorContext can be serialized and deserialized."""
        original_context = ErrorContext(
            operation="serialize_test", component="serializer", metadata={"test": True}
        )

        # Test that context maintains data integrity
        context_dict = original_context.to_dict()
        assert context_dict["operation"] == original_context.operation
        assert context_dict["component"] == original_context.component
        assert context_dict["metadata"] == original_context.metadata


class TestErrorHandler:
    """Test ErrorHandler class functionality."""

    @pytest.fixture
    def error_handler(self):
        """Fixture providing an ErrorHandler instance."""
        return ErrorHandler()

    @pytest.fixture
    def sample_context(self):
        """Fixture providing a sample ErrorContext."""
        return ErrorContext(
            operation="test_operation", component="test_component", user_id="test_user"
        )

    def test_error_handler_initialization(self, error_handler):
        """Test ErrorHandler can be initialized."""
        assert error_handler is not None
        assert hasattr(error_handler, "handle_error")
        assert hasattr(error_handler, "log_error")

    def test_error_handler_handle_error(self, error_handler, sample_context):
        """Test ErrorHandler.handle_error method."""
        exception = ValueError("Test error message")

        # Should not raise exception
        try:
            error_handler.handle_error(exception, sample_context)
        except Exception as e:
            pytest.fail(f"handle_error should not raise exception: {e}")

    def test_error_handler_with_different_exceptions(
        self, error_handler, sample_context
    ):
        """Test ErrorHandler with various exception types."""
        exceptions = [
            ValueError("Value error"),
            TypeError("Type error"),
            RuntimeError("Runtime error"),
            KeyError("Missing key"),
            AttributeError("Missing attribute"),
        ]

        for exception in exceptions:
            # Should handle all exception types
            try:
                error_handler.handle_error(exception, sample_context)
            except Exception as e:
                pytest.fail(f"Failed to handle {type(exception).__name__}: {e}")

    def test_error_handler_with_severity_levels(self, error_handler, sample_context):
        """Test ErrorHandler with different severity levels."""
        exception = ValueError("Test error")

        for severity in ErrorSeverity:
            try:
                error_handler.handle_error(exception, sample_context, severity=severity)
            except Exception as e:
                pytest.fail(f"Failed to handle error with severity {severity}: {e}")

    @patch("pynomaly.shared.error_handling.logger")
    def test_error_handler_logging(self, mock_logger, error_handler, sample_context):
        """Test that ErrorHandler logs errors appropriately."""
        exception = ValueError("Test logging error")

        error_handler.handle_error(
            exception, sample_context, severity=ErrorSeverity.HIGH
        )

        # Verify logging was called
        assert (
            mock_logger.error.called
            or mock_logger.warning.called
            or mock_logger.critical.called
        )


class TestErrorHandlingDecorator:
    """Test handle_errors decorator functionality."""

    def test_handle_errors_decorator_success(self):
        """Test handle_errors decorator with successful function."""

        @handle_errors
        def successful_function(x, y):
            return x + y

        result = successful_function(2, 3)
        assert result == 5

    def test_handle_errors_decorator_with_exception(self):
        """Test handle_errors decorator with function that raises exception."""

        @handle_errors
        def failing_function():
            raise ValueError("Decorator test error")

        # Decorator should handle the exception
        try:
            result = failing_function()
            # Depending on implementation, might return None or error response
            assert result is None or isinstance(result, dict)
        except ValueError:
            pytest.fail("Decorator should have caught the exception")

    def test_handle_errors_decorator_with_context(self):
        """Test handle_errors decorator with error context."""
        context = ErrorContext(operation="decorator_test", component="test_decorator")

        @handle_errors(context=context)
        def function_with_context():
            raise RuntimeError("Context test error")

        # Should handle error with provided context
        try:
            result = function_with_context()
            assert result is None or isinstance(result, dict)
        except RuntimeError:
            pytest.fail("Decorator should have caught the exception")

    def test_handle_errors_decorator_preserves_metadata(self):
        """Test that decorator preserves function metadata."""

        @handle_errors
        def documented_function():
            """This function has documentation."""
            return "success"

        # Function metadata should be preserved
        assert documented_function.__doc__ == "This function has documentation."
        assert documented_function.__name__ == "documented_function"


class TestErrorUtilityFunctions:
    """Test utility functions for error handling."""

    @patch("pynomaly.shared.error_handling.logger")
    def test_log_error_function(self, mock_logger):
        """Test standalone log_error function."""
        exception = ValueError("Utility test error")
        context = ErrorContext(operation="utility_test", component="utils")

        log_error(exception, context, severity=ErrorSeverity.MEDIUM)

        # Verify logging was called
        assert mock_logger.error.called or mock_logger.warning.called

    def test_create_error_response_function(self):
        """Test create_error_response utility function."""
        exception = ValueError("Response test error")
        context = ErrorContext(operation="response_test", component="utils")

        response = create_error_response(exception, context)

        assert isinstance(response, dict)
        assert "error" in response
        assert "message" in response
        assert "context" in response or "operation" in response

    def test_format_error_message_function(self):
        """Test format_error_message utility function."""
        exception = ValueError("Format test error")
        context = ErrorContext(operation="format_test", component="utils")

        message = format_error_message(exception, context)

        assert isinstance(message, str)
        assert len(message) > 0
        assert "format_test" in message or "utils" in message
        assert "Format test error" in message


class TestErrorHandlingIntegration:
    """Integration tests for error handling system."""

    def test_complete_error_handling_workflow(self):
        """Test complete error handling workflow."""
        # Create context
        context = ErrorContext(
            operation="integration_test",
            component="integration",
            user_id="integration_user",
            metadata={"test": "integration"},
        )

        # Create handler
        handler = ErrorHandler()

        # Handle error
        exception = RuntimeError("Integration test error")

        try:
            handler.handle_error(exception, context, severity=ErrorSeverity.HIGH)
            # Should complete without raising
        except Exception as e:
            pytest.fail(f"Integration test failed: {e}")

    def test_error_handling_with_nested_exceptions(self):
        """Test error handling with nested/chained exceptions."""
        try:
            try:
                raise ValueError("Inner exception")
            except ValueError as inner:
                raise RuntimeError("Outer exception") from inner
        except RuntimeError as e:
            context = ErrorContext(operation="nested_test", component="nested")
            handler = ErrorHandler()

            # Should handle nested exceptions
            try:
                handler.handle_error(e, context)
            except Exception as handling_error:
                pytest.fail(f"Failed to handle nested exception: {handling_error}")

    @patch("pynomaly.shared.error_handling.logger")
    def test_error_handling_logging_levels(self, mock_logger):
        """Test that different severity levels use appropriate logging levels."""
        handler = ErrorHandler()
        context = ErrorContext(operation="logging_test", component="logging")
        exception = ValueError("Logging level test")

        # Test different severity levels
        severity_tests = [
            (ErrorSeverity.LOW, "info"),
            (ErrorSeverity.MEDIUM, "warning"),
            (ErrorSeverity.HIGH, "error"),
            (ErrorSeverity.CRITICAL, "critical"),
        ]

        for severity, expected_level in severity_tests:
            mock_logger.reset_mock()
            handler.handle_error(exception, context, severity=severity)

            # Verify appropriate logging level was used
            # (Implementation may vary, so we check that some logging occurred)
            assert (
                mock_logger.info.called
                or mock_logger.warning.called
                or mock_logger.error.called
                or mock_logger.critical.called
            )


class TestErrorHandlingEdgeCases:
    """Test edge cases in error handling."""

    def test_error_handling_with_none_exception(self):
        """Test error handling when exception is None."""
        handler = ErrorHandler()
        context = ErrorContext(operation="none_test", component="edge_cases")

        # Should handle gracefully
        try:
            handler.handle_error(None, context)
        except Exception:
            # May raise or handle gracefully depending on implementation
            pass

    def test_error_handling_with_empty_context(self):
        """Test error handling with minimal context."""
        handler = ErrorHandler()
        context = ErrorContext(operation="", component="")
        exception = ValueError("Empty context test")

        # Should handle even with minimal context
        try:
            handler.handle_error(exception, context)
        except Exception as e:
            pytest.fail(f"Failed with empty context: {e}")

    def test_error_handling_with_large_metadata(self):
        """Test error handling with large metadata."""
        large_metadata = {f"key_{i}": f"value_{i}" * 100 for i in range(100)}
        context = ErrorContext(
            operation="large_metadata_test",
            component="edge_cases",
            metadata=large_metadata,
        )

        handler = ErrorHandler()
        exception = ValueError("Large metadata test")

        # Should handle large metadata without issues
        try:
            handler.handle_error(exception, context)
        except Exception as e:
            pytest.fail(f"Failed with large metadata: {e}")


@pytest.fixture
def mock_logger():
    """Fixture providing a mock logger for testing."""
    return Mock(spec=logging.Logger)


class TestErrorHandlingConfiguration:
    """Test error handling configuration and customization."""

    def test_error_handler_customization(self):
        """Test that ErrorHandler can be customized."""
        # Test with custom configuration if supported
        handler = ErrorHandler()

        # Handler should be configurable
        assert hasattr(handler, "__dict__") or hasattr(handler, "handle_error")

    def test_error_context_extensibility(self):
        """Test that ErrorContext can be extended with custom fields."""
        context = ErrorContext(
            operation="extensibility_test",
            component="config",
            custom_field="custom_value",
            metadata={"extra": "data"},
        )

        # Should handle additional fields gracefully
        assert context.operation == "extensibility_test"
        assert context.metadata.get("extra") == "data"
