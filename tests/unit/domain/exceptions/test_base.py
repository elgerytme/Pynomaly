"""Tests for base domain exceptions."""

import pytest

from pynomaly.domain.exceptions.base import (
    AuthenticationError,
    AuthorizationError,
    CacheError,
    ConfigurationError,
    DomainError,
    InfrastructureError,
    InvalidValueError,
    NotFittedError,
    PynamolyError,
    ValidationError,
)


class TestPynamolyError:
    """Test suite for PynamolyError base class."""

    def test_basic_creation(self):
        """Test basic error creation."""
        error = PynamolyError("Test error message")
        assert str(error) == "Test error message"
        assert error.message == "Test error message"
        assert error.details == {}
        assert error.cause is None

    def test_creation_with_details(self):
        """Test error creation with details."""
        details = {"key": "value", "number": 42}
        error = PynamolyError("Test error", details=details)
        assert error.message == "Test error"
        assert error.details == details
        assert error.cause is None

    def test_creation_with_cause(self):
        """Test error creation with cause."""
        original_error = ValueError("Original error")
        error = PynamolyError("Test error", cause=original_error)
        assert error.message == "Test error"
        assert error.cause == original_error

    def test_creation_with_details_and_cause(self):
        """Test error creation with both details and cause."""
        details = {"key": "value"}
        original_error = ValueError("Original error")
        error = PynamolyError("Test error", details=details, cause=original_error)
        assert error.message == "Test error"
        assert error.details == details
        assert error.cause == original_error

    def test_string_representation_basic(self):
        """Test string representation with basic message."""
        error = PynamolyError("Test error")
        assert str(error) == "Test error"

    def test_string_representation_with_details(self):
        """Test string representation with details."""
        details = {"field": "age", "value": -5}
        error = PynamolyError("Invalid value", details=details)
        result = str(error)
        assert "Invalid value" in result
        assert "Details:" in result
        assert "field=age" in result
        assert "value=-5" in result

    def test_string_representation_with_cause(self):
        """Test string representation with cause."""
        original_error = ValueError("Original error")
        error = PynamolyError("Test error", cause=original_error)
        result = str(error)
        assert "Test error" in result
        assert "Caused by:" in result
        assert "ValueError: Original error" in result

    def test_string_representation_with_details_and_cause(self):
        """Test string representation with both details and cause."""
        details = {"field": "age"}
        original_error = ValueError("Original error")
        error = PynamolyError("Test error", details=details, cause=original_error)
        result = str(error)
        assert "Test error" in result
        assert "Details:" in result
        assert "field=age" in result
        assert "Caused by:" in result
        assert "ValueError: Original error" in result

    def test_with_context_method(self):
        """Test with_context method."""
        error = PynamolyError("Test error")
        result = error.with_context(user_id="123", action="login")
        
        # Should return the same instance
        assert result is error
        assert error.details["user_id"] == "123"
        assert error.details["action"] == "login"

    def test_with_context_updates_existing_details(self):
        """Test with_context updates existing details."""
        error = PynamolyError("Test error", details={"existing": "value"})
        error.with_context(new_key="new_value")
        
        assert error.details["existing"] == "value"
        assert error.details["new_key"] == "new_value"

    def test_with_context_overwrites_existing_keys(self):
        """Test with_context overwrites existing keys."""
        error = PynamolyError("Test error", details={"key": "old_value"})
        error.with_context(key="new_value")
        
        assert error.details["key"] == "new_value"

    def test_inheritance_from_exception(self):
        """Test inheritance from Exception."""
        error = PynamolyError("Test error")
        assert isinstance(error, Exception)
        assert isinstance(error, PynamolyError)

    def test_exception_raising(self):
        """Test that error can be raised."""
        with pytest.raises(PynamolyError, match="Test error"):
            raise PynamolyError("Test error")

    def test_none_details_handling(self):
        """Test handling of None details."""
        error = PynamolyError("Test error", details=None)
        assert error.details == {}


class TestDomainError:
    """Test suite for DomainError."""

    def test_inheritance(self):
        """Test DomainError inheritance."""
        error = DomainError("Domain error")
        assert isinstance(error, PynamolyError)
        assert isinstance(error, DomainError)

    def test_basic_functionality(self):
        """Test basic functionality."""
        error = DomainError("Domain error", details={"key": "value"})
        assert error.message == "Domain error"
        assert error.details == {"key": "value"}

    def test_with_context(self):
        """Test with_context method."""
        error = DomainError("Domain error")
        error.with_context(context="test")
        assert error.details["context"] == "test"

    def test_string_representation(self):
        """Test string representation."""
        error = DomainError("Domain error")
        assert str(error) == "Domain error"


class TestValidationError:
    """Test suite for ValidationError."""

    def test_basic_creation(self):
        """Test basic validation error creation."""
        error = ValidationError("Validation failed")
        assert error.message == "Validation failed"
        assert error.details == {}

    def test_creation_with_field(self):
        """Test creation with field parameter."""
        error = ValidationError("Invalid value", field="age")
        assert error.message == "Invalid value"
        assert error.details["field"] == "age"

    def test_creation_with_value(self):
        """Test creation with value parameter."""
        error = ValidationError("Invalid value", value=-5)
        assert error.message == "Invalid value"
        assert error.details["value"] == -5

    def test_creation_with_field_and_value(self):
        """Test creation with field and value parameters."""
        error = ValidationError("Invalid value", field="age", value=-5)
        assert error.message == "Invalid value"
        assert error.details["field"] == "age"
        assert error.details["value"] == -5

    def test_creation_with_additional_details(self):
        """Test creation with additional details."""
        error = ValidationError(
            "Invalid value",
            field="age",
            value=-5,
            min_value=0,
            max_value=120
        )
        assert error.details["field"] == "age"
        assert error.details["value"] == -5
        assert error.details["min_value"] == 0
        assert error.details["max_value"] == 120

    def test_inheritance(self):
        """Test ValidationError inheritance."""
        error = ValidationError("Validation failed")
        assert isinstance(error, DomainError)
        assert isinstance(error, PynamolyError)
        assert isinstance(error, ValidationError)

    def test_none_field_and_value(self):
        """Test handling of None field and value."""
        error = ValidationError("Validation failed", field=None, value=None)
        assert error.message == "Validation failed"
        assert "field" not in error.details
        assert "value" not in error.details


class TestInvalidValueError:
    """Test suite for InvalidValueError."""

    def test_inheritance(self):
        """Test InvalidValueError inheritance."""
        error = InvalidValueError("Invalid value")
        assert isinstance(error, ValidationError)
        assert isinstance(error, DomainError)
        assert isinstance(error, PynamolyError)

    def test_basic_functionality(self):
        """Test basic functionality."""
        error = InvalidValueError("Invalid value", field="age", value=-5)
        assert error.message == "Invalid value"
        assert error.details["field"] == "age"
        assert error.details["value"] == -5


class TestNotFittedError:
    """Test suite for NotFittedError."""

    def test_default_creation(self):
        """Test creation with default message."""
        error = NotFittedError()
        assert error.message == "Detector must be fitted before use"
        assert error.details == {}

    def test_custom_message(self):
        """Test creation with custom message."""
        error = NotFittedError("Custom message")
        assert error.message == "Custom message"

    def test_creation_with_detector_name(self):
        """Test creation with detector name."""
        error = NotFittedError(detector_name="IsolationForest")
        assert error.message == "Detector must be fitted before use"
        assert error.details["detector_name"] == "IsolationForest"

    def test_creation_with_message_and_detector_name(self):
        """Test creation with both message and detector name."""
        error = NotFittedError("Custom message", detector_name="IsolationForest")
        assert error.message == "Custom message"
        assert error.details["detector_name"] == "IsolationForest"

    def test_creation_with_additional_details(self):
        """Test creation with additional details."""
        error = NotFittedError(
            detector_name="IsolationForest",
            model_type="anomaly_detector",
            expected_state="fitted"
        )
        assert error.details["detector_name"] == "IsolationForest"
        assert error.details["model_type"] == "anomaly_detector"
        assert error.details["expected_state"] == "fitted"

    def test_inheritance(self):
        """Test NotFittedError inheritance."""
        error = NotFittedError()
        assert isinstance(error, DomainError)
        assert isinstance(error, PynamolyError)

    def test_none_detector_name(self):
        """Test handling of None detector name."""
        error = NotFittedError(detector_name=None)
        assert "detector_name" not in error.details


class TestConfigurationError:
    """Test suite for ConfigurationError."""

    def test_basic_creation(self):
        """Test basic configuration error creation."""
        error = ConfigurationError("Invalid configuration")
        assert error.message == "Invalid configuration"
        assert error.details == {}

    def test_creation_with_parameter(self):
        """Test creation with parameter."""
        error = ConfigurationError("Invalid parameter", parameter="batch_size")
        assert error.message == "Invalid parameter"
        assert error.details["parameter"] == "batch_size"

    def test_creation_with_expected_and_actual(self):
        """Test creation with expected and actual values."""
        error = ConfigurationError(
            "Invalid value",
            parameter="batch_size",
            expected="positive integer",
            actual=-5
        )
        assert error.message == "Invalid value"
        assert error.details["parameter"] == "batch_size"
        assert error.details["expected"] == "positive integer"
        assert error.details["actual"] == -5

    def test_creation_with_additional_details(self):
        """Test creation with additional details."""
        error = ConfigurationError(
            "Invalid configuration",
            parameter="batch_size",
            expected="positive integer",
            actual=-5,
            valid_range="1-1000"
        )
        assert error.details["parameter"] == "batch_size"
        assert error.details["expected"] == "positive integer"
        assert error.details["actual"] == -5
        assert error.details["valid_range"] == "1-1000"

    def test_inheritance(self):
        """Test ConfigurationError inheritance."""
        error = ConfigurationError("Invalid configuration")
        assert isinstance(error, DomainError)
        assert isinstance(error, PynamolyError)

    def test_none_values(self):
        """Test handling of None values."""
        error = ConfigurationError(
            "Invalid configuration",
            parameter=None,
            expected=None,
            actual=None
        )
        assert error.message == "Invalid configuration"
        assert "parameter" not in error.details
        assert "expected" not in error.details
        assert "actual" not in error.details


class TestAuthenticationError:
    """Test suite for AuthenticationError."""

    def test_default_creation(self):
        """Test creation with default message."""
        error = AuthenticationError()
        assert error.message == "Authentication failed"
        assert error.details == {}

    def test_custom_message(self):
        """Test creation with custom message."""
        error = AuthenticationError("Custom auth error")
        assert error.message == "Custom auth error"

    def test_creation_with_username(self):
        """Test creation with username."""
        error = AuthenticationError(username="testuser")
        assert error.message == "Authentication failed"
        assert error.details["username"] == "testuser"

    def test_creation_with_reason(self):
        """Test creation with reason."""
        error = AuthenticationError(reason="Invalid password")
        assert error.message == "Authentication failed"
        assert error.details["reason"] == "Invalid password"

    def test_creation_with_all_parameters(self):
        """Test creation with all parameters."""
        error = AuthenticationError(
            "Login failed",
            username="testuser",
            reason="Invalid password"
        )
        assert error.message == "Login failed"
        assert error.details["username"] == "testuser"
        assert error.details["reason"] == "Invalid password"

    def test_creation_with_additional_details(self):
        """Test creation with additional details."""
        error = AuthenticationError(
            username="testuser",
            reason="Invalid password",
            attempts=3,
            locked=True
        )
        assert error.details["username"] == "testuser"
        assert error.details["reason"] == "Invalid password"
        assert error.details["attempts"] == 3
        assert error.details["locked"] is True

    def test_inheritance(self):
        """Test AuthenticationError inheritance."""
        error = AuthenticationError()
        assert isinstance(error, PynamolyError)
        assert not isinstance(error, DomainError)

    def test_none_values(self):
        """Test handling of None values."""
        error = AuthenticationError(username=None, reason=None)
        assert "username" not in error.details
        assert "reason" not in error.details


class TestAuthorizationError:
    """Test suite for AuthorizationError."""

    def test_default_creation(self):
        """Test creation with default message."""
        error = AuthorizationError()
        assert error.message == "Authorization failed"
        assert error.details == {}

    def test_custom_message(self):
        """Test creation with custom message."""
        error = AuthorizationError("Access denied")
        assert error.message == "Access denied"

    def test_creation_with_user_id(self):
        """Test creation with user ID."""
        error = AuthorizationError(user_id="user123")
        assert error.message == "Authorization failed"
        assert error.details["user_id"] == "user123"

    def test_creation_with_required_permission(self):
        """Test creation with required permission."""
        error = AuthorizationError(required_permission="read_data")
        assert error.message == "Authorization failed"
        assert error.details["required_permission"] == "read_data"

    def test_creation_with_required_role(self):
        """Test creation with required role."""
        error = AuthorizationError(required_role="admin")
        assert error.message == "Authorization failed"
        assert error.details["required_role"] == "admin"

    def test_creation_with_all_parameters(self):
        """Test creation with all parameters."""
        error = AuthorizationError(
            "Access denied",
            user_id="user123",
            required_permission="read_data",
            required_role="admin"
        )
        assert error.message == "Access denied"
        assert error.details["user_id"] == "user123"
        assert error.details["required_permission"] == "read_data"
        assert error.details["required_role"] == "admin"

    def test_creation_with_additional_details(self):
        """Test creation with additional details."""
        error = AuthorizationError(
            user_id="user123",
            required_permission="read_data",
            current_role="user",
            resource="dataset_1"
        )
        assert error.details["user_id"] == "user123"
        assert error.details["required_permission"] == "read_data"
        assert error.details["current_role"] == "user"
        assert error.details["resource"] == "dataset_1"

    def test_inheritance(self):
        """Test AuthorizationError inheritance."""
        error = AuthorizationError()
        assert isinstance(error, PynamolyError)
        assert not isinstance(error, DomainError)

    def test_none_values(self):
        """Test handling of None values."""
        error = AuthorizationError(
            user_id=None,
            required_permission=None,
            required_role=None
        )
        assert "user_id" not in error.details
        assert "required_permission" not in error.details
        assert "required_role" not in error.details


class TestCacheError:
    """Test suite for CacheError."""

    def test_default_creation(self):
        """Test creation with default message."""
        error = CacheError()
        assert error.message == "Cache operation failed"
        assert error.details == {}

    def test_custom_message(self):
        """Test creation with custom message."""
        error = CacheError("Redis connection failed")
        assert error.message == "Redis connection failed"

    def test_creation_with_operation(self):
        """Test creation with operation."""
        error = CacheError(operation="get")
        assert error.message == "Cache operation failed"
        assert error.details["operation"] == "get"

    def test_creation_with_key(self):
        """Test creation with key."""
        error = CacheError(key="user:123")
        assert error.message == "Cache operation failed"
        assert error.details["key"] == "user:123"

    def test_creation_with_all_parameters(self):
        """Test creation with all parameters."""
        error = CacheError(
            "Cache set failed",
            operation="set",
            key="user:123"
        )
        assert error.message == "Cache set failed"
        assert error.details["operation"] == "set"
        assert error.details["key"] == "user:123"

    def test_creation_with_additional_details(self):
        """Test creation with additional details."""
        error = CacheError(
            operation="get",
            key="user:123",
            timeout=30,
            retry_count=3
        )
        assert error.details["operation"] == "get"
        assert error.details["key"] == "user:123"
        assert error.details["timeout"] == 30
        assert error.details["retry_count"] == 3

    def test_inheritance(self):
        """Test CacheError inheritance."""
        error = CacheError()
        assert isinstance(error, PynamolyError)
        assert not isinstance(error, DomainError)

    def test_none_values(self):
        """Test handling of None values."""
        error = CacheError(operation=None, key=None)
        assert "operation" not in error.details
        assert "key" not in error.details


class TestInfrastructureError:
    """Test suite for InfrastructureError."""

    def test_default_creation(self):
        """Test creation with default message."""
        error = InfrastructureError()
        assert error.message == "Infrastructure operation failed"
        assert error.details == {}

    def test_custom_message(self):
        """Test creation with custom message."""
        error = InfrastructureError("Database connection failed")
        assert error.message == "Database connection failed"

    def test_creation_with_component(self):
        """Test creation with component."""
        error = InfrastructureError(component="database")
        assert error.message == "Infrastructure operation failed"
        assert error.details["component"] == "database"

    def test_creation_with_operation(self):
        """Test creation with operation."""
        error = InfrastructureError(operation="connect")
        assert error.message == "Infrastructure operation failed"
        assert error.details["operation"] == "connect"

    def test_creation_with_all_parameters(self):
        """Test creation with all parameters."""
        error = InfrastructureError(
            "Database connection failed",
            component="database",
            operation="connect"
        )
        assert error.message == "Database connection failed"
        assert error.details["component"] == "database"
        assert error.details["operation"] == "connect"

    def test_creation_with_additional_details(self):
        """Test creation with additional details."""
        error = InfrastructureError(
            component="database",
            operation="connect",
            host="localhost",
            port=5432,
            timeout=30
        )
        assert error.details["component"] == "database"
        assert error.details["operation"] == "connect"
        assert error.details["host"] == "localhost"
        assert error.details["port"] == 5432
        assert error.details["timeout"] == 30

    def test_inheritance(self):
        """Test InfrastructureError inheritance."""
        error = InfrastructureError()
        assert isinstance(error, PynamolyError)
        assert not isinstance(error, DomainError)

    def test_none_values(self):
        """Test handling of None values."""
        error = InfrastructureError(component=None, operation=None)
        assert "component" not in error.details
        assert "operation" not in error.details


class TestExceptionIntegration:
    """Test integration scenarios for exceptions."""

    def test_exception_chaining(self):
        """Test exception chaining with cause."""
        original_error = ValueError("Original error")
        domain_error = DomainError("Domain error occurred", cause=original_error)
        
        assert domain_error.cause == original_error
        assert "Caused by: ValueError: Original error" in str(domain_error)

    def test_context_building(self):
        """Test building context across exception handling."""
        error = ValidationError("Invalid input")
        error.with_context(user_id="123", action="create_model")
        error.with_context(timestamp="2023-01-01T12:00:00Z")
        
        assert error.details["user_id"] == "123"
        assert error.details["action"] == "create_model"
        assert error.details["timestamp"] == "2023-01-01T12:00:00Z"

    def test_nested_exception_handling(self):
        """Test nested exception handling."""
        try:
            try:
                raise ValueError("Inner error")
            except ValueError as e:
                raise ConfigurationError("Configuration failed", cause=e)
        except ConfigurationError as e:
            assert e.cause is not None
            assert isinstance(e.cause, ValueError)
            assert str(e.cause) == "Inner error"

    def test_exception_hierarchy(self):
        """Test exception hierarchy relationships."""
        # Domain errors inherit from PynamolyError
        domain_error = DomainError("Domain error")
        assert isinstance(domain_error, PynamolyError)
        
        # Validation errors inherit from DomainError
        validation_error = ValidationError("Validation error")
        assert isinstance(validation_error, DomainError)
        assert isinstance(validation_error, PynamolyError)
        
        # InvalidValueError inherits from ValidationError
        invalid_value_error = InvalidValueError("Invalid value")
        assert isinstance(invalid_value_error, ValidationError)
        assert isinstance(invalid_value_error, DomainError)
        assert isinstance(invalid_value_error, PynamolyError)

    def test_exception_in_try_catch(self):
        """Test exceptions in try-catch blocks."""
        with pytest.raises(ValidationError) as exc_info:
            raise ValidationError("Test validation error", field="age", value=-5)
        
        error = exc_info.value
        assert error.message == "Test validation error"
        assert error.details["field"] == "age"
        assert error.details["value"] == -5

    def test_exception_message_formatting(self):
        """Test complex exception message formatting."""
        error = ConfigurationError(
            "Invalid configuration",
            parameter="batch_size",
            expected="positive integer",
            actual=-5
        )
        error.with_context(file="config.yaml", line=42)
        
        result = str(error)
        assert "Invalid configuration" in result
        assert "Details:" in result
        assert "parameter=batch_size" in result
        assert "expected=positive integer" in result
        assert "actual=-5" in result
        assert "file=config.yaml" in result
        assert "line=42" in result

    def test_exception_with_complex_cause(self):
        """Test exception with complex cause chain."""
        original_error = ConnectionError("Network unreachable")
        cache_error = CacheError("Cache connection failed", cause=original_error)
        infrastructure_error = InfrastructureError(
            "Service unavailable",
            component="cache",
            cause=cache_error
        )
        
        result = str(infrastructure_error)
        assert "Service unavailable" in result
        assert "component=cache" in result
        assert "Caused by: CacheError: Cache connection failed" in result

    def test_error_details_immutability(self):
        """Test that error details can be modified after creation."""
        error = PynamolyError("Test error", details={"initial": "value"})
        original_details = error.details
        
        # Add new context
        error.with_context(new_key="new_value")
        
        # Should be the same dictionary object (mutable)
        assert error.details is original_details
        assert error.details["new_key"] == "new_value"

    def test_error_serialization_compatibility(self):
        """Test error compatibility with serialization."""
        error = ValidationError(
            "Validation failed",
            field="age",
            value=-5
        )
        error.with_context(user_id="123")
        
        # Should be able to extract serializable data
        error_data = {
            "message": error.message,
            "details": error.details,
            "type": type(error).__name__
        }
        
        assert error_data["message"] == "Validation failed"
        assert error_data["details"]["field"] == "age"
        assert error_data["details"]["value"] == -5
        assert error_data["details"]["user_id"] == "123"
        assert error_data["type"] == "ValidationError"