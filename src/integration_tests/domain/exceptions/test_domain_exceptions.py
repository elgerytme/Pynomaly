"""
Comprehensive tests for domain exceptions.

This module tests the domain-specific exceptions to ensure proper inheritance,
error handling, and consistent behavior across the domain layer.
"""

import pytest

from pynomaly.domain.exceptions import DomainError, InvalidValueError, ValidationError


class TestDomainError:
    """Test suite for the base domain exception class."""

    def test_domain_error_creation(self):
        """Test basic domain error creation."""
        message = "Test domain error"
        exc = DomainError(message)

        assert str(exc) == message
        assert exc.args[0] == message
        assert isinstance(exc, Exception)

    def test_domain_error_inheritance(self):
        """Test that domain error inherits from Exception."""
        exc = DomainError("test")

        assert isinstance(exc, Exception)
        assert isinstance(exc, DomainError)

    def test_domain_error_without_message(self):
        """Test domain error creation without message."""
        exc = DomainError()

        # Should have some default representation
        assert str(exc) is not None

    def test_domain_error_repr(self):
        """Test domain error representation."""
        message = "Test repr"
        exc = DomainError(message)

        repr_str = repr(exc)
        assert "DomainError" in repr_str
        assert message in repr_str

    def test_domain_error_with_complex_message(self):
        """Test domain error with complex message."""
        complex_message = (
            "Entity validation failed: field 'value' must be positive, got -5"
        )
        exc = DomainError(complex_message)

        assert str(exc) == complex_message
        assert complex_message in repr(exc)


class TestValidationError:
    """Test suite for validation-related exceptions."""

    def test_validation_error_creation(self):
        """Test validation error creation."""
        message = "Invalid field value"
        exc = ValidationError(message)

        assert str(exc) == message
        assert isinstance(exc, DomainError)
        assert isinstance(exc, ValidationError)

    def test_validation_error_inheritance_chain(self):
        """Test inheritance chain for validation error."""
        exc = ValidationError("test")

        assert isinstance(exc, Exception)
        assert isinstance(exc, DomainError)
        assert isinstance(exc, ValidationError)

    def test_validation_error_with_field_context(self):
        """Test validation error with field context."""
        message = "Field validation failed"
        field_name = "anomaly_score"
        field_value = "invalid"

        exc = ValidationError(f"{message}: {field_name}={field_value}")

        assert field_name in str(exc)
        assert field_value in str(exc)

    def test_validation_error_with_constraint_info(self):
        """Test validation error with constraint information."""
        message = "Value out of range: expected 0.0-1.0, got 1.5"
        exc = ValidationError(message)

        assert "0.0-1.0" in str(exc)
        assert "1.5" in str(exc)

    def test_validation_error_with_type_info(self):
        """Test validation error with type information."""
        message = "Type mismatch: expected float, got str"
        exc = ValidationError(message)

        assert "expected float" in str(exc)
        assert "got str" in str(exc)


class TestInvalidValueError:
    """Test suite for invalid value exceptions."""

    def test_invalid_value_error_creation(self):
        """Test invalid value error creation."""
        message = "Invalid value provided"
        exc = InvalidValueError(message)

        assert str(exc) == message
        assert isinstance(exc, DomainError)
        assert isinstance(exc, InvalidValueError)

    def test_invalid_value_error_inheritance_chain(self):
        """Test inheritance chain for invalid value error."""
        exc = InvalidValueError("test")

        assert isinstance(exc, Exception)
        assert isinstance(exc, DomainError)
        assert isinstance(exc, InvalidValueError)

    def test_invalid_value_error_with_value_info(self):
        """Test invalid value error with value information."""
        message = "Invalid contamination rate: -0.1"
        exc = InvalidValueError(message)

        assert "-0.1" in str(exc)
        assert "contamination rate" in str(exc)

    def test_invalid_value_error_with_range_info(self):
        """Test invalid value error with range information."""
        message = "Threshold must be between 0 and 1, got 2.5"
        exc = InvalidValueError(message)

        assert "0 and 1" in str(exc)
        assert "2.5" in str(exc)

    def test_invalid_value_error_with_enum_info(self):
        """Test invalid value error with enumeration information."""
        message = (
            "Invalid method: 'custom'. Valid options: ['percentile', 'fixed', 'iqr']"
        )
        exc = InvalidValueError(message)

        assert "custom" in str(exc)
        assert "percentile" in str(exc)


class TestDomainExceptionHierarchy:
    """Test the complete domain exception hierarchy."""

    def test_all_exceptions_inherit_from_domain_error(self):
        """Test that all domain exceptions inherit from DomainError."""
        exception_classes = [
            ValidationError,
            InvalidValueError,
        ]

        for exc_class in exception_classes:
            exc = exc_class("test message")
            assert isinstance(exc, DomainError)
            assert isinstance(exc, Exception)

    def test_exception_type_discrimination(self):
        """Test that different exception types can be distinguished."""
        exceptions = [
            ValidationError("validation error"),
            InvalidValueError("invalid value error"),
        ]

        # Each exception should be its own type
        for i, exc1 in enumerate(exceptions):
            for j, exc2 in enumerate(exceptions):
                if i != j:
                    assert type(exc1) != type(exc2)

    def test_exception_catching_specificity(self):
        """Test that exceptions can be caught at different levels of specificity."""

        def raise_validation_error():
            raise ValidationError("test validation error")

        # Should be catchable as specific type
        with pytest.raises(ValidationError):
            raise_validation_error()

        # Should be catchable as base domain type
        with pytest.raises(DomainError):
            raise_validation_error()

        # Should be catchable as general Exception
        with pytest.raises(Exception):
            raise_validation_error()

    def test_exception_inheritance_relationships(self):
        """Test inheritance relationships between exceptions."""
        # ValidationError inherits from DomainError
        assert issubclass(ValidationError, DomainError)
        assert issubclass(ValidationError, Exception)

        # InvalidValueError inherits from DomainError
        assert issubclass(InvalidValueError, DomainError)
        assert issubclass(InvalidValueError, Exception)

        # DomainError inherits from Exception
        assert issubclass(DomainError, Exception)


class TestDomainExceptionUsagePatterns:
    """Test common usage patterns for domain exceptions."""

    def test_validation_error_for_value_objects(self):
        """Test using ValidationError for value object validation."""

        def validate_anomaly_score(value):
            if not isinstance(value, (int, float)):
                raise ValidationError(f"Score must be numeric, got {type(value)}")
            if not (0.0 <= value <= 1.0):
                raise ValidationError(f"Score must be between 0 and 1, got {value}")

        # Valid value should not raise
        validate_anomaly_score(0.5)

        # Invalid type should raise ValidationError
        with pytest.raises(ValidationError, match="Score must be numeric"):
            validate_anomaly_score("0.5")

        # Invalid range should raise ValidationError
        with pytest.raises(ValidationError, match="Score must be between 0 and 1"):
            validate_anomaly_score(1.5)

    def test_invalid_value_error_for_configuration(self):
        """Test using InvalidValueError for configuration validation."""

        def validate_threshold_method(method):
            valid_methods = ["percentile", "fixed", "iqr", "mad", "adaptive"]
            if method not in valid_methods:
                raise InvalidValueError(
                    f"Invalid threshold method: {method}. "
                    f"Valid options: {valid_methods}"
                )

        # Valid method should not raise
        validate_threshold_method("percentile")

        # Invalid method should raise InvalidValueError
        with pytest.raises(InvalidValueError, match="Invalid threshold method"):
            validate_threshold_method("custom")

    def test_nested_exception_handling(self):
        """Test handling nested exceptions."""

        def create_value_object():
            try:
                # Simulate nested validation
                validate_inner_value()
            except ValidationError as e:
                raise DomainError(f"Failed to create value object: {e}") from e

        def validate_inner_value():
            raise ValidationError("Inner validation failed")

        with pytest.raises(DomainError) as exc_info:
            create_value_object()

        # Check exception chaining
        assert exc_info.value.__cause__ is not None
        assert isinstance(exc_info.value.__cause__, ValidationError)
        assert "Inner validation failed" in str(exc_info.value.__cause__)

    def test_exception_with_context_information(self):
        """Test exceptions with rich context information."""

        def validate_semantic_version(major, minor, patch):
            context = f"SemanticVersion(major={major}, minor={minor}, patch={patch})"

            if not isinstance(major, int) or major < 0:
                raise ValidationError(
                    f"{context}: Major version must be non-negative integer, got {major}"
                )

            if not isinstance(minor, int) or minor < 0:
                raise ValidationError(
                    f"{context}: Minor version must be non-negative integer, got {minor}"
                )

            if not isinstance(patch, int) or patch < 0:
                raise ValidationError(
                    f"{context}: Patch version must be non-negative integer, got {patch}"
                )

        # Valid versions should not raise
        validate_semantic_version(1, 2, 3)

        # Invalid major should include context
        with pytest.raises(ValidationError) as exc_info:
            validate_semantic_version(-1, 2, 3)

        assert "SemanticVersion(major=-1, minor=2, patch=3)" in str(exc_info.value)
        assert "Major version must be non-negative integer" in str(exc_info.value)

    def test_multiple_validation_errors(self):
        """Test handling multiple validation errors."""

        def validate_complex_object(data):
            errors = []

            if "score" not in data:
                errors.append("Missing required field: score")
            elif not isinstance(data["score"], (int, float)):
                errors.append("Score must be numeric")
            elif not (0.0 <= data["score"] <= 1.0):
                errors.append("Score must be between 0 and 1")

            if "threshold" not in data:
                errors.append("Missing required field: threshold")
            elif not isinstance(data["threshold"], (int, float)):
                errors.append("Threshold must be numeric")

            if errors:
                raise ValidationError(
                    f"Multiple validation errors: {'; '.join(errors)}"
                )

        # Valid data should not raise
        validate_complex_object({"score": 0.8, "threshold": 0.5})

        # Multiple errors should be collected
        with pytest.raises(ValidationError) as exc_info:
            validate_complex_object({"score": "invalid", "threshold": None})

        error_message = str(exc_info.value)
        assert "Multiple validation errors" in error_message
        assert "Score must be numeric" in error_message
        assert "Threshold must be numeric" in error_message


class TestDomainExceptionRepr:
    """Test domain exception representations."""

    def test_domain_error_repr_format(self):
        """Test DomainError repr format."""
        exc = DomainError("test message")
        repr_str = repr(exc)

        assert repr_str.startswith("DomainError(")
        assert "'test message'" in repr_str
        assert repr_str.endswith(")")

    def test_validation_error_repr_format(self):
        """Test ValidationError repr format."""
        exc = ValidationError("validation failed")
        repr_str = repr(exc)

        assert repr_str.startswith("ValidationError(")
        assert "'validation failed'" in repr_str
        assert repr_str.endswith(")")

    def test_invalid_value_error_repr_format(self):
        """Test InvalidValueError repr format."""
        exc = InvalidValueError("invalid value")
        repr_str = repr(exc)

        assert repr_str.startswith("InvalidValueError(")
        assert "'invalid value'" in repr_str
        assert repr_str.endswith(")")

    def test_exception_repr_with_special_characters(self):
        """Test exception repr with special characters."""
        special_message = "Error with quotes 'single' and \"double\" and newline\n"
        exc = ValidationError(special_message)
        repr_str = repr(exc)

        assert "ValidationError(" in repr_str
        # The repr should properly escape the special characters


class TestDomainExceptionEquality:
    """Test domain exception equality and hashing."""

    def test_same_exception_equality(self):
        """Test equality of same exception instances."""
        exc1 = ValidationError("test message")
        exc2 = ValidationError("test message")

        # Exception instances are not equal even with same message
        assert exc1 != exc2
        assert not (exc1 == exc2)

    def test_exception_identity(self):
        """Test exception identity."""
        exc = ValidationError("test message")

        # Exception is equal to itself
        assert exc == exc
        assert not (exc != exc)

    def test_exception_type_checking(self):
        """Test exception type checking."""
        validation_exc = ValidationError("validation error")
        invalid_value_exc = InvalidValueError("invalid value error")
        domain_exc = DomainError("domain error")

        # Type checking
        assert type(validation_exc) == ValidationError
        assert type(invalid_value_exc) == InvalidValueError
        assert type(domain_exc) == DomainError

        # Instance checking
        assert isinstance(validation_exc, ValidationError)
        assert isinstance(validation_exc, DomainError)
        assert isinstance(validation_exc, Exception)


class TestDomainExceptionEdgeCases:
    """Test domain exception edge cases."""

    def test_empty_message(self):
        """Test exceptions with empty messages."""
        exceptions = [
            DomainError(""),
            ValidationError(""),
            InvalidValueError(""),
        ]

        for exc in exceptions:
            assert str(exc) == ""
            assert repr(exc) is not None

    def test_very_long_message(self):
        """Test exceptions with very long messages."""
        long_message = "x" * 10000
        exc = ValidationError(long_message)

        assert str(exc) == long_message
        assert len(str(exc)) == 10000

    def test_unicode_message(self):
        """Test exceptions with unicode messages."""
        unicode_message = "异常检测错误 🚨 مشكلة في الكشف عن الشذوذ"
        exc = DomainError(unicode_message)

        assert str(exc) == unicode_message
        assert unicode_message in repr(exc)

    def test_message_with_formatting(self):
        """Test exceptions with formatted messages."""
        template = "Value {value} is invalid for field '{field}': {reason}"
        message = template.format(value=1.5, field="score", reason="out of range")
        exc = ValidationError(message)

        assert "1.5" in str(exc)
        assert "score" in str(exc)
        assert "out of range" in str(exc)

    def test_none_message(self):
        """Test exception behavior with None message."""
        # This might not be typical usage, but testing edge case
        try:
            exc = ValidationError(None)  # type: ignore
            # If this doesn't raise during construction, check string representation
            str_repr = str(exc)
            assert str_repr is not None
        except TypeError:
            # This is also acceptable behavior
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
