"""Base domain exceptions."""

from __future__ import annotations

from typing import Any


class PynamolyError(Exception):
    """Base exception for all Pynomaly errors."""

    def __init__(
        self,
        message: str,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        """Initialize exception with optional details and cause.

        Args:
            message: Error message
            details: Additional error details
            cause: Original exception that caused this error
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.cause = cause

    def __str__(self) -> str:
        """String representation of the error."""
        parts = [self.message]

        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            parts.append(f"Details: {details_str}")

        if self.cause:
            parts.append(f"Caused by: {type(self.cause).__name__}: {str(self.cause)}")

        return " | ".join(parts)

    def with_context(self, **kwargs: Any) -> PynamolyError:
        """Add context information to the exception.

        Args:
            **kwargs: Context key-value pairs

        Returns:
            The exception with added context
        """
        self.details.update(kwargs)
        return self


class DomainError(PynamolyError):
    """Base exception for domain-specific errors."""

    pass


class ValidationError(DomainError):
    """Exception raised when validation fails."""

    def __init__(
        self,
        message: str,
        field: str | None = None,
        value: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize validation error.

        Args:
            message: Error message
            field: Field that failed validation
            value: Invalid value
            **kwargs: Additional details
        """
        details = kwargs
        if field is not None:
            details["field"] = field
        if value is not None:
            details["value"] = value

        super().__init__(message, details)


class InvalidValueError(ValidationError):
    """Exception raised when a value is invalid."""

    pass


class BusinessRuleViolation(DomainError):
    """Exception raised when a business rule is violated."""

    def __init__(
        self,
        message: str,
        rule_name: str | None = None,
        rule_description: str | None = None,
        violation_context: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize business rule violation error.

        Args:
            message: Error message
            rule_name: Name of the violated business rule
            rule_description: Description of the business rule
            violation_context: Context information about the violation
            **kwargs: Additional details
        """
        details = kwargs
        if rule_name:
            details["rule_name"] = rule_name
        if rule_description:
            details["rule_description"] = rule_description
        if violation_context:
            details.update(violation_context)

        super().__init__(message, details)


class NotFittedError(DomainError):
    """Exception raised when using an unfitted model."""

    def __init__(
        self,
        message: str = "Detector must be fitted before use",
        detector_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize not fitted error.

        Args:
            message: Error message
            detector_name: Name of the unfitted detector
            **kwargs: Additional details
        """
        details = kwargs
        if detector_name:
            details["detector_name"] = detector_name

        super().__init__(message, details)


class ConfigurationError(DomainError):
    """Exception raised when configuration is invalid."""

    def __init__(
        self,
        message: str,
        parameter: str | None = None,
        expected: Any | None = None,
        actual: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize configuration error.

        Args:
            message: Error message
            parameter: Configuration parameter name
            expected: Expected value/type
            actual: Actual value
            **kwargs: Additional details
        """
        details = kwargs
        if parameter:
            details["parameter"] = parameter
        if expected is not None:
            details["expected"] = expected
        if actual is not None:
            details["actual"] = actual

        super().__init__(message, details)


class InvalidConfigurationError(ConfigurationError):
    """Exception raised when configuration is invalid or corrupted."""

    def __init__(
        self,
        message: str = "Invalid configuration",
        config_path: str | None = None,
        parameter: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize invalid configuration error.

        Args:
            message: Error message
            config_path: Path to the configuration file
            parameter: Configuration parameter that is invalid
            **kwargs: Additional details
        """
        details = kwargs
        if config_path:
            details["config_path"] = config_path

        super().__init__(message, parameter, **details)


class AuthenticationError(PynamolyError):
    """Exception raised when authentication fails."""

    def __init__(
        self,
        message: str = "Authentication failed",
        username: str | None = None,
        reason: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize authentication error.

        Args:
            message: Error message
            username: Username that failed authentication
            reason: Reason for authentication failure
            **kwargs: Additional details
        """
        details = kwargs
        if username:
            details["username"] = username
        if reason:
            details["reason"] = reason

        super().__init__(message, details)


class AuthorizationError(PynamolyError):
    """Exception raised when authorization fails."""

    def __init__(
        self,
        message: str = "Authorization failed",
        user_id: str | None = None,
        required_permission: str | None = None,
        required_role: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize authorization error.

        Args:
            message: Error message
            user_id: User ID that failed authorization
            required_permission: Permission that was required
            required_role: Role that was required
            **kwargs: Additional details
        """
        details = kwargs
        if user_id:
            details["user_id"] = user_id
        if required_permission:
            details["required_permission"] = required_permission
        if required_role:
            details["required_role"] = required_role

        super().__init__(message, details)


class CacheError(PynamolyError):
    """Exception raised when cache operations fail."""

    def __init__(
        self,
        message: str = "Cache operation failed",
        operation: str | None = None,
        key: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize cache error.

        Args:
            message: Error message
            operation: Cache operation that failed
            key: Cache key involved
            **kwargs: Additional details
        """
        details = kwargs
        if operation:
            details["operation"] = operation
        if key:
            details["key"] = key

        super().__init__(message, details)


class InfrastructureError(PynamolyError):
    """Exception raised when infrastructure operations fail."""

    def __init__(
        self,
        message: str = "Infrastructure operation failed",
        component: str | None = None,
        operation: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize infrastructure error.

        Args:
            message: Error message
            component: Infrastructure component that failed
            operation: Operation that failed
            **kwargs: Additional details
        """
        details = kwargs
        if component:
            details["component"] = component
        if operation:
            details["operation"] = operation

        super().__init__(message, details)
