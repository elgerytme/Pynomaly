"""Exception classes for enterprise applications.

This module provides a hierarchy of exceptions that can be used across
enterprise applications with structured error information.
"""

from __future__ import annotations

from typing import Any


class EnterpriseError(Exception):
    """Base exception for all enterprise application errors.

    This exception provides structured error information including
    error details and context for better debugging and monitoring.
    """

    def __init__(
        self,
        message: str,
        error_code: str | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
        self.cause = cause

    def to_dict(self) -> dict[str, Any]:
        """Convert the exception to a dictionary representation."""
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details,
            "cause": str(self.cause) if self.cause else None,
        }

    def __str__(self) -> str:
        """String representation of the exception."""
        if self.details:
            return f"{self.message} (details: {self.details})"
        return self.message


class DomainError(EnterpriseError):
    """Exception for domain layer errors.

    Raised when business rules are violated or domain invariants are broken.
    """

    pass


class ValidationError(DomainError):
    """Exception for validation errors.

    Raised when input validation fails or entity state is invalid.
    """

    pass


class BusinessRuleViolationError(DomainError):
    """Exception for business rule violations.

    Raised when a business rule is violated during domain operations.
    """

    pass


class EntityNotFoundError(DomainError):
    """Exception for when an entity is not found.

    Raised when attempting to access an entity that doesn't exist.
    """

    def __init__(
        self,
        entity_type: str,
        entity_id: Any,
        details: dict[str, Any] | None = None,
    ) -> None:
        message = f"{entity_type} with ID '{entity_id}' not found"
        details = details or {}
        details.update(
            {
                "entity_type": entity_type,
                "entity_id": str(entity_id),
            }
        )
        super().__init__(message, "ENTITY_NOT_FOUND", details)


class EntityAlreadyExistsError(DomainError):
    """Exception for when an entity already exists.

    Raised when attempting to create an entity that already exists.
    """

    def __init__(
        self,
        entity_type: str,
        entity_id: Any,
        details: dict[str, Any] | None = None,
    ) -> None:
        message = f"{entity_type} with ID '{entity_id}' already exists"
        details = details or {}
        details.update(
            {
                "entity_type": entity_type,
                "entity_id": str(entity_id),
            }
        )
        super().__init__(message, "ENTITY_ALREADY_EXISTS", details)


class ConcurrencyError(DomainError):
    """Exception for concurrency conflicts.

    Raised when optimistic concurrency checks fail.
    """

    def __init__(
        self,
        entity_type: str,
        entity_id: Any,
        expected_version: int,
        actual_version: int,
        details: dict[str, Any] | None = None,
    ) -> None:
        message = f"Concurrency conflict for {entity_type} '{entity_id}': expected version {expected_version}, actual version {actual_version}"
        details = details or {}
        details.update(
            {
                "entity_type": entity_type,
                "entity_id": str(entity_id),
                "expected_version": expected_version,
                "actual_version": actual_version,
            }
        )
        super().__init__(message, "CONCURRENCY_CONFLICT", details)


class InfrastructureError(EnterpriseError):
    """Exception for infrastructure layer errors.

    Raised when external dependencies fail or infrastructure components
    are unavailable.
    """

    pass


class RepositoryError(InfrastructureError):
    """Exception for repository errors.

    Raised when data persistence operations fail.
    """

    pass


class ExternalServiceError(InfrastructureError):
    """Exception for external service errors.

    Raised when external service calls fail.
    """

    def __init__(
        self,
        service_name: str,
        operation: str,
        message: str,
        status_code: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        full_message = (
            f"External service '{service_name}' failed during '{operation}': {message}"
        )
        details = details or {}
        details.update(
            {
                "service_name": service_name,
                "operation": operation,
                "status_code": status_code,
            }
        )
        super().__init__(full_message, "EXTERNAL_SERVICE_ERROR", details)


class CacheError(InfrastructureError):
    """Exception for cache operation errors."""

    pass


class MessageQueueError(InfrastructureError):
    """Exception for message queue operation errors."""

    pass


class DatabaseError(InfrastructureError):
    """Exception for database operation errors."""

    pass


class ConfigurationError(EnterpriseError):
    """Exception for configuration errors.

    Raised when configuration is missing, invalid, or inconsistent.
    """

    def __init__(
        self,
        config_key: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        full_message = f"Configuration error for '{config_key}': {message}"
        details = details or {}
        details.update({"config_key": config_key})
        super().__init__(full_message, "CONFIGURATION_ERROR", details)


class SecurityError(EnterpriseError):
    """Exception for security-related errors.

    Raised when security checks fail or unauthorized access is attempted.
    """

    pass


class AuthenticationError(SecurityError):
    """Exception for authentication failures."""

    def __init__(
        self,
        message: str = "Authentication failed",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, "AUTHENTICATION_FAILED", details)


class AuthorizationError(SecurityError):
    """Exception for authorization failures."""

    def __init__(
        self,
        resource: str,
        action: str,
        user: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        message = f"Access denied to {action} {resource}"
        if user:
            message += f" for user '{user}'"

        details = details or {}
        details.update(
            {
                "resource": resource,
                "action": action,
                "user": user,
            }
        )
        super().__init__(message, "ACCESS_DENIED", details)


class ApplicationError(EnterpriseError):
    """Exception for application layer errors.

    Raised when use case execution fails or application services
    encounter errors.
    """

    pass


class UseCaseError(ApplicationError):
    """Exception for use case execution errors."""

    pass


class ServiceUnavailableError(ApplicationError):
    """Exception for when a required service is unavailable."""

    def __init__(
        self,
        service_name: str,
        message: str = "Service is currently unavailable",
        details: dict[str, Any] | None = None,
    ) -> None:
        full_message = f"Service '{service_name}' is unavailable: {message}"
        details = details or {}
        details.update({"service_name": service_name})
        super().__init__(full_message, "SERVICE_UNAVAILABLE", details)


class RateLimitExceededError(ApplicationError):
    """Exception for rate limiting violations."""

    def __init__(
        self,
        resource: str,
        limit: int,
        window: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        message = f"Rate limit exceeded for {resource}: {limit} requests per {window}"
        details = details or {}
        details.update(
            {
                "resource": resource,
                "limit": limit,
                "window": window,
            }
        )
        super().__init__(message, "RATE_LIMIT_EXCEEDED", details)


class TimeoutError(ApplicationError):
    """Exception for operation timeouts."""

    def __init__(
        self,
        operation: str,
        timeout_seconds: float,
        details: dict[str, Any] | None = None,
    ) -> None:
        message = f"Operation '{operation}' timed out after {timeout_seconds} seconds"
        details = details or {}
        details.update(
            {
                "operation": operation,
                "timeout_seconds": timeout_seconds,
            }
        )
        super().__init__(message, "OPERATION_TIMEOUT", details)
