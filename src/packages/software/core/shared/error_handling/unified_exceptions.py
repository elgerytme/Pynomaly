"""Unified exception hierarchy for Software application."""

from __future__ import annotations

import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class ErrorSeverity(Enum):
    """Error severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error category classification."""

    VALIDATION = "validation"
    BUSINESS_RULE = "business_rule"
    INFRASTRUCTURE = "infrastructure"
    EXTERNAL_SERVICE = "external_service"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    CONFIGURATION = "configuration"
    PERFORMANCE = "performance"
    DATA_INTEGRITY = "data_integrity"
    NETWORK = "network"
    TIMEOUT = "timeout"
    RESOURCE_EXHAUSTION = "resource_exhaustion"


@dataclass
class ErrorContext:
    """Error context information."""

    error_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    user_id: str | None = None
    session_id: str | None = None
    request_id: str | None = None
    operation: str | None = None
    component: str | None = None
    environment: str | None = None
    additional_context: dict[str, Any] = field(default_factory=dict)


@dataclass
class ErrorDetails:
    """Detailed error information."""

    error_code: str
    message: str
    severity: ErrorSeverity
    category: ErrorCategory
    context: ErrorContext
    cause: Exception | None = None
    stack_trace: str | None = None
    recovery_suggestions: list[str] = field(default_factory=list)
    user_message: str | None = None
    technical_details: dict[str, Any] = field(default_factory=dict)
    retry_possible: bool = False
    retry_after: int | None = None  # seconds


class PynamolyError(Exception):
    """Base exception for all Software errors."""

    def __init__(
        self,
        error_code: str,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.BUSINESS_RULE,
        context: ErrorContext | None = None,
        cause: Exception | None = None,
        recovery_suggestions: list[str] | None = None,
        user_message: str | None = None,
        technical_details: dict[str, Any] | None = None,
        retry_possible: bool = False,
        retry_after: int | None = None,
    ):
        """Initialize unified Software error.

        Args:
            error_code: Unique error code for programmatic handling
            message: Technical error message
            severity: Error severity level
            category: Error category classification
            context: Error context information
            cause: Root cause exception
            recovery_suggestions: List of recovery suggestions
            user_message: User-friendly error message
            technical_details: Additional technical details
            retry_possible: Whether retry is possible
            retry_after: Retry delay in seconds
        """
        super().__init__(message)

        self.details = ErrorDetails(
            error_code=error_code,
            message=message,
            severity=severity,
            category=category,
            context=context or ErrorContext(),
            cause=cause,
            stack_trace=traceback.format_exc() if cause else None,
            recovery_suggestions=recovery_suggestions or [],
            user_message=user_message,
            technical_details=technical_details or {},
            retry_possible=retry_possible,
            retry_after=retry_after,
        )

        # Store cause for chaining
        self.__cause__ = cause

    def to_dict(self) -> dict[str, Any]:
        """Convert error to dictionary for serialization."""
        return {
            "error_id": self.details.context.error_id,
            "error_code": self.details.error_code,
            "message": self.details.message,
            "severity": self.details.severity.value,
            "category": self.details.category.value,
            "timestamp": self.details.context.timestamp.isoformat(),
            "user_id": self.details.context.user_id,
            "session_id": self.details.context.session_id,
            "request_id": self.details.context.request_id,
            "operation": self.details.context.operation,
            "component": self.details.context.component,
            "environment": self.details.context.environment,
            "user_message": self.details.user_message,
            "recovery_suggestions": self.details.recovery_suggestions,
            "retry_possible": self.details.retry_possible,
            "retry_after": self.details.retry_after,
            "technical_details": self.details.technical_details,
            "additional_context": self.details.context.additional_context,
        }

    def add_context(self, key: str, value: Any) -> None:
        """Add context information to the error."""
        self.details.context.additional_context[key] = value

    def add_recovery_suggestion(self, suggestion: str) -> None:
        """Add recovery suggestion to the error."""
        self.details.recovery_suggestions.append(suggestion)

    def set_user_message(self, message: str) -> None:
        """Set user-friendly error message."""
        self.details.user_message = message

    def set_retry_info(
        self, retry_possible: bool, retry_after: int | None = None
    ) -> None:
        """Set retry information for the error."""
        self.details.retry_possible = retry_possible
        self.details.retry_after = retry_after


# Domain-specific exceptions
class DomainError(PynamolyError):
    """Base class for domain-specific errors."""

    def __init__(self, error_code: str, message: str, **kwargs):
        super().__init__(
            error_code=error_code,
            message=message,
            category=ErrorCategory.BUSINESS_RULE,
            **kwargs,
        )


class ValidationError(PynamolyError):
    """Validation error."""

    def __init__(
        self, error_code: str, message: str, field_name: str | None = None, **kwargs
    ):
        super().__init__(
            error_code=error_code,
            message=message,
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.LOW,
            **kwargs,
        )
        if field_name:
            self.add_context("field_name", field_name)


class InfrastructureError(PynamolyError):
    """Infrastructure-related error."""

    def __init__(self, error_code: str, message: str, **kwargs):
        super().__init__(
            error_code=error_code,
            message=message,
            category=ErrorCategory.INFRASTRUCTURE,
            severity=ErrorSeverity.HIGH,
            **kwargs,
        )


class ExternalServiceError(PynamolyError):
    """External service error."""

    def __init__(self, error_code: str, message: str, service_name: str, **kwargs):
        super().__init__(
            error_code=error_code,
            message=message,
            category=ErrorCategory.EXTERNAL_SERVICE,
            severity=ErrorSeverity.HIGH,
            retry_possible=True,
            **kwargs,
        )
        self.add_context("service_name", service_name)


class AuthenticationError(PynamolyError):
    """Authentication error."""

    def __init__(self, error_code: str, message: str, **kwargs):
        super().__init__(
            error_code=error_code,
            message=message,
            category=ErrorCategory.AUTHENTICATION,
            severity=ErrorSeverity.MEDIUM,
            user_message="Authentication failed. Please check your credentials.",
            **kwargs,
        )


class AuthorizationError(PynamolyError):
    """Authorization error."""

    def __init__(self, error_code: str, message: str, **kwargs):
        super().__init__(
            error_code=error_code,
            message=message,
            category=ErrorCategory.AUTHORIZATION,
            severity=ErrorSeverity.MEDIUM,
            user_message="Access denied. You don't have permission for this operation.",
            **kwargs,
        )


class ConfigurationError(PynamolyError):
    """Configuration error."""

    def __init__(self, error_code: str, message: str, **kwargs):
        super().__init__(
            error_code=error_code,
            message=message,
            category=ErrorCategory.CONFIGURATION,
            severity=ErrorSeverity.HIGH,
            **kwargs,
        )


class PerformanceError(PynamolyError):
    """Performance-related error."""

    def __init__(self, error_code: str, message: str, **kwargs):
        super().__init__(
            error_code=error_code,
            message=message,
            category=ErrorCategory.PERFORMANCE,
            severity=ErrorSeverity.MEDIUM,
            **kwargs,
        )


class TimeoutError(PynamolyError):
    """Timeout error."""

    def __init__(self, error_code: str, message: str, timeout_seconds: float, **kwargs):
        super().__init__(
            error_code=error_code,
            message=message,
            category=ErrorCategory.TIMEOUT,
            severity=ErrorSeverity.MEDIUM,
            retry_possible=True,
            retry_after=int(timeout_seconds),
            **kwargs,
        )
        self.add_context("timeout_seconds", timeout_seconds)


class ResourceExhaustionError(PynamolyError):
    """Resource exhaustion error."""

    def __init__(self, error_code: str, message: str, resource_type: str, **kwargs):
        super().__init__(
            error_code=error_code,
            message=message,
            category=ErrorCategory.RESOURCE_EXHAUSTION,
            severity=ErrorSeverity.HIGH,
            retry_possible=True,
            retry_after=60,  # Wait 1 minute before retry
            **kwargs,
        )
        self.add_context("resource_type", resource_type)


class NetworkError(PynamolyError):
    """Network-related error."""

    def __init__(self, error_code: str, message: str, **kwargs):
        super().__init__(
            error_code=error_code,
            message=message,
            category=ErrorCategory.NETWORK,
            severity=ErrorSeverity.HIGH,
            retry_possible=True,
            retry_after=30,  # Wait 30 seconds before retry
            **kwargs,
        )


class DataIntegrityError(PynamolyError):
    """Data integrity error."""

    def __init__(self, error_code: str, message: str, **kwargs):
        super().__init__(
            error_code=error_code,
            message=message,
            category=ErrorCategory.DATA_INTEGRITY,
            severity=ErrorSeverity.CRITICAL,
            **kwargs,
        )


# Error code constants
class ErrorCodes:
    """Standard error codes for the application."""

    # Validation errors (VAL_xxx)
    VAL_INVALID_INPUT = "VAL_INVALID_INPUT"
    VAL_MISSING_REQUIRED_FIELD = "VAL_MISSING_REQUIRED_FIELD"
    VAL_INVALID_FORMAT = "VAL_INVALID_FORMAT"
    VAL_OUT_OF_RANGE = "VAL_OUT_OF_RANGE"
    VAL_DUPLICATE_VALUE = "VAL_DUPLICATE_VALUE"

    # Business rule errors (BIZ_xxx)
    BIZ_DETECTOR_NOT_FITTED = "BIZ_DETECTOR_NOT_FITTED"
    BIZ_INSUFFICIENT_DATA = "BIZ_INSUFFICIENT_DATA"
    BIZ_ALGORITHM_NOT_SUPPORTED = "BIZ_ALGORITHM_NOT_SUPPORTED"
    BIZ_FEATURE_MISMATCH = "BIZ_FEATURE_MISMATCH"
    BIZ_INVALID_OPERATION = "BIZ_INVALID_OPERATION"

    # Infrastructure errors (INF_xxx)
    INF_DATABASE_CONNECTION = "INF_DATABASE_CONNECTION"
    INF_CACHE_UNAVAILABLE = "INF_CACHE_UNAVAILABLE"
    INF_FILE_SYSTEM_ERROR = "INF_FILE_SYSTEM_ERROR"
    INF_MEMORY_EXHAUSTED = "INF_MEMORY_EXHAUSTED"

    # External service errors (EXT_xxx)
    EXT_API_TIMEOUT = "EXT_API_TIMEOUT"
    EXT_API_RATE_LIMITED = "EXT_API_RATE_LIMITED"
    EXT_SERVICE_UNAVAILABLE = "EXT_SERVICE_UNAVAILABLE"

    # Authentication/Authorization errors (AUTH_xxx)
    AUTH_INVALID_CREDENTIALS = "AUTH_INVALID_CREDENTIALS"
    AUTH_TOKEN_EXPIRED = "AUTH_TOKEN_EXPIRED"
    AUTH_INSUFFICIENT_PERMISSIONS = "AUTH_INSUFFICIENT_PERMISSIONS"

    # Configuration errors (CFG_xxx)
    CFG_INVALID_CONFIGURATION = "CFG_INVALID_CONFIGURATION"
    CFG_MISSING_CONFIGURATION = "CFG_MISSING_CONFIGURATION"
    CFG_CONFIGURATION_CONFLICT = "CFG_CONFIGURATION_CONFLICT"

    # Performance errors (PERF_xxx)
    PERF_TIMEOUT = "PERF_TIMEOUT"
    PERF_SLOW_QUERY = "PERF_SLOW_QUERY"
    PERF_RESOURCE_EXHAUSTED = "PERF_RESOURCE_EXHAUSTED"

    # Data integrity errors (DATA_xxx)
    DATA_CORRUPTION = "DATA_CORRUPTION"
    DATA_INCONSISTENCY = "DATA_INCONSISTENCY"
    DATA_CONSTRAINT_VIOLATION = "DATA_CONSTRAINT_VIOLATION"


# Factory functions for common error patterns
def create_validation_error(
    message: str, field_name: str | None = None, **kwargs
) -> ValidationError:
    """Create a validation error with standard format."""
    return ValidationError(
        error_code=ErrorCodes.VAL_INVALID_INPUT,
        message=message,
        field_name=field_name,
        **kwargs,
    )


def create_business_error(error_code: str, message: str, **kwargs) -> DomainError:
    """Create a business rule error."""
    return DomainError(error_code=error_code, message=message, **kwargs)


def create_infrastructure_error(
    error_code: str, message: str, cause: Exception | None = None, **kwargs
) -> InfrastructureError:
    """Create an infrastructure error."""
    return InfrastructureError(
        error_code=error_code, message=message, cause=cause, **kwargs
    )


def create_external_service_error(
    service_name: str, message: str, cause: Exception | None = None, **kwargs
) -> ExternalServiceError:
    """Create an external service error."""
    return ExternalServiceError(
        error_code=ErrorCodes.EXT_SERVICE_UNAVAILABLE,
        message=message,
        service_name=service_name,
        cause=cause,
        **kwargs,
    )


def create_timeout_error(
    operation: str, timeout_seconds: float, **kwargs
) -> TimeoutError:
    """Create a timeout error."""
    return TimeoutError(
        error_code=ErrorCodes.PERF_TIMEOUT,
        message=f"Operation '{operation}' timed out after {timeout_seconds} seconds",
        timeout_seconds=timeout_seconds,
        **kwargs,
    )
