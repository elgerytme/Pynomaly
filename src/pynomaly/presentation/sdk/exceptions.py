"""
SDK Exception Classes

Comprehensive exception hierarchy for the Pynomaly SDK providing clear error types
and detailed error information for proper error handling in client applications.
"""

from typing import Any


class PynomaliSDKError(Exception):
    """Base exception for all Pynomaly SDK errors."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details or {}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(message='{self.message}', status_code={self.status_code})"


class AuthenticationError(PynomaliSDKError):
    """Raised when authentication fails or credentials are invalid."""

    def __init__(self, message: str = "Authentication failed", **kwargs):
        super().__init__(message, status_code=401, **kwargs)


class AuthorizationError(PynomaliSDKError):
    """Raised when the user lacks permissions for the requested operation."""

    def __init__(self, message: str = "Insufficient permissions", **kwargs):
        super().__init__(message, status_code=403, **kwargs)


class ValidationError(PynomaliSDKError):
    """Raised when request data fails validation."""

    def __init__(
        self,
        message: str = "Validation failed",
        validation_errors: dict[str, Any] | None = None,
        **kwargs,
    ):
        super().__init__(message, status_code=400, **kwargs)
        self.validation_errors = validation_errors or {}


class ResourceNotFoundError(PynomaliSDKError):
    """Raised when a requested resource is not found."""

    def __init__(self, resource_type: str, resource_id: str, **kwargs):
        message = f"{resource_type} with ID '{resource_id}' not found"
        super().__init__(message, status_code=404, **kwargs)
        self.resource_type = resource_type
        self.resource_id = resource_id


class ConflictError(PynomaliSDKError):
    """Raised when a request conflicts with the current state."""

    def __init__(self, message: str = "Request conflicts with current state", **kwargs):
        super().__init__(message, status_code=409, **kwargs)


class RateLimitError(PynomaliSDKError):
    """Raised when rate limits are exceeded."""

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: int | None = None,
        **kwargs,
    ):
        super().__init__(message, status_code=429, **kwargs)
        self.retry_after = retry_after


class ServerError(PynomaliSDKError):
    """Raised when the server encounters an internal error."""

    def __init__(self, message: str = "Internal server error", **kwargs):
        super().__init__(message, status_code=500, **kwargs)


class ServiceUnavailableError(PynomaliSDKError):
    """Raised when the service is temporarily unavailable."""

    def __init__(self, message: str = "Service temporarily unavailable", **kwargs):
        super().__init__(message, status_code=503, **kwargs)


class TimeoutError(PynomaliSDKError):
    """Raised when a request times out."""

    def __init__(self, message: str = "Request timed out", **kwargs):
        super().__init__(message, **kwargs)


class NetworkError(PynomaliSDKError):
    """Raised when network connectivity issues occur."""

    def __init__(self, message: str = "Network error occurred", **kwargs):
        super().__init__(message, **kwargs)


class ConfigurationError(PynomaliSDKError):
    """Raised when SDK configuration is invalid."""

    def __init__(self, message: str = "Invalid SDK configuration", **kwargs):
        super().__init__(message, **kwargs)


def map_http_error(
    status_code: int, message: str, details: dict[str, Any] | None = None
) -> PynomaliSDKError:
    """Map HTTP status codes to appropriate SDK exceptions."""

    error_map = {
        400: ValidationError,
        401: AuthenticationError,
        403: AuthorizationError,
        404: lambda msg, **kwargs: ResourceNotFoundError(
            "Resource", "unknown", **kwargs
        ),
        409: ConflictError,
        429: RateLimitError,
        500: ServerError,
        503: ServiceUnavailableError,
    }

    error_class = error_map.get(status_code, PynomaliSDKError)

    if status_code == 404:
        # Extract resource info from details if available
        resource_type = (
            details.get("resource_type", "Resource") if details else "Resource"
        )
        resource_id = details.get("resource_id", "unknown") if details else "unknown"
        return ResourceNotFoundError(resource_type, resource_id, details=details)

    return error_class(message, status_code=status_code, details=details)
