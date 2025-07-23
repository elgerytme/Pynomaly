"""Exception hierarchy for SDK errors."""

from typing import Any, Dict, Optional


class SDKError(Exception):
    """Base exception for all SDK errors."""
    
    def __init__(self, message: str, status_code: Optional[int] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details or {}


class AuthenticationError(SDKError):
    """Raised when authentication fails."""
    
    def __init__(self, message: str = "Authentication failed", **kwargs):
        super().__init__(message, status_code=401, **kwargs)


class AuthorizationError(SDKError):
    """Raised when authorization fails."""
    
    def __init__(self, message: str = "Access denied", **kwargs):
        super().__init__(message, status_code=403, **kwargs)


class ValidationError(SDKError):
    """Raised when request validation fails."""
    
    def __init__(self, message: str = "Validation failed", **kwargs):
        super().__init__(message, status_code=422, **kwargs)


class NotFoundError(SDKError):
    """Raised when a resource is not found."""
    
    def __init__(self, message: str = "Resource not found", **kwargs):
        super().__init__(message, status_code=404, **kwargs)


class RateLimitError(SDKError):
    """Raised when rate limit is exceeded."""
    
    def __init__(self, message: str = "Rate limit exceeded", retry_after: Optional[int] = None, **kwargs):
        super().__init__(message, status_code=429, **kwargs)
        self.retry_after = retry_after


class ServerError(SDKError):
    """Raised when server returns 5xx error."""
    
    def __init__(self, message: str = "Internal server error", **kwargs):
        super().__init__(message, status_code=500, **kwargs)


class TimeoutError(SDKError):
    """Raised when request times out."""
    
    def __init__(self, message: str = "Request timed out", **kwargs):
        super().__init__(message, **kwargs)


class ConnectionError(SDKError):
    """Raised when connection fails."""
    
    def __init__(self, message: str = "Connection failed", **kwargs):
        super().__init__(message, **kwargs)


def create_exception_from_response(status_code: int, message: str, details: Optional[Dict[str, Any]] = None) -> SDKError:
    """Create appropriate exception based on HTTP status code."""
    
    if status_code == 401:
        return AuthenticationError(message, details=details)
    elif status_code == 403:
        return AuthorizationError(message, details=details)
    elif status_code == 404:
        return NotFoundError(message, details=details)
    elif status_code == 422:
        return ValidationError(message, details=details)
    elif status_code == 429:
        retry_after = details.get("retry_after") if details else None
        return RateLimitError(message, retry_after=retry_after, details=details)
    elif 500 <= status_code < 600:
        return ServerError(message, status_code=status_code, details=details)
    else:
        return SDKError(message, status_code=status_code, details=details)