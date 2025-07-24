"""Exception classes for the Anomaly Detection SDK."""

from typing import Optional, Dict, Any


class AnomalyDetectionSDKError(Exception):
    """Base exception for all SDK errors."""
    
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
    
    def __str__(self) -> str:
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message


class APIError(AnomalyDetectionSDKError):
    """Error from the API service."""
    
    def __init__(self, message: str, status_code: int, response_data: Optional[Dict[str, Any]] = None):
        super().__init__(message, f"HTTP_{status_code}", response_data)
        self.status_code = status_code
        self.response_data = response_data or {}


class ValidationError(AnomalyDetectionSDKError):
    """Data validation error."""
    
    def __init__(self, message: str, field: Optional[str] = None, value: Optional[Any] = None):
        super().__init__(message, "VALIDATION_ERROR", {"field": field, "value": value})
        self.field = field
        self.value = value


class ConnectionError(AnomalyDetectionSDKError):
    """Network connection error."""
    
    def __init__(self, message: str, url: Optional[str] = None):
        super().__init__(message, "CONNECTION_ERROR", {"url": url})
        self.url = url


class TimeoutError(AnomalyDetectionSDKError):
    """Request timeout error."""
    
    def __init__(self, message: str, timeout_duration: Optional[float] = None):
        super().__init__(message, "TIMEOUT_ERROR", {"timeout_duration": timeout_duration})
        self.timeout_duration = timeout_duration


class AuthenticationError(AnomalyDetectionSDKError):
    """Authentication error."""
    
    def __init__(self, message: str = "Authentication failed"):
        super().__init__(message, "AUTH_ERROR")


class RateLimitError(AnomalyDetectionSDKError):
    """Rate limit exceeded error."""
    
    def __init__(self, message: str, retry_after: Optional[int] = None):
        super().__init__(message, "RATE_LIMIT", {"retry_after": retry_after})
        self.retry_after = retry_after


class ModelNotFoundError(AnomalyDetectionSDKError):
    """Model not found error."""
    
    def __init__(self, model_id: str):
        super().__init__(f"Model not found: {model_id}", "MODEL_NOT_FOUND", {"model_id": model_id})
        self.model_id = model_id


class StreamingError(AnomalyDetectionSDKError):
    """Streaming connection error."""
    
    def __init__(self, message: str, connection_status: Optional[str] = None):
        super().__init__(message, "STREAMING_ERROR", {"connection_status": connection_status})
        self.connection_status = connection_status