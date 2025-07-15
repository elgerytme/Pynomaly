"""
Pynomaly SDK Exceptions

Custom exceptions for the Pynomaly SDK with specific error handling
for different types of failures.
"""

from typing import Optional, Dict, Any


class PynomalySDKError(Exception):
    """Base exception for all Pynomaly SDK errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}
    
    def __str__(self) -> str:
        if self.details:
            return f"{self.message} (Details: {self.details})"
        return self.message


class AuthenticationError(PynomalySDKError):
    """Raised when authentication fails."""
    
    def __init__(self, message: str = "Authentication failed", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details)


class ValidationError(PynomalySDKError):
    """Raised when request validation fails."""
    
    def __init__(self, message: str, field: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details)
        self.field = field


class APIError(PynomalySDKError):
    """Raised when API request fails."""
    
    def __init__(
        self, 
        message: str, 
        status_code: Optional[int] = None,
        response_data: Optional[Dict[str, Any]] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, details)
        self.status_code = status_code
        self.response_data = response_data or {}


class ResourceNotFoundError(APIError):
    """Raised when a requested resource is not found."""
    
    def __init__(self, resource_type: str, resource_id: str, details: Optional[Dict[str, Any]] = None):
        message = f"{resource_type} with ID '{resource_id}' not found"
        super().__init__(message, status_code=404, details=details)
        self.resource_type = resource_type
        self.resource_id = resource_id


class RateLimitError(APIError):
    """Raised when API rate limit is exceeded."""
    
    def __init__(self, retry_after: Optional[int] = None, details: Optional[Dict[str, Any]] = None):
        message = "API rate limit exceeded"
        if retry_after:
            message += f", retry after {retry_after} seconds"
        super().__init__(message, status_code=429, details=details)
        self.retry_after = retry_after


class DataError(PynomalySDKError):
    """Raised when there are issues with data processing."""
    
    def __init__(self, message: str, data_info: Optional[Dict[str, Any]] = None):
        super().__init__(message, data_info)
        self.data_info = data_info or {}


class ModelError(PynomalySDKError):
    """Raised when there are issues with model operations."""
    
    def __init__(self, message: str, model_info: Optional[Dict[str, Any]] = None):
        super().__init__(message, model_info)
        self.model_info = model_info or {}


class ConfigurationError(PynomalySDKError):
    """Raised when there are configuration issues."""
    
    def __init__(self, message: str, config_key: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details)
        self.config_key = config_key