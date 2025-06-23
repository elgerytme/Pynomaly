"""Base domain exceptions."""

from __future__ import annotations

from typing import Any, Dict, Optional


class PynamolyError(Exception):
    """Base exception for all Pynomaly errors."""
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
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
        field: Optional[str] = None,
        value: Optional[Any] = None,
        **kwargs: Any
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


class NotFittedError(DomainError):
    """Exception raised when using an unfitted model."""
    
    def __init__(
        self,
        message: str = "Detector must be fitted before use",
        detector_name: Optional[str] = None,
        **kwargs: Any
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
        parameter: Optional[str] = None,
        expected: Optional[Any] = None,
        actual: Optional[Any] = None,
        **kwargs: Any
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