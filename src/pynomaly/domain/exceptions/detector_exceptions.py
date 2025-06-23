"""Detector-specific domain exceptions."""

from __future__ import annotations

from typing import Any, Optional

from .base import ConfigurationError, DomainError, NotFittedError


class DetectorError(DomainError):
    """Base exception for detector-related errors."""
    pass


class DetectorNotFittedError(NotFittedError, DetectorError):
    """Exception raised when detector is not fitted."""
    
    def __init__(
        self,
        detector_name: str,
        operation: str = "detect",
        **kwargs: Any
    ) -> None:
        """Initialize detector not fitted error.
        
        Args:
            detector_name: Name of the detector
            operation: Operation that requires fitting
            **kwargs: Additional details
        """
        message = (
            f"Detector '{detector_name}' must be fitted before "
            f"calling {operation}()"
        )
        super().__init__(message, detector_name=detector_name, operation=operation, **kwargs)


class DetectorConfigurationError(ConfigurationError, DetectorError):
    """Exception raised when detector configuration is invalid."""
    
    def __init__(
        self,
        detector_name: str,
        message: str,
        **kwargs: Any
    ) -> None:
        """Initialize detector configuration error.
        
        Args:
            detector_name: Name of the detector
            message: Error message
            **kwargs: Additional details
        """
        full_message = f"Configuration error in detector '{detector_name}': {message}"
        super().__init__(full_message, detector_name=detector_name, **kwargs)


class InvalidAlgorithmError(DetectorError):
    """Exception raised when algorithm is invalid or not supported."""
    
    def __init__(
        self,
        algorithm_name: str,
        available_algorithms: Optional[list[str]] = None,
        **kwargs: Any
    ) -> None:
        """Initialize invalid algorithm error.
        
        Args:
            algorithm_name: Name of the invalid algorithm
            available_algorithms: List of available algorithms
            **kwargs: Additional details
        """
        message = f"Algorithm '{algorithm_name}' is not supported"
        
        if available_algorithms:
            message += f". Available algorithms: {', '.join(available_algorithms)}"
        
        details = {"algorithm_name": algorithm_name, **kwargs}
        if available_algorithms:
            details["available_algorithms"] = available_algorithms
        
        super().__init__(message, details)


class FittingError(DetectorError):
    """Exception raised when fitting fails."""
    
    def __init__(
        self,
        detector_name: str,
        reason: str,
        dataset_name: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        """Initialize fitting error.
        
        Args:
            detector_name: Name of the detector
            reason: Reason for fitting failure
            dataset_name: Name of the dataset
            **kwargs: Additional details
        """
        message = f"Failed to fit detector '{detector_name}': {reason}"
        
        details = {"detector_name": detector_name, "reason": reason, **kwargs}
        if dataset_name:
            details["dataset_name"] = dataset_name
        
        super().__init__(message, details)