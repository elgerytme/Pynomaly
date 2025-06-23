"""Dataset-specific domain exceptions."""

from __future__ import annotations

from typing import Any, Optional

from .base import DomainError, ValidationError


class DatasetError(DomainError):
    """Base exception for dataset-related errors."""
    pass


class DataValidationError(ValidationError, DatasetError):
    """Exception raised when data validation fails."""
    
    def __init__(
        self,
        message: str,
        dataset_name: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        """Initialize data validation error.
        
        Args:
            message: Error message
            dataset_name: Name of the dataset
            **kwargs: Additional details
        """
        full_message = message
        if dataset_name:
            full_message = f"Validation failed for dataset '{dataset_name}': {message}"
        
        super().__init__(full_message, dataset_name=dataset_name, **kwargs)


class InsufficientDataError(DatasetError):
    """Exception raised when dataset has insufficient data."""
    
    def __init__(
        self,
        dataset_name: str,
        n_samples: int,
        min_required: int,
        operation: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        """Initialize insufficient data error.
        
        Args:
            dataset_name: Name of the dataset
            n_samples: Number of samples in dataset
            min_required: Minimum required samples
            operation: Operation that requires more data
            **kwargs: Additional details
        """
        message = (
            f"Dataset '{dataset_name}' has insufficient data: "
            f"{n_samples} samples, but {min_required} required"
        )
        
        if operation:
            message += f" for {operation}"
        
        super().__init__(
            message,
            details={
                "dataset_name": dataset_name,
                "n_samples": n_samples,
                "min_required": min_required,
                "operation": operation,
                **kwargs
            }
        )


class DataTypeError(DatasetError):
    """Exception raised when data types are invalid."""
    
    def __init__(
        self,
        message: str,
        feature: Optional[str] = None,
        expected_type: Optional[str] = None,
        actual_type: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        """Initialize data type error.
        
        Args:
            message: Error message
            feature: Feature with wrong type
            expected_type: Expected data type
            actual_type: Actual data type
            **kwargs: Additional details
        """
        details = kwargs
        
        if feature:
            details["feature"] = feature
        if expected_type:
            details["expected_type"] = expected_type
        if actual_type:
            details["actual_type"] = actual_type
        
        super().__init__(message, details)


class FeatureMismatchError(DatasetError):
    """Exception raised when features don't match between datasets."""
    
    def __init__(
        self,
        message: str,
        expected_features: Optional[list[str]] = None,
        actual_features: Optional[list[str]] = None,
        missing_features: Optional[list[str]] = None,
        extra_features: Optional[list[str]] = None,
        **kwargs: Any
    ) -> None:
        """Initialize feature mismatch error.
        
        Args:
            message: Error message
            expected_features: Expected feature names
            actual_features: Actual feature names
            missing_features: Features that are missing
            extra_features: Features that are extra
            **kwargs: Additional details
        """
        details = kwargs
        
        if expected_features:
            details["expected_features"] = expected_features
        if actual_features:
            details["actual_features"] = actual_features
        if missing_features:
            details["missing_features"] = missing_features
        if extra_features:
            details["extra_features"] = extra_features
        
        super().__init__(message, details)