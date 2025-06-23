"""Detection result-specific domain exceptions."""

from __future__ import annotations

from typing import Any, Optional

from .base import DomainError


class ResultError(DomainError):
    """Base exception for detection result-related errors."""
    pass


class ScoreCalculationError(ResultError):
    """Exception raised when score calculation fails."""
    
    def __init__(
        self,
        message: str,
        detector_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        """Initialize score calculation error.
        
        Args:
            message: Error message
            detector_name: Name of the detector
            dataset_name: Name of the dataset
            **kwargs: Additional details
        """
        details = kwargs
        
        if detector_name:
            details["detector_name"] = detector_name
        if dataset_name:
            details["dataset_name"] = dataset_name
        
        super().__init__(message, details)


class ThresholdError(ResultError):
    """Exception raised when threshold calculation or application fails."""
    
    def __init__(
        self,
        message: str,
        threshold_value: Optional[float] = None,
        contamination_rate: Optional[float] = None,
        n_samples: Optional[int] = None,
        **kwargs: Any
    ) -> None:
        """Initialize threshold error.
        
        Args:
            message: Error message
            threshold_value: The problematic threshold
            contamination_rate: Contamination rate used
            n_samples: Number of samples
            **kwargs: Additional details
        """
        details = kwargs
        
        if threshold_value is not None:
            details["threshold_value"] = threshold_value
        if contamination_rate is not None:
            details["contamination_rate"] = contamination_rate
        if n_samples is not None:
            details["n_samples"] = n_samples
        
        super().__init__(message, details)


class InconsistentResultError(ResultError):
    """Exception raised when detection result is internally inconsistent."""
    
    def __init__(
        self,
        message: str,
        n_scores: Optional[int] = None,
        n_labels: Optional[int] = None,
        n_anomalies_expected: Optional[int] = None,
        n_anomalies_actual: Optional[int] = None,
        **kwargs: Any
    ) -> None:
        """Initialize inconsistent result error.
        
        Args:
            message: Error message
            n_scores: Number of scores
            n_labels: Number of labels
            n_anomalies_expected: Expected number of anomalies
            n_anomalies_actual: Actual number of anomalies
            **kwargs: Additional details
        """
        details = kwargs
        
        if n_scores is not None:
            details["n_scores"] = n_scores
        if n_labels is not None:
            details["n_labels"] = n_labels
        if n_anomalies_expected is not None:
            details["n_anomalies_expected"] = n_anomalies_expected
        if n_anomalies_actual is not None:
            details["n_anomalies_actual"] = n_anomalies_actual
        
        super().__init__(message, details)