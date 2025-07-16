"""Generic detection use case for any detection algorithm."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, TypeVar
from uuid import UUID

# Generic type for detection input data
T = TypeVar('T')
# Generic type for detection output results
R = TypeVar('R')


@dataclass
class GenericDetectionRequest(Generic[T]):
    """Generic request for detection."""
    
    detector_id: UUID
    data: T
    validate_input: bool = True
    save_results: bool = True
    metadata: dict[str, Any] = None

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}


@dataclass 
class GenericDetectionResponse(Generic[R]):
    """Generic response from detection."""
    
    result: R
    quality_report: dict[str, Any] | None = None
    warnings: list[str] | None = None
    metadata: dict[str, Any] | None = None


class GenericDetectionUseCase(ABC, Generic[T, R]):
    """Abstract base class for detection use cases.
    
    This provides a domain-agnostic interface for any type of detection
    (anomaly, fraud, intrusion, etc.) while maintaining clean architecture.
    """

    @abstractmethod
    async def execute(self, request: GenericDetectionRequest[T]) -> GenericDetectionResponse[R]:
        """Execute detection.
        
        Args:
            request: Detection request with input data
            
        Returns:
            Detection response with results
            
        Raises:
            ValueError: If detector is not found or not fitted
            Exception: If detection fails
        """

    @abstractmethod
    def validate_input(self, data: T) -> tuple[bool, list[str]]:
        """Validate input data.
        
        Args:
            data: Input data to validate
            
        Returns:
            Tuple of (is_valid, list_of_warnings)
        """

    @abstractmethod
    def generate_quality_report(self, data: T) -> dict[str, Any]:
        """Generate quality report for input data.
        
        Args:
            data: Input data to analyze
            
        Returns:
            Quality report dictionary
        """


class DetectionMetrics:
    """Generic metrics for any detection algorithm."""
    
    @staticmethod
    def calculate_execution_time(start_time: float, end_time: float) -> float:
        """Calculate execution time in milliseconds."""
        return (end_time - start_time) * 1000
    
    @staticmethod
    def calculate_throughput(num_samples: int, execution_time_ms: float) -> float:
        """Calculate samples processed per second."""
        if execution_time_ms <= 0:
            return 0.0
        return (num_samples * 1000) / execution_time_ms
    
    @staticmethod
    def calculate_confidence_statistics(scores: list[float]) -> dict[str, float]:
        """Calculate confidence statistics from scores."""
        if not scores:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        
        import statistics
        return {
            "mean": statistics.mean(scores),
            "std": statistics.stdev(scores) if len(scores) > 1 else 0.0,
            "min": min(scores),
            "max": max(scores),
        }