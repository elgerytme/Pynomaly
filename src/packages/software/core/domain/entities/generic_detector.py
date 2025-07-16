"""Generic detector entity for any detection algorithm."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Generic, TypeVar
from uuid import UUID, uuid4

# Generic type for detection input data
T = TypeVar('T')
# Generic type for detection output results  
R = TypeVar('R')


@dataclass
class GenericDetector(ABC, Generic[T, R]):
    """Abstract base class for any detection algorithm.
    
    This provides a domain-agnostic interface that can be implemented
    by any type of detector (anomaly, fraud, intrusion, etc.).
    
    Attributes:
        id: Unique identifier for the detector
        name: Name of the detector
        algorithm_name: Name of the underlying algorithm
        parameters: Algorithm-specific parameters
        metadata: Additional metadata
        created_at: When the detector was created
        trained_at: When the detector was last trained
        is_fitted: Whether the detector has been fitted
    """
    
    name: str
    algorithm_name: str
    id: UUID = field(default_factory=uuid4)
    parameters: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    trained_at: datetime | None = None
    is_fitted: bool = False

    def __post_init__(self) -> None:
        """Validate detector after initialization."""
        if not self.name:
            raise ValueError("Detector name cannot be empty")
        if not self.algorithm_name:
            raise ValueError("Algorithm name cannot be empty")

    @abstractmethod
    def fit(self, data: T) -> None:
        """Fit the detector to training data.
        
        Args:
            data: Training data of type T
        """

    @abstractmethod
    def detect(self, data: T) -> R:
        """Perform detection on input data.
        
        Args:
            data: Input data of type T
            
        Returns:
            Detection results of type R
        """

    @abstractmethod
    def score(self, data: T) -> list[float]:
        """Score the input data.
        
        Args:
            data: Input data of type T
            
        Returns:
            List of scores for each data point
        """

    def get_info(self) -> dict[str, Any]:
        """Get comprehensive information about the detector."""
        return {
            "id": str(self.id),
            "name": self.name,
            "algorithm": self.algorithm_name,
            "is_fitted": self.is_fitted,
            "created_at": self.created_at.isoformat(),
            "trained_at": self.trained_at.isoformat() if self.trained_at else None,
            "parameters": self.parameters,
            "metadata": self.metadata,
        }

    def mark_as_fitted(self) -> None:
        """Mark the detector as fitted."""
        self.is_fitted = True
        self.trained_at = datetime.utcnow()

    def mark_as_unfitted(self) -> None:
        """Mark the detector as not fitted."""
        self.is_fitted = False
        self.trained_at = None

    def validate_for_detection(self) -> None:
        """Validate that the detector is ready for detection.
        
        Raises:
            ValueError: If detector is not fitted
        """
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before detection")

    def __repr__(self) -> str:
        """Developer representation."""
        return (
            f"{self.__class__.__name__}(id={self.id}, name='{self.name}', "
            f"algorithm='{self.algorithm_name}', is_fitted={self.is_fitted})"
        )