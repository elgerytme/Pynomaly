"""Detector entity."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol
from uuid import UUID, uuid4

import pandas as pd

from pynomaly.domain.entities.anomaly import Anomaly
from pynomaly.domain.entities.dataset import Dataset
from pynomaly.domain.entities.detection_result import DetectionResult
from pynomaly.domain.value_objects import AnomalyScore, ContaminationRate


class DetectorAlgorithm(Protocol):
    """Protocol for detector algorithm implementations."""
    
    def fit(self, data: pd.DataFrame) -> None:
        """Fit the detector on training data."""
        ...
    
    def predict(self, data: pd.DataFrame) -> List[int]:
        """Predict anomaly labels (0=normal, 1=anomaly)."""
        ...
    
    def score(self, data: pd.DataFrame) -> List[float]:
        """Calculate anomaly scores."""
        ...


@dataclass
class Detector(ABC):
    """Abstract base entity for anomaly detectors.
    
    This is a domain entity that represents the concept of an anomaly detector,
    independent of any specific implementation or algorithm.
    
    Attributes:
        id: Unique identifier for the detector
        name: Name of the detector
        algorithm_name: Name of the underlying algorithm
        contamination_rate: Expected proportion of anomalies
        parameters: Algorithm-specific parameters
        metadata: Additional metadata
        created_at: When the detector was created
        trained_at: When the detector was last trained
        is_fitted: Whether the detector has been fitted
    """
    
    name: str
    algorithm_name: str
    contamination_rate: ContaminationRate = field(default_factory=ContaminationRate.auto)
    id: UUID = field(default_factory=uuid4)
    parameters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    trained_at: Optional[datetime] = None
    is_fitted: bool = False
    
    def __post_init__(self) -> None:
        """Validate detector after initialization."""
        if not self.name:
            raise ValueError("Detector name cannot be empty")
        
        if not self.algorithm_name:
            raise ValueError("Algorithm name cannot be empty")
        
        if not isinstance(self.contamination_rate, ContaminationRate):
            raise TypeError(
                f"Contamination rate must be ContaminationRate instance, "
                f"got {type(self.contamination_rate)}"
            )
    
    @abstractmethod
    def fit(self, dataset: Dataset) -> None:
        """Fit the detector on a dataset.
        
        This method should be implemented by infrastructure adapters.
        """
        pass
    
    @abstractmethod
    def detect(self, dataset: Dataset) -> DetectionResult:
        """Detect anomalies in a dataset.
        
        This method should be implemented by infrastructure adapters.
        """
        pass
    
    @abstractmethod
    def score(self, dataset: Dataset) -> List[AnomalyScore]:
        """Calculate anomaly scores for a dataset.
        
        This method should be implemented by infrastructure adapters.
        """
        pass
    
    def fit_detect(self, dataset: Dataset) -> DetectionResult:
        """Fit the detector and detect anomalies in one step."""
        self.fit(dataset)
        return self.detect(dataset)
    
    @property
    def requires_fitting(self) -> bool:
        """Check if detector requires fitting before detection."""
        # Most detectors require fitting, but some (like statistical tests) don't
        return self.metadata.get("requires_fitting", True)
    
    @property
    def supports_streaming(self) -> bool:
        """Check if detector supports streaming/online detection."""
        return self.metadata.get("supports_streaming", False)
    
    @property
    def supports_multivariate(self) -> bool:
        """Check if detector supports multivariate data."""
        return self.metadata.get("supports_multivariate", True)
    
    @property
    def time_complexity(self) -> Optional[str]:
        """Get time complexity of the algorithm."""
        return self.metadata.get("time_complexity")
    
    @property
    def space_complexity(self) -> Optional[str]:
        """Get space complexity of the algorithm."""
        return self.metadata.get("space_complexity")
    
    def update_metadata(self, key: str, value: Any) -> None:
        """Update detector metadata."""
        self.metadata[key] = value
    
    def update_parameters(self, **params: Any) -> None:
        """Update algorithm parameters."""
        self.parameters.update(params)
        # Reset fitted state when parameters change
        self.is_fitted = False
        self.trained_at = None
    
    def get_info(self) -> Dict[str, Any]:
        """Get comprehensive information about the detector."""
        return {
            "id": str(self.id),
            "name": self.name,
            "algorithm": self.algorithm_name,
            "contamination_rate": self.contamination_rate.value,
            "is_fitted": self.is_fitted,
            "created_at": self.created_at.isoformat(),
            "trained_at": self.trained_at.isoformat() if self.trained_at else None,
            "parameters": self.parameters,
            "metadata": self.metadata,
            "requires_fitting": self.requires_fitting,
            "supports_streaming": self.supports_streaming,
            "supports_multivariate": self.supports_multivariate,
            "time_complexity": self.time_complexity,
            "space_complexity": self.space_complexity,
        }
    
    def __repr__(self) -> str:
        """Developer representation."""
        return (
            f"Detector(id={self.id}, name='{self.name}', "
            f"algorithm='{self.algorithm_name}', is_fitted={self.is_fitted})"
        )