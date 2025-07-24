"""Anomaly score value object."""

from dataclasses import dataclass
from typing import Protocol


@dataclass
class AnomalyScore:
    """Represents an anomaly score value."""
    value: float
    threshold: float = 0.5
    
    @property
    def is_anomaly(self) -> bool:
        """Check if score indicates an anomaly."""
        return self.value > self.threshold
    
    def __float__(self) -> float:
        """Allow conversion to float."""
        return self.value


class AnomalyScoreProtocol(Protocol):
    """Protocol for anomaly score objects."""
    value: float
    
    @property
    def is_anomaly(self) -> bool:
        """Check if score indicates an anomaly."""
        ...