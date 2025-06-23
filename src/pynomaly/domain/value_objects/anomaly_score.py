"""Anomaly score value object."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class AnomalyScore:
    """Immutable value object representing an anomaly score.
    
    Attributes:
        value: The anomaly score value (higher means more anomalous)
        confidence_lower: Lower bound of confidence interval (optional)
        confidence_upper: Upper bound of confidence interval (optional)
        method: The scoring method used (optional)
    """
    
    value: float
    confidence_lower: Optional[float] = None
    confidence_upper: Optional[float] = None
    method: Optional[str] = None
    
    def __post_init__(self) -> None:
        """Validate score after initialization."""
        if not isinstance(self.value, (int, float)):
            raise ValueError(f"Score value must be numeric, got {type(self.value)}")
        
        if self.confidence_lower is not None and self.confidence_upper is not None:
            if self.confidence_lower > self.confidence_upper:
                raise ValueError(
                    f"Lower confidence bound ({self.confidence_lower}) cannot be "
                    f"greater than upper bound ({self.confidence_upper})"
                )
            if not (self.confidence_lower <= self.value <= self.confidence_upper):
                raise ValueError(
                    f"Score value ({self.value}) must be within confidence interval "
                    f"[{self.confidence_lower}, {self.confidence_upper}]"
                )
    
    @property
    def is_confident(self) -> bool:
        """Check if score has confidence intervals."""
        return self.confidence_lower is not None and self.confidence_upper is not None
    
    @property
    def confidence_width(self) -> Optional[float]:
        """Calculate width of confidence interval."""
        if self.is_confident:
            return self.confidence_upper - self.confidence_lower  # type: ignore
        return None
    
    def exceeds_threshold(self, threshold: float) -> bool:
        """Check if score exceeds a given threshold."""
        return self.value > threshold
    
    def __lt__(self, other: Any) -> bool:
        """Compare scores by value."""
        if isinstance(other, AnomalyScore):
            return self.value < other.value
        if isinstance(other, (int, float)):
            return self.value < other
        return NotImplemented
    
    def __le__(self, other: Any) -> bool:
        """Compare scores by value."""
        if isinstance(other, AnomalyScore):
            return self.value <= other.value
        if isinstance(other, (int, float)):
            return self.value <= other
        return NotImplemented
    
    def __gt__(self, other: Any) -> bool:
        """Compare scores by value."""
        if isinstance(other, AnomalyScore):
            return self.value > other.value
        if isinstance(other, (int, float)):
            return self.value > other
        return NotImplemented
    
    def __ge__(self, other: Any) -> bool:
        """Compare scores by value."""
        if isinstance(other, AnomalyScore):
            return self.value >= other.value
        if isinstance(other, (int, float)):
            return self.value >= other
        return NotImplemented