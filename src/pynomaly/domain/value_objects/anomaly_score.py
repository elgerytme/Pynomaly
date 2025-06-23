"""Anomaly score value object."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from pynomaly.domain.exceptions import InvalidValueError


@dataclass(frozen=True)
class AnomalyScore:
    """Immutable value object representing an anomaly score.
    
    Attributes:
        value: The anomaly score value (higher means more anomalous)
        confidence_interval: Confidence interval for the score (optional)
        method: The scoring method used (optional)
    """
    
    value: float
    confidence_interval: Optional['ConfidenceInterval'] = None
    method: Optional[str] = None
    
    def __post_init__(self) -> None:
        """Validate score after initialization."""
        if not isinstance(self.value, (int, float)):
            raise InvalidValueError(f"Score value must be numeric, got {type(self.value)}")
        
        if not (0.0 <= self.value <= 1.0):
            raise InvalidValueError(f"Score value must be between 0 and 1, got {self.value}")
        
        if self.confidence_interval is not None:
            if not self.confidence_interval.contains(self.value):
                raise InvalidValueError(
                    f"Score value ({self.value}) must be within confidence interval "
                    f"[{self.confidence_interval.lower}, {self.confidence_interval.upper}]"
                )
    
    def is_valid(self) -> bool:
        """Check if the score is valid."""
        return isinstance(self.value, (int, float)) and not (
            hasattr(self.value, '__isnan__') and self.value.__isnan__()
        )
    
    @property
    def is_confident(self) -> bool:
        """Check if score has confidence intervals."""
        return self.confidence_interval is not None
    
    @property
    def confidence_width(self) -> Optional[float]:
        """Calculate width of confidence interval."""
        if self.is_confident and self.confidence_interval:
            return self.confidence_interval.width()
        return None
    
    @property
    def confidence_lower(self) -> Optional[float]:
        """Get lower bound of confidence interval."""
        if self.confidence_interval:
            return self.confidence_interval.lower
        return None
    
    @property
    def confidence_upper(self) -> Optional[float]:
        """Get upper bound of confidence interval."""
        if self.confidence_interval:
            return self.confidence_interval.upper
        return None
    
    def exceeds_threshold(self, threshold: float) -> bool:
        """Check if score exceeds a given threshold."""
        return self.value > threshold
    
    def __str__(self) -> str:
        """String representation of the score."""
        return str(self.value)
    
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


# Import here to avoid circular imports
from pynomaly.domain.value_objects.confidence_interval import ConfidenceInterval
