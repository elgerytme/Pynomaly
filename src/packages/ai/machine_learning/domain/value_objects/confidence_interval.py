"""
Confidence Interval value object for machine learning domain.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ConfidenceInterval:
    """
    Represents a confidence interval value object.
    
    Attributes:
        lower_bound: Lower bound of the confidence interval
        upper_bound: Upper bound of the confidence interval
        confidence_level: Confidence level (e.g., 0.95 for 95%)
        method: Method used to calculate the interval
        metadata: Additional metadata about the interval
    """
    
    lower_bound: float
    upper_bound: float
    confidence_level: float
    method: str | None = None
    metadata: dict[str, Any] | None = None
    
    def __post_init__(self) -> None:
        """Validate the confidence interval."""
        if self.lower_bound > self.upper_bound:
            raise ValueError(f"Lower bound {self.lower_bound} must be <= upper bound {self.upper_bound}")
        
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError(f"Confidence level must be between 0.0 and 1.0, got {self.confidence_level}")
    
    @classmethod
    def create(
        cls,
        lower_bound: float,
        upper_bound: float,
        confidence_level: float,
        method: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ConfidenceInterval:
        """Create a new confidence interval."""
        return cls(
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            confidence_level=confidence_level,
            method=method,
            metadata=metadata or {},
        )
    
    @classmethod
    def from_mean_and_margin(
        cls,
        mean: float,
        margin_of_error: float,
        confidence_level: float,
        method: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ConfidenceInterval:
        """Create a confidence interval from mean and margin of error."""
        return cls(
            lower_bound=mean - margin_of_error,
            upper_bound=mean + margin_of_error,
            confidence_level=confidence_level,
            method=method,
            metadata=metadata or {},
        )
    
    def width(self) -> float:
        """Calculate the width of the confidence interval."""
        return self.upper_bound - self.lower_bound
    
    def center(self) -> float:
        """Calculate the center (mean) of the confidence interval."""
        return (self.lower_bound + self.upper_bound) / 2
    
    def margin_of_error(self) -> float:
        """Calculate the margin of error (half the width)."""
        return self.width() / 2
    
    def contains(self, value: float) -> bool:
        """Check if a value is within the confidence interval."""
        return self.lower_bound <= value <= self.upper_bound
    
    def overlaps(self, other: ConfidenceInterval) -> bool:
        """Check if this interval overlaps with another."""
        return not (self.upper_bound < other.lower_bound or other.upper_bound < self.lower_bound)
    
    def is_narrow(self, threshold: float = 0.1) -> bool:
        """Check if the interval is narrow (width < threshold)."""
        return self.width() < threshold
    
    def is_wide(self, threshold: float = 0.5) -> bool:
        """Check if the interval is wide (width > threshold)."""
        return self.width() > threshold
    
    def relative_width(self) -> float:
        """Calculate the relative width (width / center)."""
        center = self.center()
        if center == 0:
            return float('inf')
        return self.width() / abs(center)
    
    def with_confidence_level(self, confidence_level: float) -> ConfidenceInterval:
        """Create a new interval with different confidence level (scales width)."""
        import math
        
        # Approximate scaling factor (assumes normal distribution)
        # This is a simplified approach - real scaling would need the original data
        old_z = self._get_z_score(self.confidence_level)
        new_z = self._get_z_score(confidence_level)
        
        scale_factor = new_z / old_z
        center = self.center()
        new_margin = self.margin_of_error() * scale_factor
        
        return ConfidenceInterval(
            lower_bound=center - new_margin,
            upper_bound=center + new_margin,
            confidence_level=confidence_level,
            method=self.method,
            metadata=self.metadata,
        )
    
    def _get_z_score(self, confidence_level: float) -> float:
        """Get approximate z-score for confidence level."""
        # Simplified lookup - in practice, use scipy.stats.norm.ppf
        lookup = {
            0.90: 1.645,
            0.95: 1.96,
            0.99: 2.576,
        }
        return lookup.get(confidence_level, 1.96)  # Default to 95%
    
    def __str__(self) -> str:
        """String representation of the confidence interval."""
        percentage = int(self.confidence_level * 100)
        return f"[{self.lower_bound:.3f}, {self.upper_bound:.3f}] ({percentage}% CI)"
    
    def __contains__(self, value: float) -> bool:
        """Check if a value is within the confidence interval."""
        return self.contains(value)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "confidence_level": self.confidence_level,
            "method": self.method,
            "metadata": self.metadata,
            "width": self.width(),
            "center": self.center(),
        }