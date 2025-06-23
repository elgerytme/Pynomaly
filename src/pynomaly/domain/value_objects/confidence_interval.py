"""Confidence interval value object."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class ConfidenceInterval:
    """Immutable value object representing a confidence interval.
    
    Attributes:
        lower: Lower bound of the interval
        upper: Upper bound of the interval
        level: Confidence level (e.g., 0.95 for 95% CI)
    """
    
    lower: float
    upper: float
    level: float = 0.95
    
    def __post_init__(self) -> None:
        """Validate confidence interval after initialization."""
        if not isinstance(self.lower, (int, float)):
            raise TypeError(f"Lower bound must be numeric, got {type(self.lower)}")
        
        if not isinstance(self.upper, (int, float)):
            raise TypeError(f"Upper bound must be numeric, got {type(self.upper)}")
        
        if not isinstance(self.level, (int, float)):
            raise TypeError(f"Confidence level must be numeric, got {type(self.level)}")
        
        if self.lower > self.upper:
            raise ValueError(
                f"Lower bound ({self.lower}) cannot be greater than "
                f"upper bound ({self.upper})"
            )
        
        if not 0 < self.level < 1:
            raise ValueError(
                f"Confidence level must be between 0 and 1, got {self.level}"
            )
    
    @classmethod
    def from_point_estimate(
        cls,
        point: float,
        margin: float,
        level: float = 0.95
    ) -> ConfidenceInterval:
        """Create interval from point estimate and margin of error."""
        return cls(
            lower=point - margin,
            upper=point + margin,
            level=level
        )
    
    @classmethod
    def from_percentiles(
        cls,
        data: list[float],
        level: float = 0.95
    ) -> ConfidenceInterval:
        """Create interval from data percentiles."""
        if not data:
            raise ValueError("Cannot compute percentiles from empty data")
        
        sorted_data = sorted(data)
        n = len(sorted_data)
        
        # Calculate percentile positions
        alpha = 1 - level
        lower_idx = int(n * (alpha / 2))
        upper_idx = int(n * (1 - alpha / 2)) - 1
        
        # Ensure valid indices
        lower_idx = max(0, lower_idx)
        upper_idx = min(n - 1, upper_idx)
        
        return cls(
            lower=sorted_data[lower_idx],
            upper=sorted_data[upper_idx],
            level=level
        )
    
    @property
    def width(self) -> float:
        """Calculate width of the interval."""
        return self.upper - self.lower
    
    @property
    def midpoint(self) -> float:
        """Calculate midpoint of the interval."""
        return (self.lower + self.upper) / 2
    
    @property
    def margin_of_error(self) -> float:
        """Calculate margin of error (half-width)."""
        return self.width / 2
    
    @property
    def as_percentage(self) -> int:
        """Get confidence level as percentage."""
        return int(self.level * 100)
    
    def contains(self, value: float) -> bool:
        """Check if value is within the interval."""
        return self.lower <= value <= self.upper
    
    def overlaps(self, other: ConfidenceInterval) -> bool:
        """Check if this interval overlaps with another."""
        return not (self.upper < other.lower or self.lower > other.upper)
    
    def union(self, other: ConfidenceInterval) -> Optional[ConfidenceInterval]:
        """Create union of two overlapping intervals."""
        if not self.overlaps(other):
            return None
        
        # Use the more conservative (lower) confidence level
        level = min(self.level, other.level)
        
        return ConfidenceInterval(
            lower=min(self.lower, other.lower),
            upper=max(self.upper, other.upper),
            level=level
        )
    
    def intersection(self, other: ConfidenceInterval) -> Optional[ConfidenceInterval]:
        """Create intersection of two intervals."""
        if not self.overlaps(other):
            return None
        
        # Use the less conservative (higher) confidence level
        level = max(self.level, other.level)
        
        return ConfidenceInterval(
            lower=max(self.lower, other.lower),
            upper=min(self.upper, other.upper),
            level=level
        )
    
    def as_tuple(self) -> Tuple[float, float]:
        """Return interval as tuple (lower, upper)."""
        return (self.lower, self.upper)
    
    def __str__(self) -> str:
        """String representation."""
        return f"[{self.lower:.3f}, {self.upper:.3f}] ({self.as_percentage}% CI)"
    
    def __repr__(self) -> str:
        """Developer representation."""
        return (
            f"ConfidenceInterval(lower={self.lower}, upper={self.upper}, "
            f"level={self.level})"
        )