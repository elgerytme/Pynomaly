"""
Confidence interval value object for uncertainty quantification.

This module provides the ConfidenceInterval value object that represents
statistical confidence bounds for anomaly detection predictions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np

from pynomaly.domain.exceptions import InvalidValueError


@dataclass(frozen=True)
class ConfidenceInterval:
    """
    Represents a confidence interval for uncertainty quantification.
    
    A confidence interval provides a range of values that is likely to contain
    the true value with a specified level of confidence.
    """
    
    lower: float
    upper: float
    confidence_level: float = 0.95
    method: str = "unknown"
    
    def __post_init__(self) -> None:
        """Validate confidence interval parameters."""
        if not isinstance(self.lower, (int, float)):
            raise InvalidValueError(
                f"Lower bound must be numeric, got {type(self.lower)}"
            )
        
        if not isinstance(self.upper, (int, float)):
            raise InvalidValueError(
                f"Upper bound must be numeric, got {type(self.upper)}"
            )
        
        if self.lower > self.upper:
            raise InvalidValueError(
                f"Lower bound ({self.lower}) cannot be greater than "
                f"upper bound ({self.upper})"
            )
        
        if not 0.0 <= self.confidence_level <= 1.0:
            raise InvalidValueError(
                f"Confidence level must be between 0 and 1, got {self.confidence_level}"
            )
        
        if not isinstance(self.method, str) or not self.method.strip():
            raise InvalidValueError("Method must be a non-empty string")
    
    def is_valid(self) -> bool:
        """Check if the confidence interval is valid."""
        return (
            isinstance(self.lower, (int, float))
            and isinstance(self.upper, (int, float))
            and self.lower <= self.upper
            and 0.0 <= self.confidence_level <= 1.0
            and isinstance(self.method, str)
            and self.method.strip()
        )
    
    def width(self) -> float:
        """Calculate the width of the confidence interval."""
        return round(self.upper - self.lower, 10)
    
    def midpoint(self) -> float:
        """Calculate the midpoint of the confidence interval."""
        return (self.lower + self.upper) / 2
    
    @property
    def center(self) -> float:
        """Calculate center of the interval (alias for midpoint)."""
        return self.midpoint()
    
    @property
    def margin_of_error(self) -> float:
        """Calculate the margin of error (half-width)."""
        return self.width() / 2
    
    def contains(self, value: float) -> bool:
        """Check if a value falls within the confidence interval."""
        return self.lower <= value <= self.upper
    
    def overlaps(self, other: "ConfidenceInterval") -> bool:
        """Check if this interval overlaps with another confidence interval."""
        return not (self.upper < other.lower or self.lower > other.upper)
    
    def intersection(self, other: "ConfidenceInterval") -> Optional["ConfidenceInterval"]:
        """
        Calculate the intersection of two confidence intervals.
        
        Returns None if the intervals don't overlap.
        """
        if not self.overlaps(other):
            return None
        
        new_lower = max(self.lower, other.lower)
        new_upper = min(self.upper, other.upper)
        
        # Use the more conservative confidence level
        new_confidence = min(self.confidence_level, other.confidence_level)
        
        return ConfidenceInterval(
            lower=new_lower,
            upper=new_upper,
            confidence_level=new_confidence,
            method=f"intersection({self.method}, {other.method})"
        )
    
    def to_tuple(self) -> Tuple[float, float]:
        """Convert to a tuple of (lower, upper)."""
        return (self.lower, self.upper)
    
    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "lower": self.lower,
            "upper": self.upper,
            "confidence_level": self.confidence_level,
            "method": self.method,
            "width": self.width(),
            "midpoint": self.midpoint(),
            "margin_of_error": self.margin_of_error
        }
    
    @classmethod
    def from_bounds(
        cls,
        lower: float,
        upper: float,
        confidence_level: float = 0.95,
        method: str = "manual"
    ) -> "ConfidenceInterval":
        """Create confidence interval from explicit bounds."""
        return cls(
            lower=lower,
            upper=upper,
            confidence_level=confidence_level,
            method=method
        )
    
    @classmethod
    def from_center_and_margin(
        cls,
        center: float,
        margin_of_error: float,
        confidence_level: float = 0.95,
        method: str = "manual"
    ) -> "ConfidenceInterval":
        """Create confidence interval from center point and margin of error."""
        return cls(
            lower=center - margin_of_error,
            upper=center + margin_of_error,
            confidence_level=confidence_level,
            method=method
        )
    
    @classmethod
    def from_samples(
        cls,
        samples: Union[np.ndarray, list],
        confidence_level: float = 0.95,
        method: str = "percentile"
    ) -> "ConfidenceInterval":
        """
        Create confidence interval from samples using percentile method.
        
        Args:
            samples: Array of sample values
            confidence_level: Desired confidence level (0.0 to 1.0)
            method: Method used for calculation
            
        Returns:
            ConfidenceInterval based on sample percentiles
        """
        if isinstance(samples, list):
            samples = np.array(samples)
        
        if len(samples) == 0:
            raise InvalidValueError("Cannot create confidence interval from empty samples")
        
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_bound = float(np.percentile(samples, lower_percentile))
        upper_bound = float(np.percentile(samples, upper_percentile))
        
        return cls(
            lower=lower_bound,
            upper=upper_bound,
            confidence_level=confidence_level,
            method=f"{method}_percentile"
        )
    
    def __str__(self) -> str:
        """String representation of confidence interval."""
        return (
            f"[{self.lower:.4f}, {self.upper:.4f}] "
            f"({self.confidence_level:.1%} confidence, {self.method})"
        )
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return (
            f"ConfidenceInterval(lower={self.lower}, "
            f"upper={self.upper}, "
            f"confidence_level={self.confidence_level}, "
            f"method='{self.method}')"
        )
