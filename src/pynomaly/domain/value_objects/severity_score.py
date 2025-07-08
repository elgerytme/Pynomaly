"""Severity score value object."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pynomaly.domain.exceptions import InvalidValueError
from pynomaly.domain.value_objects.severity_level import SeverityLevel


@dataclass(frozen=True)
class SeverityScore:
    """Immutable value object representing a severity score.

    Attributes:
        value: The severity score value (0.0 to 1.0, higher means more severe)
        severity_level: The corresponding severity level
    """

    value: float
    severity_level: SeverityLevel

    def __post_init__(self) -> None:
        """Validate score after initialization."""
        if not isinstance(self.value, (int, float)):
            raise InvalidValueError(
                f"Severity score value must be numeric, got {type(self.value)}"
            )

        if not (0.0 <= self.value <= 1.0):
            raise InvalidValueError(
                f"Severity score value must be between 0 and 1, got {self.value}"
            )

        if not isinstance(self.severity_level, SeverityLevel):
            raise InvalidValueError(
                f"Severity level must be a SeverityLevel enum, got {type(self.severity_level)}"
            )

    @classmethod
    def from_score(cls, score: float) -> "SeverityScore":
        """Create a SeverityScore from a numeric score.
        
        Args:
            score: Anomaly score (0.0 to 1.0)
            
        Returns:
            SeverityScore instance with appropriate severity level
        """
        severity_level = SeverityLevel.from_score(score)
        return cls(value=score, severity_level=severity_level)
    
    @classmethod
    def create_minimal(cls) -> "SeverityScore":
        """Create a minimal severity score."""
        return cls(value=0.0, severity_level=SeverityLevel.LOW)

    def is_valid(self) -> bool:
        """Check if the score is valid."""
        return isinstance(self.value, (int, float)) and not (
            hasattr(self.value, "__isnan__") and self.value.__isnan__()
        )

    def exceeds_threshold(self, threshold: float) -> bool:
        """Check if score exceeds a given threshold."""
        return self.value > threshold

    def __str__(self) -> str:
        """String representation of the score."""
        return f"{self.value} ({self.severity_level})"

    def __lt__(self, other: Any) -> bool:
        """Compare scores by value."""
        if isinstance(other, SeverityScore):
            return self.value < other.value
        if isinstance(other, (int, float)):
            return self.value < other
        return NotImplemented

    def __le__(self, other: Any) -> bool:
        """Compare scores by value."""
        if isinstance(other, SeverityScore):
            return self.value <= other.value
        if isinstance(other, (int, float)):
            return self.value <= other
        return NotImplemented

    def __gt__(self, other: Any) -> bool:
        """Compare scores by value."""
        if isinstance(other, SeverityScore):
            return self.value > other.value
        if isinstance(other, (int, float)):
            return self.value > other
        return NotImplemented

    def __ge__(self, other: Any) -> bool:
        """Compare scores by value."""
        if isinstance(other, SeverityScore):
            return self.value >= other.value
        if isinstance(other, (int, float)):
            return self.value >= other
        return NotImplemented
