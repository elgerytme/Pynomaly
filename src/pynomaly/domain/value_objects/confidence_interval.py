"""Confidence interval value object."""

from __future__ import annotations

from dataclasses import dataclass

from pynomaly.domain.exceptions import InvalidValueError


@dataclass(frozen=True)
class ConfidenceInterval:
    """Immutable value object representing a confidence interval.

    Attributes:
        lower: Lower bound of the interval
        upper: Upper bound of the interval
        confidence_level: Confidence level (e.g., 0.95 for 95%)
    """

    lower: float
    upper: float
    confidence_level: float = 0.95

    def __post_init__(self) -> None:
        """Validate confidence interval after initialization."""
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
                f"Lower bound ({self.lower}) cannot be greater than upper bound ({self.upper})"
            )

        if not (0.0 <= self.confidence_level <= 1.0):
            raise InvalidValueError(
                f"Confidence level must be between 0 and 1, got {self.confidence_level}"
            )

    def is_valid(self) -> bool:
        """Check if the confidence interval is valid."""
        return (
            isinstance(self.lower, (int, float))
            and isinstance(self.upper, (int, float))
            and self.lower <= self.upper
            and 0.0 <= self.confidence_level <= 1.0
        )

    def width(self) -> float:
        """Calculate width of the interval."""
        return round(self.upper - self.lower, 10)

    def midpoint(self) -> float:
        """Calculate midpoint of the interval."""
        return (self.lower + self.upper) / 2

    @property
    def center(self) -> float:
        """Calculate center of the interval (alias for midpoint)."""
        return self.midpoint()

    def contains(self, value: float) -> bool:
        """Check if value is within the interval."""
        return self.lower <= value <= self.upper

    def __str__(self) -> str:
        """String representation."""
        return f"[{self.lower:.3f}, {self.upper:.3f}] ({self.confidence_level:.0%})"
