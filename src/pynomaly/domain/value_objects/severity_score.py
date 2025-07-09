"""Severity score value object."""

from dataclasses import dataclass
from enum import Enum


class SeverityLevel(str, Enum):
    """Enumeration for severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    def __str__(self) -> str:
        """Return the string value of the enum."""
        return self.value


@dataclass(frozen=True)
class SeverityScore:
    """Value object representing the severity of an anomaly."""

    value: float
    severity_level: SeverityLevel
    confidence: float | None = None

    def __post_init__(self) -> None:
        """Validate severity score."""
        if not 0.0 <= self.value <= 1.0:
            raise ValueError(
                f"Severity score must be between 0.0 and 1.0, got {self.value}"
            )

        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"Confidence must be between 0.0 and 1.0, got {self.confidence}"
            )

    @classmethod
    def create_minimal(cls) -> "SeverityScore":
        """Create a minimal severity score."""
        return cls(value=0.0, severity_level=SeverityLevel.LOW)

    @classmethod
    def from_score(cls, score: float) -> "SeverityScore":
        """Create severity score from a numeric score."""
        if score >= 0.8:
            level = SeverityLevel.CRITICAL
        elif score >= 0.6:
            level = SeverityLevel.HIGH
        elif score >= 0.4:
            level = SeverityLevel.MEDIUM
        else:
            level = SeverityLevel.LOW

        return cls(value=score, severity_level=level)

    def is_critical(self) -> bool:
        """Check if this is a critical severity score."""
        return self.severity_level == SeverityLevel.CRITICAL

    def is_high(self) -> bool:
        """Check if this is a high severity score."""
        return self.severity_level in [SeverityLevel.HIGH, SeverityLevel.CRITICAL]
