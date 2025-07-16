"""Anomaly score value object."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

from monorepo.domain.exceptions import ValidationError


@dataclass(frozen=True)
class AnomalyScore:
    """Immutable value object representing an anomaly score.

    Attributes:
        value: The anomaly score value (higher means more anomalous)
        threshold: The threshold for anomaly classification (default: 0.5)
        metadata: Additional metadata about the score
        confidence_interval: Confidence interval for the score (optional)
        method: The scoring method used (optional)
    """

    value: float
    threshold: float = 0.5
    metadata: dict[str, Any] = None
    confidence_interval: Any = None  # ConfidenceInterval | None
    method: str | None = None

    def __post_init__(self) -> None:
        """Validate score after initialization."""
        if self.metadata is None:
            object.__setattr__(self, "metadata", {})
        else:
            # Make a deep copy to prevent mutation of original dict
            import copy

            object.__setattr__(self, "metadata", copy.deepcopy(self.metadata))

        # Validate value
        if not isinstance(self.value, (int, float)):
            raise ValidationError(
                f"Score value must be numeric, got {type(self.value)}"
            )

        if math.isnan(self.value) or math.isinf(self.value):
            raise ValidationError(f"Score value must be finite, got {self.value}")

        if not (0.0 <= self.value <= 1.0):
            raise ValidationError(
                f"Score value must be between 0 and 1, got {self.value}"
            )

        # Validate threshold
        if not isinstance(self.threshold, (int, float)):
            raise ValidationError(
                f"Threshold must be numeric, got {type(self.threshold)}"
            )

        if math.isnan(self.threshold) or math.isinf(self.threshold):
            raise ValidationError(f"Threshold must be finite, got {self.threshold}")

        if not (0.0 <= self.threshold <= 1.0):
            raise ValidationError(
                f"Threshold must be between 0 and 1, got {self.threshold}"
            )

        # Validate metadata
        if not isinstance(self.metadata, dict):
            raise ValidationError(
                f"Metadata must be a dictionary, got {type(self.metadata)}"
            )

        # Validate confidence_interval if present
        if self.confidence_interval is not None:
            if (
                not hasattr(self.confidence_interval, "contains")
                or not hasattr(self.confidence_interval, "lower")
                or not hasattr(self.confidence_interval, "upper")
            ):
                raise ValidationError(
                    "Confidence interval must have 'contains', 'lower', and 'upper' attributes"
                )
            if not self.confidence_interval.contains(self.value):
                raise ValidationError(
                    f"Score value ({self.value}) must be within confidence interval "
                    f"[{self.confidence_interval.lower}, {self.confidence_interval.upper}]"
                )

    def is_anomaly(self) -> bool:
        """Check if score indicates an anomaly based on threshold."""
        return self.value > self.threshold

    def confidence_level(self) -> float:
        """Calculate confidence level based on distance from threshold."""
        return abs(self.value - self.threshold)

    @property
    def is_confident(self) -> bool:
        """Check if score has confidence intervals."""
        return self.confidence_interval is not None

    @property
    def confidence_width(self) -> float | None:
        """Calculate width of confidence interval."""
        if self.is_confident and self.confidence_interval:
            return self.confidence_interval.upper - self.confidence_interval.lower
        return None

    @property
    def confidence_lower(self) -> float | None:
        """Get lower bound of confidence interval."""
        if self.confidence_interval:
            return self.confidence_interval.lower
        return None

    @property
    def confidence_upper(self) -> float | None:
        """Get upper bound of confidence interval."""
        if self.confidence_interval:
            return self.confidence_interval.upper
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "value": self.value,
            "threshold": self.threshold,
            "metadata": self.metadata.copy(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AnomalyScore:
        """Create AnomalyScore from dictionary."""
        if "value" not in data:
            raise ValidationError("Missing required field: value")

        return cls(
            value=data["value"],
            threshold=data.get("threshold", 0.5),
            metadata=data.get("metadata", {}),
        )

    def to_json(self) -> str:
        """Convert to JSON representation."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> AnomalyScore:
        """Create AnomalyScore from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)

    def __str__(self) -> str:
        """String representation of the score."""
        return str(self.value)

    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"AnomalyScore(value={self.value}, threshold={self.threshold}, metadata={self.metadata})"

    def __bool__(self) -> bool:
        """Boolean conversion - True if score is non-zero."""
        return self.value != 0.0

    def __hash__(self) -> int:
        """Hash function for use in sets and dicts."""
        return hash(
            (
                self.value,
                self.threshold,
                tuple(sorted(self.metadata.items())),
                id(self.confidence_interval) if self.confidence_interval else None,
                self.method,
            )
        )

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

    def __eq__(self, other: Any) -> bool:
        """Compare scores for equality."""
        if isinstance(other, AnomalyScore):
            return (
                self.value == other.value
                and self.threshold == other.threshold
                and self.metadata == other.metadata
                and self.confidence_interval == other.confidence_interval
                and self.method == other.method
            )
        return False

    def __ne__(self, other: Any) -> bool:
        """Compare scores for inequality."""
        return not self.__eq__(other)

    def is_valid(self) -> bool:
        """Check if the score is valid."""
        return (
            isinstance(self.value, (int, float))
            and not math.isnan(self.value)
            and not math.isinf(self.value)
            and 0.0 <= self.value <= 1.0
        )

    def exceeds_threshold(self, threshold: float) -> bool:
        """Check if score exceeds a given threshold."""
        return self.value > threshold
