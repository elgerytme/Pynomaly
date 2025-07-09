"""Value objects for the domain layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AnomalyScore:
    """Value object for anomaly scores."""

    value: float

    def __post_init__(self) -> None:
        """Validate the score value."""
        if not isinstance(self.value, int | float):
            raise TypeError(f"Score value must be a number, got {type(self.value)}")

        if not 0.0 <= self.value <= 1.0:
            raise ValueError(f"Score value must be between 0 and 1, got {self.value}")

    def __float__(self) -> float:
        """Convert to float."""
        return self.value

    def __str__(self) -> str:
        """String representation."""
        return f"{self.value:.3f}"

    def __repr__(self) -> str:
        """Representation."""
        return f"AnomalyScore({self.value})"

    def __eq__(self, other: object) -> bool:
        """Equality comparison."""
        if isinstance(other, AnomalyScore):
            return self.value == other.value
        elif isinstance(other, int | float):
            return self.value == other
        return False

    def __lt__(self, other: object) -> bool:
        """Less than comparison."""
        if isinstance(other, AnomalyScore):
            return self.value < other.value
        elif isinstance(other, int | float):
            return self.value < other
        return NotImplemented

    def __le__(self, other: object) -> bool:
        """Less than or equal comparison."""
        if isinstance(other, AnomalyScore):
            return self.value <= other.value
        elif isinstance(other, int | float):
            return self.value <= other
        return NotImplemented

    def __gt__(self, other: object) -> bool:
        """Greater than comparison."""
        if isinstance(other, AnomalyScore):
            return self.value > other.value
        elif isinstance(other, int | float):
            return self.value > other
        return NotImplemented

    def __ge__(self, other: object) -> bool:
        """Greater than or equal comparison."""
        if isinstance(other, AnomalyScore):
            return self.value >= other.value
        elif isinstance(other, int | float):
            return self.value >= other
        return NotImplemented


@dataclass(frozen=True)
class DetectorConfig:
    """Value object for detector configuration."""

    algorithm: str
    parameters: dict[str, Any]

    def __post_init__(self) -> None:
        """Validate detector configuration."""
        if not isinstance(self.algorithm, str) or not self.algorithm.strip():
            raise ValueError("Algorithm must be a non-empty string")

        if not isinstance(self.parameters, dict):
            raise TypeError("Parameters must be a dictionary")

    def get_parameter(self, key: str, default: Any = None) -> Any:
        """Get a parameter value."""
        return self.parameters.get(key, default)

    def with_parameter(self, key: str, value: Any) -> DetectorConfig:
        """Return a new config with updated parameter."""
        new_params = self.parameters.copy()
        new_params[key] = value
        return DetectorConfig(self.algorithm, new_params)


@dataclass(frozen=True)
class DatasetMetadata:
    """Value object for dataset metadata."""

    name: str
    size: int
    features: list[str]
    feature_types: dict[str, str]

    def __post_init__(self) -> None:
        """Validate dataset metadata."""
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Dataset name must be a non-empty string")

        if not isinstance(self.size, int) or self.size < 0:
            raise ValueError("Dataset size must be a non-negative integer")

        if not isinstance(self.features, list):
            raise TypeError("Features must be a list")

        if not isinstance(self.feature_types, dict):
            raise TypeError("Feature types must be a dictionary")

    @property
    def num_features(self) -> int:
        """Number of features."""
        return len(self.features)

    def get_feature_type(self, feature: str) -> str | None:
        """Get the type of a feature."""
        return self.feature_types.get(feature)


@dataclass(frozen=True)
class ConfidenceInterval:
    """Value object for confidence intervals."""

    lower: float
    upper: float
    level: float = 0.95

    def __post_init__(self) -> None:
        """Validate confidence interval."""
        if not isinstance(self.lower, int | float):
            raise TypeError(f"Lower bound must be a number, got {type(self.lower)}")
        
        if not isinstance(self.upper, int | float):
            raise TypeError(f"Upper bound must be a number, got {type(self.upper)}")
        
        if self.lower > self.upper:
            raise ValueError(
                f"Lower bound ({self.lower}) must be <= upper bound ({self.upper})"
            )
        
        if not 0.0 < self.level < 1.0:
            raise ValueError(
                f"Confidence level must be between 0 and 1, got {self.level}"
            )

    @property
    def width(self) -> float:
        """Get the width of the confidence interval."""
        return self.upper - self.lower

    @property
    def center(self) -> float:
        """Get the center of the confidence interval."""
        return (self.lower + self.upper) / 2.0

    def contains(self, value: float) -> bool:
        """Check if value is within the confidence interval."""
        return self.lower <= value <= self.upper

    def __str__(self) -> str:
        """String representation."""
        return f"[{self.lower:.3f}, {self.upper:.3f}] ({self.level:.0%})"

    def __repr__(self) -> str:
        """Representation."""
        return (
            f"ConfidenceInterval(lower={self.lower}, upper={self.upper}, "
            f"level={self.level})"
        )


@dataclass(frozen=True)
class ContaminationRate:
    """Value object for contamination rate in anomaly detection."""

    value: float
    is_auto: bool = False

    def __post_init__(self) -> None:
        """Validate contamination rate."""
        if not isinstance(self.value, int | float):
            raise TypeError(
                f"Contamination rate must be a number, got {type(self.value)}"
            )
        
        if not self.is_auto and not 0.0 < self.value < 1.0:
            raise ValueError(
                f"Contamination rate must be between 0 and 1, got {self.value}"
            )

    @classmethod
    def auto(cls) -> ContaminationRate:
        """Create auto contamination rate."""
        return cls(value=0.1, is_auto=True)

    @classmethod
    def from_percentage(cls, percentage: float) -> ContaminationRate:
        """Create contamination rate from percentage."""
        return cls(value=percentage / 100.0)

    def to_percentage(self) -> float:
        """Convert to percentage."""
        return self.value * 100.0

    def __str__(self) -> str:
        """String representation."""
        if self.is_auto:
            return f"auto ({self.value:.1%})"
        return f"{self.value:.1%}"

    def __repr__(self) -> str:
        """Representation."""
        return f"ContaminationRate(value={self.value}, is_auto={self.is_auto})"
