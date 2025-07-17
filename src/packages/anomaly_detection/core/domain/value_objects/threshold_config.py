"""Threshold configuration value object."""

from __future__ import annotations

from dataclasses import dataclass

from monorepo.domain.exceptions import InvalidValueError


@dataclass(frozen=True)
class ThresholdConfig:
    """Configuration for anomaly threshold calculation."""

    method: str = "contamination"
    value: float | None = None
    auto_adjust: bool = False
    min_threshold: float | None = None
    max_threshold: float | None = None

    def __post_init__(self) -> None:
        """Validate threshold configuration."""
        if self.method not in [
            "percentile",
            "fixed",
            "iqr",
            "mad",
            "adaptive",
            "contamination",
        ]:
            raise InvalidValueError(f"Invalid threshold method: {self.method}")

        if self.method == "percentile" and self.value is not None:
            if not (0 <= self.value <= 100):
                raise InvalidValueError(
                    f"Percentile value must be between 0 and 100, got {self.value}"
                )

        if self.min_threshold is not None and self.max_threshold is not None:
            if self.min_threshold >= self.max_threshold:
                raise InvalidValueError(
                    f"min_threshold ({self.min_threshold}) must be less than "
                    f"max_threshold ({self.max_threshold})"
                )
