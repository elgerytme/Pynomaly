"""Threshold configuration value object."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from pynomaly.domain.exceptions import ValidationError


@dataclass(frozen=True)
class ThresholdConfig:
    """Configuration for anomaly threshold calculation."""
    
    method: str = "percentile"
    value: float = 0.95
    auto_adjust: bool = False
    min_threshold: Optional[float] = None
    max_threshold: Optional[float] = None
    
    def __post_init__(self) -> None:
        """Validate threshold configuration."""
        if self.method not in ["percentile", "fixed", "iqr", "mad", "adaptive"]:
            raise ValidationError(
                f"Invalid threshold method: {self.method}",
                field="method",
                value=self.method
            )
        
        if self.method == "percentile" and not 0 < self.value <= 1:
            raise ValidationError(
                f"Percentile value must be between 0 and 1, got {self.value}",
                field="value",
                value=self.value
            )
        
        if self.min_threshold is not None and self.max_threshold is not None:
            if self.min_threshold >= self.max_threshold:
                raise ValidationError(
                    f"min_threshold ({self.min_threshold}) must be less than "
                    f"max_threshold ({self.max_threshold})",
                    field="threshold_bounds"
                )