"""Contamination rate value object."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union


@dataclass(frozen=True)
class ContaminationRate:
    """Immutable value object representing the expected proportion of anomalies.
    
    Attributes:
        value: The contamination rate (between 0 and 1)
    """
    
    value: float
    
    def __post_init__(self) -> None:
        """Validate contamination rate after initialization."""
        if not isinstance(self.value, (int, float)):
            raise TypeError(f"Contamination rate must be numeric, got {type(self.value)}")
        
        if not 0 <= self.value <= 1:
            raise ValueError(
                f"Contamination rate must be between 0 and 1, got {self.value}"
            )
    
    @classmethod
    def from_percentage(cls, percentage: Union[int, float]) -> ContaminationRate:
        """Create from percentage value (0-100)."""
        if not 0 <= percentage <= 100:
            raise ValueError(
                f"Percentage must be between 0 and 100, got {percentage}"
            )
        return cls(value=percentage / 100)
    
    @classmethod
    def auto(cls) -> ContaminationRate:
        """Create auto-detection contamination rate (typically 0.1)."""
        return cls(value=0.1)
    
    @property
    def as_percentage(self) -> float:
        """Get contamination rate as percentage."""
        return self.value * 100
    
    @property
    def is_auto(self) -> bool:
        """Check if this is the default auto rate."""
        return self.value == 0.1
    
    def calculate_threshold_index(self, n_samples: int) -> int:
        """Calculate the index for threshold selection given number of samples."""
        if n_samples <= 0:
            raise ValueError(f"Number of samples must be positive, got {n_samples}")
        
        n_anomalies = int(n_samples * self.value)
        # Ensure at least 1 anomaly if contamination > 0
        if self.value > 0 and n_anomalies == 0:
            n_anomalies = 1
        # Ensure not all samples are anomalies
        if n_anomalies >= n_samples:
            n_anomalies = n_samples - 1
            
        return n_anomalies
    
    def __str__(self) -> str:
        """String representation."""
        return f"{self.as_percentage:.1f}%"
    
    def __repr__(self) -> str:
        """Developer representation."""
        return f"ContaminationRate(value={self.value})"