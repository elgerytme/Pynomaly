"""Contamination rate value object."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from pynomaly.domain.exceptions import InvalidValueError


@dataclass(frozen=True)
class ContaminationRate:
    """Immutable value object representing contamination rate.
    
    Attributes:
        value: The contamination rate (0.0 to 1.0)
    """
    
    value: float
    
    # Class constants for common rates
    AUTO: ClassVar['ContaminationRate']
    LOW: ClassVar['ContaminationRate'] 
    MEDIUM: ClassVar['ContaminationRate']
    HIGH: ClassVar['ContaminationRate']
    
    def __post_init__(self) -> None:
        """Validate contamination rate after initialization."""
        if not isinstance(self.value, (int, float)):
            raise InvalidValueError(f"Contamination rate must be numeric, got {type(self.value)}")
        
        if not (0.0 <= self.value <= 0.5):
            raise InvalidValueError(f"Contamination rate must be between 0 and 0.5, got {self.value}")
    
    def is_valid(self) -> bool:
        """Check if the contamination rate is valid."""
        return isinstance(self.value, (int, float)) and 0.0 <= self.value <= 0.5
    
    def as_percentage(self) -> float:
        """Return contamination rate as a percentage (0-100)."""
        return self.value * 100.0
    
    def __str__(self) -> str:
        """String representation."""
        # Format as percentage, removing trailing zeros after decimal
        percentage = self.value * 100
        if percentage == int(percentage):
            return f"{int(percentage)}.0%"
        else:
            return f"{percentage:.1f}%"
    
    @classmethod
    def auto(cls) -> 'ContaminationRate':
        """Create auto contamination rate (typically 0.1)."""
        return cls(0.1)
    
    @classmethod
    def low(cls) -> 'ContaminationRate':
        """Create low contamination rate."""
        return cls(0.05)
    
    @classmethod
    def medium(cls) -> 'ContaminationRate':
        """Create medium contamination rate."""
        return cls(0.1)
    
    @classmethod
    def high(cls) -> 'ContaminationRate':
        """Create high contamination rate."""
        return cls(0.2)


# Initialize class constants
ContaminationRate.AUTO = ContaminationRate(0.1)
ContaminationRate.LOW = ContaminationRate(0.05)
ContaminationRate.MEDIUM = ContaminationRate(0.1)
ContaminationRate.HIGH = ContaminationRate(0.2)
