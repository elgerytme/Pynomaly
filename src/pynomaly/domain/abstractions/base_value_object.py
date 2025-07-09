"""Base value object class for domain value objects."""

from __future__ import annotations

from abc import ABC
from typing import Any


class BaseValueObject(ABC):
    """Base class for domain value objects."""
    
    def __eq__(self, other: object) -> bool:
        """Check equality by comparing all attributes."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__
    
    def __hash__(self) -> int:
        """Hash based on all attributes."""
        return hash(tuple(sorted(self.__dict__.items())))
    
    def __repr__(self) -> str:
        """String representation."""
        attrs = ", ".join(f"{k}={v}" for k, v in self.__dict__.items())
        return f"{self.__class__.__name__}({attrs})"
    
    def to_dict(self) -> dict[str, Any]:
        """Convert value object to dictionary."""
        return self.__dict__.copy()
    
    def validate(self) -> bool:
        """Validate value object."""
        return True
