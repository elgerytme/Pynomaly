"""
BaseEntity

Abstract base class for all domain entities.
"""

from abc import ABC, abstractmethod
from typing import Any

class BaseEntity(ABC):
    """
    Abstract base class for all domain entities.
    """
    def __init__(self, id: Any):
        self._id = id

    @property
    def id(self) -> Any:
        return self._id

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BaseEntity):
            return NotImplemented
        return self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)
