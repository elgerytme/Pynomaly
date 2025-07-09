"""Base value object abstraction."""

from abc import ABC
from typing import Any

from pydantic import BaseModel


class BaseValueObject(BaseModel, ABC):
    """Base value object interface."""

    class Config:
        """Pydantic configuration."""
        allow_mutation = False

    def __hash__(self) -> int:
        """Hash based on all field values."""
        return hash(tuple(sorted(self.dict().items())))

    def __eq__(self, other: object) -> bool:
        """Equality based on all field values."""
        if not isinstance(other, self.__class__):
            return False
        return self.dict() == other.dict()

    def to_dict(self) -> dict[str, Any]:
        """Convert value object to dictionary."""
        return self.dict()
