"""Base repository abstraction."""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar
from uuid import UUID

T = TypeVar("T")


class BaseRepository(Generic[T], ABC):
    """Base repository interface."""

    @abstractmethod
    def save(self, entity: T) -> T:
        """Save an entity."""
        pass

    @abstractmethod
    def find_by_id(self, entity_id: UUID) -> T | None:
        """Find entity by ID."""
        pass

    @abstractmethod
    def find_all(self) -> list[T]:
        """Find all entities."""
        pass

    @abstractmethod
    def delete(self, entity_id: UUID) -> bool:
        """Delete entity by ID."""
        pass

    @abstractmethod
    def exists(self, entity_id: UUID) -> bool:
        """Check if entity exists."""
        pass

    @abstractmethod
    def count(self) -> int:
        """Count total entities."""
        pass
