"""Base repository abstraction."""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Optional, List
from uuid import UUID

T = TypeVar("T")


class BaseRepository(Generic[T], ABC):
    """Base repository interface."""

    @abstractmethod
    async def save(self, entity: T) -> T:
        """Save an entity."""
        pass

    @abstractmethod
    async def get_by_id(self, entity_id: UUID) -> Optional[T]:
        """Get an entity by ID."""
        pass

    @abstractmethod
    async def delete(self, entity_id: UUID) -> bool:
        """Delete an entity by ID."""
        pass

    @abstractmethod
    async def list_all(self) -> List[T]:
        """List all entities."""
        pass

    @abstractmethod
    async def exists(self, entity_id: UUID) -> bool:
        """Check if an entity exists."""
        pass
