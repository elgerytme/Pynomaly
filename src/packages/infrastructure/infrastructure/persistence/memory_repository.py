"""In-memory repository implementations for testing and development."""

from typing import Any, TypeVar
from uuid import UUID

T = TypeVar("T")


class MemoryRepository:
    """Generic in-memory repository for testing - standardized async implementation."""

    def __init__(self) -> None:
        """Initialize empty repository."""
        self._storage: dict[str, Any] = {}

    async def save(self, entity: T) -> None:
        """Save entity to memory.

        Args:
            entity: Entity to save
        """
        if hasattr(entity, "id"):
            self._storage[str(entity.id)] = entity
        else:
            # Generate a simple key for entities without ID
            key = f"{type(entity).__name__}_{len(self._storage)}"
            self._storage[key] = entity

    async def find_by_id(self, entity_id: UUID | str) -> T | None:
        """Find entity by ID.

        Args:
            entity_id: ID of entity to find (UUID or string)

        Returns:
            Entity if found, None otherwise
        """
        return self._storage.get(str(entity_id))

    async def find_all(self) -> list[T]:
        """Find all entities.

        Returns:
            List of all entities
        """
        return list(self._storage.values())

    async def delete(self, entity_id: UUID | str) -> bool:
        """Delete entity by ID.

        Args:
            entity_id: ID of entity to delete

        Returns:
            True if entity was deleted, False if not found
        """
        entity_id_str = str(entity_id)
        if entity_id_str in self._storage:
            del self._storage[entity_id_str]
            return True
        return False

    async def exists(self, entity_id: UUID | str) -> bool:
        """Check if entity exists.

        Args:
            entity_id: ID to check

        Returns:
            True if entity exists, False otherwise
        """
        return str(entity_id) in self._storage

    async def count(self) -> int:
        """Count total entities.

        Returns:
            Total number of entities
        """
        return len(self._storage)

    def clear(self) -> None:
        """Clear all entities (sync method for testing convenience)."""
        self._storage.clear()
