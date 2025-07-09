"""In-memory repository implementations for testing and development."""

from typing import Any, Dict, List, Optional, TypeVar
from uuid import UUID

T = TypeVar("T")


class MemoryRepository:
    """Generic in-memory repository for testing."""
    
    def __init__(self):
        """Initialize empty repository."""
        self._storage: Dict[str, Any] = {}
        
    async def save(self, entity: T) -> None:
        """Save entity to memory.
        
        Args:
            entity: Entity to save
        """
        if hasattr(entity, 'id'):
            self._storage[str(entity.id)] = entity
        else:
            # Generate a simple key for entities without ID
            key = f"{type(entity).__name__}_{len(self._storage)}"
            self._storage[key] = entity
            
    async def find_by_id(self, entity_id: str) -> Optional[T]:
        """Find entity by ID.
        
        Args:
            entity_id: ID of entity to find
            
        Returns:
            Entity if found, None otherwise
        """
        return self._storage.get(entity_id)
        
    async def list_all(self) -> List[T]:
        """List all entities.
        
        Returns:
            List of all entities
        """
        return list(self._storage.values())
        
    async def delete(self, entity_id: str) -> None:
        """Delete entity by ID.
        
        Args:
            entity_id: ID of entity to delete
        """
        if entity_id in self._storage:
            del self._storage[entity_id]
            
    async def exists(self, entity_id: str) -> bool:
        """Check if entity exists.
        
        Args:
            entity_id: ID to check
            
        Returns:
            True if entity exists, False otherwise
        """
        return entity_id in self._storage
        
    async def count(self) -> int:
        """Count total entities.
        
        Returns:
            Total number of entities
        """
        return len(self._storage)
        
    def clear(self) -> None:
        """Clear all entities."""
        self._storage.clear()