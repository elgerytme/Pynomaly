"""Detector repository implementation."""

from typing import List, Optional
from uuid import UUID

from pynomaly.domain.entities import Detector
from pynomaly.shared.protocols import RepositoryProtocol


class DetectorRepository(RepositoryProtocol[Detector]):
    """Repository for detector entities."""
    
    def __init__(self):
        """Initialize repository."""
        self._items: dict[UUID, Detector] = {}
    
    async def save(self, entity: Detector) -> Detector:
        """Save detector."""
        self._items[entity.id] = entity
        return entity
    
    async def get_by_id(self, entity_id: UUID) -> Optional[Detector]:
        """Get detector by ID."""
        return self._items.get(entity_id)
    
    async def get_all(self) -> List[Detector]:
        """Get all detectors."""
        return list(self._items.values())
    
    async def delete(self, entity_id: UUID) -> bool:
        """Delete detector."""
        if entity_id in self._items:
            del self._items[entity_id]
            return True
        return False
    
    async def exists(self, entity_id: UUID) -> bool:
        """Check if detector exists."""
        return entity_id in self._items
    
    async def count(self) -> int:
        """Count detectors."""
        return len(self._items)
    
    async def update(self, entity: Detector) -> Detector:
        """Update detector."""
        self._items[entity.id] = entity
        return entity