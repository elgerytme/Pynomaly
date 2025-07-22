"""Repository interface for data catalog."""

from abc import ABC, abstractmethod
from typing import List, Optional
from uuid import UUID

from ..entities.data_catalog import DataCatalogEntry


class DataCatalogRepository(ABC):
    """Abstract repository for data catalog."""
    
    @abstractmethod
    async def save(self, catalog: DataCatalogEntry) -> DataCatalogEntry:
        """Save a data catalog entry."""
        pass
    
    @abstractmethod
    async def get_by_id(self, catalog_id: UUID) -> Optional[DataCatalogEntry]:
        """Get catalog entry by ID."""
        pass
    
    @abstractmethod
    async def get_by_name(self, name: str) -> List[DataCatalogEntry]:
        """Get catalog entries by name."""
        pass
    
    @abstractmethod
    async def get_all(self) -> List[DataCatalogEntry]:
        """Get all catalog entries."""
        pass
    
    @abstractmethod
    async def delete(self, catalog_id: UUID) -> bool:
        """Delete a catalog entry."""
        pass
    
    @abstractmethod
    async def update(self, catalog: DataCatalogEntry) -> DataCatalogEntry:
        """Update a catalog entry."""
        pass