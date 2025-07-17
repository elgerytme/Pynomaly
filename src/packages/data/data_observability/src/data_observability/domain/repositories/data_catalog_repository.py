"""Repository interface for data catalog."""

from abc import ABC, abstractmethod
from typing import List, Optional

from ..entities.data_catalog import DataCatalog


class DataCatalogRepository(ABC):
    """Abstract repository for data catalog."""
    
    @abstractmethod
    async def save(self, catalog: DataCatalog) -> DataCatalog:
        """Save a data catalog entry."""
        pass
    
    @abstractmethod
    async def get_by_id(self, catalog_id: str) -> Optional[DataCatalog]:
        """Get catalog entry by ID."""
        pass
    
    @abstractmethod
    async def get_by_name(self, name: str) -> List[DataCatalog]:
        """Get catalog entries by name."""
        pass
    
    @abstractmethod
    async def get_all(self) -> List[DataCatalog]:
        """Get all catalog entries."""
        pass
    
    @abstractmethod
    async def delete(self, catalog_id: str) -> bool:
        """Delete a catalog entry."""
        pass
    
    @abstractmethod
    async def update(self, catalog: DataCatalog) -> DataCatalog:
        """Update a catalog entry."""
        pass