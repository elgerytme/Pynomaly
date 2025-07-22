"""In-memory repository implementation for data catalog."""

from typing import Dict, List, Optional
from uuid import UUID

from ...domain.entities.data_catalog import DataCatalogEntry
from ...domain.repositories.data_catalog_repository import DataCatalogRepository


class InMemoryDataCatalogRepository(DataCatalogRepository):
    """In-memory implementation of data catalog repository."""
    
    def __init__(self):
        self._catalogs: Dict[UUID, DataCatalogEntry] = {}
    
    async def save(self, catalog: DataCatalogEntry) -> DataCatalogEntry:
        """Save a data catalog entry."""
        self._catalogs[catalog.id] = catalog
        return catalog
    
    async def get_by_id(self, catalog_id: UUID) -> Optional[DataCatalogEntry]:
        """Get catalog entry by ID."""
        return self._catalogs.get(catalog_id)
    
    async def get_by_name(self, name: str) -> List[DataCatalogEntry]:
        """Get catalog entries by name."""
        return [
            catalog for catalog in self._catalogs.values()
            if catalog.name == name
        ]
    
    async def get_all(self) -> List[DataCatalogEntry]:
        """Get all catalog entries."""
        return list(self._catalogs.values())
    
    async def delete(self, catalog_id: UUID) -> bool:
        """Delete a catalog entry."""
        if catalog_id in self._catalogs:
            del self._catalogs[catalog_id]
            return True
        return False
    
    async def update(self, catalog: DataCatalogEntry) -> DataCatalogEntry:
        """Update a catalog entry."""
        if catalog.id in self._catalogs:
            self._catalogs[catalog.id] = catalog
            return catalog
        else:
            raise ValueError(f"Catalog with ID {catalog.id} not found")
    
    def clear(self) -> None:
        """Clear all catalog entries (for testing)."""
        self._catalogs.clear()
    
    def count(self) -> int:
        """Get count of catalog entries."""
        return len(self._catalogs)