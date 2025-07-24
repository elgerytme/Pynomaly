"""PostgreSQL repository implementation for data catalog."""

from datetime import datetime
from typing import Dict, List, Optional
from uuid import UUID, uuid4

from sqlalchemy import and_, func, select, delete, update
from sqlalchemy.ext.asyncio import AsyncSession

from ...domain.entities.data_catalog import DataCatalogEntry, DataAssetType, DataFormat, AccessLevel, DataClassification, DataQuality, DataSchema
from ...domain.repositories.data_catalog_repository import DataCatalogRepository
from ...infrastructure.persistence.models import DataAssetModel


class PostgresDataCatalogRepository(DataCatalogRepository):
    """PostgreSQL implementation of data catalog repository."""
    
    def __init__(self, session: AsyncSession) -> None:
        """Initialize with database session."""
        self._session = session
    
    def _to_model(self, entity: DataCatalogEntry) -> DataAssetModel:
        """Convert DataCatalogEntry entity to DataAssetModel."""
        return DataAssetModel(
            id=entity.id,
            name=entity.name,
            asset_type=entity.type.value,
            description=entity.description,
            location=entity.location,
            format=entity.format.value,
            schema_info=entity.schema_.dict() if entity.schema_ else None,
            metadata=entity.properties,
            tags=list(entity.tags),
            business_terms=list(entity.business_terms),
            owner=entity.owner,
            steward=entity.steward,
            classification=entity.classification.value,
            quality_score=entity.quality_score,
            freshness_score=entity.get_freshness_score(),
            completeness_score=entity.quality_score, # Assuming quality_score also represents completeness for now
            created_at=entity.created_at,
            updated_at=entity.updated_at,
            last_accessed_at=entity.last_accessed,
            is_active=True # Assuming all registered assets are active
        )

    def _to_entity(self, model: DataAssetModel) -> DataCatalogEntry:
        """Convert DataAssetModel to DataCatalogEntry entity."""
        return DataCatalogEntry(
            id=model.id,
            name=model.name,
            type=DataAssetType(model.asset_type),
            description=model.description,
            location=model.location,
            format=DataFormat(model.format),
            schema_=DataSchema(**model.schema_info) if model.schema_info else None,
            properties=model.metadata,
            tags=set(model.tags),
            business_terms=set(model.business_terms),
            owner=model.owner,
            steward=model.steward,
            classification=DataClassification(model.classification),
            quality_score=model.quality_score,
            created_at=model.created_at,
            updated_at=model.updated_at,
            last_accessed=model.last_accessed_at
        )

    async def save(self, catalog: DataCatalogEntry) -> DataCatalogEntry:
        """Save a data catalog entry."""
        model = self._to_model(catalog)
        self._session.add(model)
        await self._session.flush()  # To get the ID if it's a new entry
        await self._session.commit()
        return self._to_entity(model)
    
    async def get_by_id(self, catalog_id: UUID) -> Optional[DataCatalogEntry]:
        """Get catalog entry by ID."""
        result = await self._session.execute(
            select(DataAssetModel).filter(DataAssetModel.id == catalog_id)
        )
        model = result.scalar_one_or_none()
        return self._to_entity(model) if model else None
    
    async def get_by_name(self, name: str) -> List[DataCatalogEntry]:
        """Get catalog entries by name."""
        result = await self._session.execute(
            select(DataAssetModel).filter(DataAssetModel.name == name)
        )
        return [self._to_entity(model) for model in result.scalars().all()]
    
    async def get_all(self) -> List[DataCatalogEntry]:
        """Get all catalog entries."""
        result = await self._session.execute(
            select(DataAssetModel)
        )
        return [self._to_entity(model) for model in result.scalars().all()]
    
    async def delete(self, catalog_id: UUID) -> bool:
        """Delete a catalog entry."""
        result = await self._session.execute(
            delete(DataAssetModel).filter(DataAssetModel.id == catalog_id)
        )
        await self._session.commit()
        return result.rowcount > 0
    
    async def update(self, catalog: DataCatalogEntry) -> DataCatalogEntry:
        """Update a catalog entry."""
        model_data = self._to_model(catalog).dict(exclude_unset=True)
        # Exclude fields that should not be updated directly or are managed by DB
        model_data.pop("id", None)
        model_data.pop("created_at", None)
        model_data.pop("updated_at", None) # Let func.now() handle this
        
        result = await self._session.execute(
            update(DataAssetModel)
            .filter(DataAssetModel.id == catalog.id)
            .values(**model_data)
        )
        await self._session.commit()
        
        if result.rowcount > 0:
            return await self.get_by_id(catalog.id) # Fetch the updated entity
        else:
            raise ValueError(f"Catalog with ID {catalog.id} not found for update")
