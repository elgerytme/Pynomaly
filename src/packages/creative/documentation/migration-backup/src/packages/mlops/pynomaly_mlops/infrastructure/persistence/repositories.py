"""SQLAlchemy Repository Implementations

Concrete implementations of the domain repository contracts using SQLAlchemy.
"""

from typing import List, Optional, Dict, Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session, selectinload
from sqlalchemy import select, and_, or_, desc, asc, func
from sqlalchemy.exc import IntegrityError

from pynomaly_mlops.domain.entities.model import Model, ModelStatus, ModelType
from pynomaly_mlops.domain.repositories.model_repository import ModelRepository
from pynomaly_mlops.domain.value_objects.semantic_version import SemanticVersion
from pynomaly_mlops.domain.value_objects.model_metrics import ModelMetrics

from .models import ModelORM
from .mappers import ModelMapper


class SqlAlchemyModelRepository(ModelRepository):
    """SQLAlchemy implementation of ModelRepository."""
    
    def __init__(self, session: AsyncSession):
        """Initialize repository with database session.
        
        Args:
            session: SQLAlchemy async session
        """
        self.session = session
        self.mapper = ModelMapper()
    
    async def save(self, model: Model) -> Model:
        """Save or update a model.
        
        Args:
            model: Model to save
            
        Returns:
            Saved model with updated metadata
            
        Raises:
            IntegrityError: If model with same name/version already exists
        """
        try:
            # Check if model already exists
            existing = await self.get_by_id(model.id)
            
            if existing:
                # Update existing model
                model_orm = await self._get_orm_by_id(model.id)
                self.mapper.update_orm_from_domain(model_orm, model)
            else:
                # Create new model
                model_orm = self.mapper.domain_to_orm(model)
                self.session.add(model_orm)
            
            await self.session.commit()
            await self.session.refresh(model_orm)
            
            # Convert back to domain entity
            return self.mapper.orm_to_domain(model_orm)
            
        except IntegrityError as e:
            await self.session.rollback()
            if "uq_model_version" in str(e):
                raise ValueError(f"Model with name '{model.name}' and version '{model.version}' already exists")
            raise
    
    async def get_by_id(self, model_id: UUID) -> Optional[Model]:
        """Get model by ID.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Model if found, None otherwise
        """
        model_orm = await self._get_orm_by_id(model_id)
        return self.mapper.orm_to_domain(model_orm) if model_orm else None
    
    async def get_by_name_and_version(
        self, 
        name: str, 
        version: SemanticVersion
    ) -> Optional[Model]:
        """Get model by name and version.
        
        Args:
            name: Model name
            version: Model version
            
        Returns:
            Model if found, None otherwise
        """
        stmt = select(ModelORM).where(
            and_(
                ModelORM.name == name,
                ModelORM.version_major == version.major,
                ModelORM.version_minor == version.minor,
                ModelORM.version_patch == version.patch,
                ModelORM.version_prerelease == version.prerelease,
                ModelORM.version_build == version.build
            )
        )
        
        result = await self.session.execute(stmt)
        model_orm = result.scalar_one_or_none()
        
        return self.mapper.orm_to_domain(model_orm) if model_orm else None
    
    async def list_by_name(self, name: str) -> List[Model]:
        """List all versions of a model by name.
        
        Args:
            name: Model name
            
        Returns:
            List of models with the given name, sorted by version descending
        """
        stmt = (
            select(ModelORM)
            .where(ModelORM.name == name)
            .order_by(
                desc(ModelORM.version_major),
                desc(ModelORM.version_minor),
                desc(ModelORM.version_patch),
                ModelORM.version_prerelease.nulls_first(),
                desc(ModelORM.version_build)
            )
        )
        
        result = await self.session.execute(stmt)
        model_orms = result.scalars().all()
        
        return [self.mapper.orm_to_domain(orm) for orm in model_orms]
    
    async def list_by_status(self, status: ModelStatus) -> List[Model]:
        """List models by status.
        
        Args:
            status: Model status to filter by
            
        Returns:
            List of models with the given status
        """
        stmt = (
            select(ModelORM)
            .where(ModelORM.status == status.value)
            .order_by(desc(ModelORM.updated_at))
        )
        
        result = await self.session.execute(stmt)
        model_orms = result.scalars().all()
        
        return [self.mapper.orm_to_domain(orm) for orm in model_orms]
    
    async def list_by_type(self, model_type: ModelType) -> List[Model]:
        """List models by type.
        
        Args:
            model_type: Model type to filter by
            
        Returns:
            List of models with the given type
        """
        stmt = (
            select(ModelORM)
            .where(ModelORM.model_type == model_type.value)
            .order_by(desc(ModelORM.updated_at))
        )
        
        result = await self.session.execute(stmt)
        model_orms = result.scalars().all()
        
        return [self.mapper.orm_to_domain(orm) for orm in model_orms]
    
    async def search(
        self,
        name_pattern: Optional[str] = None,
        status: Optional[ModelStatus] = None,
        model_type: Optional[ModelType] = None,
        tags: Optional[List[str]] = None,
        created_by: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Model]:
        """Search models with filters.
        
        Args:
            name_pattern: Pattern to match model names (supports SQL LIKE patterns)
            status: Model status filter
            model_type: Model type filter
            tags: Tags to match (any tag matches)
            created_by: Creator filter
            limit: Maximum number of results
            offset: Number of results to skip
            
        Returns:
            List of matching models
        """
        stmt = select(ModelORM)
        
        # Build where conditions
        conditions = []
        
        if name_pattern:
            conditions.append(ModelORM.name.ilike(f"%{name_pattern}%"))
        
        if status:
            conditions.append(ModelORM.status == status.value)
        
        if model_type:
            conditions.append(ModelORM.model_type == model_type.value)
        
        if created_by:
            conditions.append(ModelORM.created_by == created_by)
        
        if tags:
            # Join with tags table for tag filtering
            from .models import model_tags_table
            tag_conditions = []
            for tag in tags:
                tag_conditions.append(model_tags_table.c.tag_name == tag)
            
            if tag_conditions:
                stmt = stmt.join(model_tags_table)
                conditions.append(or_(*tag_conditions))
        
        if conditions:
            stmt = stmt.where(and_(*conditions))
        
        # Add ordering and pagination
        stmt = stmt.order_by(desc(ModelORM.updated_at)).offset(offset).limit(limit)
        
        result = await self.session.execute(stmt)
        model_orms = result.scalars().all()
        
        return [self.mapper.orm_to_domain(orm) for orm in model_orms]
    
    async def get_latest_by_name(self, name: str) -> Optional[Model]:
        """Get the latest version of a model by name.
        
        Args:
            name: Model name
            
        Returns:
            Latest version of the model, None if not found
        """
        stmt = (
            select(ModelORM)
            .where(ModelORM.name == name)
            .order_by(
                desc(ModelORM.version_major),
                desc(ModelORM.version_minor),
                desc(ModelORM.version_patch),
                ModelORM.version_prerelease.nulls_first(),
                desc(ModelORM.version_build)
            )
            .limit(1)
        )
        
        result = await self.session.execute(stmt)
        model_orm = result.scalar_one_or_none()
        
        return self.mapper.orm_to_domain(model_orm) if model_orm else None
    
    async def get_production_models(self) -> List[Model]:
        """Get all models currently in production.
        
        Returns:
            List of production models
        """
        return await self.list_by_status(ModelStatus.PRODUCTION)
    
    async def delete(self, model_id: UUID) -> bool:
        """Delete a model.
        
        Args:
            model_id: Model identifier
            
        Returns:
            True if deleted, False if not found
        """
        model_orm = await self._get_orm_by_id(model_id)
        
        if not model_orm:
            return False
        
        await self.session.delete(model_orm)
        await self.session.commit()
        return True
    
    async def exists(self, name: str, version: SemanticVersion) -> bool:
        """Check if a model with given name and version exists.
        
        Args:
            name: Model name
            version: Model version
            
        Returns:
            True if model exists, False otherwise
        """
        model = await self.get_by_name_and_version(name, version)
        return model is not None
    
    async def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """Count models matching filters.
        
        Args:
            filters: Optional filters to apply
            
        Returns:
            Number of matching models
        """
        stmt = select(func.count(ModelORM.id))
        
        if filters:
            conditions = []
            
            if 'status' in filters:
                conditions.append(ModelORM.status == filters['status'])
            
            if 'model_type' in filters:
                conditions.append(ModelORM.model_type == filters['model_type'])
            
            if 'created_by' in filters:
                conditions.append(ModelORM.created_by == filters['created_by'])
            
            if conditions:
                stmt = stmt.where(and_(*conditions))
        
        result = await self.session.execute(stmt)
        return result.scalar()
    
    async def get_model_lineage(self, model_id: UUID) -> List[Model]:
        """Get the lineage (ancestors and descendants) of a model.
        
        Args:
            model_id: Model identifier
            
        Returns:
            List of models in the lineage chain
        """
        # Get the model itself
        base_model = await self.get_by_id(model_id)
        if not base_model:
            return []
        
        lineage = [base_model]
        
        # Get ancestors (parent models)
        current_id = model_id
        while True:
            stmt = select(ModelORM).where(ModelORM.id == current_id)
            result = await self.session.execute(stmt)
            current_orm = result.scalar_one_or_none()
            
            if not current_orm or not current_orm.parent_model_id:
                break
            
            parent_model = await self.get_by_id(current_orm.parent_model_id)
            if parent_model:
                lineage.insert(0, parent_model)
                current_id = current_orm.parent_model_id
            else:
                break
        
        # Get descendants (child models)
        descendants = await self._get_descendants(model_id)
        lineage.extend(descendants)
        
        return lineage
    
    async def _get_orm_by_id(self, model_id: UUID) -> Optional[ModelORM]:
        """Get ModelORM by ID with eager loading."""
        stmt = (
            select(ModelORM)
            .options(
                selectinload(ModelORM.parent_model),
                selectinload(ModelORM.child_models),
                selectinload(ModelORM.experiment),
                selectinload(ModelORM.deployments)
            )
            .where(ModelORM.id == model_id)
        )
        
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()
    
    async def _get_descendants(self, model_id: UUID) -> List[Model]:
        """Recursively get all descendant models."""
        descendants = []
        
        # Get direct children
        stmt = select(ModelORM).where(ModelORM.parent_model_id == model_id)
        result = await self.session.execute(stmt)
        child_orms = result.scalars().all()
        
        for child_orm in child_orms:
            child_model = self.mapper.orm_to_domain(child_orm)
            descendants.append(child_model)
            
            # Recursively get grandchildren
            grandchildren = await self._get_descendants(child_orm.id)
            descendants.extend(grandchildren)
        
        return descendants