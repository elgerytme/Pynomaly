"""
Domain Catalog Domain Entity

Represents a catalog that organizes and manages collections of domain entities
with advanced search, categorization, and relationship management capabilities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

from ..value_objects.entity_id import EntityId
from ..value_objects.version_number import VersionNumber
from ..value_objects.entity_metadata import EntityMetadata
from ..exceptions.entity_exceptions import (
    EntityValidationError, 
    InvalidEntityError,
    EntityNotFoundError,
    EntityConflictError
)
from .domain_entity import DomainEntity
from .entity_relationship import EntityRelationship


@dataclass
class CatalogIndex:
    """Represents a search index for the catalog."""
    
    name: str
    description: str
    index_type: str  # e.g., "text", "metadata", "category", "tag"
    indexed_fields: List[str]
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class CatalogStatistics:
    """Catalog statistics and metrics."""
    
    total_entities: int = 0
    total_relationships: int = 0
    categories_count: int = 0
    tags_count: int = 0
    active_entities: int = 0
    inactive_entities: int = 0
    last_calculated: datetime = field(default_factory=datetime.now)


@dataclass
class DomainCatalog:
    """
    Domain entity representing a catalog for organizing domain entities.
    
    Provides comprehensive entity management, search, categorization,
    and relationship tracking capabilities.
    """
    
    name: str
    description: str
    id: EntityId = field(default_factory=EntityId.generate)
    version: VersionNumber = field(default=VersionNumber("1.0.0"))
    metadata: EntityMetadata = field(default_factory=EntityMetadata.empty)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    # Catalog organization
    categories: Dict[str, str] = field(default_factory=dict)  # category_name -> description
    entity_ids: Set[EntityId] = field(default_factory=set)
    relationship_ids: Set[EntityId] = field(default_factory=set)
    
    # Search and indexing
    indexes: List[CatalogIndex] = field(default_factory=list)
    
    # Catalog properties
    max_entities: Optional[int] = None
    is_public: bool = False
    is_active: bool = True
    tags: Set[str] = field(default_factory=set)
    
    # Cached statistics
    _statistics: Optional[CatalogStatistics] = field(default=None, init=False, repr=False)
    
    def __post_init__(self) -> None:
        """Validate the catalog after initialization."""
        self._validate_catalog()
    
    def _validate_catalog(self) -> None:
        """Validate catalog properties and business rules."""
        if not self.name or not self.name.strip():
            raise InvalidEntityError("Catalog name cannot be empty")
        
        if not self.description or not self.description.strip():
            raise InvalidEntityError("Catalog description cannot be empty")
        
        if self.max_entities is not None and self.max_entities <= 0:
            raise EntityValidationError("Maximum entities must be positive")
    
    def add_entity(self, entity_id: EntityId) -> DomainCatalog:
        """Add an entity to the catalog."""
        if entity_id in self.entity_ids:
            raise EntityConflictError(f"Entity {entity_id} already exists in catalog")
        
        if self.max_entities and len(self.entity_ids) >= self.max_entities:
            raise EntityValidationError(f"Catalog has reached maximum capacity of {self.max_entities} entities")
        
        new_entity_ids = set(self.entity_ids)
        new_entity_ids.add(entity_id)
        
        return self._create_updated_copy(
            entity_ids=new_entity_ids,
            updated_at=datetime.now()
        )
    
    def remove_entity(self, entity_id: EntityId) -> DomainCatalog:
        """Remove an entity from the catalog."""
        if entity_id not in self.entity_ids:
            raise EntityNotFoundError(str(entity_id))
        
        new_entity_ids = set(self.entity_ids)
        new_entity_ids.remove(entity_id)
        
        return self._create_updated_copy(
            entity_ids=new_entity_ids,
            updated_at=datetime.now()
        )
    
    def add_relationship(self, relationship_id: EntityId) -> DomainCatalog:
        """Add a relationship to the catalog."""
        if relationship_id in self.relationship_ids:
            raise EntityConflictError(f"Relationship {relationship_id} already exists in catalog")
        
        new_relationship_ids = set(self.relationship_ids)
        new_relationship_ids.add(relationship_id)
        
        return self._create_updated_copy(
            relationship_ids=new_relationship_ids,
            updated_at=datetime.now()
        )
    
    def remove_relationship(self, relationship_id: EntityId) -> DomainCatalog:
        """Remove a relationship from the catalog."""
        if relationship_id not in self.relationship_ids:
            raise EntityNotFoundError(str(relationship_id))
        
        new_relationship_ids = set(self.relationship_ids)
        new_relationship_ids.remove(relationship_id)
        
        return self._create_updated_copy(
            relationship_ids=new_relationship_ids,
            updated_at=datetime.now()
        )
    
    def add_category(self, category_name: str, description: str) -> DomainCatalog:
        """Add a category to the catalog."""
        if category_name in self.categories:
            raise EntityConflictError(f"Category '{category_name}' already exists")
        
        new_categories = dict(self.categories)
        new_categories[category_name] = description
        
        return self._create_updated_copy(
            categories=new_categories,
            updated_at=datetime.now()
        )
    
    def remove_category(self, category_name: str) -> DomainCatalog:
        """Remove a category from the catalog."""
        if category_name not in self.categories:
            raise EntityNotFoundError(category_name)
        
        new_categories = dict(self.categories)
        del new_categories[category_name]
        
        return self._create_updated_copy(
            categories=new_categories,
            updated_at=datetime.now()
        )
    
    def update_category(self, category_name: str, description: str) -> DomainCatalog:
        """Update a category description."""
        if category_name not in self.categories:
            raise EntityNotFoundError(category_name)
        
        new_categories = dict(self.categories)
        new_categories[category_name] = description
        
        return self._create_updated_copy(
            categories=new_categories,
            updated_at=datetime.now()
        )
    
    def add_index(self, index: CatalogIndex) -> DomainCatalog:
        """Add a search index to the catalog."""
        # Check for duplicate index names
        existing_names = {idx.name for idx in self.indexes}
        if index.name in existing_names:
            raise EntityConflictError(f"Index '{index.name}' already exists")
        
        new_indexes = list(self.indexes)
        new_indexes.append(index)
        
        return self._create_updated_copy(
            indexes=new_indexes,
            updated_at=datetime.now()
        )
    
    def remove_index(self, index_name: str) -> DomainCatalog:
        """Remove a search index from the catalog."""
        new_indexes = [idx for idx in self.indexes if idx.name != index_name]
        
        if len(new_indexes) == len(self.indexes):
            raise EntityNotFoundError(index_name)
        
        return self._create_updated_copy(
            indexes=new_indexes,
            updated_at=datetime.now()
        )
    
    def find_entities_by_category(self, category: str) -> Set[EntityId]:
        """Find entity IDs belonging to a specific category."""
        if category not in self.categories:
            return set()
        
        # This is a placeholder - real implementation would query the entity storage
        # and filter by category
        return set()
    
    def find_entities_by_tag(self, tag: str) -> Set[EntityId]:
        """Find entity IDs that have a specific tag."""
        # This is a placeholder - real implementation would query the entity storage
        # and filter by tag
        return set()
    
    def find_related_entities(self, entity_id: EntityId, relationship_type: Optional[str] = None) -> Set[EntityId]:
        """Find entities related to the given entity."""
        if entity_id not in self.entity_ids:
            raise EntityNotFoundError(str(entity_id))
        
        # This is a placeholder - real implementation would traverse relationships
        return set()
    
    def search_entities(self, query: str, filters: Optional[Dict[str, Any]] = None) -> List[EntityId]:
        """
        Search for entities using text query and optional filters.
        
        Args:
            query: Text search query
            filters: Optional filters (category, tags, metadata, etc.)
            
        Returns:
            List of matching entity IDs
        """
        # This is a placeholder - real implementation would use search indexes
        # and apply filters to find matching entities
        return []
    
    def get_statistics(self, force_recalculate: bool = False) -> CatalogStatistics:
        """Get catalog statistics."""
        if self._statistics is None or force_recalculate:
            self._statistics = CatalogStatistics(
                total_entities=len(self.entity_ids),
                total_relationships=len(self.relationship_ids),
                categories_count=len(self.categories),
                tags_count=len(self.tags),
                active_entities=len(self.entity_ids),  # Placeholder
                inactive_entities=0,  # Placeholder
                last_calculated=datetime.now()
            )
        
        return self._statistics
    
    def validate_integrity(self) -> List[str]:
        """
        Validate the integrity of the catalog.
        
        Returns:
            List of integrity issues found
        """
        issues = []
        
        # Check for orphaned relationships
        # This is a placeholder - real implementation would validate that
        # all relationships reference entities that exist in the catalog
        
        return issues
    
    def add_tag(self, tag: str) -> DomainCatalog:
        """Add a tag to the catalog."""
        new_tags = set(self.tags)
        new_tags.add(tag.strip().lower())
        
        return self._create_updated_copy(
            tags=new_tags,
            updated_at=datetime.now()
        )
    
    def remove_tag(self, tag: str) -> DomainCatalog:
        """Remove a tag from the catalog."""
        new_tags = set(self.tags)
        new_tags.discard(tag.strip().lower())
        
        return self._create_updated_copy(
            tags=new_tags,
            updated_at=datetime.now()
        )
    
    def increment_version(self, version_type: str = "patch") -> DomainCatalog:
        """Create a new version of the catalog."""
        new_version = self.version.increment(version_type)
        
        return self._create_updated_copy(
            version=new_version,
            updated_at=datetime.now()
        )
    
    def deactivate(self) -> DomainCatalog:
        """Deactivate the catalog."""
        return self._create_updated_copy(
            is_active=False,
            updated_at=datetime.now()
        )
    
    def activate(self) -> DomainCatalog:
        """Activate the catalog."""
        return self._create_updated_copy(
            is_active=True,
            updated_at=datetime.now()
        )
    
    def _create_updated_copy(self, **kwargs) -> DomainCatalog:
        """Create an updated copy of the catalog with specified changes."""
        update_data = {
            'name': self.name,
            'description': self.description,
            'id': self.id,
            'version': self.version,
            'metadata': self.metadata,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'categories': self.categories,
            'entity_ids': self.entity_ids,
            'relationship_ids': self.relationship_ids,
            'indexes': self.indexes,
            'max_entities': self.max_entities,
            'is_public': self.is_public,
            'is_active': self.is_active,
            'tags': self.tags
        }
        update_data.update(kwargs)
        
        return DomainCatalog(**update_data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert catalog to dictionary representation."""
        return {
            'id': str(self.id),
            'name': self.name,
            'description': self.description,
            'version': str(self.version),
            'metadata': self.metadata.to_dict(),
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'categories': dict(self.categories),
            'entity_ids': [str(eid) for eid in self.entity_ids],
            'relationship_ids': [str(rid) for rid in self.relationship_ids],
            'indexes': [
                {
                    'name': idx.name,
                    'description': idx.description,
                    'index_type': idx.index_type,
                    'indexed_fields': idx.indexed_fields,
                    'is_active': idx.is_active,
                    'created_at': idx.created_at.isoformat(),
                    'last_updated': idx.last_updated.isoformat()
                }
                for idx in self.indexes
            ],
            'max_entities': self.max_entities,
            'is_public': self.is_public,
            'is_active': self.is_active,
            'tags': list(self.tags),
            'statistics': self.get_statistics().to_dict() if hasattr(self.get_statistics(), 'to_dict') else None
        }


# Add to_dict method to CatalogStatistics for completeness
def _catalog_statistics_to_dict(self) -> Dict[str, Any]:
    """Convert statistics to dictionary representation."""
    return {
        'total_entities': self.total_entities,
        'total_relationships': self.total_relationships,
        'categories_count': self.categories_count,
        'tags_count': self.tags_count,
        'active_entities': self.active_entities,
        'inactive_entities': self.inactive_entities,
        'last_calculated': self.last_calculated.isoformat()
    }

# Monkey patch the method onto CatalogStatistics
CatalogStatistics.to_dict = _catalog_statistics_to_dict