"""Enhanced base repository with generic patterns."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, Optional, TypeVar
from uuid import UUID

from .base_entity import BaseEntity
from .specification import Specification

T = TypeVar("T", bound=BaseEntity)


class BaseRepository(Generic[T], ABC):
    """Enhanced base repository with comprehensive persistence patterns.
    
    This class provides a foundation for all repository implementations
    with support for specifications, transactions, and advanced querying.
    """

    @abstractmethod
    def save(self, entity: T) -> T:
        """Save an entity to the repository.
        
        Args:
            entity: The entity to save
            
        Returns:
            The saved entity with updated metadata
        """
        ...

    @abstractmethod
    def find_by_id(self, entity_id: UUID) -> Optional[T]:
        """Find an entity by its ID.
        
        Args:
            entity_id: The UUID of the entity
            
        Returns:
            The entity if found, None otherwise
        """
        ...

    @abstractmethod
    def find_all(self, limit: Optional[int] = None, offset: int = 0) -> list[T]:
        """Find all entities in the repository.
        
        Args:
            limit: Maximum number of entities to return
            offset: Number of entities to skip
            
        Returns:
            List of entities
        """
        ...

    @abstractmethod
    def find_by_specification(self, specification: Specification[T]) -> list[T]:
        """Find entities matching a specification.
        
        Args:
            specification: The specification to match
            
        Returns:
            List of matching entities
        """
        ...

    @abstractmethod
    def delete(self, entity_id: UUID) -> bool:
        """Delete an entity by its ID.
        
        Args:
            entity_id: The UUID of the entity to delete
            
        Returns:
            True if deleted, False if not found
        """
        ...

    @abstractmethod
    def delete_by_specification(self, specification: Specification[T]) -> int:
        """Delete entities matching a specification.
        
        Args:
            specification: The specification to match
            
        Returns:
            Number of entities deleted
        """
        ...

    @abstractmethod
    def exists(self, entity_id: UUID) -> bool:
        """Check if an entity exists.
        
        Args:
            entity_id: The UUID to check
            
        Returns:
            True if exists, False otherwise
        """
        ...

    @abstractmethod
    def count(self, specification: Optional[Specification[T]] = None) -> int:
        """Count entities, optionally matching a specification.
        
        Args:
            specification: Optional specification to filter by
            
        Returns:
            Number of entities
        """
        ...

    @abstractmethod
    def update(self, entity: T) -> T:
        """Update an existing entity.
        
        Args:
            entity: The entity to update
            
        Returns:
            The updated entity
        """
        ...

    @abstractmethod
    def save_all(self, entities: list[T]) -> list[T]:
        """Save multiple entities.
        
        Args:
            entities: List of entities to save
            
        Returns:
            List of saved entities
        """
        ...

    @abstractmethod
    def delete_all(self, entities: list[T]) -> int:
        """Delete multiple entities.
        
        Args:
            entities: List of entities to delete
            
        Returns:
            Number of entities deleted
        """
        ...

    # Optional advanced methods (can be implemented by subclasses)
    
    def find_one_by_specification(self, specification: Specification[T]) -> Optional[T]:
        """Find a single entity matching a specification.
        
        Args:
            specification: The specification to match
            
        Returns:
            The first matching entity, or None if not found
        """
        results = self.find_by_specification(specification)
        return results[0] if results else None

    def exists_by_specification(self, specification: Specification[T]) -> bool:
        """Check if any entity matches a specification.
        
        Args:
            specification: The specification to check
            
        Returns:
            True if any entity matches
        """
        return self.count(specification) > 0

    def find_page(
        self, 
        page_number: int, 
        page_size: int, 
        specification: Optional[Specification[T]] = None
    ) -> dict[str, Any]:
        """Find entities with pagination.
        
        Args:
            page_number: Page number (1-based)
            page_size: Number of entities per page
            specification: Optional specification to filter by
            
        Returns:
            Dictionary containing items, total count, and pagination info
        """
        offset = (page_number - 1) * page_size
        
        if specification:
            items = self.find_by_specification(specification)
            total_count = len(items)
            items = items[offset:offset + page_size]
        else:
            items = self.find_all(limit=page_size, offset=offset)
            total_count = self.count()
        
        total_pages = (total_count + page_size - 1) // page_size
        
        return {
            "items": items,
            "page_number": page_number,
            "page_size": page_size,
            "total_count": total_count,
            "total_pages": total_pages,
            "has_next": page_number < total_pages,
            "has_previous": page_number > 1,
        }

    def get_entity_type(self) -> str:
        """Get the entity type handled by this repository.
        
        Returns:
            Entity type name
        """
        # Get the type parameter from the generic
        return getattr(self.__class__, '__orig_bases__', [None])[0]


class TransactionalRepository(BaseRepository[T], ABC):
    """Repository with transaction support."""

    @abstractmethod
    def begin_transaction(self) -> None:
        """Begin a new transaction."""
        ...

    @abstractmethod
    def commit_transaction(self) -> None:
        """Commit the current transaction."""
        ...

    @abstractmethod
    def rollback_transaction(self) -> None:
        """Rollback the current transaction."""
        ...

    @abstractmethod
    def is_in_transaction(self) -> bool:
        """Check if currently in a transaction.
        
        Returns:
            True if in transaction
        """
        ...

    def with_transaction(self, func: callable) -> Any:
        """Execute a function within a transaction.
        
        Args:
            func: Function to execute
            
        Returns:
            Function result
        """
        self.begin_transaction()
        try:
            result = func()
            self.commit_transaction()
            return result
        except Exception:
            self.rollback_transaction()
            raise


class ReadOnlyRepository(Generic[T], ABC):
    """Read-only repository interface."""

    @abstractmethod
    def find_by_id(self, entity_id: UUID) -> Optional[T]:
        """Find an entity by its ID."""
        ...

    @abstractmethod
    def find_all(self, limit: Optional[int] = None, offset: int = 0) -> list[T]:
        """Find all entities."""
        ...

    @abstractmethod
    def find_by_specification(self, specification: Specification[T]) -> list[T]:
        """Find entities matching a specification."""
        ...

    @abstractmethod
    def exists(self, entity_id: UUID) -> bool:
        """Check if an entity exists."""
        ...

    @abstractmethod
    def count(self, specification: Optional[Specification[T]] = None) -> int:
        """Count entities."""
        ...


class CacheableRepository(BaseRepository[T], ABC):
    """Repository with caching support."""

    @abstractmethod
    def invalidate_cache(self, entity_id: UUID) -> None:
        """Invalidate cache for a specific entity.
        
        Args:
            entity_id: ID of the entity to invalidate
        """
        ...

    @abstractmethod
    def clear_cache(self) -> None:
        """Clear all cache entries."""
        ...

    @abstractmethod
    def warm_cache(self, entity_ids: list[UUID]) -> None:
        """Warm cache with specific entities.
        
        Args:
            entity_ids: List of entity IDs to cache
        """
        ...

    @abstractmethod
    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        ...


class EventSourcedRepository(BaseRepository[T], ABC):
    """Repository for event-sourced entities."""

    @abstractmethod
    def save_events(self, aggregate_id: UUID, events: list[Any]) -> None:
        """Save events for an aggregate.
        
        Args:
            aggregate_id: ID of the aggregate
            events: List of domain events
        """
        ...

    @abstractmethod
    def load_events(self, aggregate_id: UUID) -> list[Any]:
        """Load events for an aggregate.
        
        Args:
            aggregate_id: ID of the aggregate
            
        Returns:
            List of domain events
        """
        ...

    @abstractmethod
    def get_event_stream(self, aggregate_id: UUID, from_version: int = 0) -> list[Any]:
        """Get event stream for an aggregate.
        
        Args:
            aggregate_id: ID of the aggregate
            from_version: Starting version
            
        Returns:
            List of domain events from the specified version
        """
        ...


class RepositoryFactory(ABC):
    """Factory for creating repository instances."""

    @abstractmethod
    def create_repository(self, entity_type: type[T]) -> BaseRepository[T]:
        """Create a repository for the given entity type.
        
        Args:
            entity_type: The entity type
            
        Returns:
            Repository instance for the entity type
        """
        ...

    @abstractmethod
    def get_supported_entity_types(self) -> list[type]:
        """Get list of supported entity types.
        
        Returns:
            List of supported entity types
        """
        ...


class UnitOfWork(ABC):
    """Unit of work pattern for managing transactions across repositories."""

    @abstractmethod
    def register_new(self, entity: BaseEntity) -> None:
        """Register a new entity.
        
        Args:
            entity: The entity to register as new
        """
        ...

    @abstractmethod
    def register_dirty(self, entity: BaseEntity) -> None:
        """Register a modified entity.
        
        Args:
            entity: The entity to register as modified
        """
        ...

    @abstractmethod
    def register_removed(self, entity: BaseEntity) -> None:
        """Register an entity for removal.
        
        Args:
            entity: The entity to register for removal
        """
        ...

    @abstractmethod
    def commit(self) -> None:
        """Commit all changes."""
        ...

    @abstractmethod
    def rollback(self) -> None:
        """Rollback all changes."""
        ...

    @abstractmethod
    def is_dirty(self) -> bool:
        """Check if there are uncommitted changes.
        
        Returns:
            True if there are uncommitted changes
        """
        ...
