"""Shared domain abstractions and base classes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, TypeVar, Generic
from uuid import UUID, uuid4


@dataclass
class BaseEntity:
    """Base class for all domain entities."""
    
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    version: int = 1
    
    def __post_init__(self) -> None:
        """Validate entity after initialization."""
        if self.version < 1:
            raise ValueError("Version must be at least 1")
    
    def update_version(self) -> None:
        """Update entity version and timestamp."""
        self.version += 1
        self.updated_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entity to dictionary."""
        return {
            "id": str(self.id),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "version": self.version
        }
    
    def __eq__(self, other: object) -> bool:
        """Check equality based on ID."""
        if not isinstance(other, BaseEntity):
            return False
        return self.id == other.id
    
    def __hash__(self) -> int:
        """Hash based on ID."""
        return hash(self.id)


@dataclass(frozen=True)
class ValueObject:
    """Base class for value objects."""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert value object to dictionary."""
        return {}


@dataclass
class DomainEvent:
    """Base class for domain events."""
    
    event_id: UUID = field(default_factory=uuid4)
    occurred_at: datetime = field(default_factory=datetime.utcnow)
    event_type: str = ""
    aggregate_id: UUID = field(default_factory=uuid4)
    aggregate_type: str = ""
    event_version: int = 1
    correlation_id: Optional[UUID] = None
    causation_id: Optional[UUID] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validate event after initialization."""
        if not self.event_type:
            raise ValueError("Event type cannot be empty")
        
        if not self.aggregate_type:
            raise ValueError("Aggregate type cannot be empty")
        
        if self.event_version < 1:
            raise ValueError("Event version must be at least 1")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return {
            "event_id": str(self.event_id),
            "occurred_at": self.occurred_at.isoformat(),
            "event_type": self.event_type,
            "aggregate_id": str(self.aggregate_id),
            "aggregate_type": self.aggregate_type,
            "event_version": self.event_version,
            "correlation_id": str(self.correlation_id) if self.correlation_id else None,
            "causation_id": str(self.causation_id) if self.causation_id else None,
            "metadata": self.metadata
        }


# Generic repository interface
T = TypeVar('T', bound=BaseEntity)

class RepositoryInterface(ABC, Generic[T]):
    """Generic repository interface."""
    
    @abstractmethod
    async def find_by_id(self, entity_id: UUID) -> Optional[T]:
        """Find entity by ID."""
        pass
    
    @abstractmethod
    async def find_all(self) -> List[T]:
        """Find all entities."""
        pass
    
    @abstractmethod
    async def save(self, entity: T) -> T:
        """Save entity."""
        pass
    
    @abstractmethod
    async def delete(self, entity_id: UUID) -> bool:
        """Delete entity by ID."""
        pass
    
    @abstractmethod
    async def exists(self, entity_id: UUID) -> bool:
        """Check if entity exists."""
        pass


# Generic service interface
class ServiceInterface(ABC):
    """Base interface for domain services."""
    pass


# Generic specification pattern
class Specification(ABC, Generic[T]):
    """Specification pattern for complex queries."""
    
    @abstractmethod
    def is_satisfied_by(self, candidate: T) -> bool:
        """Check if candidate satisfies the specification."""
        pass
    
    def and_specification(self, other: Specification[T]) -> AndSpecification[T]:
        """Combine with AND logic."""
        return AndSpecification(self, other)
    
    def or_specification(self, other: Specification[T]) -> OrSpecification[T]:
        """Combine with OR logic."""
        return OrSpecification(self, other)
    
    def not_specification(self) -> NotSpecification[T]:
        """Negate specification."""
        return NotSpecification(self)


class AndSpecification(Specification[T]):
    """AND combination of specifications."""
    
    def __init__(self, left: Specification[T], right: Specification[T]):
        self.left = left
        self.right = right
    
    def is_satisfied_by(self, candidate: T) -> bool:
        return self.left.is_satisfied_by(candidate) and self.right.is_satisfied_by(candidate)


class OrSpecification(Specification[T]):
    """OR combination of specifications."""
    
    def __init__(self, left: Specification[T], right: Specification[T]):
        self.left = left
        self.right = right
    
    def is_satisfied_by(self, candidate: T) -> bool:
        return self.left.is_satisfied_by(candidate) or self.right.is_satisfied_by(candidate)


class NotSpecification(Specification[T]):
    """NOT negation of specification."""
    
    def __init__(self, specification: Specification[T]):
        self.specification = specification
    
    def is_satisfied_by(self, candidate: T) -> bool:
        return not self.specification.is_satisfied_by(candidate)


# Unit of Work pattern
class UnitOfWorkInterface(ABC):
    """Unit of work interface for transaction management."""
    
    @abstractmethod
    async def __aenter__(self):
        """Enter async context."""
        pass
    
    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context."""
        pass
    
    @abstractmethod
    async def commit(self) -> None:
        """Commit transaction."""
        pass
    
    @abstractmethod
    async def rollback(self) -> None:
        """Rollback transaction."""
        pass


# Factory pattern
class FactoryInterface(ABC, Generic[T]):
    """Factory interface for creating entities."""
    
    @abstractmethod
    def create(self, **kwargs) -> T:
        """Create new entity."""
        pass


# Builder pattern
class BuilderInterface(ABC, Generic[T]):
    """Builder interface for complex entity construction."""
    
    @abstractmethod
    def build(self) -> T:
        """Build the entity."""
        pass
    
    @abstractmethod
    def reset(self) -> BuilderInterface[T]:
        """Reset the builder."""
        pass