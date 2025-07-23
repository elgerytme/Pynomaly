"""
Base classes for Domain-Driven Design patterns.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, TypeVar, Generic
from uuid import UUID, uuid4

from .value_objects import Identifier, Timestamp


T = TypeVar('T')


class Entity(ABC):
    """
    Base class for domain entities.
    
    Entities are objects with identity that can change over time.
    Two entities are equal if they have the same identity.
    """
    
    def __init__(self, entity_id: Identifier) -> None:
        self._id = entity_id
        self._created_at = Timestamp.now()
        self._updated_at = Timestamp.now()
        self._version = 1
    
    @property
    def id(self) -> Identifier:
        """Get the entity's unique identifier."""
        return self._id
    
    @property
    def created_at(self) -> Timestamp:
        """Get when the entity was created."""
        return self._created_at
    
    @property
    def updated_at(self) -> Timestamp:
        """Get when the entity was last updated."""
        return self._updated_at
    
    @property
    def version(self) -> int:
        """Get the entity's version for concurrency control."""
        return self._version
    
    def mark_updated(self) -> None:
        """Mark the entity as updated."""
        self._updated_at = Timestamp.now()
        self._version += 1
    
    def __eq__(self, other: Any) -> bool:
        """Entities are equal if they have the same type and ID."""
        if not isinstance(other, Entity):
            return False
        return type(self) == type(other) and self._id == other._id
    
    def __hash__(self) -> int:
        """Hash based on type and ID."""
        return hash((type(self), self._id))
    
    def __repr__(self) -> str:
        return f"{type(self).__name__}(id={self._id})"


class ValueObject(ABC):
    """
    Base class for value objects.
    
    Value objects are immutable objects that are equal when their
    attributes are equal. They have no identity.
    """
    
    def __eq__(self, other: Any) -> bool:
        """Value objects are equal when all attributes are equal."""
        if not isinstance(other, type(self)):
            return False
        return self.__dict__ == other.__dict__
    
    def __hash__(self) -> int:
        """Hash based on all attributes."""
        return hash(tuple(sorted(self.__dict__.items())))
    
    def __repr__(self) -> str:
        attrs = ', '.join(f'{k}={v!r}' for k, v in self.__dict__.items())
        return f"{type(self).__name__}({attrs})"


@dataclass(frozen=True)
class DomainEvent:
    """
    Base class for domain events.
    
    Domain events represent something interesting that happened
    in the domain that other parts of the system might want to know about.
    """
    
    event_id: Identifier
    aggregate_id: Identifier
    event_type: str
    occurred_at: Timestamp
    version: int
    metadata: Dict[str, Any]
    
    @classmethod
    def create(
        cls,
        aggregate_id: Identifier,
        event_type: str,
        version: int,
        metadata: Dict[str, Any] | None = None,
    ) -> DomainEvent:
        """Create a new domain event."""
        return cls(
            event_id=Identifier.generate("event"),
            aggregate_id=aggregate_id,
            event_type=event_type,
            occurred_at=Timestamp.now(),
            version=version,
            metadata=metadata or {},
        )
    
    def get_event_name(self) -> str:
        """Get a human-readable event name."""
        return self.event_type.replace('_', ' ').title()
    
    def is_for_aggregate(self, aggregate_id: Identifier) -> bool:
        """Check if this event is for a specific aggregate."""
        return self.aggregate_id == aggregate_id
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization."""
        return {
            'event_id': str(self.event_id),
            'aggregate_id': str(self.aggregate_id),
            'event_type': self.event_type,
            'occurred_at': self.occurred_at.to_iso(),
            'version': self.version,
            'metadata': self.metadata,
        }


class AggregateRoot(Entity):
    """
    Base class for aggregate roots.
    
    Aggregate roots are entities that serve as the entry point
    to an aggregate and maintain invariants across the aggregate.
    """
    
    def __init__(self, entity_id: Identifier) -> None:
        super().__init__(entity_id)
        self._domain_events: List[DomainEvent] = []
    
    def add_domain_event(self, event: DomainEvent) -> None:
        """Add a domain event to be published."""
        self._domain_events.append(event)
        self.mark_updated()
    
    def get_domain_events(self) -> List[DomainEvent]:
        """Get all unpublished domain events."""
        return self._domain_events.copy()
    
    def clear_domain_events(self) -> None:
        """Clear domain events after they've been published."""
        self._domain_events.clear()
    
    def has_domain_events(self) -> bool:
        """Check if there are unpublished domain events."""
        return bool(self._domain_events)


class UseCase(ABC, Generic[T]):
    """
    Base class for application use cases.
    
    Use cases represent the application's business logic and coordinate
    between domain services, repositories, and other infrastructure.
    """
    
    @abstractmethod
    def execute(self, request: T) -> Any:
        """Execute the use case with the given request."""
        pass
    
    def validate_request(self, request: T) -> None:
        """Validate the use case request. Override to add validation."""
        if request is None:
            raise ValueError("Request cannot be None")


class Repository(ABC, Generic[T]):
    """
    Base class for repositories.
    
    Repositories provide a collection-like interface for accessing
    and persisting aggregates.
    """
    
    @abstractmethod
    def get_by_id(self, entity_id: Identifier) -> T | None:
        """Get entity by ID."""
        pass
    
    @abstractmethod
    def save(self, entity: T) -> None:
        """Save entity."""
        pass
    
    @abstractmethod
    def delete(self, entity_id: Identifier) -> None:
        """Delete entity by ID."""
        pass
    
    def exists(self, entity_id: Identifier) -> bool:
        """Check if entity exists."""
        return self.get_by_id(entity_id) is not None


class DomainService(ABC):
    """
    Base class for domain services.
    
    Domain services contain domain logic that doesn't naturally
    fit within a single entity or value object.
    """
    
    pass


class Specification(ABC, Generic[T]):
    """
    Base class for specifications.
    
    Specifications encapsulate business rules and can be combined
    using boolean logic.
    """
    
    @abstractmethod
    def is_satisfied_by(self, candidate: T) -> bool:
        """Check if the candidate satisfies this specification."""
        pass
    
    def and_(self, other: Specification[T]) -> Specification[T]:
        """Combine with another specification using AND."""
        return AndSpecification(self, other)
    
    def or_(self, other: Specification[T]) -> Specification[T]:
        """Combine with another specification using OR."""
        return OrSpecification(self, other)
    
    def not_(self) -> Specification[T]:
        """Negate this specification."""
        return NotSpecification(self)


class AndSpecification(Specification[T]):
    """AND combination of two specifications."""
    
    def __init__(self, left: Specification[T], right: Specification[T]) -> None:
        self.left = left
        self.right = right
    
    def is_satisfied_by(self, candidate: T) -> bool:
        return self.left.is_satisfied_by(candidate) and self.right.is_satisfied_by(candidate)


class OrSpecification(Specification[T]):
    """OR combination of two specifications."""
    
    def __init__(self, left: Specification[T], right: Specification[T]) -> None:
        self.left = left
        self.right = right
    
    def is_satisfied_by(self, candidate: T) -> bool:
        return self.left.is_satisfied_by(candidate) or self.right.is_satisfied_by(candidate)


class NotSpecification(Specification[T]):
    """NOT negation of a specification."""
    
    def __init__(self, spec: Specification[T]) -> None:
        self.spec = spec
    
    def is_satisfied_by(self, candidate: T) -> bool:
        return not self.spec.is_satisfied_by(candidate)