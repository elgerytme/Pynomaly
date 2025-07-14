"""Domain layer abstractions for enterprise applications.

This module provides base classes and patterns for implementing domain-driven design
with clean architecture principles.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import Any, Generic, TypeVar
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

EntityId = TypeVar("EntityId")


class BaseValueObject(BaseModel):
    """Base class for value objects in domain-driven design.

    Value objects are immutable objects that represent a descriptive aspect
    of the domain with no conceptual identity.
    """

    model_config = ConfigDict(
        frozen=True,  # Immutable
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )


class DomainEvent(BaseValueObject):
    """Base class for domain events.

    Domain events represent something that happened in the domain that domain
    experts care about.
    """

    event_id: UUID = Field(default_factory=uuid.uuid4)
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    event_type: str = Field(..., description="Type of the domain event")
    version: int = Field(default=1, description="Event schema version")
    metadata: dict[str, Any] = Field(default_factory=dict)


class BaseEntity(BaseModel, Generic[EntityId]):
    """Base class for entities in domain-driven design.

    Entities are objects that have a distinct identity that runs through time
    and different representations.
    """

    model_config = ConfigDict(
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    id: EntityId = Field(..., description="Unique identifier for the entity")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When the entity was created",
    )
    updated_at: datetime | None = Field(
        default=None, description="When the entity was last updated"
    )
    version: int = Field(default=1, description="Version for optimistic concurrency")

    # Domain events that occurred on this entity
    _domain_events: list[DomainEvent] = Field(default_factory=list, exclude=True)

    def __eq__(self, other: object) -> bool:
        """Two entities are equal if they have the same type and ID."""
        if not isinstance(other, BaseEntity):
            return False
        return type(self) is type(other) and self.id == other.id

    def __hash__(self) -> int:
        """Hash based on entity type and ID."""
        return hash((type(self), self.id))

    def add_domain_event(self, event: DomainEvent) -> None:
        """Add a domain event to this entity."""
        self._domain_events.append(event)

    def clear_domain_events(self) -> list[DomainEvent]:
        """Clear and return all domain events."""
        events = self._domain_events.copy()
        self._domain_events.clear()
        return events

    def mark_as_updated(self) -> None:
        """Mark the entity as updated with current timestamp."""
        self.updated_at = datetime.now(UTC)
        self.version += 1


class AggregateRoot(BaseEntity[EntityId]):
    """Base class for aggregate roots in domain-driven design.

    An aggregate root is the only member of its aggregate that outside objects
    are allowed to hold references to.
    """

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        # Aggregate roots are responsible for maintaining consistency
        self._is_dirty = False

    def mark_as_dirty(self) -> None:
        """Mark the aggregate as having pending changes."""
        self._is_dirty = True
        self.mark_as_updated()

    def mark_as_clean(self) -> None:
        """Mark the aggregate as having no pending changes."""
        self._is_dirty = False

    @property
    def is_dirty(self) -> bool:
        """Check if the aggregate has pending changes."""
        return self._is_dirty


class DomainService(ABC):
    """Base class for domain services.

    Domain services encapsulate domain logic that doesn't naturally fit
    within entities or value objects.
    """

    @abstractmethod
    def __init__(self) -> None:
        """Initialize the domain service."""
        pass


class Specification(ABC, Generic[EntityId]):
    """Base class for specifications pattern.

    Specifications encapsulate business rules and can be combined using
    logical operators.
    """

    @abstractmethod
    def is_satisfied_by(self, candidate: BaseEntity[EntityId]) -> bool:
        """Check if the candidate satisfies this specification."""
        pass

    def and_(self, other: Specification[EntityId]) -> AndSpecification[EntityId]:
        """Combine this specification with another using AND logic."""
        return AndSpecification(self, other)

    def or_(self, other: Specification[EntityId]) -> OrSpecification[EntityId]:
        """Combine this specification with another using OR logic."""
        return OrSpecification(self, other)

    def not_(self) -> NotSpecification[EntityId]:
        """Negate this specification."""
        return NotSpecification(self)


class AndSpecification(Specification[EntityId]):
    """Specification that combines two specifications with AND logic."""

    def __init__(
        self, left: Specification[EntityId], right: Specification[EntityId]
    ) -> None:
        self.left = left
        self.right = right

    def is_satisfied_by(self, candidate: BaseEntity[EntityId]) -> bool:
        return self.left.is_satisfied_by(candidate) and self.right.is_satisfied_by(
            candidate
        )


class OrSpecification(Specification[EntityId]):
    """Specification that combines two specifications with OR logic."""

    def __init__(
        self, left: Specification[EntityId], right: Specification[EntityId]
    ) -> None:
        self.left = left
        self.right = right

    def is_satisfied_by(self, candidate: BaseEntity[EntityId]) -> bool:
        return self.left.is_satisfied_by(candidate) or self.right.is_satisfied_by(
            candidate
        )


class NotSpecification(Specification[EntityId]):
    """Specification that negates another specification."""

    def __init__(self, spec: Specification[EntityId]) -> None:
        self.spec = spec

    def is_satisfied_by(self, candidate: BaseEntity[EntityId]) -> bool:
        return not self.spec.is_satisfied_by(candidate)
