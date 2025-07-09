"""Base entity with domain event support."""

from __future__ import annotations

from abc import ABC
from datetime import datetime
from typing import Any, ClassVar, Generic, TypeVar
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from .domain_event import DomainEvent

T = TypeVar("T", bound="BaseEntity")


class BaseEntity(BaseModel, Generic[T], ABC):
    """Base entity with domain event support and lifecycle management.

    This class provides foundational functionality for all domain entities
    including identity, timestamps, event tracking, and validation.
    """

    # Entity identity
    id: UUID = Field(default_factory=uuid4, description="Unique entity identifier")

    # Lifecycle timestamps
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="Creation timestamp"
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow, description="Last update timestamp"
    )

    # Version for optimistic locking
    version: int = Field(default=1, description="Entity version for optimistic locking")

    # Domain events
    _domain_events: list[DomainEvent] = Field(
        default_factory=list, init=False, repr=False
    )

    # Entity metadata
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Entity metadata"
    )

    # Class-level configuration
    __abstract__: ClassVar[bool] = True
    __entity_type__: ClassVar[str] = ""

    class Config:
        """Pydantic configuration."""

        # Allow mutation for entity updates
        allow_mutation = True
        # Validate assignment for data integrity
        validate_assignment = True
        # Use enum values for serialization
        use_enum_values = True
        # Custom JSON serialization
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }
        # Schema extra for OpenAPI docs
        schema_extra = {
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174000",
                "created_at": "2023-12-01T10:00:00Z",
                "updated_at": "2023-12-01T10:00:00Z",
                "version": 1,
                "metadata": {},
            }
        }

    def __init__(self, **data: Any) -> None:
        """Initialize entity with validation."""
        super().__init__(**data)
        self._domain_events = []

        # Set entity type if not already set
        if not self.__entity_type__:
            self.__entity_type__ = self.__class__.__name__

    def __hash__(self) -> int:
        """Hash based on entity ID."""
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        """Equality based on entity ID and type."""
        if not isinstance(other, BaseEntity):
            return False
        return self.id == other.id and type(self) == type(other)

    def __repr__(self) -> str:
        """String representation showing entity type and ID."""
        return f"{self.__class__.__name__}(id={self.id})"

    def add_domain_event(self, event: DomainEvent) -> None:
        """Add a domain event to the entity.

        Args:
            event: The domain event to add
        """
        self._domain_events.append(event)

    def get_domain_events(self) -> list[DomainEvent]:
        """Get all domain events for this entity.

        Returns:
            List of domain events
        """
        return self._domain_events.copy()

    def clear_domain_events(self) -> None:
        """Clear all domain events."""
        self._domain_events.clear()

    def mark_as_updated(self) -> None:
        """Mark entity as updated and increment version."""
        self.updated_at = datetime.utcnow()
        self.version += 1

    def is_new(self) -> bool:
        """Check if entity is new (not persisted).

        Returns:
            True if entity is new
        """
        return self.version == 1 and self.created_at == self.updated_at

    def get_entity_type(self) -> str:
        """Get the entity type.

        Returns:
            Entity type name
        """
        return self.__entity_type__

    def to_dict(self, include_events: bool = False) -> dict[str, Any]:
        """Convert entity to dictionary.

        Args:
            include_events: Whether to include domain events

        Returns:
            Dictionary representation
        """
        data = self.dict()

        if include_events:
            data["domain_events"] = [event.dict() for event in self._domain_events]

        return data

    def validate_invariants(self) -> None:
        """Validate entity invariants.

        Override in subclasses to add domain-specific validation.

        Raises:
            ValueError: If invariants are violated
        """
        # Base validation
        if self.version < 1:
            raise ValueError("Entity version must be positive")

        if self.created_at > self.updated_at:
            raise ValueError("Created timestamp cannot be after updated timestamp")

    def clone(self: T) -> T:
        """Create a copy of the entity with a new ID.

        Returns:
            New entity instance with same data but different ID
        """
        data = self.dict()
        data["id"] = uuid4()
        data["created_at"] = datetime.utcnow()
        data["updated_at"] = datetime.utcnow()
        data["version"] = 1

        # Create new instance without domain events
        new_entity = self.__class__(**data)
        new_entity._domain_events = []

        return new_entity

    def apply_changes(self, changes: dict[str, Any]) -> None:
        """Apply changes to the entity.

        Args:
            changes: Dictionary of field changes
        """
        for field, value in changes.items():
            if hasattr(self, field):
                setattr(self, field, value)

        self.mark_as_updated()
        self.validate_invariants()

    @classmethod
    def get_identifier_field(cls) -> str:
        """Get the identifier field name.

        Returns:
            Name of the identifier field
        """
        return "id"

    @classmethod
    def create_from_dict(cls: type[T], data: dict[str, Any]) -> T:
        """Create entity from dictionary.

        Args:
            data: Dictionary with entity data

        Returns:
            New entity instance
        """
        # Remove domain events from data if present
        clean_data = {k: v for k, v in data.items() if k != "domain_events"}
        return cls(**clean_data)
