"""Base entity abstraction."""

from abc import ABC
from datetime import datetime
from typing import Any, Generic, TypeVar
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T", bound="BaseEntity")


class BaseEntity(BaseModel, Generic[T], ABC):
    """Base entity with lifecycle management."""

    id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    version: int = Field(default=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    processor_config = ConfigDict(
        allow_mutation=True,
        json_encoders={
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }
    )

    def __init__(self, **data: Any) -> None:
        """Initialize entity."""
        super().__init__(**data)

    def __hash__(self) -> int:
        """Hash based on entity ID."""
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        """Equality based on entity ID."""
        if not isinstance(other, BaseEntity):
            return False
        return self.id == other.id

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(id={self.id})"

    def mark_as_updated(self) -> None:
        """Mark entity as updated."""
        self.updated_at = datetime.utcnow()
        self.version += 1

    def is_new(self) -> bool:
        """Check if entity is new."""
        return self.version == 1

    def validate_invariants(self) -> None:
        """Validate entity invariants."""
        if self.version < 1:
            raise ValueError("Entity version must be positive")

    @classmethod
    def get_identifier_field(cls) -> str:
        """Get identifier field name."""
        return "id"
