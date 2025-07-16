"""
Entity ID Value Object

Represents a unique identifier for domain entities with validation and formatting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import UUID, uuid4


@dataclass(frozen=True)
class EntityId:
    """
    Value object representing a unique entity identifier.
    
    This value object ensures that entity IDs are valid UUIDs and provides
    utilities for ID generation and validation.
    """
    
    value: UUID
    
    def __post_init__(self) -> None:
        """Validate the entity ID."""
        if not isinstance(self.value, UUID):
            raise ValueError(f"EntityId must be a UUID, got {type(self.value)}")
    
    @classmethod
    def generate(cls) -> EntityId:
        """Generate a new random entity ID."""
        return cls(uuid4())
    
    @classmethod
    def from_string(cls, id_string: str) -> EntityId:
        """Create EntityId from string representation."""
        try:
            return cls(UUID(id_string))
        except ValueError as e:
            raise ValueError(f"Invalid UUID string: {id_string}") from e
    
    def __str__(self) -> str:
        """String representation."""
        return str(self.value)
    
    def __eq__(self, other: Any) -> bool:
        """Equality comparison."""
        if not isinstance(other, EntityId):
            return False
        return self.value == other.value
    
    def __hash__(self) -> int:
        """Hash for use in sets and dictionaries."""
        return hash(self.value)