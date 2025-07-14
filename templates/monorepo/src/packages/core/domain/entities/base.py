"""Base entity class for domain objects."""

from abc import ABC
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class Entity(BaseModel, ABC):
    """Base class for domain entities with identity and lifecycle management."""
    
    id: UUID = Field(default_factory=uuid4, description="Unique identifier")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation timestamp"
    )
    updated_at: Optional[datetime] = Field(
        default=None,
        description="Last update timestamp"
    )
    version: int = Field(default=1, description="Version for optimistic locking")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        validate_assignment = True
        arbitrary_types_allowed = True
        
    def __eq__(self, other: object) -> bool:
        """Compare entities by ID."""
        if not isinstance(other, Entity):
            return False
        return self.id == other.id
    
    def __hash__(self) -> int:
        """Hash based on ID."""
        return hash(self.id)
    
    def update_metadata(self, key: str, value: Any) -> None:
        """Update metadata and timestamp."""
        self.metadata[key] = value
        self.updated_at = datetime.now(timezone.utc)
    
    def increment_version(self) -> None:
        """Increment version for optimistic locking."""
        self.version += 1
        self.updated_at = datetime.now(timezone.utc)