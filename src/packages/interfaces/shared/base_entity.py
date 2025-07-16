"""
Base entity interface for domain entities.
Provides common functionality for all domain entities across packages.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional
from uuid import UUID, uuid4


@dataclass
class BaseEntity(ABC):
    """Base class for all domain entities."""
    
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None
    version: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization hook."""
        if self.updated_at is None:
            self.updated_at = self.created_at
    
    def update_timestamp(self) -> None:
        """Update the entity's timestamp and version."""
        self.updated_at = datetime.utcnow()
        self.version += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entity to dictionary representation."""
        return {
            "id": str(self.id),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "version": self.version,
            "metadata": self.metadata
        }
    
    @abstractmethod
    def validate(self) -> bool:
        """Validate the entity's state."""
        pass
    
    def __eq__(self, other) -> bool:
        """Equality comparison based on ID."""
        if not isinstance(other, BaseEntity):
            return False
        return self.id == other.id
    
    def __hash__(self) -> int:
        """Hash based on ID."""
        return hash(self.id)


class EntityRepository(ABC):
    """Abstract base class for entity repositories."""
    
    @abstractmethod
    async def save(self, entity: BaseEntity) -> BaseEntity:
        """Save an entity."""
        pass
    
    @abstractmethod
    async def find_by_id(self, entity_id: UUID) -> Optional[BaseEntity]:
        """Find an entity by ID."""
        pass
    
    @abstractmethod
    async def find_all(self) -> list[BaseEntity]:
        """Find all entities."""
        pass
    
    @abstractmethod
    async def delete(self, entity_id: UUID) -> bool:
        """Delete an entity by ID."""
        pass
    
    @abstractmethod
    async def update(self, entity: BaseEntity) -> BaseEntity:
        """Update an entity."""
        pass


class DomainEvent(BaseEntity):
    """Base class for domain events."""
    
    event_type: str = field(default="")
    event_data: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> bool:
        """Validate the domain event."""
        return bool(self.event_type)


class ValueObject(ABC):
    """Base class for value objects."""
    
    @abstractmethod
    def validate(self) -> bool:
        """Validate the value object."""
        pass
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Convert value object to dictionary."""
        pass


# Alias for compatibility
BaseValueObject = ValueObject