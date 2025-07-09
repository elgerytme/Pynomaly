"""Base entity class for domain models."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict
from uuid import uuid4


class BaseEntity(ABC):
    """Base class for all domain entities."""
    
    def __init__(self, entity_id: str = None):
        """Initialize base entity with ID."""
        self.id = entity_id or str(uuid4())
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Convert entity to dictionary representation."""
        pass
    
    @abstractmethod
    def validate(self) -> bool:
        """Validate entity data."""
        pass
    
    def update_timestamp(self):
        """Update the entity's timestamp."""
        self.updated_at = datetime.now()
    
    def __eq__(self, other):
        """Check equality based on ID."""
        if not isinstance(other, BaseEntity):
            return False
        return self.id == other.id
    
    def __hash__(self):
        """Hash based on ID."""
        return hash(self.id)
    
    def __repr__(self):
        """String representation of the entity."""
        return f"{self.__class__.__name__}(id={self.id})"
