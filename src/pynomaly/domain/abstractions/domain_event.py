"""Domain event abstractions for event-driven architecture."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4


class DomainEvent(ABC):
    """Base class for domain events."""
    
    def __init__(self, aggregate_id: UUID, event_id: UUID | None = None, 
                 occurred_at: datetime | None = None, **kwargs):
        """Initialize domain event."""
        self.event_id = event_id or uuid4()
        self.aggregate_id = aggregate_id
        self.occurred_at = occurred_at or datetime.utcnow()
        self.event_type = self.__class__.__name__
        self.metadata = kwargs
    
    def to_dict(self) -> dict[str, Any]:
        """Convert event to dictionary."""
        return {
            "event_id": str(self.event_id),
            "aggregate_id": str(self.aggregate_id),
            "event_type": self.event_type,
            "occurred_at": self.occurred_at.isoformat(),
            "metadata": self.metadata,
        }
    
    def __eq__(self, other: object) -> bool:
        """Check equality based on event ID."""
        if not isinstance(other, DomainEvent):
            return False
        return self.event_id == other.event_id
    
    def __hash__(self) -> int:
        """Hash based on event ID."""
        return hash(self.event_id)


class DomainEventHandler(ABC):
    """Base class for domain event handlers."""
    
    @abstractmethod
    async def handle(self, event: DomainEvent) -> None:
        """Handle domain event."""
        pass
    
    @property
    @abstractmethod
    def event_type(self) -> type[DomainEvent]:
        """Get the event type this handler processes."""
        pass
