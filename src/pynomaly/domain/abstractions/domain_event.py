"""Domain event system for architectural patterns."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Generic, Protocol, TypeVar
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

T = TypeVar("T")


class DomainEvent(BaseModel, ABC):
    """Base class for domain events.

    Domain events represent something that happened in the domain
    that is of interest to other parts of the system.
    """

    # Event identity
    event_id: UUID = Field(default_factory=uuid4, description="Unique event identifier")

    # Event metadata
    event_type: str = Field(description="Type of the event")
    occurred_at: datetime = Field(
        default_factory=datetime.utcnow, description="When the event occurred"
    )

    # Aggregate information
    aggregate_id: UUID = Field(
        description="ID of the aggregate that generated the event"
    )
    aggregate_type: str = Field(description="Type of the aggregate")
    aggregate_version: int = Field(
        description="Version of the aggregate when event was generated"
    )

    # Event data
    event_data: dict[str, Any] = Field(
        default_factory=dict, description="Event-specific data"
    )

    # Correlation and causation
    correlation_id: UUID | None = Field(None, description="Correlation ID for tracing")
    causation_id: UUID | None = Field(
        None, description="ID of the event that caused this event"
    )

    # User context
    user_id: UUID | None = Field(
        None, description="ID of the user who triggered the event"
    )
    user_context: dict[str, Any] = Field(
        default_factory=dict, description="User context information"
    )

    class Config:
        """Pydantic configuration."""

        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }

    def __init__(self, **data: Any) -> None:
        """Initialize domain event."""
        # Set event type from class name if not provided
        if "event_type" not in data:
            data["event_type"] = self.__class__.__name__

        super().__init__(**data)

    def __hash__(self) -> int:
        """Hash based on event ID."""
        return hash(self.event_id)

    def __eq__(self, other: object) -> bool:
        """Equality based on event ID."""
        if not isinstance(other, DomainEvent):
            return False
        return self.event_id == other.event_id

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(event_id={self.event_id}, aggregate_id={self.aggregate_id})"

    def to_dict(self) -> dict[str, Any]:
        """Convert event to dictionary."""
        return self.dict()

    def get_event_headers(self) -> dict[str, Any]:
        """Get event headers for messaging."""
        return {
            "event_id": str(self.event_id),
            "event_type": self.event_type,
            "occurred_at": self.occurred_at.isoformat(),
            "aggregate_id": str(self.aggregate_id),
            "aggregate_type": self.aggregate_type,
            "aggregate_version": self.aggregate_version,
            "correlation_id": str(self.correlation_id) if self.correlation_id else None,
            "causation_id": str(self.causation_id) if self.causation_id else None,
            "user_id": str(self.user_id) if self.user_id else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DomainEvent:
        """Create event from dictionary."""
        return cls(**data)

    @classmethod
    def get_event_type(cls) -> str:
        """Get the event type name."""
        return cls.__name__


class DomainEventHandler(Protocol, Generic[T]):
    """Protocol for domain event handlers."""

    @abstractmethod
    def handle(self, event: T) -> None:
        """Handle a domain event.

        Args:
            event: The domain event to handle
        """
        ...

    @abstractmethod
    def can_handle(self, event_type: str) -> bool:
        """Check if this handler can handle the given event type.

        Args:
            event_type: The event type to check

        Returns:
            True if this handler can handle the event type
        """
        ...


class DomainEventBus:
    """Simple domain event bus for dispatching events."""

    def __init__(self) -> None:
        """Initialize event bus."""
        self._handlers: dict[str, list[DomainEventHandler]] = {}

    def register_handler(self, event_type: str, handler: DomainEventHandler) -> None:
        """Register an event handler.

        Args:
            event_type: The event type to handle
            handler: The event handler
        """
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)

    def unregister_handler(self, event_type: str, handler: DomainEventHandler) -> None:
        """Unregister an event handler.

        Args:
            event_type: The event type
            handler: The event handler to remove
        """
        if event_type in self._handlers:
            self._handlers[event_type].remove(handler)

    def publish(self, event: DomainEvent) -> None:
        """Publish a domain event.

        Args:
            event: The domain event to publish
        """
        handlers = self._handlers.get(event.event_type, [])

        for handler in handlers:
            try:
                handler.handle(event)
            except Exception as e:
                # Log error but don't stop processing other handlers
                print(f"Error handling event {event.event_type}: {e}")

    def publish_all(self, events: list[DomainEvent]) -> None:
        """Publish multiple domain events.

        Args:
            events: List of domain events to publish
        """
        for event in events:
            self.publish(event)

    def get_handlers(self, event_type: str) -> list[DomainEventHandler]:
        """Get handlers for an event type.

        Args:
            event_type: The event type

        Returns:
            List of handlers for the event type
        """
        return self._handlers.get(event_type, []).copy()

    def clear_handlers(self, event_type: str | None = None) -> None:
        """Clear event handlers.

        Args:
            event_type: Event type to clear handlers for, or None to clear all
        """
        if event_type is None:
            self._handlers.clear()
        else:
            self._handlers.pop(event_type, None)


# Global event bus instance
_domain_event_bus: DomainEventBus | None = None


def get_domain_event_bus() -> DomainEventBus:
    """Get the global domain event bus."""
    global _domain_event_bus
    if _domain_event_bus is None:
        _domain_event_bus = DomainEventBus()
    return _domain_event_bus


class EventSourcedEntity(BaseModel, ABC):
    """Base class for event-sourced entities."""

    # Entity identity
    id: UUID = Field(default_factory=uuid4)

    # Event sourcing
    version: int = Field(default=0, description="Current version of the entity")
    uncommitted_events: list[DomainEvent] = Field(default_factory=list, init=False)

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True

    def __init__(self, **data: Any) -> None:
        """Initialize event-sourced entity."""
        super().__init__(**data)
        self.uncommitted_events = []

    def apply_event(self, event: DomainEvent) -> None:
        """Apply an event to the entity.

        Args:
            event: The domain event to apply
        """
        self._apply_event(event)
        self.version += 1

    def raise_event(self, event: DomainEvent) -> None:
        """Raise a new domain event.

        Args:
            event: The domain event to raise
        """
        self.apply_event(event)
        self.uncommitted_events.append(event)

    def mark_events_as_committed(self) -> None:
        """Mark all uncommitted events as committed."""
        self.uncommitted_events.clear()

    def get_uncommitted_events(self) -> list[DomainEvent]:
        """Get all uncommitted events.

        Returns:
            List of uncommitted events
        """
        return self.uncommitted_events.copy()

    @abstractmethod
    def _apply_event(self, event: DomainEvent) -> None:
        """Apply an event to update entity state.

        Args:
            event: The domain event to apply
        """
        ...

    @classmethod
    def from_history(cls, events: list[DomainEvent]) -> EventSourcedEntity:
        """Reconstruct entity from event history.

        Args:
            events: List of domain events in chronological order

        Returns:
            Entity reconstructed from events
        """
        if not events:
            raise ValueError("Cannot reconstruct entity from empty event history")

        # Create entity with ID from first event
        entity = cls(id=events[0].aggregate_id)

        # Apply all events
        for event in events:
            entity.apply_event(event)

        # Clear uncommitted events since these are from history
        entity.mark_events_as_committed()

        return entity
