"""Protocol definitions for enterprise applications.

This module defines interfaces and contracts that can be implemented by
infrastructure components while keeping the domain layer independent.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, Protocol, TypeVar, runtime_checkable

from .domain import BaseEntity, DomainEvent

EntityId = TypeVar("EntityId")
Entity = TypeVar("Entity", bound=BaseEntity)


@runtime_checkable
class Repository(Protocol, Generic[Entity, EntityId]):
    """Repository protocol for data persistence.

    Repositories encapsulate the logic needed to access data sources.
    They centralize common data access functionality.
    """

    async def save(self, entity: Entity) -> None:
        """Save an entity to the repository."""
        ...

    async def find_by_id(self, entity_id: EntityId) -> Entity | None:
        """Find an entity by its ID."""
        ...

    async def find_all(self) -> list[Entity]:
        """Find all entities."""
        ...

    async def delete(self, entity: Entity) -> None:
        """Delete an entity from the repository."""
        ...

    async def delete_by_id(self, entity_id: EntityId) -> None:
        """Delete an entity by its ID."""
        ...

    async def exists(self, entity_id: EntityId) -> bool:
        """Check if an entity exists."""
        ...

    async def count(self) -> int:
        """Count the total number of entities."""
        ...


@runtime_checkable
class UnitOfWork(Protocol):
    """Unit of Work protocol for transaction management.

    The Unit of Work pattern maintains a list of objects affected by a
    business transaction and coordinates writing out changes.
    """

    async def __aenter__(self) -> UnitOfWork:
        """Enter the unit of work context."""
        ...

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit the unit of work context."""
        ...

    async def commit(self) -> None:
        """Commit all changes in this unit of work."""
        ...

    async def rollback(self) -> None:
        """Rollback all changes in this unit of work."""
        ...


@runtime_checkable
class EventBus(Protocol):
    """Event bus protocol for domain event publishing.

    The event bus decouples event publishers from event subscribers.
    """

    async def publish(self, event: DomainEvent) -> None:
        """Publish a domain event."""
        ...

    async def publish_many(self, events: list[DomainEvent]) -> None:
        """Publish multiple domain events."""
        ...

    def subscribe(self, event_type: str, handler: EventHandler) -> None:
        """Subscribe to events of a specific type."""
        ...

    def unsubscribe(self, event_type: str, handler: EventHandler) -> None:
        """Unsubscribe from events of a specific type."""
        ...


@runtime_checkable
class EventHandler(Protocol):
    """Event handler protocol for processing domain events."""

    async def handle(self, event: DomainEvent) -> None:
        """Handle a domain event."""
        ...


@runtime_checkable
class Logger(Protocol):
    """Logger protocol for structured logging."""

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log a debug message."""
        ...

    def info(self, message: str, **kwargs: Any) -> None:
        """Log an info message."""
        ...

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log a warning message."""
        ...

    def error(self, message: str, **kwargs: Any) -> None:
        """Log an error message."""
        ...

    def critical(self, message: str, **kwargs: Any) -> None:
        """Log a critical message."""
        ...


@runtime_checkable
class Cache(Protocol):
    """Cache protocol for data caching."""

    async def get(self, key: str) -> Any | None:
        """Get a value from the cache."""
        ...

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Set a value in the cache with optional TTL."""
        ...

    async def delete(self, key: str) -> None:
        """Delete a value from the cache."""
        ...

    async def exists(self, key: str) -> bool:
        """Check if a key exists in the cache."""
        ...

    async def clear(self) -> None:
        """Clear all cached values."""
        ...


@runtime_checkable
class MessageQueue(Protocol):
    """Message queue protocol for asynchronous messaging."""

    async def send(self, queue: str, message: dict[str, Any]) -> None:
        """Send a message to a queue."""
        ...

    async def receive(self, queue: str) -> dict[str, Any] | None:
        """Receive a message from a queue."""
        ...

    async def subscribe(self, queue: str, handler: MessageHandler) -> None:
        """Subscribe to messages from a queue."""
        ...


@runtime_checkable
class MessageHandler(Protocol):
    """Message handler protocol for processing queue messages."""

    async def handle(self, message: dict[str, Any]) -> None:
        """Handle a queue message."""
        ...


@runtime_checkable
class HealthCheck(Protocol):
    """Health check protocol for service monitoring."""

    async def check(self) -> HealthStatus:
        """Perform a health check."""
        ...


class HealthStatus:
    """Health check status result."""

    def __init__(
        self,
        status: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.status = status  # "healthy", "degraded", "unhealthy"
        self.message = message
        self.details = details or {}
        self.timestamp = None  # Set by monitoring system


@runtime_checkable
class Metrics(Protocol):
    """Metrics protocol for application monitoring."""

    def counter(
        self, name: str, value: float = 1, tags: dict[str, str] | None = None
    ) -> None:
        """Increment a counter metric."""
        ...

    def gauge(
        self, name: str, value: float, tags: dict[str, str] | None = None
    ) -> None:
        """Set a gauge metric."""
        ...

    def histogram(
        self, name: str, value: float, tags: dict[str, str] | None = None
    ) -> None:
        """Record a histogram metric."""
        ...

    def timer(self, name: str) -> TimerContext:
        """Create a timer context for measuring duration."""
        ...


@runtime_checkable
class TimerContext(Protocol):
    """Timer context protocol for measuring execution time."""

    def __enter__(self) -> TimerContext:
        """Enter the timer context."""
        ...

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit the timer context and record the duration."""
        ...


@runtime_checkable
class ConfigurationProvider(Protocol):
    """Configuration provider protocol for accessing configuration values."""

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        ...

    def get_section(self, section: str) -> dict[str, Any]:
        """Get a configuration section."""
        ...

    def has(self, key: str) -> bool:
        """Check if a configuration key exists."""
        ...


@runtime_checkable
class SecretProvider(Protocol):
    """Secret provider protocol for accessing sensitive configuration."""

    async def get_secret(self, key: str) -> str | None:
        """Get a secret value."""
        ...

    async def list_secrets(self) -> list[str]:
        """List available secret keys."""
        ...


@runtime_checkable
class FeatureFlag(Protocol):
    """Feature flag protocol for controlling feature rollouts."""

    def is_enabled(self, flag: str, context: dict[str, Any] | None = None) -> bool:
        """Check if a feature flag is enabled."""
        ...

    def get_variant(self, flag: str, context: dict[str, Any] | None = None) -> str:
        """Get the variant of a feature flag."""
        ...


class ServiceAdapter(ABC):
    """Base class for service adapters.

    Service adapters provide a common interface for external services
    while hiding implementation details.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self._is_healthy = True

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the service adapter."""
        pass

    @abstractmethod
    async def health_check(self) -> HealthStatus:
        """Perform a health check on the service."""
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up resources used by the service."""
        pass

    @property
    def is_healthy(self) -> bool:
        """Check if the service is healthy."""
        return self._is_healthy
