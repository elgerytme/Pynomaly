"""
Architectural patterns and interfaces for cross-package communication.

This module defines stable interfaces that enable different domains to
communicate while maintaining proper boundaries and architectural patterns.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, List, Optional, Type, TypeVar, Union
from uuid import UUID

from .dto import BaseDTO
from .events import DomainEvent, EventBus


# Generic types
T = TypeVar('T')
E = TypeVar('E')  # Entity type
ID = TypeVar('ID')  # ID type


class Repository(Generic[E, ID], ABC):
    """
    Abstract repository pattern for data access.
    
    Provides a consistent interface for data persistence operations
    across different domains and storage backends.
    """
    
    @abstractmethod
    async def get_by_id(self, entity_id: ID) -> Optional[E]:
        """Retrieve an entity by its ID."""
        pass
    
    @abstractmethod
    async def get_all(self, limit: Optional[int] = None, offset: int = 0) -> List[E]:
        """Retrieve all entities with optional pagination."""
        pass
    
    @abstractmethod
    async def find_by_criteria(self, criteria: Dict[str, Any]) -> List[E]:
        """Find entities matching the given criteria."""
        pass
    
    @abstractmethod
    async def save(self, entity: E) -> E:
        """Save an entity (create or update)."""
        pass
    
    @abstractmethod
    async def delete(self, entity_id: ID) -> bool:
        """Delete an entity by its ID."""
        pass
    
    @abstractmethod
    async def exists(self, entity_id: ID) -> bool:
        """Check if an entity exists."""
        pass
    
    @abstractmethod
    async def count(self, criteria: Optional[Dict[str, Any]] = None) -> int:
        """Count entities matching optional criteria."""
        pass


class Service(ABC):
    """
    Abstract service pattern for domain operations.
    
    Provides a consistent interface for business logic operations
    that can be shared across domains.
    """
    
    @abstractmethod
    async def execute(self, request: BaseDTO) -> BaseDTO:
        """Execute the service operation."""
        pass
    
    @abstractmethod
    async def validate_request(self, request: BaseDTO) -> bool:
        """Validate the service request."""
        pass
    
    @abstractmethod
    def get_service_info(self) -> Dict[str, Any]:
        """Get information about the service."""
        pass


class AntiCorruptionLayer(ABC):
    """
    Anti-corruption layer pattern for external integrations.
    
    Provides translation between external APIs/formats and internal
    domain models to prevent external changes from corrupting the domain.
    """
    
    @abstractmethod
    async def translate_incoming(self, external_data: Any) -> BaseDTO:
        """Translate external data to internal DTO format."""
        pass
    
    @abstractmethod
    async def translate_outgoing(self, internal_data: BaseDTO) -> Any:
        """Translate internal DTO to external format."""
        pass
    
    @abstractmethod
    def supports_format(self, format_type: str) -> bool:
        """Check if the layer supports a specific format."""
        pass


class QueryHandler(Generic[T], ABC):
    """
    Query handler pattern for read operations.
    
    Handles queries that return data without modifying state.
    """
    
    @abstractmethod
    async def handle(self, query: T) -> Any:
        """Handle the query and return results."""
        pass
    
    @abstractmethod
    def get_query_type(self) -> Type[T]:
        """Get the type of query this handler processes."""
        pass


class CommandHandler(Generic[T], ABC):
    """
    Command handler pattern for write operations.
    
    Handles commands that modify state and may publish events.
    """
    
    @abstractmethod
    async def handle(self, command: T) -> Any:
        """Handle the command and return results."""
        pass
    
    @abstractmethod
    def get_command_type(self) -> Type[T]:
        """Get the type of command this handler processes."""
        pass


class EventHandler(ABC):
    """
    Event handler pattern for processing domain events.
    """
    
    @abstractmethod
    async def handle(self, event: DomainEvent) -> None:
        """Handle a domain event."""
        pass
    
    @abstractmethod
    def get_event_types(self) -> List[Type[DomainEvent]]:
        """Get the types of events this handler processes."""
        pass


class MessageBus(ABC):
    """
    Message bus pattern for command/query dispatch.
    
    Routes commands and queries to their appropriate handlers.
    """
    
    @abstractmethod
    async def execute_command(self, command: Any) -> Any:
        """Execute a command through its handler."""
        pass
    
    @abstractmethod
    async def execute_query(self, query: Any) -> Any:
        """Execute a query through its handler."""
        pass
    
    @abstractmethod
    def register_command_handler(self, command_type: Type[T], handler: CommandHandler[T]) -> None:
        """Register a command handler."""
        pass
    
    @abstractmethod
    def register_query_handler(self, query_type: Type[T], handler: QueryHandler[T]) -> None:
        """Register a query handler."""
        pass


class UnitOfWork(ABC):
    """
    Unit of Work pattern for transaction management.
    
    Manages transactions across multiple repositories and ensures
    consistency in data operations.
    """
    
    @abstractmethod
    async def __aenter__(self):
        """Enter the unit of work context."""
        pass
    
    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit the unit of work context."""
        pass
    
    @abstractmethod
    async def commit(self) -> None:
        """Commit all changes in the unit of work."""
        pass
    
    @abstractmethod
    async def rollback(self) -> None:
        """Rollback all changes in the unit of work."""
        pass


class Cache(Generic[T], ABC):
    """
    Cache pattern for performance optimization.
    
    Provides a consistent interface for caching operations
    across different storage backends.
    """
    
    @abstractmethod
    async def get(self, key: str) -> Optional[T]:
        """Get a value from the cache."""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: T, ttl: Optional[int] = None) -> None:
        """Set a value in the cache with optional TTL."""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete a value from the cache."""
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if a key exists in the cache."""
        pass
    
    @abstractmethod
    async def clear(self) -> None:
        """Clear all values from the cache."""
        pass


class HealthCheck(ABC):
    """
    Health check pattern for system monitoring.
    
    Provides a consistent interface for checking component health.
    """
    
    @abstractmethod
    async def check_health(self) -> Dict[str, Any]:
        """Check the health of the component."""
        pass
    
    @abstractmethod
    def get_component_name(self) -> str:
        """Get the name of the component being checked."""
        pass
    
    @abstractmethod
    async def check_dependencies(self) -> Dict[str, Dict[str, Any]]:
        """Check the health of component dependencies."""
        pass


class MetricsCollector(ABC):
    """
    Metrics collector pattern for observability.
    
    Provides a consistent interface for collecting and exposing metrics.
    """
    
    @abstractmethod
    def increment_counter(self, name: str, value: float = 1.0, tags: Optional[Dict[str, str]] = None) -> None:
        """Increment a counter metric."""
        pass
    
    @abstractmethod
    def set_gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Set a gauge metric."""
        pass
    
    @abstractmethod
    def record_histogram(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a histogram metric."""
        pass
    
    @abstractmethod
    async def get_metrics(self) -> Dict[str, Any]:
        """Get all collected metrics."""
        pass


class ConfigurationProvider(ABC):
    """
    Configuration provider pattern for settings management.
    
    Provides a consistent interface for accessing configuration
    across different sources and environments.
    """
    
    @abstractmethod
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        pass
    
    @abstractmethod
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get a configuration section."""
        pass
    
    @abstractmethod
    def has(self, key: str) -> bool:
        """Check if a configuration key exists."""
        pass
    
    @abstractmethod
    async def reload(self) -> None:
        """Reload configuration from source."""
        pass