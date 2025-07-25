"""
Event interface for ecosystem event-driven integration.

This module provides interfaces and utilities for event-driven
communication and integration across the platform ecosystem.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Callable, Union, AsyncIterator
from uuid import UUID, uuid4

import structlog

logger = structlog.get_logger(__name__)


class EventType(Enum):
    """Types of events in the ecosystem."""
    SYSTEM = "system"
    INTEGRATION = "integration"
    DATA = "data"
    MODEL = "model"
    DEPLOYMENT = "deployment"
    MONITORING = "monitoring"
    SECURITY = "security"
    GOVERNANCE = "governance"
    USER = "user"
    CUSTOM = "custom"


class EventPriority(Enum):
    """Event priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class EventStatus(Enum):
    """Event processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    PROCESSED = "processed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Event:
    """Ecosystem event data structure."""
    
    # Event identification
    id: str = field(default_factory=lambda: str(uuid4()))
    type: EventType = EventType.CUSTOM
    name: str = ""
    
    # Event content
    payload: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Event properties
    priority: EventPriority = EventPriority.MEDIUM
    status: EventStatus = EventStatus.PENDING
    
    # Source information
    source: str = "unknown"
    source_id: Optional[str] = None
    integration_id: Optional[str] = None
    
    # Timing information
    timestamp: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    processed_at: Optional[datetime] = None
    
    # Processing information
    retry_count: int = 0
    max_retries: int = 3
    error_message: Optional[str] = None
    
    # Routing information
    target_integrations: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Post-initialization validation."""
        if not self.name:
            raise ValueError("Event name is required")
    
    @property
    def is_expired(self) -> bool:
        """Check if event has expired."""
        if not self.expires_at:
            return False
        return datetime.utcnow() > self.expires_at
    
    @property
    def can_retry(self) -> bool:
        """Check if event can be retried."""
        return self.retry_count < self.max_retries
    
    def mark_processing(self) -> None:
        """Mark event as processing."""
        self.status = EventStatus.PROCESSING
        self.processed_at = datetime.utcnow()
    
    def mark_processed(self) -> None:
        """Mark event as processed."""
        self.status = EventStatus.PROCESSED
        self.processed_at = datetime.utcnow()
    
    def mark_failed(self, error: str) -> None:
        """Mark event as failed with error message."""
        self.status = EventStatus.FAILED
        self.error_message = error
        self.retry_count += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return {
            "id": self.id,
            "type": self.type.value,
            "name": self.name,
            "payload": self.payload,
            "metadata": self.metadata,
            "priority": self.priority.value,
            "status": self.status.value,
            "source": self.source,
            "source_id": self.source_id,
            "integration_id": self.integration_id,
            "timestamp": self.timestamp.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "processed_at": self.processed_at.isoformat() if self.processed_at else None,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "error_message": self.error_message,
            "target_integrations": self.target_integrations,
            "tags": self.tags
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Event":
        """Create event from dictionary."""
        event = cls(
            id=data.get("id", str(uuid4())),
            type=EventType(data.get("type", "custom")),
            name=data.get("name", ""),
            payload=data.get("payload", {}),
            metadata=data.get("metadata", {}),
            priority=EventPriority(data.get("priority", "medium")),
            status=EventStatus(data.get("status", "pending")),
            source=data.get("source", "unknown"),
            source_id=data.get("source_id"),
            integration_id=data.get("integration_id"),
            retry_count=data.get("retry_count", 0),
            max_retries=data.get("max_retries", 3),
            error_message=data.get("error_message"),
            target_integrations=data.get("target_integrations", []),
            tags=data.get("tags", [])
        )
        
        # Parse timestamps
        if "timestamp" in data:
            event.timestamp = datetime.fromisoformat(data["timestamp"])
        if "expires_at" in data and data["expires_at"]:
            event.expires_at = datetime.fromisoformat(data["expires_at"])
        if "processed_at" in data and data["processed_at"]:
            event.processed_at = datetime.fromisoformat(data["processed_at"])
        
        return event


@dataclass
class EventSubscription:
    """Event subscription configuration."""
    
    # Subscription identification
    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    
    # Subscription filters
    event_types: List[EventType] = field(default_factory=list)
    event_names: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    
    # Subscription configuration
    priority_filter: Optional[EventPriority] = None
    batch_size: int = 1
    timeout_seconds: int = 30
    
    # Subscriber information
    subscriber_id: str = ""
    callback_url: Optional[str] = None
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    active: bool = True
    
    def matches_event(self, event: Event) -> bool:
        """Check if event matches subscription filters."""
        # Check event types
        if self.event_types and event.type not in self.event_types:
            return False
        
        # Check event names
        if self.event_names and event.name not in self.event_names:
            return False
        
        # Check sources
        if self.sources and event.source not in self.sources:
            return False
        
        # Check priority filter
        if self.priority_filter and event.priority != self.priority_filter:
            return False
        
        # Check tags (at least one tag must match)
        if self.tags and not any(tag in event.tags for tag in self.tags):
            return False
        
        return True


# Type aliases for event handlers
EventHandler = Callable[[Event], Any]
AsyncEventHandler = Callable[[Event], Any]  # Can be async
BatchEventHandler = Callable[[List[Event]], Any]


class EventInterface(ABC):
    """
    Abstract interface for event-driven integration.
    
    This interface defines the contract for event publishing,
    subscription, and handling across the ecosystem.
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """Initialize event interface."""
        self.name = name
        self.config = config
        self.id = uuid4()
        self.logger = logger.bind(
            event_interface=name,
            interface_id=str(self.id)
        )
        
        self._subscriptions: Dict[str, EventSubscription] = {}
        self._handlers: Dict[str, EventHandler] = {}
        
        self.logger.info("Event interface initialized")
    
    # Connection management
    
    @abstractmethod
    async def connect(self) -> bool:
        """
        Connect to event system.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """
        Disconnect from event system.
        
        Returns:
            bool: True if disconnection successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check event system health.
        
        Returns:
            bool: True if healthy, False otherwise
        """
        pass
    
    # Event publishing
    
    @abstractmethod
    async def publish_event(self, event: Event) -> bool:
        """
        Publish event to the system.
        
        Args:
            event: Event to publish
            
        Returns:
            bool: True if publish successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def publish_batch(self, events: List[Event]) -> bool:
        """
        Publish batch of events.
        
        Args:
            events: List of events to publish
            
        Returns:
            bool: True if all events published successfully, False otherwise
        """
        pass
    
    # Event subscription
    
    @abstractmethod
    async def subscribe(
        self,
        subscription: EventSubscription,
        handler: EventHandler
    ) -> str:
        """
        Subscribe to events with handler.
        
        Args:
            subscription: Subscription configuration
            handler: Event handler function
            
        Returns:
            str: Subscription ID
        """
        pass
    
    @abstractmethod
    async def unsubscribe(self, subscription_id: str) -> bool:
        """
        Unsubscribe from events.
        
        Args:
            subscription_id: Subscription ID to cancel
            
        Returns:
            bool: True if unsubscription successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def list_subscriptions(self) -> List[EventSubscription]:
        """
        List active subscriptions.
        
        Returns:
            List[EventSubscription]: Active subscriptions
        """
        pass
    
    # Event consumption
    
    @abstractmethod
    async def consume_events(
        self,
        subscription_id: str,
        max_events: int = 10,
        timeout_seconds: Optional[int] = None
    ) -> List[Event]:
        """
        Consume events from subscription.
        
        Args:
            subscription_id: Subscription ID
            max_events: Maximum number of events to consume
            timeout_seconds: Timeout for consumption
            
        Returns:
            List[Event]: Consumed events
        """
        pass
    
    @abstractmethod
    async def consume_stream(
        self,
        subscription_id: str
    ) -> AsyncIterator[Event]:
        """
        Consume events as stream.
        
        Args:
            subscription_id: Subscription ID
            
        Yields:
            Event: Streaming events
        """
        pass
    
    # Event acknowledgment
    
    @abstractmethod
    async def acknowledge_event(self, event_id: str) -> bool:
        """
        Acknowledge event processing.
        
        Args:
            event_id: Event ID to acknowledge
            
        Returns:
            bool: True if acknowledgment successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def reject_event(
        self,
        event_id: str,
        reason: str,
        requeue: bool = True
    ) -> bool:
        """
        Reject event processing.
        
        Args:
            event_id: Event ID to reject
            reason: Rejection reason
            requeue: Whether to requeue event for retry
            
        Returns:
            bool: True if rejection successful, False otherwise
        """
        pass
    
    # Event querying
    
    @abstractmethod
    async def get_event(self, event_id: str) -> Optional[Event]:
        """
        Get event by ID.
        
        Args:
            event_id: Event ID
            
        Returns:
            Event: Event if found, None otherwise
        """
        pass
    
    @abstractmethod
    async def query_events(
        self,
        filters: Dict[str, Any],
        limit: int = 100,
        offset: int = 0
    ) -> List[Event]:
        """
        Query events with filters.
        
        Args:
            filters: Query filters
            limit: Maximum number of events to return
            offset: Query offset
            
        Returns:
            List[Event]: Matching events
        """
        pass
    
    # Event metrics and monitoring
    
    @abstractmethod
    async def get_event_metrics(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> Dict[str, Any]:
        """
        Get event metrics for time period.
        
        Args:
            start_time: Metrics start time
            end_time: Metrics end time
            
        Returns:
            Dict[str, Any]: Event metrics
        """
        pass
    
    @abstractmethod
    async def get_subscription_metrics(
        self,
        subscription_id: str
    ) -> Dict[str, Any]:
        """
        Get metrics for specific subscription.
        
        Args:
            subscription_id: Subscription ID
            
        Returns:
            Dict[str, Any]: Subscription metrics
        """
        pass
    
    # Utility methods
    
    async def create_event(
        self,
        name: str,
        event_type: EventType = EventType.CUSTOM,
        payload: Optional[Dict[str, Any]] = None,
        priority: EventPriority = EventPriority.MEDIUM,
        **kwargs
    ) -> Event:
        """
        Create new event.
        
        Args:
            name: Event name
            event_type: Event type
            payload: Event payload
            priority: Event priority
            **kwargs: Additional event properties
            
        Returns:
            Event: Created event
        """
        return Event(
            name=name,
            type=event_type,
            payload=payload or {},
            priority=priority,
            source=self.name,
            integration_id=str(self.id),
            **kwargs
        )
    
    async def create_subscription(
        self,
        name: str,
        event_types: Optional[List[EventType]] = None,
        event_names: Optional[List[str]] = None,
        **kwargs
    ) -> EventSubscription:
        """
        Create new subscription.
        
        Args:
            name: Subscription name
            event_types: Event types to subscribe to
            event_names: Event names to subscribe to
            **kwargs: Additional subscription properties
            
        Returns:
            EventSubscription: Created subscription
        """
        return EventSubscription(
            name=name,
            event_types=event_types or [],
            event_names=event_names or [],
            subscriber_id=str(self.id),
            **kwargs
        )
    
    def _store_subscription(
        self,
        subscription: EventSubscription,
        handler: EventHandler
    ) -> None:
        """Store subscription and handler locally."""
        self._subscriptions[subscription.id] = subscription
        self._handlers[subscription.id] = handler
    
    def _remove_subscription(self, subscription_id: str) -> None:
        """Remove subscription and handler locally."""
        self._subscriptions.pop(subscription_id, None)
        self._handlers.pop(subscription_id, None)
    
    async def _handle_event(self, event: Event, handler: EventHandler) -> None:
        """Handle event with error handling."""
        try:
            if asyncio.iscoroutinefunction(handler):
                await handler(event)
            else:
                handler(event)
                
            self.logger.debug("Event handled successfully", event_id=event.id)
            
        except Exception as e:
            self.logger.error(
                "Event handler failed",
                event_id=event.id,
                error=str(e)
            )
            raise
    
    def __repr__(self) -> str:
        """String representation."""
        return f"EventInterface(name={self.name}, id={self.id})"