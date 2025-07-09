"""Pure WebSocket service models without FastAPI dependencies."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from uuid import UUID, uuid4


class MessageType(Enum):
    """WebSocket message types."""
    CONNECT = "connect"
    DISCONNECT = "disconnect"
    HEARTBEAT = "heartbeat"
    NOTIFICATION = "notification"
    DATA_UPDATE = "data_update"
    DETECTION_RESULT = "detection_result"
    TRAINING_STATUS = "training_status"
    SYSTEM_ALERT = "system_alert"
    USER_ACTION = "user_action"
    ERROR = "error"


class ConnectionStatus(Enum):
    """WebSocket connection status."""
    CONNECTING = "connecting"
    CONNECTED = "connected"
    DISCONNECTING = "disconnecting"
    DISCONNECTED = "disconnected"
    ERROR = "error"


class SubscriptionTopic(Enum):
    """WebSocket subscription topics."""
    DETECTION_RESULTS = "detection_results"
    TRAINING_UPDATES = "training_updates"
    SYSTEM_HEALTH = "system_health"
    USER_NOTIFICATIONS = "user_notifications"
    DATA_CHANGES = "data_changes"
    ALERTS = "alerts"
    ACTIVITY_FEED = "activity_feed"


@dataclass
class WebSocketMessageEnvelope:
    """WebSocket message envelope."""
    message_id: str
    type: MessageType
    topic: Optional[SubscriptionTopic]
    data: Dict[str, Any]
    timestamp: datetime
    sender_id: Optional[str] = None
    recipient_ids: Optional[Set[str]] = None
    requires_auth: bool = False
    
    def __post_init__(self):
        if self.recipient_ids is None:
            self.recipient_ids = set()


@dataclass
class ConnectionRegistry:
    """WebSocket connection registry entry."""
    connection_id: str
    user_id: Optional[str]
    client_info: Dict[str, str]
    subscriptions: Set[SubscriptionTopic]
    connected_at: datetime
    last_activity: datetime
    status: ConnectionStatus = ConnectionStatus.CONNECTED
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not hasattr(self, 'subscriptions'):
            self.subscriptions = set()


@dataclass
class Subscription:
    """WebSocket subscription."""
    connection_id: str
    topic: SubscriptionTopic
    filters: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class BroadcastMessage:
    """Broadcast message configuration."""
    message: WebSocketMessageEnvelope
    target_connections: Optional[Set[str]] = None
    target_users: Optional[Set[str]] = None
    target_topics: Optional[Set[SubscriptionTopic]] = None
    exclude_connections: Optional[Set[str]] = None
    
    def __post_init__(self):
        if self.target_connections is None:
            self.target_connections = set()
        if self.target_users is None:
            self.target_users = set()
        if self.target_topics is None:
            self.target_topics = set()
        if self.exclude_connections is None:
            self.exclude_connections = set()


@dataclass
class WebSocketConfig:
    """WebSocket configuration."""
    heartbeat_interval: int = 30  # seconds
    connection_timeout: int = 300  # seconds
    max_connections_per_user: int = 5
    max_message_size: int = 64 * 1024  # 64KB
    enable_compression: bool = True
    allowed_origins: List[str] = field(default_factory=list)
    require_auth: bool = True
    
    def __post_init__(self):
        if not self.allowed_origins:
            self.allowed_origins = ["*"]


@dataclass
class RateLimitConfig:
    """Rate limiting configuration for WebSocket."""
    messages_per_minute: int = 60
    bytes_per_minute: int = 1024 * 1024  # 1MB
    connections_per_ip: int = 10
    enable_rate_limiting: bool = True


@dataclass
class NotificationPreferences:
    """User notification preferences."""
    user_id: str
    enabled_topics: Set[SubscriptionTopic]
    quiet_hours: Optional[tuple[int, int]] = None  # (start_hour, end_hour)
    delivery_methods: Set[str] = field(default_factory=lambda: {"websocket"})
    filters: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not hasattr(self, 'enabled_topics'):
            self.enabled_topics = set()


@dataclass
class MessageQueue:
    """Message queue for offline users."""
    user_id: str
    messages: List[WebSocketMessageEnvelope] = field(default_factory=list)
    max_messages: int = 100
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def add_message(self, message: WebSocketMessageEnvelope) -> None:
        """Add message to queue."""
        self.messages.append(message)
        if len(self.messages) > self.max_messages:
            self.messages.pop(0)  # Remove oldest message
    
    def get_messages(self, limit: Optional[int] = None) -> List[WebSocketMessageEnvelope]:
        """Get queued messages."""
        if limit:
            return self.messages[-limit:]
        return self.messages.copy()
    
    def clear(self) -> None:
        """Clear all messages."""
        self.messages.clear()


@dataclass
class WebSocketMetrics:
    """WebSocket metrics."""
    active_connections: int = 0
    total_connections: int = 0
    messages_sent: int = 0
    messages_received: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0
    errors: int = 0
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    def increment_connection(self) -> None:
        """Increment connection counters."""
        self.active_connections += 1
        self.total_connections += 1
        self.last_updated = datetime.utcnow()
    
    def decrement_connection(self) -> None:
        """Decrement active connections."""
        self.active_connections = max(0, self.active_connections - 1)
        self.last_updated = datetime.utcnow()
    
    def increment_message_sent(self, size_bytes: int = 0) -> None:
        """Increment sent message counters."""
        self.messages_sent += 1
        self.bytes_sent += size_bytes
        self.last_updated = datetime.utcnow()
    
    def increment_message_received(self, size_bytes: int = 0) -> None:
        """Increment received message counters."""
        self.messages_received += 1
        self.bytes_received += size_bytes
        self.last_updated = datetime.utcnow()
    
    def increment_error(self) -> None:
        """Increment error counter."""
        self.errors += 1
        self.last_updated = datetime.utcnow()


@dataclass
class WebSocketError:
    """WebSocket error information."""
    error_id: str
    connection_id: Optional[str]
    error_type: str
    message: str
    timestamp: datetime
    stack_trace: Optional[str] = None
    user_id: Optional[str] = None
    
    @classmethod
    def create(
        cls,
        error_type: str,
        message: str,
        connection_id: Optional[str] = None,
        user_id: Optional[str] = None,
        stack_trace: Optional[str] = None
    ) -> 'WebSocketError':
        """Create WebSocket error."""
        return cls(
            error_id=str(uuid4()),
            connection_id=connection_id,
            error_type=error_type,
            message=message,
            timestamp=datetime.utcnow(),
            stack_trace=stack_trace,
            user_id=user_id
        )
