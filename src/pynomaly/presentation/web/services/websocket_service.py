"""WebSocket service implementation using pure models."""

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Set
from uuid import uuid4

from ..models.websocket_models import (
    BroadcastMessage,
    ConnectionRegistry,
    ConnectionStatus,
    MessageQueue,
    MessageType,
    NotificationPreferences,
    Subscription,
    SubscriptionTopic,
    WebSocketConfig,
    WebSocketError,
    WebSocketMessageEnvelope,
    WebSocketMetrics,
)

logger = logging.getLogger(__name__)


class WebSocketService:
    """WebSocket service managing connections and message broadcasting."""
    
    def __init__(self, config: WebSocketConfig):
        """Initialize WebSocket service.
        
        Args:
            config: WebSocket configuration
        """
        self.config = config
        self.connections: Dict[str, ConnectionRegistry] = {}
        self.user_connections: Dict[str, Set[str]] = {}
        self.subscriptions: Dict[SubscriptionTopic, Set[str]] = {}
        self.message_queues: Dict[str, MessageQueue] = {}
        self.notification_preferences: Dict[str, NotificationPreferences] = {}
        self.metrics = WebSocketMetrics()
        
        # Initialize subscription topics
        for topic in SubscriptionTopic:
            self.subscriptions[topic] = set()
    
    def register_connection(
        self,
        connection_id: str,
        user_id: Optional[str] = None,
        client_info: Optional[Dict[str, str]] = None
    ) -> ConnectionRegistry:
        """Register a new WebSocket connection.
        
        Args:
            connection_id: Unique connection identifier
            user_id: Optional user ID for authenticated connections
            client_info: Client information (IP, user agent, etc.)
            
        Returns:
            Connection registry entry
        """
        now = datetime.utcnow()
        
        connection = ConnectionRegistry(
            connection_id=connection_id,
            user_id=user_id,
            client_info=client_info or {},
            subscriptions=set(),
            connected_at=now,
            last_activity=now,
            status=ConnectionStatus.CONNECTED
        )
        
        self.connections[connection_id] = connection
        
        if user_id:
            if user_id not in self.user_connections:
                self.user_connections[user_id] = set()
            self.user_connections[user_id].add(connection_id)
            
            # Send queued messages if any
            self._deliver_queued_messages(user_id, connection_id)
        
        self.metrics.increment_connection()
        logger.info(f"Registered WebSocket connection {connection_id} for user {user_id}")
        
        return connection
    
    def unregister_connection(self, connection_id: str) -> bool:
        """Unregister a WebSocket connection.
        
        Args:
            connection_id: Connection identifier
            
        Returns:
            True if connection was found and removed
        """
        if connection_id not in self.connections:
            return False
        
        connection = self.connections[connection_id]
        
        # Remove from subscriptions
        for topic_connections in self.subscriptions.values():
            topic_connections.discard(connection_id)
        
        # Remove from user connections
        if connection.user_id and connection.user_id in self.user_connections:
            self.user_connections[connection.user_id].discard(connection_id)
            if not self.user_connections[connection.user_id]:
                del self.user_connections[connection.user_id]
        
        # Remove connection
        del self.connections[connection_id]
        
        self.metrics.decrement_connection()
        logger.info(f"Unregistered WebSocket connection {connection_id}")
        
        return True
    
    def subscribe_to_topic(
        self,
        connection_id: str,
        topic: SubscriptionTopic,
        filters: Optional[Dict] = None
    ) -> bool:
        """Subscribe connection to a topic.
        
        Args:
            connection_id: Connection identifier
            topic: Subscription topic
            filters: Optional filters for the subscription
            
        Returns:
            True if subscription was successful
        """
        if connection_id not in self.connections:
            return False
        
        connection = self.connections[connection_id]
        connection.subscriptions.add(topic)
        self.subscriptions[topic].add(connection_id)
        
        logger.debug(f"Connection {connection_id} subscribed to {topic.value}")
        return True
    
    def unsubscribe_from_topic(
        self,
        connection_id: str,
        topic: SubscriptionTopic
    ) -> bool:
        """Unsubscribe connection from a topic.
        
        Args:
            connection_id: Connection identifier
            topic: Subscription topic
            
        Returns:
            True if unsubscription was successful
        """
        if connection_id not in self.connections:
            return False
        
        connection = self.connections[connection_id]
        connection.subscriptions.discard(topic)
        self.subscriptions[topic].discard(connection_id)
        
        logger.debug(f"Connection {connection_id} unsubscribed from {topic.value}")
        return True
    
    def send_message_to_connection(
        self,
        connection_id: str,
        message: WebSocketMessageEnvelope
    ) -> bool:
        """Send message to a specific connection.
        
        Args:
            connection_id: Target connection ID
            message: Message to send
            
        Returns:
            True if message was sent successfully
        """
        if connection_id not in self.connections:
            logger.warning(f"Cannot send message: connection {connection_id} not found")
            return False
        
        try:
            # This is where you would actually send the message via WebSocket
            # For now, just log it and update metrics
            logger.debug(f"Sending message to {connection_id}: {message.type.value}")
            
            # Update connection activity
            self.connections[connection_id].last_activity = datetime.utcnow()
            
            # Update metrics
            message_size = len(json.dumps(message.data))
            self.metrics.increment_message_sent(message_size)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to send message to {connection_id}: {e}")
            self.metrics.increment_error()
            return False
    
    def broadcast_message(self, broadcast: BroadcastMessage) -> int:
        """Broadcast message to multiple connections.
        
        Args:
            broadcast: Broadcast configuration
            
        Returns:
            Number of connections the message was sent to
        """
        target_connections = set()
        
        # Determine target connections
        if broadcast.target_connections:
            target_connections.update(broadcast.target_connections)
        
        if broadcast.target_users:
            for user_id in broadcast.target_users:
                if user_id in self.user_connections:
                    target_connections.update(self.user_connections[user_id])
        
        if broadcast.target_topics:
            for topic in broadcast.target_topics:
                if topic in self.subscriptions:
                    target_connections.update(self.subscriptions[topic])
        
        # Remove excluded connections
        if broadcast.exclude_connections:
            target_connections -= broadcast.exclude_connections
        
        # Send to target connections
        sent_count = 0
        for connection_id in target_connections:
            if self.send_message_to_connection(connection_id, broadcast.message):
                sent_count += 1
        
        logger.info(f"Broadcast message sent to {sent_count} connections")
        return sent_count
    
    def queue_message_for_user(
        self,
        user_id: str,
        message: WebSocketMessageEnvelope
    ) -> None:
        """Queue message for offline user.
        
        Args:
            user_id: Target user ID
            message: Message to queue
        """
        if user_id not in self.message_queues:
            self.message_queues[user_id] = MessageQueue(user_id=user_id)
        
        self.message_queues[user_id].add_message(message)
        logger.debug(f"Queued message for user {user_id}")
    
    def _deliver_queued_messages(
        self,
        user_id: str,
        connection_id: str
    ) -> None:
        """Deliver queued messages to newly connected user.
        
        Args:
            user_id: User ID
            connection_id: Connection ID
        """
        if user_id not in self.message_queues:
            return
        
        queue = self.message_queues[user_id]
        messages = queue.get_messages()
        
        for message in messages:
            self.send_message_to_connection(connection_id, message)
        
        # Clear the queue after delivery
        queue.clear()
        
        logger.info(f"Delivered {len(messages)} queued messages to user {user_id}")
    
    def send_notification(
        self,
        user_id: str,
        notification_type: str,
        title: str,
        message: str,
        data: Optional[Dict] = None
    ) -> bool:
        """Send notification to user.
        
        Args:
            user_id: Target user ID
            notification_type: Type of notification
            title: Notification title
            message: Notification message
            data: Additional notification data
            
        Returns:
            True if notification was sent or queued
        """
        envelope = WebSocketMessageEnvelope(
            message_id=str(uuid4()),
            type=MessageType.NOTIFICATION,
            topic=SubscriptionTopic.USER_NOTIFICATIONS,
            data={
                "type": notification_type,
                "title": title,
                "message": message,
                "data": data or {}
            },
            timestamp=datetime.utcnow(),
            recipient_ids={user_id}
        )
        
        # Try to send immediately if user is online
        if user_id in self.user_connections:
            broadcast = BroadcastMessage(
                message=envelope,
                target_users={user_id}
            )
            sent_count = self.broadcast_message(broadcast)
            return sent_count > 0
        else:
            # Queue for offline user
            self.queue_message_for_user(user_id, envelope)
            return True
    
    def get_active_connections(self) -> List[ConnectionRegistry]:
        """Get list of active connections.
        
        Returns:
            List of active connection registries
        """
        return [
            conn for conn in self.connections.values()
            if conn.status == ConnectionStatus.CONNECTED
        ]
    
    def get_user_connections(self, user_id: str) -> List[ConnectionRegistry]:
        """Get connections for a specific user.
        
        Args:
            user_id: User ID
            
        Returns:
            List of user's connections
        """
        if user_id not in self.user_connections:
            return []
        
        return [
            self.connections[conn_id]
            for conn_id in self.user_connections[user_id]
            if conn_id in self.connections
        ]
    
    def get_metrics(self) -> WebSocketMetrics:
        """Get WebSocket metrics.
        
        Returns:
            Current metrics
        """
        # Update active connections count
        self.metrics.active_connections = len([
            conn for conn in self.connections.values()
            if conn.status == ConnectionStatus.CONNECTED
        ])
        
        return self.metrics
    
    def cleanup_stale_connections(self, timeout_seconds: int = 300) -> int:
        """Clean up stale connections.
        
        Args:
            timeout_seconds: Connection timeout in seconds
            
        Returns:
            Number of connections cleaned up
        """
        now = datetime.utcnow()
        stale_connections = []
        
        for connection_id, connection in self.connections.items():
            time_since_activity = (now - connection.last_activity).total_seconds()
            if time_since_activity > timeout_seconds:
                stale_connections.append(connection_id)
        
        for connection_id in stale_connections:
            self.unregister_connection(connection_id)
        
        if stale_connections:
            logger.info(f"Cleaned up {len(stale_connections)} stale connections")
        
        return len(stale_connections)
