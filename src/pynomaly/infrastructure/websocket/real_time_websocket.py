"""Real-time WebSocket infrastructure for live dashboard updates.

This module provides WebSocket infrastructure for real-time monitoring dashboard
updates, including connection management, message broadcasting, and data streaming.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from ...application.services.real_time_monitoring_service import (
    RealTimeMonitoringService,
)
from ...infrastructure.config.feature_flags import require_feature

logger = logging.getLogger(__name__)


class WebSocketMessage(BaseModel):
    """WebSocket message model."""

    type: str
    timestamp: str
    data: dict[str, Any]
    client_id: str | None = None


class ConnectionInfo(BaseModel):
    """WebSocket connection information."""

    connection_id: str
    user_id: UUID | None = None
    session_id: str | None = None
    connected_at: datetime
    last_ping: datetime
    subscriptions: set[str] = set()
    metadata: dict[str, Any] = {}


class RealTimeWebSocketManager:
    """Manager for real-time WebSocket connections."""

    def __init__(self, monitoring_service: RealTimeMonitoringService | None = None):
        """Initialize WebSocket manager.

        Args:
            monitoring_service: Real-time monitoring service instance
        """
        self.monitoring_service = monitoring_service
        self.active_connections: dict[str, WebSocket] = {}
        self.connection_info: dict[str, ConnectionInfo] = {}
        self.subscription_groups: dict[str, set[str]] = {}  # topic -> connection_ids

        # Connection limits and cleanup
        self.max_connections = 1000
        self.ping_interval = 30  # seconds
        self.connection_timeout = 300  # 5 minutes

        # Background tasks
        self._background_tasks: list[asyncio.Task] = []
        self._is_running = False

    async def start(self) -> None:
        """Start the WebSocket manager."""
        if self._is_running:
            return

        self._is_running = True

        # Start background tasks
        self._background_tasks = [
            asyncio.create_task(self._ping_connections_task()),
            asyncio.create_task(self._cleanup_connections_task()),
            asyncio.create_task(self._broadcast_metrics_task()),
        ]

        logger.info("Real-time WebSocket manager started")

    async def stop(self) -> None:
        """Stop the WebSocket manager."""
        self._is_running = False

        # Close all connections
        for connection_id in list(self.active_connections.keys()):
            await self.disconnect(connection_id)

        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()

        await asyncio.gather(*self._background_tasks, return_exceptions=True)
        self._background_tasks.clear()

        logger.info("Real-time WebSocket manager stopped")

    @require_feature("real_time_monitoring")
    async def connect(
        self,
        websocket: WebSocket,
        user_id: UUID | None = None,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Connect a new WebSocket client.

        Args:
            websocket: WebSocket connection
            user_id: User ID (optional)
            session_id: Session ID (optional)
            metadata: Additional connection metadata

        Returns:
            Connection ID
        """
        # Check connection limit
        if len(self.active_connections) >= self.max_connections:
            await websocket.close(code=4008, reason="Connection limit exceeded")
            raise Exception("Connection limit exceeded")

        await websocket.accept()

        connection_id = str(uuid4())
        connection_info = ConnectionInfo(
            connection_id=connection_id,
            user_id=user_id,
            session_id=session_id,
            connected_at=datetime.utcnow(),
            last_ping=datetime.utcnow(),
            metadata=metadata or {},
        )

        self.active_connections[connection_id] = websocket
        self.connection_info[connection_id] = connection_info

        # Send connection confirmation
        await self._send_to_connection(
            connection_id,
            {
                "type": "connection_established",
                "connection_id": connection_id,
                "timestamp": datetime.utcnow().isoformat(),
            },
        )

        # Register with monitoring service
        if self.monitoring_service:
            await self.monitoring_service.add_subscriber(
                lambda message: self._handle_monitoring_event(connection_id, message)
            )

        logger.info(f"WebSocket client connected: {connection_id}")
        return connection_id

    async def disconnect(self, connection_id: str) -> None:
        """Disconnect a WebSocket client.

        Args:
            connection_id: Connection ID to disconnect
        """
        if connection_id not in self.active_connections:
            return

        # Remove from subscription groups
        for topic_connections in self.subscription_groups.values():
            topic_connections.discard(connection_id)

        # Close WebSocket connection
        websocket = self.active_connections[connection_id]
        try:
            await websocket.close()
        except Exception as e:
            logger.warning(f"Error closing WebSocket connection {connection_id}: {e}")

        # Clean up
        del self.active_connections[connection_id]
        del self.connection_info[connection_id]

        # Unregister from monitoring service
        if self.monitoring_service:
            await self.monitoring_service.remove_subscriber(
                lambda message: self._handle_monitoring_event(connection_id, message)
            )

        logger.info(f"WebSocket client disconnected: {connection_id}")

    @require_feature("real_time_monitoring")
    async def subscribe(self, connection_id: str, topics: list[str]) -> bool:
        """Subscribe connection to topics.

        Args:
            connection_id: Connection ID
            topics: List of topics to subscribe to

        Returns:
            True if successful, False if connection not found
        """
        if connection_id not in self.connection_info:
            return False

        connection_info = self.connection_info[connection_id]

        for topic in topics:
            connection_info.subscriptions.add(topic)

            if topic not in self.subscription_groups:
                self.subscription_groups[topic] = set()

            self.subscription_groups[topic].add(connection_id)

        # Send subscription confirmation
        await self._send_to_connection(
            connection_id,
            {
                "type": "subscription_confirmed",
                "topics": topics,
                "timestamp": datetime.utcnow().isoformat(),
            },
        )

        logger.debug(f"Connection {connection_id} subscribed to topics: {topics}")
        return True

    @require_feature("real_time_monitoring")
    async def unsubscribe(self, connection_id: str, topics: list[str]) -> bool:
        """Unsubscribe connection from topics.

        Args:
            connection_id: Connection ID
            topics: List of topics to unsubscribe from

        Returns:
            True if successful, False if connection not found
        """
        if connection_id not in self.connection_info:
            return False

        connection_info = self.connection_info[connection_id]

        for topic in topics:
            connection_info.subscriptions.discard(topic)

            if topic in self.subscription_groups:
                self.subscription_groups[topic].discard(connection_id)

                # Clean up empty subscription groups
                if not self.subscription_groups[topic]:
                    del self.subscription_groups[topic]

        # Send unsubscription confirmation
        await self._send_to_connection(
            connection_id,
            {
                "type": "unsubscription_confirmed",
                "topics": topics,
                "timestamp": datetime.utcnow().isoformat(),
            },
        )

        logger.debug(f"Connection {connection_id} unsubscribed from topics: {topics}")
        return True

    @require_feature("real_time_monitoring")
    async def broadcast_to_topic(self, topic: str, message: dict[str, Any]) -> int:
        """Broadcast message to all subscribers of a topic.

        Args:
            topic: Topic to broadcast to
            message: Message to broadcast

        Returns:
            Number of connections message was sent to
        """
        if topic not in self.subscription_groups:
            return 0

        connection_ids = list(self.subscription_groups[topic])
        successful_sends = 0
        failed_connections = []

        # Prepare message
        websocket_message = {
            "type": "topic_message",
            "topic": topic,
            "timestamp": datetime.utcnow().isoformat(),
            "data": message,
        }

        # Send to all subscribers
        for connection_id in connection_ids:
            try:
                await self._send_to_connection(connection_id, websocket_message)
                successful_sends += 1
            except Exception as e:
                logger.warning(f"Failed to send to connection {connection_id}: {e}")
                failed_connections.append(connection_id)

        # Clean up failed connections
        for connection_id in failed_connections:
            await self.disconnect(connection_id)

        return successful_sends

    @require_feature("real_time_monitoring")
    async def send_to_user(self, user_id: UUID, message: dict[str, Any]) -> int:
        """Send message to all connections for a specific user.

        Args:
            user_id: User ID
            message: Message to send

        Returns:
            Number of connections message was sent to
        """
        user_connections = [
            connection_id
            for connection_id, info in self.connection_info.items()
            if info.user_id == user_id
        ]

        successful_sends = 0

        for connection_id in user_connections:
            try:
                await self._send_to_connection(
                    connection_id,
                    {
                        "type": "user_message",
                        "timestamp": datetime.utcnow().isoformat(),
                        "data": message,
                    },
                )
                successful_sends += 1
            except Exception as e:
                logger.warning(
                    f"Failed to send to user connection {connection_id}: {e}"
                )
                await self.disconnect(connection_id)

        return successful_sends

    async def handle_message(self, connection_id: str, message: str) -> None:
        """Handle incoming WebSocket message.

        Args:
            connection_id: Connection ID
            message: JSON message string
        """
        try:
            data = json.loads(message)
            message_type = data.get("type")

            if message_type == "ping":
                await self._handle_ping(connection_id)
            elif message_type == "subscribe":
                topics = data.get("topics", [])
                await self.subscribe(connection_id, topics)
            elif message_type == "unsubscribe":
                topics = data.get("topics", [])
                await self.unsubscribe(connection_id, topics)
            elif message_type == "get_dashboard_data":
                await self._handle_dashboard_data_request(connection_id)
            else:
                logger.warning(
                    f"Unknown message type from {connection_id}: {message_type}"
                )

        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON message from {connection_id}: {message}")
            await self._send_error(connection_id, "Invalid JSON message")
        except Exception as e:
            logger.error(f"Error handling message from {connection_id}: {e}")
            await self._send_error(connection_id, "Internal server error")

    async def get_connection_stats(self) -> dict[str, Any]:
        """Get connection statistics.

        Returns:
            Connection statistics
        """
        now = datetime.utcnow()

        # Calculate connection duration statistics
        durations = [
            (now - info.connected_at).total_seconds()
            for info in self.connection_info.values()
        ]

        avg_duration = sum(durations) / len(durations) if durations else 0

        # Group by user
        user_connections = {}
        anonymous_connections = 0

        for info in self.connection_info.values():
            if info.user_id:
                if info.user_id not in user_connections:
                    user_connections[info.user_id] = 0
                user_connections[info.user_id] += 1
            else:
                anonymous_connections += 1

        return {
            "total_connections": len(self.active_connections),
            "authenticated_users": len(user_connections),
            "anonymous_connections": anonymous_connections,
            "avg_connection_duration_seconds": avg_duration,
            "subscription_topics": list(self.subscription_groups.keys()),
            "topic_subscriber_counts": {
                topic: len(connections)
                for topic, connections in self.subscription_groups.items()
            },
        }

    async def _send_to_connection(
        self, connection_id: str, message: dict[str, Any]
    ) -> None:
        """Send message to specific connection.

        Args:
            connection_id: Connection ID
            message: Message to send
        """
        if connection_id not in self.active_connections:
            raise Exception(f"Connection {connection_id} not found")

        websocket = self.active_connections[connection_id]

        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.warning(f"Failed to send message to {connection_id}: {e}")
            await self.disconnect(connection_id)
            raise

    async def _send_error(self, connection_id: str, error_message: str) -> None:
        """Send error message to connection.

        Args:
            connection_id: Connection ID
            error_message: Error message
        """
        try:
            await self._send_to_connection(
                connection_id,
                {
                    "type": "error",
                    "timestamp": datetime.utcnow().isoformat(),
                    "error": error_message,
                },
            )
        except Exception:
            # Connection already failed, ignore
            pass

    async def _handle_ping(self, connection_id: str) -> None:
        """Handle ping message from client.

        Args:
            connection_id: Connection ID
        """
        if connection_id in self.connection_info:
            self.connection_info[connection_id].last_ping = datetime.utcnow()

        await self._send_to_connection(
            connection_id,
            {
                "type": "pong",
                "timestamp": datetime.utcnow().isoformat(),
            },
        )

    async def _handle_dashboard_data_request(self, connection_id: str) -> None:
        """Handle request for dashboard data.

        Args:
            connection_id: Connection ID
        """
        if not self.monitoring_service:
            await self._send_error(connection_id, "Monitoring service not available")
            return

        try:
            dashboard_data = (
                await self.monitoring_service.get_real_time_dashboard_data()
            )

            await self._send_to_connection(
                connection_id,
                {
                    "type": "dashboard_data",
                    "timestamp": datetime.utcnow().isoformat(),
                    "data": dashboard_data,
                },
            )
        except Exception as e:
            logger.error(f"Error getting dashboard data for {connection_id}: {e}")
            await self._send_error(connection_id, "Failed to get dashboard data")

    async def _handle_monitoring_event(
        self, connection_id: str, message: dict[str, Any]
    ) -> None:
        """Handle monitoring event for specific connection.

        Args:
            connection_id: Connection ID
            message: Monitoring event message
        """
        if connection_id not in self.active_connections:
            return

        try:
            await self._send_to_connection(
                connection_id,
                {
                    "type": "monitoring_event",
                    "timestamp": datetime.utcnow().isoformat(),
                    "data": message,
                },
            )
        except Exception as e:
            logger.warning(f"Failed to send monitoring event to {connection_id}: {e}")

    async def _ping_connections_task(self) -> None:
        """Background task to ping connections."""
        while self._is_running:
            try:
                now = datetime.utcnow()

                for connection_id in list(self.active_connections.keys()):
                    try:
                        await self._send_to_connection(
                            connection_id,
                            {
                                "type": "server_ping",
                                "timestamp": now.isoformat(),
                            },
                        )
                    except Exception:
                        # Connection failed, will be cleaned up
                        pass

                await asyncio.sleep(self.ping_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in ping connections task: {e}")
                await asyncio.sleep(self.ping_interval)

    async def _cleanup_connections_task(self) -> None:
        """Background task to clean up stale connections."""
        while self._is_running:
            try:
                now = datetime.utcnow()
                timeout_threshold = now - timedelta(seconds=self.connection_timeout)

                stale_connections = [
                    connection_id
                    for connection_id, info in self.connection_info.items()
                    if info.last_ping < timeout_threshold
                ]

                for connection_id in stale_connections:
                    logger.info(f"Cleaning up stale connection: {connection_id}")
                    await self.disconnect(connection_id)

                await asyncio.sleep(60)  # Check every minute

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup connections task: {e}")
                await asyncio.sleep(60)

    async def _broadcast_metrics_task(self) -> None:
        """Background task to broadcast periodic metrics."""
        while self._is_running:
            try:
                if (
                    self.monitoring_service
                    and "dashboard_metrics" in self.subscription_groups
                ):
                    dashboard_data = (
                        await self.monitoring_service.get_real_time_dashboard_data()
                    )
                    await self.broadcast_to_topic("dashboard_metrics", dashboard_data)

                await asyncio.sleep(5)  # Broadcast every 5 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in broadcast metrics task: {e}")
                await asyncio.sleep(5)


# WebSocket endpoint handler
async def websocket_endpoint_handler(
    websocket: WebSocket,
    websocket_manager: RealTimeWebSocketManager,
    user_id: UUID | None = None,
    session_id: str | None = None,
) -> None:
    """Handle WebSocket endpoint connection.

    Args:
        websocket: WebSocket connection
        websocket_manager: WebSocket manager instance
        user_id: User ID (optional)
        session_id: Session ID (optional)
    """
    connection_id = None

    try:
        # Connect client
        connection_id = await websocket_manager.connect(
            websocket, user_id=user_id, session_id=session_id
        )

        # Handle messages
        while True:
            try:
                message = await websocket.receive_text()
                await websocket_manager.handle_message(connection_id, message)
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"Error handling WebSocket message: {e}")
                break

    except Exception as e:
        logger.error(f"WebSocket connection error: {e}")
    finally:
        if connection_id:
            await websocket_manager.disconnect(connection_id)
