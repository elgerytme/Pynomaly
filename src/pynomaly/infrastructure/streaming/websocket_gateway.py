"""WebSocket gateway for real-time dashboard updates."""

import asyncio
import json
import logging
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from .real_time_anomaly_pipeline import StreamingMetrics

logger = logging.getLogger(__name__)


class RealTimeMetrics(BaseModel):
    """Real-time metrics for dashboard updates."""

    dashboard_id: str
    session_id: str
    metrics: StreamingMetrics
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    uptime_seconds: float = 0.0
    status: str = "active"
    metadata: dict[str, Any] = Field(default_factory=dict)


class WebSocketConnection:
    """Represents a WebSocket connection."""

    def __init__(self, websocket, connection_id: str, dashboard_id: str):
        """Initialize WebSocket connection."""
        self.websocket = websocket
        self.connection_id = connection_id
        self.dashboard_id = dashboard_id
        self.connected_at = datetime.utcnow()
        self.last_heartbeat = datetime.utcnow()
        self.message_queue = deque()
        self.is_active = True


class WebSocketGateway:
    """WebSocket gateway for real-time dashboard updates."""

    def __init__(
        self,
        update_interval: int = 5,
        heartbeat_interval: int = 30,
        connection_timeout: int = 300,
        max_connections_per_dashboard: int = 100,
        max_message_queue_size: int = 1000,
    ):
        """Initialize WebSocket gateway.

        Args:
            update_interval: Update interval in seconds
            heartbeat_interval: Heartbeat interval in seconds
            connection_timeout: Connection timeout in seconds
            max_connections_per_dashboard: Maximum connections per dashboard
            max_message_queue_size: Maximum message queue size per connection
        """
        self.update_interval = update_interval
        self.heartbeat_interval = heartbeat_interval
        self.connection_timeout = connection_timeout
        self.max_connections_per_dashboard = max_connections_per_dashboard
        self.max_message_queue_size = max_message_queue_size

        # Connection management
        self.connections: dict[str, WebSocketConnection] = {}
        self.dashboard_connections: dict[str, set[str]] = defaultdict(set)
        self.latest_metrics: dict[str, RealTimeMetrics] = {}

        # Background tasks
        self.heartbeat_task = None
        self.cleanup_task = None
        self.update_task = None
        self.is_running = False

        # Statistics
        self.stats = {
            "total_connections": 0,
            "active_connections": 0,
            "messages_sent": 0,
            "messages_dropped": 0,
            "errors": 0,
        }

    async def start(self) -> None:
        """Start the WebSocket gateway."""
        if self.is_running:
            logger.warning("WebSocket gateway is already running")
            return

        logger.info("Starting WebSocket gateway")
        self.is_running = True

        # Start background tasks
        self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        self.update_task = asyncio.create_task(self._update_loop())

        logger.info("WebSocket gateway started successfully")

    async def stop(self) -> None:
        """Stop the WebSocket gateway."""
        if not self.is_running:
            return

        logger.info("Stopping WebSocket gateway")
        self.is_running = False

        # Cancel background tasks
        for task in [self.heartbeat_task, self.cleanup_task, self.update_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Close all connections
        for connection in list(self.connections.values()):
            await self._close_connection(connection.connection_id)

        logger.info("WebSocket gateway stopped")

    async def connect_dashboard(self, websocket, dashboard_id: str) -> str:
        """Connect to a dashboard.

        Args:
            websocket: WebSocket connection
            dashboard_id: Dashboard ID

        Returns:
            Connection ID

        Raises:
            ValueError: If dashboard has reached maximum connections
        """
        # Check connection limits
        if len(self.dashboard_connections[dashboard_id]) >= self.max_connections_per_dashboard:
            raise ValueError(f"Dashboard {dashboard_id} has reached maximum connections")

        # Create connection
        connection_id = str(uuid4())
        connection = WebSocketConnection(websocket, connection_id, dashboard_id)

        # Accept WebSocket connection
        await websocket.accept()

        # Store connection
        self.connections[connection_id] = connection
        self.dashboard_connections[dashboard_id].add(connection_id)

        # Update statistics
        self.stats["total_connections"] += 1
        self.stats["active_connections"] += 1

        logger.info(f"Connected to dashboard {dashboard_id}: {connection_id}")

        # Send welcome message
        await self._send_to_connection(connection_id, {
            "type": "welcome",
            "connection_id": connection_id,
            "dashboard_id": dashboard_id,
            "timestamp": datetime.utcnow().isoformat(),
        })

        # Send latest metrics if available
        if dashboard_id in self.latest_metrics:
            await self._send_metrics_to_connection(connection_id, self.latest_metrics[dashboard_id])

        return connection_id

    async def disconnect_dashboard(self, connection_id: str) -> None:
        """Disconnect from a dashboard.

        Args:
            connection_id: Connection ID
        """
        await self._close_connection(connection_id)

    async def update_dashboard_metrics(self, dashboard_id: str, metrics: RealTimeMetrics) -> None:
        """Update dashboard metrics.

        Args:
            dashboard_id: Dashboard ID
            metrics: Real-time metrics
        """
        # Store latest metrics
        self.latest_metrics[dashboard_id] = metrics

        # Send to all connections for this dashboard
        connection_ids = self.dashboard_connections.get(dashboard_id, set()).copy()
        for connection_id in connection_ids:
            await self._send_metrics_to_connection(connection_id, metrics)

    async def send_alert(self, dashboard_id: str, alert_data: dict[str, Any]) -> None:
        """Send alert to dashboard connections.

        Args:
            dashboard_id: Dashboard ID
            alert_data: Alert data
        """
        message = {
            "type": "alert",
            "dashboard_id": dashboard_id,
            "alert": alert_data,
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Send to all connections for this dashboard
        connection_ids = self.dashboard_connections.get(dashboard_id, set()).copy()
        for connection_id in connection_ids:
            await self._send_to_connection(connection_id, message)

    async def handle_client_message(self, connection_id: str, message: dict[str, Any]) -> None:
        """Handle message from client.

        Args:
            connection_id: Connection ID
            message: Message from client
        """
        connection = self.connections.get(connection_id)
        if not connection:
            return

        message_type = message.get("type")

        if message_type == "heartbeat":
            # Update last heartbeat
            connection.last_heartbeat = datetime.utcnow()

            # Send heartbeat response
            await self._send_to_connection(connection_id, {
                "type": "heartbeat_response",
                "timestamp": datetime.utcnow().isoformat(),
            })

        elif message_type == "subscribe":
            # Handle subscription to specific metrics
            pass

        elif message_type == "unsubscribe":
            # Handle unsubscription from specific metrics
            pass

        else:
            logger.warning(f"Unknown message type: {message_type}")

    async def _send_metrics_to_connection(self, connection_id: str, metrics: RealTimeMetrics) -> None:
        """Send metrics to a connection.

        Args:
            connection_id: Connection ID
            metrics: Real-time metrics
        """
        message = {
            "type": "metrics_update",
            "dashboard_id": metrics.dashboard_id,
            "session_id": metrics.session_id,
            "metrics": metrics.metrics.dict(),
            "timestamp": metrics.timestamp.isoformat(),
            "uptime_seconds": metrics.uptime_seconds,
            "status": metrics.status,
            "metadata": metrics.metadata,
        }

        await self._send_to_connection(connection_id, message)

    async def _send_to_connection(self, connection_id: str, message: dict[str, Any]) -> None:
        """Send message to a connection with back-pressure handling.

        Args:
            connection_id: Connection ID
            message: Message to send
        """
        connection = self.connections.get(connection_id)
        if not connection or not connection.is_active:
            return

        try:
            # Check queue size for back-pressure
            if len(connection.message_queue) >= self.max_message_queue_size:
                # Drop oldest message
                connection.message_queue.popleft()
                self.stats["messages_dropped"] += 1
                logger.warning(f"Dropped message for connection {connection_id} due to back-pressure")

            # Add to queue
            connection.message_queue.append(message)

            # Send message
            await connection.websocket.send_text(json.dumps(message))
            self.stats["messages_sent"] += 1

        except Exception as e:
            logger.error(f"Error sending message to connection {connection_id}: {e}")
            self.stats["errors"] += 1
            await self._close_connection(connection_id)

    async def _close_connection(self, connection_id: str) -> None:
        """Close a connection.

        Args:
            connection_id: Connection ID
        """
        connection = self.connections.get(connection_id)
        if not connection:
            return

        # Remove from dashboard connections
        self.dashboard_connections[connection.dashboard_id].discard(connection_id)

        # Remove from connections
        del self.connections[connection_id]

        # Update statistics
        self.stats["active_connections"] -= 1

        # Close WebSocket
        try:
            await connection.websocket.close()
        except Exception as e:
            logger.error(f"Error closing WebSocket for connection {connection_id}: {e}")

        logger.info(f"Closed connection {connection_id}")

    async def _heartbeat_loop(self) -> None:
        """Background task for sending heartbeat messages."""
        try:
            while self.is_running:
                await asyncio.sleep(self.heartbeat_interval)

                # Send heartbeat to all connections
                for connection_id in list(self.connections.keys()):
                    await self._send_to_connection(connection_id, {
                        "type": "heartbeat",
                        "timestamp": datetime.utcnow().isoformat(),
                    })

        except asyncio.CancelledError:
            pass

    async def _cleanup_loop(self) -> None:
        """Background task for cleaning up stale connections."""
        try:
            while self.is_running:
                await asyncio.sleep(60)  # Check every minute

                current_time = datetime.utcnow()
                stale_connections = []

                # Find stale connections
                for connection_id, connection in self.connections.items():
                    if (current_time - connection.last_heartbeat).total_seconds() > self.connection_timeout:
                        stale_connections.append(connection_id)

                # Close stale connections
                for connection_id in stale_connections:
                    logger.info(f"Closing stale connection {connection_id}")
                    await self._close_connection(connection_id)

        except asyncio.CancelledError:
            pass

    async def _update_loop(self) -> None:
        """Background task for periodic updates."""
        try:
            while self.is_running:
                await asyncio.sleep(self.update_interval)

                # Send periodic updates to all connections
                for dashboard_id, metrics in self.latest_metrics.items():
                    connection_ids = self.dashboard_connections.get(dashboard_id, set()).copy()
                    for connection_id in connection_ids:
                        await self._send_metrics_to_connection(connection_id, metrics)

        except asyncio.CancelledError:
            pass

    def get_stats(self) -> dict[str, Any]:
        """Get gateway statistics.

        Returns:
            Gateway statistics
        """
        return {
            **self.stats,
            "dashboard_count": len(self.dashboard_connections),
            "connections_per_dashboard": {
                dashboard_id: len(connections)
                for dashboard_id, connections in self.dashboard_connections.items()
            },
        }

    def get_connection_info(self, connection_id: str) -> dict[str, Any] | None:
        """Get connection information.

        Args:
            connection_id: Connection ID

        Returns:
            Connection information or None if not found
        """
        connection = self.connections.get(connection_id)
        if not connection:
            return None

        return {
            "connection_id": connection_id,
            "dashboard_id": connection.dashboard_id,
            "connected_at": connection.connected_at.isoformat(),
            "last_heartbeat": connection.last_heartbeat.isoformat(),
            "message_queue_size": len(connection.message_queue),
            "is_active": connection.is_active,
        }
