"""Enhanced WebSocket gateway for real-time dashboard updates with multiplexing."""

import asyncio
import json
import logging
import time
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from pynomaly.infrastructure.streaming.real_time_anomaly_pipeline import StreamingMetrics

logger = logging.getLogger(__name__)


class RealTimeMetrics(BaseModel):
    """Real-time metrics for dashboard updates."""
    
    dashboard_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metrics: StreamingMetrics
    session_id: str
    uptime_seconds: float = 0.0
    status: str = "active"
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ConnectionInfo(BaseModel):
    """Information about a WebSocket connection."""
    
    websocket: WebSocket
    dashboard_id: str
    session_id: str
    connected_at: datetime
    last_heartbeat: datetime
    is_authenticated: bool = False
    user_id: Optional[str] = None
    
    class Config:
        arbitrary_types_allowed = True


class WebSocketGateway:
    """Enhanced WebSocket gateway with dashboard multiplexing and back-pressure handling."""
    
    def __init__(self, 
                 update_interval: int = 10,
                 heartbeat_interval: int = 30,
                 connection_timeout: int = 60,
                 max_connections_per_dashboard: int = 100,
                 max_message_queue_size: int = 1000):
        """Initialize WebSocket gateway.
        
        Args:
            update_interval: Interval between metric updates (seconds)
            heartbeat_interval: Interval between heartbeat messages (seconds)
            connection_timeout: Connection timeout (seconds)
            max_connections_per_dashboard: Maximum connections per dashboard
            max_message_queue_size: Maximum message queue size per connection
        """
        self.update_interval = update_interval
        self.heartbeat_interval = heartbeat_interval
        self.connection_timeout = connection_timeout
        self.max_connections_per_dashboard = max_connections_per_dashboard
        self.max_message_queue_size = max_message_queue_size
        
        # Connection management
        self.connections: Dict[str, ConnectionInfo] = {}
        self.dashboard_connections: Dict[str, Set[str]] = defaultdict(set)
        self.connection_queues: Dict[str, asyncio.Queue] = {}
        
        # Message broadcasting
        self.broadcast_tasks: Dict[str, asyncio.Task] = {}
        
        # Metrics storage
        self.latest_metrics: Dict[str, RealTimeMetrics] = {}
        
        # Background tasks
        self.heartbeat_task: Optional[asyncio.Task] = None
        self.cleanup_task: Optional[asyncio.Task] = None
        
        # State management
        self.is_running = False
        self.stats = {
            "total_connections": 0,
            "active_connections": 0,
            "messages_sent": 0,
            "messages_dropped": 0,
            "errors": 0
        }
    
    async def start(self):
        """Start the WebSocket gateway."""
        if self.is_running:
            logger.warning("WebSocket gateway is already running")
            return
            
        self.is_running = True
        logger.info("Starting WebSocket gateway")
        
        # Start background tasks
        self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        logger.info("WebSocket gateway started successfully")
    
    async def stop(self):
        """Stop the WebSocket gateway."""
        if not self.is_running:
            return
            
        logger.info("Stopping WebSocket gateway")
        self.is_running = False
        
        # Cancel background tasks
        if self.heartbeat_task:
            self.heartbeat_task.cancel()
        if self.cleanup_task:
            self.cleanup_task.cancel()
        
        # Cancel all broadcast tasks
        for task in self.broadcast_tasks.values():
            task.cancel()
        
        # Close all connections
        for connection_id in list(self.connections.keys()):
            await self._disconnect_client(connection_id)
        
        logger.info("WebSocket gateway stopped")
    
    async def connect_dashboard(self, 
                              websocket: WebSocket, 
                              dashboard_id: str,
                              user_id: Optional[str] = None) -> str:
        """Connect a client to a dashboard channel.
        
        Args:
            websocket: WebSocket connection
            dashboard_id: Dashboard ID to connect to
            user_id: Optional user ID for authentication
            
        Returns:
            Connection ID
            
        Raises:
            ValueError: If dashboard has too many connections
        """
        # Check connection limits
        if len(self.dashboard_connections[dashboard_id]) >= self.max_connections_per_dashboard:
            raise ValueError(f"Dashboard {dashboard_id} has reached maximum connections")
        
        # Accept WebSocket connection
        await websocket.accept()
        
        # Generate connection ID
        connection_id = str(uuid4())
        session_id = str(uuid4())
        
        # Create connection info
        connection_info = ConnectionInfo(
            websocket=websocket,
            dashboard_id=dashboard_id,
            session_id=session_id,
            connected_at=datetime.utcnow(),
            last_heartbeat=datetime.utcnow(),
            is_authenticated=user_id is not None,
            user_id=user_id
        )
        
        # Store connection
        self.connections[connection_id] = connection_info
        self.dashboard_connections[dashboard_id].add(connection_id)
        
        # Create message queue for this connection
        self.connection_queues[connection_id] = asyncio.Queue(maxsize=self.max_message_queue_size)
        
        # Start broadcast task if not already running for this dashboard
        if dashboard_id not in self.broadcast_tasks:
            self.broadcast_tasks[dashboard_id] = asyncio.create_task(
                self._dashboard_broadcast_loop(dashboard_id)
            )
        
        # Update stats
        self.stats["total_connections"] += 1
        self.stats["active_connections"] = len(self.connections)
        
        logger.info(f"Client {connection_id} connected to dashboard {dashboard_id}")
        
        # Send initial connection message
        await self._send_to_connection(connection_id, {
            "type": "connection_established",
            "connection_id": connection_id,
            "dashboard_id": dashboard_id,
            "session_id": session_id,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Send latest metrics if available
        if dashboard_id in self.latest_metrics:
            await self._send_to_connection(connection_id, {
                "type": "metrics_update",
                "data": self.latest_metrics[dashboard_id].dict(),
                "timestamp": datetime.utcnow().isoformat()
            })
        
        return connection_id
    
    async def disconnect_dashboard(self, connection_id: str):
        """Disconnect a client from a dashboard channel.
        
        Args:
            connection_id: Connection ID to disconnect
        """
        await self._disconnect_client(connection_id)
    
    async def update_dashboard_metrics(self, dashboard_id: str, metrics: RealTimeMetrics):
        """Update metrics for a dashboard.
        
        Args:
            dashboard_id: Dashboard ID
            metrics: New metrics data
        """
        self.latest_metrics[dashboard_id] = metrics
        
        # Broadcast to all connections for this dashboard
        if dashboard_id in self.dashboard_connections:
            message = {
                "type": "metrics_update",
                "data": metrics.dict(),
                "timestamp": datetime.utcnow().isoformat()
            }
            await self._broadcast_to_dashboard(dashboard_id, message)
    
    async def send_alert(self, dashboard_id: str, alert_data: Dict[str, Any]):
        """Send an alert to a dashboard.
        
        Args:
            dashboard_id: Dashboard ID
            alert_data: Alert data
        """
        message = {
            "type": "alert",
            "data": alert_data,
            "timestamp": datetime.utcnow().isoformat()
        }
        await self._broadcast_to_dashboard(dashboard_id, message)
    
    async def handle_client_message(self, connection_id: str, message: Dict[str, Any]):
        """Handle incoming message from client.
        
        Args:
            connection_id: Connection ID
            message: Message data
        """
        if connection_id not in self.connections:
            logger.warning(f"Message from unknown connection {connection_id}")
            return
        
        connection = self.connections[connection_id]
        message_type = message.get("type")
        
        if message_type == "heartbeat":
            connection.last_heartbeat = datetime.utcnow()
            await self._send_to_connection(connection_id, {
                "type": "heartbeat_response",
                "timestamp": datetime.utcnow().isoformat()
            })
        
        elif message_type == "request_metrics":
            dashboard_id = connection.dashboard_id
            if dashboard_id in self.latest_metrics:
                await self._send_to_connection(connection_id, {
                    "type": "metrics_update",
                    "data": self.latest_metrics[dashboard_id].dict(),
                    "timestamp": datetime.utcnow().isoformat()
                })
        
        elif message_type == "subscribe_alerts":
            # Client wants to subscribe to alerts
            await self._send_to_connection(connection_id, {
                "type": "alert_subscription_confirmed",
                "timestamp": datetime.utcnow().isoformat()
            })
        
        else:
            logger.warning(f"Unknown message type: {message_type}")
    
    async def _dashboard_broadcast_loop(self, dashboard_id: str):
        """Background task to broadcast updates to dashboard connections."""
        try:
            while self.is_running and dashboard_id in self.dashboard_connections:
                if not self.dashboard_connections[dashboard_id]:
                    # No connections for this dashboard, break
                    break
                
                # Send periodic updates
                if dashboard_id in self.latest_metrics:
                    message = {
                        "type": "periodic_update",
                        "data": self.latest_metrics[dashboard_id].dict(),
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    await self._broadcast_to_dashboard(dashboard_id, message)
                
                await asyncio.sleep(self.update_interval)
        
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error in dashboard broadcast loop for {dashboard_id}: {e}")
        finally:
            # Clean up broadcast task
            if dashboard_id in self.broadcast_tasks:
                del self.broadcast_tasks[dashboard_id]
    
    async def _heartbeat_loop(self):
        """Background task to send heartbeat messages."""
        try:
            while self.is_running:
                current_time = datetime.utcnow()
                
                for connection_id, connection in list(self.connections.items()):
                    try:
                        # Send heartbeat
                        await self._send_to_connection(connection_id, {
                            "type": "heartbeat",
                            "timestamp": current_time.isoformat()
                        })
                    except Exception as e:
                        logger.error(f"Error sending heartbeat to {connection_id}: {e}")
                
                await asyncio.sleep(self.heartbeat_interval)
        
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error in heartbeat loop: {e}")
    
    async def _cleanup_loop(self):
        """Background task to clean up stale connections."""
        try:
            while self.is_running:
                current_time = datetime.utcnow()
                timeout_threshold = current_time - timedelta(seconds=self.connection_timeout)
                
                # Check for stale connections
                stale_connections = []
                for connection_id, connection in self.connections.items():
                    if connection.last_heartbeat < timeout_threshold:
                        stale_connections.append(connection_id)
                
                # Clean up stale connections
                for connection_id in stale_connections:
                    logger.info(f"Cleaning up stale connection {connection_id}")
                    await self._disconnect_client(connection_id)
                
                await asyncio.sleep(30)  # Check every 30 seconds
        
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error in cleanup loop: {e}")
    
    async def _broadcast_to_dashboard(self, dashboard_id: str, message: Dict[str, Any]):
        """Broadcast message to all connections for a dashboard.
        
        Args:
            dashboard_id: Dashboard ID
            message: Message to broadcast
        """
        if dashboard_id not in self.dashboard_connections:
            return
        
        connection_ids = list(self.dashboard_connections[dashboard_id])
        
        # Send to all connections with back-pressure handling
        for connection_id in connection_ids:
            try:
                await self._send_to_connection(connection_id, message)
            except Exception as e:
                logger.error(f"Error broadcasting to connection {connection_id}: {e}")
                # Don't disconnect here, let cleanup loop handle it
    
    async def _send_to_connection(self, connection_id: str, message: Dict[str, Any]):
        """Send message to a specific connection with back-pressure handling.
        
        Args:
            connection_id: Connection ID
            message: Message to send
        """
        if connection_id not in self.connections:
            return
        
        connection = self.connections[connection_id]
        queue = self.connection_queues.get(connection_id)
        
        if not queue:
            return
        
        try:
            # Try to put message in queue (non-blocking)
            queue.put_nowait(message)
            
            # Process queue if not already processing
            if not hasattr(connection, '_processing_queue'):
                connection._processing_queue = True
                asyncio.create_task(self._process_connection_queue(connection_id))
        
        except asyncio.QueueFull:
            # Handle back-pressure by dropping oldest message
            try:
                queue.get_nowait()  # Drop oldest
                queue.put_nowait(message)  # Add new
                self.stats["messages_dropped"] += 1
            except asyncio.QueueEmpty:
                pass
    
    async def _process_connection_queue(self, connection_id: str):
        """Process messages in connection queue.
        
        Args:
            connection_id: Connection ID
        """
        if connection_id not in self.connections:
            return
        
        connection = self.connections[connection_id]
        queue = self.connection_queues.get(connection_id)
        
        if not queue:
            return
        
        try:
            while not queue.empty():
                try:
                    message = queue.get_nowait()
                    await connection.websocket.send_text(json.dumps(message))
                    self.stats["messages_sent"] += 1
                except asyncio.QueueEmpty:
                    break
                except Exception as e:
                    logger.error(f"Error sending message to {connection_id}: {e}")
                    self.stats["errors"] += 1
                    await self._disconnect_client(connection_id)
                    break
        
        finally:
            # Mark queue as not being processed
            if hasattr(connection, '_processing_queue'):
                delattr(connection, '_processing_queue')
    
    async def _disconnect_client(self, connection_id: str):
        """Disconnect a client and clean up resources.
        
        Args:
            connection_id: Connection ID
        """
        if connection_id not in self.connections:
            return
        
        connection = self.connections[connection_id]
        dashboard_id = connection.dashboard_id
        
        # Remove from connections
        del self.connections[connection_id]
        
        # Remove from dashboard connections
        if dashboard_id in self.dashboard_connections:
            self.dashboard_connections[dashboard_id].discard(connection_id)
            
            # Stop broadcast task if no more connections
            if not self.dashboard_connections[dashboard_id]:
                del self.dashboard_connections[dashboard_id]
                if dashboard_id in self.broadcast_tasks:
                    self.broadcast_tasks[dashboard_id].cancel()
                    del self.broadcast_tasks[dashboard_id]
        
        # Clean up queue
        if connection_id in self.connection_queues:
            del self.connection_queues[connection_id]
        
        # Close WebSocket
        try:
            await connection.websocket.close()
        except Exception:
            pass
        
        # Update stats
        self.stats["active_connections"] = len(self.connections)
        
        logger.info(f"Client {connection_id} disconnected from dashboard {dashboard_id}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get gateway statistics.
        
        Returns:
            Statistics dictionary
        """
        return {
            **self.stats,
            "dashboard_count": len(self.dashboard_connections),
            "uptime_seconds": time.time() - (self.stats.get("start_time", time.time())),
            "connections_per_dashboard": {
                dashboard_id: len(connections)
                for dashboard_id, connections in self.dashboard_connections.items()
            }
        }


# Global gateway instance
gateway = WebSocketGateway()


async def websocket_dashboard_endpoint(websocket: WebSocket, dashboard_id: str):
    """WebSocket endpoint for dashboard connections.
    
    Args:
        websocket: WebSocket connection
        dashboard_id: Dashboard ID
    """
    connection_id = None
    
    try:
        # Connect to dashboard
        connection_id = await gateway.connect_dashboard(websocket, dashboard_id)
        
        # Handle messages
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                await gateway.handle_client_message(connection_id, message)
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Invalid JSON format",
                    "timestamp": datetime.utcnow().isoformat()
                }))
            except WebSocketDisconnect:
                break
    
    except Exception as e:
        logger.error(f"Error in dashboard WebSocket: {e}")
    
    finally:
        # Disconnect client
        if connection_id:
            await gateway.disconnect_dashboard(connection_id)
