#!/usr/bin/env python3
"""Enhanced WebSocket manager for real-time dashboard updates."""

import asyncio
import json
import logging
import weakref
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

from aiohttp import WSMsgType, web
from prometheus_client import Gauge, Counter

from pynomaly.shared.logging import get_logger
from .enhanced_metrics_collector import EnhancedMetricsCollector

logger = get_logger(__name__)


class WebSocketConnection:
    """Represents a WebSocket connection with subscription management."""
    
    def __init__(self, websocket: web.WebSocketResponse, connection_id: str):
        self.websocket = websocket
        self.connection_id = connection_id
        self.subscriptions: Set[str] = set()
        self.connected_at = datetime.utcnow()
        self.last_ping = datetime.utcnow()
        self.user_id: Optional[str] = None
        self.session_data: Dict[str, Any] = {}
    
    async def send_message(self, message: Dict[str, Any]) -> bool:
        """Send a message to the WebSocket connection."""
        try:
            if self.websocket.closed:
                return False
            
            await self.websocket.send_str(json.dumps(message))
            return True
        except Exception as e:
            logger.warning(f"Failed to send message to {self.connection_id}: {e}")
            return False
    
    def add_subscription(self, topic: str):
        """Add a subscription topic."""
        self.subscriptions.add(topic)
    
    def remove_subscription(self, topic: str):
        """Remove a subscription topic."""
        self.subscriptions.discard(topic)
    
    def is_subscribed_to(self, topic: str) -> bool:
        """Check if connection is subscribed to a topic."""
        return topic in self.subscriptions
    
    def update_last_ping(self):
        """Update the last ping timestamp."""
        self.last_ping = datetime.utcnow()
    
    def is_stale(self, timeout: timedelta) -> bool:
        """Check if the connection is stale."""
        return datetime.utcnow() - self.last_ping > timeout


class RealtimeWebSocketManager:
    """Enhanced WebSocket manager for real-time dashboard communication."""
    
    def __init__(self, metrics_collector: EnhancedMetricsCollector, ping_interval: float = 30.0):
        self.metrics_collector = metrics_collector
        self.ping_interval = ping_interval
        
        # Connection management
        self.connections: Dict[str, WebSocketConnection] = {}
        self.topic_subscribers: Dict[str, Set[str]] = {}
        
        # Background tasks
        self.cleanup_task: Optional[asyncio.Task] = None
        self.broadcast_task: Optional[asyncio.Task] = None
        self.ping_task: Optional[asyncio.Task] = None
        
        # Configuration
        self.max_connections = 1000
        self.stale_connection_timeout = timedelta(minutes=5)
        self.broadcast_interval = 2.0  # seconds
        
        # Metrics
        self.connections_gauge = Gauge('websocket_connections_current', 'Current WebSocket connections')
        self.messages_sent_counter = Counter('websocket_messages_sent_total', 'Total WebSocket messages sent', ['topic'])
        self.connections_total_counter = Counter('websocket_connections_total', 'Total WebSocket connections established')
        self.disconnections_total_counter = Counter('websocket_disconnections_total', 'Total WebSocket disconnections')
        
        # Message queues for different topics
        self.topic_queues: Dict[str, asyncio.Queue] = {
            'system_metrics': asyncio.Queue(maxsize=100),
            'application_metrics': asyncio.Queue(maxsize=100),
            'alerts': asyncio.Queue(maxsize=50),
            'dashboard_updates': asyncio.Queue(maxsize=100),
            'health_status': asyncio.Queue(maxsize=50),
        }
        
        logger.info("Enhanced WebSocket manager initialized")
    
    async def start(self):
        """Start the WebSocket manager background tasks."""
        self.cleanup_task = asyncio.create_task(self._cleanup_stale_connections())
        self.broadcast_task = asyncio.create_task(self._broadcast_metrics())
        self.ping_task = asyncio.create_task(self._ping_connections())
        logger.info("WebSocket manager started")
    
    async def stop(self):
        """Stop the WebSocket manager and cleanup resources."""
        # Cancel background tasks
        for task in [self.cleanup_task, self.broadcast_task, self.ping_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Close all connections
        for connection in list(self.connections.values()):
            await self._close_connection(connection.connection_id)
        
        logger.info("WebSocket manager stopped")
    
    async def handle_websocket(self, request: web.Request) -> web.WebSocketResponse:
        """Handle incoming WebSocket connections."""
        ws = web.WebSocketResponse(heartbeat=30)
        await ws.prepare(request)
        
        # Check connection limits
        if len(self.connections) >= self.max_connections:
            await ws.close(code=1013, message=b'Server overloaded')
            return ws
        
        connection_id = str(uuid4())
        connection = WebSocketConnection(ws, connection_id)
        
        # Extract user information from request if available
        session = await self._get_session_from_request(request)
        if session:
            connection.user_id = session.get('user_id')
            connection.session_data = session
        
        self.connections[connection_id] = connection
        self.connections_gauge.set(len(self.connections))
        self.connections_total_counter.inc()
        
        logger.info(f"WebSocket connection established: {connection_id}")
        
        # Send welcome message
        await connection.send_message({
            'type': 'connection_established',
            'connection_id': connection_id,
            'timestamp': datetime.utcnow().isoformat(),
            'server_time': datetime.utcnow().isoformat(),
        })
        
        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        await self._handle_client_message(connection, data)
                    except json.JSONDecodeError as e:
                        await connection.send_message({
                            'type': 'error',
                            'message': f'Invalid JSON: {e}',
                            'timestamp': datetime.utcnow().isoformat(),
                        })
                elif msg.type == WSMsgType.ERROR:
                    logger.error(f'WebSocket error: {ws.exception()}')
                    break
                elif msg.type == WSMsgType.CLOSE:
                    break
        except Exception as e:
            logger.error(f"Error handling WebSocket connection {connection_id}: {e}")
        finally:
            await self._close_connection(connection_id)
        
        return ws
    
    async def _handle_client_message(self, connection: WebSocketConnection, data: Dict[str, Any]):
        """Handle messages from WebSocket clients."""
        message_type = data.get('type')
        
        if message_type == 'ping':
            connection.update_last_ping()
            await connection.send_message({
                'type': 'pong',
                'timestamp': datetime.utcnow().isoformat(),
            })
        
        elif message_type == 'subscribe':
            topics = data.get('topics', [])
            for topic in topics:
                connection.add_subscription(topic)
                self._add_topic_subscriber(topic, connection.connection_id)
            
            await connection.send_message({
                'type': 'subscription_confirmed',
                'topics': topics,
                'timestamp': datetime.utcnow().isoformat(),
            })
        
        elif message_type == 'unsubscribe':
            topics = data.get('topics', [])
            for topic in topics:
                connection.remove_subscription(topic)
                self._remove_topic_subscriber(topic, connection.connection_id)
            
            await connection.send_message({
                'type': 'unsubscription_confirmed',
                'topics': topics,
                'timestamp': datetime.utcnow().isoformat(),
            })
        
        elif message_type == 'get_dashboard_data':
            dashboard_id = data.get('dashboard_id')
            if dashboard_id:
                await self._send_dashboard_data(connection, dashboard_id)
        
        elif message_type == 'request_metrics':
            metric_names = data.get('metrics', [])
            time_range = data.get('time_range', '1h')
            await self._send_metrics_data(connection, metric_names, time_range)
        
        else:
            await connection.send_message({
                'type': 'error',
                'message': f'Unknown message type: {message_type}',
                'timestamp': datetime.utcnow().isoformat(),
            })
    
    async def _close_connection(self, connection_id: str):
        """Close and cleanup a WebSocket connection."""
        if connection_id not in self.connections:
            return
        
        connection = self.connections[connection_id]
        
        # Remove from topic subscriptions
        for topic in connection.subscriptions:
            self._remove_topic_subscriber(topic, connection_id)
        
        # Close WebSocket if not already closed
        if not connection.websocket.closed:
            await connection.websocket.close()
        
        # Remove from connections
        del self.connections[connection_id]
        self.connections_gauge.set(len(self.connections))
        self.disconnections_total_counter.inc()
        
        logger.info(f"WebSocket connection closed: {connection_id}")
    
    def _add_topic_subscriber(self, topic: str, connection_id: str):
        """Add a connection to a topic's subscriber list."""
        if topic not in self.topic_subscribers:
            self.topic_subscribers[topic] = set()
        self.topic_subscribers[topic].add(connection_id)
    
    def _remove_topic_subscriber(self, topic: str, connection_id: str):
        """Remove a connection from a topic's subscriber list."""
        if topic in self.topic_subscribers:
            self.topic_subscribers[topic].discard(connection_id)
            if not self.topic_subscribers[topic]:
                del self.topic_subscribers[topic]
    
    async def broadcast_to_topic(self, topic: str, message: Dict[str, Any]):
        """Broadcast a message to all subscribers of a topic."""
        if topic not in self.topic_subscribers:
            return
        
        message['topic'] = topic
        message['timestamp'] = datetime.utcnow().isoformat()
        
        successful_sends = 0
        failed_connections = []
        
        for connection_id in self.topic_subscribers[topic].copy():
            if connection_id in self.connections:
                connection = self.connections[connection_id]
                success = await connection.send_message(message)
                if success:
                    successful_sends += 1
                else:
                    failed_connections.append(connection_id)
        
        # Cleanup failed connections
        for connection_id in failed_connections:
            await self._close_connection(connection_id)
        
        if successful_sends > 0:
            self.messages_sent_counter.labels(topic=topic).inc(successful_sends)
        
        logger.debug(f"Broadcasted to topic '{topic}': {successful_sends} successful sends")
    
    async def _cleanup_stale_connections(self):
        """Background task to cleanup stale connections."""
        while True:
            try:
                stale_connections = []
                
                for connection_id, connection in self.connections.items():
                    if connection.is_stale(self.stale_connection_timeout):
                        stale_connections.append(connection_id)
                
                for connection_id in stale_connections:
                    logger.info(f"Closing stale connection: {connection_id}")
                    await self._close_connection(connection_id)
                
                await asyncio.sleep(60)  # Check every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")
                await asyncio.sleep(10)
    
    async def _broadcast_metrics(self):
        """Background task to broadcast metrics to subscribers."""
        while True:
            try:
                # Get current metrics summary
                metrics_summary = self.metrics_collector.get_metrics_summary()
                
                # Broadcast system metrics
                await self.broadcast_to_topic('system_metrics', {
                    'type': 'metrics_update',
                    'data': metrics_summary['system'],
                })
                
                # Broadcast application metrics
                await self.broadcast_to_topic('application_metrics', {
                    'type': 'metrics_update',
                    'data': metrics_summary['application'],
                })
                
                # Broadcast health status
                health_score = metrics_summary['system'].get('health_score', 0)
                await self.broadcast_to_topic('health_status', {
                    'type': 'health_update',
                    'health_score': health_score,
                    'status': 'healthy' if health_score > 80 else 'warning' if health_score > 60 else 'critical',
                })
                
                await asyncio.sleep(self.broadcast_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in broadcast task: {e}")
                await asyncio.sleep(5)
    
    async def _ping_connections(self):
        """Background task to ping connections for health check."""
        while True:
            try:
                for connection in list(self.connections.values()):
                    await connection.send_message({
                        'type': 'server_ping',
                        'timestamp': datetime.utcnow().isoformat(),
                    })
                
                await asyncio.sleep(self.ping_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in ping task: {e}")
                await asyncio.sleep(10)
    
    async def _send_dashboard_data(self, connection: WebSocketConnection, dashboard_id: str):
        """Send dashboard data to a specific connection."""
        # This would integrate with the dashboard service
        # For now, we'll send sample data
        dashboard_data = {
            'type': 'dashboard_data',
            'dashboard_id': dashboard_id,
            'data': {
                'widgets': [],
                'timestamp': datetime.utcnow().isoformat(),
            }
        }
        await connection.send_message(dashboard_data)
    
    async def _send_metrics_data(self, connection: WebSocketConnection, metric_names: List[str], time_range: str):
        """Send metrics data to a specific connection."""
        time_delta = self._parse_time_range(time_range)
        metrics_data = {}
        
        for metric_name in metric_names:
            time_series = self.metrics_collector.get_time_series_data(metric_name, time_delta)
            metrics_data[metric_name] = time_series
        
        await connection.send_message({
            'type': 'metrics_data',
            'data': metrics_data,
            'time_range': time_range,
            'timestamp': datetime.utcnow().isoformat(),
        })
    
    def _parse_time_range(self, time_range: str) -> timedelta:
        """Parse time range string to timedelta."""
        time_map = {
            '5m': timedelta(minutes=5),
            '15m': timedelta(minutes=15),
            '30m': timedelta(minutes=30),
            '1h': timedelta(hours=1),
            '6h': timedelta(hours=6),
            '12h': timedelta(hours=12),
            '24h': timedelta(hours=24),
            '7d': timedelta(days=7),
            '30d': timedelta(days=30),
        }
        return time_map.get(time_range, timedelta(hours=1))
    
    async def _get_session_from_request(self, request: web.Request) -> Optional[Dict[str, Any]]:
        """Extract session data from request."""
        try:
            # This would integrate with your authentication system
            # For now, return None
            return None
        except Exception:
            return None
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get WebSocket connection statistics."""
        return {
            'total_connections': len(self.connections),
            'topic_subscribers': {
                topic: len(subscribers) 
                for topic, subscribers in self.topic_subscribers.items()
            },
            'max_connections': self.max_connections,
            'uptime': datetime.utcnow().isoformat(),
        }