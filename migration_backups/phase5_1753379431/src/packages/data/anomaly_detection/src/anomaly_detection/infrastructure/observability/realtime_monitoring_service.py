"""Real-time monitoring service with WebSocket support for live dashboards."""

import asyncio
import json
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Set, AsyncGenerator
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, deque
import websockets
import weakref

from .monitoring_orchestrator import MonitoringOrchestrator, MonitoringDashboard
from ..monitoring.metrics_collector import MetricsCollector, get_metrics_collector


class SubscriptionType(Enum):
    """Types of real-time subscriptions."""
    METRICS = "metrics"
    ALERTS = "alerts"
    HEALTH = "health"
    DASHBOARD = "dashboard"
    PERFORMANCE = "performance"
    BUSINESS_METRICS = "business_metrics"


class UpdateFrequency(Enum):
    """Update frequency for real-time subscriptions."""
    REAL_TIME = 1  # 1 second
    HIGH = 5  # 5 seconds
    MEDIUM = 15  # 15 seconds
    LOW = 60  # 60 seconds


@dataclass
class RealtimeSubscription:
    """Real-time monitoring subscription."""
    subscription_id: str
    client_id: str
    subscription_type: SubscriptionType
    frequency: UpdateFrequency
    filters: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    last_update: Optional[datetime] = None
    active: bool = True


@dataclass
class RealtimeMetricUpdate:
    """Real-time metric update message."""
    timestamp: datetime
    subscription_id: str
    subscription_type: SubscriptionType
    data: Dict[str, Any]
    sequence_number: int = 0


class RealtimeMonitoringService:
    """Real-time monitoring service with WebSocket support."""
    
    def __init__(
        self,
        monitoring_orchestrator: Optional[MonitoringOrchestrator] = None,
        websocket_port: int = 8765,
        max_clients: int = 100,
        metrics_buffer_size: int = 1000
    ):
        """Initialize real-time monitoring service.
        
        Args:
            monitoring_orchestrator: Monitoring orchestrator instance
            websocket_port: WebSocket server port
            max_clients: Maximum number of concurrent clients
            metrics_buffer_size: Size of metrics buffer for each subscription
        """
        self.logger = logging.getLogger(__name__)
        self.monitoring_orchestrator = monitoring_orchestrator
        self.metrics_collector = get_metrics_collector()
        self.websocket_port = websocket_port
        self.max_clients = max_clients
        self.metrics_buffer_size = metrics_buffer_size
        
        # Client management
        self.connected_clients: Dict[str, websockets.WebSocketServerProtocol] = {}
        self.client_subscriptions: Dict[str, List[RealtimeSubscription]] = defaultdict(list)
        self.subscription_registry: Dict[str, RealtimeSubscription] = {}
        
        # Data buffers and streaming
        self.metrics_buffers: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.metrics_buffer_size)
        )
        self.update_sequences: Dict[str, int] = defaultdict(int)
        
        # Background tasks
        self._running = False
        self._websocket_server = None
        self._streaming_tasks: Dict[SubscriptionType, asyncio.Task] = {}
        
        # Rate limiting
        self.client_rate_limits: Dict[str, Dict[str, datetime]] = defaultdict(dict)
        self.rate_limit_window = timedelta(seconds=1)
        
        self.logger.info(f"Real-time monitoring service initialized on port {websocket_port}")
    
    async def start(self):
        """Start the real-time monitoring service."""
        if self._running:
            return
        
        self._running = True
        
        # Start WebSocket server
        self._websocket_server = await websockets.serve(
            self._handle_websocket_connection,
            "localhost",
            self.websocket_port,
            max_size=10**6,  # 1MB max message size
            ping_interval=30,
            ping_timeout=10
        )
        
        # Start streaming tasks
        await self._start_streaming_tasks()
        
        self.logger.info(f"Real-time monitoring service started on ws://localhost:{self.websocket_port}")
    
    async def stop(self):
        """Stop the real-time monitoring service."""
        self._running = False
        
        # Stop streaming tasks
        for task in self._streaming_tasks.values():
            task.cancel()
        
        # Close WebSocket server
        if self._websocket_server:
            self._websocket_server.close()
            await self._websocket_server.wait_closed()
        
        # Disconnect all clients
        for client_id, websocket in self.connected_clients.items():
            try:
                await websocket.close()
            except Exception:
                pass
        
        self.connected_clients.clear()
        self.client_subscriptions.clear()
        
        self.logger.info("Real-time monitoring service stopped")
    
    async def _start_streaming_tasks(self):
        """Start background streaming tasks."""
        self._streaming_tasks = {
            SubscriptionType.METRICS: asyncio.create_task(self._stream_metrics()),
            SubscriptionType.ALERTS: asyncio.create_task(self._stream_alerts()),
            SubscriptionType.HEALTH: asyncio.create_task(self._stream_health()),
            SubscriptionType.DASHBOARD: asyncio.create_task(self._stream_dashboard()),
            SubscriptionType.PERFORMANCE: asyncio.create_task(self._stream_performance()),
            SubscriptionType.BUSINESS_METRICS: asyncio.create_task(self._stream_business_metrics()),
        }
    
    async def _handle_websocket_connection(self, websocket, path):
        """Handle new WebSocket connection."""
        client_id = f"client_{id(websocket)}_{datetime.now().timestamp()}"
        
        if len(self.connected_clients) >= self.max_clients:
            await websocket.close(code=1013, reason="Server at capacity")
            return
        
        self.connected_clients[client_id] = websocket
        self.logger.info(f"Client {client_id} connected from {websocket.remote_address}")
        
        try:
            # Send welcome message
            welcome_message = {
                "type": "welcome",
                "client_id": client_id,
                "timestamp": datetime.now().isoformat(),
                "server_info": {
                    "service": "anomaly_detection_realtime_monitoring",
                    "version": "1.0.0",
                    "capabilities": [t.value for t in SubscriptionType]
                }
            }
            await websocket.send(json.dumps(welcome_message))
            
            # Handle client messages
            async for message in websocket:
                await self._handle_client_message(client_id, websocket, message)
                
        except websockets.exceptions.ConnectionClosed:
            self.logger.info(f"Client {client_id} disconnected")
        except Exception as e:
            self.logger.error(f"Error handling client {client_id}: {e}")
        finally:
            # Cleanup client
            self._cleanup_client(client_id)
    
    async def _handle_client_message(self, client_id: str, websocket, message: str):
        """Handle message from client."""
        try:
            data = json.loads(message)
            message_type = data.get("type")
            
            # Rate limiting check
            if not self._check_rate_limit(client_id, message_type):
                await websocket.send(json.dumps({
                    "type": "error",
                    "message": "Rate limit exceeded",
                    "timestamp": datetime.now().isoformat()
                }))
                return
            
            if message_type == "subscribe":
                await self._handle_subscription(client_id, websocket, data)
            elif message_type == "unsubscribe":
                await self._handle_unsubscription(client_id, websocket, data)
            elif message_type == "ping":
                await websocket.send(json.dumps({
                    "type": "pong",
                    "timestamp": datetime.now().isoformat()
                }))
            elif message_type == "get_snapshot":
                await self._handle_snapshot_request(client_id, websocket, data)
            else:
                await websocket.send(json.dumps({
                    "type": "error",
                    "message": f"Unknown message type: {message_type}",
                    "timestamp": datetime.now().isoformat()
                }))
                
        except json.JSONDecodeError:
            await websocket.send(json.dumps({
                "type": "error",
                "message": "Invalid JSON message",
                "timestamp": datetime.now().isoformat()
            }))
        except Exception as e:
            self.logger.error(f"Error handling message from {client_id}: {e}")
            await websocket.send(json.dumps({
                "type": "error",
                "message": "Internal server error",
                "timestamp": datetime.now().isoformat()
            }))
    
    def _check_rate_limit(self, client_id: str, message_type: str) -> bool:
        """Check if client is within rate limits."""
        now = datetime.now()
        client_limits = self.client_rate_limits[client_id]
        
        last_request = client_limits.get(message_type)
        if last_request and (now - last_request) < self.rate_limit_window:
            return False
        
        client_limits[message_type] = now
        return True
    
    async def _handle_subscription(self, client_id: str, websocket, data: Dict[str, Any]):
        """Handle subscription request."""
        try:
            subscription_type = SubscriptionType(data.get("subscription_type"))
            frequency = UpdateFrequency(data.get("frequency", UpdateFrequency.MEDIUM.value))
            filters = data.get("filters", {})
            
            subscription = RealtimeSubscription(
                subscription_id=f"sub_{client_id}_{subscription_type.value}_{datetime.now().timestamp()}",
                client_id=client_id,
                subscription_type=subscription_type,
                frequency=frequency,
                filters=filters
            )
            
            # Register subscription
            self.client_subscriptions[client_id].append(subscription)
            self.subscription_registry[subscription.subscription_id] = subscription
            
            # Send confirmation
            await websocket.send(json.dumps({
                "type": "subscription_confirmed",
                "subscription_id": subscription.subscription_id,
                "subscription_type": subscription_type.value,
                "frequency": frequency.value,
                "timestamp": datetime.now().isoformat()
            }))
            
            self.logger.info(f"Client {client_id} subscribed to {subscription_type.value}")
            
        except ValueError as e:
            await websocket.send(json.dumps({
                "type": "error",
                "message": f"Invalid subscription parameters: {e}",
                "timestamp": datetime.now().isoformat()
            }))
    
    async def _handle_unsubscription(self, client_id: str, websocket, data: Dict[str, Any]):
        """Handle unsubscription request."""
        subscription_id = data.get("subscription_id")
        
        if subscription_id in self.subscription_registry:
            subscription = self.subscription_registry[subscription_id]
            if subscription.client_id == client_id:
                subscription.active = False
                del self.subscription_registry[subscription_id]
                
                # Remove from client subscriptions
                self.client_subscriptions[client_id] = [
                    sub for sub in self.client_subscriptions[client_id]
                    if sub.subscription_id != subscription_id
                ]
                
                await websocket.send(json.dumps({
                    "type": "unsubscription_confirmed",
                    "subscription_id": subscription_id,
                    "timestamp": datetime.now().isoformat()
                }))
                
                self.logger.info(f"Client {client_id} unsubscribed from {subscription_id}")
            else:
                await websocket.send(json.dumps({
                    "type": "error",
                    "message": "Subscription not found or unauthorized",
                    "timestamp": datetime.now().isoformat()
                }))
        else:
            await websocket.send(json.dumps({
                "type": "error",
                "message": "Subscription not found",
                "timestamp": datetime.now().isoformat()
            }))
    
    async def _handle_snapshot_request(self, client_id: str, websocket, data: Dict[str, Any]):
        """Handle snapshot data request."""
        try:
            request_type = data.get("request_type")
            
            if request_type == "dashboard":
                snapshot_data = await self._get_dashboard_snapshot()
            elif request_type == "metrics":
                snapshot_data = await self._get_metrics_snapshot(data.get("filters", {}))
            elif request_type == "alerts":
                snapshot_data = await self._get_alerts_snapshot()
            elif request_type == "health":
                snapshot_data = await self._get_health_snapshot()
            else:
                raise ValueError(f"Unknown request type: {request_type}")
            
            await websocket.send(json.dumps({
                "type": "snapshot",
                "request_type": request_type,
                "data": snapshot_data,
                "timestamp": datetime.now().isoformat()
            }))
            
        except Exception as e:
            await websocket.send(json.dumps({
                "type": "error",
                "message": f"Error getting snapshot: {e}",
                "timestamp": datetime.now().isoformat()
            }))
    
    def _cleanup_client(self, client_id: str):
        """Clean up client data."""
        # Remove client connection
        if client_id in self.connected_clients:
            del self.connected_clients[client_id]
        
        # Deactivate subscriptions
        for subscription in self.client_subscriptions[client_id]:
            subscription.active = False
            if subscription.subscription_id in self.subscription_registry:
                del self.subscription_registry[subscription.subscription_id]
        
        # Remove client subscriptions
        if client_id in self.client_subscriptions:
            del self.client_subscriptions[client_id]
        
        # Remove rate limit data
        if client_id in self.client_rate_limits:
            del self.client_rate_limits[client_id]
    
    async def _stream_metrics(self):
        """Stream metrics updates to subscribed clients."""
        while self._running:
            try:
                current_time = datetime.now()
                
                # Get active metrics subscriptions
                subscriptions = [
                    sub for sub in self.subscription_registry.values()
                    if (sub.subscription_type == SubscriptionType.METRICS and 
                        sub.active and
                        self._should_update_subscription(sub, current_time))
                ]
                
                if subscriptions and self.metrics_collector:
                    # Get recent metrics
                    recent_metrics = self.metrics_collector.get_recent_metrics(minutes=1)
                    
                    for subscription in subscriptions:
                        try:
                            # Apply filters
                            filtered_metrics = self._apply_metrics_filters(
                                recent_metrics, subscription.filters
                            )
                            
                            if filtered_metrics:
                                update = RealtimeMetricUpdate(
                                    timestamp=current_time,
                                    subscription_id=subscription.subscription_id,
                                    subscription_type=SubscriptionType.METRICS,
                                    data={"metrics": filtered_metrics},
                                    sequence_number=self.update_sequences[subscription.subscription_id]
                                )
                                
                                await self._send_update_to_client(subscription.client_id, update)
                                subscription.last_update = current_time
                                self.update_sequences[subscription.subscription_id] += 1
                                
                        except Exception as e:
                            self.logger.error(f"Error streaming metrics to {subscription.client_id}: {e}")
                
                await asyncio.sleep(1)  # Check every second
                
            except Exception as e:
                self.logger.error(f"Error in metrics streaming: {e}")
                await asyncio.sleep(1)
    
    async def _stream_alerts(self):
        """Stream alert updates to subscribed clients."""
        last_alert_check = datetime.now()
        
        while self._running:
            try:
                current_time = datetime.now()
                
                # Get active alert subscriptions
                subscriptions = [
                    sub for sub in self.subscription_registry.values()
                    if (sub.subscription_type == SubscriptionType.ALERTS and 
                        sub.active and
                        self._should_update_subscription(sub, current_time))
                ]
                
                if subscriptions and self.monitoring_orchestrator:
                    # Get recent alerts
                    alerts = self.monitoring_orchestrator.get_active_alerts()
                    
                    # Convert alerts to serializable format
                    alert_data = []
                    for alert in alerts:
                        alert_dict = {
                            "alert_id": alert.alert_id,
                            "service_type": alert.service_type.value,
                            "severity": alert.severity.value,
                            "message": alert.message,
                            "timestamp": alert.timestamp.isoformat(),
                            "context": alert.context,
                            "correlation_id": alert.correlation_id,
                            "suppressed": alert.suppressed
                        }
                        alert_data.append(alert_dict)
                    
                    for subscription in subscriptions:
                        try:
                            update = RealtimeMetricUpdate(
                                timestamp=current_time,
                                subscription_id=subscription.subscription_id,
                                subscription_type=SubscriptionType.ALERTS,
                                data={"alerts": alert_data},
                                sequence_number=self.update_sequences[subscription.subscription_id]
                            )
                            
                            await self._send_update_to_client(subscription.client_id, update)
                            subscription.last_update = current_time
                            self.update_sequences[subscription.subscription_id] += 1
                            
                        except Exception as e:
                            self.logger.error(f"Error streaming alerts to {subscription.client_id}: {e}")
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                self.logger.error(f"Error in alerts streaming: {e}")
                await asyncio.sleep(5)
    
    async def _stream_health(self):
        """Stream health updates to subscribed clients."""
        while self._running:
            try:
                current_time = datetime.now()
                
                # Get active health subscriptions
                subscriptions = [
                    sub for sub in self.subscription_registry.values()
                    if (sub.subscription_type == SubscriptionType.HEALTH and 
                        sub.active and
                        self._should_update_subscription(sub, current_time))
                ]
                
                if subscriptions and self.monitoring_orchestrator:
                    # Get health data
                    health_data = {}
                    for service_type in self.monitoring_orchestrator.service_registry.keys():
                        health = self.monitoring_orchestrator.get_service_health(service_type)
                        health_data[service_type.value] = health.value
                    
                    for subscription in subscriptions:
                        try:
                            update = RealtimeMetricUpdate(
                                timestamp=current_time,
                                subscription_id=subscription.subscription_id,
                                subscription_type=SubscriptionType.HEALTH,
                                data={"health": health_data},
                                sequence_number=self.update_sequences[subscription.subscription_id]
                            )
                            
                            await self._send_update_to_client(subscription.client_id, update)
                            subscription.last_update = current_time
                            self.update_sequences[subscription.subscription_id] += 1
                            
                        except Exception as e:
                            self.logger.error(f"Error streaming health to {subscription.client_id}: {e}")
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Error in health streaming: {e}")
                await asyncio.sleep(10)
    
    async def _stream_dashboard(self):
        """Stream dashboard updates to subscribed clients."""
        while self._running:
            try:
                current_time = datetime.now()
                
                # Get active dashboard subscriptions
                subscriptions = [
                    sub for sub in self.subscription_registry.values()
                    if (sub.subscription_type == SubscriptionType.DASHBOARD and 
                        sub.active and
                        self._should_update_subscription(sub, current_time))
                ]
                
                if subscriptions and self.monitoring_orchestrator:
                    # Get dashboard data
                    dashboard_data = self.monitoring_orchestrator.get_dashboard_data()
                    
                    if dashboard_data:
                        # Convert to serializable format
                        dashboard_dict = {
                            "overall_health": dashboard_data.overall_health.value,
                            "service_statuses": {
                                k.value: v.value for k, v in dashboard_data.service_statuses.items()
                            },
                            "active_alerts": [
                                {
                                    "alert_id": alert.alert_id,
                                    "service_type": alert.service_type.value,
                                    "severity": alert.severity.value,
                                    "message": alert.message,
                                    "timestamp": alert.timestamp.isoformat()
                                }
                                for alert in dashboard_data.active_alerts
                            ],
                            "key_metrics": dashboard_data.key_metrics,
                            "performance_summary": dashboard_data.performance_summary,
                            "business_metrics_summary": dashboard_data.business_metrics_summary,
                            "recent_events": dashboard_data.recent_events,
                            "last_updated": dashboard_data.last_updated.isoformat()
                        }
                        
                        for subscription in subscriptions:
                            try:
                                update = RealtimeMetricUpdate(
                                    timestamp=current_time,
                                    subscription_id=subscription.subscription_id,
                                    subscription_type=SubscriptionType.DASHBOARD,
                                    data={"dashboard": dashboard_dict},
                                    sequence_number=self.update_sequences[subscription.subscription_id]
                                )
                                
                                await self._send_update_to_client(subscription.client_id, update)
                                subscription.last_update = current_time
                                self.update_sequences[subscription.subscription_id] += 1
                                
                            except Exception as e:
                                self.logger.error(f"Error streaming dashboard to {subscription.client_id}: {e}")
                
                await asyncio.sleep(15)  # Check every 15 seconds
                
            except Exception as e:
                self.logger.error(f"Error in dashboard streaming: {e}")
                await asyncio.sleep(15)
    
    async def _stream_performance(self):
        """Stream performance updates to subscribed clients."""
        while self._running:
            try:
                current_time = datetime.now()
                
                # Get active performance subscriptions
                subscriptions = [
                    sub for sub in self.subscription_registry.values()
                    if (sub.subscription_type == SubscriptionType.PERFORMANCE and 
                        sub.active and
                        self._should_update_subscription(sub, current_time))
                ]
                
                if subscriptions:
                    # Get performance data (simplified)
                    performance_data = {
                        "cpu_usage": 0.0,
                        "memory_usage": 0.0,
                        "active_connections": len(self.connected_clients),
                        "active_subscriptions": len(self.subscription_registry)
                    }
                    
                    # Try to get real performance data
                    try:
                        import psutil
                        performance_data.update({
                            "cpu_usage": psutil.cpu_percent(),
                            "memory_usage": psutil.virtual_memory().percent / 100.0
                        })
                    except ImportError:
                        pass
                    
                    for subscription in subscriptions:
                        try:
                            update = RealtimeMetricUpdate(
                                timestamp=current_time,
                                subscription_id=subscription.subscription_id,
                                subscription_type=SubscriptionType.PERFORMANCE,
                                data={"performance": performance_data},
                                sequence_number=self.update_sequences[subscription.subscription_id]
                            )
                            
                            await self._send_update_to_client(subscription.client_id, update)
                            subscription.last_update = current_time
                            self.update_sequences[subscription.subscription_id] += 1
                            
                        except Exception as e:
                            self.logger.error(f"Error streaming performance to {subscription.client_id}: {e}")
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                self.logger.error(f"Error in performance streaming: {e}")
                await asyncio.sleep(5)
    
    async def _stream_business_metrics(self):
        """Stream business metrics updates to subscribed clients."""
        while self._running:
            try:
                current_time = datetime.now()
                
                # Get active business metrics subscriptions
                subscriptions = [
                    sub for sub in self.subscription_registry.values()
                    if (sub.subscription_type == SubscriptionType.BUSINESS_METRICS and 
                        sub.active and
                        self._should_update_subscription(sub, current_time))
                ]
                
                if subscriptions and self.monitoring_orchestrator and self.monitoring_orchestrator.business_metrics_service:
                    # Get business metrics data
                    dashboard_data = self.monitoring_orchestrator.business_metrics_service.generate_business_dashboard()
                    
                    for subscription in subscriptions:
                        try:
                            update = RealtimeMetricUpdate(
                                timestamp=current_time,
                                subscription_id=subscription.subscription_id,
                                subscription_type=SubscriptionType.BUSINESS_METRICS,
                                data={"business_metrics": dashboard_data},
                                sequence_number=self.update_sequences[subscription.subscription_id]
                            )
                            
                            await self._send_update_to_client(subscription.client_id, update)
                            subscription.last_update = current_time
                            self.update_sequences[subscription.subscription_id] += 1
                            
                        except Exception as e:
                            self.logger.error(f"Error streaming business metrics to {subscription.client_id}: {e}")
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error in business metrics streaming: {e}")
                await asyncio.sleep(30)
    
    def _should_update_subscription(self, subscription: RealtimeSubscription, current_time: datetime) -> bool:
        """Check if subscription should be updated based on frequency."""
        if not subscription.last_update:
            return True
        
        time_since_last = current_time - subscription.last_update
        frequency_seconds = subscription.frequency.value
        
        return time_since_last.total_seconds() >= frequency_seconds
    
    def _apply_metrics_filters(self, metrics: List[Dict[str, Any]], filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply filters to metrics data."""
        if not filters:
            return metrics
        
        filtered_metrics = []
        for metric in metrics:
            # Apply metric name filter
            if "metric_names" in filters:
                if metric.get("metric_name") not in filters["metric_names"]:
                    continue
            
            # Apply tag filters
            if "tags" in filters:
                metric_tags = metric.get("tags", {})
                if not all(metric_tags.get(k) == v for k, v in filters["tags"].items()):
                    continue
            
            # Apply value range filter
            if "value_range" in filters:
                value = metric.get("value", 0)
                min_val = filters["value_range"].get("min")
                max_val = filters["value_range"].get("max")
                
                if min_val is not None and value < min_val:
                    continue
                if max_val is not None and value > max_val:
                    continue
            
            filtered_metrics.append(metric)
        
        return filtered_metrics
    
    async def _send_update_to_client(self, client_id: str, update: RealtimeMetricUpdate):
        """Send update to specific client."""
        if client_id not in self.connected_clients:
            return
        
        websocket = self.connected_clients[client_id]
        
        try:
            message = {
                "type": "update",
                "subscription_id": update.subscription_id,
                "subscription_type": update.subscription_type.value,
                "timestamp": update.timestamp.isoformat(),
                "sequence_number": update.sequence_number,
                "data": update.data
            }
            
            await websocket.send(json.dumps(message))
            
        except websockets.exceptions.ConnectionClosed:
            # Client disconnected, cleanup will be handled elsewhere
            pass
        except Exception as e:
            self.logger.error(f"Error sending update to client {client_id}: {e}")
    
    async def _get_dashboard_snapshot(self) -> Dict[str, Any]:
        """Get current dashboard snapshot."""
        if self.monitoring_orchestrator:
            dashboard_data = self.monitoring_orchestrator.get_dashboard_data()
            if dashboard_data:
                return {
                    "overall_health": dashboard_data.overall_health.value,
                    "service_statuses": {
                        k.value: v.value for k, v in dashboard_data.service_statuses.items()
                    },
                    "active_alerts": len(dashboard_data.active_alerts),
                    "key_metrics": dashboard_data.key_metrics,
                    "last_updated": dashboard_data.last_updated.isoformat()
                }
        
        return {"status": "no_data"}
    
    async def _get_metrics_snapshot(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Get current metrics snapshot."""
        if self.metrics_collector:
            recent_metrics = self.metrics_collector.get_recent_metrics(minutes=5)
            filtered_metrics = self._apply_metrics_filters(recent_metrics, filters)
            return {"metrics": filtered_metrics, "count": len(filtered_metrics)}
        
        return {"metrics": [], "count": 0}
    
    async def _get_alerts_snapshot(self) -> Dict[str, Any]:
        """Get current alerts snapshot."""
        if self.monitoring_orchestrator:
            alerts = self.monitoring_orchestrator.get_active_alerts()
            return {
                "alerts": [
                    {
                        "alert_id": alert.alert_id,
                        "severity": alert.severity.value,
                        "message": alert.message,
                        "timestamp": alert.timestamp.isoformat()
                    }
                    for alert in alerts
                ],
                "count": len(alerts)
            }
        
        return {"alerts": [], "count": 0}
    
    async def _get_health_snapshot(self) -> Dict[str, Any]:
        """Get current health snapshot."""
        if self.monitoring_orchestrator:
            health_data = {}
            for service_type in self.monitoring_orchestrator.service_registry.keys():
                health = self.monitoring_orchestrator.get_service_health(service_type)
                health_data[service_type.value] = health.value
            
            return {"health": health_data}
        
        return {"health": {}}
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Get real-time monitoring service statistics."""
        return {
            "service_running": self._running,
            "websocket_port": self.websocket_port,
            "connected_clients": len(self.connected_clients),
            "active_subscriptions": len(self.subscription_registry),
            "subscription_types": {
                sub_type.value: len([
                    sub for sub in self.subscription_registry.values()
                    if sub.subscription_type == sub_type and sub.active
                ])
                for sub_type in SubscriptionType
            },
            "streaming_tasks": len(self._streaming_tasks),
            "metrics_buffer_size": self.metrics_buffer_size,
            "max_clients": self.max_clients
        }


# Global service instance
_realtime_monitoring_service: Optional[RealtimeMonitoringService] = None


def initialize_realtime_monitoring_service(
    monitoring_orchestrator: Optional[MonitoringOrchestrator] = None,
    websocket_port: int = 8765
) -> RealtimeMonitoringService:
    """Initialize global real-time monitoring service.
    
    Args:
        monitoring_orchestrator: Monitoring orchestrator instance
        websocket_port: WebSocket server port
        
    Returns:
        Initialized real-time monitoring service
    """
    global _realtime_monitoring_service
    _realtime_monitoring_service = RealtimeMonitoringService(
        monitoring_orchestrator=monitoring_orchestrator,
        websocket_port=websocket_port
    )
    return _realtime_monitoring_service


def get_realtime_monitoring_service() -> Optional[RealtimeMonitoringService]:
    """Get global real-time monitoring service instance.
    
    Returns:
        Real-time monitoring service instance or None
    """
    return _realtime_monitoring_service