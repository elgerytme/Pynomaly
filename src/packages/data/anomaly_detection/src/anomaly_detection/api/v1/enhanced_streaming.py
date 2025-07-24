"""Enhanced real-time WebSocket streaming for anomaly detection."""

import asyncio
import json
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any, AsyncGenerator
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from enum import Enum

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.websockets import WebSocketState
from pydantic import BaseModel, Field

from ...domain.services.streaming_service import StreamingService
from ...domain.services.detection_service import DetectionService
from ...infrastructure.logging import get_logger
from ...infrastructure.monitoring import get_metrics_collector

router = APIRouter()
logger = get_logger(__name__)
metrics = get_metrics_collector()


class MessageType(str, Enum):
    """WebSocket message types."""
    SAMPLE = "sample"
    BATCH = "batch"
    STATS = "stats"
    DRIFT = "drift"
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    PING = "ping"
    PONG = "pong"
    ERROR = "error"
    RESULT = "result"
    ALERT = "alert"
    CONFIG = "config"


class StreamingMode(str, Enum):
    """Streaming processing modes."""
    REALTIME = "realtime"  # Process immediately
    BATCH = "batch"        # Accumulate and process in batches
    ADAPTIVE = "adaptive"  # Automatically choose based on load


@dataclass
class ConnectionInfo:
    """Information about a WebSocket connection."""
    websocket: WebSocket
    session_id: str
    client_id: str
    connected_at: datetime
    last_activity: datetime
    subscriptions: Set[str]
    processing_mode: StreamingMode
    batch_size: int
    batch_timeout: float
    is_authenticated: bool = False
    user_id: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class EnhancedStreamingMessage(BaseModel):
    """Enhanced streaming message structure."""
    type: MessageType = Field(..., description="Message type")
    data: Optional[List[float]] = Field(None, description="Single sample data")
    batch_data: Optional[List[List[float]]] = Field(None, description="Batch data")
    algorithm: str = Field("isolation_forest", description="Detection algorithm")
    timestamp: Optional[str] = Field(None, description="Message timestamp")
    session_id: Optional[str] = Field(None, description="Session identifier")
    client_id: Optional[str] = Field(None, description="Client identifier")
    request_id: Optional[str] = Field(None, description="Request identifier")
    config: Optional[Dict[str, Any]] = Field(None, description="Configuration parameters")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class StreamingResponse(BaseModel):
    """Enhanced streaming response."""
    type: MessageType = Field(..., description="Response type")
    success: bool = Field(..., description="Whether operation succeeded")
    request_id: Optional[str] = Field(None, description="Original request identifier")
    timestamp: str = Field(..., description="Response timestamp")
    
    # Detection results
    sample_id: Optional[str] = Field(None, description="Sample identifier")
    is_anomaly: Optional[bool] = Field(None, description="Anomaly detection result")
    confidence_score: Optional[float] = Field(None, description="Confidence score")
    algorithm: Optional[str] = Field(None, description="Algorithm used")
    
    # Batch results
    results: Optional[List[Dict[str, Any]]] = Field(None, description="Batch results")
    
    # Statistics
    stats: Optional[Dict[str, Any]] = Field(None, description="Streaming statistics")
    
    # Drift detection
    drift_detected: Optional[bool] = Field(None, description="Drift detection result")
    drift_details: Optional[Dict[str, Any]] = Field(None, description="Drift details")
    
    # Error information
    error: Optional[str] = Field(None, description="Error message")
    error_code: Optional[str] = Field(None, description="Error code")
    
    # Metadata
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class StreamingAlert(BaseModel):
    """Streaming alert message."""
    alert_id: str = Field(..., description="Alert identifier")
    severity: str = Field(..., description="Alert severity (low, medium, high, critical)")
    title: str = Field(..., description="Alert title")
    message: str = Field(..., description="Alert message")
    timestamp: str = Field(..., description="Alert timestamp")
    source: str = Field(..., description="Alert source")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Alert metadata")


class ConnectionManager:
    """Manages WebSocket connections and message routing."""
    
    def __init__(self):
        self.active_connections: Dict[str, ConnectionInfo] = {}
        self.session_connections: Dict[str, Set[str]] = defaultdict(set)
        self.subscription_map: Dict[str, Set[str]] = defaultdict(set)  # topic -> client_ids
        self.message_buffers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.connection_stats = {
            "total_connections": 0,
            "active_connections": 0,
            "messages_sent": 0,
            "messages_received": 0,
            "errors": 0,
            "disconnections": 0
        }
        
    async def connect(self, websocket: WebSocket, session_id: str, client_id: Optional[str] = None) -> ConnectionInfo:
        """Connect a new WebSocket client."""
        await websocket.accept()
        
        if client_id is None:
            client_id = str(uuid.uuid4())
            
        connection_info = ConnectionInfo(
            websocket=websocket,
            session_id=session_id,
            client_id=client_id,
            connected_at=datetime.utcnow(),
            last_activity=datetime.utcnow(),
            subscriptions=set(),
            processing_mode=StreamingMode.REALTIME,
            batch_size=10,
            batch_timeout=1.0
        )
        
        self.active_connections[client_id] = connection_info
        self.session_connections[session_id].add(client_id)
        
        self.connection_stats["total_connections"] += 1
        self.connection_stats["active_connections"] += 1
        
        logger.info("WebSocket connected", 
                   session_id=session_id, 
                   client_id=client_id,
                   total_active=len(self.active_connections))
        
        return connection_info
    
    def disconnect(self, client_id: str):
        """Disconnect a WebSocket client."""
        if client_id in self.active_connections:
            connection_info = self.active_connections[client_id]
            session_id = connection_info.session_id
            
            # Remove from session connections
            if session_id in self.session_connections:
                self.session_connections[session_id].discard(client_id)
                if not self.session_connections[session_id]:
                    del self.session_connections[session_id]
            
            # Remove from subscriptions
            for topic in connection_info.subscriptions:
                self.subscription_map[topic].discard(client_id)
                if not self.subscription_map[topic]:
                    del self.subscription_map[topic]
            
            del self.active_connections[client_id]
            self.connection_stats["active_connections"] -= 1
            self.connection_stats["disconnections"] += 1
            
            logger.info("WebSocket disconnected", 
                       session_id=session_id, 
                       client_id=client_id,
                       total_active=len(self.active_connections))
    
    async def send_personal_message(self, message: StreamingResponse, client_id: str):
        """Send message to specific client."""
        if client_id in self.active_connections:
            connection_info = self.active_connections[client_id]
            try:
                if connection_info.websocket.client_state == WebSocketState.CONNECTED:
                    await connection_info.websocket.send_text(message.json())
                    connection_info.last_activity = datetime.utcnow()
                    self.connection_stats["messages_sent"] += 1
                else:
                    logger.warning("WebSocket not connected", client_id=client_id)
                    self.disconnect(client_id)
            except Exception as e:
                logger.error("Error sending message", client_id=client_id, error=str(e))
                self.connection_stats["errors"] += 1
                self.disconnect(client_id)
    
    async def broadcast_to_session(self, message: StreamingResponse, session_id: str):
        """Broadcast message to all clients in a session."""
        if session_id in self.session_connections:
            client_ids = list(self.session_connections[session_id])
            await asyncio.gather(
                *[self.send_personal_message(message, client_id) for client_id in client_ids],
                return_exceptions=True
            )
    
    async def broadcast_to_subscribers(self, message: StreamingResponse, topic: str):
        """Broadcast message to all subscribers of a topic."""
        if topic in self.subscription_map:
            client_ids = list(self.subscription_map[topic])
            await asyncio.gather(
                *[self.send_personal_message(message, client_id) for client_id in client_ids],
                return_exceptions=True
            )
    
    def subscribe(self, client_id: str, topic: str):
        """Subscribe client to a topic."""
        if client_id in self.active_connections:
            self.active_connections[client_id].subscriptions.add(topic)
            self.subscription_map[topic].add(client_id)
            logger.info("Client subscribed", client_id=client_id, topic=topic)
    
    def unsubscribe(self, client_id: str, topic: str):
        """Unsubscribe client from a topic."""
        if client_id in self.active_connections:
            self.active_connections[client_id].subscriptions.discard(topic)
            self.subscription_map[topic].discard(client_id)
            if not self.subscription_map[topic]:
                del self.subscription_map[topic]
            logger.info("Client unsubscribed", client_id=client_id, topic=topic)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection manager statistics."""
        return {
            **self.connection_stats,
            "active_sessions": len(self.session_connections),
            "active_subscriptions": len(self.subscription_map),
            "connections_by_session": {
                session_id: len(client_ids) 
                for session_id, client_ids in self.session_connections.items()
            }
        }


class EnhancedStreamingService:
    """Enhanced streaming service with advanced features."""
    
    def __init__(self):
        self.connection_manager = ConnectionManager()
        self.streaming_services: Dict[str, StreamingService] = {}
        self.alert_rules: Dict[str, Dict[str, Any]] = {}
        self.performance_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Background tasks will be started when needed
        self._background_tasks_started = False
        self._cleanup_task = None
        self._monitor_task = None
    
    def _start_background_tasks(self) -> None:
        """Start background tasks if not already started."""
        if not self._background_tasks_started:
            try:
                self._cleanup_task = asyncio.create_task(self._cleanup_inactive_connections())
                self._monitor_task = asyncio.create_task(self._monitor_performance())
                self._background_tasks_started = True
            except RuntimeError:
                # No event loop running, tasks will be started later
                pass
    
    def get_streaming_service(self, session_id: str, **kwargs) -> StreamingService:
        """Get or create streaming service for session."""
        if session_id not in self.streaming_services:
            detection_service = DetectionService()
            self.streaming_services[session_id] = StreamingService(
                detection_service=detection_service,
                window_size=kwargs.get('window_size', 1000),
                update_frequency=kwargs.get('update_frequency', 100)
            )
            logger.info("Created streaming service", session_id=session_id)
        
        return self.streaming_services[session_id]
    
    async def _cleanup_inactive_connections(self):
        """Background task to cleanup inactive connections."""
        while True:
            try:
                current_time = datetime.utcnow()
                inactive_threshold = timedelta(minutes=30)
                
                inactive_clients = []
                for client_id, connection_info in self.connection_manager.active_connections.items():
                    if current_time - connection_info.last_activity > inactive_threshold:
                        inactive_clients.append(client_id)
                
                for client_id in inactive_clients:
                    self.connection_manager.disconnect(client_id)
                    logger.info("Cleaned up inactive connection", client_id=client_id)
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error("Error in cleanup task", error=str(e))
                await asyncio.sleep(60)
    
    async def _monitor_performance(self):
        """Background task to monitor streaming performance."""
        while True:
            try:
                # Collect performance metrics
                stats = self.connection_manager.get_stats()
                timestamp = datetime.utcnow()
                
                # Record metrics
                metrics.record_metric('streaming.active_connections', stats['active_connections'])
                metrics.record_metric('streaming.messages_sent', stats['messages_sent'])
                metrics.record_metric('streaming.errors', stats['errors'])
                
                # Check for performance alerts
                await self._check_performance_alerts(stats)
                
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
            except Exception as e:
                logger.error("Error in performance monitoring", error=str(e))
                await asyncio.sleep(60)
    
    async def _check_performance_alerts(self, stats: Dict[str, Any]):
        """Check for performance-based alerts."""
        # High error rate alert
        error_rate = stats.get('errors', 0) / max(stats.get('messages_sent', 1), 1)
        if error_rate > 0.1:  # 10% error rate
            alert = StreamingAlert(
                alert_id=str(uuid.uuid4()),
                severity="high",
                title="High Error Rate",
                message=f"Streaming error rate is {error_rate:.2%}",
                timestamp=datetime.utcnow().isoformat(),
                source="performance_monitor",
                metadata={"error_rate": error_rate, "stats": stats}
            )
            await self._broadcast_alert(alert)
        
        # High connection count alert
        if stats.get('active_connections', 0) > 1000:
            alert = StreamingAlert(
                alert_id=str(uuid.uuid4()),
                severity="medium",
                title="High Connection Count",
                message=f"Active connections: {stats['active_connections']}",
                timestamp=datetime.utcnow().isoformat(),
                source="performance_monitor",
                metadata={"connection_count": stats['active_connections']}
            )
            await self._broadcast_alert(alert)
    
    async def _broadcast_alert(self, alert: StreamingAlert):
        """Broadcast alert to all subscribers."""
        alert_message = StreamingResponse(
            type=MessageType.ALERT,
            success=True,
            timestamp=datetime.utcnow().isoformat(),
            metadata=asdict(alert)
        )
        await self.connection_manager.broadcast_to_subscribers(alert_message, "alerts")


# Global enhanced streaming service - initialized lazily
enhanced_streaming_service = None

def get_enhanced_streaming_service() -> EnhancedStreamingService:
    """Get or create the enhanced streaming service instance."""
    global enhanced_streaming_service
    if enhanced_streaming_service is None:
        enhanced_streaming_service = EnhancedStreamingService()
    return enhanced_streaming_service


@router.websocket("/enhanced/{session_id}")
async def enhanced_websocket_streaming(websocket: WebSocket, session_id: str, client_id: Optional[str] = None):
    """Enhanced WebSocket endpoint with advanced features."""
    connection_info = None
    
    try:
        # Get service instance and start background tasks if needed
        service = get_enhanced_streaming_service()
        service._start_background_tasks()
        
        # Establish connection
        connection_info = await service.connection_manager.connect(
            websocket, session_id, client_id
        )
        
        # Get streaming service
        streaming_service = service.get_streaming_service(session_id)
        
        # Send welcome message
        welcome_message = StreamingResponse(
            type=MessageType.CONFIG,
            success=True,
            timestamp=datetime.utcnow().isoformat(),
            metadata={
                "session_id": session_id,
                "client_id": connection_info.client_id,
                "server_time": datetime.utcnow().isoformat(),
                "supported_algorithms": ["isolation_forest", "one_class_svm", "lof"],
                "processing_modes": ["realtime", "batch", "adaptive"]
            }
        )
        await service.connection_manager.send_personal_message(
            welcome_message, connection_info.client_id
        )
        
        # Message processing loop
        while True:
            try:
                # Receive message with timeout
                data = await asyncio.wait_for(websocket.receive_text(), timeout=300.0)  # 5 minute timeout
                
                try:
                    message_dict = json.loads(data)
                    message = EnhancedStreamingMessage(**message_dict)
                except (json.JSONDecodeError, ValueError) as e:
                    error_response = StreamingResponse(
                        type=MessageType.ERROR,
                        success=False,
                        timestamp=datetime.utcnow().isoformat(),
                        error=f"Invalid message format: {str(e)}",
                        error_code="INVALID_FORMAT"
                    )
                    await service.connection_manager.send_personal_message(
                        error_response, connection_info.client_id
                    )
                    continue
                
                # Update last activity
                connection_info.last_activity = datetime.utcnow()
                service.connection_manager.connection_stats["messages_received"] += 1
                
                # Process message based on type
                await _process_message(message, connection_info, streaming_service)
                
            except asyncio.TimeoutError:
                # Send ping to keep connection alive
                ping_message = StreamingResponse(
                    type=MessageType.PING,
                    success=True,
                    timestamp=datetime.utcnow().isoformat()
                )
                await service.connection_manager.send_personal_message(
                    ping_message, connection_info.client_id
                )
                
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected", session_id=session_id)
    except Exception as e:
        logger.error("WebSocket error", session_id=session_id, error=str(e))
        if connection_info and websocket.client_state == WebSocketState.CONNECTED:
            error_response = StreamingResponse(
                type=MessageType.ERROR,
                success=False,
                timestamp=datetime.utcnow().isoformat(),
                error=f"Internal server error: {str(e)}",
                error_code="INTERNAL_ERROR"
            )
            try:
                await service.connection_manager.send_personal_message(
                    error_response, connection_info.client_id
                )
            except:
                pass
    finally:
        if connection_info:
            service.connection_manager.disconnect(connection_info.client_id)


async def _process_message(
    message: EnhancedStreamingMessage, 
    connection_info: ConnectionInfo, 
    streaming_service: StreamingService
):
    """Process incoming WebSocket message."""
    try:
        if message.type == MessageType.SAMPLE:
            await _process_sample_message(message, connection_info, streaming_service)
        elif message.type == MessageType.BATCH:
            await _process_batch_message(message, connection_info, streaming_service)
        elif message.type == MessageType.STATS:
            await _process_stats_message(message, connection_info, streaming_service)
        elif message.type == MessageType.DRIFT:
            await _process_drift_message(message, connection_info, streaming_service)
        elif message.type == MessageType.SUBSCRIBE:
            await _process_subscribe_message(message, connection_info)
        elif message.type == MessageType.UNSUBSCRIBE:
            await _process_unsubscribe_message(message, connection_info)
        elif message.type == MessageType.PING:
            await _process_ping_message(message, connection_info)
        elif message.type == MessageType.CONFIG:
            await _process_config_message(message, connection_info)
        else:
            error_response = StreamingResponse(
                type=MessageType.ERROR,
                success=False,
                timestamp=datetime.utcnow().isoformat(),
                error=f"Unknown message type: {message.type}",
                error_code="UNKNOWN_MESSAGE_TYPE",
                request_id=message.request_id
            )
            await get_enhanced_streaming_service().connection_manager.send_personal_message(
                error_response, connection_info.client_id
            )
            
    except Exception as e:
        logger.error("Error processing message", 
                    client_id=connection_info.client_id, 
                    message_type=message.type, 
                    error=str(e))
        
        error_response = StreamingResponse(
            type=MessageType.ERROR,
            success=False,
            timestamp=datetime.utcnow().isoformat(),
            error=f"Message processing failed: {str(e)}",
            error_code="PROCESSING_ERROR",
            request_id=message.request_id
        )
        await get_enhanced_streaming_service().connection_manager.send_personal_message(
            error_response, connection_info.client_id
        )


async def _process_sample_message(
    message: EnhancedStreamingMessage, 
    connection_info: ConnectionInfo, 
    streaming_service: StreamingService
):
    """Process single sample message."""
    if not message.data:
        raise ValueError("Sample data is required")
    
    # Convert to numpy array
    sample_array = np.array(message.data, dtype=np.float64)
    
    # Algorithm mapping
    algorithm_map = {
        'isolation_forest': 'iforest',
        'one_class_svm': 'ocsvm',
        'lof': 'lof'
    }
    mapped_algorithm = algorithm_map.get(message.algorithm, message.algorithm)
    
    # Process sample
    start_time = time.time()
    result = streaming_service.process_sample(sample_array, mapped_algorithm)
    processing_time = time.time() - start_time
    
    # Get stats
    stats = streaming_service.get_streaming_stats()
    
    # Create response
    response = StreamingResponse(
        type=MessageType.RESULT,
        success=result.success,
        request_id=message.request_id,
        timestamp=datetime.utcnow().isoformat(),
        sample_id=str(uuid.uuid4()),
        is_anomaly=bool(result.predictions[0] == -1),
        confidence_score=float(result.confidence_scores[0]) if result.confidence_scores is not None else None,
        algorithm=message.algorithm,
        metadata={
            "processing_time_ms": processing_time * 1000,
            "buffer_size": stats['buffer_size'],
            "model_fitted": stats['model_fitted'],
            "session_id": connection_info.session_id
        }
    )
    
    # Record performance metrics
    metrics.record_metric('streaming.sample_processing_time', processing_time)
    metrics.record_metric('streaming.samples_processed', 1)
    
    await get_enhanced_streaming_service().connection_manager.send_personal_message(
        response, connection_info.client_id
    )


async def _process_batch_message(
    message: EnhancedStreamingMessage, 
    connection_info: ConnectionInfo, 
    streaming_service: StreamingService
):
    """Process batch message."""
    if not message.batch_data:
        raise ValueError("Batch data is required")
    
    # Convert to numpy array
    batch_array = np.array(message.batch_data, dtype=np.float64)
    
    # Algorithm mapping
    algorithm_map = {
        'isolation_forest': 'iforest',
        'one_class_svm': 'ocsvm',
        'lof': 'lof'
    }
    mapped_algorithm = algorithm_map.get(message.algorithm, message.algorithm)
    
    # Process batch
    start_time = time.time()
    result = streaming_service.process_batch(batch_array, mapped_algorithm)
    processing_time = time.time() - start_time
    
    # Get stats
    stats = streaming_service.get_streaming_stats()
    
    # Create results for each sample
    batch_results = []
    for i, (prediction, score) in enumerate(zip(
        result.predictions, 
        result.confidence_scores or [None] * len(result.predictions)
    )):
        batch_results.append({
            "sample_id": str(uuid.uuid4()),
            "sample_index": i,
            "is_anomaly": bool(prediction == -1),
            "confidence_score": float(score) if score is not None else None
        })
    
    # Create response
    response = StreamingResponse(
        type=MessageType.RESULT,
        success=result.success,
        request_id=message.request_id,
        timestamp=datetime.utcnow().isoformat(),
        algorithm=message.algorithm,
        results=batch_results,
        metadata={
            "processing_time_ms": processing_time * 1000,
            "batch_size": len(message.batch_data),
            "buffer_size": stats['buffer_size'],
            "model_fitted": stats['model_fitted'],
            "session_id": connection_info.session_id
        }
    )
    
    # Record performance metrics
    metrics.record_metric('streaming.batch_processing_time', processing_time)
    metrics.record_metric('streaming.batch_size', len(message.batch_data))
    
    await get_enhanced_streaming_service().connection_manager.send_personal_message(
        response, connection_info.client_id
    )


async def _process_stats_message(
    message: EnhancedStreamingMessage, 
    connection_info: ConnectionInfo, 
    streaming_service: StreamingService
):
    """Process statistics request message."""
    stats = streaming_service.get_streaming_stats()
    connection_stats = get_enhanced_streaming_service().connection_manager.get_stats()
    
    response = StreamingResponse(
        type=MessageType.STATS,
        success=True,
        request_id=message.request_id,
        timestamp=datetime.utcnow().isoformat(),
        stats={
            "streaming": stats,
            "connections": connection_stats,
            "session_id": connection_info.session_id,
            "client_id": connection_info.client_id,
            "connected_at": connection_info.connected_at.isoformat(),
            "last_activity": connection_info.last_activity.isoformat()
        }
    )
    
    await get_enhanced_streaming_service().connection_manager.send_personal_message(
        response, connection_info.client_id
    )


async def _process_drift_message(
    message: EnhancedStreamingMessage, 
    connection_info: ConnectionInfo, 
    streaming_service: StreamingService
):
    """Process drift detection message."""
    window_size = message.config.get('window_size', 200) if message.config else 200
    
    drift_result = streaming_service.detect_concept_drift(window_size)
    
    response = StreamingResponse(
        type=MessageType.DRIFT,
        success=True,
        request_id=message.request_id,
        timestamp=datetime.utcnow().isoformat(),
        drift_detected=drift_result.get('drift_detected', False),
        drift_details=drift_result
    )
    
    await get_enhanced_streaming_service().connection_manager.send_personal_message(
        response, connection_info.client_id
    )


async def _process_subscribe_message(message: EnhancedStreamingMessage, connection_info: ConnectionInfo):
    """Process subscription message."""
    topic = message.config.get('topic') if message.config else None
    if not topic:
        raise ValueError("Topic is required for subscription")
    
    get_enhanced_streaming_service().connection_manager.subscribe(connection_info.client_id, topic)
    
    response = StreamingResponse(
        type=MessageType.SUBSCRIBE,
        success=True,
        request_id=message.request_id,
        timestamp=datetime.utcnow().isoformat(),
        metadata={"topic": topic, "subscribed": True}
    )
    
    await get_enhanced_streaming_service().connection_manager.send_personal_message(
        response, connection_info.client_id
    )


async def _process_unsubscribe_message(message: EnhancedStreamingMessage, connection_info: ConnectionInfo):
    """Process unsubscription message."""
    topic = message.config.get('topic') if message.config else None
    if not topic:
        raise ValueError("Topic is required for unsubscription")
    
    get_enhanced_streaming_service().connection_manager.unsubscribe(connection_info.client_id, topic)
    
    response = StreamingResponse(
        type=MessageType.UNSUBSCRIBE,
        success=True,
        request_id=message.request_id,
        timestamp=datetime.utcnow().isoformat(),
        metadata={"topic": topic, "subscribed": False}
    )
    
    await get_enhanced_streaming_service().connection_manager.send_personal_message(
        response, connection_info.client_id
    )


async def _process_ping_message(message: EnhancedStreamingMessage, connection_info: ConnectionInfo):
    """Process ping message."""
    response = StreamingResponse(
        type=MessageType.PONG,
        success=True,
        request_id=message.request_id,
        timestamp=datetime.utcnow().isoformat(),
        metadata={"server_time": datetime.utcnow().isoformat()}
    )
    
    await get_enhanced_streaming_service().connection_manager.send_personal_message(
        response, connection_info.client_id
    )


async def _process_config_message(message: EnhancedStreamingMessage, connection_info: ConnectionInfo):
    """Process configuration message."""
    if message.config:
        # Update connection configuration
        if 'processing_mode' in message.config:
            try:
                connection_info.processing_mode = StreamingMode(message.config['processing_mode'])
            except ValueError:
                raise ValueError(f"Invalid processing mode: {message.config['processing_mode']}")
        
        if 'batch_size' in message.config:
            connection_info.batch_size = max(1, min(1000, int(message.config['batch_size'])))
        
        if 'batch_timeout' in message.config:
            connection_info.batch_timeout = max(0.1, min(60.0, float(message.config['batch_timeout'])))
    
    response = StreamingResponse(
        type=MessageType.CONFIG,
        success=True,
        request_id=message.request_id,
        timestamp=datetime.utcnow().isoformat(),
        metadata={
            "processing_mode": connection_info.processing_mode.value,
            "batch_size": connection_info.batch_size,
            "batch_timeout": connection_info.batch_timeout,
            "updated": True
        }
    )
    
    await get_enhanced_streaming_service().connection_manager.send_personal_message(
        response, connection_info.client_id
    )


# Health check endpoint for WebSocket service
@router.get("/health")
async def streaming_health():
    """Health check for streaming service."""
    stats = get_enhanced_streaming_service().connection_manager.get_stats()
    
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "streaming_service": {
            "active_connections": stats["active_connections"],
            "active_sessions": stats["active_sessions"],
            "total_messages_sent": stats["messages_sent"],
            "total_messages_received": stats["messages_received"],
            "error_rate": stats["errors"] / max(stats["messages_received"], 1)
        },
        "features": {
            "real_time_processing": True,
            "batch_processing": True,
            "concept_drift_detection": True,
            "multi_session_support": True,
            "pub_sub_messaging": True,
            "performance_monitoring": True,
            "automatic_cleanup": True
        }
    }


# Statistics endpoint
@router.get("/stats")
async def get_streaming_service_stats():
    """Get comprehensive streaming service statistics."""
    connection_stats = get_enhanced_streaming_service().connection_manager.get_stats()
    
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "connections": connection_stats,
        "streaming_services": len(get_enhanced_streaming_service().streaming_services),
        "performance": {
            "avg_processing_time_ms": metrics.get_average_metric('streaming.sample_processing_time') * 1000,
            "samples_per_second": metrics.get_rate_metric('streaming.samples_processed'),
            "error_rate": connection_stats["errors"] / max(connection_stats["messages_received"], 1)
        }
    }