"""WebSocket client for real-time streaming anomaly detection."""

import asyncio
import json
import time
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, Callable, List, AsyncGenerator
from enum import Enum
from contextlib import asynccontextmanager

import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException
import numpy as np
from pydantic import BaseModel

from ..infrastructure.logging import get_logger

logger = get_logger(__name__)


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


class StreamingClientConfig(BaseModel):
    """Configuration for streaming WebSocket client."""
    url: str = "ws://localhost:8000/api/v1/streaming/enhanced/default"
    session_id: str = "default"
    client_id: Optional[str] = None
    reconnect_attempts: int = 5
    reconnect_delay: float = 1.0
    ping_interval: float = 30.0
    ping_timeout: float = 10.0
    message_timeout: float = 30.0
    max_queue_size: int = 1000
    processing_mode: str = "realtime"
    batch_size: int = 10
    batch_timeout: float = 1.0


class StreamingMessage(BaseModel):
    """Streaming message structure."""
    type: MessageType
    data: Optional[List[float]] = None
    batch_data: Optional[List[List[float]]] = None
    algorithm: str = "isolation_forest"
    timestamp: Optional[str] = None
    session_id: Optional[str] = None
    client_id: Optional[str] = None
    request_id: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class StreamingResponse(BaseModel):
    """Streaming response structure."""
    type: MessageType
    success: bool
    request_id: Optional[str] = None
    timestamp: str
    sample_id: Optional[str] = None
    is_anomaly: Optional[bool] = None
    confidence_score: Optional[float] = None
    algorithm: Optional[str] = None
    results: Optional[List[Dict[str, Any]]] = None
    stats: Optional[Dict[str, Any]] = None
    drift_detected: Optional[bool] = None
    drift_details: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    error_code: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class StreamingWebSocketClient:
    """Advanced WebSocket client for real-time anomaly detection streaming."""
    
    def __init__(self, config: StreamingClientConfig):
        self.config = config
        self.websocket: Optional[websockets.WebSocketServerProtocol] = None
        self.is_connected = False
        self.is_connecting = False
        self.reconnect_count = 0
        self.message_queue = asyncio.Queue(maxsize=config.max_queue_size)
        self.response_callbacks: Dict[str, Callable] = {}
        self.subscription_callbacks: Dict[str, Callable] = {}
        self.error_callbacks: List[Callable] = []
        self.connection_callbacks: List[Callable] = []
        self.disconnection_callbacks: List[Callable] = []
        
        # Statistics
        self.stats = {
            "messages_sent": 0,
            "messages_received": 0,
            "errors": 0,
            "reconnections": 0,
            "samples_processed": 0,
            "anomalies_detected": 0,
            "connection_time": None,
            "last_message_time": None
        }
        
        # Background tasks
        self._background_tasks: List[asyncio.Task] = []
        self._shutdown_event = asyncio.Event()
    
    async def connect(self) -> bool:
        """Connect to WebSocket server."""
        if self.is_connected or self.is_connecting:
            return self.is_connected
        
        self.is_connecting = True
        
        try:
            logger.info("Connecting to WebSocket", url=self.config.url)
            
            # Establish WebSocket connection
            self.websocket = await websockets.connect(
                self.config.url,
                ping_interval=self.config.ping_interval,
                ping_timeout=self.config.ping_timeout,
                max_size=10**7,  # 10MB max message size
                max_queue=32
            )
            
            self.is_connected = True
            self.is_connecting = False
            self.reconnect_count = 0
            self.stats["connection_time"] = datetime.utcnow()
            
            logger.info("WebSocket connected successfully")
            
            # Start background tasks
            self._start_background_tasks()
            
            # Notify connection callbacks
            for callback in self.connection_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback()
                    else:
                        callback()
                except Exception as e:
                    logger.error("Error in connection callback", error=str(e))
            
            return True
            
        except Exception as e:
            logger.error("Failed to connect to WebSocket", error=str(e))
            self.is_connecting = False
            self.stats["errors"] += 1
            return False
    
    async def disconnect(self):
        """Disconnect from WebSocket server."""
        logger.info("Disconnecting from WebSocket")
        
        # Signal shutdown
        self._shutdown_event.set()
        
        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        # Close WebSocket connection
        if self.websocket and not self.websocket.closed:
            await self.websocket.close()
        
        self.is_connected = False
        self.websocket = None
        
        # Notify disconnection callbacks
        for callback in self.disconnection_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback()
                else:
                    callback()
            except Exception as e:
                logger.error("Error in disconnection callback", error=str(e))
        
        logger.info("WebSocket disconnected")
    
    def _start_background_tasks(self):
        """Start background tasks for message handling."""
        self._background_tasks = [
            asyncio.create_task(self._message_receiver()),
            asyncio.create_task(self._message_sender()),
            asyncio.create_task(self._ping_sender())
        ]
    
    async def _message_receiver(self):
        """Background task to receive messages from WebSocket."""
        try:
            while self.is_connected and not self._shutdown_event.is_set():
                try:
                    message = await asyncio.wait_for(
                        self.websocket.recv(),
                        timeout=self.config.message_timeout
                    )
                    
                    await self._handle_received_message(message)
                    
                except asyncio.TimeoutError:
                    logger.debug("Message receive timeout")
                    continue
                except ConnectionClosed:
                    logger.info("WebSocket connection closed")
                    break
                except Exception as e:
                    logger.error("Error receiving message", error=str(e))
                    self.stats["errors"] += 1
                    break
                    
        except Exception as e:
            logger.error("Message receiver task error", error=str(e))
        finally:
            await self._handle_disconnection()
    
    async def _message_sender(self):
        """Background task to send queued messages."""
        try:
            while self.is_connected and not self._shutdown_event.is_set():
                try:
                    # Get message from queue with timeout
                    message = await asyncio.wait_for(
                        self.message_queue.get(),
                        timeout=1.0
                    )
                    
                    if self.websocket and not self.websocket.closed:
                        await self.websocket.send(message)
                        self.stats["messages_sent"] += 1
                        self.stats["last_message_time"] = datetime.utcnow()
                        logger.debug("Message sent", message_preview=message[:100])
                    
                    self.message_queue.task_done()
                    
                except asyncio.TimeoutError:
                    continue
                except ConnectionClosed:
                    logger.info("WebSocket connection closed during send")
                    break
                except Exception as e:
                    logger.error("Error sending message", error=str(e))
                    self.stats["errors"] += 1
                    
        except Exception as e:
            logger.error("Message sender task error", error=str(e))
    
    async def _ping_sender(self):
        """Background task to send periodic pings."""
        try:
            while self.is_connected and not self._shutdown_event.is_set():
                try:
                    await asyncio.sleep(self.config.ping_interval)
                    
                    if self.is_connected:
                        await self.send_ping()
                        
                except Exception as e:
                    logger.error("Error sending ping", error=str(e))
                    
        except Exception as e:
            logger.error("Ping sender task error", error=str(e))
    
    async def _handle_received_message(self, raw_message: str):
        """Handle received message from WebSocket."""
        try:
            message_data = json.loads(raw_message)
            response = StreamingResponse(**message_data)
            
            self.stats["messages_received"] += 1
            
            logger.debug("Message received", 
                        message_type=response.type, 
                        success=response.success)
            
            # Handle different message types
            if response.type == MessageType.RESULT:
                await self._handle_result_message(response)
            elif response.type == MessageType.STATS:
                await self._handle_stats_message(response)
            elif response.type == MessageType.DRIFT:
                await self._handle_drift_message(response)
            elif response.type == MessageType.ALERT:
                await self._handle_alert_message(response)
            elif response.type == MessageType.ERROR:
                await self._handle_error_message(response)
            elif response.type == MessageType.PONG:
                logger.debug("Pong received")
            elif response.type == MessageType.CONFIG:
                await self._handle_config_message(response)
            
            # Call request-specific callback
            if response.request_id and response.request_id in self.response_callbacks:
                callback = self.response_callbacks.pop(response.request_id)
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(response)
                    else:
                        callback(response)
                except Exception as e:
                    logger.error("Error in response callback", error=str(e))
            
        except Exception as e:
            logger.error("Error handling received message", error=str(e))
            self.stats["errors"] += 1
    
    async def _handle_result_message(self, response: StreamingResponse):
        """Handle anomaly detection result message."""
        if response.is_anomaly:
            self.stats["anomalies_detected"] += 1
            logger.info("Anomaly detected", 
                       sample_id=response.sample_id,
                       confidence=response.confidence_score,
                       algorithm=response.algorithm)
        
        self.stats["samples_processed"] += 1
        
        # Call subscription callbacks for results
        for topic, callback in self.subscription_callbacks.items():
            if topic in ["results", "anomalies"] or (topic == "anomalies" and response.is_anomaly):
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(response)
                    else:
                        callback(response)
                except Exception as e:
                    logger.error("Error in subscription callback", topic=topic, error=str(e))
    
    async def _handle_stats_message(self, response: StreamingResponse):
        """Handle statistics message."""
        logger.debug("Statistics received", stats=response.stats)
        
        # Call subscription callbacks for stats
        for topic, callback in self.subscription_callbacks.items():
            if topic == "stats":
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(response)
                    else:
                        callback(response)
                except Exception as e:
                    logger.error("Error in stats callback", error=str(e))
    
    async def _handle_drift_message(self, response: StreamingResponse):
        """Handle concept drift message."""
        if response.drift_detected:
            logger.warning("Concept drift detected", details=response.drift_details)
        
        # Call subscription callbacks for drift
        for topic, callback in self.subscription_callbacks.items():
            if topic == "drift":
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(response)
                    else:
                        callback(response)
                except Exception as e:
                    logger.error("Error in drift callback", error=str(e))
    
    async def _handle_alert_message(self, response: StreamingResponse):
        """Handle alert message."""
        alert_data = response.metadata
        if alert_data:
            logger.warning("Alert received", 
                          severity=alert_data.get("severity"),
                          title=alert_data.get("title"),
                          message=alert_data.get("message"))
        
        # Call subscription callbacks for alerts
        for topic, callback in self.subscription_callbacks.items():
            if topic == "alerts":
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(response)
                    else:
                        callback(response)
                except Exception as e:
                    logger.error("Error in alert callback", error=str(e))
    
    async def _handle_error_message(self, response: StreamingResponse):
        """Handle error message."""
        logger.error("Server error received", 
                     error=response.error,
                     error_code=response.error_code)
        
        self.stats["errors"] += 1
        
        # Call error callbacks
        for callback in self.error_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(response)
                else:
                    callback(response)
            except Exception as e:
                logger.error("Error in error callback", error=str(e))
    
    async def _handle_config_message(self, response: StreamingResponse):
        """Handle configuration message."""
        logger.info("Configuration received", metadata=response.metadata)
    
    async def _handle_disconnection(self):
        """Handle WebSocket disconnection."""
        if not self.is_connected:
            return
        
        self.is_connected = False
        logger.warning("WebSocket disconnected unexpectedly")
        
        # Attempt reconnection
        if self.reconnect_count < self.config.reconnect_attempts:
            self.reconnect_count += 1
            self.stats["reconnections"] += 1
            
            logger.info("Attempting reconnection", 
                       attempt=self.reconnect_count,
                       max_attempts=self.config.reconnect_attempts)
            
            await asyncio.sleep(self.config.reconnect_delay * self.reconnect_count)
            
            if await self.connect():
                logger.info("Reconnection successful")
            else:
                logger.error("Reconnection failed")
        else:
            logger.error("Max reconnection attempts reached")
    
    async def send_sample(self, 
                         data: List[float], 
                         algorithm: str = "isolation_forest",
                         callback: Optional[Callable] = None) -> Optional[str]:
        """Send single sample for anomaly detection."""
        if not self.is_connected:
            raise RuntimeError("WebSocket not connected")
        
        request_id = str(uuid.uuid4())
        
        message = StreamingMessage(
            type=MessageType.SAMPLE,
            data=data,
            algorithm=algorithm,
            request_id=request_id,
            timestamp=datetime.utcnow().isoformat(),
            session_id=self.config.session_id,
            client_id=self.config.client_id
        )
        
        if callback:
            self.response_callbacks[request_id] = callback
        
        try:
            await self.message_queue.put(message.json())
            return request_id
        except asyncio.QueueFull:
            logger.error("Message queue full")
            raise RuntimeError("Message queue full")
    
    async def send_batch(self, 
                        batch_data: List[List[float]], 
                        algorithm: str = "isolation_forest",
                        callback: Optional[Callable] = None) -> Optional[str]:
        """Send batch of samples for anomaly detection."""
        if not self.is_connected:
            raise RuntimeError("WebSocket not connected")
        
        request_id = str(uuid.uuid4())
        
        message = StreamingMessage(
            type=MessageType.BATCH,
            batch_data=batch_data,
            algorithm=algorithm,
            request_id=request_id,
            timestamp=datetime.utcnow().isoformat(),
            session_id=self.config.session_id,
            client_id=self.config.client_id
        )
        
        if callback:
            self.response_callbacks[request_id] = callback
        
        try:
            await self.message_queue.put(message.json())
            return request_id
        except asyncio.QueueFull:
            logger.error("Message queue full")
            raise RuntimeError("Message queue full")
    
    async def request_stats(self, callback: Optional[Callable] = None) -> Optional[str]:
        """Request streaming statistics."""
        if not self.is_connected:
            raise RuntimeError("WebSocket not connected")
        
        request_id = str(uuid.uuid4())
        
        message = StreamingMessage(
            type=MessageType.STATS,
            request_id=request_id,
            timestamp=datetime.utcnow().isoformat()
        )
        
        if callback:
            self.response_callbacks[request_id] = callback
        
        try:
            await self.message_queue.put(message.json())
            return request_id
        except asyncio.QueueFull:
            logger.error("Message queue full")
            raise RuntimeError("Message queue full")
    
    async def check_drift(self, 
                         window_size: int = 200, 
                         callback: Optional[Callable] = None) -> Optional[str]:
        """Check for concept drift."""
        if not self.is_connected:
            raise RuntimeError("WebSocket not connected")
        
        request_id = str(uuid.uuid4())
        
        message = StreamingMessage(
            type=MessageType.DRIFT,
            request_id=request_id,
            config={"window_size": window_size},
            timestamp=datetime.utcnow().isoformat()
        )
        
        if callback:
            self.response_callbacks[request_id] = callback
        
        try:
            await self.message_queue.put(message.json())
            return request_id
        except asyncio.QueueFull:
            logger.error("Message queue full")
            raise RuntimeError("Message queue full")
    
    async def subscribe(self, topic: str, callback: Callable):
        """Subscribe to a topic with callback."""
        if not self.is_connected:
            raise RuntimeError("WebSocket not connected")
        
        request_id = str(uuid.uuid4())
        
        message = StreamingMessage(
            type=MessageType.SUBSCRIBE,
            request_id=request_id,
            config={"topic": topic},
            timestamp=datetime.utcnow().isoformat()
        )
        
        self.subscription_callbacks[topic] = callback
        
        try:
            await self.message_queue.put(message.json())
            return request_id
        except asyncio.QueueFull:
            logger.error("Message queue full")
            raise RuntimeError("Message queue full")
    
    async def unsubscribe(self, topic: str):
        """Unsubscribe from a topic."""
        if not self.is_connected:
            raise RuntimeError("WebSocket not connected")
        
        request_id = str(uuid.uuid4())
        
        message = StreamingMessage(
            type=MessageType.UNSUBSCRIBE,
            request_id=request_id,
            config={"topic": topic},
            timestamp=datetime.utcnow().isoformat()
        )
        
        if topic in self.subscription_callbacks:
            del self.subscription_callbacks[topic]
        
        try:
            await self.message_queue.put(message.json())
            return request_id
        except asyncio.QueueFull:
            logger.error("Message queue full")
            raise RuntimeError("Message queue full")
    
    async def send_ping(self):
        """Send ping message."""
        if not self.is_connected:
            return
        
        message = StreamingMessage(
            type=MessageType.PING,
            timestamp=datetime.utcnow().isoformat()
        )
        
        try:
            await self.message_queue.put(message.json())
        except asyncio.QueueFull:
            logger.warning("Cannot send ping - message queue full")
    
    def add_error_callback(self, callback: Callable):
        """Add error callback."""
        self.error_callbacks.append(callback)
    
    def add_connection_callback(self, callback: Callable):
        """Add connection callback."""
        self.connection_callbacks.append(callback)
    
    def add_disconnection_callback(self, callback: Callable):
        """Add disconnection callback."""
        self.disconnection_callbacks.append(callback)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics."""
        return {
            **self.stats,
            "is_connected": self.is_connected,
            "reconnect_count": self.reconnect_count,
            "queue_size": self.message_queue.qsize(),
            "active_callbacks": len(self.response_callbacks),
            "subscriptions": list(self.subscription_callbacks.keys())
        }
    
    @asynccontextmanager
    async def streaming_session(self):
        """Context manager for streaming session."""
        try:
            if not await self.connect():
                raise RuntimeError("Failed to connect to WebSocket")
            yield self
        finally:
            await self.disconnect()


# Convenience functions

async def create_streaming_client(url: str = "ws://localhost:8000/api/v1/streaming/enhanced/default",
                                session_id: str = "default",
                                **kwargs) -> StreamingWebSocketClient:
    """Create and connect streaming client."""
    config = StreamingClientConfig(
        url=url,
        session_id=session_id,
        **kwargs
    )
    
    client = StreamingWebSocketClient(config)
    await client.connect()
    return client


async def stream_samples(samples: AsyncGenerator[List[float], None],
                        algorithm: str = "isolation_forest",
                        url: str = "ws://localhost:8000/api/v1/streaming/enhanced/default",
                        session_id: str = "default") -> AsyncGenerator[StreamingResponse, None]:
    """Stream samples and yield results."""
    config = StreamingClientConfig(url=url, session_id=session_id)
    
    async with StreamingWebSocketClient(config).streaming_session() as client:
        async for sample in samples:
            response_received = asyncio.Event()
            result = None
            
            def callback(response: StreamingResponse):
                nonlocal result
                result = response
                response_received.set()
            
            await client.send_sample(sample, algorithm, callback)
            await response_received.wait()
            
            if result:
                yield result