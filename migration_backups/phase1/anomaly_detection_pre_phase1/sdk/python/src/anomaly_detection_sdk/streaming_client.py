"""Streaming client for real-time anomaly detection."""

import json
import asyncio
from typing import Dict, List, Optional, Any, Callable, Union
from urllib.parse import urlparse
import threading
from datetime import datetime

import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException

from .models import AnomalyData, StreamingConfig, AlgorithmType
from .exceptions import StreamingError, ValidationError, ConnectionError


class StreamingClient:
    """WebSocket client for real-time anomaly detection."""
    
    def __init__(
        self,
        ws_url: str,
        config: Optional[StreamingConfig] = None,
        api_key: Optional[str] = None,
        auto_reconnect: bool = True,
        reconnect_delay: float = 5.0,
    ):
        """Initialize streaming client.
        
        Args:
            ws_url: WebSocket URL for streaming
            config: Streaming configuration
            api_key: Optional API key for authentication
            auto_reconnect: Whether to automatically reconnect on disconnection
            reconnect_delay: Delay between reconnection attempts
        """
        self.ws_url = ws_url
        self.config = config or StreamingConfig()
        self.api_key = api_key
        self.auto_reconnect = auto_reconnect
        self.reconnect_delay = reconnect_delay
        
        # Connection state
        self.websocket = None
        self.connected = False
        self.running = False
        
        # Event handlers
        self._on_anomaly_handlers: List[Callable[[AnomalyData], None]] = []
        self._on_connect_handlers: List[Callable[[], None]] = []
        self._on_disconnect_handlers: List[Callable[[], None]] = []
        self._on_error_handlers: List[Callable[[Exception], None]] = []
        
        # Threading
        self._loop = None
        self._thread = None
        
        # Buffer for data points
        self._buffer: List[List[float]] = []
        self._buffer_lock = threading.Lock()
    
    def on_anomaly(self, handler: Callable[[AnomalyData], None]):
        """Register handler for anomaly events.
        
        Args:
            handler: Function to call when anomaly is detected
        """
        self._on_anomaly_handlers.append(handler)
        return handler
    
    def on_connect(self, handler: Callable[[], None]):
        """Register handler for connection events.
        
        Args:
            handler: Function to call when connected
        """
        self._on_connect_handlers.append(handler)
        return handler
    
    def on_disconnect(self, handler: Callable[[], None]):
        """Register handler for disconnection events.
        
        Args:
            handler: Function to call when disconnected
        """
        self._on_disconnect_handlers.append(handler)
        return handler
    
    def on_error(self, handler: Callable[[Exception], None]):
        """Register handler for error events.
        
        Args:
            handler: Function to call when error occurs
        """
        self._on_error_handlers.append(handler)
        return handler
    
    def start(self):
        """Start the streaming client in a background thread."""
        if self.running:
            return
        
        self.running = True
        self._thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self._thread.start()
    
    def stop(self):
        """Stop the streaming client."""
        self.running = False
        if self._loop and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._stop_async)
        
        if self._thread:
            self._thread.join(timeout=5.0)
    
    def send_data(self, data_point: List[float]):
        """Send a data point for anomaly detection.
        
        Args:
            data_point: List of feature values
        """
        if not isinstance(data_point, list):
            raise ValidationError("Data point must be a list", "data_point", data_point)
        
        with self._buffer_lock:
            self._buffer.append(data_point)
            
            # Process batch when buffer is full
            if len(self._buffer) >= self.config.batch_size:
                batch = self._buffer.copy()
                self._buffer.clear()
                
                if self._loop and not self._loop.is_closed():
                    asyncio.run_coroutine_threadsafe(
                        self._send_batch(batch),
                        self._loop
                    )
    
    def _run_event_loop(self):
        """Run the asyncio event loop in the background thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        
        try:
            self._loop.run_until_complete(self._connect_and_listen())
        except Exception as e:
            self._handle_error(e)
        finally:
            self._loop.close()
    
    async def _connect_and_listen(self):
        """Connect to WebSocket and start listening."""
        while self.running:
            try:
                await self._connect()
                await self._listen()
            except Exception as e:
                self._handle_error(e)
                
                if self.auto_reconnect and self.running:
                    await asyncio.sleep(self.reconnect_delay)
                else:
                    break
    
    async def _connect(self):
        """Establish WebSocket connection."""
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        try:
            self.websocket = await websockets.connect(
                self.ws_url,
                extra_headers=headers,
                ping_interval=20,
                ping_timeout=10,
            )
            
            # Send initial configuration
            config_message = {
                "type": "config",
                "config": self.config.model_dump()
            }
            await self.websocket.send(json.dumps(config_message))
            
            self.connected = True
            self._handle_connect()
            
        except Exception as e:
            raise ConnectionError(f"Failed to connect to {self.ws_url}: {str(e)}")
    
    async def _listen(self):
        """Listen for messages from the WebSocket."""
        try:
            async for message in self.websocket:
                await self._handle_message(message)
        except ConnectionClosed:
            self.connected = False
            self._handle_disconnect()
        except WebSocketException as e:
            raise StreamingError(f"WebSocket error: {str(e)}")
    
    async def _handle_message(self, message: str):
        """Handle incoming WebSocket message."""
        try:
            data = json.loads(message)
            message_type = data.get("type")
            
            if message_type == "anomaly":
                anomaly_data = AnomalyData(**data["data"])
                self._handle_anomaly(anomaly_data)
            elif message_type == "error":
                error_msg = data.get("message", "Unknown error")
                self._handle_error(StreamingError(error_msg))
            elif message_type == "ping":
                await self.websocket.send(json.dumps({"type": "pong"}))
                
        except json.JSONDecodeError:
            self._handle_error(StreamingError(f"Invalid JSON message: {message}"))
        except Exception as e:
            self._handle_error(StreamingError(f"Error handling message: {str(e)}"))
    
    async def _send_batch(self, batch: List[List[float]]):
        """Send a batch of data points."""
        if not self.connected or not self.websocket:
            return
        
        try:
            message = {
                "type": "batch",
                "data": batch,
                "timestamp": datetime.utcnow().isoformat()
            }
            await self.websocket.send(json.dumps(message))
        except Exception as e:
            self._handle_error(StreamingError(f"Failed to send batch: {str(e)}"))
    
    async def _stop_async(self):
        """Stop the async components."""
        if self.websocket:
            await self.websocket.close()
        self.connected = False
    
    def _handle_anomaly(self, anomaly_data: AnomalyData):
        """Handle anomaly detection."""
        for handler in self._on_anomaly_handlers:
            try:
                handler(anomaly_data)
            except Exception as e:
                self._handle_error(e)
    
    def _handle_connect(self):
        """Handle connection established."""
        for handler in self._on_connect_handlers:
            try:
                handler()
            except Exception as e:
                self._handle_error(e)
    
    def _handle_disconnect(self):
        """Handle disconnection."""
        for handler in self._on_disconnect_handlers:
            try:
                handler()
            except Exception as e:
                self._handle_error(e)
    
    def _handle_error(self, error: Exception):
        """Handle errors."""
        for handler in self._on_error_handlers:
            try:
                handler(error)
            except Exception:
                pass  # Ignore errors in error handlers
    
    @property
    def is_connected(self) -> bool:
        """Check if the client is connected."""
        return self.connected
    
    @property
    def buffer_size(self) -> int:
        """Get current buffer size."""
        with self._buffer_lock:
            return len(self._buffer)


class AsyncStreamingClient:
    """Async version of the streaming client."""
    
    def __init__(
        self,
        ws_url: str,
        config: Optional[StreamingConfig] = None,
        api_key: Optional[str] = None,
        auto_reconnect: bool = True,
        reconnect_delay: float = 5.0,
    ):
        """Initialize async streaming client."""
        self.ws_url = ws_url
        self.config = config or StreamingConfig()
        self.api_key = api_key
        self.auto_reconnect = auto_reconnect
        self.reconnect_delay = reconnect_delay
        
        # Connection state
        self.websocket = None
        self.connected = False
        self.running = False
        
        # Event handlers
        self._on_anomaly_handlers: List[Callable[[AnomalyData], None]] = []
        self._on_connect_handlers: List[Callable[[], None]] = []
        self._on_disconnect_handlers: List[Callable[[], None]] = []
        self._on_error_handlers: List[Callable[[Exception], None]] = []
        
        # Buffer for data points
        self._buffer: List[List[float]] = []
    
    def on_anomaly(self, handler: Callable[[AnomalyData], None]):
        """Register handler for anomaly events."""
        self._on_anomaly_handlers.append(handler)
        return handler
    
    def on_connect(self, handler: Callable[[], None]):
        """Register handler for connection events."""
        self._on_connect_handlers.append(handler)
        return handler
    
    def on_disconnect(self, handler: Callable[[], None]):
        """Register handler for disconnection events."""
        self._on_disconnect_handlers.append(handler)
        return handler
    
    def on_error(self, handler: Callable[[Exception], None]):
        """Register handler for error events."""
        self._on_error_handlers.append(handler)
        return handler
    
    async def start(self):
        """Start the streaming client."""
        if self.running:
            return
        
        self.running = True
        await self._connect_and_listen()
    
    async def stop(self):
        """Stop the streaming client."""
        self.running = False
        if self.websocket:
            await self.websocket.close()
        self.connected = False
    
    async def send_data(self, data_point: List[float]):
        """Send a data point for anomaly detection."""
        if not isinstance(data_point, list):
            raise ValidationError("Data point must be a list", "data_point", data_point)
        
        self._buffer.append(data_point)
        
        # Process batch when buffer is full
        if len(self._buffer) >= self.config.batch_size:
            batch = self._buffer.copy()
            self._buffer.clear()
            await self._send_batch(batch)
    
    async def _connect_and_listen(self):
        """Connect and listen for messages."""
        while self.running:
            try:
                await self._connect()
                await self._listen()
            except Exception as e:
                self._handle_error(e)
                
                if self.auto_reconnect and self.running:
                    await asyncio.sleep(self.reconnect_delay)
                else:
                    break
    
    async def _connect(self):
        """Establish WebSocket connection."""
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        try:
            self.websocket = await websockets.connect(
                self.ws_url,
                extra_headers=headers,
                ping_interval=20,
                ping_timeout=10,
            )
            
            # Send initial configuration
            config_message = {
                "type": "config",
                "config": self.config.model_dump()
            }
            await self.websocket.send(json.dumps(config_message))
            
            self.connected = True
            self._handle_connect()
            
        except Exception as e:
            raise ConnectionError(f"Failed to connect to {self.ws_url}: {str(e)}")
    
    async def _listen(self):
        """Listen for messages from the WebSocket."""
        try:
            async for message in self.websocket:
                await self._handle_message(message)
        except ConnectionClosed:
            self.connected = False
            self._handle_disconnect()
        except WebSocketException as e:
            raise StreamingError(f"WebSocket error: {str(e)}")
    
    async def _handle_message(self, message: str):
        """Handle incoming WebSocket message."""
        try:
            data = json.loads(message)
            message_type = data.get("type")
            
            if message_type == "anomaly":
                anomaly_data = AnomalyData(**data["data"])
                self._handle_anomaly(anomaly_data)
            elif message_type == "error":
                error_msg = data.get("message", "Unknown error")
                self._handle_error(StreamingError(error_msg))
            elif message_type == "ping":
                await self.websocket.send(json.dumps({"type": "pong"}))
                
        except json.JSONDecodeError:
            self._handle_error(StreamingError(f"Invalid JSON message: {message}"))
        except Exception as e:
            self._handle_error(StreamingError(f"Error handling message: {str(e)}"))
    
    async def _send_batch(self, batch: List[List[float]]):
        """Send a batch of data points."""
        if not self.connected or not self.websocket:
            return
        
        try:
            message = {
                "type": "batch",
                "data": batch,
                "timestamp": datetime.utcnow().isoformat()
            }
            await self.websocket.send(json.dumps(message))
        except Exception as e:
            self._handle_error(StreamingError(f"Failed to send batch: {str(e)}"))
    
    def _handle_anomaly(self, anomaly_data: AnomalyData):
        """Handle anomaly detection."""
        for handler in self._on_anomaly_handlers:
            try:
                handler(anomaly_data)
            except Exception as e:
                self._handle_error(e)
    
    def _handle_connect(self):
        """Handle connection established."""
        for handler in self._on_connect_handlers:
            try:
                handler()
            except Exception as e:
                self._handle_error(e)
    
    def _handle_disconnect(self):
        """Handle disconnection."""
        for handler in self._on_disconnect_handlers:
            try:
                handler()
            except Exception as e:
                self._handle_error(e)
    
    def _handle_error(self, error: Exception):
        """Handle errors."""
        for handler in self._on_error_handlers:
            try:
                handler(error)
            except Exception:
                pass  # Ignore errors in error handlers