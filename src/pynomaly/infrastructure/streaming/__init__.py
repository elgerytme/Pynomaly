"""
Streaming data processing infrastructure for real-time anomaly detection.

This module provides comprehensive streaming capabilities including:
- Real-time data processing with windowing
- Backpressure handling and flow control
- Multiple data source connectors (Kafka, Redis, RabbitMQ, WebSocket, etc.)
- Distributed processing support
- Metrics and monitoring
"""

from .streaming_processor import (
    StreamingProcessor,
    StreamingService,
    StreamRecord,
    StreamWindow,
    StreamMetrics,
    StreamState,
    WindowType,
    BackpressureHandler
)

from .connectors import (
    StreamingConnector,
    ConnectorFactory,
    ConnectorConfig,
    ConnectorType,
    KafkaConnector,
    RedisStreamsConnector,
    RabbitMQConnector,
    WebSocketConnector,
    HTTPStreamConnector
)

__all__ = [
    # Core streaming classes
    "StreamingProcessor",
    "StreamingService", 
    "StreamRecord",
    "StreamWindow",
    "StreamMetrics",
    "StreamState",
    "WindowType",
    "BackpressureHandler",
    
    # Connector classes
    "StreamingConnector",
    "ConnectorFactory",
    "ConnectorConfig", 
    "ConnectorType",
    "KafkaConnector",
    "RedisStreamsConnector",
    "RabbitMQConnector",
    "WebSocketConnector",
    "HTTPStreamConnector"
]