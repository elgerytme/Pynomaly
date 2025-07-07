"""
Streaming data connectors for various data sources.

This module provides connectors for popular streaming platforms like Kafka,
RabbitMQ, Redis Streams, and cloud services.
"""

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, AsyncIterator, Dict, List, Optional, Union
from enum import Enum

from pynomaly.infrastructure.streaming.streaming_processor import StreamRecord
from pynomaly.shared.exceptions import StreamingError, ConnectionError
from pynomaly.shared.types import TenantId

logger = logging.getLogger(__name__)


class ConnectorType(str, Enum):
    """Types of streaming connectors."""
    KAFKA = "kafka"
    REDIS_STREAMS = "redis_streams"
    RABBITMQ = "rabbitmq"
    AWS_KINESIS = "aws_kinesis"
    AZURE_EVENT_HUBS = "azure_event_hubs"
    GCP_PUBSUB = "gcp_pubsub"
    WEBSOCKET = "websocket"
    HTTP_STREAM = "http_stream"
    FILE_TAIL = "file_tail"


@dataclass
class ConnectorConfig:
    """Configuration for streaming connectors."""
    connector_type: ConnectorType
    connection_params: Dict[str, Any]
    serialization_format: str = "json"  # json, avro, protobuf
    batch_size: int = 100
    max_wait_time: float = 1.0  # seconds
    retry_attempts: int = 3
    retry_delay: float = 1.0
    health_check_interval: float = 30.0


class StreamingConnector(ABC):
    """Abstract base class for streaming data connectors."""
    
    def __init__(self, config: ConnectorConfig):
        self.config = config
        self.is_connected = False
        self.is_consuming = False
        self._stop_event = asyncio.Event()
        
    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the data source."""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to the data source."""
        pass
    
    @abstractmethod
    async def consume(self) -> AsyncIterator[StreamRecord]:
        """Consume records from the data source."""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if connection is healthy."""
        pass
    
    def _deserialize_record(self, raw_data: bytes, metadata: Dict[str, Any]) -> StreamRecord:
        """Deserialize raw data to StreamRecord."""
        try:
            if self.config.serialization_format == "json":
                data = json.loads(raw_data.decode('utf-8'))
            else:
                # For other formats, you'd implement specific deserializers
                data = {"raw": raw_data.decode('utf-8')}
            
            # Extract tenant_id from metadata or data
            tenant_id = TenantId(metadata.get('tenant_id', data.get('tenant_id', 'default')))
            
            record = StreamRecord(
                id=metadata.get('id', f"record_{int(time.time() * 1000000)}"),
                timestamp=datetime.utcnow(),
                data=data,
                tenant_id=tenant_id,
                metadata=metadata
            )
            
            return record
            
        except Exception as e:
            logger.error(f"Failed to deserialize record: {e}")
            raise StreamingError(f"Deserialization failed: {e}")


class KafkaConnector(StreamingConnector):
    """Kafka streaming connector."""
    
    def __init__(self, config: ConnectorConfig):
        super().__init__(config)
        self.consumer = None
        self._kafka_available = False
        
        # Check if kafka library is available
        try:
            import aiokafka
            self._kafka_available = True
        except ImportError:
            logger.warning("aiokafka not available - Kafka connector disabled")
    
    async def connect(self) -> None:
        """Connect to Kafka."""
        if not self._kafka_available:
            raise StreamingError("Kafka library (aiokafka) not available")
        
        from aiokafka import AIOKafkaConsumer
        
        try:
            connection_params = self.config.connection_params
            
            self.consumer = AIOKafkaConsumer(
                *connection_params.get('topics', ['anomaly-data']),
                bootstrap_servers=connection_params.get('bootstrap_servers', 'localhost:9092'),
                group_id=connection_params.get('group_id', 'pynomaly-consumer'),
                value_deserializer=lambda m: m,  # We'll handle deserialization ourselves
                key_deserializer=lambda m: m.decode('utf-8') if m else None,
                enable_auto_commit=connection_params.get('auto_commit', True),
                auto_offset_reset=connection_params.get('auto_offset_reset', 'latest'),
                session_timeout_ms=connection_params.get('session_timeout_ms', 30000),
                heartbeat_interval_ms=connection_params.get('heartbeat_interval_ms', 10000)
            )
            
            await self.consumer.start()
            self.is_connected = True
            logger.info("Connected to Kafka")
            
        except Exception as e:
            logger.error(f"Failed to connect to Kafka: {e}")
            raise ConnectionError(f"Kafka connection failed: {e}")
    
    async def disconnect(self) -> None:
        """Disconnect from Kafka."""
        if self.consumer:
            await self.consumer.stop()
            self.consumer = None
        self.is_connected = False
        logger.info("Disconnected from Kafka")
    
    async def consume(self) -> AsyncIterator[StreamRecord]:
        """Consume records from Kafka."""
        if not self.is_connected or not self.consumer:
            raise StreamingError("Not connected to Kafka")
        
        self.is_consuming = True
        
        try:
            async for message in self.consumer:
                if self._stop_event.is_set():
                    break
                
                metadata = {
                    'topic': message.topic,
                    'partition': message.partition,
                    'offset': message.offset,
                    'timestamp': message.timestamp,
                    'key': message.key.decode('utf-8') if message.key else None
                }
                
                record = self._deserialize_record(message.value, metadata)
                yield record
                
        except Exception as e:
            logger.error(f"Error consuming from Kafka: {e}")
            raise StreamingError(f"Kafka consumption failed: {e}")
        finally:
            self.is_consuming = False
    
    async def health_check(self) -> bool:
        """Check Kafka connection health."""
        if not self.is_connected or not self.consumer:
            return False
        
        try:
            # Simple check - try to get metadata
            metadata = await self.consumer.client.list_consumer_groups()
            return True
        except Exception as e:
            logger.warning(f"Kafka health check failed: {e}")
            return False


class RedisStreamsConnector(StreamingConnector):
    """Redis Streams connector."""
    
    def __init__(self, config: ConnectorConfig):
        super().__init__(config)
        self.redis_client = None
        self._redis_available = False
        
        try:
            import aioredis
            self._redis_available = True
        except ImportError:
            logger.warning("aioredis not available - Redis Streams connector disabled")
    
    async def connect(self) -> None:
        """Connect to Redis."""
        if not self._redis_available:
            raise StreamingError("Redis library (aioredis) not available")
        
        import aioredis
        
        try:
            connection_params = self.config.connection_params
            
            self.redis_client = aioredis.from_url(
                connection_params.get('url', 'redis://localhost:6379'),
                password=connection_params.get('password'),
                db=connection_params.get('db', 0),
                socket_timeout=connection_params.get('socket_timeout', 30),
                socket_connect_timeout=connection_params.get('connect_timeout', 10)
            )
            
            # Test connection
            await self.redis_client.ping()
            self.is_connected = True
            logger.info("Connected to Redis Streams")
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise ConnectionError(f"Redis connection failed: {e}")
    
    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        if self.redis_client:
            await self.redis_client.close()
            self.redis_client = None
        self.is_connected = False
        logger.info("Disconnected from Redis")
    
    async def consume(self) -> AsyncIterator[StreamRecord]:
        """Consume records from Redis Streams."""
        if not self.is_connected or not self.redis_client:
            raise StreamingError("Not connected to Redis")
        
        connection_params = self.config.connection_params
        stream_name = connection_params.get('stream_name', 'anomaly-stream')
        consumer_group = connection_params.get('consumer_group', 'pynomaly-group')
        consumer_name = connection_params.get('consumer_name', 'pynomaly-consumer')
        
        self.is_consuming = True
        
        try:
            # Create consumer group if it doesn't exist
            try:
                await self.redis_client.xgroup_create(stream_name, consumer_group, id='0', mkstream=True)
            except Exception:
                # Group might already exist
                pass
            
            while not self._stop_event.is_set():
                # Read from stream
                messages = await self.redis_client.xreadgroup(
                    consumer_group,
                    consumer_name,
                    {stream_name: '>'},
                    count=self.config.batch_size,
                    block=int(self.config.max_wait_time * 1000)  # Convert to milliseconds
                )
                
                for stream, stream_messages in messages:
                    for message_id, fields in stream_messages:
                        metadata = {
                            'stream': stream.decode('utf-8'),
                            'message_id': message_id.decode('utf-8'),
                            'consumer_group': consumer_group,
                            'consumer_name': consumer_name
                        }
                        
                        # Convert fields dict to JSON-like format
                        data = {k.decode('utf-8'): v.decode('utf-8') for k, v in fields.items()}
                        
                        record = StreamRecord(
                            id=message_id.decode('utf-8'),
                            timestamp=datetime.utcnow(),
                            data=data,
                            tenant_id=TenantId(data.get('tenant_id', 'default')),
                            metadata=metadata
                        )
                        
                        yield record
                        
                        # Acknowledge the message
                        await self.redis_client.xack(stream_name, consumer_group, message_id)
                
        except Exception as e:
            logger.error(f"Error consuming from Redis Streams: {e}")
            raise StreamingError(f"Redis Streams consumption failed: {e}")
        finally:
            self.is_consuming = False
    
    async def health_check(self) -> bool:
        """Check Redis connection health."""
        if not self.is_connected or not self.redis_client:
            return False
        
        try:
            await self.redis_client.ping()
            return True
        except Exception as e:
            logger.warning(f"Redis health check failed: {e}")
            return False


class RabbitMQConnector(StreamingConnector):
    """RabbitMQ connector using AMQP."""
    
    def __init__(self, config: ConnectorConfig):
        super().__init__(config)
        self.connection = None
        self.channel = None
        self._rabbitmq_available = False
        
        try:
            import aio_pika
            self._rabbitmq_available = True
        except ImportError:
            logger.warning("aio_pika not available - RabbitMQ connector disabled")
    
    async def connect(self) -> None:
        """Connect to RabbitMQ."""
        if not self._rabbitmq_available:
            raise StreamingError("RabbitMQ library (aio_pika) not available")
        
        import aio_pika
        
        try:
            connection_params = self.config.connection_params
            
            # Build connection URL
            url = connection_params.get('url')
            if not url:
                host = connection_params.get('host', 'localhost')
                port = connection_params.get('port', 5672)
                username = connection_params.get('username', 'guest')
                password = connection_params.get('password', 'guest')
                vhost = connection_params.get('vhost', '/')
                url = f"amqp://{username}:{password}@{host}:{port}{vhost}"
            
            self.connection = await aio_pika.connect_robust(url)
            self.channel = await self.connection.channel()
            
            # Set QoS
            await self.channel.set_qos(prefetch_count=self.config.batch_size)
            
            self.is_connected = True
            logger.info("Connected to RabbitMQ")
            
        except Exception as e:
            logger.error(f"Failed to connect to RabbitMQ: {e}")
            raise ConnectionError(f"RabbitMQ connection failed: {e}")
    
    async def disconnect(self) -> None:
        """Disconnect from RabbitMQ."""
        if self.channel:
            await self.channel.close()
            self.channel = None
        
        if self.connection:
            await self.connection.close()
            self.connection = None
        
        self.is_connected = False
        logger.info("Disconnected from RabbitMQ")
    
    async def consume(self) -> AsyncIterator[StreamRecord]:
        """Consume records from RabbitMQ."""
        if not self.is_connected or not self.channel:
            raise StreamingError("Not connected to RabbitMQ")
        
        connection_params = self.config.connection_params
        queue_name = connection_params.get('queue_name', 'anomaly-queue')
        exchange_name = connection_params.get('exchange_name', '')
        routing_key = connection_params.get('routing_key', queue_name)
        
        self.is_consuming = True
        
        try:
            # Declare queue
            queue = await self.channel.declare_queue(queue_name, durable=True)
            
            # Bind to exchange if specified
            if exchange_name:
                exchange = await self.channel.declare_exchange(exchange_name, type='direct')
                await queue.bind(exchange, routing_key)
            
            async with queue.iterator() as queue_iter:
                async for message in queue_iter:
                    if self._stop_event.is_set():
                        break
                    
                    async with message.process():
                        metadata = {
                            'queue': queue_name,
                            'exchange': message.exchange,
                            'routing_key': message.routing_key,
                            'message_id': message.message_id,
                            'headers': dict(message.headers) if message.headers else {}
                        }
                        
                        record = self._deserialize_record(message.body, metadata)
                        yield record
                        
        except Exception as e:
            logger.error(f"Error consuming from RabbitMQ: {e}")
            raise StreamingError(f"RabbitMQ consumption failed: {e}")
        finally:
            self.is_consuming = False
    
    async def health_check(self) -> bool:
        """Check RabbitMQ connection health."""
        if not self.is_connected or not self.connection:
            return False
        
        try:
            return not self.connection.is_closed
        except Exception as e:
            logger.warning(f"RabbitMQ health check failed: {e}")
            return False


class WebSocketConnector(StreamingConnector):
    """WebSocket streaming connector."""
    
    def __init__(self, config: ConnectorConfig):
        super().__init__(config)
        self.websocket = None
        self._websockets_available = False
        
        try:
            import websockets
            self._websockets_available = True
        except ImportError:
            logger.warning("websockets not available - WebSocket connector disabled")
    
    async def connect(self) -> None:
        """Connect to WebSocket."""
        if not self._websockets_available:
            raise StreamingError("WebSocket library (websockets) not available")
        
        import websockets
        
        try:
            connection_params = self.config.connection_params
            uri = connection_params.get('uri', 'ws://localhost:8000/stream')
            
            extra_headers = connection_params.get('headers', {})
            
            self.websocket = await websockets.connect(
                uri,
                extra_headers=extra_headers,
                ping_interval=connection_params.get('ping_interval', 20),
                ping_timeout=connection_params.get('ping_timeout', 10),
                close_timeout=connection_params.get('close_timeout', 10)
            )
            
            self.is_connected = True
            logger.info(f"Connected to WebSocket: {uri}")
            
        except Exception as e:
            logger.error(f"Failed to connect to WebSocket: {e}")
            raise ConnectionError(f"WebSocket connection failed: {e}")
    
    async def disconnect(self) -> None:
        """Disconnect from WebSocket."""
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
        self.is_connected = False
        logger.info("Disconnected from WebSocket")
    
    async def consume(self) -> AsyncIterator[StreamRecord]:
        """Consume records from WebSocket."""
        if not self.is_connected or not self.websocket:
            raise StreamingError("Not connected to WebSocket")
        
        self.is_consuming = True
        
        try:
            async for message in self.websocket:
                if self._stop_event.is_set():
                    break
                
                metadata = {
                    'connection': 'websocket',
                    'uri': str(self.websocket.uri),
                    'received_at': datetime.utcnow().isoformat()
                }
                
                record = self._deserialize_record(message.encode('utf-8'), metadata)
                yield record
                
        except Exception as e:
            logger.error(f"Error consuming from WebSocket: {e}")
            raise StreamingError(f"WebSocket consumption failed: {e}")
        finally:
            self.is_consuming = False
    
    async def health_check(self) -> bool:
        """Check WebSocket connection health."""
        if not self.is_connected or not self.websocket:
            return False
        
        try:
            return not self.websocket.closed
        except Exception as e:
            logger.warning(f"WebSocket health check failed: {e}")
            return False


class HTTPStreamConnector(StreamingConnector):
    """HTTP streaming connector for server-sent events (SSE)."""
    
    def __init__(self, config: ConnectorConfig):
        super().__init__(config)
        self.session = None
        self._aiohttp_available = False
        
        try:
            import aiohttp
            self._aiohttp_available = True
        except ImportError:
            logger.warning("aiohttp not available - HTTP Stream connector disabled")
    
    async def connect(self) -> None:
        """Connect to HTTP stream."""
        if not self._aiohttp_available:
            raise StreamingError("HTTP library (aiohttp) not available")
        
        import aiohttp
        
        try:
            connection_params = self.config.connection_params
            
            timeout = aiohttp.ClientTimeout(
                total=connection_params.get('timeout', 300),
                connect=connection_params.get('connect_timeout', 30)
            )
            
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                headers=connection_params.get('headers', {})
            )
            
            self.is_connected = True
            logger.info("HTTP Stream connector initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize HTTP stream: {e}")
            raise ConnectionError(f"HTTP stream initialization failed: {e}")
    
    async def disconnect(self) -> None:
        """Disconnect from HTTP stream."""
        if self.session:
            await self.session.close()
            self.session = None
        self.is_connected = False
        logger.info("Disconnected from HTTP stream")
    
    async def consume(self) -> AsyncIterator[StreamRecord]:
        """Consume records from HTTP stream."""
        if not self.is_connected or not self.session:
            raise StreamingError("Not connected to HTTP stream")
        
        connection_params = self.config.connection_params
        url = connection_params.get('url', 'http://localhost:8000/stream')
        
        self.is_consuming = True
        
        try:
            async with self.session.get(url) as response:
                if response.status != 200:
                    raise StreamingError(f"HTTP error {response.status}: {response.reason}")
                
                async for line in response.content:
                    if self._stop_event.is_set():
                        break
                    
                    line = line.decode('utf-8').strip()
                    if not line or line.startswith('#'):  # Skip empty lines and comments
                        continue
                    
                    # Parse SSE format
                    if line.startswith('data: '):
                        data_content = line[6:]  # Remove 'data: ' prefix
                        
                        metadata = {
                            'source': 'http_stream',
                            'url': url,
                            'content_type': response.headers.get('content-type', ''),
                            'received_at': datetime.utcnow().isoformat()
                        }
                        
                        record = self._deserialize_record(data_content.encode('utf-8'), metadata)
                        yield record
                        
        except Exception as e:
            logger.error(f"Error consuming from HTTP stream: {e}")
            raise StreamingError(f"HTTP stream consumption failed: {e}")
        finally:
            self.is_consuming = False
    
    async def health_check(self) -> bool:
        """Check HTTP stream connection health."""
        if not self.is_connected or not self.session:
            return False
        
        try:
            connection_params = self.config.connection_params
            health_url = connection_params.get('health_url')
            
            if health_url:
                async with self.session.get(health_url) as response:
                    return response.status == 200
            else:
                # If no health URL, assume connection is healthy if session exists
                return not self.session.closed
                
        except Exception as e:
            logger.warning(f"HTTP stream health check failed: {e}")
            return False


class ConnectorFactory:
    """Factory for creating streaming connectors."""
    
    CONNECTOR_MAPPING = {
        ConnectorType.KAFKA: KafkaConnector,
        ConnectorType.REDIS_STREAMS: RedisStreamsConnector,
        ConnectorType.RABBITMQ: RabbitMQConnector,
        ConnectorType.WEBSOCKET: WebSocketConnector,
        ConnectorType.HTTP_STREAM: HTTPStreamConnector,
    }
    
    @classmethod
    def create_connector(cls, config: ConnectorConfig) -> StreamingConnector:
        """Create a streaming connector based on configuration."""
        connector_class = cls.CONNECTOR_MAPPING.get(config.connector_type)
        
        if not connector_class:
            raise StreamingError(f"Unsupported connector type: {config.connector_type}")
        
        return connector_class(config)
    
    @classmethod
    def get_available_connectors(cls) -> List[ConnectorType]:
        """Get list of available connector types."""
        available = []
        
        for connector_type, connector_class in cls.CONNECTOR_MAPPING.items():
            # Create a dummy config to test availability
            dummy_config = ConnectorConfig(
                connector_type=connector_type,
                connection_params={}
            )
            
            try:
                # Try to instantiate the connector
                connector = connector_class(dummy_config)
                
                # Check if required libraries are available
                if hasattr(connector, '_kafka_available') and connector._kafka_available:
                    available.append(connector_type)
                elif hasattr(connector, '_redis_available') and connector._redis_available:
                    available.append(connector_type)
                elif hasattr(connector, '_rabbitmq_available') and connector._rabbitmq_available:
                    available.append(connector_type)
                elif hasattr(connector, '_websockets_available') and connector._websockets_available:
                    available.append(connector_type)
                elif hasattr(connector, '_aiohttp_available') and connector._aiohttp_available:
                    available.append(connector_type)
                elif connector_type in [ConnectorType.FILE_TAIL]:  # Always available connectors
                    available.append(connector_type)
                    
            except Exception:
                # Connector not available
                pass
        
        return available