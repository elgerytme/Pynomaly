# Message Queue Integration

This module provides comprehensive message queue integration for the Pynomaly anomaly detection system. It supports multiple message brokers and provides a unified interface for asynchronous message processing.

## Features

- **Multiple Broker Support**: RabbitMQ, Apache Kafka, Redis, and in-memory queues
- **Unified Interface**: Consistent API across all message brokers
- **Async/Await Support**: Built for modern Python async programming
- **Message Handlers**: Decorator-based message processing
- **Middleware Support**: Pluggable middleware for logging, metrics, etc.
- **Health Monitoring**: Built-in health checks and metrics
- **Configuration Management**: Environment-based configuration
- **Error Handling**: Robust error handling with retry and dead letter queues

## Quick Start

### Basic Usage

```python
import asyncio
from pynomaly.infrastructure.messaging import (
    MessageQueueFactory, 
    MessageQueueConfig, 
    MessageBroker,
    Message
)

async def main():
    # Create a message queue
    config = MessageQueueConfig(broker=MessageBroker.MEMORY)
    queue = MessageQueueFactory.create_queue(config)
    
    # Connect to the queue
    await queue.connect()
    
    # Send a message
    message = Message(body="Hello, World!", queue_name="test_queue")
    await queue.send(message)
    
    # Receive a message
    received = await queue.receive("test_queue")
    print(f"Received: {received.body}")
    
    # Disconnect
    await queue.disconnect()

asyncio.run(main())
```

### Using the Manager

```python
import asyncio
from pynomaly.infrastructure.messaging import (
    MessageQueueManager,
    MessageQueueFactory, 
    MessageQueueConfig,
    MessageBroker,
    Message
)

async def main():
    # Create manager
    config = MessageQueueConfig(broker=MessageBroker.MEMORY)
    manager = MessageQueueManager(config)
    
    # Add queues
    queue = MessageQueueFactory.create_queue(config)
    manager.add_queue("main_queue", queue)
    
    # Start manager
    await manager.start()
    
    # Send messages
    message = Message(body="Hello from manager!", queue_name="test_queue")
    await manager.send_to_queue("main_queue", message)
    
    # Check health
    health = await manager.health_check()
    print(f"System healthy: {health['overall_healthy']}")
    
    # Stop manager
    await manager.stop()

asyncio.run(main())
```

## Message Handlers

### Decorator-Based Handlers

```python
import asyncio
from pynomaly.infrastructure.messaging import (
    MessageQueueFactory,
    MessageQueueConfig,
    MessageBroker,
    Message,
    message_handler
)

# Define a message handler
async def process_anomaly_detection(message: Message) -> str:
    data = message.body
    # Process the anomaly detection request
    result = f"Processed anomaly detection for: {data}"
    return result

async def main():
    config = MessageQueueConfig(broker=MessageBroker.MEMORY)
    queue = MessageQueueFactory.create_queue(config)
    
    # Register the handler
    queue.register_function_handler("anomaly_detection", process_anomaly_detection)
    
    await queue.connect()
    
    # Send a message
    message = Message(
        body={"dataset": "sensor_data", "algorithm": "isolation_forest"},
        queue_name="anomaly_detection"
    )
    await queue.send(message)
    
    # Process the message
    received = await queue.receive("anomaly_detection")
    if received:
        result = await queue.process_message(received)
        print(f"Result: {result}")
    
    await queue.disconnect()

asyncio.run(main())
```

## Configuration

### Environment Variables

```bash
# Message broker type
export MESSAGE_BROKER=redis

# Redis configuration
export REDIS_HOST=localhost
export REDIS_PORT=6379
export REDIS_PASSWORD=mypassword
export REDIS_DB=0

# RabbitMQ configuration
export RABBITMQ_HOST=localhost
export RABBITMQ_PORT=5672
export RABBITMQ_USERNAME=admin
export RABBITMQ_PASSWORD=secret

# Kafka configuration
export KAFKA_BOOTSTRAP_SERVERS=localhost:9092,localhost:9093
export KAFKA_SECURITY_PROTOCOL=PLAINTEXT

# Performance tuning
export WORKER_POOL_SIZE=8
export MAX_CONCURRENT_MESSAGES=200
export ENABLE_METRICS=true
```

### Programmatic Configuration

```python
from pynomaly.infrastructure.messaging import (
    MessageQueueConfig,
    MessageBroker,
    ConnectionConfig,
    MessageConfig,
    QueueConfig
)

config = MessageQueueConfig(
    broker=MessageBroker.RABBITMQ,
    connection=ConnectionConfig(
        host="rabbitmq.example.com",
        port=5672,
        username="app_user",
        password="secure_password",
        virtual_host="/app"
    ),
    enable_metrics=True,
    worker_pool_size=4,
    max_concurrent_messages=100
)
```

## Supported Brokers

### Memory Queue (Development/Testing)
- **Use Case**: Testing, development, single-process applications
- **Features**: Fast, no external dependencies
- **Limitations**: Not persistent, single-process only

```python
config = MessageQueueConfig(broker=MessageBroker.MEMORY)
```

### Redis
- **Use Case**: High-performance, simple pub/sub, caching
- **Features**: Fast, persistent, clustering support
- **Requirements**: `redis` package

```python
config = MessageQueueConfig(
    broker=MessageBroker.REDIS,
    connection=ConnectionConfig(
        host="redis.example.com",
        port=6379,
        password="redis_password",
        db=0
    )
)
```

### RabbitMQ
- **Use Case**: Enterprise messaging, complex routing, reliability
- **Features**: Reliable delivery, flexible routing, clustering
- **Requirements**: `pika` package

```python
config = MessageQueueConfig(
    broker=MessageBroker.RABBITMQ,
    connection=ConnectionConfig(
        host="rabbitmq.example.com",
        port=5672,
        username="app_user",
        password="secure_password"
    )
)
```

### Apache Kafka
- **Use Case**: High-throughput, event streaming, big data
- **Features**: High throughput, partitioning, replication
- **Requirements**: `kafka-python` package

```python
config = MessageQueueConfig(
    broker=MessageBroker.KAFKA,
    connection=ConnectionConfig(
        bootstrap_servers=["kafka1:9092", "kafka2:9092", "kafka3:9092"]
    )
)
```

## Middleware

### Logging Middleware

```python
from pynomaly.infrastructure.messaging import LoggingMiddleware

queue = MessageQueueFactory.create_queue(config)
queue.add_middleware(LoggingMiddleware())
```

### Metrics Middleware

```python
from pynomaly.infrastructure.messaging import MetricsMiddleware

metrics = MetricsMiddleware()
queue.add_middleware(metrics)

# After processing messages
stats = metrics.get_metrics()
print(f"Processed: {stats['processed_count']}")
print(f"Errors: {stats['error_count']}")
print(f"Avg time: {stats['average_processing_time']:.2f}s")
```

### Custom Middleware

```python
from pynomaly.infrastructure.messaging import MessageMiddleware

class CustomMiddleware(MessageMiddleware):
    async def before_process(self, message):
        # Custom pre-processing logic
        message.headers['processed_at'] = datetime.now().isoformat()
    
    async def after_process(self, message, result):
        # Custom post-processing logic
        print(f"Message {message.id} processed successfully")
    
    async def on_error(self, message, error):
        # Custom error handling
        print(f"Message {message.id} failed: {error}")
```

## Integration with Pynomaly

### Anomaly Detection Pipeline

```python
import asyncio
from pynomaly.infrastructure.messaging import (
    MessageQueueFactory, MessageQueueConfig, MessageBroker, Message
)
from pynomaly.domain.services.detection_service import DetectionService

async def anomaly_detection_handler(message: Message) -> dict:
    """Handle anomaly detection requests."""
    request = message.body
    
    # Extract parameters
    dataset_id = request.get('dataset_id')
    algorithm = request.get('algorithm', 'isolation_forest')
    
    # Perform anomaly detection
    detection_service = DetectionService()
    result = await detection_service.detect_anomalies(
        dataset_id=dataset_id,
        algorithm=algorithm
    )
    
    return {
        'dataset_id': dataset_id,
        'algorithm': algorithm,
        'anomaly_count': result.anomaly_count,
        'execution_time': result.execution_time
    }

async def setup_anomaly_detection_queue():
    """Set up anomaly detection message queue."""
    config = MessageQueueConfig.from_env()
    queue = MessageQueueFactory.create_queue(config)
    
    # Register handler
    queue.register_function_handler("anomaly_detection", anomaly_detection_handler)
    
    # Add middleware
    queue.add_middleware(LoggingMiddleware())
    queue.add_middleware(MetricsMiddleware())
    
    await queue.connect()
    
    # Start consuming
    await queue.start_consuming("anomaly_detection")
```

### Stream Processing Integration

```python
from pynomaly.infrastructure.streaming import StreamProcessor
from pynomaly.infrastructure.messaging import MessageQueueFactory, Message

async def stream_to_queue_processor(stream_data):
    """Process stream data and send to message queue."""
    
    # Create message queue
    queue = MessageQueueFactory.create_queue_from_env()
    await queue.connect()
    
    # Convert stream data to message
    message = Message(
        body={
            'stream_id': stream_data.id,
            'data': stream_data.data,
            'timestamp': stream_data.timestamp
        },
        queue_name="stream_processing"
    )
    
    # Send to queue for processing
    await queue.send(message)
    
    await queue.disconnect()
```

## Monitoring and Health Checks

### Health Check Endpoint

```python
from fastapi import FastAPI
from pynomaly.infrastructure.messaging import get_message_queue_manager

app = FastAPI()

@app.get("/health/messaging")
async def messaging_health():
    """Check messaging system health."""
    manager = get_message_queue_manager()
    if not manager:
        return {"status": "unhealthy", "reason": "Manager not initialized"}
    
    health = await manager.health_check()
    return {
        "status": "healthy" if health["overall_healthy"] else "unhealthy",
        "details": health
    }
```

### Metrics Collection

```python
from pynomaly.infrastructure.messaging import get_message_queue_manager

async def collect_messaging_metrics():
    """Collect messaging metrics."""
    manager = get_message_queue_manager()
    if manager:
        stats = manager.get_global_stats()
        return {
            "total_messages_sent": stats["total_messages_sent"],
            "total_messages_received": stats["total_messages_received"],
            "total_messages_failed": stats["total_messages_failed"],
            "queue_count": stats["queue_count"],
            "consumer_count": stats["consumer_count"]
        }
    return {}
```

## Error Handling

### Retry Logic

```python
from pynomaly.infrastructure.messaging import MessageQueueConfig

config = MessageQueueConfig(
    broker=MessageBroker.RABBITMQ,
    max_retry_attempts=3,
    retry_delay_seconds=5,
    enable_dead_letter_queue=True
)
```

### Dead Letter Queue

```python
from pynomaly.infrastructure.messaging import QueueConfig

# Configure dead letter queue
dlq_config = QueueConfig(
    name="anomaly_detection_dlq",
    durable=True,
    arguments={
        "x-message-ttl": 86400000  # 24 hours
    }
)

# Configure main queue with DLQ
main_config = QueueConfig(
    name="anomaly_detection",
    durable=True,
    arguments={
        "x-dead-letter-exchange": "",
        "x-dead-letter-routing-key": "anomaly_detection_dlq"
    }
)
```

## Performance Tuning

### Connection Pooling

```python
config = MessageQueueConfig(
    broker=MessageBroker.RABBITMQ,
    connection=ConnectionConfig(
        max_connections=20,
        min_connections=5,
        connection_timeout=30,
        heartbeat=600
    )
)
```

### Batch Processing

```python
from pynomaly.infrastructure.messaging import MessageQueueConfig

config = MessageQueueConfig(
    broker=MessageBroker.KAFKA,
    producer=ProducerConfig(
        batch_size=16384,
        linger_ms=10,
        compression_type=CompressionType.GZIP
    ),
    consumer=ConsumerConfig(
        max_poll_records=500,
        fetch_min_bytes=1024,
        prefetch_count=10
    )
)
```

## Testing

### Unit Tests

```python
import pytest
from pynomaly.infrastructure.messaging import MessageQueueFactory, MessageBroker

@pytest.mark.asyncio
async def test_message_queue_basic_functionality():
    """Test basic message queue functionality."""
    queue = MessageQueueFactory.create_for_testing()
    
    await queue.connect()
    
    # Send message
    message = Message(body="test", queue_name="test_queue")
    message_id = await queue.send(message)
    
    # Receive message
    received = await queue.receive("test_queue")
    assert received.body == "test"
    assert received.id == message_id
    
    await queue.disconnect()
```

### Integration Tests

```python
import pytest
from pynomaly.infrastructure.messaging import (
    MessageQueueManager, MessageQueueFactory, MessageQueueConfig, MessageBroker
)

@pytest.mark.asyncio
async def test_message_queue_manager():
    """Test message queue manager functionality."""
    config = MessageQueueConfig(broker=MessageBroker.MEMORY)
    manager = MessageQueueManager(config)
    
    # Add queue
    queue = MessageQueueFactory.create_queue(config)
    manager.add_queue("test_queue", queue)
    
    # Start manager
    await manager.start()
    
    # Test health check
    health = await manager.health_check()
    assert health["overall_healthy"] is True
    
    # Stop manager
    await manager.stop()
```

## Best Practices

1. **Use Environment Configuration**: Configure message brokers through environment variables for flexibility
2. **Implement Health Checks**: Always implement health checks for monitoring
3. **Add Middleware**: Use middleware for logging, metrics, and error handling
4. **Handle Errors Gracefully**: Implement proper error handling with retries and dead letter queues
5. **Monitor Performance**: Track message processing times and throughput
6. **Use Connection Pooling**: Configure appropriate connection pools for production
7. **Implement Graceful Shutdown**: Ensure proper cleanup when shutting down
8. **Test Thoroughly**: Write comprehensive unit and integration tests

## Troubleshooting

### Common Issues

1. **Connection Failures**: Check broker availability and credentials
2. **Message Loss**: Ensure durability settings are correctly configured
3. **Performance Issues**: Monitor queue sizes and processing times
4. **Memory Leaks**: Ensure proper cleanup of connections and consumers

### Debugging

Enable debug logging to troubleshoot issues:

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('pynomaly.infrastructure.messaging')
logger.setLevel(logging.DEBUG)
```

## Contributing

When extending the message queue integration:

1. Follow the existing patterns and interfaces
2. Add comprehensive tests for new functionality
3. Update documentation and examples
4. Consider backward compatibility
5. Add appropriate error handling and logging
