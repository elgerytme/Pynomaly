# Message Queue Integration

This module provides comprehensive message queue functionality for the Pynomaly anomaly detection platform. It supports both Redis and RabbitMQ backends with a unified interface for asynchronous task processing, event-driven communication, and background job execution.

## Features

- **Multiple Backends**: Support for Redis Streams and RabbitMQ with optional backends
- **Task Management**: Submit, track, and cancel background tasks
- **Message Routing**: Intelligent message routing based on message types
- **Worker Management**: Configurable worker pools with health monitoring
- **Reliability**: Message persistence, retry logic, and dead letter queues
- **Monitoring**: Built-in metrics and health checks
- **Type Safety**: Full Pydantic model support with type validation

## Quick Start

### Basic Configuration

```python
from pynomaly.infrastructure.messaging import (
    AdapterFactory, QueueManager, MessagingSettings
)

# Configure messaging
settings = MessagingSettings(
    queue_backend="redis",  # or "rabbitmq"
    redis_queue_db=1,
    worker_concurrency=4,
    task_timeout=300
)

# Create queue manager
adapter = AdapterFactory.create_adapter(settings, "redis://localhost:6379")
queue_manager = QueueManager(adapter, settings)
```

### Sending Messages

```python
# Start the queue manager
await queue_manager.start()

# Send a simple message
message_id = await queue_manager.send_message(
    "notifications",
    {"user_id": "123", "message": "Processing complete"},
    priority=MessagePriority.HIGH
)
```

### Submitting Tasks

```python
from pynomaly.infrastructure.messaging import Task, TaskType

# Create a task
task = Task(
    task_type=TaskType.ANOMALY_DETECTION,
    name="Detect Anomalies in Dataset",
    function_name="detect_anomalies",
    kwargs={
        "dataset_id": "dataset_123",
        "algorithm": "isolation_forest",
        "parameters": {"contamination": 0.1}
    }
)

# Submit for processing
task_id = await queue_manager.submit_task(task)
```

### Processing Tasks with Workers

```python
from pynomaly.infrastructure.messaging import TaskProcessor

# Create task processor
task_processor = TaskProcessor(adapter, settings)

# Define task handler
async def handle_anomaly_detection(*args, **kwargs):
    dataset_id = kwargs["dataset_id"]
    algorithm = kwargs["algorithm"]
    
    # Your processing logic here
    result = run_anomaly_detection(dataset_id, algorithm)
    
    return result

# Register handler
task_processor.register_task_handler("anomaly_detection", handle_anomaly_detection)

# Start workers
await task_processor.start(queue_names=["anomaly_detection"])
```

## Architecture

### Components

1. **Adapters**: Backend-specific implementations (Redis, RabbitMQ)
2. **Queue Manager**: High-level queue operations
3. **Task Processor**: Worker management and task execution
4. **Message Broker**: Message routing and publishing
5. **Models**: Type-safe message and task definitions

### Message Flow

```
Producer → Message Queue → Consumer/Worker → Result Storage
    ↓
Message Broker (routing)
    ↓
Task Processor (execution)
```

### Supported Backends

#### Redis (Default)
- Uses Redis Streams for message queuing
- Consumer groups for worker coordination
- Built-in persistence and reliability
- Suitable for most use cases

#### RabbitMQ (Optional)
- Full AMQP protocol support
- Advanced routing capabilities
- Enterprise-grade reliability
- Install with: `pip install aio-pika`

## Configuration

### Environment Variables

```bash
# Queue backend
PYNOMALY_MESSAGING__QUEUE_BACKEND=redis
PYNOMALY_MESSAGING__QUEUE_URL=redis://localhost:6379/1

# Worker settings
PYNOMALY_MESSAGING__WORKER_CONCURRENCY=4
PYNOMALY_MESSAGING__TASK_TIMEOUT=300
PYNOMALY_MESSAGING__TASK_BATCH_SIZE=10

# Redis specific
PYNOMALY_MESSAGING__REDIS_QUEUE_DB=1
PYNOMALY_MESSAGING__REDIS_STREAM_MAXLEN=10000
PYNOMALY_MESSAGING__REDIS_CONSUMER_GROUP=pynomaly_workers

# Monitoring
PYNOMALY_MESSAGING__ENABLE_METRICS=true
PYNOMALY_MESSAGING__METRICS_INTERVAL=60
PYNOMALY_MESSAGING__DEAD_LETTER_QUEUE_ENABLED=true
```

### Programmatic Configuration

```python
settings = MessagingSettings(
    queue_backend="redis",
    queue_url="redis://localhost:6379/1",
    worker_concurrency=8,
    task_timeout=600,
    enable_metrics=True,
    dead_letter_queue_enabled=True
)
```

## Message Types

### Standard Messages

```python
message = Message(
    queue_name="notifications",
    payload={"data": "value"},
    priority=MessagePriority.NORMAL,
    message_type="notification"
)
```

### Task Messages

```python
task = Task(
    task_type=TaskType.DATA_PROFILING,
    name="Profile Dataset",
    function_name="profile_data",
    args=[dataset_id],
    kwargs={"include_correlations": True},
    timeout=300,
    max_retries=3
)
```

## Task Types

- `ANOMALY_DETECTION`: Anomaly detection algorithms
- `DATA_PROFILING`: Data quality and statistical profiling
- `MODEL_TRAINING`: Machine learning model training
- `DATA_PROCESSING`: General data processing tasks
- `REPORT_GENERATION`: Report and visualization generation
- `NOTIFICATION`: User notifications
- `CLEANUP`: Maintenance and cleanup tasks
- `EXPORT/IMPORT`: Data export and import operations
- `VALIDATION`: Data validation tasks

## Advanced Features

### Message Broker for Complex Routing

```python
from pynomaly.infrastructure.messaging import MessageBroker

broker = MessageBroker(adapter, settings)
await broker.start()

# Automatic routing based on message type
await broker.create_anomaly_detection_task(
    dataset_id="123",
    algorithm="lof",
    priority=1
)

await broker.publish_notification(
    recipient="admin@example.com",
    subject="Task Complete",
    content="Your analysis is ready"
)
```

### Reliability Features

```python
from pynomaly.infrastructure.messaging import ReliableAdapter

# Wrap adapter with reliability features
reliable_adapter = ReliableAdapter(
    base_adapter,
    settings,
    max_retries=3,
    retry_delay=1.0,
    exponential_backoff=True
)
```

### Monitoring and Health Checks

```python
# Queue statistics
stats = await queue_manager.get_queue_stats("anomaly_detection")
print(f"Messages: {stats['total_messages']}")
print(f"Pending: {stats['pending_messages']}")

# Worker statistics
worker_stats = task_processor.get_statistics()
print(f"Tasks processed: {worker_stats['tasks_processed']}")
print(f"Average time: {worker_stats['avg_processing_time']}")

# Health check
healthy = await queue_manager.health_check()
```

## Docker Integration

### Redis Setup

```yaml
# docker-compose.yml
services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data

  pynomaly-worker:
    image: pynomaly:latest
    environment:
      - PYNOMALY_MESSAGING__QUEUE_BACKEND=redis
      - PYNOMALY_MESSAGING__QUEUE_URL=redis://redis:6379/1
    depends_on:
      - redis
    command: python -m pynomaly.workers.main
```

### RabbitMQ Setup

```yaml
services:
  rabbitmq:
    image: rabbitmq:3-management-alpine
    ports:
      - "5672:5672"
      - "15672:15672"
    environment:
      - RABBITMQ_DEFAULT_USER=pynomaly
      - RABBITMQ_DEFAULT_PASS=password
    volumes:
      - rabbitmq_data:/var/lib/rabbitmq
```

## Testing

The module includes comprehensive test suites:

```bash
# Run unit tests
pytest src/packages/infrastructure/tests/messaging/unit/

# Run integration tests (requires Redis)
pytest src/packages/infrastructure/tests/messaging/integration/

# Run with coverage
pytest --cov=pynomaly.infrastructure.messaging
```

## Performance Considerations

1. **Batch Processing**: Use appropriate batch sizes for your workload
2. **Worker Scaling**: Scale workers based on queue depth and processing time
3. **Message Size**: Keep message payloads reasonable (<1MB)
4. **TTL Settings**: Configure appropriate TTLs for your use case
5. **Dead Letter Queues**: Monitor and handle failed messages

## Error Handling

The system provides robust error handling:

- **Connection Failures**: Automatic reconnection with exponential backoff
- **Processing Errors**: Configurable retry logic with dead letter queues
- **Timeout Handling**: Task-level timeouts with graceful cancellation
- **Resource Cleanup**: Proper cleanup on shutdown

## Examples

See `examples/basic_usage.py` for comprehensive usage examples covering:
- Basic messaging
- Task processing
- Message broker usage
- Worker management

## Dependencies

### Required
- `redis`: Redis client library
- `pydantic`: Data validation and serialization

### Optional
- `aio-pika`: RabbitMQ/AMQP support
- `uvloop`: Performance improvement for asyncio

## Contributing

When adding new features:
1. Follow the existing patterns for adapters and services
2. Add comprehensive tests
3. Update documentation
4. Consider backward compatibility
5. Add appropriate type hints

## License

This module is part of the Pynomaly project and follows the same license terms.