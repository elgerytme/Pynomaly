"""Message queue adapter implementations."""

from .adapter_factory import AdapterFactory, ReliableAdapter
from .redis_queue_adapter import RedisQueueAdapter

try:
    from .rabbitmq_adapter import RabbitMQAdapter
    __all__ = ["RedisQueueAdapter", "RabbitMQAdapter", "AdapterFactory", "ReliableAdapter"]
except ImportError:
    # RabbitMQ support is optional
    __all__ = ["RedisQueueAdapter", "AdapterFactory", "ReliableAdapter"]