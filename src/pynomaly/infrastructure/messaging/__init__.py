"""
Message Queue Integration for Pynomaly

This module provides comprehensive message queue integration supporting multiple
message brokers including RabbitMQ, Apache Kafka, Redis, and in-memory queues.
"""

from .config import (
    MessageBroker,
    MessageConfig,
    QueueConfig,
    ConsumerConfig,
    ProducerConfig,
    MessageQueueConfig,
)
from .core import (
    Message,
    MessageHandler,
    MessageQueue,
    MessageQueueManager,
)
from .producers import (
    MessageProducer,
    KafkaProducer,
    RabbitMQProducer,
    RedisProducer,
)
from .consumers import (
    MessageConsumer,
    KafkaConsumer,
    RabbitMQConsumer,
    RedisConsumer,
)
from .factory import MessageQueueFactory
from .decorators import message_handler, async_message_handler
from .middleware import MessageMiddleware, LoggingMiddleware, MetricsMiddleware

__all__ = [
    # Config
    "MessageBroker",
    "MessageConfig",
    "QueueConfig",
    "ConsumerConfig",
    "ProducerConfig",
    "MessageQueueConfig",
    # Core
    "Message",
    "MessageHandler",
    "MessageQueue",
    "MessageQueueManager",
    # Producers
    "MessageProducer",
    "KafkaProducer",
    "RabbitMQProducer",
    "RedisProducer",
    # Consumers
    "MessageConsumer",
    "KafkaConsumer",
    "RabbitMQConsumer",
    "RedisConsumer",
    # Factory
    "MessageQueueFactory",
    # Decorators
    "message_handler",
    "async_message_handler",
    # Middleware
    "MessageMiddleware",
    "LoggingMiddleware",
    "MetricsMiddleware",
]
