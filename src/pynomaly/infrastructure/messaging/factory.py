"""
Message Queue Factory

This module provides a factory for creating message queue instances based on configuration.
"""

import logging
from typing import Optional

from .config import MessageBroker, MessageQueueConfig
from .core import MessageQueue
from .implementations.memory_queue import MemoryMessageQueue
from .implementations.redis_queue import RedisMessageQueue
from .implementations.kafka_queue import KafkaMessageQueue
from .implementations.rabbitmq_queue import RabbitMQMessageQueue

logger = logging.getLogger(__name__)


class MessageQueueFactory:
    """Factory for creating message queue instances."""
    
    @staticmethod
    def create_queue(config: MessageQueueConfig) -> MessageQueue:
        """Create a message queue based on configuration."""
        
        if config.broker == MessageBroker.MEMORY:
            return MemoryMessageQueue(config)
        
        elif config.broker == MessageBroker.REDIS:
            return RedisMessageQueue(config)
        
        elif config.broker == MessageBroker.KAFKA:
            return KafkaMessageQueue(config)
        
        elif config.broker == MessageBroker.RABBITMQ:
            return RabbitMQMessageQueue(config)
        
        else:
            raise ValueError(f"Unsupported message broker: {config.broker}")
    
    @staticmethod
    def create_queue_from_env(broker: Optional[MessageBroker] = None) -> MessageQueue:
        """Create a message queue from environment variables."""
        config = MessageQueueConfig.from_env(broker)
        return MessageQueueFactory.create_queue(config)
    
    @staticmethod
    def get_available_brokers() -> list[MessageBroker]:
        """Get list of available message brokers."""
        available = [MessageBroker.MEMORY]  # Always available
        
        # Check Redis availability
        try:
            import redis
            available.append(MessageBroker.REDIS)
        except ImportError:
            pass
        
        # Check Kafka availability
        try:
            import kafka
            available.append(MessageBroker.KAFKA)
        except ImportError:
            pass
        
        # Check RabbitMQ availability
        try:
            import pika
            available.append(MessageBroker.RABBITMQ)
        except ImportError:
            pass
        
        return available
    
    @staticmethod
    def validate_broker_availability(broker: MessageBroker) -> bool:
        """Validate that a broker is available."""
        available_brokers = MessageQueueFactory.get_available_brokers()
        return broker in available_brokers
    
    @staticmethod
    def create_with_fallback(
        preferred_broker: MessageBroker,
        fallback_broker: MessageBroker = MessageBroker.MEMORY
    ) -> MessageQueue:
        """Create a message queue with fallback to another broker."""
        
        # Try preferred broker first
        if MessageQueueFactory.validate_broker_availability(preferred_broker):
            try:
                config = MessageQueueConfig.from_env(preferred_broker)
                return MessageQueueFactory.create_queue(config)
            except Exception as e:
                logger.warning(f"Failed to create {preferred_broker} queue: {e}")
        
        # Fall back to fallback broker
        logger.info(f"Falling back to {fallback_broker} message queue")
        config = MessageQueueConfig.from_env(fallback_broker)
        return MessageQueueFactory.create_queue(config)
    
    @staticmethod
    def create_for_testing() -> MessageQueue:
        """Create a message queue for testing purposes."""
        config = MessageQueueConfig(broker=MessageBroker.MEMORY)
        return MessageQueueFactory.create_queue(config)
