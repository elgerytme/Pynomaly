"""
Message Consumers

This module provides consumer classes for different message brokers.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Optional

from .core import Message
from .config import MessageQueueConfig

logger = logging.getLogger(__name__)


class MessageConsumer(ABC):
    """Abstract base class for message consumers."""
    
    def __init__(self, config: MessageQueueConfig):
        self.config = config
        self._is_connected = False
    
    @abstractmethod
    async def connect(self) -> None:
        """Connect to message broker."""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from message broker."""
        pass
    
    @abstractmethod
    async def receive(
        self, 
        queue_name: str, 
        timeout: Optional[float] = None
    ) -> Optional[Message]:
        """Receive a message."""
        pass
    
    @abstractmethod
    async def acknowledge(self, message: Message) -> None:
        """Acknowledge message processing."""
        pass
    
    @abstractmethod
    async def reject(self, message: Message, requeue: bool = False) -> None:
        """Reject a message."""
        pass


class KafkaConsumer(MessageConsumer):
    """Kafka message consumer."""
    
    def __init__(self, config: MessageQueueConfig):
        super().__init__(config)
        self._consumer = None
    
    async def connect(self) -> None:
        """Connect to Kafka."""
        try:
            from kafka import KafkaConsumer as KafkaConsumerImpl
            self._consumer = KafkaConsumerImpl(
                bootstrap_servers=self.config.connection.bootstrap_servers,
                value_deserializer=lambda m: m.decode('utf-8')
            )
            self._is_connected = True
            logger.info("Connected Kafka consumer")
        except ImportError:
            raise ImportError("kafka-python package is required")
    
    async def disconnect(self) -> None:
        """Disconnect from Kafka."""
        if self._consumer:
            self._consumer.close()
        self._is_connected = False
        logger.info("Disconnected Kafka consumer")
    
    async def receive(
        self, 
        queue_name: str, 
        timeout: Optional[float] = None
    ) -> Optional[Message]:
        """Receive a message from Kafka."""
        if not self._consumer:
            raise RuntimeError("Consumer not connected")
        
        self._consumer.subscribe([queue_name])
        
        try:
            messages = self._consumer.poll(timeout_ms=int(timeout * 1000) if timeout else 1000)
            for topic_partition, message_list in messages.items():
                for message_data in message_list:
                    message = Message.deserialize(message_data.value.encode('utf-8'))
                    logger.debug(f"Received message {message.id} from Kafka topic {queue_name}")
                    return message
        except Exception as e:
            logger.error(f"Error receiving from Kafka: {e}")
        
        return None
    
    async def acknowledge(self, message: Message) -> None:
        """Acknowledge message processing."""
        if self._consumer:
            self._consumer.commit()
        logger.debug(f"Acknowledged message {message.id}")
    
    async def reject(self, message: Message, requeue: bool = False) -> None:
        """Reject a message (not supported in Kafka)."""
        logger.debug(f"Rejected message {message.id} (requeue not supported)")


class RabbitMQConsumer(MessageConsumer):
    """RabbitMQ message consumer."""
    
    def __init__(self, config: MessageQueueConfig):
        super().__init__(config)
        self._connection = None
        self._channel = None
    
    async def connect(self) -> None:
        """Connect to RabbitMQ."""
        try:
            import pika
            credentials = pika.PlainCredentials(
                self.config.connection.username or 'guest',
                self.config.connection.password or 'guest'
            )
            parameters = pika.ConnectionParameters(
                host=self.config.connection.host,
                port=self.config.connection.port,
                credentials=credentials
            )
            self._connection = pika.BlockingConnection(parameters)
            self._channel = self._connection.channel()
            self._is_connected = True
            logger.info("Connected RabbitMQ consumer")
        except ImportError:
            raise ImportError("pika package is required")
    
    async def disconnect(self) -> None:
        """Disconnect from RabbitMQ."""
        if self._channel:
            self._channel.close()
        if self._connection:
            self._connection.close()
        self._is_connected = False
        logger.info("Disconnected RabbitMQ consumer")
    
    async def receive(
        self, 
        queue_name: str, 
        timeout: Optional[float] = None
    ) -> Optional[Message]:
        """Receive a message from RabbitMQ."""
        self._channel.queue_declare(queue=queue_name, durable=True)
        
        try:
            method_frame, header_frame, body = self._channel.basic_get(queue=queue_name)
            
            if method_frame:
                message = Message.deserialize(body)
                message.queue_name = queue_name
                message.headers['delivery_tag'] = method_frame.delivery_tag
                
                logger.debug(f"Received message {message.id} from RabbitMQ queue {queue_name}")
                return message
        except Exception as e:
            logger.error(f"Error receiving from RabbitMQ: {e}")
        
        return None
    
    async def acknowledge(self, message: Message) -> None:
        """Acknowledge message processing."""
        delivery_tag = message.headers.get('delivery_tag')
        if delivery_tag:
            self._channel.basic_ack(delivery_tag=delivery_tag)
        logger.debug(f"Acknowledged message {message.id}")
    
    async def reject(self, message: Message, requeue: bool = False) -> None:
        """Reject a message."""
        delivery_tag = message.headers.get('delivery_tag')
        if delivery_tag:
            self._channel.basic_reject(delivery_tag=delivery_tag, requeue=requeue)
        logger.debug(f"Rejected message {message.id}, requeue={requeue}")


class RedisConsumer(MessageConsumer):
    """Redis message consumer."""
    
    def __init__(self, config: MessageQueueConfig):
        super().__init__(config)
        self._redis_client = None
    
    async def connect(self) -> None:
        """Connect to Redis."""
        try:
            import redis.asyncio as redis
            self._redis_client = redis.Redis(
                host=self.config.connection.host,
                port=self.config.connection.port,
                password=self.config.connection.password,
                db=self.config.connection.db
            )
            await self._redis_client.ping()
            self._is_connected = True
            logger.info("Connected Redis consumer")
        except ImportError:
            raise ImportError("redis package is required")
    
    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        if self._redis_client:
            await self._redis_client.close()
        self._is_connected = False
        logger.info("Disconnected Redis consumer")
    
    async def receive(
        self, 
        queue_name: str, 
        timeout: Optional[float] = None
    ) -> Optional[Message]:
        """Receive a message from Redis."""
        timeout_seconds = int(timeout) if timeout else 0
        
        try:
            result = await self._redis_client.brpop(queue_name, timeout=timeout_seconds)
            if result:
                _, message_data = result
                message = Message.deserialize(message_data.encode('utf-8'))
                logger.debug(f"Received message {message.id} from Redis queue {queue_name}")
                return message
        except Exception as e:
            logger.error(f"Error receiving from Redis: {e}")
        
        return None
    
    async def acknowledge(self, message: Message) -> None:
        """Acknowledge message processing (no-op for Redis)."""
        logger.debug(f"Acknowledged message {message.id}")
    
    async def reject(self, message: Message, requeue: bool = False) -> None:
        """Reject a message."""
        if requeue and message.queue_name and self._redis_client:
            serialized_message = message.serialize().decode('utf-8')
            await self._redis_client.lpush(message.queue_name, serialized_message)
        logger.debug(f"Rejected message {message.id}, requeue={requeue}")
