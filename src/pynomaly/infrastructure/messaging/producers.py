"""
Message Producers

This module provides producer classes for different message brokers.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Optional

from .core import Message, MessageQueue
from .config import MessageQueueConfig

logger = logging.getLogger(__name__)


class MessageProducer(ABC):
    """Abstract base class for message producers."""
    
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
    async def send(
        self, 
        message: Message, 
        queue_name: Optional[str] = None,
        routing_key: Optional[str] = None
    ) -> str:
        """Send a message."""
        pass


class KafkaProducer(MessageProducer):
    """Kafka message producer."""
    
    def __init__(self, config: MessageQueueConfig):
        super().__init__(config)
        self._producer = None
    
    async def connect(self) -> None:
        """Connect to Kafka."""
        try:
            from kafka import KafkaProducer as KafkaProducerImpl
            self._producer = KafkaProducerImpl(
                bootstrap_servers=self.config.connection.bootstrap_servers,
                value_serializer=lambda v: v.encode('utf-8')
            )
            self._is_connected = True
            logger.info("Connected Kafka producer")
        except ImportError:
            raise ImportError("kafka-python package is required")
    
    async def disconnect(self) -> None:
        """Disconnect from Kafka."""
        if self._producer:
            self._producer.close()
        self._is_connected = False
        logger.info("Disconnected Kafka producer")
    
    async def send(
        self, 
        message: Message, 
        queue_name: Optional[str] = None,
        routing_key: Optional[str] = None
    ) -> str:
        """Send a message to Kafka."""
        topic = queue_name or message.queue_name
        if not topic:
            raise ValueError("Topic name must be provided")
        
        serialized_message = message.serialize()
        future = self._producer.send(topic, serialized_message)
        record_metadata = future.get(timeout=10)
        
        logger.debug(f"Sent message {message.id} to Kafka topic {topic}")
        return message.id


class RabbitMQProducer(MessageProducer):
    """RabbitMQ message producer."""
    
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
            logger.info("Connected RabbitMQ producer")
        except ImportError:
            raise ImportError("pika package is required")
    
    async def disconnect(self) -> None:
        """Disconnect from RabbitMQ."""
        if self._channel:
            self._channel.close()
        if self._connection:
            self._connection.close()
        self._is_connected = False
        logger.info("Disconnected RabbitMQ producer")
    
    async def send(
        self, 
        message: Message, 
        queue_name: Optional[str] = None,
        routing_key: Optional[str] = None
    ) -> str:
        """Send a message to RabbitMQ."""
        queue_name = queue_name or message.queue_name
        if not queue_name:
            raise ValueError("Queue name must be provided")
        
        self._channel.queue_declare(queue=queue_name, durable=True)
        message_body = message.serialize()
        self._channel.basic_publish(
            exchange='',
            routing_key=queue_name,
            body=message_body
        )
        
        logger.debug(f"Sent message {message.id} to RabbitMQ queue {queue_name}")
        return message.id


class RedisProducer(MessageProducer):
    """Redis message producer."""
    
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
            logger.info("Connected Redis producer")
        except ImportError:
            raise ImportError("redis package is required")
    
    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        if self._redis_client:
            await self._redis_client.close()
        self._is_connected = False
        logger.info("Disconnected Redis producer")
    
    async def send(
        self, 
        message: Message, 
        queue_name: Optional[str] = None,
        routing_key: Optional[str] = None
    ) -> str:
        """Send a message to Redis."""
        queue_name = queue_name or message.queue_name
        if not queue_name:
            raise ValueError("Queue name must be provided")
        
        serialized_message = message.serialize().decode('utf-8')
        await self._redis_client.lpush(queue_name, serialized_message)
        
        logger.debug(f"Sent message {message.id} to Redis queue {queue_name}")
        return message.id
