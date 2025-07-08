"""
RabbitMQ Message Queue Implementation

This module provides a RabbitMQ-based message queue implementation.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional

from ..core import MessageQueue, Message
from ..config import MessageQueueConfig, QueueConfig

logger = logging.getLogger(__name__)


class RabbitMQMessageQueue(MessageQueue):
    """RabbitMQ-based message queue implementation."""
    
    def __init__(self, config: MessageQueueConfig):
        super().__init__(config)
        self._connection = None
        self._channel = None
    
    async def connect(self) -> None:
        """Connect to RabbitMQ."""
        try:
            import pika
            
            # Create connection parameters
            credentials = pika.PlainCredentials(
                self.config.connection.username or 'guest',
                self.config.connection.password or 'guest'
            )
            
            parameters = pika.ConnectionParameters(
                host=self.config.connection.host,
                port=self.config.connection.port,
                virtual_host=self.config.connection.virtual_host,
                credentials=credentials,
                heartbeat=self.config.connection.heartbeat,
                connection_attempts=3,
                retry_delay=1
            )
            
            # Create connection and channel
            self._connection = pika.BlockingConnection(parameters)
            self._channel = self._connection.channel()
            
            self._is_connected = True
            logger.info("Connected to RabbitMQ message queue")
            
        except ImportError:
            raise ImportError("pika package is required for RabbitMQ message queue")
        except Exception as e:
            logger.error(f"Failed to connect to RabbitMQ: {e}")
            raise
    
    async def disconnect(self) -> None:
        """Disconnect from RabbitMQ."""
        if self._channel:
            self._channel.close()
        if self._connection:
            self._connection.close()
        self._is_connected = False
        logger.info("Disconnected from RabbitMQ message queue")
    
    async def send(
        self, 
        message: Message, 
        queue_name: Optional[str] = None,
        routing_key: Optional[str] = None
    ) -> str:
        """Send a message to a RabbitMQ queue."""
        queue_name = queue_name or message.queue_name
        if not queue_name:
            raise ValueError("Queue name must be provided")
        
        # Ensure queue exists
        self._channel.queue_declare(queue=queue_name, durable=True)
        
        # Send message
        message_body = message.serialize()
        self._channel.basic_publish(
            exchange='',
            routing_key=queue_name,
            body=message_body,
            properties=pika.BasicProperties(
                delivery_mode=2,  # Make message persistent
                priority=message.priority,
                correlation_id=message.correlation_id,
                reply_to=message.reply_to,
                headers=message.headers
            )
        )
        
        self.state.messages_sent += 1
        logger.debug(f"Sent message ID {message.id} to RabbitMQ queue {queue_name}")
        return message.id
    
    async def receive(
        self, 
        queue_name: str, 
        timeout: Optional[float] = None
    ) -> Optional[Message]:
        """Receive a message from a RabbitMQ queue."""
        # Ensure queue exists
        self._channel.queue_declare(queue=queue_name, durable=True)
        
        try:
            method_frame, header_frame, body = self._channel.basic_get(queue=queue_name)
            
            if method_frame:
                message = Message.deserialize(body)
                message.queue_name = queue_name
                
                # Store delivery info for acknowledgment
                message.headers['delivery_tag'] = method_frame.delivery_tag
                
                logger.debug(f"Received message ID {message.id} from RabbitMQ queue {queue_name}")
                return message
        except Exception as e:
            logger.error(f"Error receiving from RabbitMQ queue {queue_name}: {e}")
        
        return None
    
    async def acknowledge(self, message: Message) -> None:
        """Acknowledge message processing."""
        delivery_tag = message.headers.get('delivery_tag')
        if delivery_tag:
            self._channel.basic_ack(delivery_tag=delivery_tag)
        logger.debug(f"Acknowledged message ID {message.id}")
    
    async def reject(self, message: Message, requeue: bool = False) -> None:
        """Reject a message and optionally requeue it."""
        delivery_tag = message.headers.get('delivery_tag')
        if delivery_tag:
            self._channel.basic_reject(delivery_tag=delivery_tag, requeue=requeue)
        logger.debug(f"Rejected message ID {message.id}, requeue={requeue}")
    
    async def create_queue(self, queue_config: QueueConfig) -> None:
        """Create a RabbitMQ queue."""
        self._channel.queue_declare(
            queue=queue_config.name,
            durable=queue_config.durable,
            exclusive=queue_config.exclusive,
            auto_delete=queue_config.auto_delete,
            arguments=queue_config.arguments
        )
        self.state.active_queues[queue_config.name] = queue_config
        logger.info(f"Created RabbitMQ queue {queue_config.name}")
    
    async def delete_queue(self, queue_name: str) -> None:
        """Delete a RabbitMQ queue."""
        self._channel.queue_delete(queue=queue_name)
        if queue_name in self.state.active_queues:
            del self.state.active_queues[queue_name]
        logger.info(f"Deleted RabbitMQ queue {queue_name}")
    
    async def purge_queue(self, queue_name: str) -> int:
        """Purge messages from a RabbitMQ queue."""
        result = self._channel.queue_purge(queue=queue_name)
        logger.info(f"Purged {result.method.message_count} messages from RabbitMQ queue {queue_name}")
        return result.method.message_count
    
    async def get_queue_size(self, queue_name: str) -> int:
        """Get the number of messages in a RabbitMQ queue."""
        result = self._channel.queue_declare(queue=queue_name, passive=True)
        return result.method.message_count
    
    async def health_check(self) -> bool:
        """Check the health of the RabbitMQ connection."""
        try:
            return self._connection and self._connection.is_open
        except Exception:
            return False
