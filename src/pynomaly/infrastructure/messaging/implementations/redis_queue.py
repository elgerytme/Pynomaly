"""
Redis Message Queue Implementation

This module provides a Redis-based message queue implementation.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional

from ..core import MessageQueue, Message
from ..config import MessageQueueConfig, QueueConfig

logger = logging.getLogger(__name__)


class RedisMessageQueue(MessageQueue):
    """Redis-based message queue implementation."""
    
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
                db=self.config.connection.db,
                decode_responses=True
            )
            # Test connection
            await self._redis_client.ping()
            self._is_connected = True
            logger.info("Connected to Redis message queue")
        except ImportError:
            raise ImportError("redis package is required for Redis message queue")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        if self._redis_client:
            await self._redis_client.close()
        self._is_connected = False
        logger.info("Disconnected from Redis message queue")
    
    async def send(
        self, 
        message: Message, 
        queue_name: Optional[str] = None,
        routing_key: Optional[str] = None
    ) -> str:
        """Send a message to a Redis queue."""
        queue_name = queue_name or message.queue_name
        if not queue_name:
            raise ValueError("Queue name must be provided")
        
        serialized_message = message.serialize().decode('utf-8')
        await self._redis_client.lpush(queue_name, serialized_message)
        self.state.messages_sent += 1
        
        logger.debug(f"Sent message ID {message.id} to Redis queue {queue_name}")
        return message.id
    
    async def receive(
        self, 
        queue_name: str, 
        timeout: Optional[float] = None
    ) -> Optional[Message]:
        """Receive a message from a Redis queue."""
        timeout_seconds = int(timeout) if timeout else 0
        
        try:
            result = await self._redis_client.brpop(queue_name, timeout=timeout_seconds)
            if result:
                _, message_data = result
                message = Message.deserialize(message_data.encode('utf-8'))
                logger.debug(f"Received message ID {message.id} from Redis queue {queue_name}")
                return message
        except Exception as e:
            logger.error(f"Error receiving from Redis queue {queue_name}: {e}")
        
        return None
    
    async def acknowledge(self, message: Message) -> None:
        """Acknowledge message processing (no-op for basic Redis implementation)."""
        logger.debug(f"Acknowledged message ID {message.id}")
    
    async def reject(self, message: Message, requeue: bool = False) -> None:
        """Reject a message and optionally requeue it."""
        if requeue and message.queue_name:
            await self.send(message, message.queue_name)
        logger.debug(f"Rejected message ID {message.id}, requeue={requeue}")
    
    async def create_queue(self, queue_config: QueueConfig) -> None:
        """Create a Redis queue (no-op, Redis lists are created on demand)."""
        self.state.active_queues[queue_config.name] = queue_config
        logger.info(f"Created Redis queue {queue_config.name}")
    
    async def delete_queue(self, queue_name: str) -> None:
        """Delete a Redis queue."""
        await self._redis_client.delete(queue_name)
        if queue_name in self.state.active_queues:
            del self.state.active_queues[queue_name]
        logger.info(f"Deleted Redis queue {queue_name}")
    
    async def purge_queue(self, queue_name: str) -> int:
        """Purge messages from a Redis queue."""
        length = await self._redis_client.llen(queue_name)
        await self._redis_client.delete(queue_name)
        logger.info(f"Purged {length} messages from Redis queue {queue_name}")
        return length
    
    async def get_queue_size(self, queue_name: str) -> int:
        """Get the number of messages in a Redis queue."""
        return await self._redis_client.llen(queue_name)
    
    async def health_check(self) -> bool:
        """Check the health of the Redis connection."""
        try:
            await self._redis_client.ping()
            return True
        except Exception:
            return False
