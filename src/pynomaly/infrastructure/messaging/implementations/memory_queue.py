"""
In-memory Message Queue Implementation

This module provides an in-memory message queue implementation for testing and development.
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from typing import Deque, Optional, Dict

from ..core import MessageQueue, Message
from ..config import MessageQueueConfig, QueueConfig

logger = logging.getLogger(__name__)


class MemoryMessageQueue(MessageQueue):
    """In-memory message queue for testing purposes."""
    
    def __init__(self, config: MessageQueueConfig):
        super().__init__(config)
        self.queues: Dict[str, Deque[Message]] = {}
    
    async def connect(self) -> None:
        """Connect to in-memory queue (no-op)."""
        self._is_connected = True
        logger.info("Connected to in-memory message queue")
    
    async def disconnect(self) -> None:
        """Disconnect from in-memory queue (no-op)."""
        self._is_connected = False
        logger.info("Disconnected from in-memory message queue")
    
    async def send(
        self, 
        message: Message, 
        queue_name: Optional[str] = None,
        routing_key: Optional[str] = None
    ) -> str:
        """Send a message to an in-memory queue."""
        queue_name = queue_name or message.queue_name
        if not queue_name:
            raise ValueError("Queue name must be provided")
        
        if queue_name not in self.queues:
            self.queues[queue_name] = deque()
        
        self.queues[queue_name].append(message)
        self.state.messages_sent += 1
        
        logger.debug(f'Sent message ID {message.id} to queue {queue_name}')
        return message.id
    
    async def receive(
        self, 
        queue_name: str, 
        timeout: Optional[float] = None
    ) -> Optional[Message]:
        """Receive a message from an in-memory queue."""
        queue = self.queues.get(queue_name)
        if not queue:
            return None
        
        try:
            message = queue.popleft()
            logger.debug(f"Received message ID {message.id} from queue {queue_name}")
            return message
        except IndexError:
            return None
    
    async def acknowledge(self, message: Message) -> None:
        """Acknowledge message processing (no-op)."""
        logger.debug(f"Acknowledged message ID {message.id}")
    
    async def reject(self, message: Message, requeue: bool = False) -> None:
        """Reject a message and optionally requeue it."""
        if requeue:
            queue = self.queues.get(message.queue_name)
            if queue:
                queue.appendleft(message)
        logger.debug(f"Rejected message ID {message.id}, requeue={requeue}")
    
    async def create_queue(self, queue_config: QueueConfig) -> None:
        """Create an in-memory queue."""
        if queue_config.name not in self.queues:
            self.queues[queue_config.name] = deque()
        self.state.active_queues[queue_config.name] = queue_config
        logger.info(f"Created in-memory queue {queue_config.name}")
    
    async def delete_queue(self, queue_name: str) -> None:
        """Delete an in-memory queue."""
        if queue_name in self.queues:
            del self.queues[queue_name]
        if queue_name in self.state.active_queues:
            del self.state.active_queues[queue_name]
        logger.info(f"Deleted in-memory queue {queue_name}")
    
    async def purge_queue(self, queue_name: str) -> int:
        """Purge messages from an in-memory queue."""
        queue = self.queues.get(queue_name)
        if not queue:
            return 0
        count = len(queue)
        queue.clear()
        logger.info(f"Purged {count} messages from queue {queue_name}")
        return count
    
    async def get_queue_size(self, queue_name: str) -> int:
        """Get the number of messages in an in-memory queue."""
        queue = self.queues.get(queue_name)
        return len(queue) if queue else 0
    
    async def health_check(self) -> bool:
        """Check the health of the in-memory message queue (always healthy)."""
        return True
