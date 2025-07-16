"""High-level message broker service."""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator

from ..adapters.adapter_factory import MessageQueueAdapterFactory
from ..config.messaging_settings import MessagingSettings
from ..models.simple_messages import Message, Task
from ..protocols.message_queue_protocol import MessageQueueProtocol

logger = logging.getLogger(__name__)


class MessageBroker:
    """High-level message broker service for publishing and consuming messages."""

    def __init__(self, settings: MessagingSettings, queue_url: str | None = None):
        """Initialize message broker.
        
        Args:
            settings: Messaging settings
            queue_url: Optional queue URL override
        """
        self.settings = settings
        self._adapter: MessageQueueProtocol | None = None
        self._queue_url = queue_url
        self._connected = False

    async def connect(self) -> None:
        """Connect to the message queue."""
        if self._connected:
            return

        try:
            self._adapter = MessageQueueAdapterFactory.create_adapter(
                self.settings, self._queue_url
            )
            await self._adapter.connect()
            self._connected = True
            logger.info("Message broker connected successfully")
        except Exception as e:
            logger.error(f"Failed to connect message broker: {e}")
            raise

    async def disconnect(self) -> None:
        """Disconnect from the message queue."""
        if self._adapter and self._connected:
            await self._adapter.disconnect()
            self._connected = False
            logger.info("Message broker disconnected")

    async def publish(self, queue_name: str, payload: dict[str, Any], 
                     headers: dict[str, str] | None = None, priority: int = 5) -> bool:
        """Publish a message to a queue.
        
        Args:
            queue_name: Name of the queue
            payload: Message payload
            headers: Optional message headers
            priority: Message priority (0-9, higher is more important)
            
        Returns:
            True if message was published successfully
        """
        if not self._adapter or not self._connected:
            await self.connect()

        message = Message(
            payload=payload,
            headers=headers or {},
            priority=priority
        )

        try:
            success = await self._adapter.send_message(queue_name, message)
            if success:
                logger.debug(f"Published message to queue {queue_name}")
            else:
                logger.warning(f"Failed to publish message to queue {queue_name}")
            return success
        except Exception as e:
            logger.error(f"Error publishing message to queue {queue_name}: {e}")
            return False

    async def consume(self, queue_name: str, timeout: int | None = None) -> Message | None:
        """Consume a single message from a queue.
        
        Args:
            queue_name: Name of the queue
            timeout: Timeout in seconds
            
        Returns:
            Message if available, None otherwise
        """
        if not self._adapter or not self._connected:
            await self.connect()

        try:
            message = await self._adapter.receive_message(queue_name, timeout)
            if message:
                logger.debug(f"Consumed message from queue {queue_name}")
            return message
        except Exception as e:
            logger.error(f"Error consuming message from queue {queue_name}: {e}")
            return None

    async def consume_messages(self, queue_name: str, 
                              batch_size: int | None = None) -> AsyncIterator[Message]:
        """Consume messages from a queue as an async iterator.
        
        Args:
            queue_name: Name of the queue
            batch_size: Number of messages to fetch at once
            
        Yields:
            Messages from the queue
        """
        if not self._adapter or not self._connected:
            await self.connect()

        batch_size = batch_size or self.settings.task_batch_size

        try:
            async for message in self._adapter.receive_messages(queue_name, batch_size):
                logger.debug(f"Consumed message from queue {queue_name}")
                yield message
        except Exception as e:
            logger.error(f"Error consuming messages from queue {queue_name}: {e}")

    async def acknowledge(self, message: Message) -> bool:
        """Acknowledge that a message has been processed.
        
        Args:
            message: Message to acknowledge
            
        Returns:
            True if acknowledgment was successful
        """
        if not self._adapter:
            return False

        try:
            success = await self._adapter.acknowledge_message(message)
            if success:
                logger.debug(f"Acknowledged message {message.id}")
            return success
        except Exception as e:
            logger.error(f"Error acknowledging message {message.id}: {e}")
            return False

    async def reject(self, message: Message, requeue: bool = True) -> bool:
        """Reject a message.
        
        Args:
            message: Message to reject
            requeue: Whether to requeue the message
            
        Returns:
            True if rejection was successful
        """
        if not self._adapter:
            return False

        try:
            success = await self._adapter.reject_message(message, requeue)
            if success:
                action = "requeued" if requeue else "sent to DLQ"
                logger.debug(f"Rejected message {message.id} ({action})")
            return success
        except Exception as e:
            logger.error(f"Error rejecting message {message.id}: {e}")
            return False

    async def submit_task(self, task_type: str, data: dict[str, Any], 
                         priority: int = 5) -> str:
        """Submit a task for processing.
        
        Args:
            task_type: Type of task
            data: Task data
            priority: Task priority
            
        Returns:
            Task ID if successful, empty string otherwise
        """
        if not self._adapter or not self._connected:
            await self.connect()

        task = Task(
            task_type=task_type,
            data=data,
            priority=priority
        )

        try:
            task_id = await self._adapter.submit_task(task)
            if task_id:
                logger.debug(f"Submitted task {task_id} of type {task_type}")
            return task_id
        except Exception as e:
            logger.error(f"Error submitting task of type {task_type}: {e}")
            return ""

    async def get_queue_stats(self, queue_name: str) -> dict[str, Any]:
        """Get statistics for a queue.
        
        Args:
            queue_name: Name of the queue
            
        Returns:
            Queue statistics
        """
        if not self._adapter or not self._connected:
            return {}

        try:
            stats = await self._adapter.get_queue_stats(queue_name)
            logger.debug(f"Retrieved stats for queue {queue_name}")
            return stats
        except Exception as e:
            logger.error(f"Error getting stats for queue {queue_name}: {e}")
            return {"error": str(e)}

    async def purge_queue(self, queue_name: str) -> int:
        """Remove all messages from a queue.
        
        Args:
            queue_name: Name of the queue
            
        Returns:
            Number of messages removed
        """
        if not self._adapter or not self._connected:
            return 0

        try:
            count = await self._adapter.purge_queue(queue_name)
            logger.info(f"Purged {count} messages from queue {queue_name}")
            return count
        except Exception as e:
            logger.error(f"Error purging queue {queue_name}: {e}")
            return 0

    async def health_check(self) -> bool:
        """Check if the message broker is healthy.
        
        Returns:
            True if healthy
        """
        if not self._adapter or not self._connected:
            return False

        try:
            return await self._adapter.health_check()
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()