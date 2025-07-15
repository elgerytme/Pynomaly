"""Queue management service."""

from __future__ import annotations

import logging
from typing import Any

from ..adapters.redis_queue_adapter import RedisQueueAdapter
from ..config.messaging_settings import MessagingSettings
from ..models.messages import Message
from ..models.tasks import Task
from ..protocols.message_queue_protocol import MessageQueueProtocol

logger = logging.getLogger(__name__)


class QueueManager:
    """High-level queue management service."""

    def __init__(self, adapter: MessageQueueProtocol, settings: MessagingSettings):
        """Initialize queue manager.

        Args:
            adapter: Message queue adapter implementation
            settings: Messaging settings
        """
        self._adapter = adapter
        self._settings = settings
        self._connected = False

    async def start(self) -> None:
        """Start the queue manager."""
        try:
            await self._adapter.connect()
            self._connected = True
            logger.info("Queue manager started successfully")
        except Exception as e:
            logger.error(f"Failed to start queue manager: {e}")
            raise

    async def stop(self) -> None:
        """Stop the queue manager."""
        try:
            await self._adapter.disconnect()
            self._connected = False
            logger.info("Queue manager stopped")
        except Exception as e:
            logger.error(f"Error stopping queue manager: {e}")

    async def send_message(self, queue_name: str, payload: dict[str, Any], **kwargs) -> str:
        """Send a message to a queue.

        Args:
            queue_name: Name of the queue
            payload: Message payload
            **kwargs: Additional message options

        Returns:
            Message ID
        """
        if not self._connected:
            raise RuntimeError("Queue manager not connected")

        message = Message(
            queue_name=queue_name,
            payload=payload,
            **kwargs
        )

        success = await self._adapter.send_message(queue_name, message)
        if not success:
            raise RuntimeError(f"Failed to send message to queue {queue_name}")

        logger.debug(f"Message {message.id} sent to queue {queue_name}")
        return message.id

    async def submit_task(self, task: Task) -> str:
        """Submit a task for processing.

        Args:
            task: Task to submit

        Returns:
            Task ID
        """
        if not self._connected:
            raise RuntimeError("Queue manager not connected")

        task_id = await self._adapter.submit_task(task)
        logger.info(f"Task {task_id} submitted for processing")
        return task_id

    async def get_task_status(self, task_id: str) -> Task | None:
        """Get task status.

        Args:
            task_id: ID of the task

        Returns:
            Task with current status, None if not found
        """
        return await self._adapter.get_task_status(task_id)

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a task.

        Args:
            task_id: ID of the task to cancel

        Returns:
            True if task was cancelled successfully
        """
        return await self._adapter.cancel_task(task_id)

    async def get_queue_stats(self, queue_name: str) -> dict[str, Any]:
        """Get queue statistics.

        Args:
            queue_name: Name of the queue

        Returns:
            Dictionary with queue statistics
        """
        return await self._adapter.get_queue_stats(queue_name)

    async def create_queue(self, queue_name: str, **options) -> bool:
        """Create a new queue.

        Args:
            queue_name: Name of the queue
            **options: Queue-specific options

        Returns:
            True if queue was created successfully
        """
        return await self._adapter.create_queue(queue_name, **options)

    async def delete_queue(self, queue_name: str) -> bool:
        """Delete a queue.

        Args:
            queue_name: Name of the queue

        Returns:
            True if queue was deleted successfully
        """
        return await self._adapter.delete_queue(queue_name)

    async def purge_queue(self, queue_name: str) -> int:
        """Purge all messages from a queue.

        Args:
            queue_name: Name of the queue

        Returns:
            Number of messages removed
        """
        return await self._adapter.purge_queue(queue_name)

    async def health_check(self) -> bool:
        """Check if the queue manager is healthy.

        Returns:
            True if healthy
        """
        return await self._adapter.health_check()

    @classmethod
    def create_redis_manager(cls, settings: MessagingSettings, redis_url: str) -> QueueManager:
        """Create a queue manager with Redis adapter.

        Args:
            settings: Messaging settings
            redis_url: Redis connection URL

        Returns:
            Configured queue manager
        """
        adapter = RedisQueueAdapter(settings, redis_url)
        return cls(adapter, settings)