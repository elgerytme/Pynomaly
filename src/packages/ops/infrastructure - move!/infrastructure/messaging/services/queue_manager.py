"""Queue management service."""

from __future__ import annotations

import logging
from typing import Any

from .message_broker import MessageBroker

logger = logging.getLogger(__name__)


class QueueManager:
    """Service for managing queues and their lifecycle."""

    def __init__(self, message_broker: MessageBroker):
        """Initialize queue manager.
        
        Args:
            message_broker: Message broker instance
        """
        self.broker = message_broker
        self._managed_queues: set[str] = set()

    async def create_queue(self, queue_name: str, **options) -> bool:
        """Create a new queue.
        
        Args:
            queue_name: Name of the queue
            **options: Queue-specific options
            
        Returns:
            True if queue was created successfully
        """
        if not self.broker._adapter:
            await self.broker.connect()

        try:
            success = await self.broker._adapter.create_queue(queue_name, **options)
            if success:
                self._managed_queues.add(queue_name)
                logger.info(f"Created queue {queue_name}")
            return success
        except Exception as e:
            logger.error(f"Failed to create queue {queue_name}: {e}")
            return False

    async def delete_queue(self, queue_name: str) -> bool:
        """Delete a queue.
        
        Args:
            queue_name: Name of the queue
            
        Returns:
            True if queue was deleted successfully
        """
        if not self.broker._adapter:
            return False

        try:
            success = await self.broker._adapter.delete_queue(queue_name)
            if success:
                self._managed_queues.discard(queue_name)
                logger.info(f"Deleted queue {queue_name}")
            return success
        except Exception as e:
            logger.error(f"Failed to delete queue {queue_name}: {e}")
            return False

    async def get_queue_info(self, queue_name: str) -> dict[str, Any]:
        """Get information about a queue.
        
        Args:
            queue_name: Name of the queue
            
        Returns:
            Queue information dictionary
        """
        stats = await self.broker.get_queue_stats(queue_name)
        return {
            "name": queue_name,
            "managed": queue_name in self._managed_queues,
            "stats": stats
        }

    async def list_queues(self) -> list[str]:
        """List all managed queues.
        
        Returns:
            List of queue names
        """
        return list(self._managed_queues)

    async def purge_all_queues(self) -> dict[str, int]:
        """Purge all managed queues.
        
        Returns:
            Dictionary with queue names and number of messages purged
        """
        results = {}
        for queue_name in self._managed_queues:
            try:
                count = await self.broker.purge_queue(queue_name)
                results[queue_name] = count
            except Exception as e:
                logger.error(f"Failed to purge queue {queue_name}: {e}")
                results[queue_name] = -1
        
        return results

    def register_queue(self, queue_name: str) -> None:
        """Register a queue as managed without creating it.
        
        Args:
            queue_name: Name of the queue
        """
        self._managed_queues.add(queue_name)
        logger.debug(f"Registered queue {queue_name}")

    def unregister_queue(self, queue_name: str) -> None:
        """Unregister a queue from management.
        
        Args:
            queue_name: Name of the queue
        """
        self._managed_queues.discard(queue_name)
        logger.debug(f"Unregistered queue {queue_name}")