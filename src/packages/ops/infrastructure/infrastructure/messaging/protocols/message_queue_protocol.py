"""Message queue protocol interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator

from ..models.simple_messages import Message, Task


class MessageQueueProtocol(ABC):
    """Protocol interface for message queue implementations."""

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the message queue."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to the message queue."""
        pass

    @abstractmethod
    async def send_message(self, queue_name: str, message: Message) -> bool:
        """Send a message to a queue.
        
        Args:
            queue_name: Name of the queue
            message: Message to send
            
        Returns:
            True if message was sent successfully
        """
        pass

    @abstractmethod
    async def receive_message(self, queue_name: str, timeout: int | None = None) -> Message | None:
        """Receive a message from a queue.
        
        Args:
            queue_name: Name of the queue
            timeout: Timeout in seconds (None for blocking)
            
        Returns:
            Message if available, None if timeout
        """
        pass

    @abstractmethod
    async def receive_messages(self, queue_name: str, batch_size: int = 10) -> AsyncIterator[Message]:
        """Receive messages from a queue as an async iterator.
        
        Args:
            queue_name: Name of the queue
            batch_size: Number of messages to fetch at once
            
        Yields:
            Messages from the queue
        """
        pass

    @abstractmethod
    async def acknowledge_message(self, message: Message) -> bool:
        """Acknowledge that a message has been processed.
        
        Args:
            message: Message to acknowledge
            
        Returns:
            True if acknowledgment was successful
        """
        pass

    @abstractmethod
    async def reject_message(self, message: Message, requeue: bool = True) -> bool:
        """Reject a message (send to dead letter queue or requeue).
        
        Args:
            message: Message to reject
            requeue: Whether to requeue the message
            
        Returns:
            True if rejection was successful
        """
        pass

    @abstractmethod
    async def submit_task(self, task: Task) -> str:
        """Submit a task for processing.
        
        Args:
            task: Task to submit
            
        Returns:
            Task ID
        """
        pass

    @abstractmethod
    async def get_task_status(self, task_id: str) -> Task | None:
        """Get the current status of a task.
        
        Args:
            task_id: ID of the task
            
        Returns:
            Task with current status, None if not found
        """
        pass

    @abstractmethod
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending task.
        
        Args:
            task_id: ID of the task to cancel
            
        Returns:
            True if task was cancelled successfully
        """
        pass

    @abstractmethod
    async def get_queue_stats(self, queue_name: str) -> dict[str, Any]:
        """Get statistics for a queue.
        
        Args:
            queue_name: Name of the queue
            
        Returns:
            Dictionary with queue statistics
        """
        pass

    @abstractmethod
    async def purge_queue(self, queue_name: str) -> int:
        """Remove all messages from a queue.
        
        Args:
            queue_name: Name of the queue
            
        Returns:
            Number of messages removed
        """
        pass

    @abstractmethod
    async def create_queue(self, queue_name: str, **options) -> bool:
        """Create a new queue.
        
        Args:
            queue_name: Name of the queue
            **options: Queue-specific options
            
        Returns:
            True if queue was created successfully
        """
        pass

    @abstractmethod
    async def delete_queue(self, queue_name: str) -> bool:
        """Delete a queue.
        
        Args:
            queue_name: Name of the queue
            
        Returns:
            True if queue was deleted successfully
        """
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the message queue connection is healthy.
        
        Returns:
            True if connection is healthy
        """
        pass