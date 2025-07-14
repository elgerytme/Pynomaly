"""Message queue adapter factory."""

from __future__ import annotations

import logging
from typing import Type

from ..config.messaging_settings import MessagingSettings
from ..protocols.message_queue_protocol import MessageQueueProtocol
from .redis_queue_adapter import RedisQueueAdapter

logger = logging.getLogger(__name__)


class AdapterFactory:
    """Factory for creating message queue adapters."""

    _adapters: dict[str, Type[MessageQueueProtocol]] = {
        "redis": RedisQueueAdapter,
    }

    @classmethod
    def register_adapter(cls, backend_name: str, adapter_class: Type[MessageQueueProtocol]) -> None:
        """Register a message queue adapter.

        Args:
            backend_name: Name of the backend
            adapter_class: Adapter class
        """
        cls._adapters[backend_name] = adapter_class
        logger.info(f"Registered adapter for backend: {backend_name}")

    @classmethod
    def create_adapter(
        cls, 
        settings: MessagingSettings, 
        connection_url: str
    ) -> MessageQueueProtocol:
        """Create a message queue adapter based on settings.

        Args:
            settings: Messaging settings
            connection_url: Connection URL for the message queue

        Returns:
            Message queue adapter instance

        Raises:
            ValueError: If backend is not supported
        """
        backend = settings.queue_backend.lower()
        
        if backend not in cls._adapters:
            available = ", ".join(cls._adapters.keys())
            raise ValueError(
                f"Unsupported queue backend '{backend}'. "
                f"Available backends: {available}"
            )

        adapter_class = cls._adapters[backend]
        
        # Create adapter instance based on backend type
        if backend == "redis":
            return adapter_class(settings, connection_url)
        elif backend == "rabbitmq":
            return adapter_class(settings, connection_url)
        else:
            # Generic instantiation
            return adapter_class(settings, connection_url)

    @classmethod
    def get_available_backends(cls) -> list[str]:
        """Get list of available message queue backends.

        Returns:
            List of backend names
        """
        return list(cls._adapters.keys())


# Register optional adapters
try:
    from .rabbitmq_adapter import RabbitMQAdapter
    AdapterFactory.register_adapter("rabbitmq", RabbitMQAdapter)
except ImportError:
    logger.info("RabbitMQ adapter not available (aio_pika not installed)")


class ReliableAdapter(MessageQueueProtocol):
    """Wrapper adapter that adds reliability features."""

    def __init__(
        self, 
        base_adapter: MessageQueueProtocol, 
        settings: MessagingSettings,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        exponential_backoff: bool = True
    ):
        """Initialize reliable adapter wrapper.

        Args:
            base_adapter: Base adapter to wrap
            settings: Messaging settings
            max_retries: Maximum number of connection retries
            retry_delay: Initial retry delay in seconds
            exponential_backoff: Whether to use exponential backoff
        """
        self._adapter = base_adapter
        self._settings = settings
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._exponential_backoff = exponential_backoff
        self._connected = False

    async def _retry_operation(self, operation, *args, **kwargs):
        """Execute an operation with retry logic.

        Args:
            operation: Operation to execute
            *args: Operation arguments
            **kwargs: Operation keyword arguments

        Returns:
            Operation result

        Raises:
            Exception: If all retries are exhausted
        """
        last_exception = None
        delay = self._retry_delay

        for attempt in range(self._max_retries + 1):
            try:
                return await operation(*args, **kwargs)
            except Exception as e:
                last_exception = e
                
                if attempt < self._max_retries:
                    logger.warning(
                        f"Operation failed (attempt {attempt + 1}/{self._max_retries + 1}): {e}. "
                        f"Retrying in {delay}s..."
                    )
                    await asyncio.sleep(delay)
                    
                    if self._exponential_backoff:
                        delay *= 2
                else:
                    logger.error(f"Operation failed after {self._max_retries + 1} attempts: {e}")

        raise last_exception

    async def connect(self) -> None:
        """Establish connection with retry logic."""
        await self._retry_operation(self._adapter.connect)
        self._connected = True

    async def disconnect(self) -> None:
        """Close connection."""
        await self._adapter.disconnect()
        self._connected = False

    async def send_message(self, queue_name: str, message) -> bool:
        """Send message with retry logic."""
        return await self._retry_operation(self._adapter.send_message, queue_name, message)

    async def receive_message(self, queue_name: str, timeout: int | None = None):
        """Receive message with retry logic."""
        return await self._retry_operation(
            self._adapter.receive_message, queue_name, timeout
        )

    async def receive_messages(self, queue_name: str, batch_size: int = 10):
        """Receive messages with error handling."""
        async for message in self._adapter.receive_messages(queue_name, batch_size):
            yield message

    async def acknowledge_message(self, message) -> bool:
        """Acknowledge message with retry logic."""
        return await self._retry_operation(self._adapter.acknowledge_message, message)

    async def reject_message(self, message, requeue: bool = True) -> bool:
        """Reject message with retry logic."""
        return await self._retry_operation(
            self._adapter.reject_message, message, requeue
        )

    async def submit_task(self, task) -> str:
        """Submit task with retry logic."""
        return await self._retry_operation(self._adapter.submit_task, task)

    async def get_task_status(self, task_id: str):
        """Get task status with retry logic."""
        return await self._retry_operation(self._adapter.get_task_status, task_id)

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel task with retry logic."""
        return await self._retry_operation(self._adapter.cancel_task, task_id)

    async def get_queue_stats(self, queue_name: str) -> dict:
        """Get queue stats with retry logic."""
        return await self._retry_operation(self._adapter.get_queue_stats, queue_name)

    async def purge_queue(self, queue_name: str) -> int:
        """Purge queue with retry logic."""
        return await self._retry_operation(self._adapter.purge_queue, queue_name)

    async def create_queue(self, queue_name: str, **options) -> bool:
        """Create queue with retry logic."""
        return await self._retry_operation(
            self._adapter.create_queue, queue_name, **options
        )

    async def delete_queue(self, queue_name: str) -> bool:
        """Delete queue with retry logic."""
        return await self._retry_operation(self._adapter.delete_queue, queue_name)

    async def health_check(self) -> bool:
        """Health check with error handling."""
        try:
            return await self._adapter.health_check()
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False


# Import asyncio after class definitions to avoid circular imports
import asyncio