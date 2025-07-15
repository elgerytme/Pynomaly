"""Message queue adapter factory."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..config.messaging_settings import MessagingSettings
from ..protocols.message_queue_protocol import MessageQueueProtocol

if TYPE_CHECKING:
    from .redis_queue_adapter import RedisQueueAdapter


class MessageQueueAdapterFactory:
    """Factory for creating message queue adapters."""

    @staticmethod
    def create_adapter(settings: MessagingSettings, queue_url: str | None = None) -> MessageQueueProtocol:
        """Create a message queue adapter based on settings.
        
        Args:
            settings: Messaging settings
            queue_url: Optional queue URL override
            
        Returns:
            Message queue adapter instance
            
        Raises:
            ValueError: If backend is not supported
            ImportError: If required dependencies are missing
        """
        backend = settings.queue_backend.lower()
        url = queue_url or settings.queue_url or settings.get_redis_url()
        
        if backend == "redis":
            from .redis_queue_adapter import RedisQueueAdapter
            return RedisQueueAdapter(settings, url)
        elif backend == "rabbitmq":
            # Future implementation
            raise NotImplementedError("RabbitMQ adapter not yet implemented")
        elif backend == "celery":
            # Future implementation
            raise NotImplementedError("Celery adapter not yet implemented")
        else:
            raise ValueError(f"Unsupported message queue backend: {backend}")

    @staticmethod
    def get_supported_backends() -> list[str]:
        """Get list of supported backends."""
        return ["redis", "rabbitmq", "celery"]