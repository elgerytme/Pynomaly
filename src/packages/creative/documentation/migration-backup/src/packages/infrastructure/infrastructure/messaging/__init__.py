"""Messaging infrastructure module.

This module provides a comprehensive messaging system with support for
Redis-based message queues, task processing, and distributed communication.
"""

from .adapters.adapter_factory import MessageQueueAdapterFactory
from .adapters.redis_queue_adapter import RedisQueueAdapter
from .config.messaging_settings import MessagingSettings
from .models.simple_messages import Message, MessageStatus, Task, TaskStatus
from .protocols.message_queue_protocol import MessageQueueProtocol
from .services.message_broker import MessageBroker
from .services.queue_manager import QueueManager

__all__ = [
    # Configuration
    "MessagingSettings",
    
    # Models
    "Message",
    "MessageStatus", 
    "Task",
    "TaskStatus",
    
    # Protocols
    "MessageQueueProtocol",
    
    # Adapters
    "MessageQueueAdapterFactory",
    "RedisQueueAdapter",
    
    # Services
    "MessageBroker",
    "QueueManager",
]