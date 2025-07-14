"""Message queue integration for Pynomaly.

This module provides message queue functionality for:
- Asynchronous task processing
- Event-driven communication
- Background job execution
- Inter-service messaging
"""

from .adapters.adapter_factory import AdapterFactory, ReliableAdapter
from .adapters.redis_queue_adapter import RedisQueueAdapter
from .config.messaging_settings import MessagingSettings
from .models.messages import Message, MessagePriority, MessageStatus
from .models.tasks import Task, TaskStatus, TaskType
from .protocols.message_queue_protocol import MessageQueueProtocol
from .services.message_broker import MessageBroker
from .services.queue_manager import QueueManager
from .services.task_processor import TaskProcessor

__all__ = [
    "AdapterFactory",
    "ReliableAdapter",
    "RedisQueueAdapter",
    "MessagingSettings", 
    "Message",
    "MessagePriority",
    "MessageStatus",
    "Task",
    "TaskStatus",
    "TaskType",
    "MessageQueueProtocol",
    "MessageBroker",
    "QueueManager",
    "TaskProcessor",
]