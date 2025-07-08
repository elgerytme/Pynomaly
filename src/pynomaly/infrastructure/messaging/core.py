"""
Core Message Queue Components

This module defines the core classes and interfaces for message queue integration.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Union

from .config import MessageQueueConfig, MessageQueueState, QueueConfig

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """Represents a message in the queue system."""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    body: Union[str, bytes, Dict[str, Any]] = field(default="")
    headers: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Message metadata
    queue_name: Optional[str] = None
    routing_key: Optional[str] = None
    correlation_id: Optional[str] = None
    reply_to: Optional[str] = None
    content_type: str = "application/json"
    content_encoding: Optional[str] = None
    
    # Delivery information
    delivery_count: int = 0
    priority: int = 0
    ttl: Optional[int] = None
    
    # Tracing and monitoring
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        return {
            "id": self.id,
            "body": self.body,
            "headers": self.headers,
            "timestamp": self.timestamp.isoformat(),
            "queue_name": self.queue_name,
            "routing_key": self.routing_key,
            "correlation_id": self.correlation_id,
            "reply_to": self.reply_to,
            "content_type": self.content_type,
            "content_encoding": self.content_encoding,
            "delivery_count": self.delivery_count,
            "priority": self.priority,
            "ttl": self.ttl,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Message:
        """Create message from dictionary."""
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        elif timestamp is None:
            timestamp = datetime.now(timezone.utc)
        
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            body=data.get("body", ""),
            headers=data.get("headers", {}),
            timestamp=timestamp,
            queue_name=data.get("queue_name"),
            routing_key=data.get("routing_key"),
            correlation_id=data.get("correlation_id"),
            reply_to=data.get("reply_to"),
            content_type=data.get("content_type", "application/json"),
            content_encoding=data.get("content_encoding"),
            delivery_count=data.get("delivery_count", 0),
            priority=data.get("priority", 0),
            ttl=data.get("ttl"),
            trace_id=data.get("trace_id"),
            span_id=data.get("span_id"),
        )
    
    def serialize(self) -> bytes:
        """Serialize message to bytes."""
        return json.dumps(self.to_dict()).encode("utf-8")
    
    @classmethod
    def deserialize(cls, data: bytes) -> Message:
        """Deserialize message from bytes."""
        return cls.from_dict(json.loads(data.decode("utf-8")))


class MessageHandler(ABC):
    """Abstract base class for message handlers."""
    
    @abstractmethod
    async def handle(self, message: Message) -> Any:
        """Handle a message."""
        pass
    
    @abstractmethod
    async def handle_error(self, message: Message, error: Exception) -> None:
        """Handle message processing errors."""
        pass


class AsyncMessageHandler(MessageHandler):
    """Async message handler with function wrapper."""
    
    def __init__(self, handler_func: Callable[[Message], Any]):
        self.handler_func = handler_func
        self.name = getattr(handler_func, "__name__", "anonymous")
    
    async def handle(self, message: Message) -> Any:
        """Handle message with wrapped function."""
        try:
            if asyncio.iscoroutinefunction(self.handler_func):
                return await self.handler_func(message)
            else:
                return self.handler_func(message)
        except Exception as e:
            logger.error(f"Error in handler {self.name}: {e}")
            await self.handle_error(message, e)
            raise
    
    async def handle_error(self, message: Message, error: Exception) -> None:
        """Default error handling."""
        logger.error(f"Message {message.id} failed: {error}")
        
        # Increment delivery count
        message.delivery_count += 1
        
        # Could implement retry logic here
        # For now, just log the error


class MessageQueue(ABC):
    """Abstract base class for message queues."""
    
    def __init__(self, config: MessageQueueConfig):
        self.config = config
        self.state = MessageQueueState()
        self.handlers: Dict[str, MessageHandler] = {}
        self.middleware: List[Any] = []
        self._is_connected = False
        self._connection = None
    
    @abstractmethod
    async def connect(self) -> None:
        """Connect to message broker."""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from message broker."""
        pass
    
    @abstractmethod
    async def send(
        self, 
        message: Message, 
        queue_name: Optional[str] = None,
        routing_key: Optional[str] = None
    ) -> str:
        """Send a message to a queue."""
        pass
    
    @abstractmethod
    async def receive(
        self, 
        queue_name: str, 
        timeout: Optional[float] = None
    ) -> Optional[Message]:
        """Receive a message from a queue."""
        pass
    
    @abstractmethod
    async def acknowledge(self, message: Message) -> None:
        """Acknowledge message processing."""
        pass
    
    @abstractmethod
    async def reject(self, message: Message, requeue: bool = False) -> None:
        """Reject a message."""
        pass
    
    @abstractmethod
    async def create_queue(self, queue_config: QueueConfig) -> None:
        """Create a queue."""
        pass
    
    @abstractmethod
    async def delete_queue(self, queue_name: str) -> None:
        """Delete a queue."""
        pass
    
    @abstractmethod
    async def purge_queue(self, queue_name: str) -> int:
        """Purge messages from a queue."""
        pass
    
    @abstractmethod
    async def get_queue_size(self, queue_name: str) -> int:
        """Get the number of messages in a queue."""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check the health of the message queue."""
        pass
    
    # Common implementations
    
    def register_handler(self, queue_name: str, handler: MessageHandler) -> None:
        """Register a message handler for a queue."""
        self.handlers[queue_name] = handler
        logger.info(f"Registered handler for queue: {queue_name}")
    
    def register_function_handler(
        self, 
        queue_name: str, 
        handler_func: Callable[[Message], Any]
    ) -> None:
        """Register a function as a message handler."""
        handler = AsyncMessageHandler(handler_func)
        self.register_handler(queue_name, handler)
    
    def add_middleware(self, middleware: Any) -> None:
        """Add middleware to the message processing pipeline."""
        self.middleware.append(middleware)
    
    async def process_message(self, message: Message) -> Any:
        """Process a message through middleware and handlers."""
        # Apply middleware in order
        for middleware in self.middleware:
            if hasattr(middleware, 'before_process'):
                await middleware.before_process(message)
        
        try:
            # Find and execute handler
            handler = self.handlers.get(message.queue_name)
            if handler:
                result = await handler.handle(message)
                
                # Apply middleware after processing
                for middleware in reversed(self.middleware):
                    if hasattr(middleware, 'after_process'):
                        await middleware.after_process(message, result)
                
                return result
            else:
                logger.warning(f"No handler found for queue: {message.queue_name}")
                return None
                
        except Exception as e:
            # Apply error middleware
            for middleware in reversed(self.middleware):
                if hasattr(middleware, 'on_error'):
                    await middleware.on_error(message, e)
            raise
    
    async def start_consuming(self, queue_name: str) -> None:
        """Start consuming messages from a queue."""
        if not self._is_connected:
            await self.connect()
        
        logger.info(f"Starting to consume from queue: {queue_name}")
        
        while self._is_connected:
            try:
                message = await self.receive(queue_name, timeout=1.0)
                if message:
                    try:
                        await self.process_message(message)
                        await self.acknowledge(message)
                        self.state.messages_received += 1
                    except Exception as e:
                        logger.error(f"Error processing message {message.id}: {e}")
                        await self.reject(message, requeue=True)
                        self.state.messages_failed += 1
                
            except asyncio.CancelledError:
                logger.info(f"Consuming cancelled for queue: {queue_name}")
                break
            except Exception as e:
                logger.error(f"Error consuming from queue {queue_name}: {e}")
                await asyncio.sleep(1)
    
    async def send_json(
        self, 
        data: Dict[str, Any], 
        queue_name: Optional[str] = None,
        routing_key: Optional[str] = None,
        **kwargs
    ) -> str:
        """Send JSON data as a message."""
        message = Message(
            body=data,
            content_type="application/json",
            queue_name=queue_name,
            routing_key=routing_key,
            **kwargs
        )
        return await self.send(message, queue_name, routing_key)
    
    async def send_text(
        self, 
        text: str, 
        queue_name: Optional[str] = None,
        routing_key: Optional[str] = None,
        **kwargs
    ) -> str:
        """Send text as a message."""
        message = Message(
            body=text,
            content_type="text/plain",
            queue_name=queue_name,
            routing_key=routing_key,
            **kwargs
        )
        return await self.send(message, queue_name, routing_key)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        return {
            "is_connected": self._is_connected,
            "messages_sent": self.state.messages_sent,
            "messages_received": self.state.messages_received,
            "messages_failed": self.state.messages_failed,
            "error_count": self.state.error_count,
            "is_healthy": self.state.is_healthy,
            "active_queues": list(self.state.active_queues.keys()),
            "registered_handlers": list(self.handlers.keys()),
        }


class MessageQueueManager:
    """Manager for multiple message queues."""
    
    def __init__(self, config: MessageQueueConfig):
        self.config = config
        self.queues: Dict[str, MessageQueue] = {}
        self.consumers: Dict[str, asyncio.Task] = {}
        self._is_running = False
    
    def add_queue(self, name: str, queue: MessageQueue) -> None:
        """Add a queue to the manager."""
        self.queues[name] = queue
        logger.info(f"Added queue: {name}")
    
    def get_queue(self, name: str) -> Optional[MessageQueue]:
        """Get a queue by name."""
        return self.queues.get(name)
    
    async def start(self) -> None:
        """Start the message queue manager."""
        if self._is_running:
            logger.warning("Message queue manager is already running")
            return
        
        self._is_running = True
        
        # Connect all queues
        for name, queue in self.queues.items():
            try:
                await queue.connect()
                logger.info(f"Connected queue: {name}")
            except Exception as e:
                logger.error(f"Failed to connect queue {name}: {e}")
        
        # Start consumers for queues with handlers
        for name, queue in self.queues.items():
            if queue.handlers:
                for queue_name in queue.handlers.keys():
                    consumer_key = f"{name}_{queue_name}"
                    self.consumers[consumer_key] = asyncio.create_task(
                        queue.start_consuming(queue_name)
                    )
        
        logger.info("Message queue manager started")
    
    async def stop(self) -> None:
        """Stop the message queue manager."""
        if not self._is_running:
            return
        
        self._is_running = False
        
        # Stop all consumers
        for consumer_key, task in self.consumers.items():
            if not task.done():
                task.cancel()
        
        # Wait for consumers to finish
        if self.consumers:
            await asyncio.gather(*self.consumers.values(), return_exceptions=True)
        
        # Disconnect all queues
        for name, queue in self.queues.items():
            try:
                await queue.disconnect()
                logger.info(f"Disconnected queue: {name}")
            except Exception as e:
                logger.error(f"Error disconnecting queue {name}: {e}")
        
        logger.info("Message queue manager stopped")
    
    async def health_check(self) -> Dict[str, Any]:
        """Check the health of all queues."""
        health_status = {
            "is_running": self._is_running,
            "queues": {},
            "overall_healthy": True
        }
        
        for name, queue in self.queues.items():
            try:
                is_healthy = await queue.health_check()
                health_status["queues"][name] = {
                    "healthy": is_healthy,
                    "stats": queue.get_stats()
                }
                if not is_healthy:
                    health_status["overall_healthy"] = False
            except Exception as e:
                health_status["queues"][name] = {
                    "healthy": False,
                    "error": str(e)
                }
                health_status["overall_healthy"] = False
        
        return health_status
    
    async def send_to_queue(
        self, 
        queue_name: str, 
        message: Message,
        routing_key: Optional[str] = None
    ) -> str:
        """Send a message to a specific queue."""
        queue = self.get_queue(queue_name)
        if not queue:
            raise ValueError(f"Queue not found: {queue_name}")
        
        return await queue.send(message, routing_key=routing_key)
    
    def get_global_stats(self) -> Dict[str, Any]:
        """Get global statistics for all queues."""
        stats = {
            "is_running": self._is_running,
            "queue_count": len(self.queues),
            "consumer_count": len(self.consumers),
            "total_messages_sent": 0,
            "total_messages_received": 0,
            "total_messages_failed": 0,
            "queues": {}
        }
        
        for name, queue in self.queues.items():
            queue_stats = queue.get_stats()
            stats["queues"][name] = queue_stats
            stats["total_messages_sent"] += queue_stats["messages_sent"]
            stats["total_messages_received"] += queue_stats["messages_received"]
            stats["total_messages_failed"] += queue_stats["messages_failed"]
        
        return stats


# Global message queue manager instance
_global_manager: Optional[MessageQueueManager] = None


def get_message_queue_manager() -> Optional[MessageQueueManager]:
    """Get the global message queue manager."""
    return _global_manager


def set_message_queue_manager(manager: MessageQueueManager) -> None:
    """Set the global message queue manager."""
    global _global_manager
    _global_manager = manager


def init_message_queue_manager(config: MessageQueueConfig) -> MessageQueueManager:
    """Initialize the global message queue manager."""
    global _global_manager
    _global_manager = MessageQueueManager(config)
    return _global_manager
