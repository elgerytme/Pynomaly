"""
Message Processing Middleware

This module provides middleware classes for message processing.
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Optional

from .core import Message

logger = logging.getLogger(__name__)


class MessageMiddleware(ABC):
    """Abstract base class for message middleware."""
    
    @abstractmethod
    async def before_process(self, message: Message) -> None:
        """Called before message processing."""
        pass
    
    @abstractmethod
    async def after_process(self, message: Message, result: Any) -> None:
        """Called after successful message processing."""
        pass
    
    @abstractmethod
    async def on_error(self, message: Message, error: Exception) -> None:
        """Called when message processing fails."""
        pass


class LoggingMiddleware(MessageMiddleware):
    """Middleware that logs message processing events."""
    
    def __init__(self, logger_name: str = __name__):
        self.logger = logging.getLogger(logger_name)
    
    async def before_process(self, message: Message) -> None:
        """Log before message processing."""
        self.logger.info(f"Processing message {message.id} from queue {message.queue_name}")
    
    async def after_process(self, message: Message, result: Any) -> None:
        """Log after successful message processing."""
        self.logger.info(f"Successfully processed message {message.id}")
    
    async def on_error(self, message: Message, error: Exception) -> None:
        """Log message processing errors."""
        self.logger.error(f"Error processing message {message.id}: {error}")


class MetricsMiddleware(MessageMiddleware):
    """Middleware that collects metrics."""
    
    def __init__(self):
        self.processed_count = 0
        self.error_count = 0
        self.processing_times = []
        self.start_time = None
    
    async def before_process(self, message: Message) -> None:
        """Start timing message processing."""
        self.start_time = time.time()
    
    async def after_process(self, message: Message, result: Any) -> None:
        """Record successful processing metrics."""
        if self.start_time:
            processing_time = time.time() - self.start_time
            self.processing_times.append(processing_time)
            self.start_time = None
        
        self.processed_count += 1
    
    async def on_error(self, message: Message, error: Exception) -> None:
        """Record error metrics."""
        self.error_count += 1
        
        if self.start_time:
            processing_time = time.time() - self.start_time
            self.processing_times.append(processing_time)
            self.start_time = None
    
    def get_metrics(self) -> dict:
        """Get collected metrics."""
        return {
            "processed_count": self.processed_count,
            "error_count": self.error_count,
            "average_processing_time": sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0,
            "total_processing_time": sum(self.processing_times)
        }
