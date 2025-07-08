"""
Message Handler Decorators

This module provides decorators for message handlers.
"""

from typing import Callable, Any

from .core import Message


def message_handler(queue_name: str):
    """Decorator to mark a function as a message handler for a specific queue."""
    def decorator(func: Callable[[Message], Any]) -> Callable[[Message], Any]:
        func._queue_name = queue_name
        func._is_message_handler = True
        return func
    return decorator


def async_message_handler(queue_name: str):
    """Decorator to mark an async function as a message handler for a specific queue."""
    def decorator(func: Callable[[Message], Any]) -> Callable[[Message], Any]:
        func._queue_name = queue_name
        func._is_async_message_handler = True
        return func
    return decorator
