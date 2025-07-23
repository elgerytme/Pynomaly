"""Messaging infrastructure for cross-service communication.

This module provides event buses, message queues, and async communication
patterns. It supports multiple message brokers and implements reliable
delivery patterns.

Example usage:
    from infrastructure.messaging import EventBus, MessageQueue
    
    event_bus = EventBus()
    await event_bus.publish(UserCreatedEvent(user_id="123"))
    
    queue = MessageQueue("processing")
    await queue.send_message({"task": "process_data"})
"""

from .event_bus import EventBus, Event
from .message_queue import MessageQueue
from .publisher import Publisher
from .subscriber import Subscriber

__all__ = [
    "EventBus",
    "Event", 
    "MessageQueue",
    "Publisher",
    "Subscriber"
]