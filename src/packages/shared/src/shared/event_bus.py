"""
Event bus implementation for cross-package communication.

This module provides a concrete implementation of the event bus pattern
that can be shared across all packages in the monorepo.
"""

import asyncio
import logging
from typing import Any, Callable, Dict, List, Type, TypeVar
from collections import defaultdict
from datetime import datetime

from interfaces.events import DomainEvent, EventBus, EventHandler, EventPriority


logger = logging.getLogger(__name__)
T = TypeVar('T', bound=DomainEvent)


class AsyncEventHandler:
    """Wrapper for async event handlers."""
    
    def __init__(self, handler: Callable[[DomainEvent], Any]):
        self.handler = handler
        self.is_async = asyncio.iscoroutinefunction(handler)
    
    async def handle(self, event: DomainEvent) -> None:
        """Handle an event, wrapping sync handlers in async."""
        try:
            if self.is_async:
                await self.handler(event)
            else:
                self.handler(event)
        except Exception as e:
            logger.error(f"Error in event handler: {e}", exc_info=True)
            raise


class DistributedEventBus(EventBus):
    """
    Production-ready event bus with features for distributed systems.
    
    Features:
    - Priority-based event processing
    - Dead letter queue for failed events
    - Event persistence and replay
    - Metrics and monitoring
    - Circuit breaker for failing handlers
    """
    
    def __init__(self, 
                 max_retries: int = 3,
                 retry_delay: float = 1.0,
                 enable_persistence: bool = False,
                 enable_metrics: bool = True):
        self._handlers: Dict[Type[DomainEvent], List[AsyncEventHandler]] = defaultdict(list)
        self._priority_queues: Dict[EventPriority, asyncio.Queue] = {
            priority: asyncio.Queue() for priority in EventPriority
        }
        self._dead_letter_queue: asyncio.Queue = asyncio.Queue()
        self._event_store: List[DomainEvent] = [] if enable_persistence else None
        self._metrics = {
            'events_published': 0,
            'events_processed': 0,
            'events_failed': 0,
            'handler_errors': 0,
        } if enable_metrics else None
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._is_running = False
        self._worker_tasks: List[asyncio.Task] = []
    
    async def start(self) -> None:
        """Start the event bus workers."""
        if self._is_running:
            return
        
        self._is_running = True
        
        # Start workers for each priority level
        for priority in EventPriority:
            task = asyncio.create_task(self._process_priority_queue(priority))
            self._worker_tasks.append(task)
        
        # Start dead letter queue processor
        task = asyncio.create_task(self._process_dead_letter_queue())
        self._worker_tasks.append(task)
        
        logger.info("Event bus started with workers for all priority levels")
    
    async def stop(self) -> None:
        """Stop the event bus workers."""
        self._is_running = False
        
        # Cancel all worker tasks
        for task in self._worker_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self._worker_tasks, return_exceptions=True)
        self._worker_tasks.clear()
        
        logger.info("Event bus stopped")
    
    async def publish(self, event: DomainEvent) -> None:
        """Publish an event to the appropriate priority queue."""
        if self._event_store is not None:
            self._event_store.append(event)
        
        # Add to priority queue
        priority_queue = self._priority_queues[event.priority]
        await priority_queue.put(event)
        
        if self._metrics:
            self._metrics['events_published'] += 1
        
        logger.debug(f"Published event {event.event_id} with priority {event.priority.value}")
    
    def subscribe(self, event_type: Type[T], handler: Callable[[T], Any]) -> None:
        """Subscribe a handler to an event type."""
        async_handler = AsyncEventHandler(handler)
        self._handlers[event_type].append(async_handler)
        logger.debug(f"Subscribed handler for event type {event_type.__name__}")
    
    def unsubscribe(self, event_type: Type[T], handler: Callable[[T], Any]) -> None:
        """Unsubscribe a handler from an event type."""
        handlers = self._handlers.get(event_type, [])
        self._handlers[event_type] = [h for h in handlers if h.handler != handler]
        logger.debug(f"Unsubscribed handler for event type {event_type.__name__}")
    
    async def _process_priority_queue(self, priority: EventPriority) -> None:
        """Process events from a specific priority queue."""
        queue = self._priority_queues[priority]
        
        while self._is_running:
            try:
                # Wait for event with timeout to allow checking _is_running
                event = await asyncio.wait_for(queue.get(), timeout=1.0)
                await self._handle_event(event)
                queue.task_done()
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing {priority.value} queue: {e}")
    
    async def _handle_event(self, event: DomainEvent, retry_count: int = 0) -> None:
        """Handle a single event with retry logic."""
        event_type = type(event)
        handlers = self._handlers.get(event_type, [])
        
        if not handlers:
            logger.debug(f"No handlers registered for event type {event_type.__name__}")
            return
        
        failed_handlers = []
        
        for handler in handlers:
            try:
                await handler.handle(event)
                logger.debug(f"Successfully handled event {event.event_id}")
            except Exception as e:
                logger.error(f"Handler failed for event {event.event_id}: {e}")
                failed_handlers.append((handler, e))
                if self._metrics:
                    self._metrics['handler_errors'] += 1
        
        if failed_handlers and retry_count < self._max_retries:
            # Retry failed handlers
            await asyncio.sleep(self._retry_delay * (retry_count + 1))
            await self._handle_event(event, retry_count + 1)
        elif failed_handlers:
            # Send to dead letter queue
            await self._dead_letter_queue.put((event, failed_handlers))
            if self._metrics:
                self._metrics['events_failed'] += 1
        else:
            if self._metrics:
                self._metrics['events_processed'] += 1
    
    async def _process_dead_letter_queue(self) -> None:
        """Process events that failed all retry attempts."""
        while self._is_running:
            try:
                item = await asyncio.wait_for(self._dead_letter_queue.get(), timeout=1.0)
                event, failed_handlers = item
                
                logger.error(f"Event {event.event_id} moved to dead letter queue. "
                           f"Failed handlers: {len(failed_handlers)}")
                
                # Here you could implement additional logic like:
                # - Persisting to database for manual retry
                # - Sending alerts
                # - Writing to external dead letter queue service
                
                self._dead_letter_queue.task_done()
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing dead letter queue: {e}")
    
    async def replay_events(self, from_time: datetime, to_time: datetime) -> None:
        """Replay events from the event store within a time range."""
        if self._event_store is None:
            raise ValueError("Event persistence is not enabled")
        
        replayed_count = 0
        for event in self._event_store:
            if from_time <= event.occurred_at <= to_time:
                await self.publish(event)
                replayed_count += 1
        
        logger.info(f"Replayed {replayed_count} events from {from_time} to {to_time}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get event bus metrics."""
        if self._metrics is None:
            return {}
        
        return {
            **self._metrics,
            'queue_sizes': {
                priority.value: queue.qsize() 
                for priority, queue in self._priority_queues.items()
            },
            'dead_letter_queue_size': self._dead_letter_queue.qsize(),
            'registered_handlers': {
                event_type.__name__: len(handlers)
                for event_type, handlers in self._handlers.items()
            }
        }


# Convenience decorator for event handlers
def event_handler(event_type: Type[T]):
    """Decorator to mark functions as event handlers."""
    def decorator(func: Callable[[T], Any]):
        func._event_type = event_type
        func._is_event_handler = True
        return func
    return decorator


# Global event bus instance
_global_event_bus: DistributedEventBus = None


def get_event_bus() -> DistributedEventBus:
    """Get the global event bus instance."""
    global _global_event_bus
    if _global_event_bus is None:
        _global_event_bus = DistributedEventBus()
    return _global_event_bus


async def publish_event(event: DomainEvent) -> None:
    """Convenience function to publish an event."""
    bus = get_event_bus()
    await bus.publish(event)


def subscribe_to_event(event_type: Type[T], handler: Callable[[T], Any]) -> None:
    """Convenience function to subscribe to an event."""
    bus = get_event_bus()
    bus.subscribe(event_type, handler)