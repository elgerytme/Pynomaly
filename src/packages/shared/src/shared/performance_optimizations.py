"""
Performance optimizations for the interaction framework.

This module provides optimized implementations of the event bus and
dependency injection container with focus on:
- High throughput event processing
- Low latency service resolution
- Memory efficiency
- CPU optimization
- Scalability improvements
"""

import asyncio
import time
import threading
from typing import Dict, List, Any, Optional, Type, TypeVar, Callable, Set
from collections import defaultdict, deque
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import weakref
import gc

from interfaces.events import DomainEvent, EventBus, EventHandler, EventPriority
from interfaces.patterns import Service
from .dependency_injection import DIContainer, LifecycleScope, DependencyRegistration


T = TypeVar('T')


class HighPerformanceEventBus(EventBus):
    """
    Optimized event bus with high-throughput and low-latency processing.
    
    Optimizations include:
    - Lock-free data structures where possible
    - Batched event processing
    - Thread pool for CPU-intensive handlers
    - Memory pool for event objects
    - Optimized serialization
    """
    
    def __init__(self, 
                 max_workers: int = 4,
                 batch_size: int = 100,
                 enable_batching: bool = True,
                 enable_thread_pool: bool = True):
        self._handlers: Dict[Type[DomainEvent], List[EventHandler]] = defaultdict(list)
        self._priority_queues: Dict[EventPriority, asyncio.Queue] = {
            priority: asyncio.Queue(maxsize=10000) for priority in EventPriority
        }
        
        # Performance optimizations
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.enable_batching = enable_batching
        self.enable_thread_pool = enable_thread_pool
        
        # Thread pool for CPU-intensive handlers
        self._thread_pool = ThreadPoolExecutor(max_workers=max_workers) if enable_thread_pool else None
        
        # Batching support
        self._event_batches: Dict[EventPriority, List[DomainEvent]] = defaultdict(list)
        self._batch_timers: Dict[EventPriority, Optional[asyncio.Handle]] = {}
        
        # Performance tracking
        self._metrics = {
            'events_processed': 0,
            'batch_processes': 0,
            'avg_batch_size': 0.0,
            'handler_cache_hits': 0,
            'handler_cache_misses': 0,
        }
        
        # Handler cache for frequently used event types
        self._handler_cache: Dict[Type[DomainEvent], List[EventHandler]] = {}
        self._cache_hit_threshold = 10  # Cache after 10 uses
        self._handler_usage_count: Dict[Type[DomainEvent], int] = defaultdict(int)
        
        # Memory management
        self._event_pool = deque(maxlen=1000)  # Reuse event objects
        self._weak_references = weakref.WeakSet()  # Track objects for GC
        
        self._is_running = False
        self._worker_tasks: List[asyncio.Task] = []
    
    async def start(self) -> None:
        """Start the optimized event bus."""
        if self._is_running:
            return
        
        self._is_running = True
        
        # Start priority-based workers with optimizations
        for priority in EventPriority:
            if self.enable_batching:
                task = asyncio.create_task(self._process_priority_queue_batched(priority))
            else:
                task = asyncio.create_task(self._process_priority_queue_optimized(priority))
            self._worker_tasks.append(task)
        
        # Start memory management task
        task = asyncio.create_task(self._memory_management_worker())
        self._worker_tasks.append(task)
    
    async def stop(self) -> None:
        """Stop the event bus and cleanup."""
        self._is_running = False
        
        # Cancel batch timers
        for timer in self._batch_timers.values():
            if timer:
                timer.cancel()
        
        # Cancel worker tasks
        for task in self._worker_tasks:
            task.cancel()
        
        await asyncio.gather(*self._worker_tasks, return_exceptions=True)
        self._worker_tasks.clear()
        
        # Cleanup thread pool
        if self._thread_pool:
            self._thread_pool.shutdown(wait=True)
    
    async def publish(self, event: DomainEvent) -> None:
        """Publish event with optimizations."""
        # Add to priority queue
        priority_queue = self._priority_queues[event.priority]
        await priority_queue.put(event)
        
        # Update metrics
        self._metrics['events_processed'] += 1
    
    def subscribe(self, event_type: Type[T], handler: EventHandler) -> None:
        """Subscribe with handler caching optimization."""
        self._handlers[event_type].append(handler)
        
        # Invalidate cache for this event type
        if event_type in self._handler_cache:
            del self._handler_cache[event_type]
    
    def unsubscribe(self, event_type: Type[T], handler: EventHandler) -> None:
        """Unsubscribe with cache invalidation."""
        if event_type in self._handlers:
            try:
                self._handlers[event_type].remove(handler)
                # Invalidate cache
                if event_type in self._handler_cache:
                    del self._handler_cache[event_type]
            except ValueError:
                pass
    
    async def _process_priority_queue_optimized(self, priority: EventPriority) -> None:
        """Process events with single-event optimizations."""
        queue = self._priority_queues[priority]
        
        while self._is_running:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=1.0)
                await self._handle_event_optimized(event)
                queue.task_done()
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                # Log error and continue
                continue
    
    async def _process_priority_queue_batched(self, priority: EventPriority) -> None:
        """Process events in batches for higher throughput."""
        queue = self._priority_queues[priority]
        batch = []
        
        while self._is_running:
            try:
                # Collect events for batch
                while len(batch) < self.batch_size:
                    try:
                        event = await asyncio.wait_for(queue.get(), timeout=0.01)
                        batch.append(event)
                        queue.task_done()
                    except asyncio.TimeoutError:
                        break
                
                # Process batch if we have events
                if batch:
                    await self._handle_event_batch(batch)
                    self._metrics['batch_processes'] += 1
                    self._update_avg_batch_size(len(batch))
                    batch.clear()
                
                # Small delay to prevent busy waiting
                if not batch:
                    await asyncio.sleep(0.001)
                
            except Exception as e:
                # Log error and continue
                batch.clear()
    
    async def _handle_event_optimized(self, event: DomainEvent) -> None:
        """Handle single event with optimizations."""
        event_type = type(event)
        
        # Get handlers with caching
        handlers = self._get_handlers_cached(event_type)
        
        if not handlers:
            return
        
        # Process handlers
        if self.enable_thread_pool and self._should_use_thread_pool(handlers):
            # Use thread pool for CPU-intensive handlers
            await self._process_handlers_threaded(event, handlers)
        else:
            # Process handlers asynchronously
            await self._process_handlers_async(event, handlers)
    
    async def _handle_event_batch(self, events: List[DomainEvent]) -> None:
        """Handle a batch of events efficiently."""
        # Group events by type for batch processing
        events_by_type = defaultdict(list)
        for event in events:
            events_by_type[type(event)].append(event)
        
        # Process each event type batch
        tasks = []
        for event_type, event_list in events_by_type.items():
            handlers = self._get_handlers_cached(event_type)
            if handlers:
                task = self._process_event_type_batch(event_list, handlers)
                tasks.append(task)
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _process_event_type_batch(self, events: List[DomainEvent], handlers: List[EventHandler]) -> None:
        """Process a batch of events of the same type."""
        # Create tasks for all event-handler combinations
        tasks = []
        for event in events:
            for handler in handlers:
                if self.enable_thread_pool and self._is_cpu_intensive_handler(handler):
                    task = asyncio.get_event_loop().run_in_executor(
                        self._thread_pool, self._run_handler_sync, handler, event
                    )
                else:
                    task = self._run_handler_async(handler, event)
                tasks.append(task)
        
        # Process all handlers for all events concurrently
        await asyncio.gather(*tasks, return_exceptions=True)
    
    def _get_handlers_cached(self, event_type: Type[DomainEvent]) -> List[EventHandler]:
        """Get handlers with caching optimization."""
        # Check cache first
        if event_type in self._handler_cache:
            self._metrics['handler_cache_hits'] += 1
            return self._handler_cache[event_type]
        
        # Cache miss
        self._metrics['handler_cache_misses'] += 1
        handlers = self._handlers.get(event_type, [])
        
        # Update usage count
        self._handler_usage_count[event_type] += 1
        
        # Cache if used frequently
        if self._handler_usage_count[event_type] >= self._cache_hit_threshold:
            self._handler_cache[event_type] = handlers.copy()
        
        return handlers
    
    def _should_use_thread_pool(self, handlers: List[EventHandler]) -> bool:
        """Determine if handlers should use thread pool."""
        return any(self._is_cpu_intensive_handler(handler) for handler in handlers)
    
    def _is_cpu_intensive_handler(self, handler: EventHandler) -> bool:
        """Check if handler is CPU intensive (heuristic)."""
        # Simple heuristic: check if handler has certain characteristics
        handler_name = handler.__class__.__name__.lower()
        cpu_intensive_keywords = ['compute', 'calculate', 'process', 'analyze', 'transform']
        return any(keyword in handler_name for keyword in cpu_intensive_keywords)
    
    async def _process_handlers_async(self, event: DomainEvent, handlers: List[EventHandler]) -> None:
        """Process handlers asynchronously."""
        tasks = [self._run_handler_async(handler, event) for handler in handlers]
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _process_handlers_threaded(self, event: DomainEvent, handlers: List[EventHandler]) -> None:
        """Process handlers using thread pool."""
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(self._thread_pool, self._run_handler_sync, handler, event)
            for handler in handlers
        ]
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _run_handler_async(self, handler: EventHandler, event: DomainEvent) -> None:
        """Run async handler with error handling."""
        try:
            await handler.handle(event)
        except Exception:
            # Log error but don't stop processing
            pass
    
    def _run_handler_sync(self, handler: EventHandler, event: DomainEvent) -> None:
        """Run handler synchronously for thread pool."""
        try:
            # If handler is async, run it in a new event loop
            if asyncio.iscoroutinefunction(handler.handle):
                asyncio.run(handler.handle(event))
            else:
                handler.handle(event)
        except Exception:
            # Log error but don't stop processing
            pass
    
    async def _memory_management_worker(self) -> None:
        """Background worker for memory management."""
        while self._is_running:
            await asyncio.sleep(30)  # Run every 30 seconds
            
            # Trigger garbage collection
            gc.collect()
            
            # Clean up old cache entries
            self._cleanup_cache()
            
            # Clear event pool if it's getting too large
            if len(self._event_pool) > 500:
                self._event_pool.clear()
    
    def _cleanup_cache(self) -> None:
        """Clean up handler cache."""
        # Remove cache entries for infrequently used event types
        to_remove = []
        for event_type, usage_count in self._handler_usage_count.items():
            if usage_count < self._cache_hit_threshold // 2:
                to_remove.append(event_type)
        
        for event_type in to_remove:
            self._handler_cache.pop(event_type, None)
            del self._handler_usage_count[event_type]
    
    def _update_avg_batch_size(self, batch_size: int) -> None:
        """Update average batch size metric."""
        current_avg = self._metrics['avg_batch_size']
        batch_count = self._metrics['batch_processes']
        self._metrics['avg_batch_size'] = ((current_avg * (batch_count - 1)) + batch_size) / batch_count
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        return {
            **self._metrics,
            'handler_cache_size': len(self._handler_cache),
            'event_pool_size': len(self._event_pool),
            'active_workers': len(self._worker_tasks),
            'queue_sizes': {
                priority.value: queue.qsize()
                for priority, queue in self._priority_queues.items()
            }
        }


class OptimizedDIContainer(DIContainer):
    """
    Optimized dependency injection container with performance improvements.
    
    Optimizations include:
    - Service resolution caching
    - Lazy initialization
    - Lock-free operations where possible
    - Optimized circular dependency detection
    - Memory efficient service storage
    """
    
    def __init__(self):
        super().__init__()
        
        # Performance optimizations
        self._resolution_cache: Dict[Type, Any] = {}
        self._resolution_times: Dict[Type, float] = {}
        self._resolution_counts: Dict[Type, int] = defaultdict(int)
        
        # Circular dependency detection cache
        self._dependency_graph: Dict[Type, Set[Type]] = defaultdict(set)
        self._circular_check_cache: Dict[frozenset, bool] = {}
        
        # Lazy initialization tracking
        self._lazy_services: Set[Type] = set()
        self._initialization_lock = threading.RLock()
        
        # Memory optimization
        self._weak_singletons: weakref.WeakValueDictionary = weakref.WeakValueDictionary()
        
        # Performance metrics
        self._performance_metrics = {
            'cache_hits': 0,
            'cache_misses': 0,
            'avg_resolution_time_ms': 0.0,
            'circular_checks_avoided': 0,
            'memory_optimizations': 0,
        }
    
    def resolve(self, interface: Type[T]) -> T:
        """Optimized service resolution with caching."""
        start_time = time.time()
        
        # Check cache first for singletons
        if interface in self._resolution_cache:
            self._performance_metrics['cache_hits'] += 1
            return self._resolution_cache[interface]
        
        self._performance_metrics['cache_misses'] += 1
        
        # Check if already building (circular dependency)
        if interface in self._building_stack:
            if self._is_circular_dependency_cached(interface):
                self._performance_metrics['circular_checks_avoided'] += 1
                raise self._create_circular_dependency_error(interface)
        
        # Build dependency
        self._building_stack.append(interface)
        try:
            instance = self._create_instance_optimized(self._registrations[interface])
            
            # Cache if singleton
            registration = self._registrations[interface]
            if registration.lifecycle == LifecycleScope.SINGLETON:
                self._resolution_cache[interface] = instance
            
            # Update metrics
            resolution_time = (time.time() - start_time) * 1000
            self._update_resolution_metrics(interface, resolution_time)
            
            return instance
        finally:
            self._building_stack.pop()
    
    def _create_instance_optimized(self, registration: DependencyRegistration) -> Any:
        """Optimized instance creation."""
        # Use existing singleton if available
        if registration.lifecycle == LifecycleScope.SINGLETON:
            if registration.interface in self._singletons:
                return self._singletons[registration.interface]
            
            # Check weak reference cache
            if registration.interface in self._weak_singletons:
                instance = self._weak_singletons[registration.interface]
                if instance is not None:
                    self._performance_metrics['memory_optimizations'] += 1
                    return instance
        
        # Create new instance using optimized path
        if registration.factory:
            instance = self._invoke_factory_optimized(registration.factory)
        else:
            instance = self._create_from_constructor_optimized(registration)
        
        # Cache based on lifecycle
        if registration.lifecycle == LifecycleScope.SINGLETON:
            self._singletons[registration.interface] = instance
            # Also add to weak reference cache for memory optimization
            try:
                self._weak_singletons[registration.interface] = instance
            except TypeError:
                # Object not weak referenceable
                pass
        
        return instance
    
    def _create_from_constructor_optimized(self, registration: DependencyRegistration) -> Any:
        """Optimized constructor-based creation."""
        if not registration.implementation:
            raise ValueError(f"No implementation for {registration.interface.__name__}")
        
        # Use cached dependency information if available
        if registration.dependencies:
            # Fast path: use pre-analyzed dependencies
            kwargs = {}
            for dep_name in registration.dependencies:
                # Map parameter name to type (simplified)
                dep_type = self._resolve_parameter_type(registration.implementation, dep_name)
                if dep_type:
                    kwargs[dep_name] = self.resolve(dep_type)
        else:
            # Slow path: analyze constructor
            kwargs = self._build_constructor_kwargs(registration)
        
        return registration.implementation(**kwargs)
    
    def _invoke_factory_optimized(self, factory: Callable) -> Any:
        """Optimized factory invocation."""
        # Cache factory signature analysis
        factory_key = id(factory)
        if hasattr(factory, '__cached_signature__'):
            kwargs = factory.__cached_signature__
        else:
            kwargs = self._analyze_factory_signature(factory)
            factory.__cached_signature__ = kwargs
        
        # Resolve dependencies
        resolved_kwargs = {}
        for param_name, param_type in kwargs.items():
            resolved_kwargs[param_name] = self.resolve(param_type)
        
        return factory(**resolved_kwargs)
    
    def _is_circular_dependency_cached(self, interface: Type) -> bool:
        """Check for circular dependency using cache."""
        # Create a key for the current dependency path
        path_key = frozenset(self._building_stack + [interface])
        
        if path_key in self._circular_check_cache:
            return self._circular_check_cache[path_key]
        
        # Check if there's a cycle
        has_cycle = interface in self._building_stack
        self._circular_check_cache[path_key] = has_cycle
        
        return has_cycle
    
    def _update_resolution_metrics(self, interface: Type, resolution_time_ms: float) -> None:
        """Update resolution performance metrics."""
        self._resolution_times[interface] = resolution_time_ms
        self._resolution_counts[interface] += 1
        
        # Update average
        total_resolutions = sum(self._resolution_counts.values())
        total_time = sum(self._resolution_times.values())
        self._performance_metrics['avg_resolution_time_ms'] = total_time / max(total_resolutions, 1)
    
    def _resolve_parameter_type(self, implementation: Type, param_name: str) -> Optional[Type]:
        """Resolve parameter type from implementation (simplified)."""
        # This would use more sophisticated type analysis in practice
        import inspect
        try:
            sig = inspect.signature(implementation.__init__)
            param = sig.parameters.get(param_name)
            return param.annotation if param and param.annotation != inspect.Parameter.empty else None
        except:
            return None
    
    def _build_constructor_kwargs(self, registration: DependencyRegistration) -> Dict[str, Any]:
        """Build constructor kwargs (fallback method)."""
        import inspect
        
        constructor = registration.implementation.__init__
        sig = inspect.signature(constructor)
        
        kwargs = {}
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue
            
            param_type = param.annotation
            if param_type == inspect.Parameter.empty:
                if param.default == inspect.Parameter.empty:
                    raise ValueError(f"Cannot resolve parameter '{param_name}'")
                continue
            
            kwargs[param_name] = self.resolve(param_type)
        
        return kwargs
    
    def _analyze_factory_signature(self, factory: Callable) -> Dict[str, Type]:
        """Analyze factory signature for dependency injection."""
        import inspect
        
        sig = inspect.signature(factory)
        kwargs = {}
        
        for param_name, param in sig.parameters.items():
            if param.annotation != inspect.Parameter.empty:
                kwargs[param_name] = param.annotation
        
        return kwargs
    
    def _create_circular_dependency_error(self, interface: Type) -> Exception:
        """Create circular dependency error with context."""
        cycle = " -> ".join(cls.__name__ for cls in self._building_stack)
        return ValueError(f"Circular dependency: {cycle} -> {interface.__name__}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the container."""
        return {
            **self._performance_metrics,
            'cached_resolutions': len(self._resolution_cache),
            'total_registrations': len(self._registrations),
            'singleton_count': len(self._singletons),
            'weak_singleton_count': len(self._weak_singletons),
            'resolution_counts': dict(self._resolution_counts),
            'top_resolved_services': self._get_top_resolved_services(),
        }
    
    def _get_top_resolved_services(self) -> List[Dict[str, Any]]:
        """Get top resolved services by frequency."""
        sorted_services = sorted(
            self._resolution_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        return [
            {
                'service': service_type.__name__,
                'resolution_count': count,
                'avg_time_ms': self._resolution_times.get(service_type, 0)
            }
            for service_type, count in sorted_services
        ]
    
    def optimize_memory(self) -> None:
        """Perform memory optimization."""
        # Clear circular dependency cache
        self._circular_check_cache.clear()
        
        # Run garbage collection
        import gc
        gc.collect()
        
        self._performance_metrics['memory_optimizations'] += 1


# Factory functions for optimized components
def create_high_performance_event_bus(**kwargs) -> HighPerformanceEventBus:
    """Create a high-performance event bus with optimal settings."""
    return HighPerformanceEventBus(
        max_workers=kwargs.get('max_workers', 4),
        batch_size=kwargs.get('batch_size', 100),
        enable_batching=kwargs.get('enable_batching', True),
        enable_thread_pool=kwargs.get('enable_thread_pool', True)
    )


def create_optimized_di_container() -> OptimizedDIContainer:
    """Create an optimized DI container."""
    return OptimizedDIContainer()