"""Adapter for distributed tracing operations.

This adapter implements the DistributedTracingPort interface and provides
integration with external tracing systems while isolating the domain
from infrastructure concerns.
"""

import logging
import time
import uuid
from datetime import datetime
from typing import Any, Callable, Dict, Optional
from functools import wraps

from machine_learning.domain.interfaces.monitoring_operations import (
    DistributedTracingPort,
    TraceSpan,
    TracingError,
)

logger = logging.getLogger(__name__)


class DistributedTracingAdapter(DistributedTracingPort):
    """Adapter for distributed tracing operations.
    
    This adapter provides concrete implementation of distributed tracing
    while abstracting away the specific tracing system being used.
    """
    
    def __init__(self, tracing_backend: str = "local"):
        """Initialize the distributed tracing adapter.
        
        Args:
            tracing_backend: Backend to use for tracing ("local", "jaeger", "zipkin")
        """
        self._backend = tracing_backend
        self._logger = logging.getLogger(__name__)
        self._active_spans: Dict[str, TraceSpan] = {}
        
        # Initialize backend-specific components
        self._init_backend()
    
    def _init_backend(self):
        """Initialize the tracing backend."""
        if self._backend == "jaeger":
            try:
                import jaeger_client
                self._jaeger_tracer = self._setup_jaeger()
                self._logger.info("Initialized Jaeger tracing backend")
            except ImportError:
                self._logger.warning("Jaeger client not available, falling back to local tracing")
                self._backend = "local"
        
        elif self._backend == "zipkin":
            try:
                import py_zipkin
                self._zipkin_tracer = self._setup_zipkin()
                self._logger.info("Initialized Zipkin tracing backend")
            except ImportError:
                self._logger.warning("Zipkin client not available, falling back to local tracing")
                self._backend = "local"
        
        if self._backend == "local":
            self._logger.info("Using local in-memory tracing")
    
    def _setup_jaeger(self):
        """Setup Jaeger tracer."""
        # This would be implemented with actual Jaeger configuration
        # For now, we'll use a mock implementation
        return None
    
    def _setup_zipkin(self):
        """Setup Zipkin tracer."""
        # This would be implemented with actual Zipkin configuration
        # For now, we'll use a mock implementation
        return None
    
    async def start_trace(
        self,
        operation_name: str,
        parent_span_id: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None
    ) -> TraceSpan:
        """Start a new trace span."""
        try:
            span_id = str(uuid.uuid4())
            trace_id = str(uuid.uuid4()) if parent_span_id is None else self._get_trace_id(parent_span_id)
            
            span = TraceSpan(
                operation_name=operation_name,
                span_id=span_id,
                trace_id=trace_id,
                parent_span_id=parent_span_id,
                start_time=datetime.now(),
                end_time=None,
                duration_ms=None,
                tags=tags or {},
                logs=[],
                status="active"
            )
            
            # Store active span
            self._active_spans[span_id] = span
            
            # Backend-specific span creation
            if self._backend == "jaeger" and hasattr(self, '_jaeger_tracer'):
                # Create Jaeger span
                pass
            elif self._backend == "zipkin" and hasattr(self, '_zipkin_tracer'):
                # Create Zipkin span
                pass
            
            self._logger.debug(f"Started trace span: {operation_name} (span_id: {span_id})")
            return span
            
        except Exception as e:
            raise TracingError(f"Failed to start trace span: {e}")
    
    async def finish_trace(
        self,
        span: TraceSpan,
        status: str = "ok",
        error_message: Optional[str] = None
    ) -> None:
        """Finish a trace span."""
        try:
            span.end_time = datetime.now()
            span.status = status
            span.error_message = error_message
            
            if span.start_time and span.end_time:
                duration = span.end_time - span.start_time
                span.duration_ms = duration.total_seconds() * 1000
            
            # Backend-specific span finishing
            if self._backend == "jaeger" and hasattr(self, '_jaeger_tracer'):
                # Finish Jaeger span
                pass
            elif self._backend == "zipkin" and hasattr(self, '_zipkin_tracer'):
                # Finish Zipkin span
                pass
            
            # Remove from active spans
            if span.span_id in self._active_spans:
                del self._active_spans[span.span_id]
            
            self._logger.debug(
                f"Finished trace span: {span.operation_name} "
                f"(duration: {span.duration_ms:.2f}ms, status: {status})"
            )
            
        except Exception as e:
            raise TracingError(f"Failed to finish trace span: {e}")
    
    async def add_trace_tag(
        self,
        span: TraceSpan,
        key: str,
        value: Any
    ) -> None:
        """Add a tag to a trace span."""
        try:
            span.tags[key] = value
            
            # Backend-specific tag addition
            if self._backend == "jaeger" and hasattr(self, '_jaeger_tracer'):
                # Add Jaeger tag
                pass
            elif self._backend == "zipkin" and hasattr(self, '_zipkin_tracer'):
                # Add Zipkin tag
                pass
            
            self._logger.debug(f"Added tag to span {span.span_id}: {key}={value}")
            
        except Exception as e:
            raise TracingError(f"Failed to add trace tag: {e}")
    
    async def log_trace_event(
        self,
        span: TraceSpan,
        event_name: str,
        data: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log an event within a trace span."""
        try:
            event = {
                "event_name": event_name,
                "timestamp": datetime.now().isoformat(),
                "data": data or {}
            }
            
            span.logs.append(event)
            
            # Backend-specific event logging
            if self._backend == "jaeger" and hasattr(self, '_jaeger_tracer'):
                # Log Jaeger event
                pass
            elif self._backend == "zipkin" and hasattr(self, '_zipkin_tracer'):
                # Log Zipkin event
                pass
            
            self._logger.debug(f"Logged event in span {span.span_id}: {event_name}")
            
        except Exception as e:
            raise TracingError(f"Failed to log trace event: {e}")
    
    def trace_operation(
        self,
        operation_name: str,
        tags: Optional[Dict[str, Any]] = None
    ) -> Callable:
        """Decorator for tracing operations."""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                span = await self.start_trace(operation_name, tags=tags)
                try:
                    # Add function metadata as tags
                    await self.add_trace_tag(span, "function_name", func.__name__)
                    await self.add_trace_tag(span, "module", func.__module__)
                    
                    # Execute the function
                    result = await func(*args, **kwargs)
                    
                    # Record success
                    await self.add_trace_tag(span, "success", True)
                    await self.finish_trace(span, "ok")
                    
                    return result
                    
                except Exception as e:
                    # Record error
                    await self.add_trace_tag(span, "success", False)
                    await self.add_trace_tag(span, "error_type", type(e).__name__)
                    await self.finish_trace(span, "error", str(e))
                    raise
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                # For synchronous functions, create a simple wrapper
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    execution_time = (time.time() - start_time) * 1000
                    self._logger.debug(
                        f"Traced operation: {operation_name} "
                        f"(duration: {execution_time:.2f}ms, status: ok)"
                    )
                    return result
                except Exception as e:
                    execution_time = (time.time() - start_time) * 1000
                    self._logger.debug(
                        f"Traced operation: {operation_name} "
                        f"(duration: {execution_time:.2f}ms, status: error, error: {e})"
                    )
                    raise
            
            # Return appropriate wrapper based on function type
            import asyncio
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
        
        return decorator
    
    def _get_trace_id(self, span_id: str) -> str:
        """Get trace ID for a given span ID."""
        if span_id in self._active_spans:
            return self._active_spans[span_id].trace_id
        # If span not found, create new trace ID
        return str(uuid.uuid4())
    
    def get_active_spans(self) -> Dict[str, TraceSpan]:
        """Get all currently active spans."""
        return self._active_spans.copy()
    
    def get_span_by_id(self, span_id: str) -> Optional[TraceSpan]:
        """Get span by ID."""
        return self._active_spans.get(span_id)


# Utility function for backward compatibility
def trace_operation(operation_name: str, tags: Optional[Dict[str, Any]] = None):
    """Backward-compatible trace operation decorator.
    
    This function provides the same interface as the original distributed_tracing
    module while using the new adapter pattern internally.
    """
    # Create a global adapter instance for backward compatibility
    global _global_tracing_adapter
    if '_global_tracing_adapter' not in globals():
        _global_tracing_adapter = DistributedTracingAdapter()
    
    return _global_tracing_adapter.trace_operation(operation_name, tags)