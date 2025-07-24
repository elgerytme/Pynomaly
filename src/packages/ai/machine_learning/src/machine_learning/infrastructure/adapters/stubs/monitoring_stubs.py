"""Stub implementations for monitoring operations.

These stubs implement the monitoring interfaces but provide basic
functionality when external monitoring systems are not available.
"""

import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Callable, Union
from datetime import datetime
from functools import wraps

from machine_learning.domain.interfaces.monitoring_operations import (
    MonitoringPort,
    DistributedTracingPort,
    AlertingPort,
    HealthCheckPort,
    MetricValue,
    TraceSpan,
    PerformanceMetrics,
    MetricType,
    TraceLevel,
)

logger = logging.getLogger(__name__)


class MonitoringStub(MonitoringPort):
    """Stub implementation for monitoring operations.
    
    This stub provides basic functionality when external monitoring
    systems (Prometheus, etc.) are not available.
    """
    
    def __init__(self):
        """Initialize the monitoring stub."""
        self._logger = logging.getLogger(__name__)
        self._logger.warning(
            "Using monitoring stub. External monitoring systems not available. "
            "Install Prometheus client or similar libraries for full functionality."
        )
        
        # In-memory storage for stub functionality
        self._metrics: Dict[str, MetricValue] = {}
        self._performance_history: List[PerformanceMetrics] = []
    
    async def record_metric(self, metric: MetricValue) -> None:
        """Stub implementation of metric recording."""
        metric_key = f"{metric.name}_{hash(str(sorted(metric.labels.items())))}"
        self._metrics[metric_key] = metric
        
        self._logger.debug(
            f"Stub: Recorded metric {metric.name}={metric.value} "
            f"(type: {metric.metric_type.value}, labels: {metric.labels})"
        )
    
    async def record_performance_metrics(self, metrics: PerformanceMetrics) -> None:
        """Stub implementation of performance metrics recording."""
        self._performance_history.append(metrics)
        
        # Keep only last 1000 entries
        if len(self._performance_history) > 1000:
            self._performance_history = self._performance_history[-1000:]
        
        self._logger.debug(
            f"Stub: Recorded performance metrics for {metrics.operation_name} "
            f"(time: {metrics.execution_time_ms:.2f}ms, memory: {metrics.memory_usage_mb:.1f}MB)"
        )
    
    async def increment_counter(
        self,
        name: str,
        value: int = 1,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Stub implementation of counter increment."""
        labels = labels or {}
        metric_key = f"{name}_{hash(str(sorted(labels.items())))}"
        
        if metric_key in self._metrics:
            current_value = self._metrics[metric_key].value
            new_value = current_value + value
        else:
            new_value = value
        
        metric = MetricValue(
            name=name,
            value=new_value,
            metric_type=MetricType.COUNTER,
            labels=labels,
            timestamp=datetime.now()
        )
        
        await self.record_metric(metric)
    
    async def set_gauge(
        self,
        name: str,
        value: Union[int, float],
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Stub implementation of gauge setting."""
        labels = labels or {}
        
        metric = MetricValue(
            name=name,
            value=value,
            metric_type=MetricType.GAUGE,
            labels=labels,
            timestamp=datetime.now()
        )
        
        await self.record_metric(metric)
    
    async def record_histogram(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Stub implementation of histogram recording."""
        labels = labels or {}
        
        metric = MetricValue(
            name=name,
            value=value,
            metric_type=MetricType.HISTOGRAM,
            labels=labels,
            timestamp=datetime.now()
        )
        
        await self.record_metric(metric)
    
    async def get_metric_value(
        self,
        name: str,
        labels: Optional[Dict[str, str]] = None
    ) -> Optional[MetricValue]:
        """Stub implementation of metric value retrieval."""
        labels = labels or {}
        metric_key = f"{name}_{hash(str(sorted(labels.items())))}"
        
        return self._metrics.get(metric_key)


class DistributedTracingStub(DistributedTracingPort):
    """Stub implementation for distributed tracing operations.
    
    This stub provides basic functionality when external tracing
    systems (Jaeger, Zipkin, etc.) are not available.
    """
    
    def __init__(self):
        """Initialize the distributed tracing stub."""
        self._logger = logging.getLogger(__name__)
        self._logger.warning(
            "Using distributed tracing stub. External tracing systems not available. "
            "Install Jaeger, Zipkin, or OpenTelemetry for full functionality."
        )
        
        # In-memory storage for stub functionality
        self._active_spans: Dict[str, TraceSpan] = {}
        self._completed_spans: List[TraceSpan] = []
    
    async def start_trace(
        self,
        operation_name: str,
        parent_span_id: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None
    ) -> TraceSpan:
        """Stub implementation of trace starting."""
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
        
        self._active_spans[span_id] = span
        
        self._logger.debug(f"Stub: Started trace span '{operation_name}' (id: {span_id})")
        return span
    
    async def finish_trace(
        self,
        span: TraceSpan,
        status: str = "ok",
        error_message: Optional[str] = None
    ) -> None:
        """Stub implementation of trace finishing."""
        span.end_time = datetime.now()
        span.status = status
        span.error_message = error_message
        
        if span.start_time and span.end_time:
            duration = span.end_time - span.start_time
            span.duration_ms = duration.total_seconds() * 1000
        
        # Move from active to completed
        if span.span_id in self._active_spans:
            del self._active_spans[span.span_id]
        
        self._completed_spans.append(span)
        
        # Keep only last 1000 completed spans
        if len(self._completed_spans) > 1000:
            self._completed_spans = self._completed_spans[-1000:]
        
        self._logger.debug(
            f"Stub: Finished trace span '{span.operation_name}' "
            f"(duration: {span.duration_ms:.2f}ms, status: {status})"
        )
    
    async def add_trace_tag(
        self,
        span: TraceSpan,
        key: str,
        value: Any
    ) -> None:
        """Stub implementation of tag addition."""
        span.tags[key] = value
        self._logger.debug(f"Stub: Added tag to span {span.span_id}: {key}={value}")
    
    async def log_trace_event(
        self,
        span: TraceSpan,
        event_name: str,
        data: Optional[Dict[str, Any]] = None
    ) -> None:
        """Stub implementation of event logging."""
        event = {
            "event_name": event_name,
            "timestamp": datetime.now().isoformat(),
            "data": data or {}
        }
        
        span.logs.append(event)
        self._logger.debug(f"Stub: Logged event in span {span.span_id}: {event_name}")
    
    def trace_operation(
        self,
        operation_name: str,
        tags: Optional[Dict[str, Any]] = None
    ) -> Callable:
        """Stub implementation of operation tracing decorator."""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                span = await self.start_trace(operation_name, tags=tags)
                try:
                    await self.add_trace_tag(span, "function_name", func.__name__)
                    await self.add_trace_tag(span, "module", func.__module__)
                    
                    result = await func(*args, **kwargs)
                    
                    await self.add_trace_tag(span, "success", True)
                    await self.finish_trace(span, "ok")
                    
                    return result
                except Exception as e:
                    await self.add_trace_tag(span, "success", False)
                    await self.add_trace_tag(span, "error_type", type(e).__name__)
                    await self.finish_trace(span, "error", str(e))
                    raise
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    execution_time = (time.time() - start_time) * 1000
                    self._logger.debug(
                        f"Stub: Traced operation '{operation_name}' "
                        f"(duration: {execution_time:.2f}ms, status: ok)"
                    )
                    return result
                except Exception as e:
                    execution_time = (time.time() - start_time) * 1000
                    self._logger.debug(
                        f"Stub: Traced operation '{operation_name}' "
                        f"(duration: {execution_time:.2f}ms, status: error, error: {e})"
                    )
                    raise
            
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
        return str(uuid.uuid4())


class AlertingStub(AlertingPort):
    """Stub implementation for alerting operations."""
    
    def __init__(self):
        """Initialize the alerting stub."""
        self._logger = logging.getLogger(__name__)
        self._logger.warning(
            "Using alerting stub. External alerting systems not available. "
            "Configure PagerDuty, Slack, or similar services for full functionality."
        )
        
        # In-memory storage for stub functionality
        self._alerts: Dict[str, Dict[str, Any]] = {}
    
    async def create_alert(
        self,
        alert_name: str,
        message: str,
        severity: str = "medium",
        tags: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Stub implementation of alert creation."""
        alert_id = str(uuid.uuid4())
        
        alert_data = {
            "id": alert_id,
            "name": alert_name,
            "message": message,
            "severity": severity,
            "tags": tags or {},
            "metadata": metadata or {},
            "status": "active",
            "created_at": datetime.now().isoformat(),
            "resolved_at": None
        }
        
        self._alerts[alert_id] = alert_data
        
        self._logger.warning(
            f"Stub ALERT: {alert_name} ({severity}) - {message} (ID: {alert_id})"
        )
        
        return alert_id
    
    async def resolve_alert(
        self,
        alert_id: str,
        resolution_message: Optional[str] = None
    ) -> None:
        """Stub implementation of alert resolution."""
        if alert_id in self._alerts:
            self._alerts[alert_id]["status"] = "resolved"
            self._alerts[alert_id]["resolved_at"] = datetime.now().isoformat()
            self._alerts[alert_id]["resolution_message"] = resolution_message
            
            self._logger.info(f"Stub: Resolved alert {alert_id}")
        else:
            self._logger.warning(f"Stub: Alert {alert_id} not found for resolution")
    
    async def get_active_alerts(
        self,
        tags: Optional[Dict[str, str]] = None
    ) -> List[Dict[str, Any]]:
        """Stub implementation of active alerts retrieval."""
        active_alerts = []
        
        for alert_data in self._alerts.values():
            if alert_data["status"] == "active":
                # Apply tag filters if provided
                if tags:
                    alert_tags = alert_data.get("tags", {})
                    if not all(alert_tags.get(key) == value for key, value in tags.items()):
                        continue
                
                active_alerts.append(alert_data.copy())
        
        return active_alerts


class HealthCheckStub(HealthCheckPort):
    """Stub implementation for health check operations."""
    
    def __init__(self):
        """Initialize the health check stub."""
        self._logger = logging.getLogger(__name__)
        self._logger.warning(
            "Using health check stub. External health check systems not available."
        )
        
        # In-memory storage for registered checks
        self._health_checks: Dict[str, Callable] = {}
        self._check_results: Dict[str, Dict[str, Any]] = {}
    
    async def check_health(self) -> Dict[str, Any]:
        """Stub implementation of health check."""
        overall_status = "healthy"
        component_statuses = {}
        
        # Run registered health checks
        for check_name, check_function in self._health_checks.items():
            try:
                is_healthy = check_function()
                status = "healthy" if is_healthy else "unhealthy"
                
                if not is_healthy:
                    overall_status = "unhealthy"
                
                component_statuses[check_name] = {
                    "status": status,
                    "checked_at": datetime.now().isoformat()
                }
                
            except Exception as e:
                overall_status = "unhealthy"
                component_statuses[check_name] = {
                    "status": "error",
                    "error": str(e),
                    "checked_at": datetime.now().isoformat()
                }
        
        # Add default system checks
        component_statuses["memory"] = {"status": "healthy", "usage": "normal"}
        component_statuses["disk"] = {"status": "healthy", "usage": "normal"}
        component_statuses["network"] = {"status": "healthy", "connectivity": "ok"}
        
        health_result = {
            "status": overall_status,
            "timestamp": datetime.now().isoformat(),
            "components": component_statuses,
            "uptime_seconds": 3600,  # Mock uptime
            "version": "1.0.0-stub"
        }
        
        self._check_results["latest"] = health_result
        return health_result
    
    async def register_health_check(
        self,
        check_name: str,
        check_function: Callable[[], bool],
        check_interval_seconds: int = 60
    ) -> None:
        """Stub implementation of health check registration."""
        self._health_checks[check_name] = check_function
        
        self._logger.info(
            f"Stub: Registered health check '{check_name}' "
            f"(interval: {check_interval_seconds}s)"
        )
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Stub implementation of system status retrieval."""
        # Return latest health check or perform new one
        if "latest" in self._check_results:
            return self._check_results["latest"]
        else:
            return await self.check_health()