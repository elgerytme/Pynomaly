"""Monitoring Operations Interfaces (Ports).

This module defines the abstract interfaces for monitoring and observability
operations that the machine learning domain requires. These interfaces
represent the "ports" in hexagonal architecture, defining contracts for
external monitoring systems without coupling to specific implementations.

Following DDD principles, these interfaces belong to the domain layer and
define what the domain needs from external monitoring capabilities.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime
from dataclasses import dataclass
from enum import Enum


class MetricType(Enum):
    """Types of metrics that can be tracked."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"
    SUMMARY = "summary"


class TraceLevel(Enum):
    """Levels for distributed tracing."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class MetricValue:
    """A metric value with metadata."""
    name: str
    value: Union[int, float]
    metric_type: MetricType
    labels: Dict[str, str]
    timestamp: datetime
    description: Optional[str] = None


@dataclass
class TraceSpan:
    """A trace span for distributed tracing."""
    operation_name: str
    span_id: str
    trace_id: str
    parent_span_id: Optional[str]
    start_time: datetime
    end_time: Optional[datetime]
    duration_ms: Optional[float]
    tags: Dict[str, Any]
    logs: List[Dict[str, Any]]
    status: str  # "ok", "error", "timeout"
    error_message: Optional[str] = None


@dataclass
class PerformanceMetrics:
    """Performance metrics for operations."""
    operation_name: str
    execution_time_ms: float
    memory_usage_mb: float
    cpu_usage_percent: float
    success: bool
    error_message: Optional[str] = None
    additional_metrics: Dict[str, Any] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.additional_metrics is None:
            self.additional_metrics = {}


class MonitoringPort(ABC):
    """Port for monitoring operations.
    
    This interface defines the contract for collecting and reporting
    metrics, performance data, and system health information.
    """

    @abstractmethod
    async def record_metric(
        self,
        metric: MetricValue
    ) -> None:
        """Record a metric value.
        
        Args:
            metric: Metric value to record
            
        Raises:
            MetricRecordingError: If metric recording fails
        """
        pass

    @abstractmethod
    async def record_performance_metrics(
        self,
        metrics: PerformanceMetrics
    ) -> None:
        """Record performance metrics for an operation.
        
        Args:
            metrics: Performance metrics to record
            
        Raises:
            MetricRecordingError: If performance metric recording fails
        """
        pass

    @abstractmethod
    async def increment_counter(
        self,
        name: str,
        value: int = 1,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Increment a counter metric.
        
        Args:
            name: Counter name
            value: Value to increment by
            labels: Optional labels for the metric
            
        Raises:
            MetricRecordingError: If counter increment fails
        """
        pass

    @abstractmethod
    async def set_gauge(
        self,
        name: str,
        value: Union[int, float],
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Set a gauge metric value.
        
        Args:
            name: Gauge name
            value: Value to set
            labels: Optional labels for the metric
            
        Raises:
            MetricRecordingError: If gauge setting fails
        """
        pass

    @abstractmethod
    async def record_histogram(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Record a value in a histogram metric.
        
        Args:
            name: Histogram name
            value: Value to record
            labels: Optional labels for the metric
            
        Raises:
            MetricRecordingError: If histogram recording fails
        """
        pass

    @abstractmethod
    async def get_metric_value(
        self,
        name: str,
        labels: Optional[Dict[str, str]] = None
    ) -> Optional[MetricValue]:
        """Get current value of a metric.
        
        Args:
            name: Metric name
            labels: Optional labels to filter by
            
        Returns:
            Current metric value if found, None otherwise
            
        Raises:
            MetricRetrievalError: If metric retrieval fails
        """
        pass


class DistributedTracingPort(ABC):
    """Port for distributed tracing operations.
    
    This interface defines the contract for creating and managing
    distributed traces across service boundaries.
    """

    @abstractmethod
    async def start_trace(
        self,
        operation_name: str,
        parent_span_id: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None
    ) -> TraceSpan:
        """Start a new trace span.
        
        Args:
            operation_name: Name of the operation being traced
            parent_span_id: Optional parent span ID
            tags: Optional tags for the span
            
        Returns:
            New trace span
            
        Raises:
            TracingError: If trace creation fails
        """
        pass

    @abstractmethod
    async def finish_trace(
        self,
        span: TraceSpan,
        status: str = "ok",
        error_message: Optional[str] = None
    ) -> None:
        """Finish a trace span.
        
        Args:
            span: Trace span to finish
            status: Final status of the operation
            error_message: Optional error message if operation failed
            
        Raises:
            TracingError: If trace finishing fails
        """
        pass

    @abstractmethod
    async def add_trace_tag(
        self,
        span: TraceSpan,
        key: str,
        value: Any
    ) -> None:
        """Add a tag to a trace span.
        
        Args:
            span: Trace span to add tag to
            key: Tag key
            value: Tag value
            
        Raises:
            TracingError: If tag addition fails
        """
        pass

    @abstractmethod
    async def log_trace_event(
        self,
        span: TraceSpan,
        event_name: str,
        data: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log an event within a trace span.
        
        Args:
            span: Trace span to log event in
            event_name: Name of the event
            data: Optional event data
            
        Raises:
            TracingError: If event logging fails
        """
        pass

    @abstractmethod
    def trace_operation(
        self,
        operation_name: str,
        tags: Optional[Dict[str, Any]] = None
    ) -> Callable:
        """Decorator for tracing operations.
        
        Args:
            operation_name: Name of the operation to trace
            tags: Optional tags for the trace
            
        Returns:
            Decorator function for tracing
            
        Raises:
            TracingError: If decorator setup fails
        """
        pass


class AlertingPort(ABC):
    """Port for alerting operations.
    
    This interface defines the contract for creating and managing
    alerts based on system metrics and conditions.
    """

    @abstractmethod
    async def create_alert(
        self,
        alert_name: str,
        message: str,
        severity: str = "medium",
        tags: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create an alert.
        
        Args:
            alert_name: Name of the alert
            message: Alert message
            severity: Alert severity level
            tags: Optional tags for the alert
            metadata: Optional additional metadata
            
        Returns:
            Alert ID
            
        Raises:
            AlertCreationError: If alert creation fails
        """
        pass

    @abstractmethod
    async def resolve_alert(
        self,
        alert_id: str,
        resolution_message: Optional[str] = None
    ) -> None:
        """Resolve an alert.
        
        Args:
            alert_id: ID of the alert to resolve
            resolution_message: Optional resolution message
            
        Raises:
            AlertResolutionError: If alert resolution fails
        """
        pass

    @abstractmethod
    async def get_active_alerts(
        self,
        tags: Optional[Dict[str, str]] = None
    ) -> List[Dict[str, Any]]:
        """Get list of active alerts.
        
        Args:
            tags: Optional tags to filter alerts by
            
        Returns:
            List of active alerts
            
        Raises:
            AlertRetrievalError: If alert retrieval fails
        """
        pass


class HealthCheckPort(ABC):
    """Port for health check operations.
    
    This interface defines the contract for monitoring system health
    and availability.
    """

    @abstractmethod
    async def check_health(self) -> Dict[str, Any]:
        """Perform health check.
        
        Returns:
            Health check results with status and details
            
        Raises:
            HealthCheckError: If health check fails
        """
        pass

    @abstractmethod
    async def register_health_check(
        self,
        check_name: str,
        check_function: Callable[[], bool],
        check_interval_seconds: int = 60
    ) -> None:
        """Register a health check function.
        
        Args:
            check_name: Name of the health check
            check_function: Function to execute for health check
            check_interval_seconds: Interval between checks
            
        Raises:
            HealthCheckRegistrationError: If registration fails
        """
        pass

    @abstractmethod
    async def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status.
        
        Returns:
            System status with component health information
            
        Raises:
            StatusRetrievalError: If status retrieval fails
        """
        pass


# Custom exceptions for monitoring operations
class MonitoringOperationError(Exception):
    """Base exception for monitoring operation errors."""
    pass


class MetricRecordingError(MonitoringOperationError):
    """Exception raised during metric recording."""
    pass


class MetricRetrievalError(MonitoringOperationError):
    """Exception raised during metric retrieval."""
    pass


class TracingError(MonitoringOperationError):
    """Exception raised during tracing operations."""
    pass


class AlertCreationError(MonitoringOperationError):
    """Exception raised during alert creation."""
    pass


class AlertResolutionError(MonitoringOperationError):
    """Exception raised during alert resolution."""
    pass


class AlertRetrievalError(MonitoringOperationError):
    """Exception raised during alert retrieval."""
    pass


class HealthCheckError(MonitoringOperationError):
    """Exception raised during health checks."""
    pass


class HealthCheckRegistrationError(MonitoringOperationError):
    """Exception raised during health check registration."""
    pass


class StatusRetrievalError(MonitoringOperationError):
    """Exception raised during status retrieval."""
    pass