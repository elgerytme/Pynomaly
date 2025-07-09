"""Production monitoring domain models for observability and performance tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

import numpy as np


class MetricType(Enum):
    """Types of metrics collected."""
    
    # Performance metrics
    COUNTER = "counter"  # Always increasing
    GAUGE = "gauge"  # Can go up or down
    HISTOGRAM = "histogram"  # Distribution of values
    SUMMARY = "summary"  # Quantiles and count
    
    # Business metrics
    BUSINESS_KPI = "business_kpi"
    ANOMALY_DETECTION_RATE = "anomaly_detection_rate"
    MODEL_ACCURACY = "model_accuracy"
    
    # System metrics
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    DISK_USAGE = "disk_usage"
    NETWORK_IO = "network_io"
    
    # Application metrics
    REQUEST_DURATION = "request_duration"
    REQUEST_RATE = "request_rate"
    ERROR_RATE = "error_rate"
    THROUGHPUT = "throughput"


class AlertSeverity(Enum):
    """Alert severity levels."""
    
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertStatus(Enum):
    """Alert status states."""
    
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SILENCED = "silenced"


class DashboardType(Enum):
    """Types of monitoring dashboards."""
    
    SYSTEM_OVERVIEW = "system_overview"
    APPLICATION_PERFORMANCE = "application_performance"
    BUSINESS_METRICS = "business_metrics"
    SECURITY_MONITORING = "security_monitoring"
    ML_MODEL_PERFORMANCE = "ml_model_performance"
    INFRASTRUCTURE = "infrastructure"
    CUSTOM = "custom"


@dataclass
class MetricPoint:
    """Single metric data point."""
    
    timestamp: datetime
    value: Union[float, int, str]
    labels: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        if isinstance(self.value, str) and len(self.value) > 1000:
            raise ValueError("String metric values cannot exceed 1000 characters")


@dataclass
class Metric:
    """Metric definition and metadata."""
    
    metric_id: UUID
    name: str
    metric_type: MetricType
    description: str
    unit: str = ""
    
    # Metric configuration
    labels: Dict[str, str] = field(default_factory=dict)
    help_text: str = ""
    
    # Data points
    data_points: List[MetricPoint] = field(default_factory=list)
    
    # Retention settings
    retention_period: timedelta = field(default=timedelta(days=90))
    aggregation_interval: timedelta = field(default=timedelta(minutes=1))
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    source_system: str = "pynomaly"
    
    def __post_init__(self):
        if not self.name:
            raise ValueError("Metric name cannot be empty")
        if not self.description:
            raise ValueError("Metric description cannot be empty")
    
    def add_data_point(self, value: Union[float, int, str], labels: Optional[Dict[str, str]] = None) -> None:
        """Add a new data point to the metric."""
        point = MetricPoint(
            timestamp=datetime.utcnow(),
            value=value,
            labels=labels or {}
        )
        self.data_points.append(point)
        self.updated_at = datetime.utcnow()
        
        # Clean up old data points based on retention policy
        cutoff_time = datetime.utcnow() - self.retention_period
        self.data_points = [p for p in self.data_points if p.timestamp > cutoff_time]
    
    def get_latest_value(self) -> Optional[Union[float, int, str]]:
        """Get the most recent metric value."""
        if not self.data_points:
            return None
        return max(self.data_points, key=lambda p: p.timestamp).value
    
    def get_values_in_range(
        self, 
        start_time: datetime, 
        end_time: datetime
    ) -> List[MetricPoint]:
        """Get data points within a time range."""
        return [
            p for p in self.data_points 
            if start_time <= p.timestamp <= end_time
        ]
    
    def calculate_average(
        self, 
        start_time: Optional[datetime] = None, 
        end_time: Optional[datetime] = None
    ) -> Optional[float]:
        """Calculate average value over a time period."""
        if not end_time:
            end_time = datetime.utcnow()
        if not start_time:
            start_time = end_time - timedelta(hours=1)  # Default to last hour
        
        points = self.get_values_in_range(start_time, end_time)
        numeric_values = [p.value for p in points if isinstance(p.value, (int, float))]
        
        if not numeric_values:
            return None
        
        return sum(numeric_values) / len(numeric_values)


@dataclass
class AlertRule:
    """Alert rule configuration."""
    
    rule_id: UUID
    name: str
    description: str
    metric_name: str
    
    # Condition configuration
    condition: str  # e.g., "greater_than", "less_than", "equals", "not_equals"
    threshold: Union[float, int, str]
    evaluation_window: timedelta = field(default=timedelta(minutes=5))
    
    # Advanced conditions
    comparison_operator: str = ">"  # >, <, ==, !=, >=, <=
    aggregation_function: str = "avg"  # avg, sum, min, max, count
    
    # Alert configuration
    severity: AlertSeverity = AlertSeverity.WARNING
    message_template: str = "Alert: {metric_name} {condition} {threshold}"
    
    # Notification settings
    notification_channels: List[str] = field(default_factory=list)
    escalation_rules: List[Dict[str, Any]] = field(default_factory=list)
    
    # Rate limiting
    cooldown_period: timedelta = field(default=timedelta(minutes=15))
    max_alerts_per_hour: int = 10
    
    # Metadata
    is_enabled: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    created_by: UUID = field(default_factory=uuid4)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        if not self.name:
            raise ValueError("Alert rule name cannot be empty")
        if not self.metric_name:
            raise ValueError("Metric name cannot be empty")
        if self.comparison_operator not in [">", "<", "==", "!=", ">=", "<="]:
            raise ValueError("Invalid comparison operator")
        if self.aggregation_function not in ["avg", "sum", "min", "max", "count"]:
            raise ValueError("Invalid aggregation function")
    
    def evaluate(self, metric_value: Union[float, int, str]) -> bool:
        """Evaluate if the alert condition is met."""
        if not isinstance(metric_value, (int, float)) or not isinstance(self.threshold, (int, float)):
            # For string comparisons
            if self.comparison_operator == "==":
                return metric_value == self.threshold
            elif self.comparison_operator == "!=":
                return metric_value != self.threshold
            return False
        
        # Numeric comparisons
        if self.comparison_operator == ">":
            return metric_value > self.threshold
        elif self.comparison_operator == "<":
            return metric_value < self.threshold
        elif self.comparison_operator == ">=":
            return metric_value >= self.threshold
        elif self.comparison_operator == "<=":
            return metric_value <= self.threshold
        elif self.comparison_operator == "==":
            return metric_value == self.threshold
        elif self.comparison_operator == "!=":
            return metric_value != self.threshold
        
        return False
    
    def generate_alert_message(self, metric_value: Union[float, int, str]) -> str:
        """Generate alert message from template."""
        return self.message_template.format(
            metric_name=self.metric_name,
            condition=self.comparison_operator,
            threshold=self.threshold,
            value=metric_value,
            severity=self.severity.value,
        )


@dataclass
class Alert:
    """Active alert instance."""
    
    alert_id: UUID
    rule_id: UUID
    rule_name: str
    
    # Alert details
    metric_name: str
    metric_value: Union[float, int, str]
    threshold: Union[float, int, str]
    severity: AlertSeverity
    
    # Status
    status: AlertStatus = AlertStatus.ACTIVE
    message: str = ""
    
    # Timestamps
    triggered_at: datetime = field(default_factory=datetime.utcnow)
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    
    # Assignment and handling
    assigned_to: Optional[UUID] = None
    acknowledged_by: Optional[UUID] = None
    resolved_by: Optional[UUID] = None
    
    # Additional context
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    related_alerts: List[UUID] = field(default_factory=list)
    
    # Escalation tracking
    escalation_level: int = 0
    last_escalated_at: Optional[datetime] = None
    notification_count: int = 0
    
    def __post_init__(self):
        if not self.message:
            self.message = f"Alert: {self.metric_name} = {self.metric_value} (threshold: {self.threshold})"
    
    def acknowledge(self, acknowledged_by: UUID) -> None:
        """Acknowledge the alert."""
        if self.status == AlertStatus.ACTIVE:
            self.status = AlertStatus.ACKNOWLEDGED
            self.acknowledged_at = datetime.utcnow()
            self.acknowledged_by = acknowledged_by
    
    def resolve(self, resolved_by: UUID, resolution_note: Optional[str] = None) -> None:
        """Resolve the alert."""
        self.status = AlertStatus.RESOLVED
        self.resolved_at = datetime.utcnow()
        self.resolved_by = resolved_by
        
        if resolution_note:
            self.annotations["resolution_note"] = resolution_note
    
    def get_duration(self) -> timedelta:
        """Get alert duration."""
        end_time = self.resolved_at or datetime.utcnow()
        return end_time - self.triggered_at
    
    def is_escalation_due(self, escalation_interval: timedelta) -> bool:
        """Check if alert should be escalated."""
        if self.status != AlertStatus.ACTIVE:
            return False
        
        if not self.last_escalated_at:
            # First escalation check
            return datetime.utcnow() - self.triggered_at >= escalation_interval
        
        return datetime.utcnow() - self.last_escalated_at >= escalation_interval


@dataclass
class DashboardWidget:
    """Individual dashboard widget configuration."""
    
    widget_id: UUID
    title: str
    widget_type: str  # "chart", "gauge", "table", "text", "alert_list"
    
    # Data configuration
    metrics: List[str] = field(default_factory=list)
    query: str = ""
    time_range: str = "1h"  # "5m", "1h", "6h", "24h", "7d", "30d"
    
    # Display configuration
    position: Dict[str, int] = field(default_factory=lambda: {"x": 0, "y": 0, "width": 4, "height": 3})
    chart_type: str = "line"  # "line", "bar", "pie", "heatmap", "table"
    
    # Styling
    colors: List[str] = field(default_factory=list)
    thresholds: List[Dict[str, Any]] = field(default_factory=list)
    
    # Refresh settings
    auto_refresh: bool = True
    refresh_interval: int = 30  # seconds
    
    # Widget-specific options
    options: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.title:
            raise ValueError("Widget title cannot be empty")
        if self.refresh_interval < 5:
            raise ValueError("Refresh interval must be at least 5 seconds")


@dataclass
class Dashboard:
    """Monitoring dashboard configuration."""
    
    dashboard_id: UUID
    name: str
    description: str
    dashboard_type: DashboardType = DashboardType.CUSTOM
    
    # Dashboard content
    widgets: List[DashboardWidget] = field(default_factory=list)
    layout: Dict[str, Any] = field(default_factory=dict)
    
    # Access control
    is_public: bool = False
    owner_id: UUID = field(default_factory=uuid4)
    viewers: List[UUID] = field(default_factory=list)
    editors: List[UUID] = field(default_factory=list)
    
    # Display settings
    theme: str = "dark"  # "dark", "light"
    auto_refresh: bool = True
    refresh_interval: int = 30  # seconds
    time_zone: str = "UTC"
    
    # Metadata
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    last_viewed_at: Optional[datetime] = None
    view_count: int = 0
    
    def __post_init__(self):
        if not self.name:
            raise ValueError("Dashboard name cannot be empty")
        if self.refresh_interval < 5:
            raise ValueError("Refresh interval must be at least 5 seconds")
    
    def add_widget(self, widget: DashboardWidget) -> None:
        """Add widget to dashboard."""
        self.widgets.append(widget)
        self.updated_at = datetime.utcnow()
    
    def remove_widget(self, widget_id: UUID) -> bool:
        """Remove widget from dashboard."""
        initial_count = len(self.widgets)
        self.widgets = [w for w in self.widgets if w.widget_id != widget_id]
        
        if len(self.widgets) < initial_count:
            self.updated_at = datetime.utcnow()
            return True
        
        return False
    
    def can_view(self, user_id: UUID) -> bool:
        """Check if user can view dashboard."""
        return (
            self.is_public or 
            user_id == self.owner_id or 
            user_id in self.viewers or 
            user_id in self.editors
        )
    
    def can_edit(self, user_id: UUID) -> bool:
        """Check if user can edit dashboard."""
        return user_id == self.owner_id or user_id in self.editors


@dataclass
class HealthCheck:
    """System health check configuration and results."""
    
    check_id: UUID
    name: str
    description: str
    check_type: str  # "http", "tcp", "command", "database", "custom"
    
    # Check configuration
    target: str  # URL, host:port, command, etc.
    timeout: int = 30  # seconds
    interval: int = 60  # seconds
    
    # Health status
    is_healthy: bool = True
    last_check_at: Optional[datetime] = None
    last_success_at: Optional[datetime] = None
    last_failure_at: Optional[datetime] = None
    
    # Failure tracking
    consecutive_failures: int = 0
    total_failures: int = 0
    failure_threshold: int = 3
    
    # Performance tracking
    response_times: List[float] = field(default_factory=list)  # Last 100 response times
    average_response_time: float = 0.0
    
    # Check results
    last_result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    
    # Metadata
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        if not self.name:
            raise ValueError("Health check name cannot be empty")
        if self.timeout <= 0:
            raise ValueError("Timeout must be positive")
        if self.interval <= 0:
            raise ValueError("Interval must be positive")
    
    def record_success(self, response_time: float) -> None:
        """Record successful health check."""
        self.is_healthy = True
        self.last_check_at = datetime.utcnow()
        self.last_success_at = datetime.utcnow()
        self.consecutive_failures = 0
        self.error_message = None
        
        # Update response time tracking
        self.response_times.append(response_time)
        if len(self.response_times) > 100:
            self.response_times = self.response_times[-100:]
        
        self.average_response_time = sum(self.response_times) / len(self.response_times)
    
    def record_failure(self, error_message: str) -> None:
        """Record failed health check."""
        self.last_check_at = datetime.utcnow()
        self.last_failure_at = datetime.utcnow()
        self.consecutive_failures += 1
        self.total_failures += 1
        self.error_message = error_message
        
        # Mark as unhealthy if threshold exceeded
        if self.consecutive_failures >= self.failure_threshold:
            self.is_healthy = False
    
    def get_uptime_percentage(self, period: timedelta) -> float:
        """Calculate uptime percentage over a period."""
        # This would be calculated from historical data in production
        if self.total_failures == 0:
            return 100.0
        
        # Simplified calculation
        total_checks = max(1, int(period.total_seconds() / self.interval))
        successful_checks = total_checks - min(self.total_failures, total_checks)
        
        return (successful_checks / total_checks) * 100


@dataclass
class ServiceStatus:
    """Overall service status tracking."""
    
    service_name: str
    service_version: str
    
    # Overall health
    overall_status: str = "healthy"  # "healthy", "degraded", "unhealthy", "unknown"
    health_score: float = 100.0  # 0-100
    
    # Component health
    component_statuses: Dict[str, bool] = field(default_factory=dict)
    health_checks: Dict[str, HealthCheck] = field(default_factory=dict)
    
    # Performance metrics
    response_time_p50: float = 0.0
    response_time_p95: float = 0.0
    response_time_p99: float = 0.0
    error_rate: float = 0.0
    throughput: float = 0.0
    
    # Resource utilization
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    disk_usage: float = 0.0
    
    # Active issues
    active_alerts: int = 0
    critical_alerts: int = 0
    
    # Timestamps
    last_updated: datetime = field(default_factory=datetime.utcnow)
    uptime_start: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        if not self.service_name:
            raise ValueError("Service name cannot be empty")
        if not self.service_version:
            raise ValueError("Service version cannot be empty")
        if not 0 <= self.health_score <= 100:
            raise ValueError("Health score must be between 0 and 100")
    
    def add_health_check(self, health_check: HealthCheck) -> None:
        """Add health check to service status."""
        self.health_checks[health_check.name] = health_check
        self._update_overall_status()
    
    def update_component_status(self, component: str, is_healthy: bool) -> None:
        """Update component health status."""
        self.component_statuses[component] = is_healthy
        self._update_overall_status()
    
    def _update_overall_status(self) -> None:
        """Update overall service status based on components and health checks."""
        self.last_updated = datetime.utcnow()
        
        # Check health checks
        unhealthy_checks = sum(1 for check in self.health_checks.values() if not check.is_healthy)
        total_checks = len(self.health_checks)
        
        # Check components
        unhealthy_components = sum(1 for status in self.component_statuses.values() if not status)
        total_components = len(self.component_statuses)
        
        # Calculate health score
        if total_checks + total_components == 0:
            self.health_score = 100.0
        else:
            healthy_items = (total_checks - unhealthy_checks) + (total_components - unhealthy_components)
            total_items = total_checks + total_components
            self.health_score = (healthy_items / total_items) * 100
        
        # Determine overall status
        if self.health_score >= 95:
            self.overall_status = "healthy"
        elif self.health_score >= 80:
            self.overall_status = "degraded"
        elif self.health_score >= 50:
            self.overall_status = "unhealthy"
        else:
            self.overall_status = "critical"
        
        # Factor in active alerts
        if self.critical_alerts > 0:
            self.overall_status = "critical"
        elif self.active_alerts > 5:
            if self.overall_status == "healthy":
                self.overall_status = "degraded"
    
    def get_uptime_duration(self) -> timedelta:
        """Get service uptime duration."""
        return datetime.utcnow() - self.uptime_start
    
    def get_status_summary(self) -> Dict[str, Any]:
        """Get comprehensive status summary."""
        return {
            "service": {
                "name": self.service_name,
                "version": self.service_version,
                "status": self.overall_status,
                "health_score": self.health_score,
                "uptime": self.get_uptime_duration().total_seconds(),
            },
            "performance": {
                "response_time_p50": self.response_time_p50,
                "response_time_p95": self.response_time_p95,
                "response_time_p99": self.response_time_p99,
                "error_rate": self.error_rate,
                "throughput": self.throughput,
            },
            "resources": {
                "cpu_usage": self.cpu_usage,
                "memory_usage": self.memory_usage,
                "disk_usage": self.disk_usage,
            },
            "health_checks": {
                "total": len(self.health_checks),
                "healthy": sum(1 for check in self.health_checks.values() if check.is_healthy),
                "unhealthy": sum(1 for check in self.health_checks.values() if not check.is_healthy),
            },
            "components": {
                "total": len(self.component_statuses),
                "healthy": sum(1 for status in self.component_statuses.values() if status),
                "unhealthy": sum(1 for status in self.component_statuses.values() if not status),
            },
            "alerts": {
                "active": self.active_alerts,
                "critical": self.critical_alerts,
            },
            "last_updated": self.last_updated.isoformat(),
        }