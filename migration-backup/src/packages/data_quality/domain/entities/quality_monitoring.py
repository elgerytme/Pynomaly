"""Real-time Quality Monitoring Domain Entities.

Contains entities for real-time quality monitoring including streaming assessment,
alerting, and dashboard components.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union, Callable
from enum import Enum
import uuid
from collections import deque


# Value Objects
@dataclass(frozen=True)
class MonitoringJobId:
    """Monitoring job identifier value object."""
    value: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class AlertId:
    """Alert identifier value object."""
    value: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class ThresholdId:
    """Threshold identifier value object."""
    value: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class StreamId:
    """Stream identifier value object."""
    value: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    def __str__(self) -> str:
        return self.value


# Enums
class MonitoringJobStatus(Enum):
    """Status of monitoring job."""
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    FAILED = "failed"
    SUSPENDED = "suspended"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertStatus(Enum):
    """Alert status."""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


class ThresholdType(Enum):
    """Types of quality thresholds."""
    STATIC = "static"
    DYNAMIC = "dynamic"
    STATISTICAL = "statistical"
    TREND_BASED = "trend_based"


class ComparisonOperator(Enum):
    """Comparison operators for thresholds."""
    LESS_THAN = "lt"
    LESS_THAN_OR_EQUAL = "lte"
    GREATER_THAN = "gt"
    GREATER_THAN_OR_EQUAL = "gte"
    EQUAL = "eq"
    NOT_EQUAL = "ne"
    BETWEEN = "between"
    OUTSIDE = "outside"


class WindowType(Enum):
    """Types of time windows for streaming analysis."""
    FIXED = "fixed"
    SLIDING = "sliding"
    SESSION = "session"
    TUMBLING = "tumbling"


class EscalationLevel(Enum):
    """Alert escalation levels."""
    LEVEL_1 = "level_1"
    LEVEL_2 = "level_2"
    LEVEL_3 = "level_3"
    EXECUTIVE = "executive"


class DashboardRefreshMode(Enum):
    """Dashboard refresh modes."""
    REAL_TIME = "real_time"
    NEAR_REAL_TIME = "near_real_time"
    PERIODIC = "periodic"
    ON_DEMAND = "on_demand"


# Data Structures
@dataclass(frozen=True)
class QualityWindow:
    """Time window for streaming quality assessment."""
    window_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    window_type: WindowType = WindowType.SLIDING
    duration_seconds: int = 300  # 5 minutes
    slide_interval_seconds: int = 60  # 1 minute
    max_records: Optional[int] = None
    session_timeout_seconds: Optional[int] = None
    
    def __post_init__(self):
        """Validate window configuration."""
        if self.duration_seconds <= 0:
            raise ValueError("Window duration must be positive")
        
        if self.slide_interval_seconds <= 0:
            raise ValueError("Slide interval must be positive")
        
        if self.window_type == WindowType.SLIDING and self.slide_interval_seconds > self.duration_seconds:
            raise ValueError("Slide interval cannot be greater than window duration")


@dataclass(frozen=True)
class StreamingMetrics:
    """Metrics for streaming quality assessment."""
    throughput_records_per_second: float = 0.0
    latency_ms: float = 0.0
    error_rate: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    active_windows: int = 0
    processed_records: int = 0
    failed_records: int = 0
    backlog_size: int = 0
    
    def get_health_score(self) -> float:
        """Calculate overall health score (0-1)."""
        # Simple health scoring based on key metrics
        score = 1.0
        
        if self.error_rate > 0.1:  # >10% error rate
            score *= 0.5
        elif self.error_rate > 0.05:  # >5% error rate
            score *= 0.8
        
        if self.latency_ms > 1000:  # >1 second latency
            score *= 0.6
        elif self.latency_ms > 500:  # >500ms latency
            score *= 0.9
        
        if self.backlog_size > 10000:  # Large backlog
            score *= 0.7
        
        return max(0.0, score)


@dataclass(frozen=True)
class QualityThreshold:
    """Quality threshold for monitoring."""
    threshold_id: ThresholdId = field(default_factory=ThresholdId)
    name: str = ""
    description: str = ""
    threshold_type: ThresholdType = ThresholdType.STATIC
    
    # Threshold definition
    metric_name: str = ""
    operator: ComparisonOperator = ComparisonOperator.LESS_THAN
    value: Union[float, int] = 0.0
    upper_bound: Optional[Union[float, int]] = None
    lower_bound: Optional[Union[float, int]] = None
    
    # Alert configuration
    alert_severity: AlertSeverity = AlertSeverity.WARNING
    alert_message_template: str = "Quality threshold {name} violated: {metric_name} {operator} {value}"
    
    # Timing configuration
    evaluation_window: QualityWindow = field(default_factory=QualityWindow)
    grace_period_seconds: int = 0
    min_occurrences: int = 1
    
    # Advanced configuration
    enable_correlation: bool = False
    correlation_thresholds: List[str] = field(default_factory=list)
    enable_suppression: bool = False
    suppression_duration_seconds: int = 300
    
    def evaluate(self, value: Union[float, int]) -> bool:
        """Evaluate threshold against a value."""
        if self.operator == ComparisonOperator.LESS_THAN:
            return value < self.value
        elif self.operator == ComparisonOperator.LESS_THAN_OR_EQUAL:
            return value <= self.value
        elif self.operator == ComparisonOperator.GREATER_THAN:
            return value > self.value
        elif self.operator == ComparisonOperator.GREATER_THAN_OR_EQUAL:
            return value >= self.value
        elif self.operator == ComparisonOperator.EQUAL:
            return value == self.value
        elif self.operator == ComparisonOperator.NOT_EQUAL:
            return value != self.value
        elif self.operator == ComparisonOperator.BETWEEN:
            return (self.lower_bound or 0) <= value <= (self.upper_bound or float('inf'))
        elif self.operator == ComparisonOperator.OUTSIDE:
            return value < (self.lower_bound or 0) or value > (self.upper_bound or float('inf'))
        else:
            raise ValueError(f"Unknown comparison operator: {self.operator}")
    
    def format_alert_message(self, actual_value: Union[float, int], context: Dict[str, Any] = None) -> str:
        """Format alert message with actual values."""
        context = context or {}
        return self.alert_message_template.format(
            name=self.name,
            metric_name=self.metric_name,
            operator=self.operator.value,
            value=self.value,
            actual_value=actual_value,
            **context
        )


@dataclass(frozen=True)
class QualityAlert:
    """Quality alert entity."""
    alert_id: AlertId = field(default_factory=AlertId)
    threshold_id: ThresholdId = field(default_factory=ThresholdId)
    monitoring_job_id: MonitoringJobId = field(default_factory=MonitoringJobId)
    stream_id: StreamId = field(default_factory=StreamId)
    
    # Alert details
    severity: AlertSeverity = AlertSeverity.WARNING
    status: AlertStatus = AlertStatus.ACTIVE
    title: str = ""
    description: str = ""
    
    # Timing
    triggered_at: datetime = field(default_factory=datetime.now)
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    
    # Context
    metric_name: str = ""
    actual_value: Union[float, int] = 0.0
    threshold_value: Union[float, int] = 0.0
    affected_records: int = 0
    context_data: Dict[str, Any] = field(default_factory=dict)
    
    # Escalation
    escalation_level: EscalationLevel = EscalationLevel.LEVEL_1
    escalated_at: Optional[datetime] = None
    assigned_to: Optional[str] = None
    
    # Resolution
    resolution_notes: str = ""
    resolution_action: str = ""
    
    def acknowledge(self, user: str, notes: str = "") -> 'QualityAlert':
        """Acknowledge the alert."""
        return QualityAlert(
            alert_id=self.alert_id,
            threshold_id=self.threshold_id,
            monitoring_job_id=self.monitoring_job_id,
            stream_id=self.stream_id,
            severity=self.severity,
            status=AlertStatus.ACKNOWLEDGED,
            title=self.title,
            description=self.description,
            triggered_at=self.triggered_at,
            acknowledged_at=datetime.now(),
            resolved_at=self.resolved_at,
            metric_name=self.metric_name,
            actual_value=self.actual_value,
            threshold_value=self.threshold_value,
            affected_records=self.affected_records,
            context_data=self.context_data,
            escalation_level=self.escalation_level,
            escalated_at=self.escalated_at,
            assigned_to=user,
            resolution_notes=notes,
            resolution_action=self.resolution_action
        )
    
    def resolve(self, user: str, action: str, notes: str = "") -> 'QualityAlert':
        """Resolve the alert."""
        return QualityAlert(
            alert_id=self.alert_id,
            threshold_id=self.threshold_id,
            monitoring_job_id=self.monitoring_job_id,
            stream_id=self.stream_id,
            severity=self.severity,
            status=AlertStatus.RESOLVED,
            title=self.title,
            description=self.description,
            triggered_at=self.triggered_at,
            acknowledged_at=self.acknowledged_at,
            resolved_at=datetime.now(),
            metric_name=self.metric_name,
            actual_value=self.actual_value,
            threshold_value=self.threshold_value,
            affected_records=self.affected_records,
            context_data=self.context_data,
            escalation_level=self.escalation_level,
            escalated_at=self.escalated_at,
            assigned_to=user,
            resolution_notes=notes,
            resolution_action=action
        )
    
    def escalate(self, new_level: EscalationLevel) -> 'QualityAlert':
        """Escalate the alert to a higher level."""
        return QualityAlert(
            alert_id=self.alert_id,
            threshold_id=self.threshold_id,
            monitoring_job_id=self.monitoring_job_id,
            stream_id=self.stream_id,
            severity=self.severity,
            status=self.status,
            title=self.title,
            description=self.description,
            triggered_at=self.triggered_at,
            acknowledged_at=self.acknowledged_at,
            resolved_at=self.resolved_at,
            metric_name=self.metric_name,
            actual_value=self.actual_value,
            threshold_value=self.threshold_value,
            affected_records=self.affected_records,
            context_data=self.context_data,
            escalation_level=new_level,
            escalated_at=datetime.now(),
            assigned_to=self.assigned_to,
            resolution_notes=self.resolution_notes,
            resolution_action=self.resolution_action
        )
    
    def is_active(self) -> bool:
        """Check if alert is active."""
        return self.status == AlertStatus.ACTIVE
    
    def is_resolved(self) -> bool:
        """Check if alert is resolved."""
        return self.status == AlertStatus.RESOLVED
    
    def get_duration(self) -> timedelta:
        """Get alert duration."""
        end_time = self.resolved_at or datetime.now()
        return end_time - self.triggered_at
    
    def get_time_to_acknowledgment(self) -> Optional[timedelta]:
        """Get time to acknowledgment."""
        if self.acknowledged_at:
            return self.acknowledged_at - self.triggered_at
        return None
    
    def get_time_to_resolution(self) -> Optional[timedelta]:
        """Get time to resolution."""
        if self.resolved_at:
            return self.resolved_at - self.triggered_at
        return None


@dataclass(frozen=True)
class StreamingQualityAssessment:
    """Real-time quality assessment for a stream window."""
    assessment_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    stream_id: StreamId = field(default_factory=StreamId)
    window_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Assessment details
    window_start: datetime = field(default_factory=datetime.now)
    window_end: datetime = field(default_factory=datetime.now)
    records_processed: int = 0
    
    # Quality scores
    overall_score: float = 0.0
    completeness_score: float = 0.0
    accuracy_score: float = 0.0
    consistency_score: float = 0.0
    validity_score: float = 0.0
    uniqueness_score: float = 0.0
    timeliness_score: float = 0.0
    
    # Detailed metrics
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    threshold_violations: List[ThresholdId] = field(default_factory=list)
    processing_latency_ms: float = 0.0
    
    # Anomalies detected
    anomalies_detected: List[Dict[str, Any]] = field(default_factory=list)
    
    def get_window_duration(self) -> timedelta:
        """Get window duration."""
        return self.window_end - self.window_start
    
    def get_throughput(self) -> float:
        """Get processing throughput (records/second)."""
        duration = self.get_window_duration()
        if duration.total_seconds() > 0:
            return self.records_processed / duration.total_seconds()
        return 0.0
    
    def has_violations(self) -> bool:
        """Check if assessment has threshold violations."""
        return len(self.threshold_violations) > 0
    
    def get_quality_grade(self) -> str:
        """Get quality grade based on overall score."""
        if self.overall_score >= 0.95:
            return "A"
        elif self.overall_score >= 0.85:
            return "B"
        elif self.overall_score >= 0.75:
            return "C"
        elif self.overall_score >= 0.65:
            return "D"
        else:
            return "F"


@dataclass(frozen=True)
class MonitoringJobConfig:
    """Configuration for real-time monitoring job."""
    
    # Basic configuration
    job_name: str = ""
    description: str = ""
    enable_real_time: bool = True
    
    # Processing configuration
    batch_size: int = 1000
    max_concurrent_windows: int = 10
    processing_timeout_seconds: int = 30
    
    # Quality assessment
    quality_window: QualityWindow = field(default_factory=QualityWindow)
    enable_statistical_analysis: bool = True
    enable_anomaly_detection: bool = True
    enable_trend_analysis: bool = True
    
    # Thresholds
    quality_thresholds: List[QualityThreshold] = field(default_factory=list)
    
    # Alerting
    enable_alerting: bool = True
    alert_cooldown_seconds: int = 300
    max_alerts_per_hour: int = 100
    enable_alert_correlation: bool = True
    
    # Performance
    checkpoint_interval_seconds: int = 60
    state_retention_hours: int = 24
    max_memory_mb: int = 2048
    
    # Output
    enable_dashboard_updates: bool = True
    dashboard_refresh_seconds: int = 5
    enable_metrics_export: bool = True
    
    def add_threshold(self, threshold: QualityThreshold) -> 'MonitoringJobConfig':
        """Add a quality threshold to the configuration."""
        new_thresholds = self.quality_thresholds + [threshold]
        return MonitoringJobConfig(
            job_name=self.job_name,
            description=self.description,
            enable_real_time=self.enable_real_time,
            batch_size=self.batch_size,
            max_concurrent_windows=self.max_concurrent_windows,
            processing_timeout_seconds=self.processing_timeout_seconds,
            quality_window=self.quality_window,
            enable_statistical_analysis=self.enable_statistical_analysis,
            enable_anomaly_detection=self.enable_anomaly_detection,
            enable_trend_analysis=self.enable_trend_analysis,
            quality_thresholds=new_thresholds,
            enable_alerting=self.enable_alerting,
            alert_cooldown_seconds=self.alert_cooldown_seconds,
            max_alerts_per_hour=self.max_alerts_per_hour,
            enable_alert_correlation=self.enable_alert_correlation,
            checkpoint_interval_seconds=self.checkpoint_interval_seconds,
            state_retention_hours=self.state_retention_hours,
            max_memory_mb=self.max_memory_mb,
            enable_dashboard_updates=self.enable_dashboard_updates,
            dashboard_refresh_seconds=self.dashboard_refresh_seconds,
            enable_metrics_export=self.enable_metrics_export
        )


@dataclass(frozen=True)
class QualityMonitoringJob:
    """Real-time quality monitoring job entity."""
    
    job_id: MonitoringJobId = field(default_factory=MonitoringJobId)
    stream_id: StreamId = field(default_factory=StreamId)
    
    # Job details
    job_name: str = ""
    description: str = ""
    config: MonitoringJobConfig = field(default_factory=MonitoringJobConfig)
    
    # Status
    status: MonitoringJobStatus = MonitoringJobStatus.STARTING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    stopped_at: Optional[datetime] = None
    
    # Progress tracking
    current_window_id: Optional[str] = None
    windows_processed: int = 0
    total_records_processed: int = 0
    
    # Performance metrics
    streaming_metrics: StreamingMetrics = field(default_factory=StreamingMetrics)
    
    # Recent assessments (circular buffer)
    recent_assessments: List[StreamingQualityAssessment] = field(default_factory=list)
    active_alerts: List[QualityAlert] = field(default_factory=list)
    
    # Error handling
    error_count: int = 0
    last_error: Optional[str] = None
    last_error_at: Optional[datetime] = None
    
    def start(self) -> 'QualityMonitoringJob':
        """Start the monitoring job."""
        return QualityMonitoringJob(
            job_id=self.job_id,
            stream_id=self.stream_id,
            job_name=self.job_name,
            description=self.description,
            config=self.config,
            status=MonitoringJobStatus.RUNNING,
            created_at=self.created_at,
            started_at=datetime.now(),
            stopped_at=self.stopped_at,
            current_window_id=self.current_window_id,
            windows_processed=self.windows_processed,
            total_records_processed=self.total_records_processed,
            streaming_metrics=self.streaming_metrics,
            recent_assessments=self.recent_assessments,
            active_alerts=self.active_alerts,
            error_count=self.error_count,
            last_error=self.last_error,
            last_error_at=self.last_error_at
        )
    
    def stop(self) -> 'QualityMonitoringJob':
        """Stop the monitoring job."""
        return QualityMonitoringJob(
            job_id=self.job_id,
            stream_id=self.stream_id,
            job_name=self.job_name,
            description=self.description,
            config=self.config,
            status=MonitoringJobStatus.STOPPED,
            created_at=self.created_at,
            started_at=self.started_at,
            stopped_at=datetime.now(),
            current_window_id=self.current_window_id,
            windows_processed=self.windows_processed,
            total_records_processed=self.total_records_processed,
            streaming_metrics=self.streaming_metrics,
            recent_assessments=self.recent_assessments,
            active_alerts=self.active_alerts,
            error_count=self.error_count,
            last_error=self.last_error,
            last_error_at=self.last_error_at
        )
    
    def update_metrics(self, metrics: StreamingMetrics) -> 'QualityMonitoringJob':
        """Update streaming metrics."""
        return QualityMonitoringJob(
            job_id=self.job_id,
            stream_id=self.stream_id,
            job_name=self.job_name,
            description=self.description,
            config=self.config,
            status=self.status,
            created_at=self.created_at,
            started_at=self.started_at,
            stopped_at=self.stopped_at,
            current_window_id=self.current_window_id,
            windows_processed=self.windows_processed,
            total_records_processed=self.total_records_processed,
            streaming_metrics=metrics,
            recent_assessments=self.recent_assessments,
            active_alerts=self.active_alerts,
            error_count=self.error_count,
            last_error=self.last_error,
            last_error_at=self.last_error_at
        )
    
    def add_assessment(self, assessment: StreamingQualityAssessment) -> 'QualityMonitoringJob':
        """Add a new quality assessment."""
        # Keep only last 100 assessments
        new_assessments = (self.recent_assessments + [assessment])[-100:]
        
        return QualityMonitoringJob(
            job_id=self.job_id,
            stream_id=self.stream_id,
            job_name=self.job_name,
            description=self.description,
            config=self.config,
            status=self.status,
            created_at=self.created_at,
            started_at=self.started_at,
            stopped_at=self.stopped_at,
            current_window_id=assessment.window_id,
            windows_processed=self.windows_processed + 1,
            total_records_processed=self.total_records_processed + assessment.records_processed,
            streaming_metrics=self.streaming_metrics,
            recent_assessments=new_assessments,
            active_alerts=self.active_alerts,
            error_count=self.error_count,
            last_error=self.last_error,
            last_error_at=self.last_error_at
        )
    
    def add_alert(self, alert: QualityAlert) -> 'QualityMonitoringJob':
        """Add a new alert."""
        new_alerts = self.active_alerts + [alert]
        
        return QualityMonitoringJob(
            job_id=self.job_id,
            stream_id=self.stream_id,
            job_name=self.job_name,
            description=self.description,
            config=self.config,
            status=self.status,
            created_at=self.created_at,
            started_at=self.started_at,
            stopped_at=self.stopped_at,
            current_window_id=self.current_window_id,
            windows_processed=self.windows_processed,
            total_records_processed=self.total_records_processed,
            streaming_metrics=self.streaming_metrics,
            recent_assessments=self.recent_assessments,
            active_alerts=new_alerts,
            error_count=self.error_count,
            last_error=self.last_error,
            last_error_at=self.last_error_at
        )
    
    def update_alert(self, alert: QualityAlert) -> 'QualityMonitoringJob':
        """Update an existing alert."""
        new_alerts = []
        for existing_alert in self.active_alerts:
            if existing_alert.alert_id == alert.alert_id:
                new_alerts.append(alert)
            else:
                new_alerts.append(existing_alert)
        
        return QualityMonitoringJob(
            job_id=self.job_id,
            stream_id=self.stream_id,
            job_name=self.job_name,
            description=self.description,
            config=self.config,
            status=self.status,
            created_at=self.created_at,
            started_at=self.started_at,
            stopped_at=self.stopped_at,
            current_window_id=self.current_window_id,
            windows_processed=self.windows_processed,
            total_records_processed=self.total_records_processed,
            streaming_metrics=self.streaming_metrics,
            recent_assessments=self.recent_assessments,
            active_alerts=new_alerts,
            error_count=self.error_count,
            last_error=self.last_error,
            last_error_at=self.last_error_at
        )
    
    def record_error(self, error_message: str) -> 'QualityMonitoringJob':
        """Record an error."""
        return QualityMonitoringJob(
            job_id=self.job_id,
            stream_id=self.stream_id,
            job_name=self.job_name,
            description=self.description,
            config=self.config,
            status=self.status,
            created_at=self.created_at,
            started_at=self.started_at,
            stopped_at=self.stopped_at,
            current_window_id=self.current_window_id,
            windows_processed=self.windows_processed,
            total_records_processed=self.total_records_processed,
            streaming_metrics=self.streaming_metrics,
            recent_assessments=self.recent_assessments,
            active_alerts=self.active_alerts,
            error_count=self.error_count + 1,
            last_error=error_message,
            last_error_at=datetime.now()
        )
    
    def is_running(self) -> bool:
        """Check if job is running."""
        return self.status == MonitoringJobStatus.RUNNING
    
    def get_uptime(self) -> Optional[timedelta]:
        """Get job uptime."""
        if self.started_at:
            end_time = self.stopped_at or datetime.now()
            return end_time - self.started_at
        return None
    
    def get_current_quality_score(self) -> float:
        """Get current quality score from most recent assessment."""
        if self.recent_assessments:
            return self.recent_assessments[-1].overall_score
        return 0.0
    
    def get_active_alert_count(self) -> int:
        """Get count of active alerts."""
        return len([alert for alert in self.active_alerts if alert.is_active()])
    
    def get_critical_alert_count(self) -> int:
        """Get count of critical alerts."""
        return len([alert for alert in self.active_alerts 
                   if alert.is_active() and alert.severity == AlertSeverity.CRITICAL])
    
    def get_job_summary(self) -> Dict[str, Any]:
        """Get summary of monitoring job."""
        uptime = self.get_uptime()
        return {
            'job_id': str(self.job_id),
            'stream_id': str(self.stream_id),
            'job_name': self.job_name,
            'status': self.status.value,
            'uptime_seconds': uptime.total_seconds() if uptime else 0,
            'windows_processed': self.windows_processed,
            'total_records_processed': self.total_records_processed,
            'current_quality_score': self.get_current_quality_score(),
            'active_alerts': self.get_active_alert_count(),
            'critical_alerts': self.get_critical_alert_count(),
            'error_count': self.error_count,
            'health_score': self.streaming_metrics.get_health_score(),
            'throughput': self.streaming_metrics.throughput_records_per_second,
            'latency_ms': self.streaming_metrics.latency_ms
        }


@dataclass(frozen=True)
class DashboardWidget:
    """Dashboard widget configuration."""
    widget_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    widget_type: str = ""
    title: str = ""
    description: str = ""
    
    # Display configuration
    position_x: int = 0
    position_y: int = 0
    width: int = 4
    height: int = 3
    
    # Data configuration
    data_source: str = ""
    metrics: List[str] = field(default_factory=list)
    filters: Dict[str, Any] = field(default_factory=dict)
    
    # Refresh configuration
    refresh_interval_seconds: int = 5
    enable_real_time: bool = True
    
    # Visualization
    chart_type: str = "line"
    color_scheme: str = "default"
    show_legend: bool = True
    show_grid: bool = True


@dataclass(frozen=True)
class QualityDashboard:
    """Live quality dashboard configuration."""
    dashboard_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    
    # Layout
    widgets: List[DashboardWidget] = field(default_factory=list)
    layout_columns: int = 12
    
    # Refresh configuration
    refresh_mode: DashboardRefreshMode = DashboardRefreshMode.REAL_TIME
    refresh_interval_seconds: int = 5
    
    # Filters
    global_filters: Dict[str, Any] = field(default_factory=dict)
    time_range_hours: int = 24
    
    # Access control
    created_by: str = ""
    shared_with: List[str] = field(default_factory=list)
    is_public: bool = False
    
    def add_widget(self, widget: DashboardWidget) -> 'QualityDashboard':
        """Add a widget to the dashboard."""
        new_widgets = self.widgets + [widget]
        return QualityDashboard(
            dashboard_id=self.dashboard_id,
            name=self.name,
            description=self.description,
            widgets=new_widgets,
            layout_columns=self.layout_columns,
            refresh_mode=self.refresh_mode,
            refresh_interval_seconds=self.refresh_interval_seconds,
            global_filters=self.global_filters,
            time_range_hours=self.time_range_hours,
            created_by=self.created_by,
            shared_with=self.shared_with,
            is_public=self.is_public
        )
    
    def remove_widget(self, widget_id: str) -> 'QualityDashboard':
        """Remove a widget from the dashboard."""
        new_widgets = [w for w in self.widgets if w.widget_id != widget_id]
        return QualityDashboard(
            dashboard_id=self.dashboard_id,
            name=self.name,
            description=self.description,
            widgets=new_widgets,
            layout_columns=self.layout_columns,
            refresh_mode=self.refresh_mode,
            refresh_interval_seconds=self.refresh_interval_seconds,
            global_filters=self.global_filters,
            time_range_hours=self.time_range_hours,
            created_by=self.created_by,
            shared_with=self.shared_with,
            is_public=self.is_public
        )
    
    def update_widget(self, widget: DashboardWidget) -> 'QualityDashboard':
        """Update a widget in the dashboard."""
        new_widgets = []
        for existing_widget in self.widgets:
            if existing_widget.widget_id == widget.widget_id:
                new_widgets.append(widget)
            else:
                new_widgets.append(existing_widget)
        
        return QualityDashboard(
            dashboard_id=self.dashboard_id,
            name=self.name,
            description=self.description,
            widgets=new_widgets,
            layout_columns=self.layout_columns,
            refresh_mode=self.refresh_mode,
            refresh_interval_seconds=self.refresh_interval_seconds,
            global_filters=self.global_filters,
            time_range_hours=self.time_range_hours,
            created_by=self.created_by,
            shared_with=self.shared_with,
            is_public=self.is_public
        )