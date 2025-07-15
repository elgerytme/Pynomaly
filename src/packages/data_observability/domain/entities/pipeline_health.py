"""
Pipeline Health Domain Entities

Defines the domain model for pipeline health monitoring, including metrics,
health status, and performance indicators.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Enum, Field


class PipelineStatus(str):
    """Status of a pipeline."""
    
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    FAILED = "failed"
    UNKNOWN = "unknown"
    MAINTENANCE = "maintenance"


class MetricType(str, Enum):
    """Types of pipeline metrics."""
    
    THROUGHPUT = "throughput"
    LATENCY = "latency"
    ERROR_RATE = "error_rate"
    RESOURCE_USAGE = "resource_usage"
    DATA_QUALITY = "data_quality"
    AVAILABILITY = "availability"
    PERFORMANCE = "performance"


class AlertSeverity(str, Enum):
    """Severity levels for alerts."""
    
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class MetricThreshold:
    """Threshold configuration for metrics."""
    
    warning_threshold: float
    critical_threshold: float
    comparison_operator: str = ">"  # >, <, >=, <=, ==, !=
    
    def evaluate(self, value: float) -> PipelineStatus:
        """Evaluate metric value against thresholds."""
        if self.comparison_operator == ">":
            if value > self.critical_threshold:
                return PipelineStatus.CRITICAL
            elif value > self.warning_threshold:
                return PipelineStatus.WARNING
        elif self.comparison_operator == "<":
            if value < self.critical_threshold:
                return PipelineStatus.CRITICAL
            elif value < self.warning_threshold:
                return PipelineStatus.WARNING
        elif self.comparison_operator == ">=":
            if value >= self.critical_threshold:
                return PipelineStatus.CRITICAL
            elif value >= self.warning_threshold:
                return PipelineStatus.WARNING
        elif self.comparison_operator == "<=":
            if value <= self.critical_threshold:
                return PipelineStatus.CRITICAL
            elif value <= self.warning_threshold:
                return PipelineStatus.WARNING
        
        return PipelineStatus.HEALTHY


class PipelineMetric(BaseModel):
    """Represents a single pipeline metric."""
    
    id: UUID = Field(default_factory=uuid4)
    pipeline_id: UUID = Field(..., description="ID of the pipeline")
    metric_type: MetricType = Field(..., description="Type of metric")
    name: str = Field(..., description="Name of the metric")
    value: float = Field(..., description="Current metric value")
    unit: str = Field(..., description="Unit of measurement")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Metadata
    labels: Dict[str, str] = Field(default_factory=dict)
    source: Optional[str] = None
    context: Dict[str, Any] = Field(default_factory=dict)
    
    # Threshold configuration
    threshold: Optional[MetricThreshold] = None        use_enum_values = True
    
    def get_status(self) -> PipelineStatus:
        """Get status based on threshold evaluation."""
        if self.threshold:
            return self.threshold.evaluate(self.value)
        return PipelineStatus.UNKNOWN
    
    def is_healthy(self) -> bool:
        """Check if metric is healthy."""
        return self.get_status() == PipelineStatus.HEALTHY


class PipelineAlert(BaseModel):
    """Represents a pipeline alert."""
    
    id: UUID = Field(default_factory=uuid4)
    pipeline_id: UUID = Field(..., description="ID of the pipeline")
    metric_id: Optional[UUID] = None
    severity: AlertSeverity = Field(..., description="Alert severity")
    title: str = Field(..., description="Alert title")
    description: str = Field(..., description="Alert description")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    resolved_at: Optional[datetime] = None
    
    # Alert details
    triggered_by: Optional[str] = None
    current_value: Optional[float] = None
    threshold_value: Optional[float] = None
    
    # Action tracking
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None        use_enum_values = True
    
    def acknowledge(self, acknowledged_by: str) -> None:
        """Acknowledge the alert."""
        self.acknowledged = True
        self.acknowledged_by = acknowledged_by
        self.acknowledged_at = datetime.utcnow()
    
    def resolve(self) -> None:
        """Resolve the alert."""
        self.resolved_at = datetime.utcnow()
    
    def is_active(self) -> bool:
        """Check if alert is still active."""
        return self.resolved_at is None


class PipelineHealth(BaseModel):
    """Represents the overall health of a pipeline."""
    
    id: UUID = Field(default_factory=uuid4)
    pipeline_id: UUID = Field(..., description="ID of the pipeline")
    pipeline_name: str = Field(..., description="Name of the pipeline")
    status: PipelineStatus = Field(..., description="Overall pipeline status")
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    
    # Health metrics
    availability_percentage: float = Field(default=100.0, ge=0, le=100)
    performance_score: float = Field(default=1.0, ge=0, le=1)
    error_rate: float = Field(default=0.0, ge=0)
    
    # Resource utilization
    cpu_usage: Optional[float] = None
    memory_usage: Optional[float] = None
    disk_usage: Optional[float] = None
    network_usage: Optional[float] = None
    
    # Data quality metrics
    data_quality_score: Optional[float] = None
    data_freshness: Optional[timedelta] = None
    data_completeness: Optional[float] = None
    
    # Execution metrics
    last_execution: Optional[datetime] = None
    execution_duration: Optional[timedelta] = None
    successful_executions: int = 0
    failed_executions: int = 0
    
    # Current metrics and alerts
    current_metrics: List[PipelineMetric] = Field(default_factory=list)
    active_alerts: List[PipelineAlert] = Field(default_factory=list)
    
    # Health history
    health_history: List[Dict[str, Any]] = Field(default_factory=list)        use_enum_values = True
    
    def update_status(self, new_status: PipelineStatus) -> None:
        """Update pipeline status and record in history."""
        old_status = self.status
        self.status = new_status
        self.last_updated = datetime.utcnow()
        
        # Record status change in history
        self.health_history.append({
            "timestamp": self.last_updated.isoformat(),
            "old_status": old_status,
            "new_status": new_status,
            "change_type": "status_update"
        })
    
    def add_metric(self, metric: PipelineMetric) -> None:
        """Add a metric to the pipeline health."""
        # Remove existing metric of same type if it exists
        self.current_metrics = [
            m for m in self.current_metrics 
            if not (m.metric_type == metric.metric_type and m.name == metric.name)
        ]
        
        self.current_metrics.append(metric)
        self.last_updated = datetime.utcnow()
        
        # Update overall status based on metric
        metric_status = metric.get_status()
        if metric_status == PipelineStatus.CRITICAL:
            self.update_status(PipelineStatus.CRITICAL)
        elif metric_status == PipelineStatus.WARNING and self.status == PipelineStatus.HEALTHY:
            self.update_status(PipelineStatus.WARNING)
    
    def add_alert(self, alert: PipelineAlert) -> None:
        """Add an alert to the pipeline health."""
        self.active_alerts.append(alert)
        self.last_updated = datetime.utcnow()
        
        # Update status based on alert severity
        if alert.severity == AlertSeverity.EMERGENCY:
            self.update_status(PipelineStatus.FAILED)
        elif alert.severity == AlertSeverity.CRITICAL:
            self.update_status(PipelineStatus.CRITICAL)
        elif alert.severity == AlertSeverity.WARNING and self.status == PipelineStatus.HEALTHY:
            self.update_status(PipelineStatus.WARNING)
    
    def resolve_alert(self, alert_id: UUID) -> bool:
        """Resolve an alert by ID."""
        for alert in self.active_alerts:
            if alert.id == alert_id:
                alert.resolve()
                self.active_alerts.remove(alert)
                self.last_updated = datetime.utcnow()
                
                # Recalculate status after alert resolution
                self._recalculate_status()
                return True
        return False
    
    def get_critical_alerts(self) -> List[PipelineAlert]:
        """Get all critical and emergency alerts."""
        return [
            alert for alert in self.active_alerts
            if alert.severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]
        ]
    
    def get_metric_by_type(self, metric_type: MetricType) -> Optional[PipelineMetric]:
        """Get the latest metric of a specific type."""
        metrics_of_type = [m for m in self.current_metrics if m.metric_type == metric_type]
        if metrics_of_type:
            return max(metrics_of_type, key=lambda m: m.timestamp)
        return None
    
    def calculate_uptime(self, period_hours: int = 24) -> float:
        """Calculate uptime percentage for a given period."""
        if not self.health_history:
            return self.availability_percentage
        
        cutoff_time = datetime.utcnow() - timedelta(hours=period_hours)
        
        # Get status changes in the period
        relevant_history = [
            entry for entry in self.health_history
            if datetime.fromisoformat(entry["timestamp"]) >= cutoff_time
        ]
        
        if not relevant_history:
            return self.availability_percentage
        
        # Calculate uptime based on status history
        total_time = timedelta(hours=period_hours)
        downtime = timedelta()
        
        current_status = self.status
        last_timestamp = datetime.utcnow()
        
        for entry in reversed(relevant_history):
            entry_time = datetime.fromisoformat(entry["timestamp"])
            
            if current_status in [PipelineStatus.FAILED, PipelineStatus.CRITICAL]:
                downtime += last_timestamp - entry_time
            
            current_status = entry["old_status"]
            last_timestamp = entry_time
        
        # Handle remaining time to cutoff
        if current_status in [PipelineStatus.FAILED, PipelineStatus.CRITICAL]:
            downtime += last_timestamp - cutoff_time
        
        uptime_percentage = max(0, (total_time - downtime) / total_time * 100)
        return min(100, uptime_percentage)
    
    def get_health_score(self) -> float:
        """Calculate overall health score (0-1)."""
        scores = []
        
        # Availability score
        availability_score = self.availability_percentage / 100.0
        scores.append(availability_score)
        
        # Performance score
        scores.append(self.performance_score)
        
        # Error rate score (inverse)
        error_score = max(0, 1.0 - self.error_rate)
        scores.append(error_score)
        
        # Data quality score
        if self.data_quality_score is not None:
            scores.append(self.data_quality_score)
        
        # Alert penalty
        critical_alerts = len(self.get_critical_alerts())
        alert_penalty = min(0.5, critical_alerts * 0.1)  # Max 50% penalty
        alert_score = max(0, 1.0 - alert_penalty)
        scores.append(alert_score)
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def _recalculate_status(self) -> None:
        """Recalculate pipeline status based on current metrics and alerts."""
        # Check for critical alerts
        if any(alert.severity == AlertSeverity.EMERGENCY for alert in self.active_alerts):
            self.update_status(PipelineStatus.FAILED)
            return
        
        if any(alert.severity == AlertSeverity.CRITICAL for alert in self.active_alerts):
            self.update_status(PipelineStatus.CRITICAL)
            return
        
        # Check metrics
        critical_metrics = [m for m in self.current_metrics if m.get_status() == PipelineStatus.CRITICAL]
        if critical_metrics:
            self.update_status(PipelineStatus.CRITICAL)
            return
        
        warning_metrics = [m for m in self.current_metrics if m.get_status() == PipelineStatus.WARNING]
        warning_alerts = [alert for alert in self.active_alerts if alert.severity == AlertSeverity.WARNING]
        
        if warning_metrics or warning_alerts:
            self.update_status(PipelineStatus.WARNING)
            return
        
        self.update_status(PipelineStatus.HEALTHY)
    
    def to_summary_dict(self) -> Dict[str, Any]:
        """Convert to summary dictionary for reporting."""
        return {
            "pipeline_id": str(self.pipeline_id),
            "pipeline_name": self.pipeline_name,
            "status": self.status,
            "health_score": self.get_health_score(),
            "availability": self.availability_percentage,
            "error_rate": self.error_rate,
            "active_alerts_count": len(self.active_alerts),
            "critical_alerts_count": len(self.get_critical_alerts()),
            "last_updated": self.last_updated.isoformat(),
            "last_execution": self.last_execution.isoformat() if self.last_execution else None,
            "data_quality_score": self.data_quality_score
        }