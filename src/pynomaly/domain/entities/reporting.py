"""
Reporting and business metrics domain entities.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Union

from pynomaly.shared.types import UserId, TenantId, DatasetId, DetectorId


class ReportType(str, Enum):
    """Types of reports available."""
    DETECTION_SUMMARY = "detection_summary"
    BUSINESS_METRICS = "business_metrics"
    PERFORMANCE_ANALYSIS = "performance_analysis"
    USAGE_ANALYTICS = "usage_analytics"
    COST_ANALYSIS = "cost_analysis"
    COMPLIANCE_REPORT = "compliance_report"
    TREND_ANALYSIS = "trend_analysis"
    CUSTOM_DASHBOARD = "custom_dashboard"


class MetricType(str, Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    RATE = "rate"
    PERCENTAGE = "percentage"
    CURRENCY = "currency"
    DURATION = "duration"


class ReportStatus(str, Enum):
    """Report generation status."""
    PENDING = "pending"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"


class TimeGranularity(str, Enum):
    """Time granularity for metrics."""
    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    QUARTER = "quarter"
    YEAR = "year"


@dataclass
class MetricValue:
    """A single metric value with metadata."""
    value: Union[int, float, str]
    timestamp: datetime
    metric_type: MetricType
    unit: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def format_value(self) -> str:
        """Format the value for display."""
        if self.metric_type == MetricType.CURRENCY:
            return f"${self.value:,.2f}"
        elif self.metric_type == MetricType.PERCENTAGE:
            return f"{self.value:.1f}%"
        elif self.metric_type == MetricType.DURATION:
            # Assume value is in seconds
            if self.value < 60:
                return f"{self.value:.1f}s"
            elif self.value < 3600:
                return f"{self.value/60:.1f}m"
            else:
                return f"{self.value/3600:.1f}h"
        elif self.metric_type == MetricType.COUNTER:
            return f"{int(self.value):,}"
        else:
            return f"{self.value} {self.unit}".strip()


@dataclass
class Metric:
    """A business metric with time series data."""
    id: str
    name: str
    description: str
    metric_type: MetricType
    values: List[MetricValue] = field(default_factory=list)
    tags: Dict[str, str] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def latest_value(self) -> Optional[MetricValue]:
        """Get the most recent metric value."""
        return max(self.values, key=lambda v: v.timestamp) if self.values else None
    
    @property
    def current_value(self) -> Union[int, float, str, None]:
        """Get the current metric value."""
        latest = self.latest_value
        return latest.value if latest else None
    
    def get_values_in_range(self, start: datetime, end: datetime) -> List[MetricValue]:
        """Get metric values within a time range."""
        return [v for v in self.values if start <= v.timestamp <= end]
    
    def add_value(self, value: Union[int, float, str], timestamp: Optional[datetime] = None, **metadata):
        """Add a new metric value."""
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        metric_value = MetricValue(
            value=value,
            timestamp=timestamp,
            metric_type=self.metric_type,
            metadata=metadata
        )
        self.values.append(metric_value)
        self.updated_at = datetime.utcnow()


@dataclass
class DetectionMetrics:
    """Metrics specific to anomaly detection operations."""
    total_detections: int = 0
    successful_detections: int = 0
    failed_detections: int = 0
    average_detection_time: float = 0.0
    anomalies_found: int = 0
    false_positives: int = 0
    true_positives: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    
    @property
    def success_rate(self) -> float:
        """Calculate detection success rate."""
        if self.total_detections == 0:
            return 0.0
        return (self.successful_detections / self.total_detections) * 100
    
    @property
    def anomaly_rate(self) -> float:
        """Calculate anomaly detection rate."""
        if self.successful_detections == 0:
            return 0.0
        return (self.anomalies_found / self.successful_detections) * 100


@dataclass
class BusinessMetrics:
    """High-level business metrics."""
    active_users: int = 0
    total_datasets: int = 0
    total_models: int = 0
    monthly_detections: int = 0
    cost_savings: float = 0.0
    revenue_impact: float = 0.0
    customer_satisfaction_score: float = 0.0
    time_to_insight: float = 0.0  # In hours
    
    def calculate_roi(self, investment: float) -> float:
        """Calculate return on investment."""
        if investment == 0:
            return 0.0
        total_benefit = self.cost_savings + self.revenue_impact
        return ((total_benefit - investment) / investment) * 100


@dataclass
class UsageMetrics:
    """System usage metrics."""
    api_calls_today: int = 0
    api_calls_this_month: int = 0
    storage_used_gb: float = 0.0
    compute_hours_used: float = 0.0
    bandwidth_used_gb: float = 0.0
    active_sessions: int = 0
    peak_concurrent_users: int = 0
    
    def calculate_api_rate(self, time_window_hours: int = 24) -> float:
        """Calculate API calls per hour."""
        if time_window_hours == 0:
            return 0.0
        return self.api_calls_today / time_window_hours


@dataclass
class ReportFilter:
    """Filters for report generation."""
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    tenant_ids: List[TenantId] = field(default_factory=list)
    user_ids: List[UserId] = field(default_factory=list)
    dataset_ids: List[DatasetId] = field(default_factory=list)
    detector_ids: List[DetectorId] = field(default_factory=list)
    tags: Dict[str, str] = field(default_factory=dict)
    granularity: TimeGranularity = TimeGranularity.DAY


@dataclass
class ReportSection:
    """A section within a report."""
    id: str
    title: str
    description: str
    metrics: List[Metric] = field(default_factory=list)
    charts: List[Dict[str, Any]] = field(default_factory=list)
    tables: List[Dict[str, Any]] = field(default_factory=list)
    insights: List[str] = field(default_factory=list)
    order: int = 0


@dataclass
class Report:
    """Complete business report."""
    id: str
    title: str
    description: str
    report_type: ReportType
    status: ReportStatus
    tenant_id: TenantId
    created_by: UserId
    filters: ReportFilter
    sections: List[ReportSection] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_section(self, section: ReportSection) -> None:
        """Add a section to the report."""
        self.sections.append(section)
        self.sections.sort(key=lambda s: s.order)
    
    def get_section_by_id(self, section_id: str) -> Optional[ReportSection]:
        """Get a specific section by ID."""
        return next((s for s in self.sections if s.id == section_id), None)
    
    def calculate_total_metrics(self) -> Dict[str, Any]:
        """Calculate summary metrics across all sections."""
        all_metrics = []
        for section in self.sections:
            all_metrics.extend(section.metrics)
        
        return {
            "total_metrics": len(all_metrics),
            "sections_count": len(self.sections),
            "generation_time": (
                (self.completed_at - self.created_at).total_seconds()
                if self.completed_at else None
            ),
            "status": self.status.value
        }


@dataclass
class Dashboard:
    """Business metrics dashboard."""
    id: str
    name: str
    description: str
    tenant_id: TenantId
    created_by: UserId
    widgets: List[Dict[str, Any]] = field(default_factory=list)
    layout: Dict[str, Any] = field(default_factory=dict)
    refresh_interval: int = 300  # seconds
    is_public: bool = False
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    last_accessed: Optional[datetime] = None
    
    def add_widget(self, widget_config: Dict[str, Any]) -> None:
        """Add a widget to the dashboard."""
        widget_config["id"] = str(uuid.uuid4())
        widget_config["created_at"] = datetime.utcnow().isoformat()
        self.widgets.append(widget_config)
        self.updated_at = datetime.utcnow()
    
    def remove_widget(self, widget_id: str) -> bool:
        """Remove a widget from the dashboard."""
        initial_count = len(self.widgets)
        self.widgets = [w for w in self.widgets if w.get("id") != widget_id]
        if len(self.widgets) < initial_count:
            self.updated_at = datetime.utcnow()
            return True
        return False
    
    def update_widget(self, widget_id: str, updates: Dict[str, Any]) -> bool:
        """Update a widget configuration."""
        for widget in self.widgets:
            if widget.get("id") == widget_id:
                widget.update(updates)
                widget["updated_at"] = datetime.utcnow().isoformat()
                self.updated_at = datetime.utcnow()
                return True
        return False


@dataclass
class Alert:
    """Business metric alert."""
    id: str
    name: str
    description: str
    metric_id: str
    tenant_id: TenantId
    condition: str  # e.g., "value > 100" or "change_percentage > 20"
    threshold: float
    is_active: bool = True
    notification_channels: List[str] = field(default_factory=list)
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def should_trigger(self, current_value: float, previous_value: Optional[float] = None) -> bool:
        """Check if alert should trigger based on current value."""
        if not self.is_active:
            return False
        
        # Parse condition and evaluate
        if ">" in self.condition:
            return current_value > self.threshold
        elif "<" in self.condition:
            return current_value < self.threshold
        elif "change_percentage" in self.condition and previous_value is not None:
            if previous_value == 0:
                return False
            change_pct = abs((current_value - previous_value) / previous_value) * 100
            return change_pct > self.threshold
        
        return False


# Predefined metric configurations
STANDARD_METRICS = {
    "detection_success_rate": {
        "name": "Detection Success Rate",
        "description": "Percentage of successful anomaly detections",
        "metric_type": MetricType.PERCENTAGE,
        "tags": {"category": "detection", "importance": "high"}
    },
    "anomaly_detection_rate": {
        "name": "Anomaly Detection Rate",
        "description": "Percentage of data points flagged as anomalies",
        "metric_type": MetricType.PERCENTAGE,
        "tags": {"category": "detection", "importance": "medium"}
    },
    "average_detection_time": {
        "name": "Average Detection Time",
        "description": "Average time to complete anomaly detection",
        "metric_type": MetricType.DURATION,
        "tags": {"category": "performance", "importance": "medium"}
    },
    "monthly_cost_savings": {
        "name": "Monthly Cost Savings",
        "description": "Estimated cost savings from anomaly detection",
        "metric_type": MetricType.CURRENCY,
        "tags": {"category": "business", "importance": "high"}
    },
    "active_users": {
        "name": "Active Users",
        "description": "Number of active users in the system",
        "metric_type": MetricType.COUNTER,
        "tags": {"category": "usage", "importance": "medium"}
    },
    "api_usage_rate": {
        "name": "API Usage Rate",
        "description": "API calls per hour",
        "metric_type": MetricType.RATE,
        "tags": {"category": "usage", "importance": "low"}
    }
}