"""Pure dashboard models without FastAPI dependencies."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID


class DashboardTab(Enum):
    """Dashboard tab enumeration."""
    OVERVIEW = "overview"
    DETECTORS = "detectors"
    DATASETS = "datasets"
    DETECTION = "detection"
    EXPERIMENTS = "experiments"
    ENSEMBLE = "ensemble"
    AUTOML = "automl"
    VISUALIZATIONS = "visualizations"
    MONITORING = "monitoring"
    EXPLAINABILITY = "explainability"


class ChartType(Enum):
    """Chart type enumeration."""
    LINE = "line"
    BAR = "bar"
    PIE = "pie"
    SCATTER = "scatter"
    HEATMAP = "heatmap"
    HISTOGRAM = "histogram"


@dataclass
class ChartConfig:
    """Chart configuration."""
    chart_type: ChartType
    title: str
    x_axis_label: str
    y_axis_label: str
    data_source: str
    refresh_interval: int = 30  # seconds
    height: int = 400
    width: Optional[int] = None
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DashboardWidget:
    """Dashboard widget configuration."""
    id: str
    type: str
    title: str
    position: Dict[str, int]  # x, y, width, height
    config: Dict[str, Any]
    is_visible: bool = True
    requires_auth: bool = False
    required_permissions: List[str] = field(default_factory=list)


@dataclass
class DashboardLayout:
    """Dashboard layout configuration."""
    layout_id: str
    name: str
    description: str
    widgets: List[DashboardWidget]
    columns: int = 12
    row_height: int = 100
    is_default: bool = False
    created_by: Optional[str] = None
    created_at: Optional[datetime] = None
    
    
@dataclass
class MetricCard:
    """Metric card data."""
    title: str
    value: Union[int, float, str]
    unit: Optional[str] = None
    change_percent: Optional[float] = None
    trend: Optional[str] = None  # up, down, neutral
    color: str = "blue"
    icon: Optional[str] = None


@dataclass
class TimeSeriesData:
    """Time series data point."""
    timestamp: datetime
    value: Union[int, float]
    label: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ActivityLog:
    """Activity log entry."""
    id: str
    user_id: Optional[str]
    action: str
    resource_type: str
    resource_id: Optional[str]
    description: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AlertConfig:
    """Alert configuration."""
    alert_id: str
    name: str
    condition: str
    threshold: float
    metric: str
    enabled: bool = True
    notification_channels: List[str] = field(default_factory=list)
    severity: str = "medium"  # low, medium, high, critical


@dataclass
class QuickAction:
    """Quick action configuration."""
    action_id: str
    title: str
    description: str
    icon: str
    route: str
    requires_auth: bool = False
    required_permissions: List[str] = field(default_factory=list)
    category: str = "general"


@dataclass
class DashboardStats:
    """Dashboard statistics."""
    total_detectors: int
    total_datasets: int
    total_experiments: int
    active_detections: int
    avg_anomaly_rate: float
    system_health: str
    last_updated: datetime
    uptime_seconds: int


@dataclass
class RecentActivity:
    """Recent activity summary."""
    detections_last_hour: int
    new_experiments_today: int
    trained_models_today: int
    alerts_today: int
    last_detection: Optional[datetime] = None
    last_training: Optional[datetime] = None


@dataclass
class SystemStatus:
    """System status information."""
    status: str  # healthy, warning, error
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    active_connections: int
    error_rate: float
    response_time_ms: float
    last_health_check: datetime


@dataclass
class UserSession:
    """User session information."""
    session_id: str
    user_id: str
    username: str
    login_time: datetime
    last_activity: datetime
    ip_address: str
    user_agent: str
    permissions: List[str]
    preferences: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NavigationItem:
    """Navigation item configuration."""
    id: str
    label: str
    route: str
    icon: Optional[str] = None
    badge_count: Optional[int] = None
    children: List['NavigationItem'] = field(default_factory=list)
    requires_auth: bool = False
    required_permissions: List[str] = field(default_factory=list)
    is_active: bool = False
    order: int = 0


@dataclass
class BreadcrumbItem:
    """Breadcrumb navigation item."""
    label: str
    route: Optional[str] = None
    is_current: bool = False


@dataclass
class PageMeta:
    """Page metadata."""
    title: str
    description: str
    keywords: List[str] = field(default_factory=list)
    og_image: Optional[str] = None
    canonical_url: Optional[str] = None
    breadcrumbs: List[BreadcrumbItem] = field(default_factory=list)
