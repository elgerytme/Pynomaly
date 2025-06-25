"""Domain entities for visualization dashboard framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4


class DashboardStatus(Enum):
    """Dashboard generation and update status."""

    DRAFT = "draft"
    GENERATING = "generating"
    READY = "ready"
    UPDATING = "updating"
    ERROR = "error"
    ARCHIVED = "archived"


class RefreshMode(Enum):
    """Dashboard refresh modes."""

    MANUAL = "manual"
    SCHEDULED = "scheduled"
    REAL_TIME = "real_time"
    ON_DEMAND = "on_demand"


class VisualizationType(Enum):
    """Types of visualizations supported."""

    CHART = "chart"
    TABLE = "table"
    METRIC = "metric"
    MAP = "map"
    GAUGE = "gauge"
    TREE = "tree"
    NETWORK = "network"
    HEATMAP = "heatmap"


class AlertSeverity(Enum):
    """Alert severity levels for dashboard notifications."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class DashboardTheme:
    """Dashboard theme and styling configuration."""

    theme_id: UUID = field(default_factory=uuid4)
    name: str = "default"
    description: str = ""
    primary_color: str = "#1890ff"
    secondary_color: str = "#52c41a"
    background_color: str = "#ffffff"
    text_color: str = "#000000"
    accent_color: str = "#faad14"
    error_color: str = "#ff4d4f"
    warning_color: str = "#faad14"
    success_color: str = "#52c41a"
    font_family: str = "Arial, sans-serif"
    font_size: int = 14
    border_radius: int = 4
    shadow_enabled: bool = True
    animation_enabled: bool = True
    custom_css: str | None = None

    def get_color_palette(self) -> dict[str, str]:
        """Get complete color palette for theme."""
        return {
            "primary": self.primary_color,
            "secondary": self.secondary_color,
            "background": self.background_color,
            "text": self.text_color,
            "accent": self.accent_color,
            "error": self.error_color,
            "warning": self.warning_color,
            "success": self.success_color,
        }


@dataclass
class VisualizationConfig:
    """Configuration for individual visualizations."""

    config_id: UUID = field(default_factory=uuid4)
    visualization_type: VisualizationType = VisualizationType.CHART
    title: str = ""
    subtitle: str = ""
    width: int | None = None
    height: int | None = None
    responsive: bool = True
    interactive: bool = True
    animation_enabled: bool = True
    legend_enabled: bool = True
    tooltip_enabled: bool = True
    zoom_enabled: bool = False
    brush_enabled: bool = False
    export_enabled: bool = True
    refresh_interval_seconds: int | None = None
    data_source: str = ""
    query_parameters: dict[str, Any] = field(default_factory=dict)
    styling_options: dict[str, Any] = field(default_factory=dict)
    custom_options: dict[str, Any] = field(default_factory=dict)

    def is_real_time(self) -> bool:
        """Check if visualization updates in real-time."""
        return (
            self.refresh_interval_seconds is not None
            and self.refresh_interval_seconds <= 60
        )


@dataclass
class DashboardAlert:
    """Dashboard alert and notification entity."""

    alert_id: UUID = field(default_factory=uuid4)
    dashboard_id: UUID = field(default_factory=uuid4)
    title: str = ""
    message: str = ""
    severity: AlertSeverity = AlertSeverity.INFO
    timestamp: datetime = field(default_factory=datetime.utcnow)
    source_component: str = ""
    metric_name: str | None = None
    threshold_value: float | None = None
    actual_value: float | None = None
    acknowledged: bool = False
    acknowledged_by: str | None = None
    acknowledged_at: datetime | None = None
    resolved: bool = False
    resolved_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def acknowledge(self, user_id: str) -> None:
        """Acknowledge the alert."""
        self.acknowledged = True
        self.acknowledged_by = user_id
        self.acknowledged_at = datetime.utcnow()

    def resolve(self) -> None:
        """Mark alert as resolved."""
        self.resolved = True
        self.resolved_at = datetime.utcnow()

    def is_active(self) -> bool:
        """Check if alert is still active."""
        return not self.resolved

    def get_alert_summary(self) -> dict[str, Any]:
        """Get alert summary."""
        return {
            "alert_id": str(self.alert_id),
            "title": self.title,
            "severity": self.severity.value,
            "timestamp": self.timestamp.isoformat(),
            "acknowledged": self.acknowledged,
            "resolved": self.resolved,
            "is_active": self.is_active(),
            "source": self.source_component,
        }


@dataclass
class DashboardLayout:
    """Dashboard layout and positioning configuration."""

    layout_id: UUID = field(default_factory=uuid4)
    layout_type: str = "grid"  # grid, flex, absolute, flow
    columns: int = 12
    rows: int | None = None
    gap: int = 16
    padding: int = 16
    responsive_breakpoints: dict[str, int] = field(
        default_factory=lambda: {
            "xs": 480,
            "sm": 768,
            "md": 1024,
            "lg": 1280,
            "xl": 1920,
        }
    )
    component_positions: list[dict[str, Any]] = field(default_factory=list)

    def add_component(
        self,
        component_id: str,
        x: int,
        y: int,
        width: int,
        height: int,
        min_width: int = 1,
        min_height: int = 1,
    ) -> None:
        """Add component to layout."""
        self.component_positions.append(
            {
                "component_id": component_id,
                "x": x,
                "y": y,
                "width": width,
                "height": height,
                "min_width": min_width,
                "min_height": min_height,
            }
        )

    def get_component_position(self, component_id: str) -> dict[str, Any] | None:
        """Get position of specific component."""
        for position in self.component_positions:
            if position["component_id"] == component_id:
                return position
        return None


@dataclass
class DashboardFilter:
    """Dashboard filter for data selection and refinement."""

    filter_id: UUID = field(default_factory=uuid4)
    name: str = ""
    filter_type: str = "select"  # select, range, date, text, multi_select
    field_name: str = ""
    label: str = ""
    options: list[Any] = field(default_factory=list)
    default_value: Any = None
    current_value: Any = None
    required: bool = False
    visible: bool = True
    enabled: bool = True
    refresh_dependent_components: list[str] = field(default_factory=list)

    def apply_filter(self, value: Any) -> None:
        """Apply filter value."""
        if (
            self.filter_type == "range"
            and isinstance(value, (list, tuple))
            and len(value) == 2
        ):
            self.current_value = value
        elif self.filter_type == "multi_select" and isinstance(value, list):
            self.current_value = value
        else:
            self.current_value = value

    def is_active(self) -> bool:
        """Check if filter has an active value."""
        return (
            self.current_value is not None and self.current_value != self.default_value
        )

    def get_filter_state(self) -> dict[str, Any]:
        """Get current filter state."""
        return {
            "filter_id": str(self.filter_id),
            "name": self.name,
            "type": self.filter_type,
            "current_value": self.current_value,
            "is_active": self.is_active(),
            "field_name": self.field_name,
        }


@dataclass
class DashboardPermission:
    """Dashboard access and permission control."""

    permission_id: UUID = field(default_factory=uuid4)
    dashboard_id: UUID = field(default_factory=uuid4)
    user_id: str | None = None
    role: str | None = None
    group: str | None = None
    can_view: bool = True
    can_edit: bool = False
    can_delete: bool = False
    can_export: bool = True
    can_share: bool = False
    can_manage_permissions: bool = False
    granted_by: str | None = None
    granted_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: datetime | None = None
    conditions: dict[str, Any] = field(default_factory=dict)

    def is_valid(self) -> bool:
        """Check if permission is still valid."""
        if self.expires_at and datetime.utcnow() > self.expires_at:
            return False
        return True

    def has_permission(self, action: str) -> bool:
        """Check if permission allows specific action."""
        if not self.is_valid():
            return False

        permission_map = {
            "view": self.can_view,
            "edit": self.can_edit,
            "delete": self.can_delete,
            "export": self.can_export,
            "share": self.can_share,
            "manage_permissions": self.can_manage_permissions,
        }

        return permission_map.get(action, False)


@dataclass
class DashboardVersion:
    """Dashboard version control and history."""

    version_id: UUID = field(default_factory=uuid4)
    dashboard_id: UUID = field(default_factory=uuid4)
    version_number: str = "1.0.0"
    title: str = ""
    description: str = ""
    created_by: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    is_current: bool = False
    parent_version_id: UUID | None = None
    changes_summary: str = ""
    configuration_snapshot: dict[str, Any] = field(default_factory=dict)
    size_bytes: int = 0

    def get_version_info(self) -> dict[str, Any]:
        """Get version information."""
        return {
            "version_id": str(self.version_id),
            "version_number": self.version_number,
            "title": self.title,
            "created_by": self.created_by,
            "created_at": self.created_at.isoformat(),
            "is_current": self.is_current,
            "size_bytes": self.size_bytes,
            "changes_summary": self.changes_summary,
        }


@dataclass
class Dashboard:
    """Main dashboard entity with comprehensive configuration and management."""

    dashboard_id: UUID = field(default_factory=uuid4)
    name: str = ""
    title: str = ""
    description: str = ""
    dashboard_type: str = (
        "analytical"  # executive, operational, analytical, performance, compliance
    )
    status: DashboardStatus = DashboardStatus.DRAFT
    created_by: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    last_accessed_at: datetime | None = None
    refresh_mode: RefreshMode = RefreshMode.MANUAL
    refresh_interval_seconds: int | None = None
    auto_refresh_enabled: bool = False

    # Layout and styling
    layout: DashboardLayout = field(default_factory=DashboardLayout)
    theme: DashboardTheme = field(default_factory=DashboardTheme)

    # Components and data
    visualizations: list[VisualizationConfig] = field(default_factory=list)
    filters: list[DashboardFilter] = field(default_factory=list)
    alerts: list[DashboardAlert] = field(default_factory=list)

    # Access control
    permissions: list[DashboardPermission] = field(default_factory=list)
    is_public: bool = False
    is_archived: bool = False

    # Version control
    current_version: str = "1.0.0"
    versions: list[DashboardVersion] = field(default_factory=list)

    # Configuration and metadata
    configuration: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)

    # Performance metrics
    load_time_seconds: float | None = None
    last_refresh_duration_seconds: float | None = None
    view_count: int = 0
    unique_viewers: set[str] = field(default_factory=set)

    def __post_init__(self):
        """Validate dashboard configuration."""
        if not self.name:
            self.name = f"Dashboard {str(self.dashboard_id)[:8]}"

        if not self.title:
            self.title = self.name

        # Ensure updated_at is current
        self.updated_at = datetime.utcnow()

    def add_visualization(
        self,
        visualization_config: VisualizationConfig,
        x: int = 0,
        y: int = 0,
        width: int = 6,
        height: int = 4,
    ) -> None:
        """Add visualization to dashboard."""
        self.visualizations.append(visualization_config)

        # Add to layout
        self.layout.add_component(
            component_id=str(visualization_config.config_id),
            x=x,
            y=y,
            width=width,
            height=height,
        )

        self.updated_at = datetime.utcnow()

    def add_filter(self, filter_config: DashboardFilter) -> None:
        """Add filter to dashboard."""
        self.filters.append(filter_config)
        self.updated_at = datetime.utcnow()

    def add_alert(self, alert: DashboardAlert) -> None:
        """Add alert to dashboard."""
        alert.dashboard_id = self.dashboard_id
        self.alerts.append(alert)

    def get_active_alerts(self) -> list[DashboardAlert]:
        """Get all active (unresolved) alerts."""
        return [alert for alert in self.alerts if alert.is_active()]

    def get_critical_alerts(self) -> list[DashboardAlert]:
        """Get critical severity alerts."""
        return [
            alert
            for alert in self.alerts
            if alert.severity == AlertSeverity.CRITICAL and alert.is_active()
        ]

    def add_permission(self, permission: DashboardPermission) -> None:
        """Add permission to dashboard."""
        permission.dashboard_id = self.dashboard_id
        self.permissions.append(permission)

    def can_user_access(self, user_id: str, action: str = "view") -> bool:
        """Check if user can perform action on dashboard."""
        if self.is_public and action == "view":
            return True

        for permission in self.permissions:
            if permission.user_id == user_id and permission.has_permission(action):
                return True

        return False

    def create_version(
        self, version_number: str, created_by: str, changes_summary: str = ""
    ) -> DashboardVersion:
        """Create new version of dashboard."""
        # Mark current version as not current
        for version in self.versions:
            version.is_current = False

        # Create new version
        new_version = DashboardVersion(
            dashboard_id=self.dashboard_id,
            version_number=version_number,
            title=self.title,
            description=self.description,
            created_by=created_by,
            is_current=True,
            changes_summary=changes_summary,
            configuration_snapshot={
                "visualizations": len(self.visualizations),
                "filters": len(self.filters),
                "layout": self.layout.layout_type,
                "theme": self.theme.name,
            },
        )

        self.versions.append(new_version)
        self.current_version = version_number
        self.updated_at = datetime.utcnow()

        return new_version

    def update_status(self, status: DashboardStatus) -> None:
        """Update dashboard status."""
        self.status = status
        self.updated_at = datetime.utcnow()

    def record_access(self, user_id: str) -> None:
        """Record dashboard access."""
        self.last_accessed_at = datetime.utcnow()
        self.view_count += 1
        self.unique_viewers.add(user_id)

    def archive(self) -> None:
        """Archive dashboard."""
        self.is_archived = True
        self.status = DashboardStatus.ARCHIVED
        self.updated_at = datetime.utcnow()

    def restore(self) -> None:
        """Restore archived dashboard."""
        self.is_archived = False
        self.status = DashboardStatus.READY
        self.updated_at = datetime.utcnow()

    def get_dashboard_summary(self) -> dict[str, Any]:
        """Get comprehensive dashboard summary."""
        return {
            "dashboard_id": str(self.dashboard_id),
            "name": self.name,
            "title": self.title,
            "type": self.dashboard_type,
            "status": self.status.value,
            "created_by": self.created_by,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "last_accessed": self.last_accessed_at.isoformat()
            if self.last_accessed_at
            else None,
            "current_version": self.current_version,
            "visualizations_count": len(self.visualizations),
            "filters_count": len(self.filters),
            "active_alerts": len(self.get_active_alerts()),
            "critical_alerts": len(self.get_critical_alerts()),
            "permissions_count": len(self.permissions),
            "is_public": self.is_public,
            "is_archived": self.is_archived,
            "view_count": self.view_count,
            "unique_viewers": len(self.unique_viewers),
            "tags": self.tags,
            "refresh_mode": self.refresh_mode.value,
            "auto_refresh": self.auto_refresh_enabled,
        }

    def get_performance_metrics(self) -> dict[str, Any]:
        """Get dashboard performance metrics."""
        return {
            "load_time_seconds": self.load_time_seconds,
            "last_refresh_duration": self.last_refresh_duration_seconds,
            "view_count": self.view_count,
            "unique_viewers": len(self.unique_viewers),
            "visualizations_count": len(self.visualizations),
            "average_component_size": sum(
                pos.get("width", 0) * pos.get("height", 0)
                for pos in self.layout.component_positions
            )
            / max(len(self.layout.component_positions), 1),
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert dashboard to dictionary for serialization."""
        return {
            "dashboard_id": str(self.dashboard_id),
            "name": self.name,
            "title": self.title,
            "description": self.description,
            "dashboard_type": self.dashboard_type,
            "status": self.status.value,
            "created_by": self.created_by,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "configuration": self.configuration,
            "metadata": self.metadata,
            "summary": self.get_dashboard_summary(),
            "performance": self.get_performance_metrics(),
        }
