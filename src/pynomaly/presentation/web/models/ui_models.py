"""Pure UI models without FastAPI dependencies."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID


@dataclass
class DashboardData:
    """Dashboard data model."""
    detector_count: int
    dataset_count: int
    result_count: int
    recent_results: List['DetectionResult']
    current_user: Optional[str] = None
    auth_enabled: bool = False


@dataclass
class DetectionResult:
    """Detection result model."""
    id: UUID
    detector_id: UUID
    dataset_id: UUID
    timestamp: datetime
    n_anomalies: int
    n_samples: int
    anomaly_rate: float
    metrics: Dict[str, Any]
    
    
@dataclass
class DetectorInfo:
    """Detector information model."""
    id: UUID
    name: str
    algorithm: str
    description: str
    is_fitted: bool
    parameters: Dict[str, Any]
    created_at: datetime
    

@dataclass
class DatasetInfo:
    """Dataset information model."""
    id: UUID
    name: str
    description: str
    shape: tuple[int, int]
    columns: List[str]
    created_at: datetime


@dataclass
class UserInfo:
    """User information model."""
    username: str
    email: Optional[str]
    roles: List[str]
    permissions: List[str]
    is_active: bool
    last_login: Optional[datetime]


@dataclass
class WebSocketMessage:
    """WebSocket message model."""
    type: str
    data: Dict[str, Any]
    timestamp: datetime
    user_id: Optional[str] = None


@dataclass
class UINotification:
    """UI notification model."""
    id: str
    type: str  # success, error, warning, info
    title: str
    message: str
    timestamp: datetime
    auto_dismiss: bool = True
    duration_ms: int = 5000


@dataclass
class RouteGuard:
    """Route guard configuration."""
    requires_auth: bool = False
    required_roles: List[str] = None
    required_permissions: List[str] = None
    
    def __post_init__(self):
        if self.required_roles is None:
            self.required_roles = []
        if self.required_permissions is None:
            self.required_permissions = []


@dataclass
class MountConfig:
    """Mount configuration for UI components."""
    component_name: str
    mount_path: str
    static_files_path: str
    templates_path: str
    route_guards: Dict[str, RouteGuard]
    is_mounted: bool = False
    mount_timestamp: Optional[datetime] = None
    
    def mark_as_mounted(self) -> None:
        """Mark component as successfully mounted."""
        self.is_mounted = True
        self.mount_timestamp = datetime.utcnow()


@dataclass
class WebSocketConnectionInfo:
    """WebSocket connection information."""
    connection_id: str
    user_id: Optional[str]
    client_ip: str
    user_agent: str
    connected_at: datetime
    last_activity: datetime
    is_authenticated: bool = False


@dataclass
class UIPreferences:
    """User UI preferences."""
    theme: str = "light"  # light, dark, auto
    language: str = "en"
    timezone: str = "UTC"
    items_per_page: int = 20
    auto_refresh_interval: int = 30  # seconds
    show_notifications: bool = True
    compact_view: bool = False
