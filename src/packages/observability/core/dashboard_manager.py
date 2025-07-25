"""
Advanced Dashboard Management System

Provides comprehensive dashboard creation, management, and visualization
capabilities with Grafana integration and custom visualizations.
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import requests
import aiohttp

logger = logging.getLogger(__name__)


class DashboardType(Enum):
    """Types of dashboards."""
    SYSTEM_OVERVIEW = "system_overview"
    ML_MODEL_MONITORING = "ml_model_monitoring"
    BUSINESS_METRICS = "business_metrics"
    INFRASTRUCTURE = "infrastructure"
    SECURITY = "security"
    CUSTOM = "custom"


class VisualizationType(Enum):
    """Types of visualizations."""
    TIME_SERIES = "time_series"
    BAR_CHART = "bar_chart"
    PIE_CHART = "pie_chart"
    HEAT_MAP = "heat_map"
    GAUGE = "gauge"
    STAT = "stat"
    TABLE = "table"
    LOGS = "logs"
    ALERT_LIST = "alert_list"


class UserRole(Enum):
    """User roles for dashboard access."""
    ADMIN = "admin"
    DEVELOPER = "developer"
    DATA_SCIENTIST = "data_scientist"
    BUSINESS_ANALYST = "business_analyst"
    VIEWER = "viewer"


@dataclass
class Panel:
    """Dashboard panel configuration."""
    id: str
    title: str
    type: VisualizationType
    queries: List[Dict[str, Any]]
    position: Dict[str, int]  # x, y, width, height
    options: Dict[str, Any] = field(default_factory=dict)
    field_config: Dict[str, Any] = field(default_factory=dict)
    transformations: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class Dashboard:
    """Dashboard configuration."""
    id: str
    title: str
    description: str
    dashboard_type: DashboardType
    panels: List[Panel]
    tags: List[str] = field(default_factory=list)
    refresh_interval: str = "30s"
    time_range: Dict[str, str] = field(default_factory=lambda: {"from": "now-1h", "to": "now"})
    variables: List[Dict[str, Any]] = field(default_factory=list)
    permissions: Dict[UserRole, str] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class AlertRule:
    """Alert rule for dashboard panels."""
    id: str
    panel_id: str
    name: str
    condition: Dict[str, Any]
    frequency: str
    notifications: List[str]
    message: str


class DashboardManager:
    """
    Advanced dashboard management system with Grafana integration,
    role-based access control, and automated dashboard generation.
    """
    
    def __init__(
        self,
        grafana_url: str = "http://localhost:3000",
        grafana_api_key: Optional[str] = None,
        grafana_username: Optional[str] = None,
        grafana_password: Optional[str] = None,
        enable_custom_dashboards: bool = True
    ):
        self.grafana_url = grafana_url.rstrip('/')
        self.grafana_api_key = grafana_api_key
        self.grafana_username = grafana_username
        self.grafana_password = grafana_password
        self.enable_custom_dashboards = enable_custom_dashboards
        
        # Dashboard storage
        self.dashboards: Dict[str, Dashboard] = {}
        self.alert_rules: Dict[str, AlertRule] = {}
        
        # Custom visualization providers
        self.custom_visualizers: Dict[str, Callable] = {}
        
        # Template dashboards
        self.dashboard_templates: Dict[DashboardType, Callable] = {}
        
        # Setup HTTP session
        self.session = None
        
        # Initialize default templates
        self._setup_dashboard_templates()
    
    async def initialize(self) -> None:
        """Initialize the dashboard manager."""
        try:
            # Setup HTTP session
            self.session = aiohttp.ClientSession()
            
            # Test Grafana connection
            await self._test_grafana_connection()
            
            # Create default dashboards
            await self._create_default_dashboards()
            
            logger.info("Dashboard manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize dashboard manager: {e}")
            raise
    
    async def _test_grafana_connection(self) -> None:
        """Test connection to Grafana."""
        try:
            headers = self._get_auth_headers()
            
            async with self.session.get(
                f"{self.grafana_url}/api/health",
                headers=headers
            ) as response:
                if response.status == 200:
                    health_data = await response.json()
                    logger.info(f"Grafana connection successful: {health_data}")
                else:
                    logger.warning(f"Grafana health check returned {response.status}")
                    
        except Exception as e:
            logger.error(f"Failed to connect to Grafana: {e}")
            # Continue without Grafana if connection fails
    
    def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers for Grafana API."""
        headers = {"Content-Type": "application/json"}
        
        if self.grafana_api_key:
            headers["Authorization"] = f"Bearer {self.grafana_api_key}"
        elif self.grafana_username and self.grafana_password:
            import base64
            credentials = base64.b64encode(
                f"{self.grafana_username}:{self.grafana_password}".encode()
            ).decode()
            headers["Authorization"] = f"Basic {credentials}"
        
        return headers
    
    def _setup_dashboard_templates(self) -> None:
        """Setup default dashboard templates."""
        self.dashboard_templates = {
            DashboardType.SYSTEM_OVERVIEW: self._create_system_overview_dashboard,
            DashboardType.ML_MODEL_MONITORING: self._create_ml_monitoring_dashboard,
            DashboardType.BUSINESS_METRICS: self._create_business_metrics_dashboard,
            DashboardType.INFRASTRUCTURE: self._create_infrastructure_dashboard,
            DashboardType.SECURITY: self._create_security_dashboard
        }
    
    async def create_dashboard(
        self,
        dashboard_type: DashboardType,
        title: str,
        description: str = "",
        custom_config: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a new dashboard."""
        try:
            # Generate dashboard ID
            dashboard_id = f"{dashboard_type.value}_{len(self.dashboards)}"
            
            # Get template function
            template_func = self.dashboard_templates.get(dashboard_type)
            if not template_func:
                raise ValueError(f"No template found for dashboard type: {dashboard_type}")
            
            # Create dashboard from template
            dashboard = template_func(dashboard_id, title, description, custom_config)
            
            # Store dashboard
            self.dashboards[dashboard_id] = dashboard
            
            # Create in Grafana if available
            await self._create_grafana_dashboard(dashboard)
            
            logger.info(f"Created dashboard: {title} ({dashboard_id})")
            return dashboard_id
            
        except Exception as e:
            logger.error(f"Failed to create dashboard: {e}")
            raise
    
    def _create_system_overview_dashboard(
        self,
        dashboard_id: str,
        title: str,
        description: str,
        custom_config: Optional[Dict[str, Any]] = None
    ) -> Dashboard:
        """Create system overview dashboard."""
        panels = [
            Panel(
                id="cpu_usage",
                title="CPU Usage",
                type=VisualizationType.TIME_SERIES,
                queries=[{
                    "expr": "system_cpu_usage_percent",
                    "legendFormat": "CPU Usage %",
                    "refId": "A"
                }],
                position={"x": 0, "y": 0, "width": 12, "height": 8},
                options={
                    "tooltip": {"mode": "multi"},
                    "legend": {"displayMode": "table"}
                }
            ),
            Panel(
                id="memory_usage",
                title="Memory Usage",
                type=VisualizationType.TIME_SERIES,
                queries=[{
                    "expr": "system_memory_usage_bytes",
                    "legendFormat": "Memory Usage",
                    "refId": "A"
                }],
                position={"x": 12, "y": 0, "width": 12, "height": 8}
            ),
            Panel(
                id="api_requests",
                title="API Requests",
                type=VisualizationType.STAT,
                queries=[{
                    "expr": "rate(business_api_requests_total[5m])",
                    "legendFormat": "Requests/sec",
                    "refId": "A"
                }],
                position={"x": 0, "y": 8, "width": 6, "height": 4}
            ),
            Panel(
                id="error_rate",
                title="Error Rate",
                type=VisualizationType.GAUGE,
                queries=[{
                    "expr": "rate(business_api_requests_total{status_code=~\"5..\"}[5m]) / rate(business_api_requests_total[5m]) * 100",
                    "legendFormat": "Error Rate %",
                    "refId": "A"
                }],
                position={"x": 6, "y": 8, "width": 6, "height": 4},
                options={
                    "reduceOptions": {"values": False, "calcs": ["lastNotNull"]},
                    "orientation": "auto",
                    "displayMode": "gradient"
                },
                field_config={
                    "defaults": {
                        "min": 0,
                        "max": 100,
                        "thresholds": {
                            "steps": [
                                {"color": "green", "value": 0},
                                {"color": "yellow", "value": 1},
                                {"color": "red", "value": 5}
                            ]
                        }
                    }
                }
            )
        ]
        
        return Dashboard(
            id=dashboard_id,
            title=title,
            description=description,
            dashboard_type=DashboardType.SYSTEM_OVERVIEW,
            panels=panels,
            tags=["system", "overview", "monitoring"],
            permissions={
                UserRole.ADMIN: "Admin",
                UserRole.DEVELOPER: "Edit",
                UserRole.VIEWER: "View"
            }
        )
    
    def _create_ml_monitoring_dashboard(
        self,
        dashboard_id: str,
        title: str,
        description: str,
        custom_config: Optional[Dict[str, Any]] = None
    ) -> Dashboard:
        """Create ML model monitoring dashboard."""
        model_name = custom_config.get("model_name", "default") if custom_config else "default"
        
        panels = [
            Panel(
                id="prediction_latency",
                title="Prediction Latency",
                type=VisualizationType.TIME_SERIES,
                queries=[{
                    "expr": f"ml_model_prediction_latency_seconds{{model_name=\"{model_name}\"}}",
                    "legendFormat": "Latency (seconds)",
                    "refId": "A"
                }],
                position={"x": 0, "y": 0, "width": 12, "height": 8}
            ),
            Panel(
                id="prediction_volume",
                title="Prediction Volume",
                type=VisualizationType.TIME_SERIES,
                queries=[{
                    "expr": f"rate(ml_model_prediction_total{{model_name=\"{model_name}\"}}[5m])",
                    "legendFormat": "Predictions/sec",
                    "refId": "A"
                }],
                position={"x": 12, "y": 0, "width": 12, "height": 8}
            ),
            Panel(
                id="model_accuracy",
                title="Model Accuracy",
                type=VisualizationType.GAUGE,
                queries=[{
                    "expr": f"ml_model_accuracy_score{{model_name=\"{model_name}\"}}",
                    "legendFormat": "Accuracy",
                    "refId": "A"
                }],
                position={"x": 0, "y": 8, "width": 8, "height": 6},
                field_config={
                    "defaults": {
                        "min": 0,
                        "max": 1,
                        "thresholds": {
                            "steps": [
                                {"color": "red", "value": 0},
                                {"color": "yellow", "value": 0.7},
                                {"color": "green", "value": 0.9}
                            ]
                        }
                    }
                }
            ),
            Panel(
                id="drift_score",
                title="Data Drift Score",
                type=VisualizationType.TIME_SERIES,
                queries=[{
                    "expr": f"ml_model_drift_score{{model_name=\"{model_name}\"}}",
                    "legendFormat": "Drift Score",
                    "refId": "A"
                }],
                position={"x": 8, "y": 8, "width": 16, "height": 6}
            ),
            Panel(
                id="prediction_distribution",
                title="Prediction Distribution",
                type=VisualizationType.HEAT_MAP,
                queries=[{
                    "expr": f"histogram_quantile(0.95, ml_model_prediction_latency_seconds_bucket{{model_name=\"{model_name}\"}})",
                    "legendFormat": "95th Percentile",
                    "refId": "A"
                }],
                position={"x": 0, "y": 14, "width": 24, "height": 8}
            )
        ]
        
        return Dashboard(
            id=dashboard_id,
            title=title,
            description=description,
            dashboard_type=DashboardType.ML_MODEL_MONITORING,
            panels=panels,
            tags=["ml", "model", "monitoring", model_name],
            variables=[{
                "name": "model_name",
                "type": "query",
                "query": "label_values(ml_model_prediction_total, model_name)",
                "current": {"value": model_name}
            }],
            permissions={
                UserRole.ADMIN: "Admin",
                UserRole.DATA_SCIENTIST: "Edit",
                UserRole.DEVELOPER: "View",
                UserRole.VIEWER: "View"
            }
        )
    
    def _create_business_metrics_dashboard(
        self,
        dashboard_id: str,
        title: str,
        description: str,
        custom_config: Optional[Dict[str, Any]] = None
    ) -> Dashboard:
        """Create business metrics dashboard."""
        panels = [
            Panel(
                id="revenue",
                title="Revenue",
                type=VisualizationType.STAT,
                queries=[{
                    "expr": "business_revenue_total",
                    "legendFormat": "Total Revenue",
                    "refId": "A"
                }],
                position={"x": 0, "y": 0, "width": 6, "height": 4}
            ),
            Panel(
                id="active_users",
                title="Active Users",
                type=VisualizationType.TIME_SERIES,
                queries=[{
                    "expr": "business_active_users",
                    "legendFormat": "Active Users",
                    "refId": "A"
                }],
                position={"x": 6, "y": 0, "width": 18, "height": 8}
            ),
            Panel(
                id="conversion_rate",
                title="Conversion Rate",
                type=VisualizationType.BAR_CHART,
                queries=[{
                    "expr": "business_conversion_rate",
                    "legendFormat": "{{funnel_stage}}",
                    "refId": "A"
                }],
                position={"x": 0, "y": 8, "width": 12, "height": 8}
            ),
            Panel(
                id="api_usage",
                title="API Usage by Endpoint",
                type=VisualizationType.PIE_CHART,
                queries=[{
                    "expr": "sum by (endpoint) (business_api_requests_total)",
                    "legendFormat": "{{endpoint}}",
                    "refId": "A"
                }],
                position={"x": 12, "y": 8, "width": 12, "height": 8}
            )
        ]
        
        return Dashboard(
            id=dashboard_id,
            title=title,
            description=description,
            dashboard_type=DashboardType.BUSINESS_METRICS,
            panels=panels,
            tags=["business", "metrics", "kpi"],
            permissions={
                UserRole.ADMIN: "Admin",
                UserRole.BUSINESS_ANALYST: "Edit",
                UserRole.DEVELOPER: "View",
                UserRole.VIEWER: "View"
            }
        )
    
    def _create_infrastructure_dashboard(
        self,
        dashboard_id: str,
        title: str,
        description: str,
        custom_config: Optional[Dict[str, Any]] = None
    ) -> Dashboard:
        """Create infrastructure monitoring dashboard."""
        panels = [
            Panel(
                id="kubernetes_pods",
                title="Kubernetes Pods Status",
                type=VisualizationType.TABLE,
                queries=[{
                    "expr": "kube_pod_info",
                    "legendFormat": "{{pod}} - {{phase}}",
                    "refId": "A"
                }],
                position={"x": 0, "y": 0, "width": 24, "height": 8}
            ),
            Panel(
                id="container_cpu",
                title="Container CPU Usage",
                type=VisualizationType.TIME_SERIES,
                queries=[{
                    "expr": "rate(container_cpu_usage_seconds_total[5m]) * 100",
                    "legendFormat": "{{container}}",
                    "refId": "A"
                }],
                position={"x": 0, "y": 8, "width": 12, "height": 8}
            ),
            Panel(
                id="container_memory",
                title="Container Memory Usage",
                type=VisualizationType.TIME_SERIES,
                queries=[{
                    "expr": "container_memory_usage_bytes / 1024 / 1024 / 1024",
                    "legendFormat": "{{container}}",
                    "refId": "A"
                }],
                position={"x": 12, "y": 8, "width": 12, "height": 8}
            ),
            Panel(
                id="network_io",
                title="Network I/O",
                type=VisualizationType.TIME_SERIES,
                queries=[
                    {
                        "expr": "rate(container_network_receive_bytes_total[5m])",
                        "legendFormat": "RX {{container}}",
                        "refId": "A"
                    },
                    {
                        "expr": "rate(container_network_transmit_bytes_total[5m])",
                        "legendFormat": "TX {{container}}",
                        "refId": "B"
                    }
                ],
                position={"x": 0, "y": 16, "width": 24, "height": 8}
            )
        ]
        
        return Dashboard(
            id=dashboard_id,
            title=title,
            description=description,
            dashboard_type=DashboardType.INFRASTRUCTURE,
            panels=panels,
            tags=["infrastructure", "kubernetes", "containers"],
            permissions={
                UserRole.ADMIN: "Admin",
                UserRole.DEVELOPER: "Edit",
                UserRole.VIEWER: "View"
            }
        )
    
    def _create_security_dashboard(
        self,
        dashboard_id: str,
        title: str,
        description: str,
        custom_config: Optional[Dict[str, Any]] = None
    ) -> Dashboard:
        """Create security monitoring dashboard."""
        panels = [
            Panel(
                id="failed_logins",
                title="Failed Login Attempts",
                type=VisualizationType.TIME_SERIES,
                queries=[{
                    "expr": "rate(security_failed_login_attempts_total[5m])",
                    "legendFormat": "Failed Logins/sec",
                    "refId": "A"
                }],
                position={"x": 0, "y": 0, "width": 12, "height": 8}
            ),
            Panel(
                id="security_alerts",
                title="Security Alerts",
                type=VisualizationType.ALERT_LIST,
                queries=[{
                    "expr": "security_alerts_total",
                    "legendFormat": "{{severity}}",
                    "refId": "A"
                }],
                position={"x": 12, "y": 0, "width": 12, "height": 8}
            ),
            Panel(
                id="vulnerability_count",
                title="Vulnerability Count",
                type=VisualizationType.STAT,
                queries=[{
                    "expr": "security_vulnerabilities_total",
                    "legendFormat": "Vulnerabilities",
                    "refId": "A"
                }],
                position={"x": 0, "y": 8, "width": 6, "height": 4}
            ),
            Panel(
                id="threat_detection",
                title="Threat Detection Events",
                type=VisualizationType.HEAT_MAP,
                queries=[{
                    "expr": "security_threat_events_total",
                    "legendFormat": "{{threat_type}}",
                    "refId": "A"
                }],
                position={"x": 6, "y": 8, "width": 18, "height": 8}
            )
        ]
        
        return Dashboard(
            id=dashboard_id,
            title=title,
            description=description,
            dashboard_type=DashboardType.SECURITY,
            panels=panels,
            tags=["security", "threats", "vulnerabilities"],
            permissions={
                UserRole.ADMIN: "Admin",
                UserRole.DEVELOPER: "View",
                UserRole.VIEWER: "View"
            }
        )
    
    async def _create_grafana_dashboard(self, dashboard: Dashboard) -> None:
        """Create dashboard in Grafana."""
        try:
            if not self.session:
                return
            
            # Convert to Grafana format
            grafana_dashboard = self._convert_to_grafana_format(dashboard)
            
            headers = self._get_auth_headers()
            
            # Create/update dashboard
            async with self.session.post(
                f"{self.grafana_url}/api/dashboards/db",
                headers=headers,
                json={"dashboard": grafana_dashboard, "overwrite": True}
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    logger.info(f"Created Grafana dashboard: {dashboard.title}")
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to create Grafana dashboard: {error_text}")
                    
        except Exception as e:
            logger.error(f"Error creating Grafana dashboard: {e}")
    
    def _convert_to_grafana_format(self, dashboard: Dashboard) -> Dict[str, Any]:
        """Convert internal dashboard format to Grafana format."""
        grafana_panels = []
        
        for panel in dashboard.panels:
            grafana_panel = {
                "id": len(grafana_panels) + 1,
                "title": panel.title,
                "type": self._get_grafana_panel_type(panel.type),
                "gridPos": {
                    "x": panel.position["x"],
                    "y": panel.position["y"],
                    "w": panel.position["width"],
                    "h": panel.position["height"]
                },
                "targets": self._convert_queries_to_grafana(panel.queries),
                "options": panel.options,
                "fieldConfig": panel.field_config
            }
            grafana_panels.append(grafana_panel)
        
        return {
            "id": None,  # Let Grafana assign ID
            "title": dashboard.title,
            "description": dashboard.description,
            "tags": dashboard.tags,
            "refresh": dashboard.refresh_interval,
            "time": dashboard.time_range,
            "templating": {
                "list": dashboard.variables
            },
            "panels": grafana_panels,
            "schemaVersion": 30,
            "version": 1
        }
    
    def _get_grafana_panel_type(self, panel_type: VisualizationType) -> str:
        """Convert internal panel type to Grafana panel type."""
        mapping = {
            VisualizationType.TIME_SERIES: "timeseries",
            VisualizationType.BAR_CHART: "barchart",
            VisualizationType.PIE_CHART: "piechart",
            VisualizationType.HEAT_MAP: "heatmap",
            VisualizationType.GAUGE: "gauge",
            VisualizationType.STAT: "stat",
            VisualizationType.TABLE: "table",
            VisualizationType.LOGS: "logs",
            VisualizationType.ALERT_LIST: "alertlist"
        }
        return mapping.get(panel_type, "timeseries")
    
    def _convert_queries_to_grafana(self, queries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert internal query format to Grafana targets."""
        grafana_targets = []
        
        for query in queries:
            target = {
                "expr": query.get("expr", ""),
                "legendFormat": query.get("legendFormat", ""),
                "refId": query.get("refId", "A"),
                "datasource": {"type": "prometheus", "uid": "prometheus"}
            }
            grafana_targets.append(target)
        
        return grafana_targets
    
    async def _create_default_dashboards(self) -> None:
        """Create default dashboards for the platform."""
        default_dashboards = [
            {
                "type": DashboardType.SYSTEM_OVERVIEW,
                "title": "MLOps System Overview",
                "description": "Overall system health and performance metrics"
            },
            {
                "type": DashboardType.INFRASTRUCTURE,
                "title": "Infrastructure Monitoring",
                "description": "Kubernetes and container infrastructure metrics"
            },
            {
                "type": DashboardType.BUSINESS_METRICS,
                "title": "Business KPIs",
                "description": "Business metrics and key performance indicators"
            }
        ]
        
        for dashboard_config in default_dashboards:
            try:
                await self.create_dashboard(
                    dashboard_config["type"],
                    dashboard_config["title"],
                    dashboard_config["description"]
                )
            except Exception as e:
                logger.error(f"Failed to create default dashboard {dashboard_config['title']}: {e}")
    
    async def get_dashboard(self, dashboard_id: str) -> Optional[Dashboard]:
        """Get dashboard by ID."""
        return self.dashboards.get(dashboard_id)
    
    async def list_dashboards(
        self,
        dashboard_type: Optional[DashboardType] = None,
        tags: Optional[List[str]] = None,
        user_role: Optional[UserRole] = None
    ) -> List[Dashboard]:
        """List dashboards with optional filtering."""
        dashboards = list(self.dashboards.values())
        
        # Filter by type
        if dashboard_type:
            dashboards = [d for d in dashboards if d.dashboard_type == dashboard_type]
        
        # Filter by tags
        if tags:
            dashboards = [d for d in dashboards if any(tag in d.tags for tag in tags)]
        
        # Filter by user permissions
        if user_role:
            dashboards = [d for d in dashboards if user_role in d.permissions]
        
        return dashboards
    
    async def update_dashboard(self, dashboard_id: str, updates: Dict[str, Any]) -> bool:
        """Update dashboard configuration."""
        dashboard = self.dashboards.get(dashboard_id)
        if not dashboard:
            return False
        
        # Update allowed fields
        for key, value in updates.items():
            if hasattr(dashboard, key):
                setattr(dashboard, key, value)
        
        dashboard.updated_at = datetime.utcnow()
        
        # Update in Grafana
        await self._create_grafana_dashboard(dashboard)
        
        return True
    
    async def delete_dashboard(self, dashboard_id: str) -> bool:
        """Delete dashboard."""
        if dashboard_id not in self.dashboards:
            return False
        
        dashboard = self.dashboards[dashboard_id]
        
        # Delete from Grafana
        try:
            if self.session:
                headers = self._get_auth_headers()
                async with self.session.delete(
                    f"{self.grafana_url}/api/dashboards/uid/{dashboard_id}",
                    headers=headers
                ) as response:
                    if response.status != 200:
                        logger.warning(f"Failed to delete Grafana dashboard: {response.status}")
        except Exception as e:
            logger.error(f"Error deleting Grafana dashboard: {e}")
        
        # Delete locally
        del self.dashboards[dashboard_id]
        
        return True
    
    async def create_alert_rule(self, alert_rule: AlertRule) -> str:
        """Create alert rule for dashboard panel."""
        self.alert_rules[alert_rule.id] = alert_rule
        
        # Create alert in Grafana
        await self._create_grafana_alert(alert_rule)
        
        return alert_rule.id
    
    async def _create_grafana_alert(self, alert_rule: AlertRule) -> None:
        """Create alert rule in Grafana."""
        try:
            if not self.session:
                return
            
            headers = self._get_auth_headers()
            
            alert_config = {
                "alert": {
                    "name": alert_rule.name,
                    "message": alert_rule.message,
                    "frequency": alert_rule.frequency,
                    "conditions": [alert_rule.condition],
                    "notifications": alert_rule.notifications
                }
            }
            
            async with self.session.post(
                f"{self.grafana_url}/api/alerts",
                headers=headers,
                json=alert_config
            ) as response:
                if response.status == 200:
                    logger.info(f"Created Grafana alert: {alert_rule.name}")
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to create Grafana alert: {error_text}")
                    
        except Exception as e:
            logger.error(f"Error creating Grafana alert: {e}")
    
    def add_custom_visualizer(self, name: str, visualizer: Callable) -> None:
        """Add custom visualization provider."""
        self.custom_visualizers[name] = visualizer
    
    async def export_dashboard(self, dashboard_id: str) -> Optional[Dict[str, Any]]:
        """Export dashboard configuration."""
        dashboard = self.dashboards.get(dashboard_id)
        if not dashboard:
            return None
        
        return {
            "id": dashboard.id,
            "title": dashboard.title,
            "description": dashboard.description,
            "type": dashboard.dashboard_type.value,
            "panels": [
                {
                    "id": panel.id,
                    "title": panel.title,
                    "type": panel.type.value,
                    "queries": panel.queries,
                    "position": panel.position,
                    "options": panel.options,
                    "field_config": panel.field_config
                }
                for panel in dashboard.panels
            ],
            "tags": dashboard.tags,
            "refresh_interval": dashboard.refresh_interval,
            "time_range": dashboard.time_range,
            "variables": dashboard.variables,
            "created_at": dashboard.created_at.isoformat(),
            "updated_at": dashboard.updated_at.isoformat()
        }
    
    async def import_dashboard(self, dashboard_config: Dict[str, Any]) -> str:
        """Import dashboard from configuration."""
        panels = []
        for panel_config in dashboard_config.get("panels", []):
            panel = Panel(
                id=panel_config["id"],
                title=panel_config["title"],
                type=VisualizationType(panel_config["type"]),
                queries=panel_config["queries"],
                position=panel_config["position"],
                options=panel_config.get("options", {}),
                field_config=panel_config.get("field_config", {})
            )
            panels.append(panel)
        
        dashboard = Dashboard(
            id=dashboard_config["id"],
            title=dashboard_config["title"],
            description=dashboard_config["description"],
            dashboard_type=DashboardType(dashboard_config["type"]),
            panels=panels,
            tags=dashboard_config.get("tags", []),
            refresh_interval=dashboard_config.get("refresh_interval", "30s"),
            time_range=dashboard_config.get("time_range", {"from": "now-1h", "to": "now"}),
            variables=dashboard_config.get("variables", [])
        )
        
        self.dashboards[dashboard.id] = dashboard
        await self._create_grafana_dashboard(dashboard)
        
        return dashboard.id
    
    async def get_dashboard_stats(self) -> Dict[str, Any]:
        """Get dashboard statistics."""
        stats = {
            "total_dashboards": len(self.dashboards),
            "total_alert_rules": len(self.alert_rules),
            "dashboard_types": {},
            "custom_visualizers": len(self.custom_visualizers),
            "grafana_connected": self.session is not None
        }
        
        # Count dashboards by type
        for dashboard in self.dashboards.values():
            dashboard_type = dashboard.dashboard_type.value
            stats["dashboard_types"][dashboard_type] = stats["dashboard_types"].get(dashboard_type, 0) + 1
        
        return stats
    
    async def close(self) -> None:
        """Close dashboard manager and cleanup resources."""
        if self.session:
            await self.session.close()
        
        logger.info("Dashboard manager closed")


# Dashboard template configurations
DASHBOARD_TEMPLATES = {
    "ml_model_monitoring": {
        "panels": [
            {"type": "time_series", "title": "Prediction Latency", "metrics": ["ml_model_prediction_latency_seconds"]},
            {"type": "gauge", "title": "Model Accuracy", "metrics": ["ml_model_accuracy_score"]},
            {"type": "time_series", "title": "Data Drift", "metrics": ["ml_model_drift_score"]},
            {"type": "stat", "title": "Predictions/Hour", "metrics": ["ml_model_prediction_total"]}
        ]
    },
    "system_overview": {
        "panels": [
            {"type": "time_series", "title": "CPU Usage", "metrics": ["system_cpu_usage_percent"]},
            {"type": "time_series", "title": "Memory Usage", "metrics": ["system_memory_usage_bytes"]},
            {"type": "stat", "title": "API Requests", "metrics": ["business_api_requests_total"]},
            {"type": "gauge", "title": "Error Rate", "metrics": ["error_rate"]}
        ]
    }
}