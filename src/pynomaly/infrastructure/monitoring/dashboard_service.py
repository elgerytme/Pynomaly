"""Dashboard service for creating and managing monitoring dashboards."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pynomaly.domain.models.monitoring import (
    Dashboard,
    DashboardType,
    DashboardWidget,
    MetricType,
    ServiceStatus,
)


class DashboardMetrics:
    """Metrics aggregation class for dashboard widgets."""
    
    def __init__(self):
        self.metrics_data: Dict[str, Any] = {}
        self.timestamp = datetime.utcnow()
    
    def add_metric(self, name: str, value: Any, unit: str = "", tags: Optional[Dict[str, str]] = None) -> None:
        """Add a metric to the dashboard metrics."""
        self.metrics_data[name] = {
            "value": value,
            "unit": unit,
            "tags": tags or {},
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def get_metric(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a specific metric."""
        return self.metrics_data.get(name)
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics."""
        return self.metrics_data.copy()


class DashboardService:
    """Service for creating and managing monitoring dashboards."""
    
    def __init__(self, metrics_service):
        self.metrics_service = metrics_service
        self.logger = logging.getLogger(__name__)
        
        # Dashboard storage
        self.dashboards: Dict[UUID, Dashboard] = {}
        
        # Pre-built dashboard templates
        self.dashboard_templates: Dict[DashboardType, Dict[str, Any]] = {}
        
        # Initialize built-in dashboards
        self._initialize_builtin_dashboards()
        
        self.logger.info("Dashboard service initialized")
    
    def _initialize_builtin_dashboards(self) -> None:
        """Initialize built-in dashboard templates."""
        
        # System overview dashboard
        self.dashboard_templates[DashboardType.SYSTEM_OVERVIEW] = {
            "name": "System Overview",
            "description": "High-level system health and performance metrics",
            "widgets": [
                {
                    "title": "System Health",
                    "widget_type": "gauge",
                    "metrics": ["system_health_score"],
                    "chart_type": "gauge",
                    "position": {"x": 0, "y": 0, "width": 3, "height": 3},
                    "thresholds": [
                        {"value": 95, "color": "green"},
                        {"value": 80, "color": "yellow"},
                        {"value": 0, "color": "red"},
                    ],
                },
                {
                    "title": "CPU Usage",
                    "widget_type": "chart",
                    "metrics": ["cpu_usage_percent"],
                    "chart_type": "line",
                    "time_range": "1h",
                    "position": {"x": 3, "y": 0, "width": 4, "height": 3},
                },
                {
                    "title": "Memory Usage",
                    "widget_type": "chart",
                    "metrics": ["memory_usage_percent"],
                    "chart_type": "line",
                    "time_range": "1h",
                    "position": {"x": 7, "y": 0, "width": 4, "height": 3},
                },
                {
                    "title": "Active Alerts",
                    "widget_type": "alert_list",
                    "position": {"x": 0, "y": 3, "width": 6, "height": 4},
                },
                {
                    "title": "Request Rate",
                    "widget_type": "chart",
                    "metrics": ["http_requests_total"],
                    "chart_type": "line",
                    "time_range": "1h",
                    "position": {"x": 6, "y": 3, "width": 6, "height": 4},
                },
            ],
        }
        
        # Application performance dashboard
        self.dashboard_templates[DashboardType.APPLICATION_PERFORMANCE] = {
            "name": "Application Performance",
            "description": "Application-specific performance metrics",
            "widgets": [
                {
                    "title": "Response Time P95",
                    "widget_type": "gauge",
                    "metrics": ["http_request_duration_seconds"],
                    "chart_type": "gauge",
                    "position": {"x": 0, "y": 0, "width": 3, "height": 3},
                },
                {
                    "title": "Request Rate",
                    "widget_type": "chart",
                    "metrics": ["http_requests_total"],
                    "chart_type": "line",
                    "time_range": "6h",
                    "position": {"x": 3, "y": 0, "width": 4, "height": 3},
                },
                {
                    "title": "Error Rate",
                    "widget_type": "chart",
                    "metrics": ["http_errors_total"],
                    "chart_type": "line",
                    "time_range": "6h",
                    "position": {"x": 7, "y": 0, "width": 4, "height": 3},
                },
                {
                    "title": "Response Time Distribution",
                    "widget_type": "chart",
                    "metrics": ["http_request_duration_seconds"],
                    "chart_type": "heatmap",
                    "time_range": "24h",
                    "position": {"x": 0, "y": 3, "width": 6, "height": 4},
                },
                {
                    "title": "Top Endpoints by Volume",
                    "widget_type": "table",
                    "metrics": ["http_requests_total"],
                    "position": {"x": 6, "y": 3, "width": 6, "height": 4},
                },
            ],
        }
        
        # ML model performance dashboard
        self.dashboard_templates[DashboardType.ML_MODEL_PERFORMANCE] = {
            "name": "ML Model Performance",
            "description": "Machine learning model metrics and performance",
            "widgets": [
                {
                    "title": "Model Accuracy",
                    "widget_type": "gauge",
                    "metrics": ["model_accuracy"],
                    "chart_type": "gauge",
                    "position": {"x": 0, "y": 0, "width": 3, "height": 3},
                    "thresholds": [
                        {"value": 0.95, "color": "green"},
                        {"value": 0.8, "color": "yellow"},
                        {"value": 0, "color": "red"},
                    ],
                },
                {
                    "title": "Predictions per Hour",
                    "widget_type": "chart",
                    "metrics": ["model_predictions_total"],
                    "chart_type": "line",
                    "time_range": "24h",
                    "position": {"x": 3, "y": 0, "width": 4, "height": 3},
                },
                {
                    "title": "Anomaly Detection Rate",
                    "widget_type": "chart",
                    "metrics": ["anomaly_detection_rate"],
                    "chart_type": "line",
                    "time_range": "24h",
                    "position": {"x": 7, "y": 0, "width": 4, "height": 3},
                },
                {
                    "title": "Model Inference Time",
                    "widget_type": "chart",
                    "metrics": ["model_inference_duration_seconds"],
                    "chart_type": "line",
                    "time_range": "6h",
                    "position": {"x": 0, "y": 3, "width": 6, "height": 4},
                },
                {
                    "title": "Anomalies Detected by Type",
                    "widget_type": "chart",
                    "metrics": ["anomalies_detected_total"],
                    "chart_type": "pie",
                    "time_range": "24h",
                    "position": {"x": 6, "y": 3, "width": 6, "height": 4},
                },
            ],
        }
        
        # Business metrics dashboard
        self.dashboard_templates[DashboardType.BUSINESS_METRICS] = {
            "name": "Business Metrics",
            "description": "Key business performance indicators",
            "widgets": [
                {
                    "title": "Total Anomalies Detected Today",
                    "widget_type": "text",
                    "metrics": ["anomalies_detected_total"],
                    "time_range": "24h",
                    "position": {"x": 0, "y": 0, "width": 3, "height": 2},
                },
                {
                    "title": "System Uptime",
                    "widget_type": "text",
                    "metrics": ["system_uptime"],
                    "position": {"x": 3, "y": 0, "width": 3, "height": 2},
                },
                {
                    "title": "Data Processing Volume",
                    "widget_type": "chart",
                    "metrics": ["data_samples_processed"],
                    "chart_type": "bar",
                    "time_range": "7d",
                    "position": {"x": 6, "y": 0, "width": 6, "height": 4},
                },
                {
                    "title": "Detection Accuracy Trend",
                    "widget_type": "chart",
                    "metrics": ["model_accuracy"],
                    "chart_type": "line",
                    "time_range": "30d",
                    "position": {"x": 0, "y": 2, "width": 6, "height": 4},
                },
            ],
        }
    
    async def create_dashboard(
        self,
        name: str,
        description: str,
        dashboard_type: DashboardType = DashboardType.CUSTOM,
        owner_id: Optional[UUID] = None,
        is_public: bool = False,
    ) -> Dashboard:
        """Create a new dashboard."""
        
        dashboard = Dashboard(
            dashboard_id=uuid4(),
            name=name,
            description=description,
            dashboard_type=dashboard_type,
            owner_id=owner_id or uuid4(),
            is_public=is_public,
        )
        
        # Apply template if it's a predefined type
        if dashboard_type in self.dashboard_templates:
            template = self.dashboard_templates[dashboard_type]
            
            # Override template name and description if provided
            if name != template.get("name", ""):
                dashboard.name = name
            else:
                dashboard.name = template["name"]
                
            if description != template.get("description", ""):
                dashboard.description = description
            else:
                dashboard.description = template["description"]
            
            # Add template widgets
            for widget_config in template.get("widgets", []):
                widget = DashboardWidget(
                    widget_id=uuid4(),
                    title=widget_config["title"],
                    widget_type=widget_config["widget_type"],
                    metrics=widget_config.get("metrics", []),
                    time_range=widget_config.get("time_range", "1h"),
                    position=widget_config.get("position", {}),
                    chart_type=widget_config.get("chart_type", "line"),
                    thresholds=widget_config.get("thresholds", []),
                )
                dashboard.add_widget(widget)
        
        self.dashboards[dashboard.dashboard_id] = dashboard
        
        self.logger.info(f"Created dashboard: {name} ({dashboard_type.value})")
        return dashboard
    
    async def get_dashboard_data(
        self,
        dashboard_id: UUID,
        user_id: UUID,
        time_range_override: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Get dashboard data with real-time metrics."""
        
        if dashboard_id not in self.dashboards:
            return None
        
        dashboard = self.dashboards[dashboard_id]
        
        # Check access permissions
        if not dashboard.can_view(user_id):
            return None
        
        dashboard_data = {
            "dashboard": {
                "id": str(dashboard.dashboard_id),
                "name": dashboard.name,
                "description": dashboard.description,
                "type": dashboard.dashboard_type.value,
                "theme": dashboard.theme,
                "auto_refresh": dashboard.auto_refresh,
                "refresh_interval": dashboard.refresh_interval,
                "last_updated": datetime.utcnow().isoformat(),
            },
            "widgets": [],
        }
        
        # Get data for each widget
        for widget in dashboard.widgets:
            widget_data = await self._get_widget_data(widget, time_range_override)
            dashboard_data["widgets"].append(widget_data)
        
        return dashboard_data
    
    async def _get_widget_data(
        self,
        widget: DashboardWidget,
        time_range_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get data for a specific widget."""
        
        time_range = time_range_override or widget.time_range
        
        widget_data = {
            "id": str(widget.widget_id),
            "title": widget.title,
            "type": widget.widget_type,
            "chart_type": widget.chart_type,
            "position": widget.position,
            "time_range": time_range,
            "data": {},
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        # Parse time range
        time_delta = self._parse_time_range(time_range)
        
        if widget.widget_type == "gauge":
            # Single value widgets
            widget_data["data"] = await self._get_gauge_data(widget, time_delta)
        
        elif widget.widget_type == "chart":
            # Time series data
            widget_data["data"] = await self._get_chart_data(widget, time_delta)
        
        elif widget.widget_type == "table":
            # Tabular data
            widget_data["data"] = await self._get_table_data(widget, time_delta)
        
        elif widget.widget_type == "text":
            # Text/number display
            widget_data["data"] = await self._get_text_data(widget, time_delta)
        
        elif widget.widget_type == "alert_list":
            # Active alerts
            widget_data["data"] = await self._get_alert_list_data()
        
        return widget_data
    
    async def _get_gauge_data(self, widget: DashboardWidget, time_delta: timedelta) -> Dict[str, Any]:
        """Get gauge widget data."""
        
        if not widget.metrics:
            return {"value": 0, "status": "no_data"}
        
        metric_name = widget.metrics[0]
        value = await self.metrics_service.get_metric_value(metric_name, "avg", time_delta)
        
        if value is None:
            return {"value": 0, "status": "no_data"}
        
        # Determine status based on thresholds
        status = "normal"
        for threshold in sorted(widget.thresholds, key=lambda t: t.get("value", 0), reverse=True):
            if isinstance(value, (int, float)) and value >= threshold.get("value", 0):
                status = threshold.get("color", "normal")
                break
        
        return {
            "value": value,
            "status": status,
            "unit": getattr(self.metrics_service.metrics.get(metric_name), 'unit', '') if metric_name in self.metrics_service.metrics else "",
        }
    
    def _parse_time_range(self, time_range: str) -> timedelta:
        """Parse time range string to timedelta."""
        
        time_map = {
            "5m": timedelta(minutes=5),
            "15m": timedelta(minutes=15),
            "30m": timedelta(minutes=30),
            "1h": timedelta(hours=1),
            "6h": timedelta(hours=6),
            "12h": timedelta(hours=12),
            "24h": timedelta(hours=24),
            "7d": timedelta(days=7),
            "30d": timedelta(days=30),
        }
        
        return time_map.get(time_range, timedelta(hours=1))
    
    async def get_service_health_dashboard_data(self) -> Dict[str, Any]:
        """Get service health dashboard data."""
        
        health_data = await self.metrics_service.get_service_health()
        metrics_summary = await self.metrics_service.get_metrics_summary()
        
        return {
            "service_health": health_data,
            "metrics_summary": metrics_summary,
            "timestamp": datetime.utcnow().isoformat(),
        }


# Alias for backward compatibility
MonitoringDashboardService = DashboardService
