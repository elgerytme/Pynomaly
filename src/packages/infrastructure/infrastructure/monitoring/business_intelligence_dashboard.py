#!/usr/bin/env python3
"""
Business Intelligence Dashboard for Advanced Monitoring

This module provides comprehensive business intelligence dashboards including:
- Executive summary dashboards with KPIs
- Technical performance monitoring
- SLO/SLA compliance tracking
- Capacity planning visualization
- Cost analysis and optimization insights
- User behavior analytics
- System health overview
"""

import asyncio
import json
import logging
import math
import statistics
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from plotly.subplots import make_subplots

from .advanced_monitoring import (
    AdvancedMonitoringOrchestrator,
    AlertSeverity,
)

logger = logging.getLogger(__name__)


class DashboardType(Enum):
    """Dashboard types for different audiences."""

    EXECUTIVE = "executive"
    TECHNICAL = "technical"
    BUSINESS = "business"
    OPERATIONS = "operations"
    SECURITY = "security"


class MetricTrend(Enum):
    """Metric trend indicators."""

    IMPROVING = "improving"
    STABLE = "stable"
    DEGRADING = "degrading"
    VOLATILE = "volatile"


@dataclass
class KPI:
    """Key Performance Indicator definition."""

    name: str
    value: float
    target: float
    unit: str
    trend: MetricTrend
    change_percentage: float
    description: str
    category: str
    priority: int = 1


@dataclass
class BusinessMetric:
    """Business-focused metric for executive dashboards."""

    name: str
    current_value: float
    previous_value: float
    target_value: float
    unit: str
    format_type: str  # percentage, currency, number, time
    trend_direction: str  # up, down, stable
    business_impact: str  # high, medium, low
    description: str


@dataclass
class DashboardWidget:
    """Dashboard widget configuration."""

    id: str
    title: str
    widget_type: str  # chart, metric, table, alert
    data_source: str
    config: dict[str, Any]
    position: dict[str, int]  # x, y, width, height
    refresh_interval: int = 30  # seconds


class BusinessIntelligenceDashboard:
    """Main business intelligence dashboard class."""

    def __init__(self, monitoring_orchestrator: AdvancedMonitoringOrchestrator):
        self.monitor = monitoring_orchestrator
        self.app = FastAPI(title="Pynomaly BI Dashboard")
        self.templates = Jinja2Templates(directory="templates")

        # Dashboard configurations
        self.dashboards: dict[str, list[DashboardWidget]] = {}
        self.kpis: dict[str, KPI] = {}
        self.business_metrics: dict[str, BusinessMetric] = {}

        # Historical data storage
        self.metric_history: dict[str, list[tuple[datetime, float]]] = {}
        self.alert_history: list[dict[str, Any]] = []

        # WebSocket connections for real-time updates
        self.active_connections: list[WebSocket] = []

        self._setup_routes()
        self._setup_default_dashboards()
        self._setup_default_kpis()

    def _setup_routes(self):
        """Setup FastAPI routes for the dashboard."""

        @self.app.get("/", response_class=HTMLResponse)
        async def dashboard_home(request: Request):
            return self.templates.TemplateResponse(
                "dashboard_home.html", {"request": request}
            )

        @self.app.get("/dashboard/{dashboard_type}", response_class=HTMLResponse)
        async def get_dashboard(request: Request, dashboard_type: str):
            if dashboard_type not in [dt.value for dt in DashboardType]:
                return JSONResponse(
                    {"error": "Invalid dashboard type"}, status_code=400
                )

            return self.templates.TemplateResponse(
                f"dashboard_{dashboard_type}.html",
                {"request": request, "dashboard_type": dashboard_type},
            )

        @self.app.get("/api/dashboard/{dashboard_type}/data")
        async def get_dashboard_data(dashboard_type: str):
            return await self.generate_dashboard_data(DashboardType(dashboard_type))

        @self.app.get("/api/kpis")
        async def get_kpis():
            return {
                "kpis": [
                    {
                        "name": kpi.name,
                        "value": kpi.value,
                        "target": kpi.target,
                        "unit": kpi.unit,
                        "trend": kpi.trend.value,
                        "change_percentage": kpi.change_percentage,
                        "description": kpi.description,
                        "category": kpi.category,
                        "priority": kpi.priority,
                    }
                    for kpi in self.kpis.values()
                ]
            }

        @self.app.get("/api/business-metrics")
        async def get_business_metrics():
            return {
                "metrics": [
                    {
                        "name": metric.name,
                        "current_value": metric.current_value,
                        "previous_value": metric.previous_value,
                        "target_value": metric.target_value,
                        "unit": metric.unit,
                        "format_type": metric.format_type,
                        "trend_direction": metric.trend_direction,
                        "business_impact": metric.business_impact,
                        "description": metric.description,
                    }
                    for metric in self.business_metrics.values()
                ]
            }

        @self.app.get("/api/capacity-report")
        async def get_capacity_report():
            return await self.monitor.generate_capacity_report()

        @self.app.get("/api/system-health")
        async def get_system_health():
            return self.monitor.get_monitoring_status()

        @self.app.get("/api/slo-compliance")
        async def get_slo_compliance():
            compliance_data = {}
            for slo_name in self.monitor.slo_monitor.slos:
                compliance = self.monitor.slo_monitor.calculate_slo_compliance(slo_name)
                compliance_data[slo_name] = compliance
            return compliance_data

        @self.app.get("/api/charts/{chart_type}")
        async def get_chart_data(chart_type: str, time_range: str = "24h"):
            return await self.generate_chart_data(chart_type, time_range)

        @self.app.websocket("/ws/realtime")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            self.active_connections.append(websocket)
            try:
                while True:
                    await websocket.receive_text()  # Keep connection alive
            except WebSocketDisconnect:
                self.active_connections.remove(websocket)

    def _setup_default_dashboards(self):
        """Setup default dashboard configurations."""

        # Executive Dashboard
        self.dashboards[DashboardType.EXECUTIVE.value] = [
            DashboardWidget(
                id="executive_kpis",
                title="Key Performance Indicators",
                widget_type="kpi_grid",
                data_source="kpis",
                config={"layout": "grid", "columns": 4},
                position={"x": 0, "y": 0, "width": 12, "height": 3},
            ),
            DashboardWidget(
                id="business_metrics",
                title="Business Metrics Overview",
                widget_type="metric_cards",
                data_source="business_metrics",
                config={"show_trends": True, "show_targets": True},
                position={"x": 0, "y": 3, "width": 8, "height": 4},
            ),
            DashboardWidget(
                id="slo_compliance",
                title="Service Level Objectives",
                widget_type="slo_chart",
                data_source="slo_compliance",
                config={"chart_type": "gauge", "show_burn_rate": True},
                position={"x": 8, "y": 3, "width": 4, "height": 4},
            ),
            DashboardWidget(
                id="cost_analysis",
                title="Cost Analysis & Optimization",
                widget_type="cost_chart",
                data_source="cost_metrics",
                config={"time_range": "30d", "show_forecast": True},
                position={"x": 0, "y": 7, "width": 6, "height": 4},
            ),
            DashboardWidget(
                id="capacity_overview",
                title="Capacity Planning Summary",
                widget_type="capacity_chart",
                data_source="capacity_report",
                config={"show_predictions": True, "forecast_days": 30},
                position={"x": 6, "y": 7, "width": 6, "height": 4},
            ),
        ]

        # Technical Dashboard
        self.dashboards[DashboardType.TECHNICAL.value] = [
            DashboardWidget(
                id="system_health",
                title="System Health Overview",
                widget_type="health_status",
                data_source="system_health",
                config={"show_details": True},
                position={"x": 0, "y": 0, "width": 4, "height": 3},
            ),
            DashboardWidget(
                id="performance_metrics",
                title="Performance Metrics",
                widget_type="time_series",
                data_source="performance_metrics",
                config={"metrics": ["cpu_usage", "memory_usage", "response_time"]},
                position={"x": 4, "y": 0, "width": 8, "height": 3},
            ),
            DashboardWidget(
                id="error_rates",
                title="Error Rates & Anomalies",
                widget_type="error_chart",
                data_source="error_metrics",
                config={"show_anomalies": True, "time_range": "24h"},
                position={"x": 0, "y": 3, "width": 6, "height": 4},
            ),
            DashboardWidget(
                id="distributed_tracing",
                title="Distributed Tracing",
                widget_type="trace_map",
                data_source="tracing_data",
                config={"show_latency": True, "show_errors": True},
                position={"x": 6, "y": 3, "width": 6, "height": 4},
            ),
            DashboardWidget(
                id="alert_timeline",
                title="Alert Timeline",
                widget_type="alert_timeline",
                data_source="alert_history",
                config={"time_range": "24h", "show_correlations": True},
                position={"x": 0, "y": 7, "width": 12, "height": 3},
            ),
        ]

        # Operations Dashboard
        self.dashboards[DashboardType.OPERATIONS.value] = [
            DashboardWidget(
                id="live_alerts",
                title="Live Alerts",
                widget_type="alert_list",
                data_source="active_alerts",
                config={"severity_filter": ["critical", "high"], "auto_refresh": True},
                position={"x": 0, "y": 0, "width": 6, "height": 4},
            ),
            DashboardWidget(
                id="resource_utilization",
                title="Resource Utilization",
                widget_type="resource_chart",
                data_source="resource_metrics",
                config={"resources": ["cpu", "memory", "disk", "network"]},
                position={"x": 6, "y": 0, "width": 6, "height": 4},
            ),
            DashboardWidget(
                id="deployment_status",
                title="Deployment Status",
                widget_type="deployment_chart",
                data_source="deployment_metrics",
                config={"show_rollback": True, "show_health": True},
                position={"x": 0, "y": 4, "width": 8, "height": 3},
            ),
            DashboardWidget(
                id="incident_management",
                title="Incident Management",
                widget_type="incident_board",
                data_source="incident_data",
                config={"show_timeline": True, "show_impact": True},
                position={"x": 8, "y": 4, "width": 4, "height": 3},
            ),
        ]

    def _setup_default_kpis(self):
        """Setup default KPIs for monitoring."""

        self.kpis["system_availability"] = KPI(
            name="System Availability",
            value=99.95,
            target=99.9,
            unit="%",
            trend=MetricTrend.STABLE,
            change_percentage=0.05,
            description="Overall system uptime and availability",
            category="reliability",
            priority=1,
        )

        self.kpis["api_response_time"] = KPI(
            name="API Response Time",
            value=245.0,
            target=500.0,
            unit="ms",
            trend=MetricTrend.IMPROVING,
            change_percentage=-12.5,
            description="Average API response time (P95)",
            category="performance",
            priority=1,
        )

        self.kpis["error_rate"] = KPI(
            name="Error Rate",
            value=0.12,
            target=1.0,
            unit="%",
            trend=MetricTrend.IMPROVING,
            change_percentage=-35.0,
            description="Application error rate",
            category="reliability",
            priority=1,
        )

        self.kpis["throughput"] = KPI(
            name="API Throughput",
            value=1250.0,
            target=1000.0,
            unit="req/min",
            trend=MetricTrend.IMPROVING,
            change_percentage=15.0,
            description="API request throughput",
            category="performance",
            priority=2,
        )

        self.kpis["user_satisfaction"] = KPI(
            name="User Satisfaction",
            value=4.2,
            target=4.0,
            unit="/5",
            trend=MetricTrend.STABLE,
            change_percentage=2.5,
            description="User satisfaction score",
            category="business",
            priority=2,
        )

        # Business metrics
        self.business_metrics["monthly_revenue"] = BusinessMetric(
            name="Monthly Revenue",
            current_value=125000.0,
            previous_value=118000.0,
            target_value=130000.0,
            unit="USD",
            format_type="currency",
            trend_direction="up",
            business_impact="high",
            description="Total monthly revenue from API usage",
        )

        self.business_metrics["active_users"] = BusinessMetric(
            name="Active Users",
            current_value=2850.0,
            previous_value=2720.0,
            target_value=3000.0,
            unit="users",
            format_type="number",
            trend_direction="up",
            business_impact="high",
            description="Monthly active users",
        )

        self.business_metrics["conversion_rate"] = BusinessMetric(
            name="Trial to Paid Conversion",
            current_value=18.5,
            previous_value=16.8,
            target_value=20.0,
            unit="%",
            format_type="percentage",
            trend_direction="up",
            business_impact="medium",
            description="Trial to paid subscription conversion rate",
        )

    async def generate_dashboard_data(
        self, dashboard_type: DashboardType
    ) -> dict[str, Any]:
        """Generate comprehensive dashboard data."""

        widgets = self.dashboards.get(dashboard_type.value, [])
        dashboard_data = {
            "dashboard_type": dashboard_type.value,
            "timestamp": datetime.now().isoformat(),
            "widgets": {},
            "global_stats": await self._get_global_stats(),
        }

        for widget in widgets:
            try:
                widget_data = await self._generate_widget_data(widget)
                dashboard_data["widgets"][widget.id] = widget_data
            except Exception as e:
                logger.error(f"Error generating data for widget {widget.id}: {e}")
                dashboard_data["widgets"][widget.id] = {"error": str(e)}

        return dashboard_data

    async def _get_global_stats(self) -> dict[str, Any]:
        """Get global statistics for the dashboard."""

        # System health
        health_status = self.monitor.get_monitoring_status()

        # SLO compliance
        slo_compliance = {}
        total_compliance = 0
        slo_count = 0

        for slo_name in self.monitor.slo_monitor.slos:
            compliance = self.monitor.slo_monitor.calculate_slo_compliance(slo_name)
            slo_compliance[slo_name] = compliance.get("compliance", 0)
            total_compliance += compliance.get("compliance", 0)
            slo_count += 1

        avg_compliance = total_compliance / slo_count if slo_count > 0 else 0

        # Active alerts
        active_alerts = len(
            [
                alert
                for alert in self.monitor.alert_correlator.alert_history
                if not alert.resolved
            ]
        )

        # Critical alerts
        critical_alerts = len(
            [
                alert
                for alert in self.monitor.alert_correlator.alert_history
                if not alert.resolved and alert.severity == AlertSeverity.CRITICAL
            ]
        )

        return {
            "overall_health": health_status.get("overall_status", "unknown"),
            "avg_slo_compliance": round(avg_compliance, 2),
            "active_alerts": active_alerts,
            "critical_alerts": critical_alerts,
            "components_healthy": len(
                [
                    comp
                    for comp in health_status.get("components", {}).values()
                    if comp.get("status") == "active"
                ]
            ),
            "last_updated": datetime.now().isoformat(),
        }

    async def _generate_widget_data(self, widget: DashboardWidget) -> dict[str, Any]:
        """Generate data for a specific widget."""

        widget_data = {
            "id": widget.id,
            "title": widget.title,
            "type": widget.widget_type,
            "timestamp": datetime.now().isoformat(),
            "config": widget.config,
        }

        if widget.data_source == "kpis":
            widget_data["data"] = await self._get_kpi_data()
        elif widget.data_source == "business_metrics":
            widget_data["data"] = await self._get_business_metrics_data()
        elif widget.data_source == "slo_compliance":
            widget_data["data"] = await self._get_slo_compliance_data()
        elif widget.data_source == "system_health":
            widget_data["data"] = self.monitor.get_monitoring_status()
        elif widget.data_source == "capacity_report":
            widget_data["data"] = await self.monitor.generate_capacity_report()
        elif widget.data_source == "performance_metrics":
            widget_data["data"] = await self._get_performance_metrics_data(
                widget.config
            )
        elif widget.data_source == "error_metrics":
            widget_data["data"] = await self._get_error_metrics_data(widget.config)
        elif widget.data_source == "alert_history":
            widget_data["data"] = await self._get_alert_history_data(widget.config)
        elif widget.data_source == "active_alerts":
            widget_data["data"] = await self._get_active_alerts_data(widget.config)
        elif widget.data_source == "cost_metrics":
            widget_data["data"] = await self._get_cost_metrics_data(widget.config)
        else:
            widget_data["data"] = {
                "message": f"Data source '{widget.data_source}' not implemented"
            }

        return widget_data

    async def _get_kpi_data(self) -> list[dict[str, Any]]:
        """Get KPI data for dashboard."""
        return [
            {
                "name": kpi.name,
                "value": kpi.value,
                "target": kpi.target,
                "unit": kpi.unit,
                "trend": kpi.trend.value,
                "change_percentage": kpi.change_percentage,
                "description": kpi.description,
                "category": kpi.category,
                "priority": kpi.priority,
                "status": "good"
                if kpi.value >= kpi.target
                else "warning"
                if kpi.value >= kpi.target * 0.9
                else "critical",
            }
            for kpi in sorted(self.kpis.values(), key=lambda x: x.priority)
        ]

    async def _get_business_metrics_data(self) -> list[dict[str, Any]]:
        """Get business metrics data."""
        return [
            {
                "name": metric.name,
                "current_value": metric.current_value,
                "previous_value": metric.previous_value,
                "target_value": metric.target_value,
                "unit": metric.unit,
                "format_type": metric.format_type,
                "trend_direction": metric.trend_direction,
                "business_impact": metric.business_impact,
                "description": metric.description,
                "change_value": metric.current_value - metric.previous_value,
                "change_percentage": (
                    (metric.current_value - metric.previous_value)
                    / metric.previous_value
                    * 100
                )
                if metric.previous_value > 0
                else 0,
                "target_progress": (metric.current_value / metric.target_value * 100)
                if metric.target_value > 0
                else 0,
            }
            for metric in self.business_metrics.values()
        ]

    async def _get_slo_compliance_data(self) -> dict[str, Any]:
        """Get SLO compliance data."""
        slo_data = {}

        for slo_name, slo in self.monitor.slo_monitor.slos.items():
            compliance = self.monitor.slo_monitor.calculate_slo_compliance(slo_name)
            burn_rate_alerts = self.monitor.slo_monitor.check_burn_rate_alerts(slo_name)

            slo_data[slo_name] = {
                "name": slo_name,
                "description": slo.description,
                "target_percentage": slo.target_percentage,
                "current_compliance": compliance.get("compliance", 0),
                "error_budget_remaining": compliance.get("error_budget_remaining", 0),
                "burn_rate": compliance.get("burn_rate", 0),
                "measurement_count": compliance.get("measurement_count", 0),
                "time_period": compliance.get("time_period", ""),
                "status": "good"
                if compliance.get("compliance", 0) >= slo.target_percentage
                else "bad",
                "burn_rate_alerts": len(burn_rate_alerts),
                "sli_type": slo.sli.type.value,
                "sli_target": slo.sli.target_value,
            }

        return slo_data

    async def _get_performance_metrics_data(
        self, config: dict[str, Any]
    ) -> dict[str, Any]:
        """Get performance metrics data for charts."""
        metrics = config.get("metrics", ["cpu_usage", "memory_usage", "response_time"])
        time_range = config.get("time_range", "24h")

        # Parse time range
        if time_range.endswith("h"):
            hours = int(time_range[:-1])
            start_time = datetime.now() - timedelta(hours=hours)
        elif time_range.endswith("d"):
            days = int(time_range[:-1])
            start_time = datetime.now() - timedelta(days=days)
        else:
            start_time = datetime.now() - timedelta(hours=24)

        performance_data = {}

        for metric in metrics:
            # Generate sample data (in a real implementation, this would query actual metrics)
            timestamps = []
            values = []

            current_time = start_time
            while current_time <= datetime.now():
                timestamps.append(current_time.isoformat())

                # Generate realistic sample data
                if metric == "cpu_usage":
                    base_value = 45 + 15 * math.sin(
                        (current_time.hour + current_time.minute / 60) * math.pi / 12
                    )
                    noise = np.random.normal(0, 5)
                    values.append(max(0, min(100, base_value + noise)))
                elif metric == "memory_usage":
                    base_value = 60 + 10 * math.sin(
                        (current_time.hour + current_time.minute / 60) * math.pi / 8
                    )
                    noise = np.random.normal(0, 3)
                    values.append(max(0, min(100, base_value + noise)))
                elif metric == "response_time":
                    base_value = 200 + 50 * math.sin(
                        (current_time.hour + current_time.minute / 60) * math.pi / 6
                    )
                    noise = np.random.normal(0, 20)
                    values.append(max(50, base_value + noise))

                current_time += timedelta(minutes=5)

            performance_data[metric] = {
                "timestamps": timestamps,
                "values": values,
                "unit": "%"
                if metric.endswith("_usage")
                else "ms"
                if metric == "response_time"
                else "",
                "current_value": values[-1] if values else 0,
                "avg_value": statistics.mean(values) if values else 0,
                "max_value": max(values) if values else 0,
                "min_value": min(values) if values else 0,
            }

        return performance_data

    async def _get_error_metrics_data(self, config: dict[str, Any]) -> dict[str, Any]:
        """Get error metrics and anomaly data."""
        time_range = config.get("time_range", "24h")
        show_anomalies = config.get("show_anomalies", True)

        # Generate sample error data
        timestamps = []
        error_rates = []
        anomalies = []

        start_time = datetime.now() - timedelta(hours=24)
        current_time = start_time

        while current_time <= datetime.now():
            timestamps.append(current_time.isoformat())

            # Generate realistic error rate data
            base_rate = 0.5 + 0.3 * math.sin(
                (current_time.hour + current_time.minute / 60) * math.pi / 12
            )

            # Add occasional spikes
            if np.random.random() < 0.05:  # 5% chance of spike
                spike = np.random.uniform(2.0, 5.0)
                error_rate = base_rate + spike
                anomalies.append(True)
            else:
                noise = np.random.normal(0, 0.1)
                error_rate = max(0, base_rate + noise)
                anomalies.append(False)

            error_rates.append(error_rate)
            current_time += timedelta(minutes=10)

        # Calculate error categories
        error_categories = {
            "4xx_errors": np.random.poisson(50, len(timestamps)).tolist(),
            "5xx_errors": np.random.poisson(10, len(timestamps)).tolist(),
            "timeout_errors": np.random.poisson(5, len(timestamps)).tolist(),
            "connection_errors": np.random.poisson(3, len(timestamps)).tolist(),
        }

        return {
            "error_rate": {
                "timestamps": timestamps,
                "values": error_rates,
                "anomalies": anomalies if show_anomalies else None,
                "unit": "%",
                "current_rate": error_rates[-1] if error_rates else 0,
                "avg_rate": statistics.mean(error_rates) if error_rates else 0,
                "anomaly_count": sum(anomalies) if show_anomalies else 0,
            },
            "error_categories": {
                category: {
                    "timestamps": timestamps,
                    "values": values,
                    "total_count": sum(values),
                }
                for category, values in error_categories.items()
            },
            "summary": {
                "total_errors": sum(
                    sum(values) for values in error_categories.values()
                ),
                "error_rate_trend": "increasing"
                if error_rates[-1] > error_rates[0]
                else "decreasing",
                "most_common_error": max(
                    error_categories.keys(), key=lambda k: sum(error_categories[k])
                ),
            },
        }

    async def _get_alert_history_data(self, config: dict[str, Any]) -> dict[str, Any]:
        """Get alert history data."""
        time_range = config.get("time_range", "24h")
        show_correlations = config.get("show_correlations", True)

        # Get alerts from monitoring system
        alerts = self.monitor.alert_correlator.alert_history

        # Filter by time range
        if time_range.endswith("h"):
            hours = int(time_range[:-1])
            cutoff_time = datetime.now() - timedelta(hours=hours)
        else:
            cutoff_time = datetime.now() - timedelta(hours=24)

        filtered_alerts = [alert for alert in alerts if alert.timestamp > cutoff_time]

        # Organize alerts by severity and time
        alert_timeline = []
        severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0, "info": 0}
        correlation_groups = {}

        for alert in filtered_alerts:
            alert_data = {
                "id": alert.id,
                "timestamp": alert.timestamp.isoformat(),
                "severity": alert.severity.value,
                "source": alert.source,
                "title": alert.title,
                "description": alert.description,
                "resolved": alert.resolved,
                "correlation_id": alert.correlation_id,
            }
            alert_timeline.append(alert_data)

            # Count by severity
            severity_counts[alert.severity.value] += 1

            # Group by correlation
            if show_correlations and alert.correlation_id:
                if alert.correlation_id not in correlation_groups:
                    correlation_groups[alert.correlation_id] = []
                correlation_groups[alert.correlation_id].append(alert_data)

        return {
            "timeline": sorted(
                alert_timeline, key=lambda x: x["timestamp"], reverse=True
            ),
            "severity_distribution": severity_counts,
            "correlation_groups": correlation_groups if show_correlations else {},
            "summary": {
                "total_alerts": len(filtered_alerts),
                "active_alerts": len([a for a in filtered_alerts if not a.resolved]),
                "resolved_alerts": len([a for a in filtered_alerts if a.resolved]),
                "correlation_groups_count": len(correlation_groups),
                "avg_resolution_time": "15m",  # Would calculate from actual data
            },
        }

    async def _get_active_alerts_data(self, config: dict[str, Any]) -> dict[str, Any]:
        """Get active alerts data."""
        severity_filter = config.get(
            "severity_filter", ["critical", "high", "medium", "low"]
        )

        active_alerts = [
            alert
            for alert in self.monitor.alert_correlator.alert_history
            if not alert.resolved and alert.severity.value in severity_filter
        ]

        alert_list = []
        for alert in active_alerts:
            alert_list.append(
                {
                    "id": alert.id,
                    "timestamp": alert.timestamp.isoformat(),
                    "severity": alert.severity.value,
                    "source": alert.source,
                    "title": alert.title,
                    "description": alert.description,
                    "labels": alert.labels,
                    "correlation_id": alert.correlation_id,
                    "age_minutes": int(
                        (datetime.now() - alert.timestamp).total_seconds() / 60
                    ),
                }
            )

        # Sort by severity and age
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        alert_list.sort(
            key=lambda x: (severity_order.get(x["severity"], 4), -x["age_minutes"])
        )

        return {
            "alerts": alert_list,
            "count_by_severity": {
                severity: len([a for a in alert_list if a["severity"] == severity])
                for severity in severity_filter
            },
            "oldest_alert_age": max([a["age_minutes"] for a in alert_list], default=0),
            "newest_alert_age": min([a["age_minutes"] for a in alert_list], default=0),
        }

    async def _get_cost_metrics_data(self, config: dict[str, Any]) -> dict[str, Any]:
        """Get cost analysis data."""
        time_range = config.get("time_range", "30d")
        show_forecast = config.get("show_forecast", True)

        # Generate sample cost data
        if time_range.endswith("d"):
            days = int(time_range[:-1])
        else:
            days = 30

        timestamps = []
        daily_costs = []

        for i in range(days):
            date = datetime.now() - timedelta(days=days - i - 1)
            timestamps.append(date.date().isoformat())

            # Generate realistic cost data
            base_cost = 850 + 100 * math.sin(i * math.pi / 15)  # Seasonal variation
            daily_variation = np.random.normal(0, 50)
            weekend_factor = 0.7 if date.weekday() >= 5 else 1.0
            daily_costs.append(max(0, (base_cost + daily_variation) * weekend_factor))

        # Cost breakdown by service
        cost_breakdown = {
            "compute": sum(daily_costs) * 0.45,
            "storage": sum(daily_costs) * 0.25,
            "network": sum(daily_costs) * 0.15,
            "monitoring": sum(daily_costs) * 0.08,
            "other": sum(daily_costs) * 0.07,
        }

        # Generate forecast if requested
        forecast_data = None
        if show_forecast:
            forecast_days = 7
            forecast_timestamps = []
            forecast_costs = []

            for i in range(forecast_days):
                date = datetime.now() + timedelta(days=i + 1)
                forecast_timestamps.append(date.date().isoformat())

                # Simple trend-based forecast
                trend = (
                    (daily_costs[-1] - daily_costs[-7]) / 7
                    if len(daily_costs) >= 7
                    else 0
                )
                forecast_cost = daily_costs[-1] + trend * (i + 1)
                forecast_costs.append(max(0, forecast_cost))

            forecast_data = {
                "timestamps": forecast_timestamps,
                "values": forecast_costs,
                "total_forecast": sum(forecast_costs),
            }

        return {
            "daily_costs": {
                "timestamps": timestamps,
                "values": daily_costs,
                "total": sum(daily_costs),
                "average": statistics.mean(daily_costs),
                "currency": "USD",
            },
            "cost_breakdown": cost_breakdown,
            "forecast": forecast_data,
            "optimization_opportunities": [
                {
                    "service": "compute",
                    "potential_savings": cost_breakdown["compute"] * 0.15,
                    "recommendation": "Right-size instances based on utilization patterns",
                },
                {
                    "service": "storage",
                    "potential_savings": cost_breakdown["storage"] * 0.20,
                    "recommendation": "Implement lifecycle policies for data archival",
                },
            ],
            "monthly_projection": sum(daily_costs) * (30 / days) if days > 0 else 0,
        }

    async def generate_chart_data(
        self, chart_type: str, time_range: str
    ) -> dict[str, Any]:
        """Generate data for specific chart types."""

        if chart_type == "performance_overview":
            return await self._generate_performance_overview_chart(time_range)
        elif chart_type == "slo_burn_rate":
            return await self._generate_slo_burn_rate_chart(time_range)
        elif chart_type == "capacity_utilization":
            return await self._generate_capacity_utilization_chart(time_range)
        elif chart_type == "error_distribution":
            return await self._generate_error_distribution_chart(time_range)
        elif chart_type == "business_kpis":
            return await self._generate_business_kpis_chart(time_range)
        else:
            return {"error": f"Chart type '{chart_type}' not supported"}

    async def _generate_performance_overview_chart(
        self, time_range: str
    ) -> dict[str, Any]:
        """Generate performance overview chart data."""

        # This would typically query actual performance metrics
        # For demo purposes, we'll generate realistic sample data

        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=("Response Time", "Throughput", "Error Rate", "CPU Usage"),
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
            ],
        )

        # Generate time series data
        timestamps = pd.date_range(start=f"-{time_range}", end="now", freq="5min")

        # Response time
        response_times = (
            200
            + 50 * np.sin(np.arange(len(timestamps)) * 0.1)
            + np.random.normal(0, 20, len(timestamps))
        )
        fig.add_trace(
            go.Scatter(x=timestamps, y=response_times, name="Response Time (ms)"),
            row=1,
            col=1,
        )

        # Throughput
        throughput = (
            1000
            + 200 * np.sin(np.arange(len(timestamps)) * 0.05)
            + np.random.normal(0, 50, len(timestamps))
        )
        fig.add_trace(
            go.Scatter(x=timestamps, y=throughput, name="Requests/min"), row=1, col=2
        )

        # Error rate
        error_rate = (
            0.5
            + 0.2 * np.sin(np.arange(len(timestamps)) * 0.02)
            + np.random.normal(0, 0.1, len(timestamps))
        )
        error_rate = np.maximum(0, error_rate)
        fig.add_trace(
            go.Scatter(x=timestamps, y=error_rate, name="Error Rate (%)"), row=2, col=1
        )

        # CPU usage
        cpu_usage = (
            45
            + 15 * np.sin(np.arange(len(timestamps)) * 0.08)
            + np.random.normal(0, 5, len(timestamps))
        )
        cpu_usage = np.clip(cpu_usage, 0, 100)
        fig.add_trace(
            go.Scatter(x=timestamps, y=cpu_usage, name="CPU Usage (%)"), row=2, col=2
        )

        fig.update_layout(
            height=600, showlegend=False, title_text="Performance Overview"
        )

        return {
            "chart_data": fig.to_dict(),
            "summary": {
                "avg_response_time": float(np.mean(response_times)),
                "avg_throughput": float(np.mean(throughput)),
                "avg_error_rate": float(np.mean(error_rate)),
                "avg_cpu_usage": float(np.mean(cpu_usage)),
            },
        }

    async def broadcast_realtime_update(self, data: dict[str, Any]):
        """Broadcast real-time updates to connected WebSocket clients."""
        if not self.active_connections:
            return

        message = json.dumps(
            {
                "type": "realtime_update",
                "timestamp": datetime.now().isoformat(),
                "data": data,
            }
        )

        # Send to all connected clients
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.warning(f"Failed to send update to client: {e}")
                disconnected.append(connection)

        # Remove disconnected clients
        for connection in disconnected:
            self.active_connections.remove(connection)

    async def update_kpi(
        self, kpi_name: str, value: float, trend: MetricTrend | None = None
    ):
        """Update a KPI value and broadcast to connected clients."""
        if kpi_name in self.kpis:
            old_value = self.kpis[kpi_name].value
            self.kpis[kpi_name].value = value

            if trend:
                self.kpis[kpi_name].trend = trend

            # Calculate change percentage
            if old_value > 0:
                change_pct = ((value - old_value) / old_value) * 100
                self.kpis[kpi_name].change_percentage = change_pct

            # Broadcast update
            await self.broadcast_realtime_update(
                {
                    "type": "kpi_update",
                    "kpi_name": kpi_name,
                    "value": value,
                    "change_percentage": self.kpis[kpi_name].change_percentage,
                    "trend": self.kpis[kpi_name].trend.value,
                }
            )

    def run(self, host: str = "0.0.0.0", port: int = 8080, debug: bool = False):
        """Run the dashboard server."""
        import uvicorn

        uvicorn.run(self.app, host=host, port=port, debug=debug)


# HTML Templates (would typically be in separate files)
DASHBOARD_TEMPLATES = {
    "dashboard_home.html": """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pynomaly Business Intelligence Dashboard</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        .dashboard-card { margin-bottom: 20px; }
        .kpi-card { text-align: center; padding: 20px; }
        .kpi-value { font-size: 2em; font-weight: bold; }
        .kpi-trend { font-size: 0.9em; }
        .trend-up { color: #28a745; }
        .trend-down { color: #dc3545; }
        .trend-stable { color: #6c757d; }
    </style>
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark bg-primary">
        <div class="container-fluid">
            <a class="navbar-brand" href="#">Pynomaly BI Dashboard</a>
            <div class="navbar-nav">
                <a class="nav-link" href="/dashboard/executive">Executive</a>
                <a class="nav-link" href="/dashboard/technical">Technical</a>
                <a class="nav-link" href="/dashboard/operations">Operations</a>
            </div>
        </div>
    </nav>

    <div class="container-fluid mt-4">
        <div class="row">
            <div class="col-md-12">
                <h1>Welcome to Pynomaly Business Intelligence Dashboard</h1>
                <p class="lead">Select a dashboard type to view detailed metrics and insights.</p>

                <div class="row">
                    <div class="col-md-4">
                        <div class="card dashboard-card">
                            <div class="card-body">
                                <h5 class="card-title">Executive Dashboard</h5>
                                <p class="card-text">High-level business metrics and KPIs for executives and stakeholders.</p>
                                <a href="/dashboard/executive" class="btn btn-primary">View Dashboard</a>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="card dashboard-card">
                            <div class="card-body">
                                <h5 class="card-title">Technical Dashboard</h5>
                                <p class="card-text">Detailed technical metrics, performance data, and system health.</p>
                                <a href="/dashboard/technical" class="btn btn-primary">View Dashboard</a>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="card dashboard-card">
                            <div class="card-body">
                                <h5 class="card-title">Operations Dashboard</h5>
                                <p class="card-text">Real-time operations data, alerts, and incident management.</p>
                                <a href="/dashboard/operations" class="btn btn-primary">View Dashboard</a>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
""",
    "dashboard_executive.html": """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Executive Dashboard - Pynomaly</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark bg-primary">
        <div class="container-fluid">
            <a class="navbar-brand" href="/">Pynomaly BI Dashboard</a>
            <span class="navbar-text">Executive Dashboard</span>
        </div>
    </nav>

    <div class="container-fluid mt-4">
        <div id="dashboard-content">
            <div class="text-center">
                <div class="spinner-border" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
                <p>Loading dashboard data...</p>
            </div>
        </div>
    </div>

    <script>
        // Load dashboard data
        $(document).ready(function() {
            loadDashboardData();
            setInterval(loadDashboardData, 30000); // Refresh every 30 seconds
        });

        function loadDashboardData() {
            $.get('/api/dashboard/executive/data', function(data) {
                renderDashboard(data);
            }).fail(function() {
                $('#dashboard-content').html('<div class="alert alert-danger">Failed to load dashboard data</div>');
            });
        }

        function renderDashboard(data) {
            let html = '<div class="row">';

            // Render KPIs
            if (data.widgets.executive_kpis) {
                html += renderKPIGrid(data.widgets.executive_kpis.data);
            }

            // Render business metrics
            if (data.widgets.business_metrics) {
                html += renderBusinessMetrics(data.widgets.business_metrics.data);
            }

            // Render SLO compliance
            if (data.widgets.slo_compliance) {
                html += renderSLOCompliance(data.widgets.slo_compliance.data);
            }

            html += '</div>';
            $('#dashboard-content').html(html);
        }

        function renderKPIGrid(kpis) {
            let html = '<div class="col-12"><h3>Key Performance Indicators</h3><div class="row">';

            kpis.forEach(function(kpi) {
                let statusClass = kpi.status === 'good' ? 'text-success' :
                                 kpi.status === 'warning' ? 'text-warning' : 'text-danger';
                let trendIcon = kpi.trend === 'improving' ? '↗️' :
                               kpi.trend === 'degrading' ? '↘️' : '➡️';

                html += `
                    <div class="col-md-3">
                        <div class="card">
                            <div class="card-body text-center">
                                <h6 class="card-title">${kpi.name}</h6>
                                <div class="kpi-value ${statusClass}">${kpi.value} ${kpi.unit}</div>
                                <div class="kpi-trend">
                                    ${trendIcon} ${kpi.change_percentage > 0 ? '+' : ''}${kpi.change_percentage.toFixed(1)}%
                                </div>
                                <small class="text-muted">Target: ${kpi.target} ${kpi.unit}</small>
                            </div>
                        </div>
                    </div>
                `;
            });

            html += '</div></div>';
            return html;
        }

        function renderBusinessMetrics(metrics) {
            let html = '<div class="col-8"><h3>Business Metrics</h3><div class="row">';

            metrics.forEach(function(metric) {
                let trendClass = metric.trend_direction === 'up' ? 'text-success' :
                                metric.trend_direction === 'down' ? 'text-danger' : 'text-muted';

                html += `
                    <div class="col-md-6">
                        <div class="card">
                            <div class="card-body">
                                <h6 class="card-title">${metric.name}</h6>
                                <div class="row">
                                    <div class="col-6">
                                        <div class="metric-value">
                                            ${formatValue(metric.current_value, metric.format_type, metric.unit)}
                                        </div>
                                        <small class="text-muted">Current</small>
                                    </div>
                                    <div class="col-6">
                                        <div class="${trendClass}">
                                            ${formatValue(metric.change_value, metric.format_type, metric.unit)}
                                        </div>
                                        <small class="text-muted">Change</small>
                                    </div>
                                </div>
                                <div class="progress mt-2">
                                    <div class="progress-bar" style="width: ${metric.target_progress}%"></div>
                                </div>
                                <small class="text-muted">
                                    Target: ${formatValue(metric.target_value, metric.format_type, metric.unit)}
                                </small>
                            </div>
                        </div>
                    </div>
                `;
            });

            html += '</div></div>';
            return html;
        }

        function renderSLOCompliance(slos) {
            let html = '<div class="col-4"><h3>SLO Compliance</h3>';

            Object.values(slos).forEach(function(slo) {
                let complianceClass = slo.status === 'good' ? 'text-success' : 'text-danger';

                html += `
                    <div class="card mb-2">
                        <div class="card-body">
                            <h6 class="card-title">${slo.name}</h6>
                            <div class="row">
                                <div class="col-6">
                                    <div class="${complianceClass}">
                                        ${slo.current_compliance.toFixed(2)}%
                                    </div>
                                    <small class="text-muted">Compliance</small>
                                </div>
                                <div class="col-6">
                                    <div>
                                        ${slo.error_budget_remaining.toFixed(1)}%
                                    </div>
                                    <small class="text-muted">Error Budget</small>
                                </div>
                            </div>
                        </div>
                    </div>
                `;
            });

            html += '</div>';
            return html;
        }

        function formatValue(value, type, unit) {
            if (type === 'currency') {
                return '$' + value.toLocaleString();
            } else if (type === 'percentage') {
                return value.toFixed(1) + '%';
            } else {
                return value.toLocaleString() + (unit ? ' ' + unit : '');
            }
        }
    </script>
</body>
</html>
""",
}


async def create_demo_dashboard():
    """Create a demo dashboard with sample data."""

    # Initialize monitoring orchestrator
    from .advanced_monitoring import AdvancedMonitoringOrchestrator

    monitor = AdvancedMonitoringOrchestrator()

    # Initialize dashboard
    dashboard = BusinessIntelligenceDashboard(monitor)

    # Simulate some data updates
    await dashboard.update_kpi("system_availability", 99.95, MetricTrend.STABLE)
    await dashboard.update_kpi("api_response_time", 245.0, MetricTrend.IMPROVING)
    await dashboard.update_kpi("error_rate", 0.12, MetricTrend.IMPROVING)

    return dashboard


if __name__ == "__main__":
    # Demo the dashboard
    async def main():
        dashboard = await create_demo_dashboard()

        # Generate sample dashboard data
        exec_data = await dashboard.generate_dashboard_data(DashboardType.EXECUTIVE)
        print("Executive Dashboard Data:")
        print(json.dumps(exec_data, indent=2, default=str))

        # Start the dashboard server (comment out for testing)
        # dashboard.run(debug=True)

    asyncio.run(main())
