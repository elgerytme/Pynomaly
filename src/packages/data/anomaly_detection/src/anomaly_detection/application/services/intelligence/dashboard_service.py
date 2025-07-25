"""Dashboard service for managing business intelligence dashboards and reporting."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Union
from dataclasses import dataclass, field
from enum import Enum
import threading
from pathlib import Path

from .analytics_engine import (
    AnalyticsEngine,
    AnalyticsQuery,
    MetricType,
    ChartType,
    AggregationType,
    Dashboard,
    DashboardWidget,
    get_analytics_engine
)
from ....infrastructure.logging import get_logger

logger = get_logger(__name__)

# Lazy import metrics collector to avoid None issues
def get_safe_metrics_collector():
    """Get metrics collector with safe fallback."""
    try:
        from ....infrastructure.monitoring import get_metrics_collector
        return get_metrics_collector()
    except Exception:
        class MockMetricsCollector:
            def record_metric(self, *args, **kwargs):
                pass
        return MockMetricsCollector()


class DashboardTemplate(Enum):
    """Pre-defined dashboard templates."""
    SYSTEM_OVERVIEW = "system_overview"
    ANOMALY_DETECTION = "anomaly_detection"
    PERFORMANCE_MONITORING = "performance_monitoring"
    SECURITY_MONITORING = "security_monitoring"
    BUSINESS_METRICS = "business_metrics"
    EXECUTIVE_SUMMARY = "executive_summary"


class ReportFormat(Enum):
    """Report export formats."""
    PDF = "pdf"
    HTML = "html"
    JSON = "json"
    CSV = "csv"
    EXCEL = "excel"


@dataclass
class ReportConfig:
    """Report generation configuration."""
    title: str
    description: str
    dashboard_ids: List[str]
    format: ReportFormat
    schedule: Optional[str] = None  # Cron expression
    recipients: List[str] = field(default_factory=list)
    filters: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class DashboardMetrics:
    """Dashboard usage metrics."""
    dashboard_id: str
    views: int = 0
    unique_viewers: Set[str] = field(default_factory=set)
    avg_load_time: float = 0.0
    last_viewed: Optional[datetime] = None
    errors: int = 0


class DashboardTemplateFactory:
    """Factory for creating dashboard templates."""
    
    @staticmethod
    def create_system_overview_dashboard() -> Dashboard:
        """Create system overview dashboard."""
        now = datetime.utcnow()
        time_range = (now - timedelta(hours=24), now)
        
        widgets = [
            # CPU Usage
            DashboardWidget(
                widget_id="cpu_usage",
                title="CPU Usage Over Time",
                chart_type=ChartType.LINE,
                query=AnalyticsQuery(
                    metric_type=MetricType.SYSTEM_METRICS,
                    time_range=time_range,
                    group_by=["timestamp", "service"],
                    aggregation=AggregationType.MEAN
                ),
                width=6,
                height=300,
                config={
                    "x_column": "timestamp",
                    "y_column": "cpu_usage",
                    "color_column": "service",
                    "title": "CPU Usage by Service"
                }
            ),
            
            # Memory Usage
            DashboardWidget(
                widget_id="memory_usage",
                title="Memory Usage Over Time",
                chart_type=ChartType.LINE,
                query=AnalyticsQuery(
                    metric_type=MetricType.SYSTEM_METRICS,
                    time_range=time_range,
                    group_by=["timestamp", "service"],
                    aggregation=AggregationType.MEAN
                ),
                width=6,
                height=300,
                config={
                    "x_column": "timestamp",
                    "y_column": "memory_usage",
                    "color_column": "service",
                    "title": "Memory Usage by Service"
                }
            ),
            
            # Service Uptime
            DashboardWidget(
                widget_id="service_uptime",
                title="Service Uptime",
                chart_type=ChartType.BAR,
                query=AnalyticsQuery(
                    metric_type=MetricType.SYSTEM_METRICS,
                    time_range=time_range,
                    group_by=["service"],
                    aggregation=AggregationType.MEAN
                ),
                width=6,
                height=300,
                config={
                    "x_column": "service",
                    "y_column": "uptime",
                    "title": "Average Service Uptime"
                }
            ),
            
            # Error Counts
            DashboardWidget(
                widget_id="error_counts",
                title="Error Counts by Service",
                chart_type=ChartType.BAR,
                query=AnalyticsQuery(
                    metric_type=MetricType.SYSTEM_METRICS,
                    time_range=time_range,
                    group_by=["service"],
                    aggregation=AggregationType.SUM
                ),
                width=6,
                height=300,
                config={
                    "x_column": "service",
                    "y_column": "error_count",
                    "title": "Total Errors by Service"
                }
            )
        ]
        
        return Dashboard(
            dashboard_id="system_overview",
            title="System Overview",
            description="Overall system health and performance metrics",
            widgets=widgets,
            tags=["system", "monitoring", "overview"]
        )
    
    @staticmethod
    def create_anomaly_detection_dashboard() -> Dashboard:
        """Create anomaly detection dashboard."""
        now = datetime.utcnow()
        time_range = (now - timedelta(hours=24), now)
        
        widgets = [
            # Anomalies Over Time
            DashboardWidget(
                widget_id="anomalies_timeline",
                title="Anomalies Detected Over Time",
                chart_type=ChartType.LINE,
                query=AnalyticsQuery(
                    metric_type=MetricType.DETECTION_METRICS,
                    time_range=time_range,
                    group_by=["timestamp"],
                    aggregation=AggregationType.SUM
                ),
                width=12,
                height=300,
                config={
                    "x_column": "timestamp",
                    "y_column": "anomalies_detected",
                    "title": "Anomalies Detected Over Time"
                }
            ),
            
            # Algorithm Performance
            DashboardWidget(
                widget_id="algorithm_accuracy",
                title="Algorithm Accuracy Comparison",
                chart_type=ChartType.BAR,
                query=AnalyticsQuery(
                    metric_type=MetricType.DETECTION_METRICS,
                    time_range=time_range,
                    group_by=["algorithm"],
                    aggregation=AggregationType.MEAN
                ),
                width=6,
                height=300,
                config={
                    "x_column": "algorithm",
                    "y_column": "accuracy",
                    "title": "Average Accuracy by Algorithm"
                }
            ),
            
            # Processing Time Distribution
            DashboardWidget(
                widget_id="processing_time_dist",
                title="Processing Time Distribution",
                chart_type=ChartType.HISTOGRAM,
                query=AnalyticsQuery(
                    metric_type=MetricType.DETECTION_METRICS,
                    time_range=time_range
                ),
                width=6,
                height=300,
                config={
                    "x_column": "processing_time",
                    "bins": 30,
                    "title": "Processing Time Distribution"
                }
            ),
            
            # Algorithm Usage
            DashboardWidget(
                widget_id="algorithm_usage",
                title="Algorithm Usage",
                chart_type=ChartType.PIE,
                query=AnalyticsQuery(
                    metric_type=MetricType.DETECTION_METRICS,
                    time_range=time_range,
                    group_by=["algorithm"],
                    aggregation=AggregationType.COUNT
                ),
                width=6,
                height=300,
                config={
                    "values_column": "count",
                    "names_column": "algorithm",
                    "title": "Algorithm Usage Distribution"
                }
            ),
            
            # Performance Metrics Heatmap
            DashboardWidget(
                widget_id="performance_heatmap",
                title="Algorithm Performance Heatmap",
                chart_type=ChartType.HEATMAP,
                query=AnalyticsQuery(
                    metric_type=MetricType.DETECTION_METRICS,
                    time_range=time_range,
                    group_by=["algorithm"],
                    aggregation=AggregationType.MEAN
                ),
                width=6,
                height=300,
                config={
                    "correlation": True,
                    "title": "Algorithm Performance Correlation"
                }
            )
        ]
        
        return Dashboard(
            dashboard_id="anomaly_detection",
            title="Anomaly Detection Analytics",
            description="Comprehensive anomaly detection performance and metrics",
            widgets=widgets,
            tags=["anomaly", "detection", "ml", "performance"]
        )
    
    @staticmethod
    def create_performance_monitoring_dashboard() -> Dashboard:
        """Create performance monitoring dashboard."""
        now = datetime.utcnow()
        time_range = (now - timedelta(hours=24), now)
        
        widgets = [
            # Response Time Trends
            DashboardWidget(
                widget_id="response_time_trends",
                title="API Response Time Trends",
                chart_type=ChartType.LINE,
                query=AnalyticsQuery(
                    metric_type=MetricType.PERFORMANCE_METRICS,
                    time_range=time_range,
                    group_by=["timestamp", "endpoint"],
                    aggregation=AggregationType.MEAN
                ),
                width=12,
                height=300,
                config={
                    "x_column": "timestamp",
                    "y_column": "response_time",
                    "color_column": "endpoint",
                    "title": "Response Time by Endpoint"
                }
            ),
            
            # Throughput
            DashboardWidget(
                widget_id="throughput",
                title="API Throughput",
                chart_type=ChartType.BAR,
                query=AnalyticsQuery(
                    metric_type=MetricType.PERFORMANCE_METRICS,
                    time_range=time_range,
                    group_by=["endpoint"],
                    aggregation=AggregationType.MEAN
                ),
                width=6,
                height=300,
                config={
                    "x_column": "endpoint",
                    "y_column": "requests_per_second",
                    "title": "Average Requests per Second"
                }
            ),
            
            # Error Rate
            DashboardWidget(
                widget_id="error_rate",
                title="Error Rate by Endpoint",
                chart_type=ChartType.BAR,
                query=AnalyticsQuery(
                    metric_type=MetricType.PERFORMANCE_METRICS,
                    time_range=time_range,
                    group_by=["endpoint"],
                    aggregation=AggregationType.MEAN
                ),
                width=6,
                height=300,
                config={
                    "x_column": "endpoint",
                    "y_column": "error_rate",
                    "title": "Average Error Rate"
                }
            ),
            
            # Resource Usage
            DashboardWidget(
                widget_id="resource_usage",
                title="Resource Usage Over Time",
                chart_type=ChartType.LINE,
                query=AnalyticsQuery(
                    metric_type=MetricType.PERFORMANCE_METRICS,
                    time_range=time_range,
                    group_by=["timestamp"],
                    aggregation=AggregationType.MEAN
                ),
                width=12,
                height=300,
                config={
                    "x_column": "timestamp",
                    "y_column": "cpu_usage",
                    "title": "CPU and Memory Usage"
                }
            )
        ]
        
        return Dashboard(
            dashboard_id="performance_monitoring",
            title="Performance Monitoring",
            description="API and system performance metrics",
            widgets=widgets,
            tags=["performance", "api", "monitoring"]
        )
    
    @staticmethod
    def create_security_monitoring_dashboard() -> Dashboard:
        """Create security monitoring dashboard."""
        now = datetime.utcnow()
        time_range = (now - timedelta(hours=24), now)
        
        widgets = [
            # Threat Detection Timeline
            DashboardWidget(
                widget_id="threat_timeline",
                title="Threat Detection Timeline",
                chart_type=ChartType.LINE,
                query=AnalyticsQuery(
                    metric_type=MetricType.SECURITY_METRICS,
                    time_range=time_range,
                    group_by=["timestamp", "threat_type"],
                    aggregation=AggregationType.SUM
                ),
                width=12,
                height=300,
                config={
                    "x_column": "timestamp",
                    "y_column": "incidents_detected",
                    "color_column": "threat_type",
                    "title": "Security Incidents by Type"
                }
            ),
            
            # Threat Type Distribution
            DashboardWidget(
                widget_id="threat_distribution",
                title="Threat Type Distribution",
                chart_type=ChartType.PIE,
                query=AnalyticsQuery(
                    metric_type=MetricType.SECURITY_METRICS,
                    time_range=time_range,
                    group_by=["threat_type"],
                    aggregation=AggregationType.SUM
                ),
                width=6,
                height=300,
                config={
                    "values_column": "incidents_detected",
                    "names_column": "threat_type",
                    "title": "Threat Types"
                }
            ),
            
            # Blocked vs Detected
            DashboardWidget(
                widget_id="blocked_vs_detected",
                title="Blocked vs Detected Threats",
                chart_type=ChartType.BAR,
                query=AnalyticsQuery(
                    metric_type=MetricType.SECURITY_METRICS,
                    time_range=time_range,
                    group_by=["threat_type"],
                    aggregation=AggregationType.SUM
                ),
                width=6,
                height=300,
                config={
                    "x_column": "threat_type",
                    "y_column": "blocked_attempts",
                    "title": "Blocked Attempts by Threat Type"
                }
            ),
            
            # Security Response Time
            DashboardWidget(
                widget_id="security_response_time",
                title="Security Response Time",
                chart_type=ChartType.BOX,
                query=AnalyticsQuery(
                    metric_type=MetricType.SECURITY_METRICS,
                    time_range=time_range
                ),
                width=6,
                height=300,
                config={
                    "y_column": "response_time",
                    "x_column": "threat_type",
                    "title": "Response Time by Threat Type"
                }
            ),
            
            # Mitigation Success Rate
            DashboardWidget(
                widget_id="mitigation_success",
                title="Mitigation Success Rate",
                chart_type=ChartType.BAR,
                query=AnalyticsQuery(
                    metric_type=MetricType.SECURITY_METRICS,
                    time_range=time_range,
                    group_by=["threat_type"],
                    aggregation=AggregationType.MEAN
                ),
                width=6,
                height=300,
                config={
                    "x_column": "threat_type",
                    "y_column": "mitigation_success",
                    "title": "Average Mitigation Success Rate"
                }
            )
        ]
        
        return Dashboard(
            dashboard_id="security_monitoring",
            title="Security Monitoring",
            description="Security threats and incident monitoring",
            widgets=widgets,
            tags=["security", "threats", "monitoring"]
        )
    
    @staticmethod
    def create_business_metrics_dashboard() -> Dashboard:
        """Create business metrics dashboard."""
        now = datetime.utcnow()
        time_range = (now - timedelta(days=30), now)
        
        widgets = [
            # Revenue Trends
            DashboardWidget(
                widget_id="revenue_trends",
                title="Revenue Trends",
                chart_type=ChartType.LINE,
                query=AnalyticsQuery(
                    metric_type=MetricType.BUSINESS_METRICS,
                    time_range=time_range,
                    group_by=["timestamp", "product"],
                    aggregation=AggregationType.SUM
                ),
                width=12,
                height=300,
                config={
                    "x_column": "timestamp",
                    "y_column": "revenue",
                    "color_column": "product",
                    "title": "Revenue by Product"
                }
            ),
            
            # User Growth
            DashboardWidget(
                widget_id="user_growth",
                title="User Growth by Region",
                chart_type=ChartType.BAR,
                query=AnalyticsQuery(
                    metric_type=MetricType.BUSINESS_METRICS,
                    time_range=time_range,
                    group_by=["region"],
                    aggregation=AggregationType.SUM
                ),
                width=6,
                height=300,
                config={
                    "x_column": "region",
                    "y_column": "users",
                    "title": "Total Users by Region"
                }
            ),
            
            # API Usage
            DashboardWidget(
                widget_id="api_usage",
                title="API Usage by Product",
                chart_type=ChartType.TREEMAP,
                query=AnalyticsQuery(
                    metric_type=MetricType.BUSINESS_METRICS,
                    time_range=time_range,
                    group_by=["product", "region"],
                    aggregation=AggregationType.SUM
                ),
                width=6,
                height=300,
                config={
                    "path_columns": ["product", "region"],
                    "values_column": "api_calls",
                    "title": "API Calls by Product and Region"
                }
            ),
            
            # Customer Satisfaction
            DashboardWidget(
                widget_id="customer_satisfaction",
                title="Customer Satisfaction",
                chart_type=ChartType.SCATTER,
                query=AnalyticsQuery(
                    metric_type=MetricType.BUSINESS_METRICS,
                    time_range=time_range,
                    group_by=["product"],
                    aggregation=AggregationType.MEAN
                ),
                width=6,
                height=300,
                config={
                    "x_column": "customer_satisfaction",
                    "y_column": "churn_rate",
                    "color_column": "product",
                    "title": "Satisfaction vs Churn Rate"
                }
            ),
            
            # Cost Analysis
            DashboardWidget(
                widget_id="cost_analysis",
                title="Cost vs Revenue Analysis",
                chart_type=ChartType.SCATTER,
                query=AnalyticsQuery(
                    metric_type=MetricType.BUSINESS_METRICS,
                    time_range=time_range,
                    group_by=["product", "region"],
                    aggregation=AggregationType.SUM
                ),
                width=6,
                height=300,
                config={
                    "x_column": "cost",
                    "y_column": "revenue",
                    "color_column": "product",
                    "size_column": "users",
                    "title": "Cost vs Revenue by Product"
                }
            )
        ]
        
        return Dashboard(
            dashboard_id="business_metrics",
            title="Business Metrics",
            description="Key business performance indicators",
            widgets=widgets,
            tags=["business", "kpi", "revenue", "users"]
        )
    
    @staticmethod
    def create_executive_summary_dashboard() -> Dashboard:
        """Create executive summary dashboard."""
        now = datetime.utcnow()
        time_range = (now - timedelta(days=7), now)
        
        widgets = [
            # Key Metrics Summary
            DashboardWidget(
                widget_id="key_metrics",
                title="Key Performance Indicators",
                chart_type=ChartType.BAR,
                query=AnalyticsQuery(
                    metric_type=MetricType.BUSINESS_METRICS,
                    time_range=time_range,
                    group_by=["product"],
                    aggregation=AggregationType.SUM
                ),
                width=12,
                height=200,
                config={
                    "x_column": "product",
                    "y_column": "revenue",
                    "title": "Weekly Revenue by Product"
                }
            ),
            
            # System Health
            DashboardWidget(
                widget_id="system_health",
                title="System Health Overview",
                chart_type=ChartType.PIE,
                query=AnalyticsQuery(
                    metric_type=MetricType.SYSTEM_METRICS,
                    time_range=time_range,
                    group_by=["service"],
                    aggregation=AggregationType.MEAN
                ),
                width=6,
                height=300,
                config={
                    "values_column": "uptime",
                    "names_column": "service",
                    "title": "Service Uptime"
                }
            ),
            
            # Security Status
            DashboardWidget(
                widget_id="security_status",
                title="Security Incidents This Week",
                chart_type=ChartType.BAR,
                query=AnalyticsQuery(
                    metric_type=MetricType.SECURITY_METRICS,
                    time_range=time_range,
                    group_by=["severity"],
                    aggregation=AggregationType.COUNT
                ),
                width=6,
                height=300,
                config={
                    "x_column": "severity",
                    "y_column": "count",
                    "title": "Security Incidents by Severity"
                }
            )
        ]
        
        return Dashboard(
            dashboard_id="executive_summary",
            title="Executive Summary",
            description="High-level business and system overview",
            widgets=widgets,
            tags=["executive", "summary", "overview"]
        )


class DashboardService:
    """Service for managing dashboards and reports."""
    
    def __init__(self):
        self.analytics_engine = get_analytics_engine()
        self.dashboard_metrics: Dict[str, DashboardMetrics] = {}
        self.reports: Dict[str, ReportConfig] = {}
        self.lock = threading.Lock()
        self.metrics_collector = get_safe_metrics_collector()
        
        # Initialize default dashboards
        self._initialize_default_dashboards()
        
        logger.info("Dashboard service initialized")
    
    def _initialize_default_dashboards(self):
        """Initialize default dashboard templates."""
        try:
            templates = [
                DashboardTemplateFactory.create_system_overview_dashboard(),
                DashboardTemplateFactory.create_anomaly_detection_dashboard(),
                DashboardTemplateFactory.create_performance_monitoring_dashboard(),
                DashboardTemplateFactory.create_security_monitoring_dashboard(),
                DashboardTemplateFactory.create_business_metrics_dashboard(),
                DashboardTemplateFactory.create_executive_summary_dashboard()
            ]
            
            for dashboard in templates:
                asyncio.create_task(self.analytics_engine.create_dashboard(dashboard))
                self.dashboard_metrics[dashboard.dashboard_id] = DashboardMetrics(
                    dashboard_id=dashboard.dashboard_id
                )
            
            logger.info(f"Initialized {len(templates)} default dashboards")
            
        except Exception as e:
            logger.error(f"Failed to initialize default dashboards: {e}")
    
    async def list_dashboards(self, tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """List available dashboards."""
        try:
            dashboards = []
            
            for dashboard_id, metrics in self.dashboard_metrics.items():
                dashboard = await self.analytics_engine.get_dashboard(dashboard_id)
                if dashboard:
                    # Filter by tags if provided
                    if tags and not any(tag in dashboard.tags for tag in tags):
                        continue
                    
                    dashboards.append({
                        "dashboard_id": dashboard.dashboard_id,
                        "title": dashboard.title,
                        "description": dashboard.description,
                        "tags": dashboard.tags,
                        "widgets_count": len(dashboard.widgets),
                        "views": metrics.views,
                        "unique_viewers": len(metrics.unique_viewers),
                        "last_viewed": metrics.last_viewed.isoformat() if metrics.last_viewed else None,
                        "avg_load_time": metrics.avg_load_time,
                        "created_at": dashboard.created_at.isoformat(),
                        "updated_at": dashboard.updated_at.isoformat()
                    })
            
            return sorted(dashboards, key=lambda x: x["views"], reverse=True)
            
        except Exception as e:
            logger.error(f"Failed to list dashboards: {e}")
            return []
    
    async def get_dashboard(self, dashboard_id: str, viewer_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get dashboard by ID and track usage."""
        try:
            start_time = datetime.utcnow()
            
            # Get dashboard
            dashboard_data = await self.analytics_engine.render_dashboard(dashboard_id)
            
            if "error" in dashboard_data:
                return dashboard_data
            
            # Track usage metrics
            load_time = (datetime.utcnow() - start_time).total_seconds()
            await self._track_dashboard_usage(dashboard_id, viewer_id, load_time)
            
            # Add usage metrics to response
            metrics = self.dashboard_metrics.get(dashboard_id)
            if metrics:
                dashboard_data["usage_metrics"] = {
                    "views": metrics.views,
                    "unique_viewers": len(metrics.unique_viewers),
                    "avg_load_time": metrics.avg_load_time,
                    "last_viewed": metrics.last_viewed.isoformat() if metrics.last_viewed else None
                }
            
            return dashboard_data
            
        except Exception as e:
            logger.error(f"Failed to get dashboard {dashboard_id}: {e}")
            
            # Track error
            if dashboard_id in self.dashboard_metrics:
                self.dashboard_metrics[dashboard_id].errors += 1
            
            return {"error": str(e)}
    
    async def _track_dashboard_usage(self, dashboard_id: str, viewer_id: Optional[str], load_time: float):
        """Track dashboard usage metrics."""
        try:
            with self.lock:
                if dashboard_id not in self.dashboard_metrics:
                    self.dashboard_metrics[dashboard_id] = DashboardMetrics(dashboard_id=dashboard_id)
                
                metrics = self.dashboard_metrics[dashboard_id]
                metrics.views += 1
                metrics.last_viewed = datetime.utcnow()
                
                if viewer_id:
                    metrics.unique_viewers.add(viewer_id)
                
                # Update average load time
                if metrics.avg_load_time == 0:
                    metrics.avg_load_time = load_time
                else:
                    metrics.avg_load_time = (metrics.avg_load_time + load_time) / 2
            
            # Record metric
            self.metrics_collector.record_metric(
                "dashboard.viewed",
                1,
                {
                    "dashboard_id": dashboard_id,
                    "load_time": load_time
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to track dashboard usage: {e}")
    
    async def create_custom_dashboard(self, dashboard_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create a custom dashboard."""
        try:
            # Validate required fields
            required_fields = ["dashboard_id", "title", "widgets"]
            for field in required_fields:
                if field not in dashboard_config:
                    return {"error": f"Missing required field: {field}"}
            
            # Parse widgets
            widgets = []
            for widget_config in dashboard_config["widgets"]:
                try:
                    widget = self._parse_widget_config(widget_config)
                    widgets.append(widget)
                except Exception as e:
                    return {"error": f"Invalid widget configuration: {e}"}
            
            # Create dashboard
            dashboard = Dashboard(
                dashboard_id=dashboard_config["dashboard_id"],
                title=dashboard_config["title"],
                description=dashboard_config.get("description", ""),
                widgets=widgets,
                tags=dashboard_config.get("tags", []),
                is_public=dashboard_config.get("is_public", False)
            )
            
            # Save dashboard
            success = await self.analytics_engine.create_dashboard(dashboard)
            
            if success:
                # Initialize metrics
                self.dashboard_metrics[dashboard.dashboard_id] = DashboardMetrics(
                    dashboard_id=dashboard.dashboard_id
                )
                
                return {
                    "success": True,
                    "dashboard_id": dashboard.dashboard_id,
                    "message": "Dashboard created successfully"
                }
            else:
                return {"error": "Failed to create dashboard"}
            
        except Exception as e:
            logger.error(f"Failed to create custom dashboard: {e}")
            return {"error": str(e)}
    
    def _parse_widget_config(self, widget_config: Dict[str, Any]) -> DashboardWidget:
        """Parse widget configuration from dictionary."""
        # Parse query
        query_config = widget_config["query"]
        
        # Parse time range
        if isinstance(query_config["time_range"], list) and len(query_config["time_range"]) == 2:
            time_range = (
                datetime.fromisoformat(query_config["time_range"][0]),
                datetime.fromisoformat(query_config["time_range"][1])
            )
        else:
            # Default to last 24 hours
            now = datetime.utcnow()
            time_range = (now - timedelta(hours=24), now)
        
        query = AnalyticsQuery(
            metric_type=MetricType(query_config["metric_type"]),
            time_range=time_range,
            filters=query_config.get("filters", {}),
            group_by=query_config.get("group_by", []),
            aggregation=AggregationType(query_config.get("aggregation", "count")),
            limit=query_config.get("limit"),
            sort_by=query_config.get("sort_by"),
            sort_desc=query_config.get("sort_desc", True)
        )
        
        return DashboardWidget(
            widget_id=widget_config["widget_id"],
            title=widget_config["title"],
            chart_type=ChartType(widget_config["chart_type"]),
            query=query,
            refresh_interval=widget_config.get("refresh_interval", 300),
            width=widget_config.get("width", 6),
            height=widget_config.get("height", 300),
            config=widget_config.get("config", {})
        )
    
    async def get_dashboard_insights(self, dashboard_id: str) -> Dict[str, Any]:
        """Get automated insights for a dashboard."""
        try:
            dashboard = await self.analytics_engine.get_dashboard(dashboard_id)
            if not dashboard:
                return {"error": "Dashboard not found"}
            
            insights = []
            
            # Get insights for each widget's metric type
            metric_types_processed = set()
            
            for widget in dashboard.widgets:
                metric_type = widget.query.metric_type
                
                if metric_type not in metric_types_processed:
                    widget_insights = await self.analytics_engine.get_insights(
                        metric_type,
                        widget.query.time_range
                    )
                    
                    if "insights" in widget_insights:
                        insights.extend(widget_insights["insights"])
                    
                    metric_types_processed.add(metric_type)
            
            # Add dashboard-specific insights
            usage_metrics = self.dashboard_metrics.get(dashboard_id)
            if usage_metrics:
                if usage_metrics.views > 100:
                    insights.append({
                        "type": "usage",
                        "description": f"This dashboard is popular with {usage_metrics.views} views",
                        "confidence": "high"
                    })
                
                if usage_metrics.avg_load_time > 2.0:
                    insights.append({
                        "type": "performance",
                        "description": f"Dashboard load time is high ({usage_metrics.avg_load_time:.1f}s)",
                        "confidence": "high"
                    })
            
            return {
                "dashboard_id": dashboard_id,
                "insights": insights,
                "generated_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get dashboard insights: {e}")
            return {"error": str(e)}
    
    async def export_dashboard(self, dashboard_id: str, format: ReportFormat) -> Dict[str, Any]:
        """Export dashboard in specified format."""
        try:
            dashboard_data = await self.analytics_engine.render_dashboard(dashboard_id)
            
            if "error" in dashboard_data:
                return dashboard_data
            
            if format == ReportFormat.JSON:
                return {
                    "success": True,
                    "format": "json",
                    "data": dashboard_data
                }
            elif format == ReportFormat.HTML:
                html_content = self._generate_html_report(dashboard_data)
                return {
                    "success": True,
                    "format": "html",
                    "content": html_content
                }
            else:
                return {"error": f"Export format {format.value} not yet implemented"}
            
        except Exception as e:
            logger.error(f"Failed to export dashboard {dashboard_id}: {e}")
            return {"error": str(e)}
    
    def _generate_html_report(self, dashboard_data: Dict[str, Any]) -> str:
        """Generate HTML report from dashboard data."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{dashboard_data['title']}</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ text-align: center; margin-bottom: 30px; }}
                .widget {{ margin-bottom: 30px; border: 1px solid #ddd; padding: 15px; }}
                .widget-title {{ font-size: 18px; font-weight: bold; margin-bottom: 10px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{dashboard_data['title']}</h1>
                <p>{dashboard_data.get('description', '')}</p>
                <p><small>Generated: {dashboard_data.get('generated_at', '')}</small></p>
            </div>
        """
        
        for i, widget in enumerate(dashboard_data.get('widgets', [])):
            if 'chart' in widget and widget['chart']:
                html += f"""
                <div class="widget">
                    <div class="widget-title">{widget.get('title', f'Widget {i+1}')}</div>
                    <div id="chart_{i}"></div>
                    <script>
                        Plotly.newPlot('chart_{i}', {widget['chart']});
                    </script>
                </div>
                """
        
        html += """
        </body>
        </html>
        """
        
        return html
    
    async def get_dashboard_metrics(self) -> Dict[str, Any]:
        """Get overall dashboard usage metrics."""
        try:
            total_views = sum(m.views for m in self.dashboard_metrics.values())
            total_unique_viewers = len(set().union(*(m.unique_viewers for m in self.dashboard_metrics.values())))
            avg_load_time = sum(m.avg_load_time for m in self.dashboard_metrics.values()) / len(self.dashboard_metrics) if self.dashboard_metrics else 0
            total_errors = sum(m.errors for m in self.dashboard_metrics.values())
            
            # Top dashboards by views
            top_dashboards = sorted(
                [
                    {
                        "dashboard_id": dashboard_id,
                        "views": metrics.views,
                        "unique_viewers": len(metrics.unique_viewers),
                        "avg_load_time": metrics.avg_load_time
                    }
                    for dashboard_id, metrics in self.dashboard_metrics.items()
                ],
                key=lambda x: x["views"],
                reverse=True
            )[:5]
            
            return {
                "overview": {
                    "total_dashboards": len(self.dashboard_metrics),
                    "total_views": total_views,
                    "unique_viewers": total_unique_viewers,
                    "avg_load_time": avg_load_time,
                    "total_errors": total_errors
                },
                "top_dashboards": top_dashboards,
                "generated_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get dashboard metrics: {e}")
            return {"error": str(e)}


# Global dashboard service instance
_dashboard_service: Optional[DashboardService] = None


def get_dashboard_service() -> DashboardService:
    """Get the global dashboard service instance."""
    global _dashboard_service
    
    if _dashboard_service is None:
        _dashboard_service = DashboardService()
    
    return _dashboard_service


def initialize_dashboard_service() -> DashboardService:
    """Initialize the global dashboard service."""
    global _dashboard_service
    _dashboard_service = DashboardService()
    return _dashboard_service