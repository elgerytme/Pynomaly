"""Advanced monitoring dashboards for domain-driven anomaly detection service."""

import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class DashboardType(str, Enum):
    """Dashboard types."""
    DOMAIN_OVERVIEW = "domain_overview"
    PERFORMANCE_METRICS = "performance_metrics"
    BUSINESS_METRICS = "business_metrics"
    SECURITY_DASHBOARD = "security_dashboard"
    OPERATIONAL_HEALTH = "operational_health"


class ChartType(str, Enum):
    """Chart types for dashboards."""
    LINE_CHART = "line_chart"
    BAR_CHART = "bar_chart"
    PIE_CHART = "pie_chart"
    HEATMAP = "heatmap"
    GAUGE = "gauge"
    TABLE = "table"


@dataclass
class DashboardWidget:
    """Dashboard widget configuration."""
    id: str
    title: str
    chart_type: ChartType
    data_source: str
    query: str
    refresh_interval_seconds: int = 60
    width: int = 6
    height: int = 4
    position: Dict[str, int] = None
    config: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.position is None:
            self.position = {"x": 0, "y": 0}
        if self.config is None:
            self.config = {}


@dataclass
class Dashboard:
    """Dashboard configuration."""
    id: str
    name: str
    description: str
    dashboard_type: DashboardType
    widgets: List[DashboardWidget]
    refresh_interval_seconds: int = 30
    auto_refresh: bool = True
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()


class AdvancedDashboardManager:
    """Manager for advanced monitoring dashboards."""
    
    def __init__(self):
        """Initialize dashboard manager."""
        self.dashboards: Dict[str, Dashboard] = {}
        self.widget_data_cache: Dict[str, Any] = {}
        self.cache_ttl = 300  # 5 minutes
        
    def create_domain_overview_dashboard(self) -> Dashboard:
        """Create domain overview dashboard."""
        widgets = [
            # Domain Health Overview
            DashboardWidget(
                id="domain_health_overview",
                title="Domain Health Status",
                chart_type=ChartType.PIE_CHART,
                data_source="domain_health_monitor",
                query="SELECT domain, status, COUNT(*) FROM domain_health GROUP BY domain, status",
                position={"x": 0, "y": 0},
                width=6,
                height=4,
                config={
                    "colors": {
                        "healthy": "#28a745",
                        "degraded": "#ffc107", 
                        "unhealthy": "#dc3545"
                    }
                }
            ),
            
            # Request Volume by Domain
            DashboardWidget(
                id="request_volume_by_domain",
                title="Request Volume by Domain (Last 24h)",
                chart_type=ChartType.BAR_CHART,
                data_source="metrics",
                query="""
                SELECT 
                    domain,
                    COUNT(*) as requests
                FROM request_logs 
                WHERE timestamp >= NOW() - INTERVAL '24 hours'
                GROUP BY domain
                ORDER BY requests DESC
                """,
                position={"x": 6, "y": 0},
                width=6,
                height=4
            ),
            
            # Response Time Heatmap
            DashboardWidget(
                id="response_time_heatmap",
                title="Response Time Distribution by Domain",
                chart_type=ChartType.HEATMAP,
                data_source="metrics",
                query="""
                SELECT 
                    domain,
                    EXTRACT(HOUR FROM timestamp) as hour,
                    AVG(response_time_ms) as avg_response_time
                FROM request_logs 
                WHERE timestamp >= NOW() - INTERVAL '7 days'
                GROUP BY domain, hour
                """,
                position={"x": 0, "y": 4},
                width=12,
                height=4,
                config={
                    "color_scale": ["#green", "#yellow", "#red"],
                    "max_value": 1000
                }
            ),
            
            # Error Rate by Domain
            DashboardWidget(
                id="error_rate_by_domain",
                title="Error Rate by Domain (Last 24h)",
                chart_type=ChartType.LINE_CHART,
                data_source="metrics",
                query="""
                SELECT 
                    domain,
                    DATE_TRUNC('hour', timestamp) as hour,
                    (COUNT(CASE WHEN status_code >= 400 THEN 1 END) * 100.0 / COUNT(*)) as error_rate
                FROM request_logs 
                WHERE timestamp >= NOW() - INTERVAL '24 hours'
                GROUP BY domain, hour
                ORDER BY hour
                """,
                position={"x": 0, "y": 8},
                width=12,
                height=4,
                config={
                    "y_axis_max": 100,
                    "alert_threshold": 5.0
                }
            )
        ]
        
        dashboard = Dashboard(
            id="domain_overview",
            name="Domain Overview Dashboard",
            description="High-level overview of all domain health and performance",
            dashboard_type=DashboardType.DOMAIN_OVERVIEW,
            widgets=widgets
        )
        
        self.dashboards[dashboard.id] = dashboard
        return dashboard
    
    def create_performance_dashboard(self) -> Dashboard:
        """Create performance metrics dashboard."""
        widgets = [
            # Algorithm Performance Comparison
            DashboardWidget(
                id="algorithm_performance",
                title="Algorithm Performance Comparison",
                chart_type=ChartType.BAR_CHART,
                data_source="ml_metrics",
                query="""
                SELECT 
                    algorithm,
                    AVG(execution_time_ms) as avg_execution_time,
                    AVG(accuracy_score) as avg_accuracy,
                    COUNT(*) as total_runs
                FROM ml_execution_logs 
                WHERE timestamp >= NOW() - INTERVAL '7 days'
                GROUP BY algorithm
                """,
                position={"x": 0, "y": 0},
                width=8,
                height=4
            ),
            
            # Memory Usage Trends
            DashboardWidget(
                id="memory_usage_trends",
                title="Memory Usage Trends",
                chart_type=ChartType.LINE_CHART,
                data_source="system_metrics",
                query="""
                SELECT 
                    timestamp,
                    service_name,
                    memory_usage_mb
                FROM system_metrics 
                WHERE timestamp >= NOW() - INTERVAL '24 hours'
                AND metric_name = 'memory_usage'
                ORDER BY timestamp
                """,
                position={"x": 8, "y": 0},
                width=4,
                height=4,
                config={
                    "alert_threshold": 1024
                }
            ),
            
            # Throughput Metrics
            DashboardWidget(
                id="throughput_metrics",
                title="Request Throughput (req/sec)",
                chart_type=ChartType.GAUGE,
                data_source="metrics",
                query="""
                SELECT 
                    COUNT(*) / 60.0 as requests_per_second
                FROM request_logs 
                WHERE timestamp >= NOW() - INTERVAL '1 minute'
                """,
                position={"x": 0, "y": 4},
                width=3,
                height=3,
                config={
                    "min": 0,
                    "max": 1000,
                    "thresholds": [
                        {"value": 800, "color": "green"},
                        {"value": 900, "color": "yellow"},
                        {"value": 950, "color": "red"}
                    ]
                }
            ),
            
            # Cache Hit Rate
            DashboardWidget(
                id="cache_hit_rate",
                title="Cache Hit Rate",
                chart_type=ChartType.GAUGE,
                data_source="cache_metrics",
                query="""
                SELECT 
                    (SUM(CASE WHEN cache_hit = true THEN 1 ELSE 0 END) * 100.0 / COUNT(*)) as hit_rate_percent
                FROM cache_access_logs 
                WHERE timestamp >= NOW() - INTERVAL '5 minutes'
                """,
                position={"x": 3, "y": 4},
                width=3,
                height=3,
                config={
                    "min": 0,
                    "max": 100,
                    "thresholds": [
                        {"value": 80, "color": "red"},
                        {"value": 90, "color": "yellow"},
                        {"value": 95, "color": "green"}
                    ]
                }
            ),
            
            # Model Accuracy Over Time
            DashboardWidget(
                id="model_accuracy_trends",
                title="Model Accuracy Trends",
                chart_type=ChartType.LINE_CHART,
                data_source="ml_metrics",
                query="""
                SELECT 
                    DATE_TRUNC('day', timestamp) as date,
                    model_id,
                    AVG(accuracy_score) as avg_accuracy
                FROM model_evaluation_logs 
                WHERE timestamp >= NOW() - INTERVAL '30 days'
                GROUP BY date, model_id
                ORDER BY date
                """,
                position={"x": 6, "y": 4},
                width=6,
                height=3,
                config={
                    "y_axis_min": 0,
                    "y_axis_max": 1,
                    "alert_threshold": 0.85
                }
            )
        ]
        
        dashboard = Dashboard(
            id="performance_metrics",
            name="Performance Metrics Dashboard", 
            description="Detailed performance metrics across all domains",
            dashboard_type=DashboardType.PERFORMANCE_METRICS,
            widgets=widgets
        )
        
        self.dashboards[dashboard.id] = dashboard
        return dashboard
    
    def create_business_metrics_dashboard(self) -> Dashboard:
        """Create business metrics dashboard."""
        widgets = [
            # Anomaly Detection Volume
            DashboardWidget(
                id="anomaly_detection_volume",
                title="Daily Anomaly Detection Volume",
                chart_type=ChartType.LINE_CHART,
                data_source="ml_metrics",
                query="""
                SELECT 
                    DATE_TRUNC('day', timestamp) as date,
                    COUNT(*) as detections,
                    SUM(anomaly_count) as total_anomalies
                FROM detection_logs 
                WHERE timestamp >= NOW() - INTERVAL '30 days'
                GROUP BY date
                ORDER BY date
                """,
                position={"x": 0, "y": 0},
                width=8,
                height=4
            ),
            
            # Top Anomaly Types
            DashboardWidget(
                id="top_anomaly_types",
                title="Most Common Anomaly Patterns",
                chart_type=ChartType.PIE_CHART,
                data_source="ml_metrics",
                query="""
                SELECT 
                    anomaly_type,
                    COUNT(*) as occurrences
                FROM anomaly_classification_logs 
                WHERE timestamp >= NOW() - INTERVAL '7 days'
                GROUP BY anomaly_type
                ORDER BY occurrences DESC
                LIMIT 10
                """,
                position={"x": 8, "y": 0},
                width=4,
                height=4
            ),
            
            # Customer Usage Patterns
            DashboardWidget(
                id="customer_usage_patterns",
                title="API Usage by Customer Tier",
                chart_type=ChartType.BAR_CHART,
                data_source="usage_metrics",
                query="""
                SELECT 
                    customer_tier,
                    COUNT(*) as api_calls,
                    AVG(response_time_ms) as avg_response_time
                FROM api_usage_logs 
                WHERE timestamp >= NOW() - INTERVAL '7 days'
                GROUP BY customer_tier
                """,
                position={"x": 0, "y": 4},
                width=6,
                height=4
            ),
            
            # Model Performance ROI
            DashboardWidget(
                id="model_performance_roi",
                title="Model Performance vs. Cost",
                chart_type=ChartType.LINE_CHART,
                data_source="business_metrics",
                query="""
                SELECT 
                    DATE_TRUNC('week', timestamp) as week,
                    SUM(true_positives) / (SUM(true_positives) + SUM(false_negatives)) as recall,
                    SUM(compute_cost_usd) as weekly_cost
                FROM model_performance_metrics 
                WHERE timestamp >= NOW() - INTERVAL '12 weeks'
                GROUP BY week
                ORDER BY week
                """,
                position={"x": 6, "y": 4},
                width=6,
                height=4,
                config={
                    "dual_axis": True,
                    "left_axis": "recall",
                    "right_axis": "cost"
                }
            )
        ]
        
        dashboard = Dashboard(
            id="business_metrics",
            name="Business Metrics Dashboard",
            description="Business-focused KPIs and usage analytics",
            dashboard_type=DashboardType.BUSINESS_METRICS,
            widgets=widgets
        )
        
        self.dashboards[dashboard.id] = dashboard
        return dashboard
    
    def create_security_dashboard(self) -> Dashboard:
        """Create security monitoring dashboard."""
        widgets = [
            # Authentication Attempts
            DashboardWidget(
                id="auth_attempts",
                title="Authentication Attempts (Last 24h)",
                chart_type=ChartType.LINE_CHART,
                data_source="security_logs",
                query="""
                SELECT 
                    DATE_TRUNC('hour', timestamp) as hour,
                    COUNT(CASE WHEN success = true THEN 1 END) as successful,
                    COUNT(CASE WHEN success = false THEN 1 END) as failed
                FROM authentication_logs 
                WHERE timestamp >= NOW() - INTERVAL '24 hours'
                GROUP BY hour
                ORDER BY hour
                """,
                position={"x": 0, "y": 0},
                width=8,
                height=4,
                config={
                    "colors": {
                        "successful": "#28a745",
                        "failed": "#dc3545"
                    }
                }
            ),
            
            # Blocked IPs
            DashboardWidget(
                id="blocked_ips",
                title="Top Blocked IP Addresses",
                chart_type=ChartType.TABLE,
                data_source="security_logs",
                query="""
                SELECT 
                    ip_address,
                    COUNT(*) as block_count,
                    MAX(timestamp) as last_blocked,
                    reason
                FROM ip_block_logs 
                WHERE timestamp >= NOW() - INTERVAL '24 hours'
                GROUP BY ip_address, reason
                ORDER BY block_count DESC
                LIMIT 20
                """,
                position={"x": 8, "y": 0},
                width=4,
                height=4
            ),
            
            # Rate Limiting Violations
            DashboardWidget(
                id="rate_limit_violations",
                title="Rate Limiting Violations",
                chart_type=ChartType.BAR_CHART,
                data_source="security_logs",
                query="""
                SELECT 
                    endpoint,
                    COUNT(*) as violations
                FROM rate_limit_violations 
                WHERE timestamp >= NOW() - INTERVAL '24 hours'
                GROUP BY endpoint
                ORDER BY violations DESC
                """,
                position={"x": 0, "y": 4},
                width=6,
                height=4
            ),
            
            # Security Alerts
            DashboardWidget(
                id="security_alerts",
                title="Active Security Alerts",
                chart_type=ChartType.TABLE,
                data_source="security_alerts",
                query="""
                SELECT 
                    alert_type,
                    severity,
                    COUNT(*) as count,
                    MAX(timestamp) as latest
                FROM security_alerts 
                WHERE status = 'active'
                GROUP BY alert_type, severity
                ORDER BY severity DESC, count DESC
                """,
                position={"x": 6, "y": 4},
                width=6,
                height=4,
                config={
                    "severity_colors": {
                        "critical": "#dc3545",
                        "high": "#fd7e14",
                        "medium": "#ffc107",
                        "low": "#28a745"
                    }
                }
            )
        ]
        
        dashboard = Dashboard(
            id="security_dashboard",
            name="Security Monitoring Dashboard",
            description="Security events, threats, and compliance monitoring",
            dashboard_type=DashboardType.SECURITY_DASHBOARD,
            widgets=widgets
        )
        
        self.dashboards[dashboard.id] = dashboard
        return dashboard
    
    def create_operational_health_dashboard(self) -> Dashboard:
        """Create operational health dashboard."""
        widgets = [
            # Service Uptime
            DashboardWidget(
                id="service_uptime",
                title="Service Uptime (Last 30 days)",
                chart_type=ChartType.HEATMAP,
                data_source="uptime_metrics",
                query="""
                SELECT 
                    service_name,
                    DATE_TRUNC('day', timestamp) as date,
                    AVG(CASE WHEN status = 'healthy' THEN 1 ELSE 0 END) * 100 as uptime_percent
                FROM service_health_checks 
                WHERE timestamp >= NOW() - INTERVAL '30 days'
                GROUP BY service_name, date
                """,
                position={"x": 0, "y": 0},
                width=8,
                height=4,
                config={
                    "color_scale": ["#dc3545", "#ffc107", "#28a745"],
                    "min_value": 90,
                    "max_value": 100
                }
            ),
            
            # Deployment Status
            DashboardWidget(
                id="deployment_status",
                title="Recent Deployments",
                chart_type=ChartType.TABLE,
                data_source="deployment_logs",
                query="""
                SELECT 
                    service_name,
                    version,
                    status,
                    deployed_at,
                    deployment_duration_minutes
                FROM deployments 
                WHERE deployed_at >= NOW() - INTERVAL '7 days'
                ORDER BY deployed_at DESC
                LIMIT 20
                """,
                position={"x": 8, "y": 0},
                width=4,
                height=4
            ),
            
            # Infrastructure Costs
            DashboardWidget(
                id="infrastructure_costs",
                title="Daily Infrastructure Costs",
                chart_type=ChartType.LINE_CHART,
                data_source="cost_metrics",
                query="""
                SELECT 
                    DATE_TRUNC('day', timestamp) as date,
                    SUM(cost_usd) as daily_cost,
                    service_name
                FROM cost_tracking 
                WHERE timestamp >= NOW() - INTERVAL '30 days'
                GROUP BY date, service_name
                ORDER BY date
                """,
                position={"x": 0, "y": 4},
                width=6,
                height=4,
                config={
                    "stacked": True
                }
            ),
            
            # Alert Summary
            DashboardWidget(
                id="alert_summary",
                title="Alert Summary (Last 24h)",
                chart_type=ChartType.PIE_CHART,
                data_source="alerts",
                query="""
                SELECT 
                    severity,
                    COUNT(*) as count
                FROM alerts 
                WHERE created_at >= NOW() - INTERVAL '24 hours'
                GROUP BY severity
                """,
                position={"x": 6, "y": 4},
                width=6,
                height=4,
                config={
                    "colors": {
                        "critical": "#dc3545",
                        "high": "#fd7e14", 
                        "medium": "#ffc107",
                        "low": "#28a745"
                    }
                }
            )
        ]
        
        dashboard = Dashboard(
            id="operational_health",
            name="Operational Health Dashboard",
            description="Overall system health, deployments, and operational metrics",
            dashboard_type=DashboardType.OPERATIONAL_HEALTH,
            widgets=widgets
        )
        
        self.dashboards[dashboard.id] = dashboard
        return dashboard
    
    def initialize_default_dashboards(self) -> None:
        """Initialize all default dashboards."""
        logger.info("Initializing default dashboards")
        
        dashboards = [
            self.create_domain_overview_dashboard(),
            self.create_performance_dashboard(),
            self.create_business_metrics_dashboard(),
            self.create_security_dashboard(),
            self.create_operational_health_dashboard()
        ]
        
        logger.info(f"Created {len(dashboards)} dashboards")
    
    def get_dashboard(self, dashboard_id: str) -> Optional[Dashboard]:
        """Get dashboard by ID."""
        return self.dashboards.get(dashboard_id)
    
    def list_dashboards(self) -> List[Dashboard]:
        """List all available dashboards."""
        return list(self.dashboards.values())
    
    def export_dashboard_config(self, dashboard_id: str) -> Dict[str, Any]:
        """Export dashboard configuration as JSON."""
        dashboard = self.get_dashboard(dashboard_id)
        if not dashboard:
            raise ValueError(f"Dashboard {dashboard_id} not found")
        
        return asdict(dashboard)
    
    def export_all_dashboards(self) -> Dict[str, Any]:
        """Export all dashboard configurations."""
        return {
            "dashboards": [asdict(dashboard) for dashboard in self.dashboards.values()],
            "exported_at": datetime.utcnow().isoformat(),
            "version": "1.0"
        }
    
    async def refresh_widget_data(self, widget: DashboardWidget) -> Dict[str, Any]:
        """Refresh data for a specific widget."""
        # This would integrate with actual data sources
        # For now, return mock data structure
        
        cache_key = f"{widget.id}_{widget.data_source}"
        
        # Check cache
        cached_data = self.widget_data_cache.get(cache_key)
        if cached_data and (datetime.utcnow() - cached_data["timestamp"]).seconds < self.cache_ttl:
            return cached_data["data"]
        
        # Simulate data fetching
        await asyncio.sleep(0.1)
        
        # Mock data based on chart type
        if widget.chart_type == ChartType.LINE_CHART:
            data = {
                "type": "line",
                "data": {
                    "labels": ["Hour 1", "Hour 2", "Hour 3", "Hour 4"],
                    "datasets": [{
                        "label": widget.title,
                        "data": [65, 59, 80, 81],
                        "borderColor": "rgb(75, 192, 192)",
                        "tension": 0.1
                    }]
                }
            }
        elif widget.chart_type == ChartType.PIE_CHART:
            data = {
                "type": "pie",
                "data": {
                    "labels": ["Healthy", "Degraded", "Unhealthy"],
                    "datasets": [{
                        "data": [80, 15, 5],
                        "backgroundColor": ["#28a745", "#ffc107", "#dc3545"]
                    }]
                }
            }
        elif widget.chart_type == ChartType.GAUGE:
            data = {
                "type": "gauge",
                "value": 85,
                "max": 100,
                "thresholds": widget.config.get("thresholds", [])
            }
        else:
            data = {
                "type": widget.chart_type.value,
                "data": {"message": "Mock data for " + widget.title}
            }
        
        # Cache the data
        self.widget_data_cache[cache_key] = {
            "data": data,
            "timestamp": datetime.utcnow()
        }
        
        return data


# Global dashboard manager instance
_dashboard_manager = None

def get_dashboard_manager() -> AdvancedDashboardManager:
    """Get or create the global dashboard manager."""
    global _dashboard_manager
    if _dashboard_manager is None:
        _dashboard_manager = AdvancedDashboardManager()
        _dashboard_manager.initialize_default_dashboards()
    
    return _dashboard_manager


async def generate_dashboard_html(dashboard_id: str) -> str:
    """Generate HTML for a dashboard."""
    manager = get_dashboard_manager()
    dashboard = manager.get_dashboard(dashboard_id)
    
    if not dashboard:
        return "<html><body><h1>Dashboard not found</h1></body></html>"
    
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{dashboard.name}</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .dashboard-header {{ margin-bottom: 20px; }}
            .widget-grid {{ display: grid; grid-template-columns: repeat(12, 1fr); gap: 20px; }}
            .widget {{ border: 1px solid #ddd; padding: 15px; border-radius: 5px; }}
            .widget h3 {{ margin-top: 0; }}
            canvas {{ max-width: 100%; }}
        </style>
    </head>
    <body>
        <div class="dashboard-header">
            <h1>{dashboard.name}</h1>
            <p>{dashboard.description}</p>
        </div>
        <div class="widget-grid">
    """
    
    for widget in dashboard.widgets:
        widget_data = await manager.refresh_widget_data(widget)
        
        html_template += f"""
            <div class="widget" style="grid-column: span {widget.width};">
                <h3>{widget.title}</h3>
                <canvas id="chart_{widget.id}"></canvas>
                <script>
                    const ctx_{widget.id} = document.getElementById('chart_{widget.id}').getContext('2d');
                    const chart_{widget.id} = new Chart(ctx_{widget.id}, {json.dumps(widget_data)});
                </script>
            </div>
        """
    
    html_template += """
        </div>
        <script>
            // Auto-refresh every 30 seconds
            setTimeout(() => location.reload(), 30000);
        </script>
    </body>
    </html>
    """
    
    return html_template


if __name__ == "__main__":
    # Example usage
    import asyncio
    
    async def demo():
        manager = get_dashboard_manager()
        
        # List available dashboards
        dashboards = manager.list_dashboards()
        print(f"Available dashboards: {len(dashboards)}")
        
        for dashboard in dashboards:
            print(f"- {dashboard.name} ({dashboard.dashboard_type.value})")
        
        # Generate HTML for domain overview dashboard
        html = await generate_dashboard_html("domain_overview")
        print(f"Generated HTML length: {len(html)} characters")
        
        # Export configuration
        config = manager.export_all_dashboards()
        print(f"Exported {len(config['dashboards'])} dashboard configurations")
    
    asyncio.run(demo())