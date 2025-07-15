"""Live Quality Dashboard Service.

Provides real-time quality dashboard with metrics visualization, trend analysis,
drill-down capabilities, quality heatmaps, and customizable layouts.
"""

import asyncio
import json
import logging
from typing import List, Dict, Any, Optional, Callable, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import statistics
import math

from ...domain.entities.quality_monitoring import (
    QualityDashboard, DashboardWidget, DashboardRefreshMode,
    StreamingQualityAssessment, QualityAlert, StreamingMetrics,
    QualityMonitoringJob, MonitoringJobId, StreamId
)

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of dashboard metrics."""
    GAUGE = "gauge"
    LINE_CHART = "line_chart"
    BAR_CHART = "bar_chart"
    HEATMAP = "heatmap"
    TABLE = "table"
    ALERT_LIST = "alert_list"
    TREND_INDICATOR = "trend_indicator"
    DISTRIBUTION = "distribution"
    CORRELATION = "correlation"
    SPARKLINE = "sparkline"


class TimeRange(Enum):
    """Time ranges for dashboard data."""
    LAST_5_MINUTES = "5m"
    LAST_15_MINUTES = "15m"
    LAST_30_MINUTES = "30m"
    LAST_1_HOUR = "1h"
    LAST_3_HOURS = "3h"
    LAST_6_HOURS = "6h"
    LAST_12_HOURS = "12h"
    LAST_24_HOURS = "24h"
    LAST_7_DAYS = "7d"
    LAST_30_DAYS = "30d"


@dataclass
class DashboardMetric:
    """Individual dashboard metric."""
    metric_id: str
    metric_name: str
    metric_type: MetricType
    current_value: float
    previous_value: Optional[float] = None
    unit: str = ""
    format_string: str = "{:.2f}"
    thresholds: Dict[str, float] = field(default_factory=dict)
    trend: Optional[str] = None  # "up", "down", "stable"
    trend_percentage: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def get_status(self) -> str:
        """Get metric status based on thresholds."""
        if not self.thresholds:
            return "normal"
        
        if "critical" in self.thresholds and self.current_value <= self.thresholds["critical"]:
            return "critical"
        elif "warning" in self.thresholds and self.current_value <= self.thresholds["warning"]:
            return "warning"
        elif "good" in self.thresholds and self.current_value >= self.thresholds["good"]:
            return "good"
        else:
            return "normal"
    
    def calculate_trend(self) -> None:
        """Calculate trend information."""
        if self.previous_value is None:
            self.trend = "stable"
            self.trend_percentage = 0.0
            return
        
        if self.previous_value == 0:
            self.trend = "up" if self.current_value > 0 else "stable"
            self.trend_percentage = 100.0 if self.current_value > 0 else 0.0
            return
        
        change = self.current_value - self.previous_value
        percentage_change = (change / self.previous_value) * 100
        
        if abs(percentage_change) < 1.0:  # Less than 1% change
            self.trend = "stable"
        elif percentage_change > 0:
            self.trend = "up"
        else:
            self.trend = "down"
        
        self.trend_percentage = abs(percentage_change)
    
    def format_value(self) -> str:
        """Format metric value for display."""
        return self.format_string.format(self.current_value)


@dataclass
class DashboardData:
    """Dashboard data structure."""
    dashboard_id: str
    generated_at: datetime
    metrics: List[DashboardMetric]
    time_series_data: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    alerts: List[QualityAlert] = field(default_factory=list)
    summary_stats: Dict[str, Any] = field(default_factory=dict)
    
    def get_metric_by_id(self, metric_id: str) -> Optional[DashboardMetric]:
        """Get metric by ID."""
        for metric in self.metrics:
            if metric.metric_id == metric_id:
                return metric
        return None
    
    def get_metrics_by_type(self, metric_type: MetricType) -> List[DashboardMetric]:
        """Get metrics by type."""
        return [metric for metric in self.metrics if metric.metric_type == metric_type]


@dataclass
class DashboardServiceConfig:
    """Configuration for dashboard service."""
    
    # Update intervals
    real_time_update_interval_seconds: int = 1
    near_real_time_update_interval_seconds: int = 5
    periodic_update_interval_seconds: int = 30
    
    # Data retention
    time_series_retention_hours: int = 24
    metrics_buffer_size: int = 1000
    
    # Performance
    max_concurrent_dashboards: int = 50
    max_widgets_per_dashboard: int = 20
    data_point_limit: int = 500
    
    # Caching
    enable_caching: bool = True
    cache_ttl_seconds: int = 30
    
    # Alerts
    max_alerts_per_dashboard: int = 100
    alert_age_limit_hours: int = 24
    
    # Rendering
    default_chart_colors: List[str] = field(default_factory=lambda: [
        "#007bff", "#28a745", "#ffc107", "#dc3545", "#17a2b8", "#6c757d"
    ])
    
    # WebSocket
    enable_websocket: bool = True
    websocket_port: int = 8765
    max_websocket_connections: int = 100


class DashboardDataProvider:
    """Provides data for dashboard widgets."""
    
    def __init__(self):
        """Initialize dashboard data provider."""
        self.data_cache: Dict[str, Any] = {}
        self.cache_timestamps: Dict[str, datetime] = {}
        self.time_series_data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.metric_aggregators: Dict[str, Callable] = {}
        
    def register_metric_aggregator(self, metric_name: str, aggregator: Callable) -> None:
        """Register a metric aggregator function."""
        self.metric_aggregators[metric_name] = aggregator
    
    def add_time_series_point(self, metric_name: str, value: float, timestamp: datetime = None) -> None:
        """Add a time series data point."""
        if timestamp is None:
            timestamp = datetime.now()
        
        self.time_series_data[metric_name].append({
            'timestamp': timestamp,
            'value': value
        })
    
    def get_time_series_data(self, metric_name: str, time_range: TimeRange) -> List[Dict[str, Any]]:
        """Get time series data for a metric."""
        if metric_name not in self.time_series_data:
            return []
        
        # Calculate time range
        now = datetime.now()
        range_mapping = {
            TimeRange.LAST_5_MINUTES: timedelta(minutes=5),
            TimeRange.LAST_15_MINUTES: timedelta(minutes=15),
            TimeRange.LAST_30_MINUTES: timedelta(minutes=30),
            TimeRange.LAST_1_HOUR: timedelta(hours=1),
            TimeRange.LAST_3_HOURS: timedelta(hours=3),
            TimeRange.LAST_6_HOURS: timedelta(hours=6),
            TimeRange.LAST_12_HOURS: timedelta(hours=12),
            TimeRange.LAST_24_HOURS: timedelta(hours=24),
            TimeRange.LAST_7_DAYS: timedelta(days=7),
            TimeRange.LAST_30_DAYS: timedelta(days=30)
        }
        
        cutoff_time = now - range_mapping.get(time_range, timedelta(hours=1))
        
        # Filter data
        filtered_data = [
            point for point in self.time_series_data[metric_name]
            if point['timestamp'] >= cutoff_time
        ]
        
        return filtered_data
    
    def get_aggregated_metric(self, metric_name: str, aggregation: str = "avg") -> float:
        """Get aggregated metric value."""
        if metric_name not in self.time_series_data:
            return 0.0
        
        values = [point['value'] for point in self.time_series_data[metric_name]]
        if not values:
            return 0.0
        
        if aggregation == "avg":
            return statistics.mean(values)
        elif aggregation == "sum":
            return sum(values)
        elif aggregation == "min":
            return min(values)
        elif aggregation == "max":
            return max(values)
        elif aggregation == "median":
            return statistics.median(values)
        elif aggregation == "last":
            return values[-1]
        else:
            return values[-1]  # Default to last value
    
    def calculate_metric_trend(self, metric_name: str, lookback_minutes: int = 30) -> Dict[str, Any]:
        """Calculate trend for a metric."""
        if metric_name not in self.time_series_data:
            return {'trend': 'stable', 'change': 0.0, 'percentage': 0.0}
        
        now = datetime.now()
        cutoff_time = now - timedelta(minutes=lookback_minutes)
        
        # Get recent data
        recent_data = [
            point for point in self.time_series_data[metric_name]
            if point['timestamp'] >= cutoff_time
        ]
        
        if len(recent_data) < 2:
            return {'trend': 'stable', 'change': 0.0, 'percentage': 0.0}
        
        # Calculate trend
        values = [point['value'] for point in recent_data]
        first_value = values[0]
        last_value = values[-1]
        
        change = last_value - first_value
        percentage_change = (change / first_value * 100) if first_value != 0 else 0.0
        
        if abs(percentage_change) < 1.0:
            trend = 'stable'
        elif percentage_change > 0:
            trend = 'up'
        else:
            trend = 'down'
        
        return {
            'trend': trend,
            'change': change,
            'percentage': abs(percentage_change)
        }


class LiveQualityDashboardService:
    """Live quality dashboard service."""
    
    def __init__(self, config: DashboardServiceConfig = None):
        """Initialize live quality dashboard service."""
        self.config = config or DashboardServiceConfig()
        self.data_provider = DashboardDataProvider()
        
        # Active dashboards
        self.active_dashboards: Dict[str, QualityDashboard] = {}
        self.dashboard_subscriptions: Dict[str, Set[str]] = defaultdict(set)  # dashboard_id -> client_ids
        
        # Data streams
        self.monitoring_jobs: Dict[MonitoringJobId, QualityMonitoringJob] = {}
        self.recent_assessments: Dict[StreamId, deque] = defaultdict(lambda: deque(maxlen=100))
        self.active_alerts: Dict[StreamId, List[QualityAlert]] = defaultdict(list)
        
        # Background tasks
        self.background_tasks: List[asyncio.Task] = []
        self.shutdown_event = asyncio.Event()
        
        # WebSocket connections
        self.websocket_connections: Dict[str, Any] = {}  # client_id -> websocket
        
        # Performance metrics
        self.metrics = {
            'dashboards_active': 0,
            'clients_connected': 0,
            'updates_sent': 0,
            'data_points_processed': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        logger.info("Live quality dashboard service initialized")
    
    async def start(self) -> None:
        """Start the dashboard service."""
        logger.info("Starting live quality dashboard service")
        
        # Start background tasks
        task = asyncio.create_task(self._dashboard_update_loop())
        self.background_tasks.append(task)
        
        task = asyncio.create_task(self._metrics_collection_loop())
        self.background_tasks.append(task)
        
        task = asyncio.create_task(self._cleanup_loop())
        self.background_tasks.append(task)
        
        # Start WebSocket server if enabled
        if self.config.enable_websocket:
            task = asyncio.create_task(self._start_websocket_server())
            self.background_tasks.append(task)
        
        logger.info("Live quality dashboard service started")
    
    async def stop(self) -> None:
        """Stop the dashboard service."""
        logger.info("Stopping live quality dashboard service")
        
        self.shutdown_event.set()
        
        # Cancel background tasks
        for task in self.background_tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        # Close WebSocket connections
        for client_id, websocket in self.websocket_connections.items():
            try:
                await websocket.close()
            except Exception as e:
                logger.warning(f"Error closing websocket for client {client_id}: {str(e)}")
        
        logger.info("Live quality dashboard service stopped")
    
    async def create_dashboard(self, dashboard: QualityDashboard) -> str:
        """Create a new dashboard."""
        dashboard_id = dashboard.dashboard_id
        self.active_dashboards[dashboard_id] = dashboard
        self.metrics['dashboards_active'] = len(self.active_dashboards)
        
        logger.info(f"Created dashboard: {dashboard_id}")
        return dashboard_id
    
    async def update_dashboard(self, dashboard: QualityDashboard) -> bool:
        """Update an existing dashboard."""
        dashboard_id = dashboard.dashboard_id
        
        if dashboard_id not in self.active_dashboards:
            logger.warning(f"Dashboard {dashboard_id} not found")
            return False
        
        self.active_dashboards[dashboard_id] = dashboard
        
        # Notify subscribers
        await self._notify_dashboard_subscribers(dashboard_id, "dashboard_updated")
        
        logger.info(f"Updated dashboard: {dashboard_id}")
        return True
    
    async def delete_dashboard(self, dashboard_id: str) -> bool:
        """Delete a dashboard."""
        if dashboard_id not in self.active_dashboards:
            logger.warning(f"Dashboard {dashboard_id} not found")
            return False
        
        del self.active_dashboards[dashboard_id]
        
        # Clean up subscriptions
        if dashboard_id in self.dashboard_subscriptions:
            del self.dashboard_subscriptions[dashboard_id]
        
        self.metrics['dashboards_active'] = len(self.active_dashboards)
        
        logger.info(f"Deleted dashboard: {dashboard_id}")
        return True
    
    async def get_dashboard(self, dashboard_id: str) -> Optional[QualityDashboard]:
        """Get a dashboard by ID."""
        return self.active_dashboards.get(dashboard_id)
    
    async def get_dashboard_data(self, dashboard_id: str) -> Optional[DashboardData]:
        """Get dashboard data."""
        dashboard = self.active_dashboards.get(dashboard_id)
        if not dashboard:
            return None
        
        # Generate dashboard data
        dashboard_data = await self._generate_dashboard_data(dashboard)
        
        return dashboard_data
    
    async def subscribe_to_dashboard(self, dashboard_id: str, client_id: str) -> bool:
        """Subscribe a client to dashboard updates."""
        if dashboard_id not in self.active_dashboards:
            logger.warning(f"Dashboard {dashboard_id} not found")
            return False
        
        self.dashboard_subscriptions[dashboard_id].add(client_id)
        self.metrics['clients_connected'] = sum(len(clients) for clients in self.dashboard_subscriptions.values())
        
        logger.info(f"Client {client_id} subscribed to dashboard {dashboard_id}")
        return True
    
    async def unsubscribe_from_dashboard(self, dashboard_id: str, client_id: str) -> bool:
        """Unsubscribe a client from dashboard updates."""
        if dashboard_id in self.dashboard_subscriptions:
            self.dashboard_subscriptions[dashboard_id].discard(client_id)
            if not self.dashboard_subscriptions[dashboard_id]:
                del self.dashboard_subscriptions[dashboard_id]
        
        self.metrics['clients_connected'] = sum(len(clients) for clients in self.dashboard_subscriptions.values())
        
        logger.info(f"Client {client_id} unsubscribed from dashboard {dashboard_id}")
        return True
    
    async def register_websocket_client(self, client_id: str, websocket: Any) -> None:
        """Register a WebSocket client."""
        self.websocket_connections[client_id] = websocket
        logger.info(f"Registered WebSocket client: {client_id}")
    
    async def unregister_websocket_client(self, client_id: str) -> None:
        """Unregister a WebSocket client."""
        if client_id in self.websocket_connections:
            del self.websocket_connections[client_id]
        
        # Clean up subscriptions
        for dashboard_id in list(self.dashboard_subscriptions.keys()):
            self.dashboard_subscriptions[dashboard_id].discard(client_id)
            if not self.dashboard_subscriptions[dashboard_id]:
                del self.dashboard_subscriptions[dashboard_id]
        
        self.metrics['clients_connected'] = sum(len(clients) for clients in self.dashboard_subscriptions.values())
        
        logger.info(f"Unregistered WebSocket client: {client_id}")
    
    async def update_monitoring_job(self, job: QualityMonitoringJob) -> None:
        """Update monitoring job data."""
        self.monitoring_jobs[job.job_id] = job
        
        # Update data provider with new assessments
        for assessment in job.recent_assessments:
            self._process_assessment(assessment)
        
        # Update alerts
        self.active_alerts[job.stream_id] = job.active_alerts
    
    async def add_quality_assessment(self, assessment: StreamingQualityAssessment) -> None:
        """Add a new quality assessment."""
        self.recent_assessments[assessment.stream_id].append(assessment)
        self._process_assessment(assessment)
        
        # Notify relevant dashboards
        await self._notify_assessment_update(assessment)
    
    async def add_quality_alert(self, alert: QualityAlert) -> None:
        """Add a new quality alert."""
        self.active_alerts[alert.stream_id].append(alert)
        
        # Notify relevant dashboards
        await self._notify_alert_update(alert)
    
    def get_service_statistics(self) -> Dict[str, Any]:
        """Get service statistics."""
        return {
            'active_dashboards': len(self.active_dashboards),
            'total_subscriptions': sum(len(clients) for clients in self.dashboard_subscriptions.values()),
            'websocket_connections': len(self.websocket_connections),
            'monitoring_jobs': len(self.monitoring_jobs),
            'active_streams': len(self.recent_assessments),
            'cache_hit_rate': self.metrics['cache_hits'] / max(1, self.metrics['cache_hits'] + self.metrics['cache_misses']) * 100,
            'updates_sent': self.metrics['updates_sent'],
            'data_points_processed': self.metrics['data_points_processed']
        }
    
    # Private methods
    
    def _process_assessment(self, assessment: StreamingQualityAssessment) -> None:
        """Process quality assessment for dashboard data."""
        timestamp = assessment.window_end
        
        # Add time series data points
        self.data_provider.add_time_series_point("overall_score", assessment.overall_score, timestamp)
        self.data_provider.add_time_series_point("completeness_score", assessment.completeness_score, timestamp)
        self.data_provider.add_time_series_point("accuracy_score", assessment.accuracy_score, timestamp)
        self.data_provider.add_time_series_point("consistency_score", assessment.consistency_score, timestamp)
        self.data_provider.add_time_series_point("validity_score", assessment.validity_score, timestamp)
        self.data_provider.add_time_series_point("uniqueness_score", assessment.uniqueness_score, timestamp)
        self.data_provider.add_time_series_point("timeliness_score", assessment.timeliness_score, timestamp)
        self.data_provider.add_time_series_point("records_processed", assessment.records_processed, timestamp)
        self.data_provider.add_time_series_point("processing_latency", assessment.processing_latency_ms, timestamp)
        self.data_provider.add_time_series_point("anomalies_detected", len(assessment.anomalies_detected), timestamp)
        
        # Add custom metrics
        for metric_name, value in assessment.quality_metrics.items():
            self.data_provider.add_time_series_point(metric_name, value, timestamp)
        
        self.metrics['data_points_processed'] += 1
    
    async def _generate_dashboard_data(self, dashboard: QualityDashboard) -> DashboardData:
        """Generate dashboard data."""
        metrics = []
        time_series_data = {}
        
        # Process each widget
        for widget in dashboard.widgets:
            widget_metrics = await self._generate_widget_metrics(widget)
            metrics.extend(widget_metrics)
            
            # Get time series data if needed
            if widget.widget_type in ["line_chart", "area_chart", "sparkline"]:
                for metric_name in widget.metrics:
                    time_range = TimeRange.LAST_1_HOUR  # Default time range
                    time_series_data[metric_name] = self.data_provider.get_time_series_data(metric_name, time_range)
        
        # Get recent alerts
        recent_alerts = []
        for stream_alerts in self.active_alerts.values():
            recent_alerts.extend([alert for alert in stream_alerts if alert.is_active()])
        
        # Limit alerts
        recent_alerts = recent_alerts[:self.config.max_alerts_per_dashboard]
        
        # Generate summary stats
        summary_stats = self._generate_summary_stats(metrics)
        
        return DashboardData(
            dashboard_id=dashboard.dashboard_id,
            generated_at=datetime.now(),
            metrics=metrics,
            time_series_data=time_series_data,
            alerts=recent_alerts,
            summary_stats=summary_stats
        )
    
    async def _generate_widget_metrics(self, widget: DashboardWidget) -> List[DashboardMetric]:
        """Generate metrics for a widget."""
        metrics = []
        
        for metric_name in widget.metrics:
            # Get current value
            current_value = self.data_provider.get_aggregated_metric(metric_name, "last")
            
            # Get previous value for trend calculation
            previous_value = self.data_provider.get_aggregated_metric(metric_name, "avg")
            
            # Create metric
            metric = DashboardMetric(
                metric_id=f"{widget.widget_id}_{metric_name}",
                metric_name=metric_name,
                metric_type=MetricType(widget.chart_type) if widget.chart_type in [mt.value for mt in MetricType] else MetricType.GAUGE,
                current_value=current_value,
                previous_value=previous_value,
                unit=self._get_metric_unit(metric_name),
                format_string=self._get_metric_format(metric_name),
                thresholds=self._get_metric_thresholds(metric_name)
            )
            
            # Calculate trend
            metric.calculate_trend()
            
            metrics.append(metric)
        
        return metrics
    
    def _get_metric_unit(self, metric_name: str) -> str:
        """Get unit for metric."""
        unit_mapping = {
            "overall_score": "%",
            "completeness_score": "%",
            "accuracy_score": "%",
            "consistency_score": "%",
            "validity_score": "%",
            "uniqueness_score": "%",
            "timeliness_score": "%",
            "records_processed": "records",
            "processing_latency": "ms",
            "anomalies_detected": "count",
            "throughput": "records/s",
            "error_rate": "%"
        }
        
        return unit_mapping.get(metric_name, "")
    
    def _get_metric_format(self, metric_name: str) -> str:
        """Get format string for metric."""
        format_mapping = {
            "overall_score": "{:.1f}",
            "completeness_score": "{:.1f}",
            "accuracy_score": "{:.1f}",
            "consistency_score": "{:.1f}",
            "validity_score": "{:.1f}",
            "uniqueness_score": "{:.1f}",
            "timeliness_score": "{:.1f}",
            "records_processed": "{:.0f}",
            "processing_latency": "{:.2f}",
            "anomalies_detected": "{:.0f}",
            "throughput": "{:.2f}",
            "error_rate": "{:.2f}"
        }
        
        return format_mapping.get(metric_name, "{:.2f}")
    
    def _get_metric_thresholds(self, metric_name: str) -> Dict[str, float]:
        """Get thresholds for metric."""
        threshold_mapping = {
            "overall_score": {"critical": 0.5, "warning": 0.7, "good": 0.9},
            "completeness_score": {"critical": 0.5, "warning": 0.7, "good": 0.9},
            "accuracy_score": {"critical": 0.5, "warning": 0.7, "good": 0.9},
            "consistency_score": {"critical": 0.5, "warning": 0.7, "good": 0.9},
            "validity_score": {"critical": 0.5, "warning": 0.7, "good": 0.9},
            "uniqueness_score": {"critical": 0.5, "warning": 0.7, "good": 0.9},
            "timeliness_score": {"critical": 0.5, "warning": 0.7, "good": 0.9},
            "processing_latency": {"critical": 1000, "warning": 500, "good": 100},
            "error_rate": {"critical": 0.1, "warning": 0.05, "good": 0.01}
        }
        
        return threshold_mapping.get(metric_name, {})
    
    def _generate_summary_stats(self, metrics: List[DashboardMetric]) -> Dict[str, Any]:
        """Generate summary statistics."""
        if not metrics:
            return {}
        
        # Calculate summary statistics
        total_metrics = len(metrics)
        critical_metrics = len([m for m in metrics if m.get_status() == "critical"])
        warning_metrics = len([m for m in metrics if m.get_status() == "warning"])
        good_metrics = len([m for m in metrics if m.get_status() == "good"])
        
        # Overall health score
        health_score = (good_metrics / total_metrics) * 100 if total_metrics > 0 else 0
        
        # Trending metrics
        trending_up = len([m for m in metrics if m.trend == "up"])
        trending_down = len([m for m in metrics if m.trend == "down"])
        
        return {
            "total_metrics": total_metrics,
            "critical_metrics": critical_metrics,
            "warning_metrics": warning_metrics,
            "good_metrics": good_metrics,
            "health_score": health_score,
            "trending_up": trending_up,
            "trending_down": trending_down,
            "last_updated": datetime.now().isoformat()
        }
    
    async def _notify_dashboard_subscribers(self, dashboard_id: str, event_type: str) -> None:
        """Notify dashboard subscribers."""
        if dashboard_id not in self.dashboard_subscriptions:
            return
        
        subscribers = self.dashboard_subscriptions[dashboard_id]
        
        # Generate fresh dashboard data
        dashboard = self.active_dashboards[dashboard_id]
        dashboard_data = await self._generate_dashboard_data(dashboard)
        
        # Send updates to subscribers
        for client_id in subscribers:
            await self._send_update_to_client(client_id, {
                "type": event_type,
                "dashboard_id": dashboard_id,
                "data": self._serialize_dashboard_data(dashboard_data)
            })
    
    async def _notify_assessment_update(self, assessment: StreamingQualityAssessment) -> None:
        """Notify dashboards about assessment updates."""
        # Find dashboards that should be notified
        for dashboard_id, dashboard in self.active_dashboards.items():
            if self._should_notify_dashboard(dashboard, assessment):
                await self._notify_dashboard_subscribers(dashboard_id, "assessment_update")
    
    async def _notify_alert_update(self, alert: QualityAlert) -> None:
        """Notify dashboards about alert updates."""
        # Find dashboards that should be notified
        for dashboard_id, dashboard in self.active_dashboards.items():
            if self._should_notify_dashboard_alert(dashboard, alert):
                await self._notify_dashboard_subscribers(dashboard_id, "alert_update")
    
    def _should_notify_dashboard(self, dashboard: QualityDashboard, assessment: StreamingQualityAssessment) -> bool:
        """Check if dashboard should be notified about assessment."""
        # Check if dashboard has widgets that use assessment data
        for widget in dashboard.widgets:
            if widget.data_source == "quality_assessments":
                return True
            if str(assessment.stream_id) in widget.filters.get("stream_ids", []):
                return True
        
        return False
    
    def _should_notify_dashboard_alert(self, dashboard: QualityDashboard, alert: QualityAlert) -> bool:
        """Check if dashboard should be notified about alert."""
        # Check if dashboard has alert widgets
        for widget in dashboard.widgets:
            if widget.widget_type == "alert_list":
                return True
            if str(alert.stream_id) in widget.filters.get("stream_ids", []):
                return True
        
        return False
    
    async def _send_update_to_client(self, client_id: str, update: Dict[str, Any]) -> None:
        """Send update to a specific client."""
        if client_id not in self.websocket_connections:
            return
        
        websocket = self.websocket_connections[client_id]
        
        try:
            await websocket.send(json.dumps(update))
            self.metrics['updates_sent'] += 1
        except Exception as e:
            logger.error(f"Failed to send update to client {client_id}: {str(e)}")
            # Remove failed connection
            await self.unregister_websocket_client(client_id)
    
    def _serialize_dashboard_data(self, dashboard_data: DashboardData) -> Dict[str, Any]:
        """Serialize dashboard data for transmission."""
        return {
            "dashboard_id": dashboard_data.dashboard_id,
            "generated_at": dashboard_data.generated_at.isoformat(),
            "metrics": [
                {
                    "metric_id": metric.metric_id,
                    "metric_name": metric.metric_name,
                    "metric_type": metric.metric_type.value,
                    "current_value": metric.current_value,
                    "formatted_value": metric.format_value(),
                    "unit": metric.unit,
                    "status": metric.get_status(),
                    "trend": metric.trend,
                    "trend_percentage": metric.trend_percentage,
                    "timestamp": metric.timestamp.isoformat()
                }
                for metric in dashboard_data.metrics
            ],
            "time_series_data": {
                metric_name: [
                    {
                        "timestamp": point["timestamp"].isoformat(),
                        "value": point["value"]
                    }
                    for point in points
                ]
                for metric_name, points in dashboard_data.time_series_data.items()
            },
            "alerts": [
                {
                    "alert_id": str(alert.alert_id),
                    "severity": alert.severity.value,
                    "title": alert.title,
                    "description": alert.description,
                    "triggered_at": alert.triggered_at.isoformat(),
                    "status": alert.status.value,
                    "stream_id": str(alert.stream_id)
                }
                for alert in dashboard_data.alerts
            ],
            "summary_stats": dashboard_data.summary_stats
        }
    
    async def _dashboard_update_loop(self) -> None:
        """Main dashboard update loop."""
        while not self.shutdown_event.is_set():
            try:
                # Update all active dashboards
                for dashboard_id in list(self.active_dashboards.keys()):
                    if dashboard_id in self.dashboard_subscriptions:
                        await self._notify_dashboard_subscribers(dashboard_id, "periodic_update")
                
                # Wait for next update
                await asyncio.sleep(self.config.periodic_update_interval_seconds)
                
            except Exception as e:
                logger.error(f"Error in dashboard update loop: {str(e)}")
                await asyncio.sleep(5)  # Wait before retry
    
    async def _metrics_collection_loop(self) -> None:
        """Metrics collection loop."""
        while not self.shutdown_event.is_set():
            try:
                # Collect metrics from monitoring jobs
                for job in self.monitoring_jobs.values():
                    if job.is_running():
                        self._collect_job_metrics(job)
                
                # Wait for next collection
                await asyncio.sleep(self.config.real_time_update_interval_seconds)
                
            except Exception as e:
                logger.error(f"Error in metrics collection loop: {str(e)}")
                await asyncio.sleep(5)  # Wait before retry
    
    def _collect_job_metrics(self, job: QualityMonitoringJob) -> None:
        """Collect metrics from a monitoring job."""
        timestamp = datetime.now()
        
        # Collect streaming metrics
        metrics = job.streaming_metrics
        self.data_provider.add_time_series_point("throughput", metrics.throughput_records_per_second, timestamp)
        self.data_provider.add_time_series_point("latency", metrics.latency_ms, timestamp)
        self.data_provider.add_time_series_point("error_rate", metrics.error_rate, timestamp)
        self.data_provider.add_time_series_point("memory_usage", metrics.memory_usage_mb, timestamp)
        self.data_provider.add_time_series_point("cpu_usage", metrics.cpu_usage_percent, timestamp)
        self.data_provider.add_time_series_point("active_windows", metrics.active_windows, timestamp)
        self.data_provider.add_time_series_point("backlog_size", metrics.backlog_size, timestamp)
        
        # Collect job-specific metrics
        self.data_provider.add_time_series_point("windows_processed", job.windows_processed, timestamp)
        self.data_provider.add_time_series_point("total_records_processed", job.total_records_processed, timestamp)
        self.data_provider.add_time_series_point("active_alerts", job.get_active_alert_count(), timestamp)
        self.data_provider.add_time_series_point("critical_alerts", job.get_critical_alert_count(), timestamp)
        self.data_provider.add_time_series_point("current_quality_score", job.get_current_quality_score(), timestamp)
        self.data_provider.add_time_series_point("error_count", job.error_count, timestamp)
    
    async def _cleanup_loop(self) -> None:
        """Cleanup loop."""
        while not self.shutdown_event.is_set():
            try:
                # Cleanup old time series data
                self._cleanup_time_series_data()
                
                # Cleanup old alerts
                self._cleanup_old_alerts()
                
                # Wait for next cleanup
                await asyncio.sleep(3600)  # 1 hour
                
            except Exception as e:
                logger.error(f"Error in cleanup loop: {str(e)}")
                await asyncio.sleep(300)  # Wait before retry
    
    def _cleanup_time_series_data(self) -> None:
        """Clean up old time series data."""
        retention_time = datetime.now() - timedelta(hours=self.config.time_series_retention_hours)
        
        for metric_name, time_series in self.data_provider.time_series_data.items():
            # Remove old data points
            while time_series and time_series[0]['timestamp'] < retention_time:
                time_series.popleft()
    
    def _cleanup_old_alerts(self) -> None:
        """Clean up old alerts."""
        age_limit = datetime.now() - timedelta(hours=self.config.alert_age_limit_hours)
        
        for stream_id, alerts in self.active_alerts.items():
            # Keep only recent alerts
            self.active_alerts[stream_id] = [
                alert for alert in alerts
                if alert.triggered_at >= age_limit or alert.is_active()
            ]
    
    async def _start_websocket_server(self) -> None:
        """Start WebSocket server."""
        try:
            import websockets
            
            async def handle_client(websocket, path):
                client_id = f"client_{int(datetime.now().timestamp())}"
                await self.register_websocket_client(client_id, websocket)
                
                try:
                    async for message in websocket:
                        await self._handle_websocket_message(client_id, message)
                except websockets.exceptions.ConnectionClosed:
                    pass
                finally:
                    await self.unregister_websocket_client(client_id)
            
            # Start server
            server = await websockets.serve(
                handle_client,
                "localhost",
                self.config.websocket_port,
                max_size=None,
                max_queue=None
            )
            
            logger.info(f"WebSocket server started on port {self.config.websocket_port}")
            
            # Wait for shutdown
            await self.shutdown_event.wait()
            
            # Close server
            server.close()
            await server.wait_closed()
            
        except ImportError:
            logger.warning("websockets library not available - WebSocket server disabled")
        except Exception as e:
            logger.error(f"Error starting WebSocket server: {str(e)}")
    
    async def _handle_websocket_message(self, client_id: str, message: str) -> None:
        """Handle WebSocket message from client."""
        try:
            data = json.loads(message)
            message_type = data.get("type")
            
            if message_type == "subscribe":
                dashboard_id = data.get("dashboard_id")
                if dashboard_id:
                    await self.subscribe_to_dashboard(dashboard_id, client_id)
            
            elif message_type == "unsubscribe":
                dashboard_id = data.get("dashboard_id")
                if dashboard_id:
                    await self.unsubscribe_from_dashboard(dashboard_id, client_id)
            
            elif message_type == "get_dashboard_data":
                dashboard_id = data.get("dashboard_id")
                if dashboard_id:
                    dashboard_data = await self.get_dashboard_data(dashboard_id)
                    if dashboard_data:
                        await self._send_update_to_client(client_id, {
                            "type": "dashboard_data",
                            "dashboard_id": dashboard_id,
                            "data": self._serialize_dashboard_data(dashboard_data)
                        })
            
        except Exception as e:
            logger.error(f"Error handling WebSocket message from client {client_id}: {str(e)}")