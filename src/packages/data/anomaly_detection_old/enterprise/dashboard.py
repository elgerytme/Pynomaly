"""
Enterprise Dashboard and Reporting for Pynomaly Detection
==========================================================

Comprehensive enterprise dashboard providing:
- Real-time monitoring and alerting
- Advanced analytics and reporting
- Executive dashboards and KPIs
- Custom report generation
- Data visualization and insights
"""

import logging
import json
import time
import threading
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque
import uuid

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

logger = logging.getLogger(__name__)

class DashboardType(Enum):
    """Dashboard type enumeration."""
    EXECUTIVE = "executive"
    OPERATIONAL = "operational"
    SECURITY = "security"
    COMPLIANCE = "compliance"
    PERFORMANCE = "performance"
    TENANT = "tenant"
    CUSTOM = "custom"

class ReportType(Enum):
    """Report type enumeration."""
    SUMMARY = "summary"
    DETAILED = "detailed"
    TRENDING = "trending"
    COMPARATIVE = "comparative"
    COMPLIANCE = "compliance"
    SECURITY = "security"
    PERFORMANCE = "performance"

class ChartType(Enum):
    """Chart type enumeration."""
    LINE = "line"
    BAR = "bar"
    PIE = "pie"
    SCATTER = "scatter"
    HEATMAP = "heatmap"
    GAUGE = "gauge"
    TABLE = "table"

class MetricFrequency(Enum):
    """Metric collection frequency."""
    REAL_TIME = "real_time"
    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"

@dataclass
class DashboardWidget:
    """Dashboard widget definition."""
    widget_id: str
    title: str
    chart_type: ChartType
    data_source: str
    query: Dict[str, Any]
    refresh_interval: int = 300  # seconds
    size: Dict[str, int] = field(default_factory=lambda: {"width": 6, "height": 4})
    position: Dict[str, int] = field(default_factory=lambda: {"x": 0, "y": 0})
    config: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = True

@dataclass
class Dashboard:
    """Dashboard definition."""
    dashboard_id: str
    name: str
    dashboard_type: DashboardType
    description: str = ""
    widgets: List[DashboardWidget] = field(default_factory=list)
    tenant_id: Optional[str] = None
    created_by: Optional[str] = None
    is_public: bool = False
    created_date: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)

@dataclass
class ReportConfig:
    """Report configuration."""
    report_id: str
    name: str
    report_type: ReportType
    description: str = ""
    data_sources: List[str] = field(default_factory=list)
    filters: Dict[str, Any] = field(default_factory=dict)
    grouping: List[str] = field(default_factory=list)
    metrics: List[str] = field(default_factory=list)
    date_range: Dict[str, Any] = field(default_factory=dict)
    format_type: str = "json"  # json, csv, pdf, excel
    schedule: Optional[Dict[str, Any]] = None  # For scheduled reports
    recipients: List[str] = field(default_factory=list)

@dataclass
class KPI:
    """Key Performance Indicator definition."""
    kpi_id: str
    name: str
    description: str
    metric_source: str
    calculation: str
    target_value: Optional[float] = None
    warning_threshold: Optional[float] = None
    critical_threshold: Optional[float] = None
    unit: str = ""
    frequency: MetricFrequency = MetricFrequency.HOUR
    is_active: bool = True

class EnterpriseDashboard:
    """Comprehensive enterprise dashboard system."""
    
    def __init__(self, tenant_manager=None, audit_logger=None, 
                 security_manager=None, rate_limiter=None):
        """Initialize enterprise dashboard.
        
        Args:
            tenant_manager: Tenant manager instance
            audit_logger: Audit logger instance
            security_manager: Security manager instance
            rate_limiter: Rate limiter instance
        """
        self.tenant_manager = tenant_manager
        self.audit_logger = audit_logger
        self.security_manager = security_manager
        self.rate_limiter = rate_limiter
        
        # Dashboard management
        self.dashboards: Dict[str, Dashboard] = {}
        self.kpis: Dict[str, KPI] = {}
        
        # Real-time data
        self.real_time_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.metric_collectors: Dict[str, Callable] = {}
        
        # Caching
        self.widget_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl = 300  # 5 minutes
        self.cache_timestamps: Dict[str, datetime] = {}
        
        # Threading
        self.lock = threading.RLock()
        self.collection_threads: Dict[str, threading.Thread] = {}
        self.is_running = False
        
        # Initialize default dashboards and KPIs
        self._initialize_default_dashboards()
        self._initialize_default_kpis()
        
        logger.info("Enterprise Dashboard initialized")
    
    def start_monitoring(self):
        """Start real-time monitoring and data collection."""
        if self.is_running:
            logger.warning("Dashboard monitoring already running")
            return
        
        self.is_running = True
        
        # Start metric collection threads
        for kpi_id, kpi in self.kpis.items():
            if kpi.is_active and kpi.frequency == MetricFrequency.REAL_TIME:
                thread = threading.Thread(
                    target=self._collect_kpi_data,
                    args=(kpi,),
                    daemon=True
                )
                thread.start()
                self.collection_threads[kpi_id] = thread
        
        logger.info("Dashboard monitoring started")
    
    def stop_monitoring(self):
        """Stop real-time monitoring."""
        self.is_running = False
        
        # Wait for collection threads to finish
        for thread in self.collection_threads.values():
            if thread.is_alive():
                thread.join(timeout=5.0)
        
        self.collection_threads.clear()
        logger.info("Dashboard monitoring stopped")
    
    def create_dashboard(self, dashboard: Dashboard) -> bool:
        """Create new dashboard.
        
        Args:
            dashboard: Dashboard configuration
            
        Returns:
            Success status
        """
        try:
            with self.lock:
                if dashboard.dashboard_id in self.dashboards:
                    logger.error(f"Dashboard already exists: {dashboard.dashboard_id}")
                    return False
                
                self.dashboards[dashboard.dashboard_id] = dashboard
            
            logger.info(f"Dashboard created: {dashboard.dashboard_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create dashboard: {e}")
            return False
    
    def get_dashboard(self, dashboard_id: str, user_id: Optional[str] = None) -> Optional[Dashboard]:
        """Get dashboard by ID.
        
        Args:
            dashboard_id: Dashboard identifier
            user_id: Optional user identifier for access control
            
        Returns:
            Dashboard or None
        """
        try:
            dashboard = self.dashboards.get(dashboard_id)
            if not dashboard:
                return None
            
            # Check access permissions
            if not self._check_dashboard_access(dashboard, user_id):
                return None
            
            return dashboard
            
        except Exception as e:
            logger.error(f"Failed to get dashboard {dashboard_id}: {e}")
            return None
    
    def get_dashboard_data(self, dashboard_id: str, user_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get dashboard data with widget values.
        
        Args:
            dashboard_id: Dashboard identifier
            user_id: Optional user identifier
            
        Returns:
            Dashboard data or None
        """
        try:
            dashboard = self.get_dashboard(dashboard_id, user_id)
            if not dashboard:
                return None
            
            dashboard_data = {
                'dashboard_id': dashboard.dashboard_id,
                'name': dashboard.name,
                'type': dashboard.dashboard_type.value,
                'description': dashboard.description,
                'last_updated': dashboard.last_updated.isoformat(),
                'widgets': []
            }
            
            # Get data for each widget
            for widget in dashboard.widgets:
                if widget.is_active:
                    widget_data = self._get_widget_data(widget)
                    dashboard_data['widgets'].append(widget_data)
            
            return dashboard_data
            
        except Exception as e:
            logger.error(f"Failed to get dashboard data: {e}")
            return None
    
    def add_widget(self, dashboard_id: str, widget: DashboardWidget) -> bool:
        """Add widget to dashboard.
        
        Args:
            dashboard_id: Dashboard identifier
            widget: Widget to add
            
        Returns:
            Success status
        """
        try:
            with self.lock:
                dashboard = self.dashboards.get(dashboard_id)
                if not dashboard:
                    logger.error(f"Dashboard not found: {dashboard_id}")
                    return False
                
                dashboard.widgets.append(widget)
                dashboard.last_updated = datetime.now()
            
            logger.info(f"Widget added to dashboard {dashboard_id}: {widget.widget_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add widget: {e}")
            return False
    
    def remove_widget(self, dashboard_id: str, widget_id: str) -> bool:
        """Remove widget from dashboard.
        
        Args:
            dashboard_id: Dashboard identifier
            widget_id: Widget identifier
            
        Returns:
            Success status
        """
        try:
            with self.lock:
                dashboard = self.dashboards.get(dashboard_id)
                if not dashboard:
                    return False
                
                dashboard.widgets = [w for w in dashboard.widgets if w.widget_id != widget_id]
                dashboard.last_updated = datetime.now()
            
            logger.info(f"Widget removed from dashboard {dashboard_id}: {widget_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove widget: {e}")
            return False
    
    def create_kpi(self, kpi: KPI) -> bool:
        """Create new KPI.
        
        Args:
            kpi: KPI configuration
            
        Returns:
            Success status
        """
        try:
            with self.lock:
                self.kpis[kpi.kpi_id] = kpi
            
            # Start collection if real-time
            if kpi.is_active and kpi.frequency == MetricFrequency.REAL_TIME and self.is_running:
                thread = threading.Thread(
                    target=self._collect_kpi_data,
                    args=(kpi,),
                    daemon=True
                )
                thread.start()
                self.collection_threads[kpi.kpi_id] = thread
            
            logger.info(f"KPI created: {kpi.kpi_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create KPI: {e}")
            return False
    
    def get_kpi_data(self, kpi_id: str, time_range: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Get KPI data.
        
        Args:
            kpi_id: KPI identifier
            time_range: Optional time range filter
            
        Returns:
            KPI data or None
        """
        try:
            kpi = self.kpis.get(kpi_id)
            if not kpi:
                return None
            
            # Get current value
            current_value = self._calculate_kpi_value(kpi)
            
            # Get historical data
            historical_data = list(self.real_time_metrics[kpi_id])
            
            # Apply time range filter if specified
            if time_range and historical_data:
                start_time = time_range.get('start')
                end_time = time_range.get('end')
                
                if start_time or end_time:
                    filtered_data = []
                    for entry in historical_data:
                        timestamp = entry.get('timestamp')
                        if timestamp:
                            if start_time and timestamp < start_time:
                                continue
                            if end_time and timestamp > end_time:
                                continue
                            filtered_data.append(entry)
                    historical_data = filtered_data
            
            # Determine status
            status = self._get_kpi_status(kpi, current_value)
            
            return {
                'kpi_id': kpi.kpi_id,
                'name': kpi.name,
                'description': kpi.description,
                'current_value': current_value,
                'target_value': kpi.target_value,
                'unit': kpi.unit,
                'status': status,
                'historical_data': historical_data[-100:],  # Last 100 points
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get KPI data: {e}")
            return None
    
    def get_executive_summary(self, tenant_id: Optional[str] = None) -> Dict[str, Any]:
        """Get executive dashboard summary.
        
        Args:
            tenant_id: Optional tenant filter
            
        Returns:
            Executive summary
        """
        try:
            summary = {
                'timestamp': datetime.now().isoformat(),
                'tenant_id': tenant_id,
                'system_health': self._get_system_health(),
                'key_metrics': self._get_key_metrics(tenant_id),
                'alerts': self._get_active_alerts(tenant_id),
                'performance': self._get_performance_summary(tenant_id),
                'compliance': self._get_compliance_summary(tenant_id),
                'security': self._get_security_summary(tenant_id)
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate executive summary: {e}")
            return {}
    
    def generate_chart(self, chart_type: ChartType, data: Dict[str, Any], 
                      config: Dict[str, Any] = None) -> Optional[str]:
        """Generate chart visualization.
        
        Args:
            chart_type: Type of chart
            data: Chart data
            config: Chart configuration
            
        Returns:
            Chart HTML or JSON representation
        """
        try:
            if not PLOTLY_AVAILABLE:
                logger.error("Plotly not available for chart generation")
                return None
            
            config = config or {}
            
            if chart_type == ChartType.LINE:
                return self._generate_line_chart(data, config)
            elif chart_type == ChartType.BAR:
                return self._generate_bar_chart(data, config)
            elif chart_type == ChartType.PIE:
                return self._generate_pie_chart(data, config)
            elif chart_type == ChartType.SCATTER:
                return self._generate_scatter_chart(data, config)
            elif chart_type == ChartType.HEATMAP:
                return self._generate_heatmap(data, config)
            elif chart_type == ChartType.GAUGE:
                return self._generate_gauge_chart(data, config)
            elif chart_type == ChartType.TABLE:
                return self._generate_table(data, config)
            else:
                logger.error(f"Unsupported chart type: {chart_type}")
                return None
                
        except Exception as e:
            logger.error(f"Chart generation failed: {e}")
            return None
    
    def list_dashboards(self, user_id: Optional[str] = None, 
                       tenant_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List available dashboards.
        
        Args:
            user_id: Optional user filter
            tenant_id: Optional tenant filter
            
        Returns:
            List of dashboard summaries
        """
        try:
            dashboard_list = []
            
            for dashboard in self.dashboards.values():
                # Apply filters
                if tenant_id and dashboard.tenant_id != tenant_id:
                    continue
                
                # Check access permissions
                if not self._check_dashboard_access(dashboard, user_id):
                    continue
                
                dashboard_summary = {
                    'dashboard_id': dashboard.dashboard_id,
                    'name': dashboard.name,
                    'type': dashboard.dashboard_type.value,
                    'description': dashboard.description,
                    'widget_count': len(dashboard.widgets),
                    'last_updated': dashboard.last_updated.isoformat(),
                    'tags': dashboard.tags
                }
                
                dashboard_list.append(dashboard_summary)
            
            return dashboard_list
            
        except Exception as e:
            logger.error(f"Failed to list dashboards: {e}")
            return []
    
    def _initialize_default_dashboards(self):
        """Initialize default enterprise dashboards."""
        # Executive Dashboard
        executive_widgets = [
            DashboardWidget(
                widget_id="system_health",
                title="System Health",
                chart_type=ChartType.GAUGE,
                data_source="system_metrics",
                query={"metric": "overall_health"},
                position={"x": 0, "y": 0},
                size={"width": 3, "height": 3}
            ),
            DashboardWidget(
                widget_id="total_requests",
                title="Total Requests (24h)",
                chart_type=ChartType.BAR,
                data_source="api_metrics",
                query={"metric": "request_count", "period": "24h"},
                position={"x": 3, "y": 0},
                size={"width": 3, "height": 3}
            ),
            DashboardWidget(
                widget_id="anomaly_detection_rate",
                title="Anomaly Detection Rate",
                chart_type=ChartType.LINE,
                data_source="detection_metrics",
                query={"metric": "detection_rate", "period": "7d"},
                position={"x": 0, "y": 3},
                size={"width": 6, "height": 4}
            )
        ]
        
        executive_dashboard = Dashboard(
            dashboard_id="executive_overview",
            name="Executive Overview",
            dashboard_type=DashboardType.EXECUTIVE,
            description="High-level executive dashboard with key business metrics",
            widgets=executive_widgets,
            is_public=True
        )
        
        # Security Dashboard
        security_widgets = [
            DashboardWidget(
                widget_id="security_threats",
                title="Security Threats",
                chart_type=ChartType.PIE,
                data_source="security_metrics",
                query={"metric": "threat_distribution"},
                position={"x": 0, "y": 0},
                size={"width": 4, "height": 4}
            ),
            DashboardWidget(
                widget_id="failed_logins",
                title="Failed Login Attempts",
                chart_type=ChartType.LINE,
                data_source="security_metrics",
                query={"metric": "failed_logins", "period": "24h"},
                position={"x": 4, "y": 0},
                size={"width": 4, "height": 4}
            )
        ]
        
        security_dashboard = Dashboard(
            dashboard_id="security_overview",
            name="Security Overview",
            dashboard_type=DashboardType.SECURITY,
            description="Security monitoring and threat analysis dashboard",
            widgets=security_widgets,
            is_public=False
        )
        
        # Add dashboards
        self.dashboards[executive_dashboard.dashboard_id] = executive_dashboard
        self.dashboards[security_dashboard.dashboard_id] = security_dashboard
    
    def _initialize_default_kpis(self):
        """Initialize default KPIs."""
        default_kpis = [
            KPI(
                kpi_id="system_uptime",
                name="System Uptime",
                description="System availability percentage",
                metric_source="system_metrics",
                calculation="uptime_percentage",
                target_value=99.9,
                warning_threshold=99.0,
                critical_threshold=95.0,
                unit="%",
                frequency=MetricFrequency.MINUTE
            ),
            KPI(
                kpi_id="detection_accuracy",
                name="Detection Accuracy",
                description="Anomaly detection accuracy rate",
                metric_source="detection_metrics",
                calculation="accuracy_rate",
                target_value=95.0,
                warning_threshold=90.0,
                critical_threshold=85.0,
                unit="%",
                frequency=MetricFrequency.HOUR
            ),
            KPI(
                kpi_id="api_response_time",
                name="API Response Time",
                description="Average API response time",
                metric_source="api_metrics",
                calculation="avg_response_time",
                target_value=200.0,
                warning_threshold=500.0,
                critical_threshold=1000.0,
                unit="ms",
                frequency=MetricFrequency.MINUTE
            ),
            KPI(
                kpi_id="security_incidents",
                name="Security Incidents",
                description="Number of security incidents per day",
                metric_source="security_metrics",
                calculation="incident_count",
                target_value=0.0,
                warning_threshold=1.0,
                critical_threshold=5.0,
                unit="incidents",
                frequency=MetricFrequency.DAY
            )
        ]
        
        for kpi in default_kpis:
            self.kpis[kpi.kpi_id] = kpi
    
    def _get_widget_data(self, widget: DashboardWidget) -> Dict[str, Any]:
        """Get data for dashboard widget."""
        # Check cache first
        cache_key = f"{widget.widget_id}_{hash(str(widget.query))}"
        if cache_key in self.widget_cache:
            cache_time = self.cache_timestamps.get(cache_key)
            if cache_time and (datetime.now() - cache_time).total_seconds() < self.cache_ttl:
                cached_data = self.widget_cache[cache_key]
                cached_data['widget_id'] = widget.widget_id
                cached_data['title'] = widget.title
                return cached_data
        
        # Fetch fresh data
        data = self._fetch_widget_data(widget.data_source, widget.query)
        
        # Generate chart
        chart_html = self.generate_chart(widget.chart_type, data, widget.config)
        
        widget_data = {
            'widget_id': widget.widget_id,
            'title': widget.title,
            'chart_type': widget.chart_type.value,
            'data': data,
            'chart': chart_html,
            'position': widget.position,
            'size': widget.size,
            'last_updated': datetime.now().isoformat()
        }
        
        # Cache the result
        self.widget_cache[cache_key] = widget_data
        self.cache_timestamps[cache_key] = datetime.now()
        
        return widget_data
    
    def _fetch_widget_data(self, data_source: str, query: Dict[str, Any]) -> Dict[str, Any]:
        """Fetch data for widget from data source."""
        try:
            if data_source == "system_metrics":
                return self._get_system_metrics_data(query)
            elif data_source == "api_metrics":
                return self._get_api_metrics_data(query)
            elif data_source == "detection_metrics":
                return self._get_detection_metrics_data(query)
            elif data_source == "security_metrics":
                return self._get_security_metrics_data(query)
            elif data_source == "tenant_metrics":
                return self._get_tenant_metrics_data(query)
            else:
                logger.warning(f"Unknown data source: {data_source}")
                return {"values": [], "labels": []}
                
        except Exception as e:
            logger.error(f"Failed to fetch widget data: {e}")
            return {"values": [], "labels": []}
    
    def _get_system_metrics_data(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """Get system metrics data."""
        metric = query.get("metric", "")
        
        if metric == "overall_health":
            # Calculate system health score
            health_score = 95.5  # Mock value
            return {"value": health_score, "status": "healthy"}
        
        elif metric == "uptime":
            return {"value": 99.9, "unit": "%"}
        
        else:
            return {"values": [90, 95, 98, 96, 99], "labels": ["Mon", "Tue", "Wed", "Thu", "Fri"]}
    
    def _get_api_metrics_data(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """Get API metrics data."""
        metric = query.get("metric", "")
        period = query.get("period", "24h")
        
        if metric == "request_count":
            # Mock request count data
            if period == "24h":
                return {
                    "values": [150, 200, 180, 220, 250, 300, 280],
                    "labels": ["00:00", "04:00", "08:00", "12:00", "16:00", "20:00", "24:00"]
                }
        
        elif metric == "response_time":
            return {
                "values": [120, 150, 140, 160, 130, 145, 135],
                "labels": ["00:00", "04:00", "08:00", "12:00", "16:00", "20:00", "24:00"],
                "unit": "ms"
            }
        
        return {"values": [], "labels": []}
    
    def _get_detection_metrics_data(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """Get detection metrics data."""
        metric = query.get("metric", "")
        
        if metric == "detection_rate":
            return {
                "values": [92, 94, 95, 93, 96, 94, 95],
                "labels": ["Day 1", "Day 2", "Day 3", "Day 4", "Day 5", "Day 6", "Day 7"],
                "unit": "%"
            }
        
        elif metric == "anomaly_count":
            return {
                "values": [15, 12, 18, 10, 20, 14, 16],
                "labels": ["Day 1", "Day 2", "Day 3", "Day 4", "Day 5", "Day 6", "Day 7"]
            }
        
        return {"values": [], "labels": []}
    
    def _get_security_metrics_data(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """Get security metrics data."""
        metric = query.get("metric", "")
        
        if metric == "threat_distribution":
            return {
                "values": [5, 3, 2, 8, 1],
                "labels": ["Brute Force", "Suspicious Access", "Data Exfiltration", "Failed Logins", "Other"]
            }
        
        elif metric == "failed_logins":
            return {
                "values": [8, 12, 6, 15, 9, 11, 7],
                "labels": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            }
        
        return {"values": [], "labels": []}
    
    def _get_tenant_metrics_data(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """Get tenant-specific metrics data."""
        # Mock tenant metrics
        return {
            "values": [100, 120, 110, 130, 125],
            "labels": ["Tenant A", "Tenant B", "Tenant C", "Tenant D", "Tenant E"]
        }
    
    def _collect_kpi_data(self, kpi: KPI):
        """Collect real-time KPI data."""
        while self.is_running and kpi.is_active:
            try:
                value = self._calculate_kpi_value(kpi)
                
                data_point = {
                    'timestamp': datetime.now().isoformat(),
                    'value': value,
                    'kpi_id': kpi.kpi_id
                }
                
                self.real_time_metrics[kpi.kpi_id].append(data_point)
                
                # Sleep based on frequency
                if kpi.frequency == MetricFrequency.REAL_TIME:
                    time.sleep(30)  # 30 second intervals for real-time
                elif kpi.frequency == MetricFrequency.MINUTE:
                    time.sleep(60)
                elif kpi.frequency == MetricFrequency.HOUR:
                    time.sleep(3600)
                
            except Exception as e:
                logger.error(f"KPI data collection failed for {kpi.kpi_id}: {e}")
                time.sleep(60)
    
    def _calculate_kpi_value(self, kpi: KPI) -> float:
        """Calculate current KPI value."""
        try:
            if kpi.metric_source == "system_metrics":
                if kpi.calculation == "uptime_percentage":
                    return 99.5  # Mock value
                elif kpi.calculation == "cpu_usage":
                    return 45.2
            
            elif kpi.metric_source == "api_metrics":
                if kpi.calculation == "avg_response_time":
                    if self.rate_limiter:
                        stats = self.rate_limiter.get_statistics()
                        return stats.get('avg_evaluation_time', 0.0) * 1000  # Convert to ms
                    return 150.0
            
            elif kpi.metric_source == "detection_metrics":
                if kpi.calculation == "accuracy_rate":
                    return 94.5
            
            elif kpi.metric_source == "security_metrics":
                if kpi.calculation == "incident_count":
                    if self.security_manager:
                        report = self.security_manager.get_security_report()
                        return len(report.get('recent_threats', []))
                    return 0.0
            
            return 0.0
            
        except Exception as e:
            logger.error(f"KPI calculation failed for {kpi.kpi_id}: {e}")
            return 0.0
    
    def _get_kpi_status(self, kpi: KPI, current_value: float) -> str:
        """Get KPI status based on thresholds."""
        if kpi.critical_threshold is not None:
            if (kpi.target_value is not None and current_value < kpi.target_value and 
                current_value <= kpi.critical_threshold):
                return "critical"
            elif (kpi.target_value is not None and current_value > kpi.target_value and 
                  current_value >= kpi.critical_threshold):
                return "critical"
        
        if kpi.warning_threshold is not None:
            if (kpi.target_value is not None and current_value < kpi.target_value and 
                current_value <= kpi.warning_threshold):
                return "warning"
            elif (kpi.target_value is not None and current_value > kpi.target_value and 
                  current_value >= kpi.warning_threshold):
                return "warning"
        
        return "normal"
    
    def _check_dashboard_access(self, dashboard: Dashboard, user_id: Optional[str]) -> bool:
        """Check if user has access to dashboard."""
        # Public dashboards are accessible to all
        if dashboard.is_public:
            return True
        
        # Check if user owns the dashboard
        if dashboard.created_by == user_id:
            return True
        
        # Additional access control logic would go here
        return True  # Default to allow for now
    
    def _get_system_health(self) -> Dict[str, Any]:
        """Get system health summary."""
        return {
            "overall_score": 95.5,
            "status": "healthy",
            "components": {
                "api": {"status": "healthy", "score": 98.0},
                "database": {"status": "healthy", "score": 96.0},
                "detection_engine": {"status": "healthy", "score": 94.0},
                "security": {"status": "warning", "score": 92.0}
            }
        }
    
    def _get_key_metrics(self, tenant_id: Optional[str]) -> Dict[str, Any]:
        """Get key business metrics."""
        return {
            "total_requests_24h": 15420,
            "anomalies_detected_24h": 48,
            "active_users": 156,
            "system_uptime": 99.9,
            "average_response_time": 145.6
        }
    
    def _get_active_alerts(self, tenant_id: Optional[str]) -> List[Dict[str, Any]]:
        """Get active alerts."""
        return [
            {
                "alert_id": "alert_001",
                "severity": "warning",
                "message": "API response time above threshold",
                "timestamp": datetime.now().isoformat()
            },
            {
                "alert_id": "alert_002", 
                "severity": "info",
                "message": "New anomaly pattern detected",
                "timestamp": (datetime.now() - timedelta(minutes=15)).isoformat()
            }
        ]
    
    def _get_performance_summary(self, tenant_id: Optional[str]) -> Dict[str, Any]:
        """Get performance summary."""
        return {
            "avg_response_time": 145.6,
            "throughput": 1245.8,
            "error_rate": 0.02,
            "cpu_usage": 45.2,
            "memory_usage": 68.9
        }
    
    def _get_compliance_summary(self, tenant_id: Optional[str]) -> Dict[str, Any]:
        """Get compliance summary."""
        return {
            "overall_score": 92.5,
            "frameworks": {
                "gdpr": {"score": 95.0, "status": "compliant"},
                "soc2": {"score": 90.0, "status": "compliant"},
                "hipaa": {"score": 88.0, "status": "warning"}
            },
            "violations": 2,
            "last_audit": "2024-01-15"
        }
    
    def _get_security_summary(self, tenant_id: Optional[str]) -> Dict[str, Any]:
        """Get security summary."""
        return {
            "threat_level": "low",
            "incidents_24h": 3,
            "blocked_attempts": 15,
            "security_score": 92.0,
            "last_scan": datetime.now().isoformat()
        }
    
    # Chart generation methods (simplified implementations)
    def _generate_line_chart(self, data: Dict[str, Any], config: Dict[str, Any]) -> str:
        """Generate line chart."""
        if not PLOTLY_AVAILABLE:
            return "<div>Chart unavailable (Plotly not installed)</div>"
        
        try:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=data.get('labels', []),
                y=data.get('values', []),
                mode='lines+markers',
                name=config.get('title', 'Data')
            ))
            
            fig.update_layout(
                title=config.get('title', ''),
                xaxis_title=config.get('x_title', ''),
                yaxis_title=config.get('y_title', ''),
                height=400
            )
            
            return fig.to_html()
            
        except Exception as e:
            logger.error(f"Line chart generation failed: {e}")
            return f"<div>Chart error: {e}</div>"
    
    def _generate_bar_chart(self, data: Dict[str, Any], config: Dict[str, Any]) -> str:
        """Generate bar chart."""
        if not PLOTLY_AVAILABLE:
            return "<div>Chart unavailable (Plotly not installed)</div>"
        
        try:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=data.get('labels', []),
                y=data.get('values', []),
                name=config.get('title', 'Data')
            ))
            
            fig.update_layout(
                title=config.get('title', ''),
                xaxis_title=config.get('x_title', ''),
                yaxis_title=config.get('y_title', ''),
                height=400
            )
            
            return fig.to_html()
            
        except Exception as e:
            logger.error(f"Bar chart generation failed: {e}")
            return f"<div>Chart error: {e}</div>"
    
    def _generate_pie_chart(self, data: Dict[str, Any], config: Dict[str, Any]) -> str:
        """Generate pie chart."""
        if not PLOTLY_AVAILABLE:
            return "<div>Chart unavailable (Plotly not installed)</div>"
        
        try:
            fig = go.Figure()
            fig.add_trace(go.Pie(
                labels=data.get('labels', []),
                values=data.get('values', []),
                name=config.get('title', 'Data')
            ))
            
            fig.update_layout(
                title=config.get('title', ''),
                height=400
            )
            
            return fig.to_html()
            
        except Exception as e:
            logger.error(f"Pie chart generation failed: {e}")
            return f"<div>Chart error: {e}</div>"
    
    def _generate_scatter_chart(self, data: Dict[str, Any], config: Dict[str, Any]) -> str:
        """Generate scatter chart."""
        if not PLOTLY_AVAILABLE:
            return "<div>Chart unavailable (Plotly not installed)</div>"
        
        try:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=data.get('x_values', data.get('labels', [])),
                y=data.get('y_values', data.get('values', [])),
                mode='markers',
                name=config.get('title', 'Data')
            ))
            
            fig.update_layout(
                title=config.get('title', ''),
                xaxis_title=config.get('x_title', ''),
                yaxis_title=config.get('y_title', ''),
                height=400
            )
            
            return fig.to_html()
            
        except Exception as e:
            logger.error(f"Scatter chart generation failed: {e}")
            return f"<div>Chart error: {e}</div>"
    
    def _generate_heatmap(self, data: Dict[str, Any], config: Dict[str, Any]) -> str:
        """Generate heatmap."""
        if not PLOTLY_AVAILABLE:
            return "<div>Chart unavailable (Plotly not installed)</div>"
        
        try:
            z_values = data.get('z_values', [[1, 2, 3], [4, 5, 6], [7, 8, 9]])
            
            fig = go.Figure()
            fig.add_trace(go.Heatmap(
                z=z_values,
                x=data.get('x_labels', []),
                y=data.get('y_labels', []),
                colorscale='Viridis'
            ))
            
            fig.update_layout(
                title=config.get('title', ''),
                height=400
            )
            
            return fig.to_html()
            
        except Exception as e:
            logger.error(f"Heatmap generation failed: {e}")
            return f"<div>Chart error: {e}</div>"
    
    def _generate_gauge_chart(self, data: Dict[str, Any], config: Dict[str, Any]) -> str:
        """Generate gauge chart."""
        if not PLOTLY_AVAILABLE:
            return "<div>Chart unavailable (Plotly not installed)</div>"
        
        try:
            value = data.get('value', 0)
            max_value = config.get('max_value', 100)
            
            fig = go.Figure()
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=value,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': config.get('title', 'Gauge')},
                gauge={
                    'axis': {'range': [None, max_value]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, max_value * 0.5], 'color': "lightgray"},
                        {'range': [max_value * 0.5, max_value * 0.8], 'color': "gray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': max_value * 0.9
                    }
                }
            ))
            
            fig.update_layout(height=400)
            
            return fig.to_html()
            
        except Exception as e:
            logger.error(f"Gauge chart generation failed: {e}")
            return f"<div>Chart error: {e}</div>"
    
    def _generate_table(self, data: Dict[str, Any], config: Dict[str, Any]) -> str:
        """Generate data table."""
        try:
            headers = data.get('headers', [])
            rows = data.get('rows', [])
            
            table_html = f"<table class='table table-striped'>"
            
            # Headers
            if headers:
                table_html += "<thead><tr>"
                for header in headers:
                    table_html += f"<th>{header}</th>"
                table_html += "</tr></thead>"
            
            # Rows
            table_html += "<tbody>"
            for row in rows:
                table_html += "<tr>"
                for cell in row:
                    table_html += f"<td>{cell}</td>"
                table_html += "</tr>"
            table_html += "</tbody></table>"
            
            return table_html
            
        except Exception as e:
            logger.error(f"Table generation failed: {e}")
            return f"<div>Table error: {e}</div>"


class ReportingService:
    """Advanced reporting service for enterprise analytics."""
    
    def __init__(self, dashboard: EnterpriseDashboard):
        """Initialize reporting service.
        
        Args:
            dashboard: Enterprise dashboard instance
        """
        self.dashboard = dashboard
        self.report_configs: Dict[str, ReportConfig] = {}
        self.scheduled_reports: Dict[str, Dict[str, Any]] = {}
        
        logger.info("Reporting Service initialized")
    
    def create_report(self, config: ReportConfig) -> bool:
        """Create new report configuration.
        
        Args:
            config: Report configuration
            
        Returns:
            Success status
        """
        try:
            self.report_configs[config.report_id] = config
            
            # Schedule if needed
            if config.schedule:
                self._schedule_report(config)
            
            logger.info(f"Report created: {config.report_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create report: {e}")
            return False
    
    def generate_report(self, report_id: str, custom_filters: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Generate report.
        
        Args:
            report_id: Report identifier
            custom_filters: Optional custom filters
            
        Returns:
            Generated report or None
        """
        try:
            config = self.report_configs.get(report_id)
            if not config:
                logger.error(f"Report configuration not found: {report_id}")
                return None
            
            # Merge filters
            filters = config.filters.copy()
            if custom_filters:
                filters.update(custom_filters)
            
            # Generate report data
            report_data = self._generate_report_data(config, filters)
            
            # Format report
            formatted_report = self._format_report(report_data, config)
            
            return formatted_report
            
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            return None
    
    def _generate_report_data(self, config: ReportConfig, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate report data based on configuration."""
        # This would typically query databases and aggregate data
        # For now, return mock data
        
        return {
            'timestamp': datetime.now().isoformat(),
            'filters': filters,
            'metrics': {
                'total_requests': 15420,
                'anomalies_detected': 48,
                'system_uptime': 99.9
            },
            'trends': {
                'request_growth': 12.5,
                'accuracy_improvement': 2.1
            }
        }
    
    def _format_report(self, data: Dict[str, Any], config: ReportConfig) -> Dict[str, Any]:
        """Format report data according to configuration."""
        return {
            'report_id': config.report_id,
            'name': config.name,
            'type': config.report_type.value,
            'generated_at': datetime.now().isoformat(),
            'data': data,
            'format': config.format_type
        }
    
    def _schedule_report(self, config: ReportConfig):
        """Schedule automatic report generation."""
        # This would integrate with a scheduler like APScheduler
        # For now, just store the schedule configuration
        self.scheduled_reports[config.report_id] = {
            'config': config,
            'next_run': datetime.now() + timedelta(hours=24)  # Default daily
        }