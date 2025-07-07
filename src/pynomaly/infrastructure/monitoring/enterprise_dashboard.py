"""Enterprise monitoring dashboard and metrics aggregation service."""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from pynomaly.shared.config import Config
from .opentelemetry_service import get_telemetry_service


logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics tracked by the dashboard."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class AlertSeverity(Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Metric:
    """Individual metric data point."""
    name: str
    value: Union[int, float]
    metric_type: MetricType
    timestamp: datetime
    labels: Dict[str, str]
    unit: Optional[str] = None
    description: Optional[str] = None


@dataclass
class Alert:
    """Alert data structure."""
    id: str
    metric_name: str
    severity: AlertSeverity
    threshold: Union[int, float]
    current_value: Union[int, float]
    message: str
    timestamp: datetime
    resolved: bool = False
    resolution_time: Optional[datetime] = None


@dataclass
class DashboardData:
    """Complete dashboard data structure."""
    system_metrics: Dict[str, Any]
    business_metrics: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    alerts: List[Alert]
    last_updated: datetime
    uptime_seconds: float


class EnterpriseDashboard:
    """Enterprise monitoring dashboard service."""
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize enterprise dashboard."""
        self.config = config or Config()
        self.metrics_store: Dict[str, List[Metric]] = {}
        self.alerts: List[Alert] = []
        self.start_time = time.time()
        self.telemetry = get_telemetry_service()
        
        # Configuration
        self.retention_hours = self.config.get("dashboard.retention_hours", 24)
        self.alert_thresholds = self._load_alert_thresholds()
        self.refresh_interval = self.config.get("dashboard.refresh_interval", 30)
        
        # Performance tracking
        self.performance_history: List[Dict[str, Any]] = []
        self.max_history_size = self.config.get("dashboard.max_history_size", 1000)
        
        logger.info("Enterprise dashboard initialized")
    
    def _load_alert_thresholds(self) -> Dict[str, Dict[str, Union[int, float]]]:
        """Load alert thresholds from configuration."""
        return self.config.get("dashboard.alert_thresholds", {
            "cpu_usage": {"high": 80.0, "critical": 95.0},
            "memory_usage": {"high": 85.0, "critical": 95.0},
            "disk_usage": {"high": 80.0, "critical": 90.0},
            "detection_latency": {"high": 5.0, "critical": 10.0},
            "error_rate": {"high": 0.05, "critical": 0.1},
            "queue_depth": {"high": 100, "critical": 500},
        })
    
    def record_metric(self, metric: Metric) -> None:
        """Record a metric in the dashboard."""
        with self.telemetry.trace_operation("dashboard.record_metric") as span:
            if span:
                span.set_attribute("metric.name", metric.name)
                span.set_attribute("metric.type", metric.metric_type.value)
            
            if metric.name not in self.metrics_store:
                self.metrics_store[metric.name] = []
            
            self.metrics_store[metric.name].append(metric)
            
            # Check for alerts
            self._check_alert_conditions(metric)
            
            # Clean up old metrics
            self._cleanup_old_metrics()
    
    def record_business_metric(
        self, 
        name: str, 
        value: Union[int, float],
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Record a business metric."""
        metric = Metric(
            name=name,
            value=value,
            metric_type=MetricType.GAUGE,
            timestamp=datetime.now(),
            labels=labels or {},
            description=f"Business metric: {name}"
        )
        self.record_metric(metric)
    
    def record_performance_metric(
        self, 
        operation: str, 
        duration: float,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Record a performance metric."""
        metric = Metric(
            name=f"performance.{operation}.duration",
            value=duration,
            metric_type=MetricType.HISTOGRAM,
            timestamp=datetime.now(),
            labels=labels or {},
            unit="seconds",
            description=f"Performance metric for {operation}"
        )
        self.record_metric(metric)
        
        # Store in performance history for trend analysis
        self.performance_history.append({
            "operation": operation,
            "duration": duration,
            "timestamp": datetime.now().isoformat(),
            "labels": labels or {}
        })
        
        # Limit history size
        if len(self.performance_history) > self.max_history_size:
            self.performance_history = self.performance_history[-self.max_history_size:]
    
    def _check_alert_conditions(self, metric: Metric) -> None:
        """Check if metric triggers any alerts."""
        thresholds = self.alert_thresholds.get(metric.name)
        if not thresholds:
            return
        
        current_value = metric.value
        alert_triggered = False
        severity = AlertSeverity.LOW
        
        # Check thresholds in order of severity
        if "critical" in thresholds and current_value >= thresholds["critical"]:
            severity = AlertSeverity.CRITICAL
            alert_triggered = True
        elif "high" in thresholds and current_value >= thresholds["high"]:
            severity = AlertSeverity.HIGH
            alert_triggered = True
        elif "medium" in thresholds and current_value >= thresholds["medium"]:
            severity = AlertSeverity.MEDIUM
            alert_triggered = True
        
        if alert_triggered:
            alert = Alert(
                id=f"{metric.name}_{int(time.time())}",
                metric_name=metric.name,
                severity=severity,
                threshold=thresholds[severity.value],
                current_value=current_value,
                message=f"{metric.name} value {current_value} exceeds {severity.value} threshold {thresholds[severity.value]}",
                timestamp=datetime.now()
            )
            self.alerts.append(alert)
            
            # Log alert
            logger.warning(f"Alert triggered: {alert.message}")
            
            # Record alert metric
            self.telemetry.record_detection_metrics(
                duration=0,
                anomaly_count=1,
                algorithm="alert_system"
            )
    
    def _cleanup_old_metrics(self) -> None:
        """Remove metrics older than retention period."""
        cutoff_time = datetime.now() - timedelta(hours=self.retention_hours)
        
        for metric_name in list(self.metrics_store.keys()):
            self.metrics_store[metric_name] = [
                m for m in self.metrics_store[metric_name] 
                if m.timestamp > cutoff_time
            ]
            
            # Remove empty metric lists
            if not self.metrics_store[metric_name]:
                del self.metrics_store[metric_name]
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics."""
        metrics = {}
        
        if PSUTIL_AVAILABLE:
            try:
                # CPU metrics
                cpu_percent = psutil.cpu_percent(interval=0.1)
                metrics["cpu_usage"] = cpu_percent
                metrics["cpu_count"] = psutil.cpu_count()
                metrics["cpu_freq"] = psutil.cpu_freq().current if psutil.cpu_freq() else 0
                
                # Memory metrics
                memory = psutil.virtual_memory()
                metrics["memory_usage"] = memory.percent
                metrics["memory_total"] = memory.total
                metrics["memory_available"] = memory.available
                metrics["memory_used"] = memory.used
                
                # Disk metrics
                disk = psutil.disk_usage('/')
                metrics["disk_usage"] = (disk.used / disk.total) * 100
                metrics["disk_total"] = disk.total
                metrics["disk_free"] = disk.free
                metrics["disk_used"] = disk.used
                
                # Network metrics
                network = psutil.net_io_counters()
                metrics["network_bytes_sent"] = network.bytes_sent
                metrics["network_bytes_recv"] = network.bytes_recv
                metrics["network_packets_sent"] = network.packets_sent
                metrics["network_packets_recv"] = network.packets_recv
                
                # Process metrics
                process = psutil.Process()
                metrics["process_memory"] = process.memory_info().rss
                metrics["process_cpu"] = process.cpu_percent()
                metrics["process_threads"] = process.num_threads()
                
            except Exception as e:
                logger.warning(f"Error collecting system metrics: {e}")
                metrics["error"] = str(e)
        else:
            metrics["error"] = "psutil not available"
        
        return metrics
    
    def get_business_metrics(self) -> Dict[str, Any]:
        """Get business-specific metrics."""
        metrics = {}
        
        # Aggregate metrics from store
        for metric_name, metric_list in self.metrics_store.items():
            if metric_name.startswith("business.") or metric_name in [
                "detections_total", "model_accuracy", "anomaly_score"
            ]:
                if metric_list:
                    latest = metric_list[-1]
                    metrics[metric_name] = {
                        "value": latest.value,
                        "timestamp": latest.timestamp.isoformat(),
                        "labels": latest.labels
                    }
        
        # Calculate aggregated metrics
        if "detections_total" in metrics:
            # Detection rate (per hour)
            hour_ago = datetime.now() - timedelta(hours=1)
            recent_detections = [
                m for m in self.metrics_store.get("detections_total", [])
                if m.timestamp > hour_ago
            ]
            metrics["detection_rate_per_hour"] = len(recent_detections)
        
        return metrics
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics and trends."""
        metrics = {}
        
        # Aggregate performance data
        if self.performance_history:
            # Group by operation
            operations = {}
            for entry in self.performance_history:
                op = entry["operation"]
                if op not in operations:
                    operations[op] = []
                operations[op].append(entry["duration"])
            
            # Calculate statistics for each operation
            for op, durations in operations.items():
                if durations:
                    metrics[f"{op}_avg_duration"] = sum(durations) / len(durations)
                    metrics[f"{op}_max_duration"] = max(durations)
                    metrics[f"{op}_min_duration"] = min(durations)
                    metrics[f"{op}_count"] = len(durations)
                    
                    # Calculate percentiles
                    sorted_durations = sorted(durations)
                    p50_idx = int(len(sorted_durations) * 0.5)
                    p95_idx = int(len(sorted_durations) * 0.95)
                    p99_idx = int(len(sorted_durations) * 0.99)
                    
                    metrics[f"{op}_p50_duration"] = sorted_durations[p50_idx]
                    metrics[f"{op}_p95_duration"] = sorted_durations[p95_idx]
                    metrics[f"{op}_p99_duration"] = sorted_durations[p99_idx]
        
        # Add uptime
        metrics["uptime_seconds"] = time.time() - self.start_time
        
        return metrics
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get list of active alerts."""
        active_alerts = [alert for alert in self.alerts if not alert.resolved]
        return [asdict(alert) for alert in active_alerts]
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an active alert."""
        for alert in self.alerts:
            if alert.id == alert_id and not alert.resolved:
                alert.resolved = True
                alert.resolution_time = datetime.now()
                logger.info(f"Alert {alert_id} resolved")
                return True
        return False
    
    def get_dashboard_data(self) -> DashboardData:
        """Get complete dashboard data."""
        with self.telemetry.trace_operation("dashboard.get_data") as span:
            system_metrics = self.get_system_metrics()
            business_metrics = self.get_business_metrics()
            performance_metrics = self.get_performance_metrics()
            active_alerts = self.get_active_alerts()
            
            # Record system metrics in telemetry
            for name, value in system_metrics.items():
                if isinstance(value, (int, float)) and name in ["cpu_usage", "memory_usage"]:
                    metric = Metric(
                        name=name,
                        value=value,
                        metric_type=MetricType.GAUGE,
                        timestamp=datetime.now(),
                        labels={"source": "system"}
                    )
                    self.record_metric(metric)
            
            dashboard_data = DashboardData(
                system_metrics=system_metrics,
                business_metrics=business_metrics,
                performance_metrics=performance_metrics,
                alerts=active_alerts,
                last_updated=datetime.now(),
                uptime_seconds=time.time() - self.start_time
            )
            
            if span:
                span.set_attribute("dashboard.metrics_count", len(self.metrics_store))
                span.set_attribute("dashboard.alerts_count", len(active_alerts))
            
            return dashboard_data
    
    def export_metrics(self, format_type: str = "json") -> str:
        """Export metrics in specified format."""
        dashboard_data = self.get_dashboard_data()
        
        if format_type.lower() == "json":
            return json.dumps(asdict(dashboard_data), default=str, indent=2)
        elif format_type.lower() == "prometheus":
            return self._export_prometheus_format()
        else:
            raise ValueError(f"Unsupported export format: {format_type}")
    
    def _export_prometheus_format(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        
        for metric_name, metric_list in self.metrics_store.items():
            if not metric_list:
                continue
                
            latest = metric_list[-1]
            
            # Add help and type comments
            lines.append(f"# HELP {metric_name} {latest.description or 'No description'}")
            lines.append(f"# TYPE {metric_name} {latest.metric_type.value}")
            
            # Add metric value with labels
            labels_str = ""
            if latest.labels:
                label_pairs = [f'{k}="{v}"' for k, v in latest.labels.items()]
                labels_str = "{" + ",".join(label_pairs) + "}"
            
            lines.append(f"{metric_name}{labels_str} {latest.value}")
        
        return "\n".join(lines)
    
    async def start_monitoring_loop(self) -> None:
        """Start the monitoring loop for continuous metric collection."""
        logger.info("Starting enterprise monitoring loop")
        
        while True:
            try:
                # Collect system metrics
                system_metrics = self.get_system_metrics()
                
                # Record key system metrics
                for name, value in system_metrics.items():
                    if isinstance(value, (int, float)) and name in [
                        "cpu_usage", "memory_usage", "disk_usage", 
                        "process_memory", "process_cpu"
                    ]:
                        metric = Metric(
                            name=name,
                            value=value,
                            metric_type=MetricType.GAUGE,
                            timestamp=datetime.now(),
                            labels={"source": "system", "instance": "local"}
                        )
                        self.record_metric(metric)
                
                await asyncio.sleep(self.refresh_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.refresh_interval)
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status."""
        dashboard_data = self.get_dashboard_data()
        
        # Determine health based on alerts and metrics
        critical_alerts = [a for a in dashboard_data.alerts if a.get("severity") == "critical"]
        high_alerts = [a for a in dashboard_data.alerts if a.get("severity") == "high"]
        
        if critical_alerts:
            health_status = "critical"
            health_message = f"{len(critical_alerts)} critical alerts active"
        elif high_alerts:
            health_status = "degraded"
            health_message = f"{len(high_alerts)} high-priority alerts active"
        elif dashboard_data.alerts:
            health_status = "warning"
            health_message = f"{len(dashboard_data.alerts)} alerts active"
        else:
            health_status = "healthy"
            health_message = "All systems operational"
        
        return {
            "status": health_status,
            "message": health_message,
            "uptime_seconds": dashboard_data.uptime_seconds,
            "last_updated": dashboard_data.last_updated.isoformat(),
            "metrics_collected": len(self.metrics_store),
            "active_alerts": len(dashboard_data.alerts)
        }


# Global dashboard instance
_enterprise_dashboard: Optional[EnterpriseDashboard] = None


def get_enterprise_dashboard(config: Optional[Config] = None) -> EnterpriseDashboard:
    """Get the global enterprise dashboard instance."""
    global _enterprise_dashboard
    if _enterprise_dashboard is None:
        _enterprise_dashboard = EnterpriseDashboard(config)
    return _enterprise_dashboard


def record_business_metric(name: str, value: Union[int, float], labels: Optional[Dict[str, str]] = None) -> None:
    """Convenience function to record business metrics."""
    dashboard = get_enterprise_dashboard()
    dashboard.record_business_metric(name, value, labels)


def record_performance_metric(operation: str, duration: float, labels: Optional[Dict[str, str]] = None) -> None:
    """Convenience function to record performance metrics."""
    dashboard = get_enterprise_dashboard()
    dashboard.record_performance_metric(operation, duration, labels)