"""Advanced health monitoring service for anomaly detection system."""

import asyncio
import time
import psutil
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import structlog

logger = structlog.get_logger(__name__)


class HealthStatus(Enum):
    """System health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class HealthMetric:
    """Individual health metric."""
    name: str
    value: float
    unit: str
    status: HealthStatus
    threshold_warning: float
    threshold_critical: float
    timestamp: datetime
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['status'] = self.status.value
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class SystemAlert:
    """System alert information."""
    alert_id: str
    severity: AlertSeverity
    title: str
    message: str
    metric_name: str
    current_value: float
    threshold_value: float
    timestamp: datetime
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['severity'] = self.severity.value
        data['timestamp'] = self.timestamp.isoformat()
        if self.resolved_at:
            data['resolved_at'] = self.resolved_at.isoformat()
        return data


@dataclass
class HealthReport:
    """Comprehensive health report."""
    overall_status: HealthStatus
    overall_score: float
    metrics: List[HealthMetric]
    active_alerts: List[SystemAlert]
    performance_summary: Dict[str, Any]
    uptime_seconds: float
    timestamp: datetime
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'overall_status': self.overall_status.value,
            'overall_score': self.overall_score,
            'metrics': [metric.to_dict() for metric in self.metrics],
            'active_alerts': [alert.to_dict() for alert in self.active_alerts],
            'performance_summary': self.performance_summary,
            'uptime_seconds': self.uptime_seconds,
            'timestamp': self.timestamp.isoformat(),
            'recommendations': self.recommendations
        }


class PerformanceTracker:
    """Tracks performance metrics over time."""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.response_times: List[float] = []
        self.error_counts: List[int] = []
        self.throughput_data: List[float] = []
        self.timestamps: List[datetime] = []
        self._lock = asyncio.Lock()
    
    async def record_response_time(self, response_time_ms: float):
        """Record API response time."""
        async with self._lock:
            self.response_times.append(response_time_ms)
            self.timestamps.append(datetime.utcnow())
            self._trim_history()
    
    async def record_error(self):
        """Record an error occurrence."""
        async with self._lock:
            if self.error_counts:
                self.error_counts[-1] += 1
            else:
                self.error_counts.append(1)
    
    async def record_throughput(self, requests_per_second: float):
        """Record throughput measurement."""
        async with self._lock:
            self.throughput_data.append(requests_per_second)
            self._trim_history()
    
    def _trim_history(self):
        """Trim history to max_history length."""
        if len(self.response_times) > self.max_history:
            self.response_times = self.response_times[-self.max_history:]
        if len(self.error_counts) > self.max_history:
            self.error_counts = self.error_counts[-self.max_history:]
        if len(self.throughput_data) > self.max_history:
            self.throughput_data = self.throughput_data[-self.max_history:]
        if len(self.timestamps) > self.max_history:
            self.timestamps = self.timestamps[-self.max_history:]
    
    async def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics."""
        async with self._lock:
            summary = {
                'response_time_stats': {},
                'error_stats': {},
                'throughput_stats': {},
                'data_points': len(self.response_times)
            }
            
            if self.response_times:
                summary['response_time_stats'] = {
                    'avg_ms': statistics.mean(self.response_times),
                    'median_ms': statistics.median(self.response_times),
                    'min_ms': min(self.response_times),
                    'max_ms': max(self.response_times),
                    'p95_ms': self._percentile(self.response_times, 95),
                    'p99_ms': self._percentile(self.response_times, 99)
                }
            
            if self.error_counts:
                summary['error_stats'] = {
                    'total_errors': sum(self.error_counts),
                    'error_rate_percent': (sum(self.error_counts) / len(self.error_counts)) * 100,
                    'recent_errors': sum(self.error_counts[-10:])
                }
            
            if self.throughput_data:
                summary['throughput_stats'] = {
                    'avg_rps': statistics.mean(self.throughput_data),
                    'max_rps': max(self.throughput_data),
                    'min_rps': min(self.throughput_data)
                }
            
            return summary
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile value."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int((percentile / 100) * len(sorted_data))
        return sorted_data[min(index, len(sorted_data) - 1)]


class AlertManager:
    """Manages system alerts and notifications."""
    
    def __init__(self):
        self.active_alerts: Dict[str, SystemAlert] = {}
        self.alert_history: List[SystemAlert] = []
        self.alert_handlers: List[Callable[[SystemAlert], None]] = []
        self._lock = asyncio.Lock()
    
    def add_alert_handler(self, handler: Callable[[SystemAlert], None]):
        """Add alert notification handler."""
        self.alert_handlers.append(handler)
    
    async def create_alert(
        self,
        alert_id: str,
        severity: AlertSeverity,
        title: str,
        message: str,
        metric_name: str,
        current_value: float,
        threshold_value: float
    ) -> SystemAlert:
        """Create and register a new alert."""
        alert = SystemAlert(
            alert_id=alert_id,
            severity=severity,
            title=title,
            message=message,
            metric_name=metric_name,
            current_value=current_value,
            threshold_value=threshold_value,
            timestamp=datetime.utcnow()
        )
        
        async with self._lock:
            self.active_alerts[alert_id] = alert
            self.alert_history.append(alert)
            
            # Keep history limited
            if len(self.alert_history) > 1000:
                self.alert_history = self.alert_history[-1000:]
        
        # Notify handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error("Alert handler failed", error=str(e), alert_id=alert_id)
        
        logger.warning(
            "Alert created",
            alert_id=alert_id,
            severity=severity.value,
            title=title,
            metric=metric_name,
            value=current_value,
            threshold=threshold_value
        )
        
        return alert
    
    async def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an active alert."""
        async with self._lock:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.resolved = True
                alert.resolved_at = datetime.utcnow()
                del self.active_alerts[alert_id]
                
                logger.info("Alert resolved", alert_id=alert_id)
                return True
            return False
    
    async def get_active_alerts(self, severity_filter: Optional[AlertSeverity] = None) -> List[SystemAlert]:
        """Get active alerts, optionally filtered by severity."""
        async with self._lock:
            alerts = list(self.active_alerts.values())
            if severity_filter:
                alerts = [alert for alert in alerts if alert.severity == severity_filter]
            return sorted(alerts, key=lambda a: a.timestamp, reverse=True)
    
    async def get_alert_history(self, hours: int = 24) -> List[SystemAlert]:
        """Get alert history for specified time period."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        async with self._lock:
            return [
                alert for alert in self.alert_history
                if alert.timestamp >= cutoff_time
            ]


class HealthMonitoringService:
    """Advanced health monitoring service."""
    
    def __init__(self, check_interval: int = 30):
        self.check_interval = check_interval
        self.start_time = datetime.utcnow()
        self.performance_tracker = PerformanceTracker()
        self.alert_manager = AlertManager()
        self.monitoring_enabled = False
        self.monitoring_task: Optional[asyncio.Task] = None
        
        # Health thresholds
        self.thresholds = {
            'cpu_usage': {'warning': 70.0, 'critical': 90.0},
            'memory_usage': {'warning': 80.0, 'critical': 95.0},
            'disk_usage': {'warning': 85.0, 'critical': 95.0},
            'response_time': {'warning': 1000.0, 'critical': 3000.0},  # ms
            'error_rate': {'warning': 5.0, 'critical': 10.0},  # %
            'queue_length': {'warning': 50, 'critical': 100}
        }
        
        # Add default alert handler
        self.alert_manager.add_alert_handler(self._default_alert_handler)
    
    def _default_alert_handler(self, alert: SystemAlert):
        """Default alert handler that logs alerts."""
        logger.warning(
            "System Alert",
            alert_id=alert.alert_id,
            severity=alert.severity.value,
            title=alert.title,
            message=alert.message
        )
    
    async def start_monitoring(self):
        """Start continuous health monitoring."""
        if self.monitoring_enabled:
            return
        
        self.monitoring_enabled = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Health monitoring started", interval=self.check_interval)
    
    async def stop_monitoring(self):
        """Stop health monitoring."""
        self.monitoring_enabled = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("Health monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_enabled:
            try:
                # Perform health checks
                await self._perform_health_checks()
                
                # Wait for next check
                await asyncio.sleep(self.check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Health monitoring loop error", error=str(e))
                await asyncio.sleep(self.check_interval)
    
    async def _perform_health_checks(self):
        """Perform comprehensive health checks."""
        try:
            # System resource checks
            await self._check_system_resources()
            
            # Performance checks
            await self._check_performance_metrics()
            
            # Application-specific checks
            await self._check_application_health()
            
        except Exception as e:
            logger.error("Health check failed", error=str(e))
    
    async def _check_system_resources(self):
        """Check system resource utilization."""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        await self._evaluate_metric(
            "cpu_usage", cpu_percent, "%",
            "CPU utilization percentage"
        )
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        await self._evaluate_metric(
            "memory_usage", memory_percent, "%",
            "Memory utilization percentage"
        )
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        await self._evaluate_metric(
            "disk_usage", disk_percent, "%",
            "Disk utilization percentage"
        )
    
    async def _check_performance_metrics(self):
        """Check application performance metrics."""
        perf_summary = await self.performance_tracker.get_performance_summary()
        
        # Response time check
        if perf_summary.get('response_time_stats'):
            avg_response_time = perf_summary['response_time_stats']['avg_ms']
            await self._evaluate_metric(
                "response_time", avg_response_time, "ms",
                "Average API response time"
            )
        
        # Error rate check
        if perf_summary.get('error_stats'):
            error_rate = perf_summary['error_stats']['error_rate_percent']
            await self._evaluate_metric(
                "error_rate", error_rate, "%",
                "Application error rate"
            )
    
    async def _check_application_health(self):
        """Check application-specific health metrics."""
        # This would be extended with specific application metrics
        # For now, we'll add some mock checks
        
        # Mock queue length check
        import random
        queue_length = random.randint(0, 80)  # Mock queue length
        await self._evaluate_metric(
            "queue_length", queue_length, "jobs",
            "Job queue length"
        )
    
    async def _evaluate_metric(
        self,
        metric_name: str,
        value: float,
        unit: str,
        description: str
    ):
        """Evaluate a metric against thresholds and create alerts if needed."""
        thresholds = self.thresholds.get(metric_name, {})
        warning_threshold = thresholds.get('warning', float('inf'))
        critical_threshold = thresholds.get('critical', float('inf'))
        
        # Determine status
        if value >= critical_threshold:
            status = HealthStatus.CRITICAL
        elif value >= warning_threshold:
            status = HealthStatus.WARNING
        else:
            status = HealthStatus.HEALTHY
        
        # Create or resolve alerts
        alert_id = f"{metric_name}_threshold"
        
        if status in [HealthStatus.WARNING, HealthStatus.CRITICAL]:
            # Check if alert already exists
            active_alerts = await self.alert_manager.get_active_alerts()
            existing_alert = next((a for a in active_alerts if a.alert_id == alert_id), None)
            
            if not existing_alert:
                severity = AlertSeverity.CRITICAL if status == HealthStatus.CRITICAL else AlertSeverity.WARNING
                threshold = critical_threshold if status == HealthStatus.CRITICAL else warning_threshold
                
                await self.alert_manager.create_alert(
                    alert_id=alert_id,
                    severity=severity,
                    title=f"{metric_name.replace('_', ' ').title()} Threshold Exceeded",
                    message=f"{description} is {value:.1f}{unit}, exceeding {severity.value} threshold of {threshold:.1f}{unit}",
                    metric_name=metric_name,
                    current_value=value,
                    threshold_value=threshold
                )
        else:
            # Resolve alert if metric is healthy
            await self.alert_manager.resolve_alert(alert_id)
    
    async def get_health_report(self, include_history: bool = False) -> HealthReport:
        """Generate comprehensive health report."""
        # Collect current metrics
        metrics = await self._collect_current_metrics()
        
        # Get active alerts
        active_alerts = await self.alert_manager.get_active_alerts()
        
        # Calculate overall health score
        overall_score = self._calculate_overall_score(metrics)
        overall_status = self._determine_overall_status(metrics, active_alerts)
        
        # Get performance summary
        performance_summary = await self.performance_tracker.get_performance_summary()
        
        # Calculate uptime
        uptime_seconds = (datetime.utcnow() - self.start_time).total_seconds()
        
        # Generate recommendations
        recommendations = self._generate_recommendations(metrics, active_alerts)
        
        return HealthReport(
            overall_status=overall_status,
            overall_score=overall_score,
            metrics=metrics,
            active_alerts=active_alerts,
            performance_summary=performance_summary,
            uptime_seconds=uptime_seconds,
            timestamp=datetime.utcnow(),
            recommendations=recommendations
        )
    
    async def _collect_current_metrics(self) -> List[HealthMetric]:
        """Collect current system metrics."""
        metrics = []
        
        # System metrics
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        metrics.extend([
            HealthMetric(
                name="cpu_usage",
                value=cpu_percent,
                unit="%",
                status=self._get_metric_status("cpu_usage", cpu_percent),
                threshold_warning=self.thresholds["cpu_usage"]["warning"],
                threshold_critical=self.thresholds["cpu_usage"]["critical"],
                timestamp=datetime.utcnow(),
                description="CPU utilization percentage"
            ),
            HealthMetric(
                name="memory_usage",
                value=memory.percent,
                unit="%",
                status=self._get_metric_status("memory_usage", memory.percent),
                threshold_warning=self.thresholds["memory_usage"]["warning"],
                threshold_critical=self.thresholds["memory_usage"]["critical"],
                timestamp=datetime.utcnow(),
                description="Memory utilization percentage"
            ),
            HealthMetric(
                name="disk_usage",
                value=(disk.used / disk.total) * 100,
                unit="%",
                status=self._get_metric_status("disk_usage", (disk.used / disk.total) * 100),
                threshold_warning=self.thresholds["disk_usage"]["warning"],
                threshold_critical=self.thresholds["disk_usage"]["critical"],
                timestamp=datetime.utcnow(),
                description="Disk utilization percentage"
            )
        ])
        
        # Performance metrics
        perf_summary = await self.performance_tracker.get_performance_summary()
        if perf_summary.get('response_time_stats'):
            avg_response_time = perf_summary['response_time_stats']['avg_ms']
            metrics.append(
                HealthMetric(
                    name="response_time",
                    value=avg_response_time,
                    unit="ms",
                    status=self._get_metric_status("response_time", avg_response_time),
                    threshold_warning=self.thresholds["response_time"]["warning"],
                    threshold_critical=self.thresholds["response_time"]["critical"],
                    timestamp=datetime.utcnow(),
                    description="Average API response time"
                )
            )
        
        return metrics
    
    def _get_metric_status(self, metric_name: str, value: float) -> HealthStatus:
        """Get health status for a metric value."""
        thresholds = self.thresholds.get(metric_name, {})
        warning_threshold = thresholds.get('warning', float('inf'))
        critical_threshold = thresholds.get('critical', float('inf'))
        
        if value >= critical_threshold:
            return HealthStatus.CRITICAL
        elif value >= warning_threshold:
            return HealthStatus.WARNING
        else:
            return HealthStatus.HEALTHY
    
    def _calculate_overall_score(self, metrics: List[HealthMetric]) -> float:
        """Calculate overall health score (0-100)."""
        if not metrics:
            return 0.0
        
        total_score = 0.0
        for metric in metrics:
            if metric.status == HealthStatus.HEALTHY:
                total_score += 100
            elif metric.status == HealthStatus.WARNING:
                total_score += 70
            elif metric.status == HealthStatus.CRITICAL:
                total_score += 30
            else:  # UNKNOWN
                total_score += 50
        
        return total_score / len(metrics)
    
    def _determine_overall_status(
        self,
        metrics: List[HealthMetric],
        alerts: List[SystemAlert]
    ) -> HealthStatus:
        """Determine overall system health status."""
        # Check for critical alerts
        critical_alerts = [a for a in alerts if a.severity == AlertSeverity.CRITICAL]
        if critical_alerts:
            return HealthStatus.CRITICAL
        
        # Check for critical metrics
        critical_metrics = [m for m in metrics if m.status == HealthStatus.CRITICAL]
        if critical_metrics:
            return HealthStatus.CRITICAL
        
        # Check for warning conditions
        warning_alerts = [a for a in alerts if a.severity == AlertSeverity.WARNING]
        warning_metrics = [m for m in metrics if m.status == HealthStatus.WARNING]
        
        if warning_alerts or warning_metrics:
            return HealthStatus.WARNING
        
        return HealthStatus.HEALTHY
    
    def _generate_recommendations(
        self,
        metrics: List[HealthMetric],
        alerts: List[SystemAlert]
    ) -> List[str]:
        """Generate health improvement recommendations."""
        recommendations = []
        
        for metric in metrics:
            if metric.status == HealthStatus.CRITICAL:
                if metric.name == "cpu_usage":
                    recommendations.append("Consider scaling up compute resources or optimizing CPU-intensive operations")
                elif metric.name == "memory_usage":
                    recommendations.append("Monitor memory leaks and consider increasing available memory")
                elif metric.name == "disk_usage":
                    recommendations.append("Clean up temporary files or increase disk capacity")
                elif metric.name == "response_time":
                    recommendations.append("Optimize slow API endpoints and consider caching strategies")
            elif metric.status == HealthStatus.WARNING:
                if metric.name == "cpu_usage":
                    recommendations.append("Monitor CPU usage trends and prepare for scaling")
                elif metric.name == "memory_usage":
                    recommendations.append("Review memory usage patterns and optimize if possible")
        
        # Remove duplicates and limit recommendations
        recommendations = list(set(recommendations))[:5]
        
        if not recommendations:
            recommendations.append("System is operating within normal parameters")
        
        return recommendations
    
    async def record_api_call(self, response_time_ms: float, success: bool = True):
        """Record API call metrics."""
        await self.performance_tracker.record_response_time(response_time_ms)
        if not success:
            await self.performance_tracker.record_error()
    
    async def get_alert_history(self, hours: int = 24) -> List[SystemAlert]:
        """Get alert history."""
        return await self.alert_manager.get_alert_history(hours)
    
    def set_threshold(self, metric_name: str, warning: float, critical: float):
        """Set custom thresholds for a metric."""
        self.thresholds[metric_name] = {
            'warning': warning,
            'critical': critical
        }
    
    def add_alert_handler(self, handler: Callable[[SystemAlert], None]):
        """Add custom alert handler."""
        self.alert_manager.add_alert_handler(handler)