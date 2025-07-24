"""Enhanced analytics service combining dashboard insights and health monitoring."""

import asyncio
import json
import time
import psutil
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from enum import Enum

import numpy as np
import pandas as pd
import structlog

from ..entities.detection_result import DetectionResult
from ...infrastructure.monitoring.metrics_collector import get_metrics_collector, MetricsCollector

logger = structlog.get_logger(__name__)


# ==============================================================================
# ENUMS AND DATA CLASSES
# ==============================================================================

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
class PerformanceMetrics:
    """Performance metrics for the system."""
    total_detections: int = 0
    total_anomalies: int = 0
    average_detection_time: float = 0.0
    success_rate: float = 0.0
    throughput: float = 0.0  # detections per second
    error_rate: float = 0.0


@dataclass
class AlgorithmStats:
    """Statistics for a specific algorithm."""
    algorithm: str
    detections_count: int = 0
    anomalies_found: int = 0
    average_score: float = 0.0
    average_time: float = 0.0
    success_rate: float = 0.0
    last_used: Optional[datetime] = None


@dataclass
class DataQualityMetrics:
    """Data quality metrics."""
    total_samples: int = 0
    missing_values: int = 0
    duplicate_samples: int = 0
    outliers_count: int = 0
    data_drift_events: int = 0


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
    id: str
    severity: AlertSeverity
    title: str
    message: str
    component: str
    timestamp: datetime
    resolved: bool = False
    resolution_time: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['severity'] = self.severity.value
        data['timestamp'] = self.timestamp.isoformat()
        if self.resolution_time:
            data['resolution_time'] = self.resolution_time.isoformat()
        return data


class EnhancedAnalyticsService:
    """
    Enhanced analytics service combining:
    - Dashboard analytics and reporting
    - System health monitoring
    - Performance tracking
    - Alert management
    """
    
    def __init__(
        self,
        metrics_collector: Optional[MetricsCollector] = None,
        health_check_interval: int = 60,
        history_retention_days: int = 30
    ):
        # Analytics components
        self.metrics_collector = metrics_collector or get_metrics_collector()
        self.performance_metrics = PerformanceMetrics()
        self.algorithm_stats: Dict[str, AlgorithmStats] = {}
        self.data_quality_metrics = DataQualityMetrics()
        
        # Health monitoring components
        self.health_check_interval = health_check_interval
        self.history_retention_days = history_retention_days
        self._health_metrics: Dict[str, HealthMetric] = {}
        self._system_alerts: List[SystemAlert] = []
        self._alert_counter = 0
        
        # Historical data storage
        self._detection_history: deque = deque(maxlen=10000)
        self._performance_history: deque = deque(maxlen=1000)
        self._health_history: deque = deque(maxlen=1000)
        
        # Monitoring state
        self._monitoring_active = False
        self._monitoring_task: Optional[asyncio.Task] = None
        
        # Thresholds for health monitoring
        self._health_thresholds = {
            'cpu_usage': {'warning': 70.0, 'critical': 90.0},
            'memory_usage': {'warning': 80.0, 'critical': 95.0},
            'disk_usage': {'warning': 80.0, 'critical': 95.0},
            'detection_latency': {'warning': 1000.0, 'critical': 5000.0},  # ms
            'error_rate': {'warning': 5.0, 'critical': 10.0},  # percentage
            'throughput': {'warning': 10.0, 'critical': 5.0}  # requests/sec (lower is worse)
        }

    # ==============================================================================
    # ANALYTICS METHODS
    # ==============================================================================
    
    def record_detection(
        self,
        result: DetectionResult,
        processing_time: float,
        data_size: int,
        success: bool = True
    ) -> None:
        """Record a detection event for analytics."""
        
        # Update performance metrics
        self.performance_metrics.total_detections += 1
        if hasattr(result, 'is_anomaly') and any(result.is_anomaly):
            self.performance_metrics.total_anomalies += len([x for x in result.is_anomaly if x])
        
        # Update timing metrics
        current_avg = self.performance_metrics.average_detection_time
        total_detections = self.performance_metrics.total_detections
        self.performance_metrics.average_detection_time = (
            (current_avg * (total_detections - 1) + processing_time) / total_detections
        )
        
        # Update success rate
        if success:
            success_count = self.performance_metrics.success_rate * (total_detections - 1) + 1
            self.performance_metrics.success_rate = success_count / total_detections
        else:
            success_count = self.performance_metrics.success_rate * (total_detections - 1)
            self.performance_metrics.success_rate = success_count / total_detections
        
        # Update algorithm-specific stats
        algorithm = getattr(result, 'algorithm', 'unknown')
        if algorithm not in self.algorithm_stats:
            self.algorithm_stats[algorithm] = AlgorithmStats(algorithm=algorithm)
        
        algo_stats = self.algorithm_stats[algorithm]
        algo_stats.detections_count += 1
        if hasattr(result, 'is_anomaly') and any(result.is_anomaly):
            algo_stats.anomalies_found += len([x for x in result.is_anomaly if x])
        
        # Update algorithm timing
        current_algo_avg = algo_stats.average_time
        algo_stats.average_time = (
            (current_algo_avg * (algo_stats.detections_count - 1) + processing_time) / 
            algo_stats.detections_count
        )
        algo_stats.last_used = datetime.utcnow()
        
        # Record in history
        detection_record = {
            'timestamp': datetime.utcnow(),
            'algorithm': algorithm,
            'processing_time': processing_time,
            'data_size': data_size,
            'anomalies_found': len([x for x in result.is_anomaly if x]) if hasattr(result, 'is_anomaly') else 0,
            'success': success
        }
        self._detection_history.append(detection_record)
        
        logger.debug("Detection recorded", 
                    algorithm=algorithm,
                    processing_time=processing_time,
                    success=success)

    def get_dashboard_data(self, time_range: Optional[timedelta] = None) -> Dict[str, Any]:
        """Get comprehensive dashboard data."""
        
        if time_range is None:
            time_range = timedelta(hours=24)
        
        cutoff_time = datetime.utcnow() - time_range
        
        # Filter historical data by time range
        recent_detections = [
            record for record in self._detection_history
            if record['timestamp'] >= cutoff_time
        ]
        
        # Calculate time-based metrics
        time_based_metrics = self._calculate_time_based_metrics(recent_detections, time_range)
        
        dashboard_data = {
            'overview': {
                'total_detections': self.performance_metrics.total_detections,
                'total_anomalies': self.performance_metrics.total_anomalies,
                'success_rate': round(self.performance_metrics.success_rate * 100, 2),
                'average_detection_time': round(self.performance_metrics.average_detection_time, 2),
                'error_rate': round(self.performance_metrics.error_rate, 2)
            },
            'time_based': time_based_metrics,
            'algorithms': {
                name: {
                    'detections_count': stats.detections_count,
                    'anomalies_found': stats.anomalies_found,
                    'average_time': round(stats.average_time, 2),
                    'success_rate': round(stats.success_rate * 100, 2),
                    'last_used': stats.last_used.isoformat() if stats.last_used else None
                }
                for name, stats in self.algorithm_stats.items()
            },
            'data_quality': asdict(self.data_quality_metrics),
            'system_health': self.get_health_summary(),
            'alerts': self.get_active_alerts(),
            'timestamp': datetime.utcnow().isoformat()
        }
        
        return dashboard_data

    def get_performance_trends(self, hours: int = 24) -> Dict[str, Any]:
        """Get performance trends over time."""
        
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        recent_detections = [
            record for record in self._detection_history
            if record['timestamp'] >= cutoff_time
        ]
        
        # Group by hour
        hourly_data = defaultdict(list)
        for record in recent_detections:
            hour_key = record['timestamp'].replace(minute=0, second=0, microsecond=0)
            hourly_data[hour_key].append(record)
        
        trends = {
            'hourly_throughput': [],
            'hourly_avg_latency': [],
            'hourly_anomaly_rate': [],
            'timestamps': []
        }
        
        for hour in sorted(hourly_data.keys()):
            records = hourly_data[hour]
            
            trends['timestamps'].append(hour.isoformat())
            trends['hourly_throughput'].append(len(records))
            
            avg_latency = np.mean([r['processing_time'] for r in records])
            trends['hourly_avg_latency'].append(round(avg_latency, 2))
            
            total_anomalies = sum(r['anomalies_found'] for r in records)
            anomaly_rate = (total_anomalies / len(records)) * 100 if records else 0
            trends['hourly_anomaly_rate'].append(round(anomaly_rate, 2))
        
        return trends

    # ==============================================================================
    # HEALTH MONITORING METHODS
    # ==============================================================================
    
    async def start_health_monitoring(self) -> None:
        """Start continuous health monitoring."""
        if self._monitoring_active:
            logger.warning("Health monitoring already active")
            return
        
        self._monitoring_active = True
        self._monitoring_task = asyncio.create_task(self._health_monitoring_loop())
        
        logger.info("Health monitoring started", interval=self.health_check_interval)

    async def stop_health_monitoring(self) -> None:
        """Stop health monitoring."""
        self._monitoring_active = False
        
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Health monitoring stopped")

    async def _health_monitoring_loop(self) -> None:
        """Main health monitoring loop."""
        while self._monitoring_active:
            try:
                await self._perform_health_check()
                await asyncio.sleep(self.health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Health monitoring error", error=str(e))
                await asyncio.sleep(self.health_check_interval)

    async def _perform_health_check(self) -> None:
        """Perform comprehensive health check."""
        timestamp = datetime.utcnow()
        
        # System resource metrics
        await self._check_system_resources(timestamp)
        
        # Application performance metrics
        await self._check_application_performance(timestamp)
        
        # Service availability metrics
        await self._check_service_availability(timestamp)
        
        # Store health snapshot
        health_snapshot = {
            'timestamp': timestamp,
            'metrics': {name: metric.to_dict() for name, metric in self._health_metrics.items()},
            'overall_status': self._calculate_overall_health_status()
        }
        self._health_history.append(health_snapshot)
        
        # Cleanup old alerts
        self._cleanup_old_alerts()

    async def _check_system_resources(self, timestamp: datetime) -> None:
        """Check system resource utilization."""
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_status = self._determine_status(cpu_percent, 'cpu_usage')
        self._health_metrics['cpu_usage'] = HealthMetric(
            name='cpu_usage',
            value=cpu_percent,
            unit='%',
            status=cpu_status,
            threshold_warning=self._health_thresholds['cpu_usage']['warning'],
            threshold_critical=self._health_thresholds['cpu_usage']['critical'],
            timestamp=timestamp,
            description='CPU utilization percentage'
        )
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_status = self._determine_status(memory.percent, 'memory_usage')
        self._health_metrics['memory_usage'] = HealthMetric(
            name='memory_usage',
            value=memory.percent,
            unit='%',
            status=memory_status,
            threshold_warning=self._health_thresholds['memory_usage']['warning'],
            threshold_critical=self._health_thresholds['memory_usage']['critical'],
            timestamp=timestamp,
            description='Memory utilization percentage'
        )
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        disk_status = self._determine_status(disk_percent, 'disk_usage')
        self._health_metrics['disk_usage'] = HealthMetric(
            name='disk_usage',
            value=disk_percent,
            unit='%',
            status=disk_status,
            threshold_warning=self._health_thresholds['disk_usage']['warning'],
            threshold_critical=self._health_thresholds['disk_usage']['critical'],
            timestamp=timestamp,
            description='Disk space utilization percentage'
        )

    async def _check_application_performance(self, timestamp: datetime) -> None:
        """Check application performance metrics."""
        
        # Detection latency
        avg_latency_ms = self.performance_metrics.average_detection_time * 1000
        latency_status = self._determine_status(avg_latency_ms, 'detection_latency')
        self._health_metrics['detection_latency'] = HealthMetric(
            name='detection_latency',
            value=avg_latency_ms,
            unit='ms',
            status=latency_status,
            threshold_warning=self._health_thresholds['detection_latency']['warning'],
            threshold_critical=self._health_thresholds['detection_latency']['critical'],
            timestamp=timestamp,
            description='Average detection processing time'
        )
        
        # Error rate
        error_rate = (1 - self.performance_metrics.success_rate) * 100
        error_status = self._determine_status(error_rate, 'error_rate')
        self._health_metrics['error_rate'] = HealthMetric(
            name='error_rate',
            value=error_rate,
            unit='%',
            status=error_status,
            threshold_warning=self._health_thresholds['error_rate']['warning'],
            threshold_critical=self._health_thresholds['error_rate']['critical'],
            timestamp=timestamp,
            description='Detection error rate percentage'
        )
        
        # Throughput (recent detections per second)
        recent_detections = [
            r for r in self._detection_history
            if (timestamp - r['timestamp']).total_seconds() <= 60
        ]
        throughput = len(recent_detections) / 60.0
        # For throughput, lower values are worse, so we invert the logic
        throughput_status = self._determine_throughput_status(throughput)
        self._health_metrics['throughput'] = HealthMetric(
            name='throughput',
            value=throughput,
            unit='req/s',
            status=throughput_status,
            threshold_warning=self._health_thresholds['throughput']['warning'],
            threshold_critical=self._health_thresholds['throughput']['critical'],
            timestamp=timestamp,
            description='Detection requests per second'
        )

    async def _check_service_availability(self, timestamp: datetime) -> None:
        """Check service availability and dependencies."""
        
        # Service uptime
        uptime_hours = (timestamp - datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds() / 3600
        uptime_status = HealthStatus.HEALTHY  # Simplified for now
        
        self._health_metrics['uptime'] = HealthMetric(
            name='uptime',
            value=uptime_hours,
            unit='hours',
            status=uptime_status,
            threshold_warning=24.0,
            threshold_critical=1.0,
            timestamp=timestamp,
            description='Service uptime in hours'
        )

    def get_health_summary(self) -> Dict[str, Any]:
        """Get current health summary."""
        
        if not self._health_metrics:
            return {
                'overall_status': HealthStatus.UNKNOWN.value,
                'metrics': {},
                'last_check': None
            }
        
        return {
            'overall_status': self._calculate_overall_health_status().value,
            'metrics': {name: metric.to_dict() for name, metric in self._health_metrics.items()},
            'last_check': max(metric.timestamp for metric in self._health_metrics.values()).isoformat(),
            'active_alerts': len(self.get_active_alerts())
        }

    def get_health_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get health history for specified time period."""
        
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        return [
            snapshot for snapshot in self._health_history
            if snapshot['timestamp'] >= cutoff_time
        ]

    # ==============================================================================
    # ALERT MANAGEMENT METHODS
    # ==============================================================================
    
    def create_alert(
        self,
        severity: AlertSeverity,
        title: str,
        message: str,
        component: str
    ) -> SystemAlert:
        """Create a new system alert."""
        
        self._alert_counter += 1
        alert = SystemAlert(
            id=f"alert_{self._alert_counter}_{int(time.time())}",
            severity=severity,
            title=title,
            message=message,
            component=component,
            timestamp=datetime.utcnow()
        )
        
        self._system_alerts.append(alert)
        
        logger.warning("Alert created",
                      alert_id=alert.id,
                      severity=severity.value,
                      title=title,
                      component=component)
        
        return alert

    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert by ID."""
        
        for alert in self._system_alerts:
            if alert.id == alert_id and not alert.resolved:
                alert.resolved = True
                alert.resolution_time = datetime.utcnow()
                
                logger.info("Alert resolved", alert_id=alert_id)
                return True
        
        return False

    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get all active (unresolved) alerts."""
        
        return [
            alert.to_dict() for alert in self._system_alerts
            if not alert.resolved
        ]

    def get_all_alerts(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get all alerts within specified time period."""
        
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        return [
            alert.to_dict() for alert in self._system_alerts
            if alert.timestamp >= cutoff_time
        ]

    # ==============================================================================
    # UTILITY METHODS
    # ==============================================================================
    
    def _calculate_time_based_metrics(
        self,
        detections: List[Dict[str, Any]],
        time_range: timedelta
    ) -> Dict[str, Any]:
        """Calculate metrics for a specific time period."""
        
        if not detections:
            return {
                'detections_in_period': 0,
                'anomalies_in_period': 0,
                'average_processing_time': 0.0,
                'throughput': 0.0,
                'success_rate': 0.0
            }
        
        total_detections = len(detections)
        total_anomalies = sum(d['anomalies_found'] for d in detections)
        successful_detections = sum(1 for d in detections if d['success'])
        
        avg_processing_time = np.mean([d['processing_time'] for d in detections])
        throughput = total_detections / time_range.total_seconds()
        success_rate = (successful_detections / total_detections) * 100
        
        return {
            'detections_in_period': total_detections,
            'anomalies_in_period': total_anomalies,
            'average_processing_time': round(avg_processing_time, 3),
            'throughput': round(throughput, 3),
            'success_rate': round(success_rate, 2)
        }

    def _determine_status(self, value: float, metric_name: str) -> HealthStatus:
        """Determine health status based on value and thresholds."""
        
        thresholds = self._health_thresholds.get(metric_name, {})
        critical_threshold = thresholds.get('critical', 100.0)
        warning_threshold = thresholds.get('warning', 80.0)
        
        if value >= critical_threshold:
            self._check_and_create_alert(metric_name, value, AlertSeverity.CRITICAL)
            return HealthStatus.CRITICAL
        elif value >= warning_threshold:
            self._check_and_create_alert(metric_name, value, AlertSeverity.WARNING)
            return HealthStatus.WARNING
        else:
            return HealthStatus.HEALTHY

    def _determine_throughput_status(self, throughput: float) -> HealthStatus:
        """Determine status for throughput (lower is worse)."""
        
        thresholds = self._health_thresholds['throughput']
        critical_threshold = thresholds['critical']
        warning_threshold = thresholds['warning']
        
        if throughput <= critical_threshold:
            self._check_and_create_alert('throughput', throughput, AlertSeverity.CRITICAL)
            return HealthStatus.CRITICAL
        elif throughput <= warning_threshold:
            self._check_and_create_alert('throughput', throughput, AlertSeverity.WARNING)
            return HealthStatus.WARNING
        else:
            return HealthStatus.HEALTHY

    def _check_and_create_alert(self, metric_name: str, value: float, severity: AlertSeverity) -> None:
        """Check if alert should be created for metric threshold breach."""
        
        # Check if similar alert already exists and is active
        existing_alert = None
        for alert in self._system_alerts:
            if (alert.component == metric_name and 
                not alert.resolved and 
                alert.severity == severity):
                existing_alert = alert
                break
        
        if existing_alert:
            return  # Don't create duplicate alerts
        
        # Create new alert
        title = f"{metric_name.replace('_', ' ').title()} {severity.value.title()}"
        message = f"{metric_name} is {value} which exceeds {severity.value} threshold"
        
        self.create_alert(severity, title, message, metric_name)

    def _calculate_overall_health_status(self) -> HealthStatus:
        """Calculate overall system health status."""
        
        if not self._health_metrics:
            return HealthStatus.UNKNOWN
        
        statuses = [metric.status for metric in self._health_metrics.values()]
        
        if HealthStatus.CRITICAL in statuses:
            return HealthStatus.CRITICAL
        elif HealthStatus.WARNING in statuses:
            return HealthStatus.WARNING
        else:
            return HealthStatus.HEALTHY

    def _cleanup_old_alerts(self) -> None:
        """Clean up old resolved alerts."""
        
        cutoff_time = datetime.utcnow() - timedelta(days=self.history_retention_days)
        
        self._system_alerts = [
            alert for alert in self._system_alerts
            if not alert.resolved or alert.timestamp >= cutoff_time
        ]

    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information."""
        
        return {
            'service': {
                'name': 'Enhanced Analytics Service',
                'version': '1.0.0',
                'uptime': (datetime.utcnow() - datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()
            },
            'monitoring': {
                'active': self._monitoring_active,
                'interval_seconds': self.health_check_interval,
                'retention_days': self.history_retention_days
            },
            'statistics': {
                'total_detections_recorded': len(self._detection_history),
                'health_checks_performed': len(self._health_history),
                'alerts_created': self._alert_counter,
                'active_alerts': len(self.get_active_alerts())
            },
            'thresholds': self._health_thresholds
        }

    async def __aenter__(self):
        """Async context manager entry."""
        await self.start_health_monitoring()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop_health_monitoring()