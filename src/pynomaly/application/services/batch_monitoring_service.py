"""Batch Processing Monitoring Service.

This module provides comprehensive monitoring, progress tracking, and
real-time metrics for batch processing operations.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum

import psutil
from pydantic import BaseModel

from ...infrastructure.config.settings import Settings

logger = logging.getLogger(__name__)


class AlertLevel(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(str, Enum):
    """Types of metrics to track."""
    GAUGE = "gauge"        # Current value
    COUNTER = "counter"    # Cumulative count
    HISTOGRAM = "histogram" # Distribution of values
    RATE = "rate"          # Change over time


@dataclass
class SystemMetrics:
    """System resource metrics."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_available_gb: float
    disk_usage_percent: float
    network_bytes_sent: int
    network_bytes_recv: int
    process_count: int
    
    @classmethod
    def capture_current(cls) -> SystemMetrics:
        """Capture current system metrics."""
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        network = psutil.net_io_counters()
        
        return cls(
            timestamp=datetime.now(timezone.utc),
            cpu_percent=psutil.cpu_percent(interval=0.1),
            memory_percent=memory.percent,
            memory_available_gb=memory.available / (1024**3),
            disk_usage_percent=disk.percent,
            network_bytes_sent=network.bytes_sent,
            network_bytes_recv=network.bytes_recv,
            process_count=len(psutil.pids())
        )


@dataclass
class BatchJobMetrics:
    """Metrics for a specific batch job."""
    job_id: str
    timestamp: datetime
    status: str
    progress_percentage: float
    items_processed: int
    items_total: int
    batches_completed: int
    batches_total: int
    processing_rate_items_per_sec: float
    error_count: int
    memory_usage_mb: float
    estimated_completion: Optional[datetime]
    last_checkpoint: Optional[datetime]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        # Convert datetime objects to ISO strings
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat() if value else None
        return data


class BatchAlert(BaseModel):
    """Alert for batch processing issues."""
    id: str
    job_id: Optional[str] = None
    level: AlertLevel
    title: str
    message: str
    timestamp: datetime
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = {}


class ProgressEvent(BaseModel):
    """Event for progress updates."""
    job_id: str
    timestamp: datetime
    event_type: str  # started, progress, completed, failed, cancelled
    progress_percentage: float
    message: str
    metadata: Dict[str, Any] = {}


class MetricSeries:
    """Time series of metric values."""
    
    def __init__(self, name: str, metric_type: MetricType, max_points: int = 1000):
        self.name = name
        self.metric_type = metric_type
        self.max_points = max_points
        self.points: List[tuple[datetime, float]] = []
    
    def add_point(self, timestamp: datetime, value: float) -> None:
        """Add a metric point."""
        self.points.append((timestamp, value))
        
        # Keep only the most recent points
        if len(self.points) > self.max_points:
            self.points = self.points[-self.max_points:]
    
    def get_recent_points(self, minutes: int = 60) -> List[tuple[datetime, float]]:
        """Get points from the last N minutes."""
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=minutes)
        return [(ts, val) for ts, val in self.points if ts >= cutoff]
    
    def get_current_value(self) -> Optional[float]:
        """Get the most recent value."""
        return self.points[-1][1] if self.points else None
    
    def get_average(self, minutes: int = 60) -> Optional[float]:
        """Get average value over the last N minutes."""
        recent_points = self.get_recent_points(minutes)
        if not recent_points:
            return None
        return sum(val for _, val in recent_points) / len(recent_points)


class BatchMonitoringService:
    """Service for monitoring batch processing operations."""
    
    def __init__(self, settings: Optional[Settings] = None):
        """Initialize the monitoring service.
        
        Args:
            settings: Application settings
        """
        self.settings = settings or Settings()
        self.logger = logging.getLogger(__name__)
        
        # Metrics storage
        self.job_metrics: Dict[str, List[BatchJobMetrics]] = {}
        self.system_metrics: List[SystemMetrics] = []
        self.metric_series: Dict[str, MetricSeries] = {}
        
        # Alerts and events
        self.alerts: List[BatchAlert] = []
        self.events: List[ProgressEvent] = []
        
        # Progress callbacks
        self.progress_callbacks: List[Callable[[ProgressEvent], None]] = []
        self.alert_callbacks: List[Callable[[BatchAlert], None]] = []
        
        # Monitoring configuration
        self.monitoring_interval = 30  # seconds
        self.metrics_retention_hours = 24
        self.events_retention_hours = 48
        self.alerts_retention_days = 7
        
        # Monitoring task
        self._monitoring_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Initialize standard metrics
        self._initialize_metrics()
    
    def _initialize_metrics(self) -> None:
        """Initialize standard metric series."""
        system_metrics = [
            ("system.cpu_percent", MetricType.GAUGE),
            ("system.memory_percent", MetricType.GAUGE),
            ("system.memory_available_gb", MetricType.GAUGE),
            ("system.disk_usage_percent", MetricType.GAUGE),
            ("system.process_count", MetricType.GAUGE),
            ("jobs.running_count", MetricType.GAUGE),
            ("jobs.scheduled_count", MetricType.GAUGE),
            ("jobs.completed_count", MetricType.COUNTER),
            ("jobs.failed_count", MetricType.COUNTER),
            ("processing.items_per_second", MetricType.GAUGE),
            ("processing.error_rate", MetricType.GAUGE),
        ]
        
        for name, metric_type in system_metrics:
            self.metric_series[name] = MetricSeries(name, metric_type)
    
    async def start_monitoring(self) -> None:
        """Start the monitoring service."""
        if self._running:
            return
        
        self._running = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.logger.info("Batch monitoring service started")
    
    async def stop_monitoring(self) -> None:
        """Stop the monitoring service."""
        self._running = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Batch monitoring service stopped")
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self._running:
            try:
                await self._collect_system_metrics()
                await self._check_alerts()
                await self._cleanup_old_data()
                await asyncio.sleep(self.monitoring_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.monitoring_interval)
    
    async def _collect_system_metrics(self) -> None:
        """Collect current system metrics."""
        try:
            metrics = SystemMetrics.capture_current()
            self.system_metrics.append(metrics)
            
            # Update metric series
            timestamp = metrics.timestamp
            self.metric_series["system.cpu_percent"].add_point(timestamp, metrics.cpu_percent)
            self.metric_series["system.memory_percent"].add_point(timestamp, metrics.memory_percent)
            self.metric_series["system.memory_available_gb"].add_point(timestamp, metrics.memory_available_gb)
            self.metric_series["system.disk_usage_percent"].add_point(timestamp, metrics.disk_usage_percent)
            self.metric_series["system.process_count"].add_point(timestamp, float(metrics.process_count))
            
            # Keep only recent system metrics
            cutoff = datetime.now(timezone.utc) - timedelta(hours=self.metrics_retention_hours)
            self.system_metrics = [m for m in self.system_metrics if m.timestamp >= cutoff]
            
        except Exception as e:
            self.logger.error(f"Failed to collect system metrics: {e}")
    
    async def _check_alerts(self) -> None:
        """Check for alert conditions."""
        timestamp = datetime.now(timezone.utc)
        
        # System resource alerts
        if self.system_metrics:
            latest = self.system_metrics[-1]
            
            # High CPU usage
            if latest.cpu_percent > 90:
                await self._create_alert(
                    level=AlertLevel.CRITICAL,
                    title="High CPU Usage",
                    message=f"CPU usage is {latest.cpu_percent:.1f}% - batch processing may be slow",
                    metadata={"cpu_percent": latest.cpu_percent}
                )
            elif latest.cpu_percent > 80:
                await self._create_alert(
                    level=AlertLevel.WARNING,
                    title="Elevated CPU Usage",
                    message=f"CPU usage is {latest.cpu_percent:.1f}%",
                    metadata={"cpu_percent": latest.cpu_percent}
                )
            
            # High memory usage
            if latest.memory_percent > 95:
                await self._create_alert(
                    level=AlertLevel.CRITICAL,
                    title="Critical Memory Usage",
                    message=f"Memory usage is {latest.memory_percent:.1f}% - risk of OOM errors",
                    metadata={"memory_percent": latest.memory_percent}
                )
            elif latest.memory_percent > 85:
                await self._create_alert(
                    level=AlertLevel.WARNING,
                    title="High Memory Usage",
                    message=f"Memory usage is {latest.memory_percent:.1f}%",
                    metadata={"memory_percent": latest.memory_percent}
                )
            
            # Low available memory
            if latest.memory_available_gb < 0.5:
                await self._create_alert(
                    level=AlertLevel.ERROR,
                    title="Low Available Memory",
                    message=f"Only {latest.memory_available_gb:.2f}GB memory available",
                    metadata={"memory_available_gb": latest.memory_available_gb}
                )
            
            # High disk usage
            if latest.disk_usage_percent > 95:
                await self._create_alert(
                    level=AlertLevel.CRITICAL,
                    title="Disk Space Critical",
                    message=f"Disk usage is {latest.disk_usage_percent:.1f}%",
                    metadata={"disk_usage_percent": latest.disk_usage_percent}
                )
    
    async def _create_alert(self,
                           level: AlertLevel,
                           title: str,
                           message: str,
                           job_id: Optional[str] = None,
                           metadata: Optional[Dict[str, Any]] = None) -> BatchAlert:
        """Create a new alert."""
        import uuid
        
        alert = BatchAlert(
            id=str(uuid.uuid4()),
            job_id=job_id,
            level=level,
            title=title,
            message=message,
            timestamp=datetime.now(timezone.utc),
            metadata=metadata or {}
        )
        
        # Check for duplicate alerts (same title and job_id in last 5 minutes)
        recent_cutoff = datetime.now(timezone.utc) - timedelta(minutes=5)
        duplicate = any(
            a.title == title and a.job_id == job_id and a.timestamp >= recent_cutoff
            for a in self.alerts if not a.resolved
        )
        
        if not duplicate:
            self.alerts.append(alert)
            self.logger.warning(f"Alert created: {title} - {message}")
            
            # Notify callbacks
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    self.logger.error(f"Alert callback failed: {e}")
        
        return alert
    
    async def _cleanup_old_data(self) -> None:
        """Clean up old metrics and events."""
        now = datetime.now(timezone.utc)
        
        # Clean up old events
        events_cutoff = now - timedelta(hours=self.events_retention_hours)
        self.events = [e for e in self.events if e.timestamp >= events_cutoff]
        
        # Clean up old alerts
        alerts_cutoff = now - timedelta(days=self.alerts_retention_days)
        self.alerts = [a for a in self.alerts if a.timestamp >= alerts_cutoff]
        
        # Clean up old job metrics
        metrics_cutoff = now - timedelta(hours=self.metrics_retention_hours)
        for job_id in list(self.job_metrics.keys()):
            self.job_metrics[job_id] = [
                m for m in self.job_metrics[job_id] 
                if m.timestamp >= metrics_cutoff
            ]
            # Remove empty job metric lists
            if not self.job_metrics[job_id]:
                del self.job_metrics[job_id]
    
    def record_job_metrics(self, job_metrics: BatchJobMetrics) -> None:
        """Record metrics for a batch job."""
        if job_metrics.job_id not in self.job_metrics:
            self.job_metrics[job_metrics.job_id] = []
        
        self.job_metrics[job_metrics.job_id].append(job_metrics)
        
        # Update global metrics
        timestamp = job_metrics.timestamp
        
        # Processing rate
        if job_metrics.processing_rate_items_per_sec > 0:
            self.metric_series["processing.items_per_second"].add_point(
                timestamp, job_metrics.processing_rate_items_per_sec
            )
        
        # Error rate
        error_rate = (
            job_metrics.error_count / max(job_metrics.items_processed, 1)
            if job_metrics.items_processed > 0 else 0
        )
        self.metric_series["processing.error_rate"].add_point(timestamp, error_rate)
    
    def record_progress_event(self, event: ProgressEvent) -> None:
        """Record a progress event."""
        self.events.append(event)
        
        # Notify callbacks
        for callback in self.progress_callbacks:
            try:
                callback(event)
            except Exception as e:
                self.logger.error(f"Progress callback failed: {e}")
        
        # Check for job-specific alerts
        if event.event_type == "failed":
            asyncio.create_task(self._create_alert(
                level=AlertLevel.ERROR,
                title=f"Job Failed",
                message=f"Batch job {event.job_id} failed: {event.message}",
                job_id=event.job_id,
                metadata=event.metadata
            ))
        elif event.event_type == "stalled":
            asyncio.create_task(self._create_alert(
                level=AlertLevel.WARNING,
                title=f"Job Stalled",
                message=f"Batch job {event.job_id} appears to be stalled",
                job_id=event.job_id,
                metadata=event.metadata
            ))
    
    def add_progress_callback(self, callback: Callable[[ProgressEvent], None]) -> None:
        """Add a progress event callback."""
        self.progress_callbacks.append(callback)
    
    def add_alert_callback(self, callback: Callable[[BatchAlert], None]) -> None:
        """Add an alert callback."""
        self.alert_callbacks.append(callback)
    
    def get_job_metrics_history(self, 
                               job_id: str,
                               minutes: int = 60) -> List[BatchJobMetrics]:
        """Get metrics history for a job.
        
        Args:
            job_id: Job ID
            minutes: Number of minutes of history to return
            
        Returns:
            List of job metrics
        """
        if job_id not in self.job_metrics:
            return []
        
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=minutes)
        return [m for m in self.job_metrics[job_id] if m.timestamp >= cutoff]
    
    def get_system_metrics_history(self, minutes: int = 60) -> List[SystemMetrics]:
        """Get system metrics history.
        
        Args:
            minutes: Number of minutes of history to return
            
        Returns:
            List of system metrics
        """
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=minutes)
        return [m for m in self.system_metrics if m.timestamp >= cutoff]
    
    def get_active_alerts(self, 
                         level_filter: Optional[AlertLevel] = None,
                         job_id_filter: Optional[str] = None) -> List[BatchAlert]:
        """Get active (unresolved) alerts.
        
        Args:
            level_filter: Filter by alert level
            job_id_filter: Filter by job ID
            
        Returns:
            List of active alerts
        """
        alerts = [a for a in self.alerts if not a.resolved]
        
        if level_filter:
            alerts = [a for a in alerts if a.level == level_filter]
        
        if job_id_filter:
            alerts = [a for a in alerts if a.job_id == job_id_filter]
        
        return sorted(alerts, key=lambda a: a.timestamp, reverse=True)
    
    def get_recent_events(self, 
                         job_id: Optional[str] = None,
                         event_type: Optional[str] = None,
                         minutes: int = 60) -> List[ProgressEvent]:
        """Get recent progress events.
        
        Args:
            job_id: Filter by job ID
            event_type: Filter by event type
            minutes: Number of minutes of history
            
        Returns:
            List of progress events
        """
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=minutes)
        events = [e for e in self.events if e.timestamp >= cutoff]
        
        if job_id:
            events = [e for e in events if e.job_id == job_id]
        
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        return sorted(events, key=lambda e: e.timestamp, reverse=True)
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert.
        
        Args:
            alert_id: Alert ID to resolve
            
        Returns:
            True if alert was found and resolved
        """
        for alert in self.alerts:
            if alert.id == alert_id and not alert.resolved:
                alert.resolved = True
                alert.resolved_at = datetime.now(timezone.utc)
                self.logger.info(f"Resolved alert: {alert.title}")
                return True
        return False
    
    def get_monitoring_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data.
        
        Returns:
            Dashboard data including metrics, alerts, and events
        """
        now = datetime.now(timezone.utc)
        
        # System metrics summary
        system_summary = {}
        if self.system_metrics:
            latest = self.system_metrics[-1]
            system_summary = {
                "cpu_percent": latest.cpu_percent,
                "memory_percent": latest.memory_percent,
                "memory_available_gb": latest.memory_available_gb,
                "disk_usage_percent": latest.disk_usage_percent,
                "last_updated": latest.timestamp.isoformat()
            }
        
        # Active alerts summary
        active_alerts = self.get_active_alerts()
        alerts_summary = {
            "total": len(active_alerts),
            "critical": len([a for a in active_alerts if a.level == AlertLevel.CRITICAL]),
            "errors": len([a for a in active_alerts if a.level == AlertLevel.ERROR]),
            "warnings": len([a for a in active_alerts if a.level == AlertLevel.WARNING])
        }
        
        # Recent events summary
        recent_events = self.get_recent_events(minutes=60)
        events_summary = {
            "total": len(recent_events),
            "completed": len([e for e in recent_events if e.event_type == "completed"]),
            "failed": len([e for e in recent_events if e.event_type == "failed"]),
            "started": len([e for e in recent_events if e.event_type == "started"])
        }
        
        # Job counts
        job_counts = {
            "total_jobs": len(self.job_metrics),
            "active_jobs": len([
                job_id for job_id, metrics in self.job_metrics.items()
                if metrics and metrics[-1].status in ["running", "pending"]
            ])
        }
        
        # Performance metrics
        performance_metrics = {}
        if "processing.items_per_second" in self.metric_series:
            avg_rate = self.metric_series["processing.items_per_second"].get_average(60)
            current_rate = self.metric_series["processing.items_per_second"].get_current_value()
            performance_metrics["average_processing_rate"] = avg_rate
            performance_metrics["current_processing_rate"] = current_rate
        
        return {
            "timestamp": now.isoformat(),
            "system_summary": system_summary,
            "alerts_summary": alerts_summary,
            "events_summary": events_summary,
            "job_counts": job_counts,
            "performance_metrics": performance_metrics,
            "monitoring_status": {
                "running": self._running,
                "monitoring_interval": self.monitoring_interval
            }
        }