"""Monitoring dashboard service for comprehensive system overview."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from .health_service import HealthService, SystemMetrics


@dataclass
class DashboardMetrics:
    """Combined metrics for dashboard display."""
    
    # System overview
    system_health: str
    uptime_seconds: float
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    
    # Application metrics
    total_detectors: int
    total_datasets: int
    total_results: int
    active_sessions: int
    
    # Performance metrics
    avg_detection_time_ms: float
    total_detections_today: int
    error_rate_24h: float
    
    # Health status by component
    component_health: dict[str, str]
    
    # Recent activity
    recent_detections: list[dict[str, Any]]
    recent_errors: list[dict[str, Any]]
    
    # Timestamp
    timestamp: datetime = field(default_factory=datetime.utcnow)


class MonitoringDashboardService:
    """Service for generating monitoring dashboard data."""

    def __init__(self, health_service: HealthService):
        """Initialize dashboard service.
        
        Args:
            health_service: Health monitoring service
        """
        self.health_service = health_service
        self._cache_ttl = 60  # Cache for 60 seconds
        self._cached_metrics: DashboardMetrics | None = None
        self._last_update: datetime | None = None

    async def get_dashboard_metrics(
        self,
        container,
        force_refresh: bool = False,
    ) -> DashboardMetrics:
        """Get comprehensive dashboard metrics.
        
        Args:
            container: Dependency injection container
            force_refresh: Force refresh of cached data
            
        Returns:
            Dashboard metrics
        """
        # Return cached data if still fresh
        if (
            not force_refresh
            and self._cached_metrics
            and self._last_update
            and (datetime.utcnow() - self._last_update).seconds < self._cache_ttl
        ):
            return self._cached_metrics

        # Collect fresh metrics
        metrics = await self._collect_metrics(container)
        
        # Cache results
        self._cached_metrics = metrics
        self._last_update = datetime.utcnow()
        
        return metrics

    async def _collect_metrics(self, container) -> DashboardMetrics:
        """Collect all dashboard metrics."""
        # Get system metrics
        system_metrics = self.health_service.get_system_metrics()
        
        # Get health checks
        health_checks = await self.health_service.perform_comprehensive_health_check()
        overall_health = self.health_service.get_overall_status(health_checks)
        
        # Get application metrics
        app_metrics = await self._get_application_metrics(container)
        
        # Get performance metrics
        perf_metrics = await self._get_performance_metrics(container)
        
        # Get component health
        component_health = {
            name: check.status.value
            for name, check in health_checks.items()
        }
        
        # Get recent activity
        recent_activity = await self._get_recent_activity(container)
        
        return DashboardMetrics(
            # System overview
            system_health=overall_health.value,
            uptime_seconds=system_metrics.uptime_seconds,
            cpu_percent=system_metrics.cpu_percent,
            memory_percent=system_metrics.memory_percent,
            disk_percent=system_metrics.disk_percent,
            
            # Application metrics
            total_detectors=app_metrics["detectors"],
            total_datasets=app_metrics["datasets"],
            total_results=app_metrics["results"],
            active_sessions=app_metrics["sessions"],
            
            # Performance metrics
            avg_detection_time_ms=perf_metrics["avg_detection_time"],
            total_detections_today=perf_metrics["detections_today"],
            error_rate_24h=perf_metrics["error_rate"],
            
            # Component health
            component_health=component_health,
            
            # Recent activity
            recent_detections=recent_activity["detections"],
            recent_errors=recent_activity["errors"],
        )

    async def _get_application_metrics(self, container) -> dict[str, Any]:
        """Get application-specific metrics."""
        try:
            detector_count = container.detector_repository().count()
            dataset_count = container.dataset_repository().count()
            result_count = container.result_repository().count()
            
            # For active sessions, we'd need session management
            # For now, use a placeholder
            active_sessions = 0
            
            return {
                "detectors": detector_count,
                "datasets": dataset_count,
                "results": result_count,
                "sessions": active_sessions,
            }
            
        except Exception:
            return {
                "detectors": 0,
                "datasets": 0,
                "results": 0,
                "sessions": 0,
            }

    async def _get_performance_metrics(self, container) -> dict[str, Any]:
        """Get performance metrics."""
        try:
            # Get recent detection results for performance analysis
            results = container.result_repository().find_recent(100)
            
            # Calculate average detection time
            if results:
                total_time = sum(r.execution_time_ms or 0 for r in results)
                avg_time = total_time / len(results)
            else:
                avg_time = 0.0
            
            # Count detections today
            today = datetime.utcnow().date()
            detections_today = sum(
                1 for r in results
                if r.timestamp.date() == today
            )
            
            # Calculate error rate (placeholder - would need error tracking)
            error_rate = 0.0
            
            return {
                "avg_detection_time": avg_time,
                "detections_today": detections_today,
                "error_rate": error_rate,
            }
            
        except Exception:
            return {
                "avg_detection_time": 0.0,
                "detections_today": 0,
                "error_rate": 0.0,
            }

    async def _get_recent_activity(self, container) -> dict[str, Any]:
        """Get recent activity data."""
        try:
            # Get recent detection results
            recent_results = container.result_repository().find_recent(10)
            
            detections = [
                {
                    "id": str(r.id),
                    "timestamp": r.timestamp.isoformat(),
                    "detector_id": str(r.detector_id),
                    "dataset_id": str(r.dataset_id),
                    "anomalies_found": r.n_anomalies,
                    "execution_time_ms": r.execution_time_ms,
                }
                for r in recent_results
            ]
            
            # Recent errors would come from error tracking
            # For now, use placeholder
            errors = []
            
            return {
                "detections": detections,
                "errors": errors,
            }
            
        except Exception:
            return {
                "detections": [],
                "errors": [],
            }

    def get_alert_thresholds(self) -> dict[str, float]:
        """Get alerting thresholds."""
        return {
            "cpu_percent": 80.0,
            "memory_percent": 85.0,
            "disk_percent": 90.0,
            "error_rate": 5.0,
            "avg_response_time_ms": 1000.0,
        }

    def check_alert_conditions(self, metrics: DashboardMetrics) -> list[dict[str, Any]]:
        """Check if any alert conditions are met."""
        alerts = []
        thresholds = self.get_alert_thresholds()
        
        # CPU alert
        if metrics.cpu_percent > thresholds["cpu_percent"]:
            alerts.append({
                "type": "resource",
                "severity": "warning",
                "metric": "cpu_percent",
                "value": metrics.cpu_percent,
                "threshold": thresholds["cpu_percent"],
                "message": f"High CPU usage: {metrics.cpu_percent:.1f}%",
            })
        
        # Memory alert
        if metrics.memory_percent > thresholds["memory_percent"]:
            alerts.append({
                "type": "resource",
                "severity": "warning",
                "metric": "memory_percent",
                "value": metrics.memory_percent,
                "threshold": thresholds["memory_percent"],
                "message": f"High memory usage: {metrics.memory_percent:.1f}%",
            })
        
        # Disk alert
        if metrics.disk_percent > thresholds["disk_percent"]:
            alerts.append({
                "type": "resource",
                "severity": "critical",
                "metric": "disk_percent",
                "value": metrics.disk_percent,
                "threshold": thresholds["disk_percent"],
                "message": f"High disk usage: {metrics.disk_percent:.1f}%",
            })
        
        # Error rate alert
        if metrics.error_rate_24h > thresholds["error_rate"]:
            alerts.append({
                "type": "application",
                "severity": "warning",
                "metric": "error_rate",
                "value": metrics.error_rate_24h,
                "threshold": thresholds["error_rate"],
                "message": f"High error rate: {metrics.error_rate_24h:.1f}%",
            })
        
        # Response time alert
        if metrics.avg_detection_time_ms > thresholds["avg_response_time_ms"]:
            alerts.append({
                "type": "performance",
                "severity": "warning",
                "metric": "avg_detection_time_ms",
                "value": metrics.avg_detection_time_ms,
                "threshold": thresholds["avg_response_time_ms"],
                "message": f"Slow detection time: {metrics.avg_detection_time_ms:.0f}ms",
            })
        
        return alerts

    async def export_metrics_for_prometheus(self, metrics: DashboardMetrics) -> str:
        """Export metrics in Prometheus format."""
        prometheus_metrics = [
            "# HELP pynomaly_system_health System health status (1=healthy, 0.5=degraded, 0=unhealthy)",
            "# TYPE pynomaly_system_health gauge",
            f"pynomaly_system_health {self._health_to_numeric(metrics.system_health)}",
            "",
            "# HELP pynomaly_uptime_seconds System uptime in seconds",
            "# TYPE pynomaly_uptime_seconds counter",
            f"pynomaly_uptime_seconds {metrics.uptime_seconds}",
            "",
            "# HELP pynomaly_cpu_percent CPU utilization percentage",
            "# TYPE pynomaly_cpu_percent gauge",
            f"pynomaly_cpu_percent {metrics.cpu_percent}",
            "",
            "# HELP pynomaly_memory_percent Memory utilization percentage",
            "# TYPE pynomaly_memory_percent gauge",
            f"pynomaly_memory_percent {metrics.memory_percent}",
            "",
            "# HELP pynomaly_disk_percent Disk utilization percentage",
            "# TYPE pynomaly_disk_percent gauge",
            f"pynomaly_disk_percent {metrics.disk_percent}",
            "",
            "# HELP pynomaly_total_detectors Total number of detectors",
            "# TYPE pynomaly_total_detectors gauge",
            f"pynomaly_total_detectors {metrics.total_detectors}",
            "",
            "# HELP pynomaly_total_datasets Total number of datasets",
            "# TYPE pynomaly_total_datasets gauge",
            f"pynomaly_total_datasets {metrics.total_datasets}",
            "",
            "# HELP pynomaly_total_results Total number of detection results",
            "# TYPE pynomaly_total_results gauge",
            f"pynomaly_total_results {metrics.total_results}",
            "",
            "# HELP pynomaly_avg_detection_time_ms Average detection time in milliseconds",
            "# TYPE pynomaly_avg_detection_time_ms gauge",
            f"pynomaly_avg_detection_time_ms {metrics.avg_detection_time_ms}",
            "",
            "# HELP pynomaly_detections_today Total detections performed today",
            "# TYPE pynomaly_detections_today counter",
            f"pynomaly_detections_today {metrics.total_detections_today}",
            "",
            "# HELP pynomaly_error_rate_24h Error rate over the last 24 hours",
            "# TYPE pynomaly_error_rate_24h gauge",
            f"pynomaly_error_rate_24h {metrics.error_rate_24h}",
        ]
        
        return "\n".join(prometheus_metrics)

    def _health_to_numeric(self, health_status: str) -> float:
        """Convert health status to numeric value for Prometheus."""
        mapping = {
            "healthy": 1.0,
            "degraded": 0.5,
            "unhealthy": 0.0,
            "unknown": -1.0,
        }
        return mapping.get(health_status, -1.0)