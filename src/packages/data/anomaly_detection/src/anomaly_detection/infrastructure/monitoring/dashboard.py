"""Monitoring dashboard data aggregation and reporting."""

from __future__ import annotations

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

from .metrics_collector import get_metrics_collector
from .health_checker import get_health_checker
from .performance_monitor import get_performance_monitor
from ..logging import get_logger

logger = get_logger(__name__)


@dataclass
class DashboardSummary:
    """Summary data for monitoring dashboard."""
    
    # System health
    overall_health_status: str
    healthy_checks: int
    degraded_checks: int
    unhealthy_checks: int
    
    # Performance
    total_operations: int
    operations_last_hour: int
    avg_response_time_ms: float
    success_rate: float
    
    # Resource usage
    current_memory_mb: Optional[float]
    current_cpu_percent: Optional[float]
    peak_memory_mb: Optional[float]
    
    # Model metrics
    total_models: int
    active_detections: int
    anomalies_detected_today: int
    
    # Alerts and issues
    active_alerts: int
    recent_errors: int
    slow_operations: int
    
    # Timestamp
    generated_at: datetime


class MonitoringDashboard:
    """Centralized dashboard data provider."""
    
    def __init__(self):
        """Initialize monitoring dashboard."""
        self.metrics_collector = get_metrics_collector()
        self.health_checker = get_health_checker()
        self.performance_monitor = get_performance_monitor()
        
        logger.info("Monitoring dashboard initialized")
    
    async def get_dashboard_summary(self) -> DashboardSummary:
        """Get comprehensive dashboard summary.
        
        Returns:
            DashboardSummary with current system status
        """
        # Run health checks first
        await self.health_checker.run_all_checks()
        
        # Get health summary
        health_summary = self.health_checker.get_health_summary()
        
        # Get performance summary
        perf_summary = self.performance_monitor.get_performance_summary()
        
        # Get metrics summary
        metrics_summary = self.metrics_collector.get_summary_stats()
        
        # Calculate time-based metrics
        now = datetime.utcnow()
        one_hour_ago = now - timedelta(hours=1)
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Get recent performance data
        recent_profiles = self.performance_monitor.get_recent_profiles(
            since=one_hour_ago
        )
        
        # Get recent model metrics
        model_metrics = self.metrics_collector.get_model_metrics(
            since=today_start
        )
        
        # Calculate derived metrics
        operations_last_hour = len(recent_profiles)
        
        # Success rate calculation
        success_rate = 0.0
        if recent_profiles:
            successful = sum(1 for p in recent_profiles if p.success)
            success_rate = successful / len(recent_profiles)
        
        # Average response time
        avg_response_time = 0.0
        if recent_profiles:
            total_time = sum(p.total_duration_ms for p in recent_profiles)
            avg_response_time = total_time / len(recent_profiles)
        
        # Count anomalies detected today
        anomalies_today = sum(
            m.anomalies_detected or 0 
            for m in model_metrics 
            if m.anomalies_detected and m.timestamp >= today_start
        )
        
        # Count slow operations (> 5 seconds)
        slow_operations = sum(
            1 for p in recent_profiles 
            if p.total_duration_ms > 5000
        )
        
        # Count recent errors
        recent_errors = sum(
            1 for p in recent_profiles 
            if not p.success
        )
        
        # Get resource usage
        resource_usage = self.performance_monitor.get_resource_usage(limit=1)
        current_memory_mb = resource_usage[0].memory_mb if resource_usage else None
        current_cpu_percent = resource_usage[0].cpu_percent if resource_usage else None
        
        # Calculate peak memory from recent profiles
        peak_memory_mb = None
        if recent_profiles:
            peak_memories = [p.peak_memory_mb for p in recent_profiles if p.peak_memory_mb]
            peak_memory_mb = max(peak_memories) if peak_memories else None
        
        # Count health check statuses
        health_counts = health_summary.get("status_counts", {})
        healthy_checks = health_counts.get("healthy", 0)
        degraded_checks = health_counts.get("degraded", 0)
        unhealthy_checks = health_counts.get("unhealthy", 0)
        
        # Active alerts (currently just unhealthy + degraded checks + errors)
        active_alerts = unhealthy_checks + (1 if recent_errors > 5 else 0) + (1 if slow_operations > 3 else 0)
        
        return DashboardSummary(
            overall_health_status=health_summary.get("overall_status", "unknown"),
            healthy_checks=healthy_checks,
            degraded_checks=degraded_checks,
            unhealthy_checks=unhealthy_checks,
            total_operations=perf_summary.get("total_operations", 0),
            operations_last_hour=operations_last_hour,
            avg_response_time_ms=avg_response_time,
            success_rate=success_rate,
            current_memory_mb=current_memory_mb,
            current_cpu_percent=current_cpu_percent,
            peak_memory_mb=peak_memory_mb,
            total_models=0,  # Would be calculated from model repository
            active_detections=operations_last_hour,  # Approximation
            anomalies_detected_today=anomalies_today,
            active_alerts=active_alerts,
            recent_errors=recent_errors,
            slow_operations=slow_operations,
            generated_at=now
        )
    
    def get_performance_trends(self, hours: int = 24) -> Dict[str, Any]:
        """Get performance trends over time.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            Dictionary with trend data
        """
        since = datetime.utcnow() - timedelta(hours=hours)
        
        # Get performance profiles
        profiles = self.performance_monitor.get_recent_profiles(since=since)
        
        # Group by hour
        hourly_data = {}
        for profile in profiles:
            hour_key = profile.timestamp.replace(minute=0, second=0, microsecond=0)
            
            if hour_key not in hourly_data:
                hourly_data[hour_key] = {
                    "timestamp": hour_key,
                    "operations": 0,
                    "successes": 0,
                    "failures": 0,
                    "total_duration": 0,
                    "peak_memory": 0
                }
            
            data = hourly_data[hour_key]
            data["operations"] += 1
            data["total_duration"] += profile.total_duration_ms
            
            if profile.success:
                data["successes"] += 1
            else:
                data["failures"] += 1
            
            if profile.peak_memory_mb:
                data["peak_memory"] = max(data["peak_memory"], profile.peak_memory_mb)
        
        # Calculate averages and rates
        trend_data = []
        for hour_key in sorted(hourly_data.keys()):
            data = hourly_data[hour_key]
            
            avg_duration = 0
            success_rate = 0
            if data["operations"] > 0:
                avg_duration = data["total_duration"] / data["operations"]
                success_rate = data["successes"] / data["operations"]
            
            trend_data.append({
                "timestamp": hour_key.isoformat(),
                "operations_count": data["operations"],
                "avg_duration_ms": avg_duration,
                "success_rate": success_rate,
                "peak_memory_mb": data["peak_memory"]
            })
        
        return {
            "period_hours": hours,
            "data_points": len(trend_data),
            "trends": trend_data
        }
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of current alerts and issues.
        
        Returns:
            Dictionary with alert information
        """
        alerts = []
        
        # Health check alerts
        health_summary = self.health_checker.get_health_summary()
        for check_name, check_info in health_summary.get("checks", {}).items():
            if check_info["status"] == "unhealthy":
                alerts.append({
                    "type": "health_check",
                    "severity": "critical" if check_info.get("critical", False) else "warning",
                    "message": f"Health check '{check_name}' is unhealthy: {check_info['message']}",
                    "timestamp": check_info["timestamp"],
                    "source": check_name
                })
            elif check_info["status"] == "degraded":
                alerts.append({
                    "type": "health_check",
                    "severity": "warning",
                    "message": f"Health check '{check_name}' is degraded: {check_info['message']}",
                    "timestamp": check_info["timestamp"],
                    "source": check_name
                })
        
        # Performance alerts
        recent_profiles = self.performance_monitor.get_recent_profiles(
            since=datetime.utcnow() - timedelta(hours=1)
        )
        
        # Slow operations alert
        slow_ops = [p for p in recent_profiles if p.total_duration_ms > 10000]
        if len(slow_ops) > 3:
            alerts.append({
                "type": "performance",
                "severity": "warning",
                "message": f"High number of slow operations: {len(slow_ops)} operations > 10s in last hour",
                "timestamp": datetime.utcnow().isoformat(),
                "source": "performance_monitor"
            })
        
        # High error rate alert
        failed_ops = [p for p in recent_profiles if not p.success]
        if recent_profiles and len(failed_ops) / len(recent_profiles) > 0.1:
            error_rate = len(failed_ops) / len(recent_profiles)
            alerts.append({
                "type": "error_rate",
                "severity": "critical" if error_rate > 0.2 else "warning",
                "message": f"High error rate: {error_rate:.1%} in last hour",
                "timestamp": datetime.utcnow().isoformat(),
                "source": "performance_monitor"
            })
        
        # Memory usage alert
        resource_usage = self.performance_monitor.get_resource_usage(limit=1)
        if resource_usage and resource_usage[0].memory_percent > 85:
            alerts.append({
                "type": "resource",
                "severity": "warning",
                "message": f"High memory usage: {resource_usage[0].memory_percent:.1f}%",
                "timestamp": resource_usage[0].timestamp.isoformat(),
                "source": "resource_monitor"
            })
        
        # Sort alerts by severity and timestamp
        severity_order = {"critical": 0, "warning": 1, "info": 2}
        alerts.sort(key=lambda x: (severity_order.get(x["severity"], 3), x["timestamp"]))
        
        return {
            "total_alerts": len(alerts),
            "critical_count": sum(1 for a in alerts if a["severity"] == "critical"),
            "warning_count": sum(1 for a in alerts if a["severity"] == "warning"),
            "alerts": alerts[:20]  # Limit to 20 most important alerts
        }
    
    def get_operation_breakdown(self) -> Dict[str, Any]:
        """Get breakdown of operations by type and performance.
        
        Returns:
            Dictionary with operation statistics
        """
        # Get operation stats from performance monitor
        operation_stats = self.performance_monitor.get_operation_stats()
        
        # Sort operations by total duration
        sorted_operations = sorted(
            operation_stats.items(),
            key=lambda x: x[1].get("total_duration_ms", 0),
            reverse=True
        )
        
        operation_breakdown = []
        for op_name, stats in sorted_operations[:10]:  # Top 10 operations
            operation_breakdown.append({
                "operation": op_name,
                "total_calls": stats.get("count", 0),
                "success_count": stats.get("success_count", 0),
                "error_count": stats.get("error_count", 0),
                "success_rate": stats.get("success_rate", 0.0),
                "avg_duration_ms": stats.get("avg_duration_ms", 0.0),
                "min_duration_ms": stats.get("min_duration_ms", 0.0),
                "max_duration_ms": stats.get("max_duration_ms", 0.0),
                "total_duration_ms": stats.get("total_duration_ms", 0.0)
            })
        
        return {
            "total_operations_monitored": len(operation_stats),
            "top_operations": operation_breakdown,
            "generated_at": datetime.utcnow().isoformat()
        }


# Global dashboard instance
_global_dashboard: Optional[MonitoringDashboard] = None


def get_monitoring_dashboard() -> MonitoringDashboard:
    """Get the global monitoring dashboard instance."""
    global _global_dashboard
    
    if _global_dashboard is None:
        _global_dashboard = MonitoringDashboard()
    
    return _global_dashboard