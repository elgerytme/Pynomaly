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
        anomalies_today = sum(\n            m.anomalies_detected or 0 \n            for m in model_metrics \n            if m.anomalies_detected and m.timestamp >= today_start\n        )\n        \n        # Count slow operations (> 5 seconds)\n        slow_operations = sum(\n            1 for p in recent_profiles \n            if p.total_duration_ms > 5000\n        )\n        \n        # Count recent errors\n        recent_errors = sum(\n            1 for p in recent_profiles \n            if not p.success\n        )\n        \n        # Get resource usage\n        resource_usage = self.performance_monitor.get_resource_usage(limit=1)\n        current_memory_mb = resource_usage[0].memory_mb if resource_usage else None\n        current_cpu_percent = resource_usage[0].cpu_percent if resource_usage else None\n        \n        # Calculate peak memory from recent profiles\n        peak_memory_mb = None\n        if recent_profiles:\n            peak_memories = [p.peak_memory_mb for p in recent_profiles if p.peak_memory_mb]\n            peak_memory_mb = max(peak_memories) if peak_memories else None\n        \n        # Count health check statuses\n        health_counts = health_summary.get(\"status_counts\", {})\n        healthy_checks = health_counts.get(\"healthy\", 0)\n        degraded_checks = health_counts.get(\"degraded\", 0)\n        unhealthy_checks = health_counts.get(\"unhealthy\", 0)\n        \n        # Active alerts (currently just unhealthy + degraded checks + errors)\n        active_alerts = unhealthy_checks + (1 if recent_errors > 5 else 0) + (1 if slow_operations > 3 else 0)\n        \n        return DashboardSummary(\n            overall_health_status=health_summary.get(\"overall_status\", \"unknown\"),\n            healthy_checks=healthy_checks,\n            degraded_checks=degraded_checks,\n            unhealthy_checks=unhealthy_checks,\n            total_operations=perf_summary.get(\"total_operations\", 0),\n            operations_last_hour=operations_last_hour,\n            avg_response_time_ms=avg_response_time,\n            success_rate=success_rate,\n            current_memory_mb=current_memory_mb,\n            current_cpu_percent=current_cpu_percent,\n            peak_memory_mb=peak_memory_mb,\n            total_models=0,  # Would be calculated from model repository\n            active_detections=operations_last_hour,  # Approximation\n            anomalies_detected_today=anomalies_today,\n            active_alerts=active_alerts,\n            recent_errors=recent_errors,\n            slow_operations=slow_operations,\n            generated_at=now\n        )\n    \n    def get_performance_trends(self, hours: int = 24) -> Dict[str, Any]:\n        \"\"\"Get performance trends over time.\n        \n        Args:\n            hours: Number of hours to look back\n            \n        Returns:\n            Dictionary with trend data\n        \"\"\"\n        since = datetime.utcnow() - timedelta(hours=hours)\n        \n        # Get performance profiles\n        profiles = self.performance_monitor.get_recent_profiles(since=since)\n        \n        # Group by hour\n        hourly_data = {}\n        for profile in profiles:\n            hour_key = profile.timestamp.replace(minute=0, second=0, microsecond=0)\n            \n            if hour_key not in hourly_data:\n                hourly_data[hour_key] = {\n                    \"timestamp\": hour_key,\n                    \"operations\": 0,\n                    \"successes\": 0,\n                    \"failures\": 0,\n                    \"total_duration\": 0,\n                    \"peak_memory\": 0\n                }\n            \n            data = hourly_data[hour_key]\n            data[\"operations\"] += 1\n            data[\"total_duration\"] += profile.total_duration_ms\n            \n            if profile.success:\n                data[\"successes\"] += 1\n            else:\n                data[\"failures\"] += 1\n            \n            if profile.peak_memory_mb:\n                data[\"peak_memory\"] = max(data[\"peak_memory\"], profile.peak_memory_mb)\n        \n        # Calculate averages and rates\n        trend_data = []\n        for hour_key in sorted(hourly_data.keys()):\n            data = hourly_data[hour_key]\n            \n            avg_duration = 0\n            success_rate = 0\n            if data[\"operations\"] > 0:\n                avg_duration = data[\"total_duration\"] / data[\"operations\"]\n                success_rate = data[\"successes\"] / data[\"operations\"]\n            \n            trend_data.append({\n                \"timestamp\": hour_key.isoformat(),\n                \"operations_count\": data[\"operations\"],\n                \"avg_duration_ms\": avg_duration,\n                \"success_rate\": success_rate,\n                \"peak_memory_mb\": data[\"peak_memory\"]\n            })\n        \n        return {\n            \"period_hours\": hours,\n            \"data_points\": len(trend_data),\n            \"trends\": trend_data\n        }\n    \n    def get_alert_summary(self) -> Dict[str, Any]:\n        \"\"\"Get summary of current alerts and issues.\n        \n        Returns:\n            Dictionary with alert information\n        \"\"\"\n        alerts = []\n        \n        # Health check alerts\n        health_summary = self.health_checker.get_health_summary()\n        for check_name, check_info in health_summary.get(\"checks\", {}).items():\n            if check_info[\"status\"] == \"unhealthy\":\n                alerts.append({\n                    \"type\": \"health_check\",\n                    \"severity\": \"critical\" if check_info.get(\"critical\", False) else \"warning\",\n                    \"message\": f\"Health check '{check_name}' is unhealthy: {check_info['message']}\",\n                    \"timestamp\": check_info[\"timestamp\"],\n                    \"source\": check_name\n                })\n            elif check_info[\"status\"] == \"degraded\":\n                alerts.append({\n                    \"type\": \"health_check\",\n                    \"severity\": \"warning\",\n                    \"message\": f\"Health check '{check_name}' is degraded: {check_info['message']}\",\n                    \"timestamp\": check_info[\"timestamp\"],\n                    \"source\": check_name\n                })\n        \n        # Performance alerts\n        recent_profiles = self.performance_monitor.get_recent_profiles(\n            since=datetime.utcnow() - timedelta(hours=1)\n        )\n        \n        # Slow operations alert\n        slow_ops = [p for p in recent_profiles if p.total_duration_ms > 10000]\n        if len(slow_ops) > 3:\n            alerts.append({\n                \"type\": \"performance\",\n                \"severity\": \"warning\",\n                \"message\": f\"High number of slow operations: {len(slow_ops)} operations > 10s in last hour\",\n                \"timestamp\": datetime.utcnow().isoformat(),\n                \"source\": \"performance_monitor\"\n            })\n        \n        # High error rate alert\n        failed_ops = [p for p in recent_profiles if not p.success]\n        if recent_profiles and len(failed_ops) / len(recent_profiles) > 0.1:\n            error_rate = len(failed_ops) / len(recent_profiles)\n            alerts.append({\n                \"type\": \"error_rate\",\n                \"severity\": \"critical\" if error_rate > 0.2 else \"warning\",\n                \"message\": f\"High error rate: {error_rate:.1%} in last hour\",\n                \"timestamp\": datetime.utcnow().isoformat(),\n                \"source\": \"performance_monitor\"\n            })\n        \n        # Memory usage alert\n        resource_usage = self.performance_monitor.get_resource_usage(limit=1)\n        if resource_usage and resource_usage[0].memory_percent > 85:\n            alerts.append({\n                \"type\": \"resource\",\n                \"severity\": \"warning\",\n                \"message\": f\"High memory usage: {resource_usage[0].memory_percent:.1f}%\",\n                \"timestamp\": resource_usage[0].timestamp.isoformat(),\n                \"source\": \"resource_monitor\"\n            })\n        \n        # Sort alerts by severity and timestamp\n        severity_order = {\"critical\": 0, \"warning\": 1, \"info\": 2}\n        alerts.sort(key=lambda x: (severity_order.get(x[\"severity\"], 3), x[\"timestamp\"]))\n        \n        return {\n            \"total_alerts\": len(alerts),\n            \"critical_count\": sum(1 for a in alerts if a[\"severity\"] == \"critical\"),\n            \"warning_count\": sum(1 for a in alerts if a[\"severity\"] == \"warning\"),\n            \"alerts\": alerts[:20]  # Limit to 20 most important alerts\n        }\n    \n    def get_operation_breakdown(self) -> Dict[str, Any]:\n        \"\"\"Get breakdown of operations by type and performance.\n        \n        Returns:\n            Dictionary with operation statistics\n        \"\"\"\n        # Get operation stats from performance monitor\n        operation_stats = self.performance_monitor.get_operation_stats()\n        \n        # Sort operations by total duration\n        sorted_operations = sorted(\n            operation_stats.items(),\n            key=lambda x: x[1].get(\"total_duration_ms\", 0),\n            reverse=True\n        )\n        \n        operation_breakdown = []\n        for op_name, stats in sorted_operations[:10]:  # Top 10 operations\n            operation_breakdown.append({\n                \"operation\": op_name,\n                \"total_calls\": stats.get(\"count\", 0),\n                \"success_count\": stats.get(\"success_count\", 0),\n                \"error_count\": stats.get(\"error_count\", 0),\n                \"success_rate\": stats.get(\"success_rate\", 0.0),\n                \"avg_duration_ms\": stats.get(\"avg_duration_ms\", 0.0),\n                \"min_duration_ms\": stats.get(\"min_duration_ms\", 0.0),\n                \"max_duration_ms\": stats.get(\"max_duration_ms\", 0.0),\n                \"total_duration_ms\": stats.get(\"total_duration_ms\", 0.0)\n            })\n        \n        return {\n            \"total_operations_monitored\": len(operation_stats),\n            \"top_operations\": operation_breakdown,\n            \"generated_at\": datetime.utcnow().isoformat()\n        }\n\n\n# Global dashboard instance\n_global_dashboard: Optional[MonitoringDashboard] = None\n\n\ndef get_monitoring_dashboard() -> MonitoringDashboard:\n    \"\"\"Get the global monitoring dashboard instance.\"\"\"\n    global _global_dashboard\n    \n    if _global_dashboard is None:\n        _global_dashboard = MonitoringDashboard()\n    \n    return _global_dashboard"