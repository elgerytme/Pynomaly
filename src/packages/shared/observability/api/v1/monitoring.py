"""Monitoring and metrics endpoints."""

from datetime import datetime, timedelta
from typing import Dict, Any

from fastapi import APIRouter, HTTPException, status

from ...infrastructure.monitoring import (
    get_metrics_collector,
    get_performance_monitor
)
from ...infrastructure.monitoring.dashboard import get_monitoring_dashboard
from ...infrastructure.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)

metrics_collector = get_metrics_collector()
performance_monitor = get_performance_monitor()
monitoring_dashboard = get_monitoring_dashboard()


@router.get("/metrics")
async def get_metrics() -> Dict[str, Any]:
    """Get system metrics and statistics."""
    try:
        return {
            "metrics_summary": metrics_collector.get_summary_stats(),
            "performance_summary": performance_monitor.get_performance_summary(),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error("Failed to get metrics", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve metrics: {str(e)}"
        )


@router.get("/metrics/performance/{operation}")
async def get_operation_performance(operation: str) -> Dict[str, Any]:
    """Get performance metrics for a specific operation."""
    try:
        stats = performance_monitor.get_operation_stats(operation)
        recent_profiles = performance_monitor.get_recent_profiles(
            operation=operation,
            limit=10
        )
        
        return {
            "operation": operation,
            "statistics": stats,
            "recent_profiles": [
                {
                    "timestamp": p.timestamp.isoformat(),
                    "duration_ms": p.total_duration_ms,
                    "success": p.success,
                    "memory_mb": p.memory_usage_mb,
                    "peak_memory_mb": p.peak_memory_mb
                }
                for p in recent_profiles
            ]
        }
    except Exception as e:
        logger.error("Failed to get operation performance", 
                    operation=operation, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve performance metrics: {str(e)}"
        )


@router.get("/resources")
async def get_resource_usage() -> Dict[str, Any]:
    """Get system resource usage information."""
    try:
        recent_usage = performance_monitor.get_resource_usage(
            since=datetime.utcnow() - timedelta(hours=1),
            limit=60  # Last hour of data
        )
        
        return {
            "resource_usage": [
                {
                    "timestamp": usage.timestamp.isoformat(),
                    "cpu_percent": usage.cpu_percent,
                    "memory_mb": usage.memory_mb,
                    "memory_percent": usage.memory_percent,
                    "disk_io_read_mb": usage.disk_io_read_mb,
                    "disk_io_write_mb": usage.disk_io_write_mb,
                    "network_sent_mb": usage.network_sent_mb,
                    "network_received_mb": usage.network_received_mb
                }
                for usage in recent_usage
            ],
            "count": len(recent_usage),
            "period_hours": 1
        }
    except Exception as e:
        logger.error("Failed to get resource usage", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve resource usage: {str(e)}"
        )


@router.get("/dashboard/summary")
async def get_dashboard_summary() -> Dict[str, Any]:
    """Get comprehensive dashboard summary."""
    try:
        summary = await monitoring_dashboard.get_dashboard_summary()
        
        return {
            "summary": {
                "overall_health_status": summary.overall_health_status,
                "healthy_checks": summary.healthy_checks,
                "degraded_checks": summary.degraded_checks,
                "unhealthy_checks": summary.unhealthy_checks,
                "total_operations": summary.total_operations,
                "operations_last_hour": summary.operations_last_hour,
                "avg_response_time_ms": summary.avg_response_time_ms,
                "success_rate": summary.success_rate,
                "current_memory_mb": summary.current_memory_mb,
                "current_cpu_percent": summary.current_cpu_percent,
                "peak_memory_mb": summary.peak_memory_mb,
                "total_models": summary.total_models,
                "active_detections": summary.active_detections,
                "anomalies_detected_today": summary.anomalies_detected_today,
                "active_alerts": summary.active_alerts,
                "recent_errors": summary.recent_errors,
                "slow_operations": summary.slow_operations,
                "generated_at": summary.generated_at.isoformat()
            }
        }
    except Exception as e:
        logger.error("Failed to get dashboard summary", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve dashboard summary: {str(e)}"
        )


@router.get("/dashboard/trends")
async def get_performance_trends(
    hours: int = 24
) -> Dict[str, Any]:
    """Get performance trends over time."""
    try:
        if hours < 1 or hours > 168:  # Limit to 1 hour - 1 week
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Hours must be between 1 and 168"
            )
        
        trends = monitoring_dashboard.get_performance_trends(hours)
        return trends
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get performance trends", 
                    hours=hours, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve performance trends: {str(e)}"
        )


@router.get("/dashboard/alerts")
async def get_alerts() -> Dict[str, Any]:
    """Get current alerts and issues."""
    try:
        alerts = monitoring_dashboard.get_alert_summary()
        return alerts
    except Exception as e:
        logger.error("Failed to get alerts", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve alerts: {str(e)}"
        )


@router.get("/dashboard/operations")
async def get_operation_breakdown() -> Dict[str, Any]:
    """Get breakdown of operations by type and performance."""
    try:
        breakdown = monitoring_dashboard.get_operation_breakdown()
        return breakdown
    except Exception as e:
        logger.error("Failed to get operation breakdown", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve operation breakdown: {str(e)}"
        )