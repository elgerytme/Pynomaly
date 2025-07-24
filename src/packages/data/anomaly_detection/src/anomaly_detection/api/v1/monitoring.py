"""Monitoring API endpoints."""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional, Dict, Any, List
from datetime import datetime

from ...domain.services.health_monitoring_service import (
    get_health_monitoring_service,
    SystemHealth,
    HealthCheck,
    HealthStatus
)
from ...infrastructure.observability.metrics import get_metrics_collector

router = APIRouter(prefix="/monitoring", tags=["monitoring"])


@router.get("/health", response_model=Dict[str, Any])
async def get_health() -> Dict[str, Any]:
    """Get system health status."""
    try:
        health_service = get_health_monitoring_service()
        health = await health_service.get_health_status()
        
        return {
            "status": health.overall_status.value,
            "timestamp": health.timestamp.isoformat(),
            "uptime_seconds": health.uptime_seconds,
            "checks": [
                {
                    "name": check.name,
                    "status": check.status.value,
                    "message": check.message,
                    "timestamp": check.timestamp.isoformat(),
                    "response_time_ms": check.response_time_ms,
                    "metadata": check.metadata or {}
                }
                for check in health.checks
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@router.get("/health/liveness")
async def liveness_check() -> Dict[str, str]:
    """Kubernetes liveness probe endpoint."""
    try:
        health_service = get_health_monitoring_service()
        is_alive = await health_service.get_liveness_check()
        
        if is_alive:
            return {"status": "alive"}
        else:
            raise HTTPException(status_code=503, detail="Service not alive")
            
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Liveness check failed: {str(e)}")


@router.get("/health/readiness")
async def readiness_check() -> Dict[str, str]:
    """Kubernetes readiness probe endpoint."""
    try:
        health_service = get_health_monitoring_service()
        is_ready = await health_service.get_readiness_check()
        
        if is_ready:
            return {"status": "ready"}
        else:
            raise HTTPException(status_code=503, detail="Service not ready")
            
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Readiness check failed: {str(e)}")


@router.get("/metrics", response_model=Dict[str, Any])
async def get_metrics(
    metric_name: Optional[str] = Query(None, description="Filter by metric name"),
    start_time: Optional[datetime] = Query(None, description="Start time for metrics"),
    end_time: Optional[datetime] = Query(None, description="End time for metrics")
) -> Dict[str, Any]:
    """Get system metrics."""
    try:
        metrics_collector = get_metrics_collector()
        
        # Get all metrics
        all_metrics = metrics_collector.get_all_metrics()
        
        # Filter by metric name if provided
        if metric_name:
            filtered_metrics = {k: v for k, v in all_metrics.items() if metric_name in k}
        else:
            filtered_metrics = all_metrics
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": filtered_metrics,
            "filters": {
                "metric_name": metric_name,
                "start_time": start_time.isoformat() if start_time else None,
                "end_time": end_time.isoformat() if end_time else None
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")


@router.get("/status", response_model=Dict[str, Any])
async def get_system_status() -> Dict[str, Any]:
    """Get comprehensive system status."""
    try:
        health_service = get_health_monitoring_service()
        metrics_collector = get_metrics_collector()
        
        # Get health status
        health = await health_service.get_health_status()
        
        # Get key metrics
        metrics = metrics_collector.get_all_metrics()
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "overall_status": health.overall_status.value,
            "uptime_seconds": health.uptime_seconds,
            "health_summary": {
                "total_checks": len(health.checks),
                "healthy_checks": len([c for c in health.checks if c.status == HealthStatus.HEALTHY]),
                "warning_checks": len([c for c in health.checks if c.status == HealthStatus.WARNING]),
                "critical_checks": len([c for c in health.checks if c.status == HealthStatus.CRITICAL])
            },
            "key_metrics": {
                "requests_total": metrics.get("requests_total", 0),
                "errors_total": metrics.get("errors_total", 0),
                "response_time_avg": metrics.get("response_time_avg", 0),
                "active_connections": metrics.get("active_connections", 0)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get system status: {str(e)}")


@router.post("/health/check/{check_name}")
async def trigger_health_check(check_name: str) -> Dict[str, Any]:
    """Manually trigger a specific health check."""
    try:
        health_service = get_health_monitoring_service()
        
        if check_name not in health_service.health_checks:
            available_checks = list(health_service.health_checks.keys())
            raise HTTPException(
                status_code=404, 
                detail=f"Health check '{check_name}' not found. Available checks: {available_checks}"
            )
        
        # Run the specific check
        check_func = health_service.health_checks[check_name]
        
        try:
            if hasattr(check_func, '__await__'):
                result = await check_func()
            else:
                result = check_func()
            
            if isinstance(result, HealthCheck):
                return {
                    "name": result.name,
                    "status": result.status.value,
                    "message": result.message,
                    "timestamp": result.timestamp.isoformat(),
                    "response_time_ms": result.response_time_ms,
                    "metadata": result.metadata or {}
                }
            else:
                return {
                    "name": check_name,
                    "status": "healthy" if result else "critical",
                    "message": "Check passed" if result else "Check failed",
                    "timestamp": datetime.utcnow().isoformat()
                }
                
        except Exception as check_e:
            return {
                "name": check_name,
                "status": "critical",
                "message": f"Check failed with error: {str(check_e)}",
                "timestamp": datetime.utcnow().isoformat(),
                "metadata": {"error": str(check_e)}
            }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to trigger health check: {str(e)}")


@router.get("/version", response_model=Dict[str, str])
async def get_version() -> Dict[str, str]:
    """Get service version information."""
    return {
        "service": "anomaly_detection",
        "version": "0.1.0",
        "build_date": "2024-01-01",
        "git_commit": "unknown",
        "python_version": "3.12+"
    }