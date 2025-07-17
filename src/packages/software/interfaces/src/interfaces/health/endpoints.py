"""
Health Check Endpoints for Package Monitoring

This module provides HTTP endpoints for health checks and monitoring.
"""

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum
import asyncio
import json
import logging

logger = logging.getLogger(__name__)

# Import health monitor if available
try:
    from ...ops.infrastructure.src.infrastructure.monitoring.health_monitor import (
        HealthMonitor, HealthStatus, PackageHealth
    )
    HEALTH_MONITOR_AVAILABLE = True
except ImportError:
    HEALTH_MONITOR_AVAILABLE = False
    logger.warning("Health monitor not available, using basic health checks")


class HealthCheckStatus(str, Enum):
    """Health check status enumeration"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class HealthCheckResponse(BaseModel):
    """Health check response model"""
    status: HealthCheckStatus
    timestamp: datetime
    checks: Dict[str, Any]
    version: str = "0.1.0"
    uptime: Optional[float] = None


class PackageHealthResponse(BaseModel):
    """Package health response model"""
    package_name: str
    version: str
    status: HealthCheckStatus
    last_check: datetime
    metrics: List[Dict[str, Any]]
    dependencies_status: Dict[str, str]
    error_count: int = 0
    warning_count: int = 0


class SystemHealthResponse(BaseModel):
    """System health response model"""
    timestamp: datetime
    system_metrics: Dict[str, Any]
    package_summary: Dict[str, int]
    packages: Dict[str, PackageHealthResponse]


# Create router
router = APIRouter(prefix="/health", tags=["health"])

# Global health monitor instance
health_monitor: Optional[HealthMonitor] = None


async def get_health_monitor() -> Optional[HealthMonitor]:
    """Get or create health monitor instance"""
    global health_monitor
    if health_monitor is None and HEALTH_MONITOR_AVAILABLE:
        health_monitor = HealthMonitor()
    return health_monitor


@router.get("/", response_model=HealthCheckResponse)
async def basic_health_check() -> HealthCheckResponse:
    """Basic health check endpoint"""
    try:
        # Basic checks
        checks = {
            "database": "healthy",  # Would check database connection
            "redis": "healthy",     # Would check Redis connection
            "filesystem": "healthy", # Would check filesystem
            "memory": "healthy"     # Would check memory usage
        }
        
        # Determine overall status
        status = HealthCheckStatus.HEALTHY
        if any(check != "healthy" for check in checks.values()):
            status = HealthCheckStatus.DEGRADED
        
        return HealthCheckResponse(
            status=status,
            timestamp=datetime.now(),
            checks=checks,
            uptime=None  # Would calculate actual uptime
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthCheckResponse(
            status=HealthCheckStatus.UNHEALTHY,
            timestamp=datetime.now(),
            checks={"error": str(e)}
        )


@router.get("/liveness")
async def liveness_probe() -> JSONResponse:
    """Kubernetes liveness probe endpoint"""
    return JSONResponse(
        status_code=200,
        content={"status": "alive", "timestamp": datetime.now().isoformat()}
    )


@router.get("/readiness")
async def readiness_probe() -> JSONResponse:
    """Kubernetes readiness probe endpoint"""
    try:
        # Check if service is ready to handle requests
        # This could include checking database connections, etc.
        ready = True
        
        if ready:
            return JSONResponse(
                status_code=200,
                content={"status": "ready", "timestamp": datetime.now().isoformat()}
            )
        else:
            return JSONResponse(
                status_code=503,
                content={"status": "not ready", "timestamp": datetime.now().isoformat()}
            )
            
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={"status": "error", "error": str(e), "timestamp": datetime.now().isoformat()}
        )


@router.get("/package/{package_name}", response_model=PackageHealthResponse)
async def get_package_health(
    package_name: str,
    monitor: Optional[HealthMonitor] = Depends(get_health_monitor)
) -> PackageHealthResponse:
    """Get health information for a specific package"""
    try:
        if monitor:
            health = await monitor.check_package_health(package_name)
            return PackageHealthResponse(
                package_name=health.package_name,
                version=health.version,
                status=HealthCheckStatus(health.status.value),
                last_check=health.last_check,
                metrics=[
                    {
                        "name": metric.name,
                        "value": metric.value,
                        "unit": metric.unit,
                        "timestamp": metric.timestamp,
                        "status": metric.status.value
                    }
                    for metric in health.metrics
                ],
                dependencies_status={
                    name: status.value for name, status in health.dependencies_status.items()
                },
                error_count=health.error_count,
                warning_count=health.warning_count
            )
        else:
            # Fallback basic health check
            return PackageHealthResponse(
                package_name=package_name,
                version="unknown",
                status=HealthCheckStatus.UNKNOWN,
                last_check=datetime.now(),
                metrics=[],
                dependencies_status={},
                error_count=0,
                warning_count=0
            )
            
    except Exception as e:
        logger.error(f"Failed to get package health for {package_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get package health: {e}")


@router.get("/system", response_model=SystemHealthResponse)
async def get_system_health(
    monitor: Optional[HealthMonitor] = Depends(get_health_monitor)
) -> SystemHealthResponse:
    """Get comprehensive system health information"""
    try:
        if monitor:
            report = await monitor.generate_health_report()
            
            # Convert to response format
            packages = {}
            for package_name, package_data in report["packages"].items():
                packages[package_name] = PackageHealthResponse(
                    package_name=package_data["package_name"],
                    version=package_data["version"],
                    status=HealthCheckStatus(package_data["status"]),
                    last_check=datetime.fromisoformat(package_data["last_check"]),
                    metrics=package_data["metrics"],
                    dependencies_status=package_data["dependencies_status"],
                    error_count=package_data.get("error_count", 0),
                    warning_count=package_data.get("warning_count", 0)
                )
            
            return SystemHealthResponse(
                timestamp=datetime.fromisoformat(report["timestamp"]),
                system_metrics=report["system_health"],
                package_summary=report["summary"],
                packages=packages
            )
        else:
            # Fallback system health
            return SystemHealthResponse(
                timestamp=datetime.now(),
                system_metrics={},
                package_summary={
                    "total_packages": 0,
                    "healthy_packages": 0,
                    "degraded_packages": 0,
                    "unhealthy_packages": 0,
                    "unknown_packages": 0
                },
                packages={}
            )
            
    except Exception as e:
        logger.error(f"Failed to get system health: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get system health: {e}")


@router.get("/metrics/prometheus")
async def get_prometheus_metrics(
    monitor: Optional[HealthMonitor] = Depends(get_health_monitor)
) -> str:
    """Get metrics in Prometheus format"""
    try:
        if monitor:
            report = await monitor.generate_health_report()
            
            # Convert to Prometheus format
            metrics_lines = []
            
            # System metrics
            for metric_name, metric_data in report["system_health"].items():
                value = metric_data["value"]
                metrics_lines.append(f"system_{metric_name} {value}")
            
            # Package metrics
            for package_name, package_data in report["packages"].items():
                status_value = 1 if package_data["status"] == "healthy" else 0
                metrics_lines.append(f'package_health{{package="{package_name}"}} {status_value}')
                
                for metric in package_data["metrics"]:
                    metric_name = metric["name"]
                    value = metric["value"]
                    metrics_lines.append(f'package_{metric_name}{{package="{package_name}"}} {value}')
            
            return "\\n".join(metrics_lines)
        else:
            return "# Health monitor not available"
            
    except Exception as e:
        logger.error(f"Failed to get Prometheus metrics: {e}")
        return f"# Error: {e}"


@router.post("/alert")
async def trigger_alert(
    alert_data: Dict[str, Any],
    monitor: Optional[HealthMonitor] = Depends(get_health_monitor)
) -> JSONResponse:
    """Trigger a health alert"""
    try:
        # In a real implementation, this would integrate with alerting systems
        logger.warning(f"Health alert triggered: {alert_data}")
        
        return JSONResponse(
            status_code=200,
            content={"status": "alert_received", "timestamp": datetime.now().isoformat()}
        )
        
    except Exception as e:
        logger.error(f"Failed to trigger alert: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "error": str(e), "timestamp": datetime.now().isoformat()}
        )


@router.get("/dashboard")
async def get_health_dashboard_data(
    monitor: Optional[HealthMonitor] = Depends(get_health_monitor)
) -> Dict[str, Any]:
    """Get data for health monitoring dashboard"""
    try:
        if monitor:
            report = await monitor.generate_health_report()
            
            # Create dashboard-friendly data structure
            dashboard_data = {
                "timestamp": report["timestamp"],
                "overview": {
                    "total_packages": report["summary"]["total_packages"],
                    "healthy_count": report["summary"]["healthy_packages"],
                    "degraded_count": report["summary"]["degraded_packages"],
                    "unhealthy_count": report["summary"]["unhealthy_packages"],
                    "unknown_count": report["summary"]["unknown_packages"]
                },
                "system_metrics": [
                    {
                        "name": name,
                        "value": data["value"],
                        "unit": data["unit"],
                        "status": data["status"],
                        "threshold_warning": data.get("threshold_warning"),
                        "threshold_critical": data.get("threshold_critical")
                    }
                    for name, data in report["system_health"].items()
                ],
                "packages": [
                    {
                        "name": name,
                        "status": data["status"],
                        "version": data["version"],
                        "metrics_count": len(data["metrics"]),
                        "dependencies_count": len(data["dependencies_status"]),
                        "error_count": data.get("error_count", 0),
                        "warning_count": data.get("warning_count", 0)
                    }
                    for name, data in report["packages"].items()
                ]
            }
            
            return dashboard_data
        else:
            return {"error": "Health monitor not available"}
            
    except Exception as e:
        logger.error(f"Failed to get dashboard data: {e}")
        return {"error": str(e)}


# Export router for inclusion in main FastAPI app
__all__ = ["router"]