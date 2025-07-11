"""
Comprehensive health check endpoints for Pynomaly production deployment.
Provides detailed health, readiness, and liveness checks for Kubernetes.
"""

import logging
import os
import time
from datetime import datetime
from typing import Any

import psutil
import redis
from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from pynomaly.infrastructure.config.settings import get_settings
from pynomaly.infrastructure.database.session import get_async_session

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/health", tags=["health"])


class HealthStatus(BaseModel):
    """Health status model"""

    status: str = Field(..., description="Overall health status")
    timestamp: datetime = Field(..., description="Health check timestamp")
    version: str = Field(..., description="Application version")
    uptime_seconds: float = Field(..., description="Application uptime in seconds")
    checks: dict[str, Any] = Field(..., description="Individual health checks")


class ReadinessStatus(BaseModel):
    """Readiness status model"""

    ready: bool = Field(..., description="Application readiness status")
    timestamp: datetime = Field(..., description="Readiness check timestamp")
    services: dict[str, bool] = Field(..., description="Service readiness status")
    dependencies: dict[str, Any] = Field(..., description="Dependency status")


class LivenessStatus(BaseModel):
    """Liveness status model"""

    alive: bool = Field(..., description="Application liveness status")
    timestamp: datetime = Field(..., description="Liveness check timestamp")
    memory_usage_mb: float = Field(..., description="Memory usage in MB")
    cpu_usage_percent: float = Field(..., description="CPU usage percentage")
    disk_usage_percent: float = Field(..., description="Disk usage percentage")


class StartupStatus(BaseModel):
    """Startup status model"""

    started: bool = Field(..., description="Application startup status")
    timestamp: datetime = Field(..., description="Startup check timestamp")
    initialization: dict[str, bool] = Field(..., description="Initialization status")
    startup_time_seconds: float = Field(..., description="Time taken to start")


# Global variables for tracking application state
_app_start_time = time.time()
_startup_completed = False
_initialization_status = {
    "database_connection": False,
    "redis_connection": False,
    "models_loaded": False,
    "configuration_loaded": False,
    "migrations_completed": False,
}


class HealthChecker:
    """Comprehensive health checking utility"""

    def __init__(self):
        self.settings = get_settings()

    async def check_database_health(self, db: AsyncSession) -> dict[str, Any]:
        """Check PostgreSQL database health"""
        try:
            start_time = time.time()
            result = await db.execute(text("SELECT 1"))
            response_time = (time.time() - start_time) * 1000

            # Check connection pool status
            pool_info = db.get_bind().pool
            pool_status = {
                "size": pool_info.size(),
                "checked_out": pool_info.checkedout(),
                "overflow": pool_info.overflow(),
                "invalidated": pool_info.invalidated(),
            }

            return {
                "status": "healthy",
                "response_time_ms": round(response_time, 2),
                "connection_pool": pool_status,
                "database_version": str(result.fetchone()[0]) if result else "unknown",
            }
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return {"status": "unhealthy", "error": str(e), "response_time_ms": None}

    async def check_redis_health(self) -> dict[str, Any]:
        """Check Redis cache health"""
        try:
            redis_client = redis.Redis.from_url(
                self.settings.REDIS_URL, socket_connect_timeout=5, socket_timeout=5
            )

            start_time = time.time()
            ping_result = redis_client.ping()
            response_time = (time.time() - start_time) * 1000

            # Get Redis info
            redis_info = redis_client.info()

            return {
                "status": "healthy" if ping_result else "unhealthy",
                "response_time_ms": round(response_time, 2),
                "version": redis_info.get("redis_version"),
                "memory_used_mb": round(
                    redis_info.get("used_memory", 0) / 1024 / 1024, 2
                ),
                "connected_clients": redis_info.get("connected_clients", 0),
            }
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return {"status": "unhealthy", "error": str(e), "response_time_ms": None}

    def check_system_resources(self) -> dict[str, Any]:
        """Check system resource usage"""
        try:
            # Memory usage
            memory = psutil.virtual_memory()
            memory_usage = {
                "total_mb": round(memory.total / 1024 / 1024, 2),
                "available_mb": round(memory.available / 1024 / 1024, 2),
                "used_mb": round(memory.used / 1024 / 1024, 2),
                "percent": memory.percent,
            }

            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()

            # Disk usage
            disk = psutil.disk_usage("/")
            disk_usage = {
                "total_gb": round(disk.total / 1024 / 1024 / 1024, 2),
                "used_gb": round(disk.used / 1024 / 1024 / 1024, 2),
                "free_gb": round(disk.free / 1024 / 1024 / 1024, 2),
                "percent": round((disk.used / disk.total) * 100, 2),
            }

            return {
                "memory": memory_usage,
                "cpu": {"percent": cpu_percent, "count": cpu_count},
                "disk": disk_usage,
                "load_average": list(os.getloadavg())
                if hasattr(os, "getloadavg")
                else None,
            }
        except Exception as e:
            logger.error(f"System resource check failed: {e}")
            return {"error": str(e), "memory": None, "cpu": None, "disk": None}

    async def check_external_services(self) -> dict[str, Any]:
        """Check external service dependencies"""
        external_checks = {}

        # Check if we can resolve DNS
        try:
            import socket

            socket.gethostbyname("google.com")
            external_checks["dns_resolution"] = {"status": "healthy"}
        except Exception as e:
            external_checks["dns_resolution"] = {"status": "unhealthy", "error": str(e)}

        # Check internet connectivity
        try:
            import urllib.request

            urllib.request.urlopen("https://httpbin.org/status/200", timeout=5)
            external_checks["internet_connectivity"] = {"status": "healthy"}
        except Exception as e:
            external_checks["internet_connectivity"] = {
                "status": "unhealthy",
                "error": str(e),
            }

        return external_checks


health_checker = HealthChecker()


@router.get("/", response_model=HealthStatus, summary="Comprehensive Health Check")
async def health_check(
    response: Response, db: AsyncSession = Depends(get_async_session)
) -> HealthStatus:
    """
    Comprehensive health check endpoint.
    Returns detailed status of all application components.
    """
    start_time = time.time()
    timestamp = datetime.utcnow()
    uptime = time.time() - _app_start_time

    # Perform all health checks
    checks = {}
    overall_status = "healthy"

    try:
        # Database health
        checks["database"] = await health_checker.check_database_health(db)
        if checks["database"]["status"] != "healthy":
            overall_status = "degraded"

        # Redis health
        checks["redis"] = await health_checker.check_redis_health()
        if checks["redis"]["status"] != "healthy":
            overall_status = "degraded"

        # System resources
        checks["system"] = health_checker.check_system_resources()
        memory_percent = checks["system"].get("memory", {}).get("percent", 0)
        cpu_percent = checks["system"].get("cpu", {}).get("percent", 0)
        disk_percent = checks["system"].get("disk", {}).get("percent", 0)

        # Check resource thresholds
        if memory_percent > 90 or cpu_percent > 90 or disk_percent > 90:
            overall_status = "degraded"

        # External services
        checks["external"] = await health_checker.check_external_services()

        # Application-specific checks
        checks["application"] = {
            "startup_completed": _startup_completed,
            "configuration_loaded": _initialization_status.get(
                "configuration_loaded", False
            ),
            "models_loaded": _initialization_status.get("models_loaded", False),
        }

        if not _startup_completed:
            overall_status = "starting"

        checks["response_time_ms"] = round((time.time() - start_time) * 1000, 2)

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        overall_status = "unhealthy"
        checks["error"] = str(e)

    # Set HTTP status based on health
    if overall_status == "unhealthy":
        response.status_code = 503
    elif overall_status == "degraded":
        response.status_code = 200  # Still operational but degraded

    return HealthStatus(
        status=overall_status,
        timestamp=timestamp,
        version=os.getenv("PYNOMALY_VERSION", "unknown"),
        uptime_seconds=round(uptime, 2),
        checks=checks,
    )


@router.get("/live", response_model=LivenessStatus, summary="Liveness Probe")
async def liveness_check(response: Response) -> LivenessStatus:
    """
    Kubernetes liveness probe endpoint.
    Checks if the application is running and responsive.
    """
    timestamp = datetime.utcnow()

    try:
        # Get system resource usage
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=0.1)
        disk = psutil.disk_usage("/")

        memory_usage_mb = memory.used / 1024 / 1024
        disk_usage_percent = (disk.used / disk.total) * 100

        # Check if resources are within acceptable limits
        alive = (
            memory_usage_mb < 4096  # Less than 4GB
            and cpu_percent < 95  # Less than 95% CPU
            and disk_usage_percent < 95  # Less than 95% disk
        )

        if not alive:
            response.status_code = 503

        return LivenessStatus(
            alive=alive,
            timestamp=timestamp,
            memory_usage_mb=round(memory_usage_mb, 2),
            cpu_usage_percent=round(cpu_percent, 2),
            disk_usage_percent=round(disk_usage_percent, 2),
        )

    except Exception as e:
        logger.error(f"Liveness check failed: {e}")
        response.status_code = 503
        return LivenessStatus(
            alive=False,
            timestamp=timestamp,
            memory_usage_mb=0,
            cpu_usage_percent=0,
            disk_usage_percent=0,
        )


@router.get("/ready", response_model=ReadinessStatus, summary="Readiness Probe")
async def readiness_check(
    response: Response, db: AsyncSession = Depends(get_async_session)
) -> ReadinessStatus:
    """
    Kubernetes readiness probe endpoint.
    Checks if the application is ready to serve traffic.
    """
    timestamp = datetime.utcnow()
    services = {}
    dependencies = {}

    try:
        # Check database connectivity
        try:
            await db.execute(text("SELECT 1"))
            services["database"] = True
            dependencies["database"] = {"status": "ready", "response_time_ms": "< 100"}
        except Exception as e:
            services["database"] = False
            dependencies["database"] = {"status": "not_ready", "error": str(e)}

        # Check Redis connectivity
        try:
            redis_client = redis.Redis.from_url(
                health_checker.settings.REDIS_URL, socket_timeout=2
            )
            redis_client.ping()
            services["redis"] = True
            dependencies["redis"] = {"status": "ready"}
        except Exception as e:
            services["redis"] = False
            dependencies["redis"] = {"status": "not_ready", "error": str(e)}

        # Check if startup is completed
        services["startup"] = _startup_completed
        dependencies["startup"] = {
            "status": "ready" if _startup_completed else "not_ready",
            "initialization": _initialization_status,
        }

        # Overall readiness
        ready = all(services.values())

        if not ready:
            response.status_code = 503

        return ReadinessStatus(
            ready=ready,
            timestamp=timestamp,
            services=services,
            dependencies=dependencies,
        )

    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        response.status_code = 503
        return ReadinessStatus(
            ready=False,
            timestamp=timestamp,
            services={"error": True},
            dependencies={"error": str(e)},
        )


@router.get("/startup", response_model=StartupStatus, summary="Startup Probe")
async def startup_check(response: Response) -> StartupStatus:
    """
    Kubernetes startup probe endpoint.
    Checks if the application has completed startup initialization.
    """
    timestamp = datetime.utcnow()
    startup_time = time.time() - _app_start_time

    try:
        # Check initialization status
        started = _startup_completed and all(_initialization_status.values())

        if not started:
            response.status_code = 503

        return StartupStatus(
            started=started,
            timestamp=timestamp,
            initialization=_initialization_status.copy(),
            startup_time_seconds=round(startup_time, 2),
        )

    except Exception as e:
        logger.error(f"Startup check failed: {e}")
        response.status_code = 503
        return StartupStatus(
            started=False,
            timestamp=timestamp,
            initialization={"error": str(e)},
            startup_time_seconds=round(startup_time, 2),
        )


@router.get("/metrics", summary="Prometheus Metrics")
async def metrics_endpoint():
    """
    Prometheus metrics endpoint for monitoring.
    Returns metrics in Prometheus format.
    """
    try:
        # Get system metrics
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent()
        disk = psutil.disk_usage("/")

        uptime = time.time() - _app_start_time

        metrics = [
            "# HELP pynomaly_uptime_seconds Application uptime in seconds",
            "# TYPE pynomaly_uptime_seconds counter",
            f"pynomaly_uptime_seconds {uptime}",
            "",
            "# HELP pynomaly_memory_usage_bytes Memory usage in bytes",
            "# TYPE pynomaly_memory_usage_bytes gauge",
            f"pynomaly_memory_usage_bytes {memory.used}",
            "",
            "# HELP pynomaly_cpu_usage_percent CPU usage percentage",
            "# TYPE pynomaly_cpu_usage_percent gauge",
            f"pynomaly_cpu_usage_percent {cpu_percent}",
            "",
            "# HELP pynomaly_disk_usage_bytes Disk usage in bytes",
            "# TYPE pynomaly_disk_usage_bytes gauge",
            f"pynomaly_disk_usage_bytes {disk.used}",
            "",
            "# HELP pynomaly_startup_completed Application startup status",
            "# TYPE pynomaly_startup_completed gauge",
            f"pynomaly_startup_completed {1 if _startup_completed else 0}",
        ]

        return Response(
            content="\n".join(metrics),
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    except Exception as e:
        logger.error(f"Metrics endpoint failed: {e}")
        raise HTTPException(status_code=500, detail="Metrics collection failed")


def mark_startup_completed():
    """Mark application startup as completed"""
    global _startup_completed
    _startup_completed = True
    logger.info("Application startup completed")


def update_initialization_status(component: str, status: bool):
    """Update initialization status for a specific component"""
    global _initialization_status
    _initialization_status[component] = status
    logger.info(f"Initialization status updated: {component} = {status}")
