"""Health check endpoints with comprehensive OpenAPI documentation."""

from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from pynomaly.infrastructure.config import Container
from pynomaly.infrastructure.monitoring.health_service import HealthService
from pynomaly.presentation.api.deps import get_container
from pynomaly.presentation.api.docs.response_models import HTTPResponses
from pynomaly.presentation.api.docs.schema_examples import SchemaExamples

router = APIRouter(
    prefix="/health",
    tags=["Health"],
    responses={
        500: HTTPResponses.server_error_500("Health check service unavailable"),
        503: HTTPResponses.server_error_500("Service temporarily unavailable"),
    },
)

# Global health service instance
health_service = HealthService()


class HealthCheckResponse(BaseModel):
    """Individual health check response."""

    name: str = Field(..., description="Name of the health check")
    status: str = Field(
        ...,
        description="Health check status",
        enum=["healthy", "degraded", "unhealthy"],
    )
    message: str = Field(..., description="Human-readable status message")
    duration_ms: float = Field(
        ..., description="Check execution time in milliseconds", ge=0
    )
    timestamp: datetime = Field(..., description="Check execution timestamp")
    details: dict[str, Any] = Field(
        default_factory=dict, description="Additional check details"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "name": "database",
                "status": "healthy",
                "message": "Database connection successful",
                "duration_ms": 15.2,
                "timestamp": "2024-12-25T10:30:00Z",
                "details": {"connection_pool_size": 10, "active_connections": 3},
            }
        }


class HealthResponse(BaseModel):
    """Comprehensive health check response."""

    overall_status: str = Field(
        ...,
        description="Overall system health status",
        enum=["healthy", "degraded", "unhealthy"],
    )
    timestamp: datetime = Field(..., description="Response generation timestamp")
    version: str = Field(..., description="Application version")
    uptime_seconds: float = Field(..., description="System uptime in seconds", ge=0)
    checks: dict[str, HealthCheckResponse] = Field(
        ..., description="Individual health check results"
    )
    summary: dict[str, Any] = Field(..., description="Health summary statistics")

    class Config:
        json_schema_extra = {"example": SchemaExamples.health_check_response()["value"]}


class SystemMetricsResponse(BaseModel):
    """System resource metrics response."""

    cpu_percent: float = Field(
        ..., description="CPU utilization percentage", ge=0, le=100
    )
    memory_percent: float = Field(
        ..., description="Memory utilization percentage", ge=0, le=100
    )
    disk_percent: float = Field(
        ..., description="Disk utilization percentage", ge=0, le=100
    )
    memory_available_mb: float = Field(..., description="Available memory in MB", ge=0)
    disk_available_gb: float = Field(
        ..., description="Available disk space in GB", ge=0
    )
    load_average: list[float] = Field(
        ..., description="System load average (1, 5, 15 minutes)"
    )
    network_io: dict[str, int] = Field(..., description="Network I/O statistics")
    process_count: int = Field(..., description="Number of active processes", ge=0)
    uptime_seconds: float = Field(..., description="System uptime in seconds", ge=0)

    class Config:
        json_schema_extra = {
            "example": {
                "cpu_percent": 23.5,
                "memory_percent": 68.2,
                "disk_percent": 45.8,
                "memory_available_mb": 2048.0,
                "disk_available_gb": 128.5,
                "load_average": [0.85, 0.92, 1.15],
                "network_io": {
                    "bytes_sent": 1048576,
                    "bytes_recv": 2097152,
                    "packets_sent": 1024,
                    "packets_recv": 2048,
                },
                "process_count": 127,
                "uptime_seconds": 86400.0,
            }
        }


@router.get(
    "/",
    response_model=HealthResponse,
    summary="Comprehensive Health Check",
    description="""
    Perform a comprehensive health check of all system components.

    This endpoint checks:
    - **System Resources**: CPU, memory, disk usage
    - **Database Connectivity**: Database connection and response time
    - **Cache Connectivity**: Redis/cache service availability
    - **Repository Access**: Data access layer functionality
    - **Algorithm Adapters**: ML library availability
    - **Configuration**: Security and environment settings

    The response includes individual check results and an overall status.
    Use this endpoint for detailed monitoring and troubleshooting.

    **Rate Limit**: 60 requests per minute
    """,
    responses={
        200: {
            "description": "Health check completed successfully",
            "content": {
                "application/json": {
                    "example": SchemaExamples.health_check_response()["value"]
                }
            },
        },
        503: HTTPResponses.server_error_500("Service unhealthy or unavailable"),
    },
)
async def health_check(
    container: Container = Depends(get_container),
    include_system: bool = Query(
        True, description="Include system resource checks (CPU, memory, disk)"
    ),
    include_database: bool = Query(
        True, description="Include database connectivity checks"
    ),
    include_cache: bool = Query(
        True, description="Include cache connectivity checks (Redis)"
    ),
) -> HealthResponse:
    """Comprehensive application health check with detailed component status."""
    settings = container.config()

    # Get database engine if available
    database_engine = None
    if include_database:
        try:
            if hasattr(container, "database_manager") and container.database_manager():
                database_engine = container.database_manager().engine
        except Exception:
            pass  # Database not configured

    # Get Redis client if available
    redis_client = None
    if include_cache and settings.cache_enabled:
        try:
            if hasattr(container, "redis_cache") and container.redis_cache():
                redis_client = container.redis_cache().client
        except Exception:
            pass  # Redis not configured

    # Custom application checks
    custom_checks = {
        "repositories": lambda: _check_repositories(container),
        "adapters": lambda: _check_adapters(container),
        "configuration": lambda: _check_configuration(settings),
    }

    # Perform comprehensive health check
    health_checks = await health_service.perform_comprehensive_health_check(
        database_engine=database_engine,
        redis_client=redis_client,
        custom_checks=custom_checks,
    )

    # Convert to response format
    check_responses = {
        name: HealthCheckResponse(
            name=check.name,
            status=check.status.value,
            message=check.message,
            duration_ms=check.duration_ms,
            timestamp=check.timestamp,
            details=check.details,
        )
        for name, check in health_checks.items()
    }

    overall_status = health_service.get_overall_status(health_checks)
    summary = health_service.get_health_summary()
    metrics = health_service.get_system_metrics()

    return HealthResponse(
        overall_status=overall_status.value,
        timestamp=datetime.now(UTC),
        version=settings.app.version,
        uptime_seconds=metrics.uptime_seconds,
        checks=check_responses,
        summary=summary,
    )


@router.get("/metrics", response_model=SystemMetricsResponse)
async def system_metrics() -> SystemMetricsResponse:
    """Get detailed system metrics."""
    metrics = health_service.get_system_metrics()

    return SystemMetricsResponse(
        cpu_percent=metrics.cpu_percent,
        memory_percent=metrics.memory_percent,
        disk_percent=metrics.disk_percent,
        memory_available_mb=metrics.memory_available_mb,
        disk_available_gb=metrics.disk_available_gb,
        load_average=metrics.load_average,
        network_io=metrics.network_io,
        process_count=metrics.process_count,
        uptime_seconds=metrics.uptime_seconds,
    )


@router.get("/history")
async def health_history(
    hours: int = Query(24, ge=1, le=168, description="Hours of history to retrieve"),
) -> list[dict[str, Any]]:
    """Get health check history."""
    history = health_service.get_health_history(hours=hours)

    # Convert to serializable format
    serializable_history = []
    for checks in history:
        serializable_checks = {
            name: {
                "name": check.name,
                "status": check.status.value,
                "message": check.message,
                "duration_ms": check.duration_ms,
                "timestamp": check.timestamp.isoformat(),
                "details": check.details,
            }
            for name, check in checks.items()
        }
        serializable_history.append(serializable_checks)

    return serializable_history


@router.get("/summary")
async def health_summary() -> dict[str, Any]:
    """Get health status summary."""
    return health_service.get_health_summary()


@router.get("/ready")
async def readiness_check(
    container: Container = Depends(get_container),
) -> dict[str, str]:
    """Kubernetes readiness probe - fast check for readiness to serve requests."""
    try:
        # Check critical components quickly
        container.config()

        # Check repositories are accessible
        container.detector_repository().count()
        container.dataset_repository().count()

        # Check if database is responsive (if configured)
        try:
            if hasattr(container, "database_manager") and container.database_manager():
                with container.database_manager().engine.connect() as conn:
                    conn.execute("SELECT 1").fetchone()
        except Exception:
            pass  # Database not critical for readiness

        return {"status": "ready", "timestamp": datetime.now(UTC).isoformat()}

    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "not_ready",
                "error": str(e),
                "timestamp": datetime.now(UTC).isoformat(),
            },
        )


@router.get("/live")
async def liveness_check() -> dict[str, str]:
    """Kubernetes liveness probe - simple check that application is alive."""
    try:
        # Basic application responsiveness check
        metrics = health_service.get_system_metrics()

        # Check if memory usage is reasonable (not leaked)
        if metrics.memory_percent > 98:
            raise HTTPException(
                status_code=503,
                detail={
                    "status": "unhealthy",
                    "reason": "memory_exhausted",
                    "memory_percent": metrics.memory_percent,
                    "timestamp": datetime.now(UTC).isoformat(),
                },
            )

        return {
            "status": "alive",
            "uptime_seconds": metrics.uptime_seconds,
            "timestamp": datetime.now(UTC).isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now(UTC).isoformat(),
            },
        )


def _check_repositories(container: Container) -> dict[str, Any]:
    """Check repository health."""
    try:
        detector_count = container.detector_repository().count()
        dataset_count = container.dataset_repository().count()
        result_count = container.result_repository().count()

        return {
            "status": "healthy",
            "message": f"Repositories accessible: {detector_count} detectors, {dataset_count} datasets, {result_count} results",
            "details": {
                "detectors": detector_count,
                "datasets": dataset_count,
                "results": result_count,
            },
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "message": f"Repository access failed: {e}",
            "details": {"error": str(e)},
        }


def _check_adapters(container: Container) -> dict[str, Any]:
    """Check algorithm adapter availability."""
    try:
        adapters = {}

        # Check PyOD adapter
        try:
            container.pyod_adapter()
            adapters["pyod"] = "available"
        except Exception:
            adapters["pyod"] = "unavailable"

        # Check sklearn adapter
        try:
            container.sklearn_adapter()
            adapters["sklearn"] = "available"
        except Exception:
            adapters["sklearn"] = "unavailable"

        # Check optional adapters
        optional_adapters = ["pytorch_adapter", "pygod_adapter"]
        for adapter_name in optional_adapters:
            try:
                if hasattr(container, adapter_name):
                    getattr(container, adapter_name)()
                    adapters[adapter_name.replace("_adapter", "")] = "available"
                else:
                    adapters[adapter_name.replace("_adapter", "")] = "not_configured"
            except Exception:
                adapters[adapter_name.replace("_adapter", "")] = "unavailable"

        available_count = sum(
            1 for status in adapters.values() if status == "available"
        )

        return {
            "status": "healthy" if available_count >= 2 else "degraded",
            "message": f"{available_count} adapters available",
            "details": adapters,
        }

    except Exception as e:
        return {
            "status": "unhealthy",
            "message": f"Adapter check failed: {e}",
            "details": {"error": str(e)},
        }


def _check_configuration(settings) -> dict[str, Any]:
    """Check configuration health."""
    try:
        issues = []

        # Check critical settings
        if not settings.secret_key or settings.secret_key == "your-secret-key":
            issues.append("Default secret key in use")

        if settings.debug and settings.app.environment == "production":
            issues.append("Debug mode enabled in production")

        if not settings.cors_origins:
            issues.append("No CORS origins configured")

        status = "healthy" if not issues else "degraded"
        message = (
            "Configuration OK"
            if not issues
            else f"Configuration issues: {', '.join(issues)}"
        )

        return {
            "status": status,
            "message": message,
            "details": {
                "environment": settings.app.environment,
                "debug": settings.debug,
                "issues": issues,
            },
        }

    except Exception as e:
        return {
            "status": "unhealthy",
            "message": f"Configuration check failed: {e}",
            "details": {"error": str(e)},
        }
