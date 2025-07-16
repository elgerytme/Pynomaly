"""Monitoring and observability API endpoints."""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from pydantic import BaseModel

from monorepo.infrastructure.auth import (
    UserModel,
    require_super_admin,
    require_tenant_admin,
)
from monorepo.infrastructure.monitoring.health_checks import (
    ProbeResponse,
    SystemHealth,
    get_health_checker,
    liveness_probe,
    readiness_probe,
)
from monorepo.infrastructure.monitoring.prometheus_metrics import get_metrics_service
from monorepo.infrastructure.monitoring.telemetry import get_telemetry
from monorepo.presentation.api.docs.response_models import (
    HTTPResponses,
    SuccessResponse,
)

router = APIRouter(
    prefix="/monitoring",
    tags=["Monitoring & Observability"],
    responses={
        500: HTTPResponses.server_error_500(),
    },
)


class MetricsResponse(BaseModel):
    """Response model for metrics data."""

    format: str
    size_bytes: int
    timestamp: str


@router.get(
    "/health",
    response_model=SuccessResponse[SystemHealth],
    summary="System Health Check",
    description="""
    Get comprehensive system health status including all components.

    This endpoint performs health checks on:
    - **System Resources**: CPU, memory, disk usage
    - **Application Components**: Model repository, detector service, streaming service
    - **External Dependencies**: Database, cache, file system

    **Health Status Levels:**
    - `healthy`: All systems operational
    - `degraded`: Some components experiencing issues but system functional
    - `unhealthy`: Critical components failing, system may not function properly
    - `unknown`: Unable to determine component status

    **Use Cases:**
    - Application monitoring and alerting
    - Load balancer health checks
    - Troubleshooting system issues
    - Capacity planning and optimization

    **Response Includes:**
    - Overall system status and message
    - Individual component health results
    - Response times for each health check
    - Detailed metrics and resource usage
    - System uptime and version information
    """,
    responses={
        200: HTTPResponses.ok_200("System health retrieved successfully"),
        503: {
            "description": "Service Unavailable - System unhealthy",
            "content": {
                "application/json": {
                    "example": {
                        "success": False,
                        "message": "System unhealthy: 2 component(s) failing",
                        "data": {
                            "status": "unhealthy",
                            "message": "System unhealthy: 2 component(s) failing",
                            "checks": [],
                        },
                    }
                }
            },
        },
    },
)
async def get_system_health(
    current_user: UserModel = Depends(require_tenant_admin),
) -> SuccessResponse[SystemHealth]:
    """Get comprehensive system health status."""
    try:
        checker = get_health_checker()
        health = await checker.get_system_health(version="1.0.0")

        # Return appropriate HTTP status based on health
        if health.status.value == "unhealthy":
            raise HTTPException(
                status_code=503,
                detail={
                    "success": False,
                    "message": health.message,
                    "data": health.to_dict(),
                },
            )

        return SuccessResponse(
            data=health, message="System health retrieved successfully"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to check system health: {str(e)}"
        )


@router.get(
    "/health/live",
    response_model=ProbeResponse,
    summary="Liveness Probe",
    description="""
    Kubernetes liveness probe endpoint for basic application health.

    This is a lightweight check that verifies the application process is running
    and responsive. It does not check external dependencies or detailed component health.

    **Kubernetes Configuration:**
    ```yaml
    livenessProbe:
      httpGet:
        path: /api/monitoring/health/live
        port: 8000
      initialDelaySeconds: 30
      periodSeconds: 10
      timeoutSeconds: 5
      failureThreshold: 3
    ```

    **Response:**
    - `status`: Always "alive" if application is running
    - `timestamp`: Current timestamp
    - `details`: Basic application information (uptime)
    """,
    responses={
        200: HTTPResponses.ok_200("Application is alive"),
    },
)
async def liveness_check() -> ProbeResponse:
    """Kubernetes liveness probe - basic application health."""
    return await liveness_probe()


@router.get(
    "/health/ready",
    response_model=ProbeResponse,
    summary="Readiness Probe",
    description="""
    Kubernetes readiness probe endpoint for traffic readiness.

    This endpoint checks if the application is ready to receive traffic by
    verifying critical components are healthy and operational.

    **Critical Components Checked:**
    - Memory usage within acceptable limits
    - File system accessible and writable
    - Model repository operational

    **Kubernetes Configuration:**
    ```yaml
    readinessProbe:
      httpGet:
        path: /api/monitoring/health/ready
        port: 8000
      initialDelaySeconds: 15
      periodSeconds: 5
      timeoutSeconds: 3
      failureThreshold: 2
    ```

    **Response:**
    - `status`: "ready" or "not_ready"
    - `details`: Status of each critical component
    """,
    responses={
        200: HTTPResponses.ok_200("Application is ready"),
        503: {
            "description": "Service Unavailable - Application not ready",
            "content": {
                "application/json": {
                    "example": {
                        "status": "not_ready",
                        "timestamp": "2024-12-26T10:30:00Z",
                        "details": {
                            "memory": "healthy",
                            "filesystem": "unhealthy",
                            "model_repository": "healthy",
                        },
                    }
                }
            },
        },
    },
)
async def readiness_check(response: Response) -> ProbeResponse:
    """Kubernetes readiness probe - traffic readiness check."""
    probe_result = await readiness_probe()

    if probe_result.status == "not_ready":
        response.status_code = 503

    return probe_result


@router.get(
    "/metrics",
    response_class=Response,
    summary="Prometheus Metrics",
    description="""
    Prometheus metrics endpoint for scraping application metrics.

    Returns metrics in Prometheus exposition format for monitoring and alerting.

    **Metric Categories:**
    - **HTTP Metrics**: Request rates, response times, status codes
    - **Detection Metrics**: Detection rates, accuracy, processing times
    - **Training Metrics**: Training duration, model sizes, success rates
    - **Streaming Metrics**: Throughput, buffer utilization, backpressure events
    - **Ensemble Metrics**: Prediction rates, agreement ratios, voting strategies
    - **System Metrics**: CPU, memory, active models/streams
    - **Cache Metrics**: Hit ratios, operation counts, cache sizes
    - **Error Metrics**: Error rates by type and component
    - **Quality Metrics**: Data quality scores, prediction confidence
    - **Business Metrics**: Datasets processed, API response sizes

    **Prometheus Configuration:**
    ```yaml
    scrape_configs:
      - job_name: 'monorepo'
        static_configs:
          - targets: ['localhost:8000']
        scrape_interval: 30s
        metrics_path: '/api/monitoring/metrics'
    ```

    **Example Metrics:**
    ```
    # HELP pynomaly_http_requests_total Total HTTP requests
    # TYPE pynomaly_http_requests_total counter
    pynomaly_http_requests_total{method="GET",endpoint="/api/detect",status="200"} 1250

    # HELP pynomaly_detection_duration_seconds Anomaly detection duration
    # TYPE pynomaly_detection_duration_seconds histogram
    pynomaly_detection_duration_seconds_bucket{algorithm="IsolationForest",le="0.1"} 850
    ```
    """,
    responses={
        200: {
            "description": "Prometheus metrics data",
            "content": {
                "text/plain": {
                    "example": '# HELP pynomaly_http_requests_total Total HTTP requests\\n# TYPE pynomaly_http_requests_total counter\\npynomaly_http_requests_total{method="GET",endpoint="/api/detect",status="200"} 1250\\n'
                }
            },
        }
    },
)
async def get_prometheus_metrics(
    current_user: UserModel = Depends(require_super_admin),
) -> Response:
    """Get Prometheus metrics in exposition format."""
    try:
        metrics_service = get_metrics_service()

        if not metrics_service:
            # Return empty metrics if service not available
            content = "# Prometheus metrics service not available\n"
        else:
            metrics_data = metrics_service.get_metrics_data()
            content = metrics_data.decode("utf-8")

        return Response(
            content=content, media_type="text/plain; version=0.0.4; charset=utf-8"
        )

    except Exception as e:
        # Return error metrics instead of HTTP error for Prometheus compatibility
        content = f"# ERROR: Failed to collect metrics: {str(e)}\n"
        return Response(
            content=content, media_type="text/plain; version=0.0.4; charset=utf-8"
        )


@router.get(
    "/metrics/info",
    response_model=SuccessResponse[MetricsResponse],
    summary="Metrics Information",
    description="""
    Get information about available metrics without returning the full data.

    Useful for:
    - Checking metrics service availability
    - Monitoring metrics data size
    - Verifying metrics endpoint functionality
    - Debugging metrics collection issues
    """,
    responses={
        200: HTTPResponses.ok_200("Metrics information retrieved"),
    },
)
async def get_metrics_info(
    current_user: UserModel = Depends(require_tenant_admin),
) -> SuccessResponse[MetricsResponse]:
    """Get information about available metrics."""
    try:
        metrics_service = get_metrics_service()

        if not metrics_service:
            raise HTTPException(status_code=503, detail="Metrics service not available")

        metrics_data = metrics_service.get_metrics_data()

        from datetime import datetime

        metrics_info = MetricsResponse(
            format="prometheus",
            size_bytes=len(metrics_data),
            timestamp=datetime.now().isoformat(),
        )

        return SuccessResponse(
            data=metrics_info, message="Metrics information retrieved successfully"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get metrics info: {str(e)}"
        )


@router.get(
    "/telemetry/status",
    response_model=SuccessResponse[dict[str, Any]],
    summary="Telemetry Status",
    description="""
    Get status and configuration of telemetry services.

    **Telemetry Components:**
    - **Tracing**: Distributed tracing with OpenTelemetry
    - **Metrics**: Custom metrics collection and export
    - **Logging**: Structured logging and correlation

    **Information Provided:**
    - Service availability and configuration
    - Export endpoints and destinations
    - Instrumentation status
    - Resource information (service name, version, environment)
    """,
    responses={
        200: HTTPResponses.ok_200("Telemetry status retrieved"),
    },
)
async def get_telemetry_status(
    current_user: UserModel = Depends(require_tenant_admin),
) -> SuccessResponse[dict[str, Any]]:
    """Get telemetry service status and configuration."""
    try:
        telemetry = get_telemetry()
        metrics_service = get_metrics_service()

        status = {
            "telemetry_service": {
                "available": telemetry is not None,
                "tracing_enabled": telemetry.tracer is not None if telemetry else False,
                "metrics_enabled": telemetry.meter is not None if telemetry else False,
            },
            "prometheus_service": {
                "available": metrics_service is not None,
                "metrics_count": len(metrics_service.metrics) if metrics_service else 0,
                "server_started": (
                    getattr(metrics_service, "server_started", False)
                    if metrics_service
                    else False
                ),
            },
        }

        if telemetry and hasattr(telemetry, "settings"):
            status["configuration"] = {
                "service_name": getattr(telemetry.settings, "app", {}).get(
                    "name", "monorepo"
                ),
                "environment": getattr(telemetry.settings, "app", {}).get(
                    "environment", "unknown"
                ),
                "version": getattr(telemetry.settings, "app", {}).get(
                    "version", "unknown"
                ),
            }

        return SuccessResponse(
            data=status, message="Telemetry status retrieved successfully"
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get telemetry status: {str(e)}"
        )


@router.post(
    "/metrics/application-info",
    response_model=SuccessResponse[str],
    summary="Set Application Info",
    description="""
    Set application information in metrics for tracking and identification.

    This endpoint allows updating the application info metric with current
    deployment information, useful for:
    - Version tracking across deployments
    - Environment identification
    - Build and deployment correlation
    - Monitoring dashboard labeling
    """,
    responses={
        200: HTTPResponses.ok_200("Application info updated"),
    },
)
async def set_application_info(
    version: str = Query(..., description="Application version"),
    environment: str = Query("production", description="Environment name"),
    build_time: str = Query(..., description="Build timestamp"),
    git_commit: str = Query("unknown", description="Git commit hash"),
) -> SuccessResponse[str]:
    """Set application information in metrics."""
    try:
        metrics_service = get_metrics_service()

        if not metrics_service:
            raise HTTPException(status_code=503, detail="Metrics service not available")

        metrics_service.set_application_info(
            version=version,
            environment=environment,
            build_time=build_time,
            git_commit=git_commit,
        )

        return SuccessResponse(
            data="Application info updated",
            message=f"Set application info: {version} ({environment})",
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to set application info: {str(e)}"
        )


@router.get(
    "/components",
    response_model=SuccessResponse[dict[str, Any]],
    summary="List Monitored Components",
    description="""
    Get list of all monitored components and their last known status.

    **Component Categories:**
    - **System**: CPU, memory, filesystem, network
    - **Application**: Model repository, detector service, streaming service
    - **External**: Database, cache, external APIs

    **Status Information:**
    - Current health status
    - Last check timestamp
    - Response time
    - Component-specific details
    """,
    responses={
        200: HTTPResponses.ok_200("Component list retrieved"),
    },
)
async def list_monitored_components() -> SuccessResponse[dict[str, Any]]:
    """Get list of all monitored components and their status."""
    try:
        checker = get_health_checker()
        cached_results = checker.get_cached_results()

        components = {}
        for name, result in cached_results.items():
            components[name] = {
                "component_type": result.component_type.value,
                "status": result.status.value,
                "message": result.message,
                "last_check": result.timestamp.isoformat(),
                "response_time_ms": result.response_time_ms,
            }

        summary = {
            "total_components": len(components),
            "components": components,
            "registered_checks": list(checker._check_functions.keys()),
        }

        return SuccessResponse(
            data=summary, message=f"Retrieved {len(components)} monitored components"
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to list components: {str(e)}"
        )
