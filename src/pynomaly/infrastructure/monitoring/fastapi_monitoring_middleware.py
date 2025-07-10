"""FastAPI monitoring middleware for Pynomaly.

This middleware integrates with the production monitoring system to:
- Track HTTP requests and responses
- Monitor API performance
- Record anomaly detection metrics
- Handle health checks
- Integrate with alerting system
"""

import time
from collections.abc import Callable
from typing import Any
from uuid import uuid4

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from ...shared.logging import get_logger
from .production_monitoring_integration import ProductionMonitoringIntegration

logger = get_logger(__name__)


class MonitoringMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for production monitoring."""

    def __init__(self, app: ASGIApp, monitoring: ProductionMonitoringIntegration):
        super().__init__(app)
        self.monitoring = monitoring
        self.active_requests: dict[str, dict[str, Any]] = {}

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with monitoring."""
        # Generate request ID
        request_id = str(uuid4())

        # Skip monitoring for certain paths
        if self._should_skip_monitoring(request):
            return await call_next(request)

        # Record request start
        start_time = time.time()
        request_info = {
            "start_time": start_time,
            "method": request.method,
            "path": request.url.path,
            "client_ip": self._get_client_ip(request),
            "user_agent": request.headers.get("user-agent", ""),
        }

        self.active_requests[request_id] = request_info

        # Add monitoring context to request
        request.state.monitoring_request_id = request_id
        request.state.monitoring_start_time = start_time

        response = None
        status_code = 500
        exception_occurred = False

        try:
            # Process request
            response = await call_next(request)
            status_code = response.status_code

        except Exception as e:
            exception_occurred = True
            logger.error(f"Request {request_id} failed with exception: {e}")
            # Re-raise the exception
            raise

        finally:
            # Record request completion
            end_time = time.time()
            duration = end_time - start_time

            # Clean up active requests
            self.active_requests.pop(request_id, None)

            # Record metrics
            await self._record_request_metrics(
                request, response, duration, status_code, exception_occurred
            )

        return response

    def _should_skip_monitoring(self, request: Request) -> bool:
        """Check if monitoring should be skipped for this request."""
        skip_paths = {
            "/metrics",
            "/health",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/favicon.ico"
        }

        path = request.url.path.lower()
        return any(path.startswith(skip_path) for skip_path in skip_paths)

    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address from request."""
        # Check for forwarded headers first
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip

        # Fall back to client address
        if hasattr(request, "client") and request.client:
            return request.client.host

        return "unknown"

    async def _record_request_metrics(
        self,
        request: Request,
        response: Response | None,
        duration: float,
        status_code: int,
        exception_occurred: bool
    ):
        """Record request metrics."""
        try:
            method = request.method
            endpoint = self._normalize_endpoint(request.url.path)

            # Record HTTP request metrics
            await self.monitoring.record_http_request(
                method=method,
                endpoint=endpoint,
                status_code=status_code,
                duration=duration
            )

            # Log request details
            log_level = logger.warning if status_code >= 400 else logger.info
            log_level(
                f"{method} {endpoint} - {status_code} - {duration:.3f}s"
            )

            # Record additional context if available
            await self._record_additional_context(request, response, duration)

        except Exception as e:
            logger.error(f"Failed to record request metrics: {e}")

    def _normalize_endpoint(self, path: str) -> str:
        """Normalize endpoint path for metrics."""
        # Replace path parameters with placeholders
        import re

        # Replace UUIDs
        path = re.sub(
            r'/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}',
            '/{id}',
            path
        )

        # Replace numeric IDs
        path = re.sub(r'/\d+', '/{id}', path)

        # Replace file extensions
        path = re.sub(r'\.[a-zA-Z0-9]+$', '.{ext}', path)

        return path

    async def _record_additional_context(
        self,
        request: Request,
        response: Response | None,
        duration: float
    ):
        """Record additional context-specific metrics."""
        try:
            path = request.url.path

            # Anomaly detection endpoints
            if "/api/v1/detect" in path:
                await self._record_anomaly_detection_context(request, response, duration)

            # Model training endpoints
            elif "/api/v1/train" in path:
                await self._record_training_context(request, response, duration)

            # Dataset endpoints
            elif "/api/v1/datasets" in path:
                await self._record_dataset_context(request, response, duration)

        except Exception as e:
            logger.error(f"Failed to record additional context: {e}")

    async def _record_anomaly_detection_context(
        self,
        request: Request,
        response: Response | None,
        duration: float
    ):
        """Record anomaly detection specific metrics."""
        try:
            # Try to extract algorithm from request body or query params
            algorithm = "unknown"
            success = response and response.status_code < 400

            # Extract algorithm from request if possible
            if hasattr(request.state, 'algorithm'):
                algorithm = request.state.algorithm
            elif "algorithm" in request.query_params:
                algorithm = request.query_params["algorithm"]

            await self.monitoring.record_anomaly_detection(
                algorithm=algorithm,
                duration=duration,
                success=success
            )

        except Exception as e:
            logger.error(f"Failed to record anomaly detection context: {e}")

    async def _record_training_context(
        self,
        request: Request,
        response: Response | None,
        duration: float
    ):
        """Record model training specific metrics."""
        try:
            algorithm = "unknown"
            success = response and response.status_code < 400
            accuracy = None

            # Extract algorithm from request
            if hasattr(request.state, 'algorithm'):
                algorithm = request.state.algorithm
            elif "algorithm" in request.query_params:
                algorithm = request.query_params["algorithm"]

            # Extract accuracy from response if available
            if hasattr(request.state, 'model_accuracy'):
                accuracy = request.state.model_accuracy

            await self.monitoring.record_model_training(
                algorithm=algorithm,
                success=success,
                accuracy=accuracy
            )

        except Exception as e:
            logger.error(f"Failed to record training context: {e}")

    async def _record_dataset_context(
        self,
        request: Request,
        response: Response | None,
        duration: float
    ):
        """Record dataset operation specific metrics."""
        try:
            # Dataset operations might have specific metrics
            # This can be extended based on specific requirements
            pass

        except Exception as e:
            logger.error(f"Failed to record dataset context: {e}")

    async def get_active_requests_info(self) -> dict[str, Any]:
        """Get information about currently active requests."""
        current_time = time.time()
        active_info = {}

        for request_id, info in self.active_requests.items():
            duration = current_time - info["start_time"]
            active_info[request_id] = {
                **info,
                "duration": duration,
                "status": "active"
            }

        return {
            "total_active": len(self.active_requests),
            "requests": active_info
        }


class HealthCheckMiddleware(BaseHTTPMiddleware):
    """Middleware for enhanced health checks."""

    def __init__(self, app: ASGIApp, monitoring: ProductionMonitoringIntegration):
        super().__init__(app)
        self.monitoring = monitoring
        self.last_health_check = time.time()

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with health check enhancements."""
        # Handle health check requests
        if request.url.path == "/health":
            return await self._handle_health_check(request)

        # Handle metrics requests
        elif request.url.path == "/metrics":
            return await self._handle_metrics_request(request)

        # Handle monitoring status requests
        elif request.url.path == "/monitoring/status":
            return await self._handle_monitoring_status(request)

        # Process other requests normally
        return await call_next(request)

    async def _handle_health_check(self, request: Request) -> Response:
        """Handle health check requests with enhanced monitoring."""
        try:
            # Get detailed health status
            health_status = await self.monitoring.get_monitoring_status()

            # Determine response status code
            status_code = 200
            if health_status["status"] == "unhealthy":
                status_code = 503
            elif health_status["status"] == "degraded":
                status_code = 200  # Still serve traffic but warn

            # Simple health check for load balancers
            if request.query_params.get("simple") == "true":
                if status_code == 200:
                    return Response(content="OK", status_code=200)
                else:
                    return Response(content="UNHEALTHY", status_code=503)

            # Detailed health check
            self.last_health_check = time.time()
            return Response(
                content=health_status,
                status_code=status_code,
                media_type="application/json"
            )

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return Response(
                content={"status": "error", "message": str(e)},
                status_code=503,
                media_type="application/json"
            )

    async def _handle_metrics_request(self, request: Request) -> Response:
        """Handle Prometheus metrics requests."""
        # This is typically handled by prometheus_client
        # but we can add custom logic here if needed
        return await self.app(request.scope, request.receive, request._send)

    async def _handle_monitoring_status(self, request: Request) -> Response:
        """Handle monitoring system status requests."""
        try:
            status = await self.monitoring.get_monitoring_status()
            return Response(
                content=status,
                status_code=200,
                media_type="application/json"
            )
        except Exception as e:
            logger.error(f"Failed to get monitoring status: {e}")
            return Response(
                content={"error": str(e)},
                status_code=500,
                media_type="application/json"
            )


class AlertingMiddleware(BaseHTTPMiddleware):
    """Middleware for alerting integration."""

    def __init__(self, app: ASGIApp, monitoring: ProductionMonitoringIntegration):
        super().__init__(app)
        self.monitoring = monitoring
        self.error_counts: dict[str, int] = {}
        self.last_error_reset = time.time()

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with alerting integration."""
        try:
            response = await call_next(request)

            # Monitor error patterns
            await self._monitor_error_patterns(request, response)

            return response

        except Exception as e:
            # Record exception for alerting
            await self._record_exception(request, e)
            raise

    async def _monitor_error_patterns(self, request: Request, response: Response):
        """Monitor error patterns for alerting."""
        try:
            # Reset error counts every hour
            current_time = time.time()
            if current_time - self.last_error_reset > 3600:
                self.error_counts.clear()
                self.last_error_reset = current_time

            # Track error rates by endpoint
            if response.status_code >= 400:
                endpoint = self._normalize_endpoint(request.url.path)
                self.error_counts[endpoint] = self.error_counts.get(endpoint, 0) + 1

                # Trigger alert if error rate is high
                if self.error_counts[endpoint] > 10:  # More than 10 errors per hour
                    await self._trigger_error_rate_alert(endpoint, self.error_counts[endpoint])

        except Exception as e:
            logger.error(f"Error monitoring patterns: {e}")

    async def _record_exception(self, request: Request, exception: Exception):
        """Record exception for alerting."""
        try:
            endpoint = self._normalize_endpoint(request.url.path)

            # Record exception metrics
            logger.error(f"Exception in {endpoint}: {exception}")

            # Could trigger immediate alerts for critical exceptions
            if isinstance(exception, (ConnectionError, TimeoutError)):
                await self._trigger_critical_alert(endpoint, str(exception))

        except Exception as e:
            logger.error(f"Failed to record exception: {e}")

    async def _trigger_error_rate_alert(self, endpoint: str, error_count: int):
        """Trigger alert for high error rate."""
        try:
            # Record metric for alerting system
            if self.monitoring.alerting_service:
                await self.monitoring.alerting_service.record_metric(
                    f"endpoint_error_rate_{endpoint}", error_count
                )

        except Exception as e:
            logger.error(f"Failed to trigger error rate alert: {e}")

    async def _trigger_critical_alert(self, endpoint: str, error_message: str):
        """Trigger critical alert for severe errors."""
        try:
            # This could trigger immediate notifications
            logger.critical(f"Critical error in {endpoint}: {error_message}")

        except Exception as e:
            logger.error(f"Failed to trigger critical alert: {e}")

    def _normalize_endpoint(self, path: str) -> str:
        """Normalize endpoint path for alerting."""
        import re

        # Replace path parameters with placeholders
        path = re.sub(r'/[0-9a-f-]{36}', '/{uuid}', path)  # UUIDs
        path = re.sub(r'/\d+', '/{id}', path)  # Numeric IDs

        return path


def setup_monitoring_middleware(app, monitoring: ProductionMonitoringIntegration):
    """Set up all monitoring middleware for the FastAPI app."""
    # Add middleware in reverse order (last added = first executed)
    app.add_middleware(AlertingMiddleware, monitoring=monitoring)
    app.add_middleware(HealthCheckMiddleware, monitoring=monitoring)
    app.add_middleware(MonitoringMiddleware, monitoring=monitoring)

    logger.info("Monitoring middleware configured")
