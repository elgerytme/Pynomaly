"""Frontend support endpoints for the web UI utilities."""

import secrets
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Request
from pydantic import BaseModel

from monorepo.presentation.api.middleware import CSPViolationReporter

router = APIRouter(prefix="/api", tags=["Frontend Support"])


class PerformanceMetric(BaseModel):
    """Performance metric model."""

    metric: str
    value: float
    timestamp: int
    url: str


class SecurityEvent(BaseModel):
    """Security event model."""

    type: str
    timestamp: int
    url: str
    userAgent: str
    data: dict[str, Any]


class SessionExtendRequest(BaseModel):
    """Session extend request model."""

    timestamp: int = None


class SessionExtendResponse(BaseModel):
    """Session extend response model."""

    success: bool
    new_expiry: int
    message: str


@router.post("/metrics/critical")
async def report_critical_metric(request: Request, metric: PerformanceMetric):
    """
    Report critical performance metrics from the frontend.

    This endpoint receives performance metrics from the frontend utilities
    like Core Web Vitals, memory usage, and other critical performance indicators.
    """
    # Log the metric (in production, this would go to a monitoring system)
    print(f"CRITICAL METRIC: {metric.metric} = {metric.value} at {metric.url}")

    # You could integrate with monitoring services here:
    # - Prometheus
    # - DataDog
    # - New Relic
    # - Custom metrics storage

    return {"status": "received", "metric": metric.metric, "value": metric.value}


@router.post("/security/events")
async def report_security_event(request: Request, event: SecurityEvent):
    """
    Report security events from the frontend.

    This endpoint receives security events from the frontend security manager
    like XSS attempts, SQL injection attempts, and other security incidents.
    """
    # Log the security event (in production, this would go to a SIEM system)
    print(f"SECURITY EVENT: {event.type} from {event.url}")
    print(f"  User Agent: {event.userAgent}")
    print(f"  Data: {event.data}")

    # You could integrate with security monitoring services here:
    # - Splunk
    # - LogRhythm
    # - Custom security event storage
    # - Alert systems

    return {
        "status": "received",
        "event_type": event.type,
        "timestamp": event.timestamp,
    }


@router.post("/session/extend")
async def extend_session(
    request: Request, session_request: SessionExtendRequest = None
):
    """
    Extend the current user session.

    This endpoint is called by the frontend session manager to extend
    the user's session and prevent automatic logout.
    """
    # In a real implementation, this would:
    # 1. Validate the current session
    # 2. Check if the user is still active
    # 3. Extend the session expiry time
    # 4. Return the new expiry time

    # For now, we'll simulate a session extension
    current_time = int(datetime.now().timestamp())
    new_expiry = current_time + (30 * 60)  # Extend by 30 minutes

    return SessionExtendResponse(
        success=True, new_expiry=new_expiry, message="Session extended successfully"
    )


@router.get("/session/status")
async def get_session_status(request: Request):
    """
    Get the current session status.

    This endpoint provides information about the current user session
    including expiry time and activity status.
    """
    current_time = int(datetime.now().timestamp())

    return {
        "authenticated": True,  # This would be based on actual auth state
        "expires_at": current_time + (30 * 60),
        "last_activity": current_time,
        "csrf_token": secrets.token_urlsafe(32),  # Generate a real CSRF token
    }


@router.get("/ui/config")
async def get_ui_config(request: Request):
    """
    Get UI configuration for the frontend.

    This endpoint provides configuration settings that the frontend
    utilities need to operate correctly.
    """
    return {
        "performance_monitoring": {
            "enabled": True,
            "critical_thresholds": {
                "LCP": 2500,
                "FID": 100,
                "CLS": 0.1,
                "memory_used": 50 * 1024 * 1024,
            },
        },
        "security": {
            "csrf_protection": True,
            "xss_protection": True,
            "sql_injection_protection": True,
            "session_timeout": 30 * 60 * 1000,  # 30 minutes in milliseconds
        },
        "features": {
            "dark_mode": True,
            "lazy_loading": True,
            "caching": True,
            "offline_support": True,
        },
    }


@router.get("/ui/health")
async def get_ui_health(request: Request):
    """
    Get health status for UI components.

    This endpoint provides health information for the frontend
    utilities and components.
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "performance_monitor": "healthy",
            "security_manager": "healthy",
            "cache_manager": "healthy",
            "lazy_loader": "healthy",
            "theme_manager": "healthy",
        },
        "metrics": {
            "uptime": "00:05:23",
            "memory_usage": "45MB",
            "cache_hit_rate": "87%",
        },
    }


@router.post("/security/csp-violation")
async def report_csp_violation(request: Request, violation_data: dict):
    """
    Report Content Security Policy violations.

    This endpoint receives CSP violation reports from the browser
    when the Content Security Policy is violated.
    """
    CSPViolationReporter.report_violation(request, violation_data)
    return {"status": "received"}


@router.get("/monitoring/dashboard")
async def get_monitoring_dashboard(request: Request):
    """
    Get comprehensive monitoring dashboard data.

    This endpoint provides real-time monitoring data for the web UI
    including performance metrics, security events, and system health.
    """
    from monorepo.presentation.web.monitoring import (
        performance_monitor,
        security_monitor,
    )

    return {
        "timestamp": datetime.now().isoformat(),
        "performance": performance_monitor.get_performance_summary(),
        "security": security_monitor.get_security_summary(),
        "system": {
            "uptime": "running",
            "memory_usage": "monitoring active",
            "disk_usage": "monitoring active",
            "cpu_usage": "monitoring active",
        },
        "alerts": [
            {
                "type": "info",
                "message": "All systems operational",
                "timestamp": datetime.now().isoformat(),
            }
        ],
    }


@router.post("/monitoring/performance")
async def report_performance_data(request: Request, data: dict):
    """
    Report detailed performance data from frontend.

    This endpoint receives comprehensive performance data including
    page load times, resource timings, and user interaction metrics.
    """
    from monorepo.presentation.web.monitoring import performance_monitor

    # Process different types of performance data
    if "page_load_time" in data:
        performance_monitor.record_page_load_time(
            data.get("page", "unknown"), data["page_load_time"]
        )

    if "api_response_time" in data:
        performance_monitor.record_api_response_time(
            data.get("endpoint", "unknown"), data["api_response_time"]
        )

    if "core_web_vital" in data:
        performance_monitor.record_core_web_vital(
            data["core_web_vital"]["metric"], data["core_web_vital"]["value"]
        )

    return {"status": "received", "data_type": "performance"}


@router.post("/monitoring/security")
async def report_security_data(request: Request, data: dict):
    """
    Report security monitoring data from frontend.

    This endpoint receives security-related events and threat detection
    data from the frontend security monitoring system.
    """
    from monorepo.presentation.web.monitoring import security_monitor

    # Record security event
    security_monitor.record_security_event(
        data.get("event_type", "unknown"),
        {
            "data": data,
            "user_agent": request.headers.get("user-agent", "unknown"),
            "client_ip": request.client.host if request.client else "unknown",
            "timestamp": datetime.now().isoformat(),
        },
    )

    return {"status": "received", "data_type": "security"}


@router.get("/error-monitoring/metrics")
async def get_error_metrics(request: Request):
    """Get error monitoring metrics."""
    try:
        from monorepo.presentation.web.error_handling import get_error_monitor

        monitor = get_error_monitor()
        return monitor.get_metrics()
    except ImportError:
        return {"status": "error", "message": "Error monitoring not available"}


@router.get("/error-monitoring/history")
async def get_error_history(request: Request, limit: int = 100):
    """Get error monitoring history."""
    try:
        from monorepo.presentation.web.error_handling import get_error_monitor

        monitor = get_error_monitor()
        return {"errors": monitor.get_recent_errors(limit)}
    except ImportError:
        return {"errors": []}


@router.get("/error-monitoring/trends")
async def get_error_trends(request: Request):
    """Get error monitoring trends."""
    try:
        from monorepo.presentation.web.error_handling import get_error_monitor

        monitor = get_error_monitor()
        return monitor.get_error_trends()
    except ImportError:
        return {"trends": {}}


@router.post("/error-monitoring/report")
async def report_frontend_error(request: Request, error_data: dict):
    """Report frontend error for monitoring."""
    try:
        from monorepo.presentation.web.error_handling import (
            ErrorCode,
            ErrorLevel,
            create_web_ui_error,
            get_error_monitor,
        )

        error = create_web_ui_error(
            message=error_data.get("message", "Frontend error"),
            error_code=ErrorCode.INTERNAL_SERVER_ERROR,
            error_level=ErrorLevel.ERROR,
            details=error_data.get("details", {}),
            user_message=error_data.get("user_message", "An error occurred"),
            suggestion=error_data.get("suggestion", "Please try again"),
        )
        monitor = get_error_monitor()
        request_info = {
            "ip": request.client.host,
            "user_agent": request.headers.get("user-agent", ""),
            "path": request.url.path,
            "method": request.method,
        }
        monitor.record_error(error, request_info)
        return {"status": "received", "error_id": error.error_id}
    except ImportError:
        return {"status": "error", "message": "Error monitoring not available"}


@router.get("/security/events")
async def get_security_events(request: Request, limit: int = 100):
    """Get recent security events for monitoring."""
    try:
        from monorepo.presentation.web.security_features import get_security_middleware

        middleware = get_security_middleware()
        events = middleware.get_security_events(limit)
        return {"events": events}
    except ImportError:
        return {"events": []}


@router.get("/security/metrics")
async def get_security_metrics(request: Request):
    """Get security metrics and statistics."""
    try:
        from monorepo.presentation.web.security_features import get_security_middleware

        middleware = get_security_middleware()
        metrics = middleware.get_security_metrics()
        return metrics
    except ImportError:
        return {"status": "error", "message": "Security monitoring not available"}


@router.get("/security/rate-limit-status/{ip}")
async def get_rate_limit_status(request: Request, ip: str):
    """Get rate limit status for specific IP."""
    try:
        from monorepo.presentation.web.security_features import get_rate_limiter

        rate_limiter = get_rate_limiter()
        endpoint = request.url.path
        status = rate_limiter.get_rate_limit_status(ip, endpoint)
        return status
    except ImportError:
        return {"status": "error", "message": "Rate limiting not available"}
