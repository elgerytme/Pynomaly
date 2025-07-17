"""Real-time monitoring dashboard API endpoints.

This module provides API endpoints for the comprehensive monitoring dashboard,
including real-time measurements, error tracking, user analytics, and system health.
"""

from datetime import timedelta
from typing import Any
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, WebSocket
from pydantic import BaseModel

from ...application.services.error_tracking_service import ErrorTrackingService
from ...application.services.privacy_compliance_service import (
    ConsentType,
    PrivacyComplianceService,
)
from ...application.services.real_time_monitoring_service import (
    RealTimeMonitoringService,
)
from ...domain.models.monitoring import UserEventType, WebVitalType
from ...infrastructure.config.container import Container
from ...infrastructure.websocket.real_time_websocket import (
    RealTimeWebSocketManager,
    websocket_endpoint_handler,
)
from ..api.deps import get_current_user, get_optional_current_user

router = APIRouter()

# Initialize services (these would be dependency injected in production)
container = Container()


class DashboardMetricsResponse(BaseModel):
    """Response processor for dashboard measurements."""

    timestamp: str
    system_health: dict[str, Any]
    performance: dict[str, Any]
    users: dict[str, Any]
    api: dict[str, Any]
    errors: dict[str, Any]
    alerts: list[dict[str, Any]]


class UserEventRequest(BaseModel):
    """Request processor for tracking user events."""

    session_id: str
    event_type: UserEventType
    event_name: str
    properties: dict[str, Any] | None = None
    page_url: str | None = ""
    response_time_ms: int | None = None
    error_message: str | None = None


class WebVitalRequest(BaseModel):
    """Request processor for tracking Web Vitals."""

    session_id: str
    vital_type: WebVitalType
    value: float
    page_url: str | None = ""
    device_type: str | None = ""
    connection_type: str | None = ""


class SessionStartRequest(BaseModel):
    """Request processor for starting a user session."""

    session_id: str
    ip_address: str | None = ""
    user_agent: str | None = ""
    referrer: str | None = ""
    landing_page: str | None = ""


class ApiRequestMetrics(BaseModel):
    """Request processor for API request measurements."""

    endpoint: str
    method: str
    response_time_ms: int
    status_code: int
    error_message: str | None = None


# Dependency to get monitoring service
def get_monitoring_service() -> RealTimeMonitoringService:
    """Get real-time monitoring service instance."""
    # In production, this would be dependency injected
    if not hasattr(get_monitoring_service, "_instance"):
        get_monitoring_service._instance = RealTimeMonitoringService()
    return get_monitoring_service._instance


# Dependency to get error tracking service
def get_error_tracking_service() -> ErrorTrackingService:
    """Get error tracking service instance."""
    if not hasattr(get_error_tracking_service, "_instance"):
        get_error_tracking_service._instance = ErrorTrackingService()
    return get_error_tracking_service._instance


# Dependency to get privacy compliance service
def get_privacy_service() -> PrivacyComplianceService:
    """Get privacy compliance service instance."""
    if not hasattr(get_privacy_service, "_instance"):
        get_privacy_service._instance = PrivacyComplianceService()
    return get_privacy_service._instance


# Dependency to get WebSocket manager
def get_websocket_manager() -> RealTimeWebSocketManager:
    """Get WebSocket manager instance."""
    if not hasattr(get_websocket_manager, "_instance"):
        monitoring_service = get_monitoring_service()
        get_websocket_manager._instance = RealTimeWebSocketManager(monitoring_service)
    return get_websocket_manager._instance


@router.get(
    "/dashboard/measurements",
    response_processor=DashboardMetricsResponse,
    summary="Get real-time dashboard measurements",
    description="Get comprehensive real-time monitoring dashboard data including system health, performance, and analytics.",
)
async def get_dashboard_measurements(
    current_user: dict | None = Depends(get_optional_current_user),
    monitoring_service: RealTimeMonitoringService = Depends(get_monitoring_service),
) -> DashboardMetricsResponse:
    """Get real-time dashboard measurements."""
    try:
        dashboard_data = await monitoring_service.get_real_time_dashboard_data()
        return DashboardMetricsResponse(**dashboard_data)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get dashboard measurements: {str(e)}"
        )


@router.get(
    "/dashboard/performance",
    summary="Get performance measurements",
    description="Get detailed performance measurements including API response times, throughput, and system resources.",
)
async def get_performance_measurements(
    time_window_hours: int = Query(default=1, ge=1, le=168),  # 1 hour to 1 week
    current_user: dict | None = Depends(get_optional_current_user),
    monitoring_service: RealTimeMonitoringService = Depends(get_monitoring_service),
) -> dict[str, Any]:
    """Get performance measurements for the specified time window."""
    try:
        # Create a performance snapshot
        snapshot = await monitoring_service.create_performance_snapshot()
        return snapshot.get_performance_summary()
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get performance measurements: {str(e)}"
        )


@router.get(
    "/dashboard/errors",
    summary="Get error analytics",
    description="Get comprehensive error analytics including error trends, categorization, and top errors.",
)
async def get_error_analytics(
    time_window_hours: int = Query(default=1, ge=1, le=168),
    current_user: dict | None = Depends(get_optional_current_user),
    monitoring_service: RealTimeMonitoringService = Depends(get_monitoring_service),
) -> dict[str, Any]:
    """Get error analytics for the specified time window."""
    try:
        time_window = timedelta(hours=time_window_hours)
        error_analytics = await monitoring_service.get_error_analytics(time_window)
        return error_analytics
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get error analytics: {str(e)}"
        )


@router.get(
    "/dashboard/users",
    summary="Get user analytics",
    description="Get user analytics including active users, session data, and feature usage.",
)
async def get_user_analytics(
    time_window_hours: int = Query(default=1, ge=1, le=168),
    current_user: dict | None = Depends(get_optional_current_user),
    monitoring_service: RealTimeMonitoringService = Depends(get_monitoring_service),
) -> dict[str, Any]:
    """Get user analytics for the specified time window."""
    try:
        time_window = timedelta(hours=time_window_hours)
        user_analytics = await monitoring_service.get_user_analytics(time_window)
        return user_analytics
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get user analytics: {str(e)}"
        )


@router.get(
    "/dashboard/alerts",
    summary="Get active alerts",
    description="Get all active monitoring alerts with their details and status.",
)
async def get_active_alerts(
    current_user: dict | None = Depends(get_optional_current_user),
    error_service: ErrorTrackingService = Depends(get_error_tracking_service),
) -> dict[str, Any]:
    """Get active alerts."""
    try:
        return {
            "alerts": [
                alert.to_dict() for alert in error_service.active_alerts.values()
            ],
            "total_alerts": len(error_service.active_alerts),
            "critical_alerts": len(
                [
                    alert
                    for alert in error_service.active_alerts.values()
                    if alert.severity.value == "critical"
                ]
            ),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get alerts: {str(e)}")


@router.post(
    "/events/track",
    summary="Track user event",
    description="Track a user interaction event for analytics and monitoring.",
)
async def track_user_event(
    event: UserEventRequest,
    current_user: dict | None = Depends(get_optional_current_user),
    monitoring_service: RealTimeMonitoringService = Depends(get_monitoring_service),
    privacy_service: PrivacyComplianceService = Depends(get_privacy_service),
) -> dict[str, Any]:
    """Track a user event."""
    try:
        user_id = (
            UUID(current_user["user_id"])
            if current_user and "user_id" in current_user
            else None
        )

        tracked_event = await monitoring_service.track_user_event(
            session_id=event.session_id,
            event_type=event.event_type,
            event_name=event.event_name,
            user_id=user_id,
            properties=event.properties,
            page_url=event.page_url,
            response_time_ms=event.response_time_ms,
            error_message=event.error_message,
        )

        # Apply privacy compliance anonymization
        anonymized_event = await privacy_service.anonymize_user_event(tracked_event)

        return {"event_id": str(anonymized_event.event_id), "status": "tracked"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to track event: {str(e)}")


@router.post(
    "/events/web-vitals",
    summary="Track Web Vitals",
    description="Track Core Web Vitals measurements for performance monitoring.",
)
async def track_web_vital(
    vital: WebVitalRequest,
    current_user: dict | None = Depends(get_optional_current_user),
    monitoring_service: RealTimeMonitoringService = Depends(get_monitoring_service),
    privacy_service: PrivacyComplianceService = Depends(get_privacy_service),
) -> dict[str, Any]:
    """Track a Web Vital metric."""
    try:
        user_id = (
            UUID(current_user["user_id"])
            if current_user and "user_id" in current_user
            else None
        )

        tracked_vital = await monitoring_service.track_web_vital(
            session_id=vital.session_id,
            vital_type=vital.vital_type,
            value=vital.value,
            page_url=vital.page_url,
            user_id=user_id,
            device_type=vital.device_type,
            connection_type=vital.connection_type,
        )

        # Apply privacy compliance anonymization
        anonymized_vital = await privacy_service.anonymize_web_vital(tracked_vital)

        return {
            "metric_id": str(anonymized_vital.metric_id),
            "rating": anonymized_vital.rating,
            "status": "tracked",
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to track Web Vital: {str(e)}"
        )


@router.post(
    "/sessions/start",
    summary="Start user session",
    description="Start a new user session for analytics tracking.",
)
async def start_user_session(
    session_request: SessionStartRequest,
    current_user: dict | None = Depends(get_optional_current_user),
    monitoring_service: RealTimeMonitoringService = Depends(get_monitoring_service),
    privacy_service: PrivacyComplianceService = Depends(get_privacy_service),
) -> dict[str, Any]:
    """Start a new user session."""
    try:
        user_id = (
            UUID(current_user["user_id"])
            if current_user and "user_id" in current_user
            else None
        )

        session = await monitoring_service.start_user_session(
            session_id=session_request.session_id,
            user_id=user_id,
            ip_address=session_request.ip_address,
            user_agent=session_request.user_agent,
            referrer=session_request.referrer,
            landing_page=session_request.landing_page,
        )

        # Apply privacy compliance anonymization
        anonymized_session = await privacy_service.anonymize_user_session(session)

        return {"session_id": anonymized_session.session_id, "status": "started"}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to start session: {str(e)}"
        )


@router.post(
    "/sessions/{session_id}/end",
    summary="End user session",
    description="End a user session and finalize analytics data.",
)
async def end_user_session(
    session_id: str,
    current_user: dict | None = Depends(get_optional_current_user),
    monitoring_service: RealTimeMonitoringService = Depends(get_monitoring_service),
) -> dict[str, Any]:
    """End a user session."""
    try:
        session = await monitoring_service.end_user_session(session_id)

        if session:
            return {
                "session_id": session.session_id,
                "duration_seconds": session.duration_seconds,
                "status": "ended",
            }
        else:
            raise HTTPException(status_code=404, detail="Session not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to end session: {str(e)}")


@router.post(
    "/api/measurements",
    summary="Record API request measurements",
    description="Record API request performance measurements for monitoring.",
)
async def record_api_measurements(
    measurements: ApiRequestMetrics,
    current_user: dict | None = Depends(get_optional_current_user),
    monitoring_service: RealTimeMonitoringService = Depends(get_monitoring_service),
) -> dict[str, Any]:
    """Record API request measurements."""
    try:
        user_id = (
            UUID(current_user["user_id"])
            if current_user and "user_id" in current_user
            else None
        )

        await monitoring_service.record_api_request(
            endpoint=measurements.endpoint,
            method=measurements.method,
            response_time_ms=measurements.response_time_ms,
            status_code=measurements.status_code,
            user_id=user_id,
            error_message=measurements.error_message,
        )

        return {"status": "recorded"}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to record API measurements: {str(e)}"
        )


@router.get(
    "/errors/groups",
    summary="Get error groups",
    description="Get grouped errors for error tracking and management.",
)
async def get_error_groups(
    limit: int = Query(default=50, ge=1, le=500),
    current_user: dict | None = Depends(get_optional_current_user),
    error_service: ErrorTrackingService = Depends(get_error_tracking_service),
) -> dict[str, Any]:
    """Get error groups."""
    try:
        # Get recent error groups sorted by count
        error_groups = sorted(
            error_service.error_groups.values(), key=lambda g: g.count, reverse=True
        )[:limit]

        return {
            "error_groups": [group.to_dict() for group in error_groups],
            "total_groups": len(error_service.error_groups),
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get error groups: {str(e)}"
        )


@router.post(
    "/errors/groups/{fingerprint}/resolve",
    summary="Resolve error group",
    description="Mark an error group as resolved.",
)
async def resolve_error_group(
    fingerprint: str,
    resolution_note: str = "",
    current_user: dict = Depends(get_current_user),
    error_service: ErrorTrackingService = Depends(get_error_tracking_service),
) -> dict[str, Any]:
    """Resolve an error group."""
    try:
        user_id = UUID(current_user["user_id"])
        success = error_service.resolve_error_group(
            fingerprint, user_id, resolution_note
        )

        if success:
            return {"status": "resolved", "fingerprint": fingerprint}
        else:
            raise HTTPException(status_code=404, detail="Error group not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to resolve error group: {str(e)}"
        )


@router.get(
    "/websocket/stats",
    summary="Get WebSocket connection statistics",
    description="Get statistics about active WebSocket connections.",
)
async def get_websocket_stats(
    current_user: dict | None = Depends(get_optional_current_user),
    websocket_manager: RealTimeWebSocketManager = Depends(get_websocket_manager),
) -> dict[str, Any]:
    """Get WebSocket connection statistics."""
    try:
        return await websocket_manager.get_connection_stats()
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get WebSocket stats: {str(e)}"
        )


@router.websocket("/ws")
async def websocket_dashboard_endpoint(
    websocket: WebSocket,
    websocket_manager: RealTimeWebSocketManager = Depends(get_websocket_manager),
) -> None:
    """WebSocket endpoint for real-time dashboard updates."""
    try:
        # In production, you would extract user info from token/auth
        await websocket_endpoint_handler(
            websocket=websocket,
            websocket_manager=websocket_manager,
            user_id=None,  # Would be extracted from auth
            session_id=None,  # Would be extracted from query params or auth
        )
    except Exception:
        # Log error but don't raise (WebSocket is already closed)
        pass


# Initialize services when module is imported
async def initialize_monitoring_services():
    """Initialize monitoring services."""
    monitoring_service = get_monitoring_service()
    websocket_manager = get_websocket_manager()

    await monitoring_service.start_monitoring()
    await websocket_manager.start()


# Privacy compliance endpoints


class ConsentUpdateRequest(BaseModel):
    """Request processor for updating user consent."""

    consent_type: ConsentType
    granted: bool
    consent_method: str = "web_form"


class PrivacyDashboardResponse(BaseModel):
    """Response processor for privacy dashboard data."""

    user_id: str
    session_id: str
    consents: dict[str, bool]
    last_consent_update: str
    gdpr_compliant: bool
    data_categories_processed: list[str]
    retention_policies: dict[str, str]
    anonymization_status: dict[str, bool]


@router.post(
    "/privacy/consent/update",
    summary="Update user consent preferences",
    description="Update user consent preferences for data processing categories.",
)
async def update_user_consent(
    consent_request: ConsentUpdateRequest,
    session_id: str = Query(..., description="User session ID"),
    current_user: dict | None = Depends(get_optional_current_user),
    privacy_service: PrivacyComplianceService = Depends(get_privacy_service),
) -> dict[str, Any]:
    """Update user consent preferences."""
    try:
        success = await privacy_service.update_user_consent(
            session_id=session_id,
            consent_type=consent_request.consent_type,
            granted=consent_request.granted,
            consent_method=consent_request.consent_method,
        )

        if success:
            return {
                "status": "updated",
                "consent_type": consent_request.consent_type.value,
                "granted": consent_request.granted,
            }
        else:
            raise HTTPException(status_code=404, detail="User session not found")
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to update consent: {str(e)}"
        )


@router.get(
    "/privacy/dashboard/{session_id}",
    response_processor=PrivacyDashboardResponse,
    summary="Get privacy dashboard data",
    description="Get user's privacy dashboard data including consent status and data processing information.",
)
async def get_privacy_dashboard(
    session_id: str,
    current_user: dict | None = Depends(get_optional_current_user),
    privacy_service: PrivacyComplianceService = Depends(get_privacy_service),
) -> PrivacyDashboardResponse:
    """Get privacy dashboard data for user."""
    try:
        dashboard_data = await privacy_service.get_privacy_dashboard_data(session_id)

        if "error" in dashboard_data:
            raise HTTPException(status_code=404, detail=dashboard_data["error"])

        return PrivacyDashboardResponse(**dashboard_data)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get privacy dashboard: {str(e)}"
        )


@router.post(
    "/privacy/profile/create",
    summary="Create privacy-compliant user profile",
    description="Create a new privacy-compliant user profile with initial consent preferences.",
)
async def create_privacy_profile(
    session_id: str = Query(..., description="User session ID"),
    ip_address: str = Query(default="", description="User IP address"),
    user_agent: str = Query(default="", description="User agent string"),
    initial_consents: dict[str, bool] = Query(
        default={}, description="Initial consent preferences"
    ),
    current_user: dict | None = Depends(get_optional_current_user),
    privacy_service: PrivacyComplianceService = Depends(get_privacy_service),
) -> dict[str, Any]:
    """Create privacy-compliant user profile."""
    try:
        user_id = (
            UUID(current_user["user_id"])
            if current_user and "user_id" in current_user
            else uuid4()
        )

        # Convert string keys to ConsentType enums
        consent_dict = {}
        for consent_str, granted in initial_consents.items():
            try:
                consent_type = ConsentType(consent_str)
                consent_dict[consent_type] = granted
            except ValueError:
                continue  # Skip invalid consent types

        profile = await privacy_service.create_user_profile(
            user_id=user_id,
            session_id=session_id,
            ip_address=ip_address,
            user_agent=user_agent,
            initial_consents=consent_dict if consent_dict else None,
        )

        return {
            "status": "created",
            "user_id": str(profile.user_id),
            "session_id": profile.session_id,
            "consents": {ct.value: granted for ct, granted in profile.consents.items()},
            "gdpr_compliant": profile.is_gdpr_compliant(),
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to create privacy profile: {str(e)}"
        )


@router.post(
    "/privacy/data-request",
    summary="Submit data subject rights request",
    description="Submit a data subject rights request (GDPR Article 15-22).",
)
async def submit_data_request(
    request_type: str = Query(
        ..., description="Type of request (access, erasure, rectification, portability)"
    ),
    email: str = Query(..., description="User email for verification"),
    request_details: dict[str, Any] = Query(
        default={}, description="Additional request details"
    ),
    current_user: dict = Depends(get_current_user),
    privacy_service: PrivacyComplianceService = Depends(get_privacy_service),
) -> dict[str, Any]:
    """Submit data subject rights request."""
    try:
        user_id = UUID(current_user["user_id"])

        result = await privacy_service.handle_data_subject_request(
            request_type=request_type,
            user_id=user_id,
            email=email,
            request_details=request_details,
        )

        return result
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to submit data request: {str(e)}"
        )


# Cleanup function for graceful shutdown
async def cleanup_monitoring_services():
    """Cleanup monitoring services."""
    try:
        monitoring_service = get_monitoring_service()
        websocket_manager = get_websocket_manager()

        await monitoring_service.stop_monitoring()
        await websocket_manager.stop()
    except Exception:
        pass  # Ignore cleanup errors
