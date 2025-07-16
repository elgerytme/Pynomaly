"""
Security Dashboard API Endpoints

Provides API endpoints for:
- Real-time security monitoring
- Session management
- Threat analysis
- Security event tracking
"""

from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from pynomaly_detection.infrastructure.auth.enhanced_dependencies import get_current_user
from pynomaly_detection.infrastructure.security.rbac_middleware import require_permissions
from pynomaly_detection.infrastructure.security.session_manager import get_session_manager


# Response Models
class SecurityOverviewResponse(BaseModel):
    """Security overview response model."""

    threat_level: str
    active_sessions: int
    blocked_threats: int
    security_score: int
    session_growth_percentage: float
    last_updated: datetime


class ThreatStatistics(BaseModel):
    """Threat statistics model."""

    type: str
    count: int
    percentage: float
    color: str
    severity: str


class SecurityEvent(BaseModel):
    """Security event model."""

    id: str
    timestamp: datetime
    event_type: str
    source_ip: str
    severity: str
    details: dict[str, Any]
    user_id: str | None = None
    session_id: str | None = None


class ActiveSessionInfo(BaseModel):
    """Active session information model."""

    session_id: str
    user_email: str | None
    user_role: str | None
    ip_address: str
    created_at: datetime
    last_accessed: datetime
    login_method: str
    security_level: str
    is_current: bool


class ThreatTimelineData(BaseModel):
    """Threat timeline data model."""

    timestamp: datetime
    threat_count: int
    blocked_count: int
    severity_distribution: dict[str, int]


# Create router
router = APIRouter(prefix="/api/security", tags=["security"])


@router.get("/overview", response_model=SecurityOverviewResponse)
async def get_security_overview(
    current_user=Depends(get_current_user),
    _permission=Depends(require_permissions(["view_security_dashboard"])),
) -> SecurityOverviewResponse:
    """Get security overview metrics."""

    session_manager = get_session_manager()

    # Get active session count (mock implementation)
    # In production, this would query Redis
    active_sessions = 23  # Mock data

    # Calculate threat level based on recent activity
    threat_level = _calculate_threat_level()

    # Get blocked threats count
    blocked_threats = _get_blocked_threats_count()

    # Calculate security score
    security_score = _calculate_security_score()

    # Calculate session growth
    session_growth = _calculate_session_growth()

    return SecurityOverviewResponse(
        threat_level=threat_level,
        active_sessions=active_sessions,
        blocked_threats=blocked_threats,
        security_score=security_score,
        session_growth_percentage=session_growth,
        last_updated=datetime.now(UTC),
    )


@router.get("/threat-statistics", response_model=list[ThreatStatistics])
async def get_threat_statistics(
    timeframe: str = "24h",
    current_user=Depends(get_current_user),
    _permission=Depends(require_permissions(["view_security_dashboard"])),
) -> list[ThreatStatistics]:
    """Get threat type distribution statistics."""

    # Mock data - in production, this would query WAF logs
    threat_data = [
        {
            "type": "SQL Injection",
            "count": 45,
            "percentage": 29.8,
            "color": "#ef4444",
            "severity": "critical",
        },
        {
            "type": "XSS",
            "count": 32,
            "percentage": 21.2,
            "color": "#f97316",
            "severity": "high",
        },
        {
            "type": "CSRF",
            "count": 28,
            "percentage": 18.5,
            "color": "#eab308",
            "severity": "medium",
        },
        {
            "type": "Brute Force",
            "count": 21,
            "percentage": 13.9,
            "color": "#84cc16",
            "severity": "medium",
        },
        {
            "type": "Path Traversal",
            "count": 15,
            "percentage": 9.9,
            "color": "#06b6d4",
            "severity": "high",
        },
        {
            "type": "Code Injection",
            "count": 10,
            "percentage": 6.6,
            "color": "#8b5cf6",
            "severity": "critical",
        },
    ]

    return [ThreatStatistics(**threat) for threat in threat_data]


@router.get("/events", response_model=list[SecurityEvent])
async def get_security_events(
    limit: int = 50,
    severity: str | None = None,
    event_type: str | None = None,
    current_user=Depends(get_current_user),
    _permission=Depends(require_permissions(["view_security_events"])),
) -> list[SecurityEvent]:
    """Get recent security events."""

    # Mock data - in production, this would query security event logs
    now = datetime.now(UTC)
    events = [
        SecurityEvent(
            id="evt_001",
            timestamp=now - timedelta(minutes=5),
            event_type="Failed Login Attempt",
            source_ip="192.168.1.100",
            severity="warning",
            details={"attempts": 3, "user": "admin", "user_agent": "Mozilla/5.0..."},
            user_id="user_123",
        ),
        SecurityEvent(
            id="evt_002",
            timestamp=now - timedelta(minutes=10),
            event_type="SQL Injection Blocked",
            source_ip="10.0.0.45",
            severity="critical",
            details={
                "payload": "UNION SELECT * FROM users",
                "endpoint": "/api/search",
                "waf_rule": "SQLi_001",
            },
        ),
        SecurityEvent(
            id="evt_003",
            timestamp=now - timedelta(minutes=15),
            event_type="Session Created",
            source_ip="172.16.0.22",
            severity="info",
            details={"login_method": "jwt", "user_agent": "Chrome/120.0.0.0"},
            user_id="user_456",
            session_id="sess_789",
        ),
        SecurityEvent(
            id="evt_004",
            timestamp=now - timedelta(minutes=20),
            event_type="XSS Attempt Blocked",
            source_ip="203.0.113.45",
            severity="high",
            details={
                "payload": "<script>alert('xss')</script>",
                "endpoint": "/api/comments",
                "waf_rule": "XSS_002",
            },
        ),
        SecurityEvent(
            id="evt_005",
            timestamp=now - timedelta(minutes=25),
            event_type="Rate Limit Exceeded",
            source_ip="198.51.100.23",
            severity="warning",
            details={"endpoint": "/api/login", "requests_per_minute": 120, "limit": 10},
        ),
    ]

    # Apply filters
    if severity:
        events = [e for e in events if e.severity == severity]

    if event_type:
        events = [e for e in events if e.event_type == event_type]

    return events[:limit]


@router.get("/sessions", response_model=list[ActiveSessionInfo])
async def get_active_sessions(
    current_user=Depends(get_current_user),
    _permission=Depends(require_permissions(["manage_sessions"])),
) -> list[ActiveSessionInfo]:
    """Get all active sessions."""

    session_manager = get_session_manager()

    # Mock data - in production, this would query all active sessions
    now = datetime.now(UTC)
    sessions = [
        ActiveSessionInfo(
            session_id="sess_001",
            user_email="john.doe@company.com",
            user_role="Data Scientist",
            ip_address="192.168.1.100",
            created_at=now - timedelta(hours=2),
            last_accessed=now - timedelta(minutes=5),
            login_method="jwt",
            security_level="standard",
            is_current=False,
        ),
        ActiveSessionInfo(
            session_id="sess_002",
            user_email="jane.smith@company.com",
            user_role="Admin",
            ip_address="10.0.0.45",
            created_at=now - timedelta(hours=1),
            last_accessed=now - timedelta(minutes=2),
            login_method="jwt",
            security_level="elevated",
            is_current=False,
        ),
        ActiveSessionInfo(
            session_id="sess_003",
            user_email="mike.wilson@company.com",
            user_role="Analyst",
            ip_address="172.16.0.22",
            created_at=now - timedelta(minutes=30),
            last_accessed=now - timedelta(seconds=30),
            login_method="oauth",
            security_level="standard",
            is_current=True,
        ),
    ]

    return sessions


@router.delete("/sessions/{session_id}")
async def terminate_session(
    session_id: str,
    current_user=Depends(get_current_user),
    _permission=Depends(require_permissions(["manage_sessions"])),
) -> dict[str, Any]:
    """Terminate a specific session."""

    session_manager = get_session_manager()

    success = await session_manager.terminate_session(
        session_id, reason="admin_termination"
    )

    if not success:
        raise HTTPException(
            status_code=404, detail="Session not found or already terminated"
        )

    return {
        "success": True,
        "message": f"Session {session_id} terminated successfully",
        "timestamp": datetime.now(UTC).isoformat(),
    }


@router.get("/sessions/user/{user_id}", response_model=list[ActiveSessionInfo])
async def get_user_sessions(
    user_id: str,
    current_user=Depends(get_current_user),
    _permission=Depends(require_permissions(["view_user_sessions"])),
) -> list[ActiveSessionInfo]:
    """Get all sessions for a specific user."""

    session_manager = get_session_manager()

    session_details = await session_manager.get_session_details(user_id)

    return [
        ActiveSessionInfo(
            session_id=session["session_id"],
            user_email=None,  # Privacy protection
            user_role=None,
            ip_address=session["ip_address"],
            created_at=datetime.fromisoformat(session["created_at"]),
            last_accessed=datetime.fromisoformat(session["last_accessed"]),
            login_method=session["login_method"],
            security_level=session["security_level"],
            is_current=session["is_current"],
        )
        for session in session_details
    ]


@router.delete("/sessions/user/{user_id}")
async def terminate_user_sessions(
    user_id: str,
    except_current: bool = True,
    current_user=Depends(get_current_user),
    _permission=Depends(require_permissions(["manage_sessions"])),
) -> dict[str, Any]:
    """Terminate all sessions for a user."""

    session_manager = get_session_manager()

    # Get current session ID if protecting current session
    current_session_id = None
    if except_current and hasattr(current_user, "session_id"):
        current_session_id = current_user.session_id

    terminated_count = await session_manager.terminate_user_sessions(
        user_id, except_session_id=current_session_id
    )

    return {
        "success": True,
        "message": f"Terminated {terminated_count} sessions for user {user_id}",
        "terminated_count": terminated_count,
        "timestamp": datetime.now(UTC).isoformat(),
    }


@router.get("/threat-timeline", response_model=list[ThreatTimelineData])
async def get_threat_timeline(
    timeframe: str = "24h",
    current_user=Depends(get_current_user),
    _permission=Depends(require_permissions(["view_security_dashboard"])),
) -> list[ThreatTimelineData]:
    """Get threat activity timeline data."""

    # Mock data - in production, this would query WAF logs
    now = datetime.now(UTC)
    timeline_data = []

    # Generate data points based on timeframe
    if timeframe == "1h":
        intervals = 12  # 5-minute intervals
        delta = timedelta(minutes=5)
    elif timeframe == "24h":
        intervals = 24  # 1-hour intervals
        delta = timedelta(hours=1)
    else:  # 7d
        intervals = 7  # 1-day intervals
        delta = timedelta(days=1)

    for i in range(intervals):
        timestamp = now - (delta * (intervals - i - 1))

        # Generate mock threat data
        threat_count = max(0, int(15 + (i % 3) * 5 + (i % 7) * 2))
        blocked_count = int(threat_count * 0.85)  # 85% blocked

        timeline_data.append(
            ThreatTimelineData(
                timestamp=timestamp,
                threat_count=threat_count,
                blocked_count=blocked_count,
                severity_distribution={
                    "critical": int(threat_count * 0.1),
                    "high": int(threat_count * 0.2),
                    "medium": int(threat_count * 0.4),
                    "low": int(threat_count * 0.3),
                },
            )
        )

    return timeline_data


@router.post("/sessions/{session_id}/refresh")
async def refresh_session(
    session_id: str,
    current_user=Depends(get_current_user),
    _permission=Depends(require_permissions(["manage_own_session"])),
) -> dict[str, Any]:
    """Refresh a session's expiration time."""

    session_manager = get_session_manager()

    success = await session_manager.refresh_session(session_id)

    if not success:
        raise HTTPException(status_code=404, detail="Session not found or expired")

    return {
        "success": True,
        "message": "Session refreshed successfully",
        "new_expiry": (datetime.now(UTC) + timedelta(hours=24)).isoformat(),
    }


# Helper functions for security calculations


def _calculate_threat_level() -> str:
    """Calculate current threat level based on recent activity."""
    # Mock implementation - in production, this would analyze real threat data
    import random

    levels = ["LOW", "MEDIUM", "HIGH"]
    weights = [0.7, 0.25, 0.05]  # Bias toward LOW
    return random.choices(levels, weights=weights)[0]


def _get_blocked_threats_count() -> int:
    """Get count of blocked threats in the last 24 hours."""
    # Mock implementation
    import random

    return random.randint(100, 200)


def _calculate_security_score() -> int:
    """Calculate overall security score (0-100)."""
    # Mock implementation - in production, this would consider:
    # - WAF effectiveness
    # - Failed login attempts
    # - Session security
    # - Configuration compliance
    import random

    return random.randint(85, 95)


def _calculate_session_growth() -> float:
    """Calculate session growth percentage."""
    # Mock implementation
    import random

    return round(random.uniform(-5.0, 15.0), 1)
