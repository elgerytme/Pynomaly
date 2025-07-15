"""
Enhanced Session Management System

Provides comprehensive session management with:
- Server-side session storage
- Session fixation protection
- Concurrent session management
- Session timeout handling
- Real-time session monitoring
"""

import json
import secrets
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

import redis
from fastapi import Request, Response
from pydantic import BaseModel

from pynomaly.infrastructure.config.settings import get_settings


class SessionData(BaseModel):
    """Session data model."""

    session_id: str
    user_id: str | None = None
    user_email: str | None = None
    user_role: str | None = None
    ip_address: str
    user_agent: str
    created_at: datetime
    last_accessed: datetime
    expires_at: datetime
    is_active: bool = True
    csrf_token: str | None = None
    login_method: str | None = None  # jwt, api_key, oauth, etc.
    security_level: str = "standard"  # standard, elevated, admin
    concurrent_session_count: int = 1

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat(), UUID: lambda v: str(v)}


class SessionSecurityEvent(BaseModel):
    """Session security event model."""

    event_id: str
    session_id: str
    event_type: str  # created, accessed, expired, terminated, suspicious
    timestamp: datetime
    ip_address: str
    user_agent: str
    details: dict[str, Any]
    severity: str = "info"  # info, warning, critical


class EnhancedSessionManager:
    """Enhanced session manager with server-side storage and security features."""

    def __init__(self):
        self.settings = get_settings()
        self._redis = None
        self._session_timeout = timedelta(hours=24)
        self._max_concurrent_sessions = 5
        self._session_cookie_name = "pynomaly_session_id"
        self._security_events: list[SessionSecurityEvent] = []

    @property
    def redis(self) -> redis.Redis:
        """Get Redis connection for session storage."""
        if self._redis is None:
            self._redis = redis.Redis(
                host=getattr(self.settings, "redis_host", "localhost"),
                port=getattr(self.settings, "redis_port", 6379),
                db=getattr(self.settings, "redis_session_db", 1),
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
            )
        return self._redis

    async def create_session(
        self,
        request: Request,
        user_id: str | None = None,
        user_email: str | None = None,
        user_role: str | None = None,
        login_method: str = "jwt",
        security_level: str = "standard",
    ) -> SessionData:
        """Create a new session with security checks."""

        # Check concurrent session limits
        if user_id:
            await self._enforce_concurrent_session_limits(user_id)

        # Generate secure session ID
        session_id = self._generate_session_id()

        # Get client information
        ip_address = self._get_client_ip(request)
        user_agent = request.headers.get("user-agent", "Unknown")

        # Create session data
        now = datetime.now(UTC)
        expires_at = now + self._session_timeout

        session_data = SessionData(
            session_id=session_id,
            user_id=user_id,
            user_email=user_email,
            user_role=user_role,
            ip_address=ip_address,
            user_agent=user_agent,
            created_at=now,
            last_accessed=now,
            expires_at=expires_at,
            csrf_token=secrets.token_urlsafe(32),
            login_method=login_method,
            security_level=security_level,
        )

        # Store session in Redis
        await self._store_session(session_data)

        # Track user sessions
        if user_id:
            await self._add_user_session(user_id, session_id)

        # Log security event
        await self._log_security_event(
            session_id=session_id,
            event_type="created",
            ip_address=ip_address,
            user_agent=user_agent,
            details={
                "user_id": user_id,
                "login_method": login_method,
                "security_level": security_level,
            },
        )

        return session_data

    async def get_session(self, session_id: str) -> SessionData | None:
        """Get session data by session ID."""
        try:
            session_json = self.redis.get(f"session:{session_id}")
            if not session_json:
                return None

            session_dict = json.loads(session_json)
            session_data = SessionData(**session_dict)

            # Check if session is expired
            if datetime.now(UTC) > session_data.expires_at:
                await self.terminate_session(session_id, reason="expired")
                return None

            # Update last accessed time
            session_data.last_accessed = datetime.now(UTC)
            await self._store_session(session_data)

            return session_data

        except Exception as e:
            print(f"Error getting session {session_id}: {e}")
            return None

    async def refresh_session(self, session_id: str) -> bool:
        """Refresh session expiration time."""
        session_data = await self.get_session(session_id)
        if not session_data:
            return False

        # Extend expiration
        session_data.expires_at = datetime.now(UTC) + self._session_timeout
        session_data.last_accessed = datetime.now(UTC)

        await self._store_session(session_data)
        return True

    async def terminate_session(self, session_id: str, reason: str = "logout") -> bool:
        """Terminate a session."""
        session_data = await self.get_session(session_id)
        if not session_data:
            return False

        # Remove from Redis
        self.redis.delete(f"session:{session_id}")

        # Remove from user sessions
        if session_data.user_id:
            await self._remove_user_session(session_data.user_id, session_id)

        # Log security event
        await self._log_security_event(
            session_id=session_id,
            event_type="terminated",
            ip_address=session_data.ip_address,
            user_agent=session_data.user_agent,
            details={"reason": reason},
        )

        return True

    async def terminate_user_sessions(
        self, user_id: str, except_session_id: str | None = None
    ) -> int:
        """Terminate all sessions for a user except specified session."""
        user_sessions = await self.get_user_sessions(user_id)
        terminated_count = 0

        for session_id in user_sessions:
            if session_id != except_session_id:
                if await self.terminate_session(session_id, reason="admin_termination"):
                    terminated_count += 1

        return terminated_count

    async def get_user_sessions(self, user_id: str) -> list[str]:
        """Get all active session IDs for a user."""
        try:
            session_ids = self.redis.smembers(f"user_sessions:{user_id}")
            return list(session_ids) if session_ids else []
        except Exception:
            return []

    async def get_session_details(self, user_id: str) -> list[dict[str, Any]]:
        """Get detailed session information for a user."""
        session_ids = await self.get_user_sessions(user_id)
        session_details = []

        for session_id in session_ids:
            session_data = await self.get_session(session_id)
            if session_data:
                session_details.append(
                    {
                        "session_id": session_id,
                        "ip_address": session_data.ip_address,
                        "user_agent": session_data.user_agent,
                        "created_at": session_data.created_at.isoformat(),
                        "last_accessed": session_data.last_accessed.isoformat(),
                        "login_method": session_data.login_method,
                        "security_level": session_data.security_level,
                        "is_current": False,  # Will be set by caller
                    }
                )

        return session_details

    async def validate_session_security(
        self, session_id: str, request: Request
    ) -> dict[str, Any]:
        """Validate session security and detect anomalies."""
        session_data = await self.get_session(session_id)
        if not session_data:
            return {"valid": False, "reason": "session_not_found"}

        current_ip = self._get_client_ip(request)
        current_user_agent = request.headers.get("user-agent", "Unknown")

        security_issues = []

        # Check IP address change
        if session_data.ip_address != current_ip:
            security_issues.append(
                {
                    "type": "ip_change",
                    "message": f"IP changed from {session_data.ip_address} to {current_ip}",
                    "severity": "warning",
                }
            )

        # Check user agent change
        if session_data.user_agent != current_user_agent:
            security_issues.append(
                {
                    "type": "user_agent_change",
                    "message": "User agent changed",
                    "severity": "warning",
                }
            )

        # Check session age
        session_age = datetime.now(UTC) - session_data.created_at
        if session_age > timedelta(days=7):
            security_issues.append(
                {
                    "type": "old_session",
                    "message": f"Session is {session_age.days} days old",
                    "severity": "info",
                }
            )

        # Log suspicious activity
        if security_issues:
            await self._log_security_event(
                session_id=session_id,
                event_type="suspicious",
                ip_address=current_ip,
                user_agent=current_user_agent,
                details={"security_issues": security_issues},
                severity="warning",
            )

        return {
            "valid": True,
            "security_issues": security_issues,
            "requires_reauth": len(
                [i for i in security_issues if i["severity"] == "critical"]
            )
            > 0,
        }

    async def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions."""
        # This would typically be run by a background task
        expired_count = 0

        # Get all session keys
        session_keys = self.redis.keys("session:*")

        for key in session_keys:
            try:
                session_json = self.redis.get(key)
                if session_json:
                    session_dict = json.loads(session_json)
                    expires_at = datetime.fromisoformat(session_dict["expires_at"])

                    if datetime.now(UTC) > expires_at:
                        session_id = key.replace("session:", "")
                        await self.terminate_session(session_id, reason="expired")
                        expired_count += 1

            except Exception as e:
                print(f"Error processing session {key}: {e}")

        return expired_count

    def set_session_cookie(
        self, response: Response, session_id: str, secure: bool = True
    ) -> None:
        """Set session cookie with secure attributes."""
        response.set_cookie(
            key=self._session_cookie_name,
            value=session_id,
            max_age=int(self._session_timeout.total_seconds()),
            httponly=True,
            secure=secure and self.settings.environment == "production",
            samesite="strict" if self.settings.environment == "production" else "lax",
            path="/",
        )

    def clear_session_cookie(self, response: Response) -> None:
        """Clear session cookie."""
        response.delete_cookie(
            key=self._session_cookie_name,
            path="/",
        )

    def get_session_id_from_request(self, request: Request) -> str | None:
        """Extract session ID from request."""
        return request.cookies.get(self._session_cookie_name)

    # Private helper methods

    def _generate_session_id(self) -> str:
        """Generate a cryptographically secure session ID."""
        return secrets.token_urlsafe(32)

    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address from request."""
        # Check for forwarded headers
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        return str(request.client.host) if request.client else "unknown"

    async def _store_session(self, session_data: SessionData) -> None:
        """Store session data in Redis."""
        session_json = session_data.model_dump_json()
        ttl = int((session_data.expires_at - datetime.now(UTC)).total_seconds())

        self.redis.setex(f"session:{session_data.session_id}", ttl, session_json)

    async def _add_user_session(self, user_id: str, session_id: str) -> None:
        """Add session to user's session set."""
        self.redis.sadd(f"user_sessions:{user_id}", session_id)

    async def _remove_user_session(self, user_id: str, session_id: str) -> None:
        """Remove session from user's session set."""
        self.redis.srem(f"user_sessions:{user_id}", session_id)

    async def _enforce_concurrent_session_limits(self, user_id: str) -> None:
        """Enforce concurrent session limits for a user."""
        user_sessions = await self.get_user_sessions(user_id)

        if len(user_sessions) >= self._max_concurrent_sessions:
            # Remove oldest sessions
            sessions_to_remove = len(user_sessions) - self._max_concurrent_sessions + 1

            # Get session details to find oldest
            session_details = []
            for session_id in user_sessions:
                session_data = await self.get_session(session_id)
                if session_data:
                    session_details.append((session_id, session_data.created_at))

            # Sort by creation time and remove oldest
            session_details.sort(key=lambda x: x[1])
            for session_id, _ in session_details[:sessions_to_remove]:
                await self.terminate_session(session_id, reason="concurrent_limit")

    async def _log_security_event(
        self,
        session_id: str,
        event_type: str,
        ip_address: str,
        user_agent: str,
        details: dict[str, Any],
        severity: str = "info",
    ) -> None:
        """Log security event."""
        event = SessionSecurityEvent(
            event_id=str(uuid4()),
            session_id=session_id,
            event_type=event_type,
            timestamp=datetime.now(UTC),
            ip_address=ip_address,
            user_agent=user_agent,
            details=details,
            severity=severity,
        )

        # Store in Redis with TTL
        event_json = event.model_dump_json()
        self.redis.setex(
            f"security_event:{event.event_id}",
            86400 * 30,  # 30 days
            event_json,
        )

        # Add to security events list (for real-time monitoring)
        self.redis.lpush("security_events", event_json)
        self.redis.ltrim("security_events", 0, 1000)  # Keep last 1000 events


# Singleton instance
session_manager = EnhancedSessionManager()


def get_session_manager() -> EnhancedSessionManager:
    """Get session manager instance."""
    return session_manager
