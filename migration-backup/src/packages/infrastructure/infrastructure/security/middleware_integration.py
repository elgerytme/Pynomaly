"""Integration middleware for security components with FastAPI.

This module provides easy integration of all security components with FastAPI applications.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI

from .audit_logger import AuditLogger
from .encryption import DataEncryption
from .input_sanitizer import InputSanitizer
from .security_headers import SecurityHeaders, SecurityHeadersMiddleware
from .security_monitor import SecurityMonitor
from .sql_protection import SQLInjectionProtector
from .user_tracking import UserActionTracker, UserTrackingMiddleware

logger = logging.getLogger(__name__)


class SecurityMiddlewareStack:
    """Complete security middleware stack for FastAPI applications."""

    def __init__(
        self,
        app: FastAPI,
        security_headers: SecurityHeaders | None = None,
        user_tracker: UserActionTracker | None = None,
        audit_logger: AuditLogger | None = None,
        security_monitor: SecurityMonitor | None = None,
        input_sanitizer: InputSanitizer | None = None,
        sql_protector: SQLInjectionProtector | None = None,
        data_encryption: DataEncryption | None = None,
    ):
        """Initialize complete security middleware stack.

        Args:
            app: FastAPI application
            security_headers: Security headers configuration
            user_tracker: User action tracker
            audit_logger: Audit logger
            security_monitor: Security monitor
            input_sanitizer: Input sanitizer
            sql_protector: SQL injection protector
            data_encryption: Data encryption service
        """
        self.app = app
        self.security_headers = security_headers
        self.user_tracker = user_tracker
        self.audit_logger = audit_logger
        self.security_monitor = security_monitor
        self.input_sanitizer = input_sanitizer
        self.sql_protector = sql_protector
        self.data_encryption = data_encryption

        self._setup_middleware()

    def _setup_middleware(self) -> None:
        """Setup all security middleware in correct order."""

        # 1. Security Headers (outermost)
        if self.security_headers:
            self.app.add_middleware(
                SecurityHeadersMiddleware, config=self.security_headers
            )

        # 2. User Action Tracking
        if self.user_tracker:
            self.app.add_middleware(UserTrackingMiddleware, tracker=self.user_tracker)

        logger.info("Security middleware stack initialized")

    async def startup(self) -> None:
        """Initialize security services on startup."""
        # Start security monitoring if available
        if self.security_monitor:
            await self.security_monitor.start_monitoring()
            logger.info("Security monitoring started")

    async def shutdown(self) -> None:
        """Cleanup security services on shutdown."""
        # Stop security monitoring if available
        if self.security_monitor:
            await self.security_monitor.stop_monitoring()
            logger.info("Security monitoring stopped")


def setup_security_middleware(
    app: FastAPI,
    enable_security_headers: bool = True,
    enable_user_tracking: bool = True,
    enable_audit_logging: bool = True,
    enable_security_monitoring: bool = True,
    development_mode: bool = False,
) -> SecurityMiddlewareStack:
    """Setup complete security middleware stack with default configurations.

    Args:
        app: FastAPI application
        enable_security_headers: Enable security headers middleware
        enable_user_tracking: Enable user action tracking
        enable_audit_logging: Enable audit logging
        enable_security_monitoring: Enable security monitoring
        development_mode: Use development-friendly configurations

    Returns:
        Configured security middleware stack
    """
    from .security_headers import create_development_headers, create_production_headers

    # Security headers
    security_headers = None
    if enable_security_headers:
        security_headers = (
            create_development_headers()
            if development_mode
            else create_production_headers()
        )

    # Audit logger
    audit_logger = None
    if enable_audit_logging:
        audit_logger = AuditLogger(
            enable_structured_logging=not development_mode,
            enable_compliance_logging=not development_mode,
        )

    # Security monitor
    security_monitor = None
    if enable_security_monitoring:
        security_monitor = SecurityMonitor(audit_logger)

    # User tracker
    user_tracker = None
    if enable_user_tracking:
        user_tracker = UserActionTracker(audit_logger, security_monitor)

    # Create middleware stack
    middleware_stack = SecurityMiddlewareStack(
        app=app,
        security_headers=security_headers,
        user_tracker=user_tracker,
        audit_logger=audit_logger,
        security_monitor=security_monitor,
    )

    # Setup startup and shutdown events
    @app.on_event("startup")
    async def startup_event():
        await middleware_stack.startup()

    @app.on_event("shutdown")
    async def shutdown_event():
        await middleware_stack.shutdown()

    return middleware_stack


def add_security_endpoints(
    app: FastAPI,
    security_monitor: SecurityMonitor | None = None,
    user_tracker: UserActionTracker | None = None,
) -> None:
    """Add security-related API endpoints.

    Args:
        app: FastAPI application
        security_monitor: Security monitor instance
        user_tracker: User tracker instance
    """

    @app.get("/api/security/status")
    async def get_security_status():
        """Get security monitoring status."""
        if not security_monitor:
            return {
                "status": "disabled",
                "message": "Security monitoring not configured",
            }

        return security_monitor.get_security_summary()

    @app.get("/api/security/alerts")
    async def get_security_alerts(threat_level: str | None = None):
        """Get active security alerts."""
        if not security_monitor:
            return {"alerts": [], "message": "Security monitoring not configured"}

        from .security_monitor import ThreatLevel

        level_filter = None
        if threat_level:
            try:
                level_filter = ThreatLevel(threat_level.upper())
            except ValueError:
                pass

        alerts = security_monitor.get_active_alerts(threat_level=level_filter)
        return {
            "alerts": [
                {
                    "alert_id": alert.alert_id,
                    "alert_type": alert.alert_type,
                    "threat_level": alert.threat_level,
                    "title": alert.title,
                    "description": alert.description,
                    "timestamp": alert.timestamp.isoformat(),
                    "source_ip": alert.source_ip,
                    "user_id": alert.user_id,
                    "indicators": alert.indicators,
                    "recommended_actions": alert.recommended_actions,
                }
                for alert in alerts
            ],
            "total": len(alerts),
        }

    @app.post("/api/security/alerts/{alert_id}/acknowledge")
    async def acknowledge_alert(alert_id: str):
        """Acknowledge a security alert."""
        if not security_monitor:
            return {"success": False, "message": "Security monitoring not configured"}

        success = security_monitor.acknowledge_alert(alert_id)
        return {
            "success": success,
            "message": "Alert acknowledged" if success else "Alert not found",
        }

    @app.get("/api/security/users/{user_id}/activity")
    async def get_user_activity(user_id: str, hours: int = 24):
        """Get user activity summary."""
        if not user_tracker:
            return {"activity": {}, "message": "User tracking not configured"}

        activity = user_tracker.get_user_activity_summary(user_id, hours)
        return activity


# Utility functions for common security tasks


def sanitize_request_data(
    request_data: dict, sanitizer: InputSanitizer | None = None
) -> dict:
    """Sanitize request data using the input sanitizer.

    Args:
        request_data: Request data to sanitize
        sanitizer: Input sanitizer instance

    Returns:
        Sanitized request data
    """
    if not sanitizer:
        from .input_sanitizer import get_sanitizer

        sanitizer = get_sanitizer()

    return sanitizer.sanitize_dict(request_data)


def validate_sql_query(query: str, parameters: dict | None = None) -> bool:
    """Validate SQL query for injection risks.

    Args:
        query: SQL query to validate
        parameters: Query parameters

    Returns:
        True if query is safe
    """
    from .sql_protection import get_sql_protector

    protector = get_sql_protector()
    return protector.validate_query_safety(query, parameters)


def encrypt_sensitive_data(data: any) -> str:
    """Encrypt sensitive data for storage.

    Args:
        data: Data to encrypt

    Returns:
        Encrypted data as base64 string
    """
    from .encryption import get_encryption_service

    encryption_service = get_encryption_service()
    return encryption_service.encrypt_sensitive_data(data)


def decrypt_sensitive_data(encrypted_data: str, return_type: type = str) -> any:
    """Decrypt sensitive data from storage.

    Args:
        encrypted_data: Encrypted data
        return_type: Expected return type

    Returns:
        Decrypted data
    """
    from .encryption import get_encryption_service

    encryption_service = get_encryption_service()
    return encryption_service.decrypt_sensitive_data(encrypted_data, return_type)
