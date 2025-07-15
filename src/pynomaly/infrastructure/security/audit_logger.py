"""Comprehensive audit logging system for security events and compliance.

This module provides detailed audit logging capabilities including:
- Security event logging
- User action tracking
- API access logging
- Compliance audit trails
- Structured logging with correlation IDs
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from functools import wraps
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


class SecurityEventType(str, Enum):
    """Types of security events to log."""

    # Authentication Events
    AUTH_LOGIN_SUCCESS = "auth.login.success"
    AUTH_LOGIN_FAILURE = "auth.login.failure"
    AUTH_LOGOUT = "auth.logout"
    AUTH_TOKEN_REFRESH = "auth.token.refresh"
    AUTH_TOKEN_REVOKED = "auth.token.revoked"
    AUTH_PASSWORD_CHANGE = "auth.password.change"
    AUTH_ACCOUNT_LOCKED = "auth.account.locked"
    AUTH_ACCOUNT_UNLOCKED = "auth.account.unlocked"

    # Authorization Events
    AUTHZ_ACCESS_DENIED = "authz.access.denied"
    AUTHZ_PERMISSION_GRANTED = "authz.permission.granted"
    AUTHZ_PRIVILEGE_ESCALATION = "authz.privilege.escalation"
    AUTHZ_ROLE_CHANGED = "authz.role.changed"

    # Data Access Events
    DATA_ACCESS_READ = "data.access.read"
    DATA_ACCESS_WRITE = "data.access.write"
    DATA_ACCESS_DELETE = "data.access.delete"
    DATA_EXPORT = "data.export"
    DATA_IMPORT = "data.import"
    DATA_ENCRYPTION = "data.encryption"
    DATA_DECRYPTION = "data.decryption"

    # Security Violations
    SECURITY_SQL_INJECTION = "security.sql_injection"
    SECURITY_XSS_ATTEMPT = "security.xss_attempt"
    SECURITY_CSRF_VIOLATION = "security.csrf_violation"
    SECURITY_RATE_LIMIT_EXCEEDED = "security.rate_limit_exceeded"
    SECURITY_SUSPICIOUS_ACTIVITY = "security.suspicious_activity"
    SECURITY_MALWARE_DETECTED = "security.malware_detected"

    # System Events
    SYSTEM_CONFIG_CHANGE = "system.config.change"
    SYSTEM_SERVICE_START = "system.service.start"
    SYSTEM_SERVICE_STOP = "system.service.stop"
    SYSTEM_BACKUP_CREATED = "system.backup.created"
    SYSTEM_BACKUP_RESTORED = "system.backup.restored"
    SYSTEM_UPDATE_APPLIED = "system.update.applied"

    # API Events
    API_KEY_CREATED = "api.key.created"
    API_KEY_REVOKED = "api.key.revoked"
    API_ENDPOINT_ACCESS = "api.endpoint.access"
    API_RATE_LIMITED = "api.rate_limited"
    API_ERROR = "api.error"
    API_REQUEST = "api.request"

    # Generic Data Access (for advanced threat detection)
    DATA_ACCESS = "data.access"

    # Model Events
    MODEL_TRAINED = "model.trained"
    MODEL_DEPLOYED = "model.deployed"
    MODEL_DELETED = "model.deleted"
    MODEL_PREDICTION = "model.prediction"
    MODEL_ANOMALY_DETECTED = "model.anomaly_detected"

    # MFA Events
    MFA_TOTP_SETUP_INITIATED = "mfa.totp.setup.initiated"
    MFA_TOTP_SETUP_FAILED = "mfa.totp.setup.failed"
    MFA_TOTP_ENABLED = "mfa.totp.enabled"
    MFA_TOTP_VERIFIED = "mfa.totp.verified"
    MFA_TOTP_FAILED = "mfa.totp.failed"
    MFA_TOTP_VERIFICATION_FAILED = "mfa.totp.verification.failed"
    MFA_SMS_SETUP_FAILED = "mfa.sms.setup.failed"
    MFA_SMS_SENT = "mfa.sms.sent"
    MFA_SMS_FAILED = "mfa.sms.failed"
    MFA_SMS_VERIFIED = "mfa.sms.verified"
    MFA_SMS_VERIFICATION_FAILED = "mfa.sms.verification.failed"
    MFA_EMAIL_SETUP_FAILED = "mfa.email.setup.failed"
    MFA_EMAIL_SENT = "mfa.email.sent"
    MFA_EMAIL_FAILED = "mfa.email.failed"
    MFA_EMAIL_VERIFIED = "mfa.email.verified"
    MFA_EMAIL_VERIFICATION_FAILED = "mfa.email.verification.failed"
    MFA_METHOD_ENABLED = "mfa.method.enabled"
    MFA_METHOD_DISABLED = "mfa.method.disabled"
    MFA_ENABLE_FAILED = "mfa.enable.failed"
    MFA_DISABLE_FAILED = "mfa.disable.failed"
    MFA_LOGIN_SUCCESS = "mfa.login.success"
    MFA_LOGIN_FAILED = "mfa.login.failed"
    MFA_BACKUP_CODES_GENERATED = "mfa.backup_codes.generated"
    MFA_BACKUP_CODES_REGENERATED = "mfa.backup_codes.regenerated"
    MFA_BACKUP_CODES_REGENERATION_FAILED = "mfa.backup_codes.regeneration.failed"
    MFA_BACKUP_CODES_ACCESS_FAILED = "mfa.backup_codes.access.failed"
    MFA_BACKUP_CODE_USED = "mfa.backup_code.used"
    MFA_BACKUP_CODE_FAILED = "mfa.backup_code.failed"
    MFA_RECOVERY_SUCCESS = "mfa.recovery.success"
    MFA_RECOVERY_FAILED = "mfa.recovery.failed"
    MFA_DEVICE_REMEMBERED = "mfa.device.remembered"
    MFA_DEVICE_REVOKED = "mfa.device.revoked"
    MFA_DEVICE_REVOCATION_FAILED = "mfa.device.revocation.failed"
    MFA_TRUSTED_DEVICES_ACCESS_FAILED = "mfa.trusted_devices.access.failed"
    MFA_STATUS_CHECK_FAILED = "mfa.status.check.failed"
    MFA_SETTINGS_ACCESS_FAILED = "mfa.settings.access.failed"
    MFA_STATISTICS_ACCESS = "mfa.statistics.access"
    MFA_STATISTICS_ACCESS_FAILED = "mfa.statistics.access.failed"


class AuditLevel(str, Enum):
    """Audit logging levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ComplianceStandard(str, Enum):
    """Compliance standards for audit logging."""

    SOX = "SOX"  # Sarbanes-Oxley
    GDPR = "GDPR"  # General Data Protection Regulation
    HIPAA = "HIPAA"  # Health Insurance Portability and Accountability Act
    PCI_DSS = "PCI_DSS"  # Payment Card Industry Data Security Standard
    ISO27001 = "ISO27001"  # ISO/IEC 27001
    NIST = "NIST"  # NIST Cybersecurity Framework


@dataclass
class AuditContext:
    """Context information for audit events."""

    correlation_id: str
    session_id: str | None = None
    user_id: str | None = None
    user_name: str | None = None
    user_roles: list[str] | None = None
    ip_address: str | None = None
    user_agent: str | None = None
    request_id: str | None = None
    timestamp: float = Field(default_factory=time.time)


class SecurityEvent(BaseModel):
    """Security event data structure."""

    event_type: SecurityEventType
    level: AuditLevel
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # Context
    correlation_id: str
    session_id: str | None = None
    user_id: str | None = None
    user_name: str | None = None
    user_roles: list[str] | None = None

    # Request information
    ip_address: str | None = None
    user_agent: str | None = None
    request_method: str | None = None
    request_path: str | None = None
    request_id: str | None = None

    # Event details
    message: str
    details: dict[str, Any] = Field(default_factory=dict)

    # Security metadata
    risk_score: int | None = None  # 0-100
    threat_indicators: list[str] | None = None
    mitigation_actions: list[str] | None = None

    # Compliance
    compliance_standards: list[ComplianceStandard] | None = None
    data_classification: str | None = None  # PUBLIC, INTERNAL, CONFIDENTIAL, RESTRICTED

    # Technical details
    source_component: str | None = None
    error_code: str | None = None
    stack_trace: str | None = None        json_encoders = {datetime: lambda v: v.isoformat()}


class AuditEvent(BaseModel):
    """General audit event for non-security activities."""

    event_type: str
    level: AuditLevel
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # Context
    correlation_id: str
    user_id: str | None = None
    user_name: str | None = None

    # Event details
    resource: str | None = None  # Resource being accessed/modified
    action: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)

    # Before/after state for change tracking
    before_state: dict[str, Any] | None = None
    after_state: dict[str, Any] | None = None

    # Compliance
    compliance_standards: list[ComplianceStandard] | None = None
    retention_period_days: int | None = None        json_encoders = {datetime: lambda v: v.isoformat()}


class AuditLogger:
    """Service for comprehensive audit logging."""

    def __init__(
        self,
        logger_name: str = "pynomaly.audit",
        enable_structured_logging: bool = True,
        enable_compliance_logging: bool = True,
        default_retention_days: int = 2555,
    ):  # 7 years default
        """Initialize audit logger.

        Args:
            logger_name: Logger name
            enable_structured_logging: Enable JSON structured logging
            enable_compliance_logging: Enable compliance-specific logging
            default_retention_days: Default retention period in days
        """
        self.audit_logger = logging.getLogger(logger_name)
        self.enable_structured_logging = enable_structured_logging
        self.enable_compliance_logging = enable_compliance_logging
        self.default_retention_days = default_retention_days

        # Context storage for correlation
        self._context_stack: list[AuditContext] = []

        # Event handlers for specific compliance standards
        self._compliance_handlers: dict[ComplianceStandard, callable] = {}

        # Initialize structured logging if enabled
        if enable_structured_logging:
            self._setup_structured_logging()

    def _setup_structured_logging(self) -> None:
        """Setup structured JSON logging."""

        # Create custom formatter for audit logs
        class AuditFormatter(logging.Formatter):
            def format(self, record):
                if hasattr(record, "audit_data"):
                    return json.dumps(
                        record.audit_data, default=str, ensure_ascii=False
                    )
                return super().format(record)

        # Add handler with audit formatter
        handler = logging.StreamHandler()
        handler.setFormatter(AuditFormatter())
        self.audit_logger.addHandler(handler)
        self.audit_logger.setLevel(logging.INFO)

    @contextmanager
    def audit_context(self, context: AuditContext):
        """Context manager for audit context tracking.

        Args:
            context: Audit context information
        """
        self._context_stack.append(context)
        try:
            yield context
        finally:
            if self._context_stack:
                self._context_stack.pop()

    def get_current_context(self) -> AuditContext | None:
        """Get current audit context."""
        return self._context_stack[-1] if self._context_stack else None

    def log_security_event(
        self,
        event_type: SecurityEventType,
        message: str,
        level: AuditLevel = AuditLevel.INFO,
        details: dict[str, Any] | None = None,
        risk_score: int | None = None,
        compliance_standards: list[ComplianceStandard] | None = None,
        **kwargs,
    ) -> None:
        """Log a security event.

        Args:
            event_type: Type of security event
            message: Event message
            level: Audit level
            details: Additional event details
            risk_score: Risk score (0-100)
            compliance_standards: Applicable compliance standards
            **kwargs: Additional event attributes
        """
        context = self.get_current_context()

        # Create security event
        event = SecurityEvent(
            event_type=event_type,
            level=level,
            message=message,
            details=details or {},
            risk_score=risk_score,
            compliance_standards=compliance_standards,
            correlation_id=(
                context.correlation_id if context else self._generate_correlation_id()
            ),
            session_id=kwargs.get("session_id") or (context.session_id if context else None),
            user_id=kwargs.get("user_id") or (context.user_id if context else None),
            user_name=kwargs.get("user_name") or (context.user_name if context else None),
            user_roles=kwargs.get("user_roles") or (context.user_roles if context else None),
            ip_address=kwargs.get("ip_address") or (context.ip_address if context else None),
            user_agent=kwargs.get("user_agent") or (context.user_agent if context else None),
            request_id=kwargs.get("request_id") or (context.request_id if context else None),
            **{k: v for k, v in kwargs.items() if k not in ["session_id", "user_id", "user_name", "user_roles", "ip_address", "user_agent", "request_id"]},
        )

        # Log the event
        self._log_event(event, is_security=True)

        # Handle compliance-specific logging
        if self.enable_compliance_logging and compliance_standards:
            self._handle_compliance_logging(event, compliance_standards)

    def log_audit_event(
        self,
        event_type: str,
        action: str,
        message: str,
        resource: str | None = None,
        level: AuditLevel = AuditLevel.INFO,
        details: dict[str, Any] | None = None,
        before_state: dict[str, Any] | None = None,
        after_state: dict[str, Any] | None = None,
        compliance_standards: list[ComplianceStandard] | None = None,
        **kwargs,
    ) -> None:
        """Log a general audit event.

        Args:
            event_type: Type of audit event
            action: Action being performed
            message: Event message
            resource: Resource being accessed/modified
            level: Audit level
            details: Additional event details
            before_state: State before the action
            after_state: State after the action
            compliance_standards: Applicable compliance standards
            **kwargs: Additional event attributes
        """
        context = self.get_current_context()

        # Create audit event
        event = AuditEvent(
            event_type=event_type,
            action=action,
            message=message,
            resource=resource,
            level=level,
            details=details or {},
            before_state=before_state,
            after_state=after_state,
            compliance_standards=compliance_standards,
            retention_period_days=self.default_retention_days,
            correlation_id=(
                context.correlation_id if context else self._generate_correlation_id()
            ),
            user_id=context.user_id if context else None,
            user_name=context.user_name if context else None,
            **kwargs,
        )

        # Log the event
        self._log_event(event, is_security=False)

        # Handle compliance-specific logging
        if self.enable_compliance_logging and compliance_standards:
            self._handle_compliance_logging(event, compliance_standards)

    def _log_event(self, event: SecurityEvent | AuditEvent, is_security: bool) -> None:
        """Log an event to the audit trail."""
        event_dict = event.model_dump()

        # Determine log level
        log_level = getattr(logging, event.level.value)

        if self.enable_structured_logging:
            # Create log record with structured data
            record = self.audit_logger.makeRecord(
                name=self.audit_logger.name,
                level=log_level,
                fn="",
                lno=0,
                msg=event.message,
                args=(),
                exc_info=None,
            )
            record.audit_data = {
                **event_dict,
                "audit_type": "security" if is_security else "general",
                "logger_timestamp": datetime.now(UTC).isoformat(),
            }
            self.audit_logger.handle(record)
        else:
            # Simple text logging
            self.audit_logger.log(
                log_level,
                f"AUDIT[{event.event_type}]: {event.message} | Context: {event.correlation_id}",
            )

    def _handle_compliance_logging(
        self,
        event: SecurityEvent | AuditEvent,
        standards: list[ComplianceStandard],
    ) -> None:
        """Handle compliance-specific logging requirements."""
        for standard in standards:
            handler = self._compliance_handlers.get(standard)
            if handler:
                try:
                    handler(event)
                except Exception as e:
                    logger.error(f"Compliance logging error for {standard}: {e}")
            else:
                # Default compliance logging
                self._default_compliance_logging(event, standard)

    def _default_compliance_logging(
        self, event: SecurityEvent | AuditEvent, standard: ComplianceStandard
    ) -> None:
        """Default compliance logging behavior."""
        compliance_logger = logging.getLogger(f"pynomaly.compliance.{standard.lower()}")

        compliance_record = {
            "standard": standard,
            "event": event.model_dump(),
            "compliance_timestamp": datetime.now(UTC).isoformat(),
            "retention_required": True,
            "immutable": True,
        }

        # Log with WARNING level to ensure it's captured
        compliance_logger.warning(
            json.dumps(compliance_record, default=str, ensure_ascii=False)
        )

    def register_compliance_handler(
        self, standard: ComplianceStandard, handler: callable
    ) -> None:
        """Register a custom compliance handler.

        Args:
            standard: Compliance standard
            handler: Handler function that takes an event
        """
        self._compliance_handlers[standard] = handler
        logger.info(f"Registered compliance handler for {standard}")

    def _generate_correlation_id(self) -> str:
        """Generate a unique correlation ID."""
        import uuid

        return str(uuid.uuid4())

    # Convenience methods for common security events

    def log_login_success(self, user_id: str, user_name: str, ip_address: str) -> None:
        """Log successful login."""
        self.log_security_event(
            SecurityEventType.AUTH_LOGIN_SUCCESS,
            f"User {user_name} logged in successfully",
            details={"user_id": user_id, "ip_address": ip_address},
            compliance_standards=[ComplianceStandard.SOX, ComplianceStandard.GDPR],
        )

    def log_login_failure(self, username: str, ip_address: str, reason: str) -> None:
        """Log failed login attempt."""
        self.log_security_event(
            SecurityEventType.AUTH_LOGIN_FAILURE,
            f"Login failed for user {username}: {reason}",
            level=AuditLevel.WARNING,
            details={
                "username": username,
                "ip_address": ip_address,
                "failure_reason": reason,
            },
            risk_score=60,
            compliance_standards=[ComplianceStandard.SOX, ComplianceStandard.GDPR],
        )

    def log_access_denied(
        self, user_id: str, resource: str, required_permission: str
    ) -> None:
        """Log access denied event."""
        self.log_security_event(
            SecurityEventType.AUTHZ_ACCESS_DENIED,
            f"Access denied to {resource} for user {user_id}",
            level=AuditLevel.WARNING,
            details={
                "user_id": user_id,
                "resource": resource,
                "required_permission": required_permission,
            },
            risk_score=40,
            compliance_standards=[ComplianceStandard.SOX],
        )

    def log_data_access(
        self,
        action: str,
        resource: str,
        user_id: str,
        record_count: int | None = None,
    ) -> None:
        """Log data access event."""
        event_type = {
            "read": SecurityEventType.DATA_ACCESS_READ,
            "write": SecurityEventType.DATA_ACCESS_WRITE,
            "delete": SecurityEventType.DATA_ACCESS_DELETE,
        }.get(action.lower(), SecurityEventType.DATA_ACCESS_READ)

        self.log_security_event(
            event_type,
            f"Data {action} on {resource} by user {user_id}",
            details={
                "action": action,
                "resource": resource,
                "user_id": user_id,
                "record_count": record_count,
            },
            compliance_standards=[ComplianceStandard.GDPR, ComplianceStandard.SOX],
        )

    def log_sql_injection_attempt(self, query: str, ip_address: str) -> None:
        """Log SQL injection attempt."""
        self.log_security_event(
            SecurityEventType.SECURITY_SQL_INJECTION,
            "SQL injection attempt detected",
            level=AuditLevel.ERROR,
            details={
                "suspicious_query": query[:200],  # Truncate for logging
                "ip_address": ip_address,
            },
            risk_score=90,
            threat_indicators=["sql_injection"],
            mitigation_actions=["block_request", "alert_security_team"],
        )


# Global audit logger instance
_audit_logger: AuditLogger | None = None


def get_audit_logger() -> AuditLogger:
    """Get global audit logger instance."""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger


def init_audit_logger(
    logger_name: str = "pynomaly.audit",
    enable_structured_logging: bool = True,
    enable_compliance_logging: bool = True,
) -> AuditLogger:
    """Initialize global audit logger.

    Args:
        logger_name: Logger name
        enable_structured_logging: Enable JSON structured logging
        enable_compliance_logging: Enable compliance-specific logging

    Returns:
        Audit logger instance
    """
    global _audit_logger
    _audit_logger = AuditLogger(
        logger_name=logger_name,
        enable_structured_logging=enable_structured_logging,
        enable_compliance_logging=enable_compliance_logging,
    )
    return _audit_logger


def log_security_event(event_type: SecurityEventType, message: str, **kwargs) -> None:
    """Convenience function to log security events.

    Args:
        event_type: Type of security event
        message: Event message
        **kwargs: Additional event attributes
    """
    audit_logger = get_audit_logger()
    audit_logger.log_security_event(event_type, message, **kwargs)


def audit_context(
    correlation_id: str,
    user_id: str | None = None,
    session_id: str | None = None,
    ip_address: str | None = None,
    user_agent: str | None = None,
):
    """Create audit context for tracking related events.

    Args:
        correlation_id: Correlation ID for related events
        user_id: User ID
        session_id: Session ID
        ip_address: Client IP address
        user_agent: User agent string

    Returns:
        Audit context manager
    """
    audit_logger = get_audit_logger()
    context = AuditContext(
        correlation_id=correlation_id,
        user_id=user_id,
        session_id=session_id,
        ip_address=ip_address,
        user_agent=user_agent,
    )
    return audit_logger.audit_context(context)


def audit_action(
    action: str, resource: str | None = None, event_type: str = "user_action"
):
    """Decorator to audit function calls.

    Args:
        action: Action being performed
        resource: Resource being accessed
        event_type: Type of audit event

    Returns:
        Decorator function
    """

    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            audit_logger = get_audit_logger()

            try:
                result = await func(*args, **kwargs)
                audit_logger.log_audit_event(
                    event_type=event_type,
                    action=action,
                    resource=resource or func.__name__,
                    message=f"Successfully executed {action}",
                    level=AuditLevel.INFO,
                )
                return result
            except Exception as e:
                audit_logger.log_audit_event(
                    event_type=event_type,
                    action=action,
                    resource=resource or func.__name__,
                    message=f"Failed to execute {action}: {str(e)}",
                    level=AuditLevel.ERROR,
                    details={"error": str(e), "function": func.__name__},
                )
                raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            audit_logger = get_audit_logger()

            try:
                result = func(*args, **kwargs)
                audit_logger.log_audit_event(
                    event_type=event_type,
                    action=action,
                    resource=resource or func.__name__,
                    message=f"Successfully executed {action}",
                    level=AuditLevel.INFO,
                )
                return result
            except Exception as e:
                audit_logger.log_audit_event(
                    event_type=event_type,
                    action=action,
                    resource=resource or func.__name__,
                    message=f"Failed to execute {action}: {str(e)}",
                    level=AuditLevel.ERROR,
                    details={"error": str(e), "function": func.__name__},
                )
                raise

        # Return appropriate wrapper based on function type
        import asyncio

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator
