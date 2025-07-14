"""Audit logger protocol for domain layer."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Optional


class AuditLevel(Enum):
    """Audit log levels."""
    
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class SecurityEventType(Enum):
    """Security event types."""
    
    MFA_TOTP_SETUP_INITIATED = "mfa_totp_setup_initiated"
    MFA_TOTP_VERIFIED = "mfa_totp_verified"
    MFA_TOTP_FAILED = "mfa_totp_failed"
    MFA_TOTP_ENABLED = "mfa_totp_enabled"
    MFA_SMS_SENT = "mfa_sms_sent"
    MFA_SMS_VERIFIED = "mfa_sms_verified"
    MFA_SMS_FAILED = "mfa_sms_failed"
    MFA_EMAIL_SENT = "mfa_email_sent"
    MFA_EMAIL_VERIFIED = "mfa_email_verified"
    MFA_EMAIL_FAILED = "mfa_email_failed"
    MFA_BACKUP_CODES_GENERATED = "mfa_backup_codes_generated"
    MFA_BACKUP_CODE_USED = "mfa_backup_code_used"
    MFA_BACKUP_CODE_FAILED = "mfa_backup_code_failed"
    MFA_DEVICE_REMEMBERED = "mfa_device_remembered"
    MFA_DEVICE_REVOKED = "mfa_device_revoked"
    MFA_METHOD_DISABLED = "mfa_method_disabled"


class AuditLoggerProtocol(ABC):
    """Protocol for audit logging in domain layer."""
    
    @abstractmethod
    def log_security_event(
        self,
        event_type: SecurityEventType,
        message: str,
        level: AuditLevel = AuditLevel.INFO,
        user_id: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
        risk_score: Optional[int] = None,
    ) -> None:
        """Log a security event."""
        pass