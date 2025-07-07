"""Audit logging for security events and compliance."""

import json
import logging
from datetime import datetime, UTC
from enum import Enum
from typing import Any, Dict, Optional
from dataclasses import dataclass, asdict
from contextlib import asynccontextmanager

from pynomaly.infrastructure.observability import get_logger


class AuditEventType(str, Enum):
    """Types of audit events."""
    
    # Authentication events
    LOGIN_SUCCESS = "auth.login.success"
    LOGIN_FAILURE = "auth.login.failure"
    LOGOUT = "auth.logout"
    TOKEN_CREATED = "auth.token.created"
    TOKEN_REVOKED = "auth.token.revoked"
    API_KEY_CREATED = "auth.api_key.created"
    API_KEY_REVOKED = "auth.api_key.revoked"
    PASSWORD_CHANGED = "auth.password.changed"
    PASSWORD_RESET = "auth.password.reset"
    
    # Authorization events
    ACCESS_GRANTED = "authz.access.granted"
    ACCESS_DENIED = "authz.access.denied"
    PERMISSION_GRANTED = "authz.permission.granted"
    PERMISSION_REVOKED = "authz.permission.revoked"
    ROLE_ASSIGNED = "authz.role.assigned"
    ROLE_REMOVED = "authz.role.removed"
    
    # Data events
    DATA_CREATED = "data.created"
    DATA_UPDATED = "data.updated"
    DATA_DELETED = "data.deleted"
    DATA_ACCESSED = "data.accessed"
    DATA_EXPORTED = "data.exported"
    
    # System events
    CONFIG_CHANGED = "system.config.changed"
    SERVICE_STARTED = "system.service.started"
    SERVICE_STOPPED = "system.service.stopped"
    MAINTENANCE_MODE = "system.maintenance.enabled"
    
    # Security events
    SECURITY_ALERT = "security.alert"
    RATE_LIMIT_EXCEEDED = "security.rate_limit.exceeded"
    SUSPICIOUS_ACTIVITY = "security.suspicious.activity"
    THREAT_DETECTED = "security.threat.detected"
    ACCOUNT_LOCKED = "security.account.locked"
    
    # User management events
    USER_CREATED = "user.created"
    USER_UPDATED = "user.updated"
    USER_DELETED = "user.deleted"
    USER_ACTIVATED = "user.activated"
    USER_DEACTIVATED = "user.deactivated"
    
    # Model/detector events
    MODEL_CREATED = "model.created"
    MODEL_TRAINED = "model.trained"
    MODEL_DEPLOYED = "model.deployed"
    MODEL_DELETED = "model.deleted"
    DETECTION_RUN = "model.detection.run"
    
    # Admin events
    ADMIN_ACTION = "admin.action"
    BULK_OPERATION = "admin.bulk.operation"
    SYSTEM_BACKUP = "admin.backup"
    SYSTEM_RESTORE = "admin.restore"


class AuditSeverity(str, Enum):
    """Severity levels for audit events."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AuditEvent:
    """Audit event data structure."""
    
    event_type: AuditEventType
    timestamp: datetime
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    resource: Optional[str] = None
    action: Optional[str] = None
    outcome: str = "success"  # success, failure, error
    severity: AuditSeverity = AuditSeverity.LOW
    message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    correlation_id: Optional[str] = None
    tenant_id: Optional[str] = None
    risk_score: Optional[int] = None  # 0-100
    
    def __post_init__(self):
        """Post-initialization processing."""
        if self.timestamp is None:
            self.timestamp = datetime.now(UTC)
        
        if self.details is None:
            self.details = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=str)


class AuditLogger:
    """Centralized audit logging service."""
    
    def __init__(self, logger_name: str = "pynomaly.audit"):
        """Initialize audit logger.
        
        Args:
            logger_name: Name for the audit logger
        """
        self.logger = get_logger(logger_name)
        self.logger.setLevel(logging.INFO)
        
        # Create audit-specific handler
        self._setup_audit_handler()
        
        # Event processors for enrichment
        self._event_processors = []
        
        # Risk scoring rules
        self._risk_rules = self._init_risk_rules()
    
    def _setup_audit_handler(self):
        """Set up dedicated audit log handler."""
        # In production, this should write to a separate audit log file
        # or send to a centralized audit system
        audit_handler = logging.StreamHandler()
        audit_formatter = logging.Formatter(
            '%(asctime)s - AUDIT - %(levelname)s - %(message)s'
        )
        audit_handler.setFormatter(audit_formatter)
        
        # Create dedicated audit logger
        self.audit_logger = logging.getLogger("audit")
        self.audit_logger.setLevel(logging.INFO)
        self.audit_logger.addHandler(audit_handler)
        self.audit_logger.propagate = False
    
    def _init_risk_rules(self) -> Dict[AuditEventType, int]:
        """Initialize risk scoring rules.
        
        Returns:
            Dictionary mapping event types to base risk scores
        """
        return {
            # High risk events
            AuditEventType.LOGIN_FAILURE: 30,
            AuditEventType.ACCESS_DENIED: 25,
            AuditEventType.ACCOUNT_LOCKED: 60,
            AuditEventType.SUSPICIOUS_ACTIVITY: 70,
            AuditEventType.THREAT_DETECTED: 80,
            AuditEventType.RATE_LIMIT_EXCEEDED: 40,
            
            # Medium risk events
            AuditEventType.PASSWORD_CHANGED: 20,
            AuditEventType.API_KEY_CREATED: 15,
            AuditEventType.DATA_DELETED: 25,
            AuditEventType.CONFIG_CHANGED: 30,
            AuditEventType.ADMIN_ACTION: 35,
            
            # Low risk events
            AuditEventType.LOGIN_SUCCESS: 5,
            AuditEventType.DATA_ACCESSED: 5,
            AuditEventType.DETECTION_RUN: 5,
            AuditEventType.DATA_CREATED: 10,
        }
    
    def add_event_processor(self, processor):
        """Add event processor for enrichment.
        
        Args:
            processor: Function that takes and returns an AuditEvent
        """
        self._event_processors.append(processor)
    
    def log_event(self, event: AuditEvent) -> None:
        """Log an audit event.
        
        Args:
            event: Audit event to log
        """
        # Process event through enrichment processors
        for processor in self._event_processors:
            try:
                event = processor(event)
            except Exception as e:
                self.logger.warning(f"Event processor failed: {e}")
        
        # Calculate risk score if not set
        if event.risk_score is None:
            event.risk_score = self._calculate_risk_score(event)
        
        # Set severity based on risk score
        if event.severity == AuditSeverity.LOW:
            event.severity = self._calculate_severity(event.risk_score)
        
        # Log to audit logger
        log_data = event.to_dict()
        
        # Choose log level based on severity
        if event.severity == AuditSeverity.CRITICAL:
            self.audit_logger.critical(json.dumps(log_data))
        elif event.severity == AuditSeverity.HIGH:
            self.audit_logger.error(json.dumps(log_data))
        elif event.severity == AuditSeverity.MEDIUM:
            self.audit_logger.warning(json.dumps(log_data))
        else:
            self.audit_logger.info(json.dumps(log_data))
        
        # Also log to application logger for correlation
        self.logger.info(
            f"Audit event: {event.event_type} - {event.outcome} - "
            f"User: {event.user_id} - IP: {event.ip_address} - "
            f"Risk: {event.risk_score}",
            extra={
                'audit_event': log_data,
                'correlation_id': event.correlation_id
            }
        )
    
    def _calculate_risk_score(self, event: AuditEvent) -> int:
        """Calculate risk score for event.
        
        Args:
            event: Audit event
            
        Returns:
            Risk score (0-100)
        """
        base_score = self._risk_rules.get(event.event_type, 10)
        
        # Adjust based on outcome
        if event.outcome == "failure":
            base_score += 20
        elif event.outcome == "error":
            base_score += 15
        
        # Adjust based on context
        if event.details:
            # Multiple failed attempts
            if "attempt_count" in event.details and event.details["attempt_count"] > 3:
                base_score += 30
            
            # Suspicious patterns
            if "threat_type" in event.details:
                base_score += 40
            
            # Admin actions
            if event.event_type in [AuditEventType.ADMIN_ACTION, AuditEventType.CONFIG_CHANGED]:
                if "bulk" in str(event.details).lower():
                    base_score += 20
        
        return min(100, max(0, base_score))
    
    def _calculate_severity(self, risk_score: int) -> AuditSeverity:
        """Calculate severity based on risk score.
        
        Args:
            risk_score: Risk score (0-100)
            
        Returns:
            Audit severity
        """
        if risk_score >= 80:
            return AuditSeverity.CRITICAL
        elif risk_score >= 60:
            return AuditSeverity.HIGH
        elif risk_score >= 30:
            return AuditSeverity.MEDIUM
        else:
            return AuditSeverity.LOW
    
    # Convenience methods for common events
    
    def log_authentication(self, 
                          event_type: AuditEventType,
                          user_id: Optional[str] = None,
                          outcome: str = "success",
                          ip_address: Optional[str] = None,
                          user_agent: Optional[str] = None,
                          details: Optional[Dict[str, Any]] = None,
                          correlation_id: Optional[str] = None):
        """Log authentication event."""
        event = AuditEvent(
            event_type=event_type,
            timestamp=datetime.now(UTC),
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            outcome=outcome,
            details=details or {},
            correlation_id=correlation_id
        )
        self.log_event(event)
    
    def log_authorization(self,
                         event_type: AuditEventType,
                         user_id: str,
                         resource: str,
                         action: str,
                         outcome: str = "granted",
                         details: Optional[Dict[str, Any]] = None,
                         correlation_id: Optional[str] = None):
        """Log authorization event."""
        event = AuditEvent(
            event_type=event_type,
            timestamp=datetime.now(UTC),
            user_id=user_id,
            resource=resource,
            action=action,
            outcome=outcome,
            details=details or {},
            correlation_id=correlation_id
        )
        self.log_event(event)
    
    def log_data_access(self,
                       event_type: AuditEventType,
                       user_id: str,
                       resource: str,
                       action: str,
                       outcome: str = "success",
                       details: Optional[Dict[str, Any]] = None,
                       correlation_id: Optional[str] = None):
        """Log data access event."""
        event = AuditEvent(
            event_type=event_type,
            timestamp=datetime.now(UTC),
            user_id=user_id,
            resource=resource,
            action=action,
            outcome=outcome,
            details=details or {},
            correlation_id=correlation_id
        )
        self.log_event(event)
    
    def log_security_alert(self,
                          alert_type: str,
                          message: str,
                          user_id: Optional[str] = None,
                          ip_address: Optional[str] = None,
                          severity: AuditSeverity = AuditSeverity.HIGH,
                          details: Optional[Dict[str, Any]] = None,
                          correlation_id: Optional[str] = None):
        """Log security alert."""
        event = AuditEvent(
            event_type=AuditEventType.SECURITY_ALERT,
            timestamp=datetime.now(UTC),
            user_id=user_id,
            ip_address=ip_address,
            message=message,
            severity=severity,
            outcome="alert",
            details={"alert_type": alert_type, **(details or {})},
            correlation_id=correlation_id
        )
        self.log_event(event)


# Context manager for audit logging
@asynccontextmanager
async def audit_context(audit_logger: AuditLogger,
                       user_id: str,
                       action: str,
                       resource: str,
                       correlation_id: Optional[str] = None):
    """Context manager for auditing operations.
    
    Args:
        audit_logger: Audit logger instance
        user_id: User performing the action
        action: Action being performed
        resource: Resource being acted upon
        correlation_id: Correlation ID for request tracking
        
    Yields:
        Dictionary for collecting audit details
    """
    audit_details = {}
    start_time = datetime.now(UTC)
    
    try:
        yield audit_details
        
        # Log success
        audit_logger.log_data_access(
            event_type=AuditEventType.DATA_ACCESSED,
            user_id=user_id,
            resource=resource,
            action=action,
            outcome="success",
            details={
                "duration_ms": int((datetime.now(UTC) - start_time).total_seconds() * 1000),
                **audit_details
            },
            correlation_id=correlation_id
        )
        
    except Exception as e:
        # Log failure
        audit_logger.log_data_access(
            event_type=AuditEventType.DATA_ACCESSED,
            user_id=user_id,
            resource=resource,
            action=action,
            outcome="failure",
            details={
                "error": str(e),
                "duration_ms": int((datetime.now(UTC) - start_time).total_seconds() * 1000),
                **audit_details
            },
            correlation_id=correlation_id
        )
        raise


# Global audit logger instance
_audit_logger: Optional[AuditLogger] = None


def get_audit_logger() -> AuditLogger:
    """Get global audit logger instance.
    
    Returns:
        Audit logger instance
    """
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger


def init_audit_logging() -> AuditLogger:
    """Initialize audit logging system.
    
    Returns:
        Audit logger instance
    """
    global _audit_logger
    _audit_logger = AuditLogger()
    
    # Add event enrichment processors
    def add_environment_info(event: AuditEvent) -> AuditEvent:
        """Add environment information to events."""
        if event.details is None:
            event.details = {}
        event.details["environment"] = "production"  # Get from config
        event.details["service"] = "pynomaly"
        return event
    
    _audit_logger.add_event_processor(add_environment_info)
    
    return _audit_logger