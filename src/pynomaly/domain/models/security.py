"""Enterprise security domain models for RBAC, audit trails, and compliance."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from uuid import UUID, uuid4

import numpy as np


class AuditEventType(Enum):
    """Types of audit events."""
    
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    MODEL_TRAINING = "model_training"
    MODEL_INFERENCE = "model_inference"
    DATA_ACCESS = "data_access"
    CONFIG_CHANGE = "config_change"
    PERMISSION_CHANGE = "permission_change"
    SECURITY_VIOLATION = "security_violation"
    SYSTEM_ERROR = "system_error"


class SecuritySeverity(Enum):
    """Security severity levels."""
    
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class PermissionType(Enum):
    """Types of permissions in the system."""
    
    # Read permissions
    READ_MODELS = "read_models"
    READ_DATASETS = "read_datasets"
    READ_METRICS = "read_metrics"
    READ_LOGS = "read_logs"
    READ_CONFIGS = "read_configs"
    READ_USERS = "read_users"
    
    # Write permissions
    WRITE_MODELS = "write_models"
    WRITE_DATASETS = "write_datasets"
    WRITE_CONFIGS = "write_configs"
    WRITE_USERS = "write_users"
    
    # Execute permissions
    EXECUTE_TRAINING = "execute_training"
    EXECUTE_INFERENCE = "execute_inference"
    EXECUTE_DEPLOYMENT = "execute_deployment"
    EXECUTE_ANALYSIS = "execute_analysis"
    
    # Admin permissions
    ADMIN_SYSTEM = "admin_system"
    ADMIN_SECURITY = "admin_security"
    ADMIN_AUDIT = "admin_audit"
    ADMIN_COMPLIANCE = "admin_compliance"
    
    # Special permissions
    EXPORT_DATA = "export_data"
    DELETE_DATA = "delete_data"
    MANAGE_SECRETS = "manage_secrets"
    VIEW_SENSITIVE_DATA = "view_sensitive_data"


class UserRole(Enum):
    """User roles with predefined permission sets."""
    
    ADMIN = "admin"
    DATA_SCIENTIST = "data_scientist"
    ML_ENGINEER = "ml_engineer"
    ANALYST = "analyst"
    VIEWER = "viewer"
    AUDITOR = "auditor"
    COMPLIANCE_OFFICER = "compliance_officer"
    GUEST = "guest"


class ActionType(Enum):
    """Types of actions for audit logging."""
    
    # Authentication actions
    LOGIN = "login"
    LOGOUT = "logout"
    LOGIN_FAILED = "login_failed"
    PASSWORD_CHANGED = "password_changed"
    
    # Data actions
    DATA_CREATED = "data_created"
    DATA_READ = "data_read"
    DATA_UPDATED = "data_updated"
    DATA_DELETED = "data_deleted"
    DATA_EXPORTED = "data_exported"
    
    # Model actions
    MODEL_TRAINED = "model_trained"
    MODEL_DEPLOYED = "model_deployed"
    MODEL_INFERENCE = "model_inference"
    MODEL_DELETED = "model_deleted"
    
    # Configuration actions
    CONFIG_CREATED = "config_created"
    CONFIG_UPDATED = "config_updated"
    CONFIG_DELETED = "config_deleted"
    
    # Admin actions
    USER_CREATED = "user_created"
    USER_UPDATED = "user_updated"
    USER_DELETED = "user_deleted"
    ROLE_ASSIGNED = "role_assigned"
    PERMISSION_GRANTED = "permission_granted"
    PERMISSION_REVOKED = "permission_revoked"
    
    # Security actions
    SECRET_ACCESSED = "secret_accessed"
    SENSITIVE_DATA_ACCESSED = "sensitive_data_accessed"
    SECURITY_POLICY_UPDATED = "security_policy_updated"
    
    # System actions
    SYSTEM_BACKUP = "system_backup"
    SYSTEM_RESTORE = "system_restore"
    SYSTEM_MAINTENANCE = "system_maintenance"


class ComplianceFramework(Enum):
    """Supported compliance frameworks."""
    
    GDPR = "gdpr"  # General Data Protection Regulation
    HIPAA = "hipaa"  # Health Insurance Portability and Accountability Act
    SOX = "sox"  # Sarbanes-Oxley Act
    PCI_DSS = "pci_dss"  # Payment Card Industry Data Security Standard
    CCPA = "ccpa"  # California Consumer Privacy Act
    SOC2 = "soc2"  # Service Organization Control 2
    ISO27001 = "iso27001"  # ISO/IEC 27001
    NIST = "nist"  # NIST Cybersecurity Framework


class SecurityFramework(Enum):
    """Security frameworks and standards."""
    
    OWASP = "owasp"  # OWASP Security Framework
    NIST_CSF = "nist_csf"  # NIST Cybersecurity Framework
    ISO27001 = "iso27001"  # ISO/IEC 27001
    CIS_CONTROLS = "cis_controls"  # CIS Critical Security Controls
    SANS_TOP25 = "sans_top25"  # SANS Top 25 Software Errors
    ZERO_TRUST = "zero_trust"  # Zero Trust Architecture


@dataclass
class User:
    """User entity with security attributes."""
    
    user_id: UUID
    username: str
    email: str
    
    # Authentication
    password_hash: str
    salt: str
    mfa_enabled: bool = False
    mfa_secret: Optional[str] = None
    
    # Profile
    first_name: str = ""
    last_name: str = ""
    department: Optional[str] = None
    job_title: Optional[str] = None
    
    # Account status
    is_active: bool = True
    is_verified: bool = False
    is_locked: bool = False
    failed_login_attempts: int = 0
    
    # Roles and permissions
    roles: Set[UserRole] = field(default_factory=set)
    custom_permissions: Set[PermissionType] = field(default_factory=set)
    
    # Session management
    last_login: Optional[datetime] = None
    last_activity: Optional[datetime] = None
    session_timeout: int = 3600  # seconds
    
    # Security settings
    password_expires_at: Optional[datetime] = None
    must_change_password: bool = False
    allowed_ip_ranges: List[str] = field(default_factory=list)
    
    # Audit fields
    created_at: datetime = field(default_factory=datetime.utcnow)
    created_by: Optional[UUID] = None
    updated_at: datetime = field(default_factory=datetime.utcnow)
    updated_by: Optional[UUID] = None
    
    def __post_init__(self):
        if not self.username:
            raise ValueError("Username cannot be empty")
        if not self.email:
            raise ValueError("Email cannot be empty")
        if self.failed_login_attempts < 0:
            raise ValueError("Failed login attempts must be non-negative")
    
    def get_all_permissions(self) -> Set[PermissionType]:
        """Get all permissions from roles and custom permissions."""
        permissions = set(self.custom_permissions)
        
        # Add role-based permissions
        for role in self.roles:
            permissions.update(self._get_role_permissions(role))
        
        return permissions
    
    def has_permission(self, permission: PermissionType) -> bool:
        """Check if user has specific permission."""
        return permission in self.get_all_permissions()
    
    def is_session_valid(self) -> bool:
        """Check if user session is still valid."""
        if not self.is_active or self.is_locked:
            return False
        
        if self.last_activity:
            time_since_activity = (datetime.utcnow() - self.last_activity).total_seconds()
            return time_since_activity < self.session_timeout
        
        return False
    
    def _get_role_permissions(self, role: UserRole) -> Set[PermissionType]:
        """Get permissions for a specific role."""
        role_permissions = {
            UserRole.ADMIN: {
                PermissionType.ADMIN_SYSTEM,
                PermissionType.ADMIN_SECURITY,
                PermissionType.ADMIN_AUDIT,
                PermissionType.READ_MODELS,
                PermissionType.WRITE_MODELS,
                PermissionType.READ_DATASETS,
                PermissionType.WRITE_DATASETS,
                PermissionType.EXECUTE_TRAINING,
                PermissionType.EXECUTE_INFERENCE,
                PermissionType.EXECUTE_DEPLOYMENT,
                PermissionType.MANAGE_SECRETS,
                PermissionType.VIEW_SENSITIVE_DATA,
            },
            UserRole.DATA_SCIENTIST: {
                PermissionType.READ_MODELS,
                PermissionType.WRITE_MODELS,
                PermissionType.READ_DATASETS,
                PermissionType.WRITE_DATASETS,
                PermissionType.EXECUTE_TRAINING,
                PermissionType.EXECUTE_INFERENCE,
                PermissionType.EXECUTE_ANALYSIS,
                PermissionType.READ_METRICS,
            },
            UserRole.ML_ENGINEER: {
                PermissionType.READ_MODELS,
                PermissionType.WRITE_MODELS,
                PermissionType.EXECUTE_DEPLOYMENT,
                PermissionType.EXECUTE_INFERENCE,
                PermissionType.READ_CONFIGS,
                PermissionType.WRITE_CONFIGS,
                PermissionType.READ_METRICS,
            },
            UserRole.ANALYST: {
                PermissionType.READ_MODELS,
                PermissionType.READ_DATASETS,
                PermissionType.EXECUTE_ANALYSIS,
                PermissionType.READ_METRICS,
                PermissionType.EXPORT_DATA,
            },
            UserRole.VIEWER: {
                PermissionType.READ_MODELS,
                PermissionType.READ_DATASETS,
                PermissionType.READ_METRICS,
            },
            UserRole.AUDITOR: {
                PermissionType.READ_LOGS,
                PermissionType.READ_METRICS,
                PermissionType.ADMIN_AUDIT,
                PermissionType.VIEW_SENSITIVE_DATA,
            },
            UserRole.COMPLIANCE_OFFICER: {
                PermissionType.READ_LOGS,
                PermissionType.ADMIN_COMPLIANCE,
                PermissionType.ADMIN_AUDIT,
                PermissionType.VIEW_SENSITIVE_DATA,
            },
            UserRole.GUEST: {
                PermissionType.READ_METRICS,
            },
        }
        
        return role_permissions.get(role, set())


@dataclass
class AuditEvent:
    """Audit event for security and compliance tracking."""
    
    event_id: UUID
    user_id: Optional[UUID]  # None for system events
    username: Optional[str]
    
    # Event details
    action: ActionType
    resource_type: str  # e.g., "model", "dataset", "user"
    resource_id: Optional[str]
    resource_name: Optional[str]
    
    # Context information
    ip_address: Optional[str]
    user_agent: Optional[str]
    session_id: Optional[str]
    request_id: Optional[str]
    
    # Event outcome
    success: bool
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    
    # Additional data
    additional_data: Dict[str, Any] = field(default_factory=dict)
    
    # Security context
    security_level: str = "INFO"  # INFO, WARNING, CRITICAL
    compliance_relevant: bool = False
    requires_retention: bool = True
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.utcnow)
    server_timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        if self.security_level not in ["INFO", "WARNING", "CRITICAL"]:
            raise ValueError("Security level must be INFO, WARNING, or CRITICAL")
    
    def is_security_event(self) -> bool:
        """Check if this is a security-relevant event."""
        security_actions = {
            ActionType.LOGIN_FAILED,
            ActionType.SECRET_ACCESSED,
            ActionType.SENSITIVE_DATA_ACCESSED,
            ActionType.SECURITY_POLICY_UPDATED,
            ActionType.PERMISSION_GRANTED,
            ActionType.PERMISSION_REVOKED,
        }
        return self.action in security_actions or self.security_level in ["WARNING", "CRITICAL"]
    
    def get_retention_period_days(self) -> int:
        """Get retention period based on event type and compliance requirements."""
        if not self.requires_retention:
            return 30  # Default short retention
        
        if self.is_security_event():
            return 2555  # 7 years for security events
        elif self.compliance_relevant:
            return 2190  # 6 years for compliance events
        else:
            return 365  # 1 year for regular audit events


@dataclass
class SecurityPolicy:
    """Security policy configuration."""
    
    policy_id: UUID
    policy_name: str
    description: str
    
    # Password policy
    password_min_length: int = 12
    password_require_uppercase: bool = True
    password_require_lowercase: bool = True
    password_require_numbers: bool = True
    password_require_special_chars: bool = True
    password_expiry_days: int = 90
    password_history_size: int = 5
    
    # Account lockout policy
    max_failed_login_attempts: int = 5
    account_lockout_duration: int = 1800  # seconds
    
    # Session policy
    session_timeout: int = 3600  # seconds
    max_concurrent_sessions: int = 3
    session_idle_timeout: int = 1800  # seconds
    
    # MFA policy
    mfa_required_for_admins: bool = True
    mfa_required_for_sensitive_actions: bool = True
    mfa_methods_allowed: List[str] = field(default_factory=lambda: ["totp", "sms", "email"])
    
    # Access control
    ip_whitelist_enabled: bool = False
    allowed_ip_ranges: List[str] = field(default_factory=list)
    geo_blocking_enabled: bool = False
    blocked_countries: List[str] = field(default_factory=list)
    
    # Data protection
    encryption_at_rest: bool = True
    encryption_in_transit: bool = True
    data_masking_enabled: bool = True
    pii_detection_enabled: bool = True
    
    # Audit and compliance
    audit_all_actions: bool = True
    compliance_frameworks: List[ComplianceFramework] = field(default_factory=list)
    retention_period_days: int = 2555  # 7 years default
    
    # API security
    rate_limiting_enabled: bool = True
    max_requests_per_minute: int = 1000
    api_key_required: bool = True
    cors_enabled: bool = False
    allowed_origins: List[str] = field(default_factory=list)
    
    # Security monitoring
    anomaly_detection_enabled: bool = True
    threat_detection_enabled: bool = True
    security_alerts_enabled: bool = True
    
    # Policy metadata
    version: str = "1.0"
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    created_by: UUID = field(default_factory=uuid4)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    updated_by: UUID = field(default_factory=uuid4)
    
    def __post_init__(self):
        if self.password_min_length < 8:
            raise ValueError("Password minimum length must be at least 8")
        if self.max_failed_login_attempts < 1:
            raise ValueError("Max failed login attempts must be at least 1")
        if self.session_timeout < 60:
            raise ValueError("Session timeout must be at least 60 seconds")


@dataclass
class AccessRequest:
    """Access request for elevated permissions or resources."""
    
    request_id: UUID
    requester_id: UUID
    requester_username: str
    
    # Request details
    requested_permission: PermissionType
    resource_type: str
    resource_id: Optional[str]
    justification: str
    
    # Approval workflow
    approver_id: Optional[UUID] = None
    approver_username: Optional[str] = None
    approval_status: str = "pending"  # pending, approved, rejected, expired
    approval_timestamp: Optional[datetime] = None
    approval_comments: Optional[str] = None
    
    # Time bounds
    requested_start_time: Optional[datetime] = None
    requested_end_time: Optional[datetime] = None
    granted_start_time: Optional[datetime] = None
    granted_end_time: Optional[datetime] = None
    
    # Request metadata
    request_timestamp: datetime = field(default_factory=datetime.utcnow)
    urgency: str = "normal"  # low, normal, high, critical
    business_context: Optional[str] = None
    
    def __post_init__(self):
        if self.approval_status not in ["pending", "approved", "rejected", "expired"]:
            raise ValueError("Invalid approval status")
        if self.urgency not in ["low", "normal", "high", "critical"]:
            raise ValueError("Invalid urgency level")
        if not self.justification.strip():
            raise ValueError("Justification cannot be empty")
    
    def is_expired(self) -> bool:
        """Check if access request has expired."""
        if self.granted_end_time:
            return datetime.utcnow() > self.granted_end_time
        return False
    
    def is_active(self) -> bool:
        """Check if access is currently active."""
        if self.approval_status != "approved":
            return False
        
        now = datetime.utcnow()
        
        if self.granted_start_time and now < self.granted_start_time:
            return False
        
        if self.granted_end_time and now > self.granted_end_time:
            return False
        
        return True


@dataclass
class SecurityIncident:
    """Security incident record."""
    
    incident_id: UUID
    incident_type: str  # e.g., "unauthorized_access", "data_breach", "malware"
    title: str
    description: str
    severity: str = "medium"  # low, medium, high, critical
    affected_systems: List[str] = field(default_factory=list)
    affected_users: List[UUID] = field(default_factory=list)
    affected_data: List[str] = field(default_factory=list)
    
    # Detection
    detected_at: datetime = field(default_factory=datetime.utcnow)
    detected_by: Optional[UUID] = None  # User ID or system
    detection_method: str = "manual"  # manual, automated, third_party
    
    # Response
    response_status: str = "open"  # open, investigating, contained, resolved, closed
    assigned_to: Optional[UUID] = None
    response_started_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    
    # Impact assessment
    confidentiality_impact: str = "none"  # none, low, medium, high
    integrity_impact: str = "none"
    availability_impact: str = "none"
    financial_impact: Optional[float] = None
    
    # Root cause and remediation
    root_cause: Optional[str] = None
    remediation_actions: List[str] = field(default_factory=list)
    lessons_learned: Optional[str] = None
    
    # Compliance and reporting
    regulatory_notification_required: bool = False
    regulatory_notification_sent: bool = False
    customer_notification_required: bool = False
    customer_notification_sent: bool = False
    
    # Incident metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        valid_severities = ["low", "medium", "high", "critical"]
        if self.severity not in valid_severities:
            raise ValueError(f"Severity must be one of: {valid_severities}")
        
        valid_statuses = ["open", "investigating", "contained", "resolved", "closed"]
        if self.response_status not in valid_statuses:
            raise ValueError(f"Response status must be one of: {valid_statuses}")
        
        valid_impacts = ["none", "low", "medium", "high"]
        if self.confidentiality_impact not in valid_impacts:
            raise ValueError(f"Confidentiality impact must be one of: {valid_impacts}")
    
    def get_overall_impact_score(self) -> int:
        """Calculate overall impact score (0-12)."""
        impact_values = {"none": 0, "low": 1, "medium": 2, "high": 3}
        
        confidentiality_score = impact_values.get(self.confidentiality_impact, 0)
        integrity_score = impact_values.get(self.integrity_impact, 0)
        availability_score = impact_values.get(self.availability_impact, 0)
        
        return confidentiality_score + integrity_score + availability_score
    
    def is_major_incident(self) -> bool:
        """Check if this is a major security incident."""
        return (
            self.severity in ["high", "critical"] or
            self.get_overall_impact_score() >= 6 or
            self.regulatory_notification_required
        )


@dataclass
class ComplianceReport:
    """Compliance assessment report."""
    
    report_id: UUID
    framework: ComplianceFramework
    assessment_period_start: datetime
    assessment_period_end: datetime
    
    # Report metadata
    generated_at: datetime = field(default_factory=datetime.utcnow)
    generated_by: UUID = field(default_factory=uuid4)
    report_version: str = "1.0"
    
    # Compliance status
    overall_compliance_score: float = 0.0  # 0-100%
    compliant_controls: int = 0
    non_compliant_controls: int = 0
    total_controls: int = 0
    
    # Control assessments
    control_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Findings and recommendations
    critical_findings: List[str] = field(default_factory=list)
    high_findings: List[str] = field(default_factory=list)
    medium_findings: List[str] = field(default_factory=list)
    low_findings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    # Risk assessment
    overall_risk_level: str = "medium"  # low, medium, high, critical
    identified_risks: List[Dict[str, Any]] = field(default_factory=list)
    
    # Evidence and documentation
    evidence_collected: List[str] = field(default_factory=list)
    documentation_gaps: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not 0 <= self.overall_compliance_score <= 100:
            raise ValueError("Overall compliance score must be between 0 and 100")
        
        valid_risk_levels = ["low", "medium", "high", "critical"]
        if self.overall_risk_level not in valid_risk_levels:
            raise ValueError(f"Overall risk level must be one of: {valid_risk_levels}")
    
    def calculate_compliance_score(self) -> float:
        """Calculate overall compliance score based on control results."""
        if self.total_controls == 0:
            return 0.0
        
        return (self.compliant_controls / self.total_controls) * 100
    
    def get_critical_control_failures(self) -> List[str]:
        """Get list of critical control failures."""
        critical_failures = []
        
        for control_id, result in self.control_results.items():
            if result.get("status") == "failed" and result.get("criticality") == "critical":
                critical_failures.append(control_id)
        
        return critical_failures