"""
Audit Log domain entities for comprehensive audit trail management.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator


class AuditEventType(str, Enum):
    """Audit event type enumeration."""
    # Authentication Events
    USER_LOGIN = "user.login"
    USER_LOGOUT = "user.logout"
    USER_LOGIN_FAILED = "user.login_failed"
    USER_LOCKED = "user.locked"
    USER_PASSWORD_CHANGED = "user.password_changed"
    USER_MFA_ENABLED = "user.mfa_enabled"
    USER_MFA_DISABLED = "user.mfa_disabled"
    
    # User Management Events
    USER_CREATED = "user.created"
    USER_UPDATED = "user.updated"
    USER_DELETED = "user.deleted"
    USER_ACTIVATED = "user.activated"
    USER_DEACTIVATED = "user.deactivated"
    USER_ROLE_ASSIGNED = "user.role_assigned"
    USER_ROLE_REVOKED = "user.role_revoked"
    
    # Tenant Management Events
    TENANT_CREATED = "tenant.created"
    TENANT_UPDATED = "tenant.updated"
    TENANT_SUSPENDED = "tenant.suspended"
    TENANT_ACTIVATED = "tenant.activated"
    TENANT_PLAN_CHANGED = "tenant.plan_changed"
    
    # Data Events
    DATA_ACCESSED = "data.accessed"
    DATA_CREATED = "data.created"
    DATA_UPDATED = "data.updated"
    DATA_DELETED = "data.deleted"
    DATA_EXPORTED = "data.exported"
    DATA_IMPORTED = "data.imported"
    
    # System Events
    SYSTEM_BACKUP_CREATED = "system.backup_created"
    SYSTEM_RESTORE_COMPLETED = "system.restore_completed"
    SYSTEM_CONFIG_CHANGED = "system.config_changed"
    SYSTEM_MAINTENANCE_START = "system.maintenance_start"
    SYSTEM_MAINTENANCE_END = "system.maintenance_end"
    
    # Security Events
    SECURITY_BREACH_DETECTED = "security.breach_detected"
    SECURITY_POLICY_VIOLATED = "security.policy_violated"
    SECURITY_SCAN_COMPLETED = "security.scan_completed"
    SECURITY_INCIDENT_CREATED = "security.incident_created"
    
    # API Events
    API_ACCESS = "api.access"
    API_RATE_LIMITED = "api.rate_limited"
    API_KEY_CREATED = "api.key_created"
    API_KEY_REVOKED = "api.key_revoked"
    
    # Compliance Events
    COMPLIANCE_REPORT_GENERATED = "compliance.report_generated"
    COMPLIANCE_AUDIT_STARTED = "compliance.audit_started"
    COMPLIANCE_AUDIT_COMPLETED = "compliance.audit_completed"
    COMPLIANCE_VIOLATION_DETECTED = "compliance.violation_detected"


class AuditSeverity(str, Enum):
    """Audit event severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class AuditStatus(str, Enum):
    """Audit log entry status."""
    PENDING = "pending"
    PROCESSED = "processed"
    ARCHIVED = "archived"
    DELETED = "deleted"


class AuditLog(BaseModel):
    """
    Audit Log domain entity for tracking all system activities.
    
    Provides comprehensive audit trail for security, compliance,
    and operational monitoring purposes.
    """
    
    id: UUID = Field(default_factory=uuid4, description="Unique audit log identifier")
    
    # Event Classification
    event_type: AuditEventType = Field(..., description="Type of audit event")
    category: str = Field(..., description="Event category for grouping")
    severity: AuditSeverity = Field(..., description="Event severity level")
    
    # Event Context
    tenant_id: Optional[UUID] = Field(None, description="Associated tenant ID")
    user_id: Optional[UUID] = Field(None, description="User who performed the action")
    session_id: Optional[UUID] = Field(None, description="User session ID")
    
    # Event Details
    message: str = Field(..., description="Human-readable event description")
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional event details")
    
    # Resource Information
    resource_type: Optional[str] = Field(None, description="Type of resource affected")
    resource_id: Optional[UUID] = Field(None, description="ID of resource affected")
    resource_name: Optional[str] = Field(None, description="Name of resource affected")
    
    # Request Context
    ip_address: Optional[str] = Field(None, description="Client IP address")
    user_agent: Optional[str] = Field(None, description="Client user agent")
    request_id: Optional[str] = Field(None, description="Request correlation ID")
    
    # Operational Data
    operation: Optional[str] = Field(None, description="Operation performed")
    status_code: Optional[int] = Field(None, description="HTTP status code")
    response_time_ms: Optional[float] = Field(None, description="Response time in milliseconds")
    
    # Data Changes
    old_values: Optional[Dict[str, Any]] = Field(None, description="Previous values (for updates)")
    new_values: Optional[Dict[str, Any]] = Field(None, description="New values (for updates)")
    
    # Compliance and Regulatory
    compliance_tags: List[str] = Field(default_factory=list, description="Compliance framework tags")
    retention_policy: Optional[str] = Field(None, description="Data retention policy")
    
    # Metadata
    source_system: str = Field(..., description="System that generated the event")
    source_component: Optional[str] = Field(None, description="Component that generated the event")
    environment: str = Field(..., description="Environment (dev, staging, production)")
    
    # Timestamps
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Event timestamp")
    processed_at: Optional[datetime] = Field(None, description="When log was processed")
    
    # Status and Lifecycle
    status: AuditStatus = Field(default=AuditStatus.PENDING)
    checksum: Optional[str] = Field(None, description="Integrity checksum")
    encrypted: bool = Field(default=False, description="Whether sensitive data is encrypted")
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        validate_assignment = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }
    
    @validator('compliance_tags', pre=True)
    def validate_compliance_tags(cls, v):
        """Validate compliance tags."""
        if isinstance(v, str):
            return [v]
        return v or []
    
    @validator('ip_address')
    def validate_ip_address(cls, v):
        """Validate IP address format."""
        if v and not (v == "unknown" or "." in v or ":" in v):
            raise ValueError("Invalid IP address format")
        return v
    
    def add_compliance_tag(self, tag: str) -> None:
        """Add a compliance framework tag."""
        if tag not in self.compliance_tags:
            self.compliance_tags.append(tag)
    
    def remove_compliance_tag(self, tag: str) -> None:
        """Remove a compliance framework tag."""
        if tag in self.compliance_tags:
            self.compliance_tags.remove(tag)
    
    def is_security_event(self) -> bool:
        """Check if this is a security-related event."""
        security_events = [
            AuditEventType.USER_LOGIN_FAILED,
            AuditEventType.USER_LOCKED,
            AuditEventType.SECURITY_BREACH_DETECTED,
            AuditEventType.SECURITY_POLICY_VIOLATED,
            AuditEventType.SECURITY_INCIDENT_CREATED,
        ]
        return self.event_type in security_events
    
    def is_compliance_event(self) -> bool:
        """Check if this is a compliance-related event."""
        compliance_events = [
            AuditEventType.COMPLIANCE_REPORT_GENERATED,
            AuditEventType.COMPLIANCE_AUDIT_STARTED,
            AuditEventType.COMPLIANCE_AUDIT_COMPLETED,
            AuditEventType.COMPLIANCE_VIOLATION_DETECTED,
        ]
        return self.event_type in compliance_events
    
    def is_critical(self) -> bool:
        """Check if this is a critical severity event."""
        return self.severity == AuditSeverity.CRITICAL
    
    def requires_immediate_attention(self) -> bool:
        """Check if this event requires immediate attention."""
        return (
            self.severity in [AuditSeverity.CRITICAL, AuditSeverity.HIGH] or
            self.is_security_event()
        )
    
    def sanitize_for_export(self) -> Dict[str, Any]:
        """
        Sanitize audit log for external export.
        
        Removes sensitive information while preserving audit trail integrity.
        """
        export_data = self.dict()
        
        # Remove or redact sensitive fields
        sensitive_fields = ['old_values', 'new_values', 'details']
        for field in sensitive_fields:
            if field in export_data and export_data[field]:
                export_data[field] = "***REDACTED***"
        
        # Redact IP addresses if required
        if self.ip_address and len(self.ip_address.split('.')) == 4:
            parts = self.ip_address.split('.')
            export_data['ip_address'] = f"{parts[0]}.{parts[1]}.xxx.xxx"
        
        return export_data
    
    def calculate_checksum(self) -> str:
        """Calculate integrity checksum for tamper detection."""
        import hashlib
        import json
        
        # Create canonical representation
        checksum_data = {
            'id': str(self.id),
            'event_type': self.event_type,
            'timestamp': self.timestamp.isoformat(),
            'tenant_id': str(self.tenant_id) if self.tenant_id else None,
            'user_id': str(self.user_id) if self.user_id else None,
            'message': self.message,
            'resource_type': self.resource_type,
            'resource_id': str(self.resource_id) if self.resource_id else None,
        }
        
        canonical_json = json.dumps(checksum_data, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(canonical_json.encode()).hexdigest()
    
    def verify_integrity(self) -> bool:
        """Verify audit log integrity using checksum."""
        if not self.checksum:
            return False
        
        calculated_checksum = self.calculate_checksum()
        return calculated_checksum == self.checksum
    
    def mark_processed(self) -> None:
        """Mark audit log as processed."""
        self.status = AuditStatus.PROCESSED
        self.processed_at = datetime.utcnow()
        
        # Calculate and store checksum for integrity
        if not self.checksum:
            self.checksum = self.calculate_checksum()
    
    def archive(self) -> None:
        """Archive audit log for long-term retention."""
        self.status = AuditStatus.ARCHIVED
    
    def to_dict(self, include_sensitive: bool = False) -> Dict[str, Any]:
        """
        Convert audit log to dictionary.
        
        Args:
            include_sensitive: Whether to include sensitive information
        """
        if include_sensitive:
            return self.dict()
        else:
            return self.sanitize_for_export()


class AuditQuery(BaseModel):
    """
    Query object for searching audit logs with filtering and pagination.
    """
    
    # Time range filters
    start_time: Optional[datetime] = Field(None, description="Start time filter")
    end_time: Optional[datetime] = Field(None, description="End time filter")
    
    # Entity filters
    tenant_id: Optional[UUID] = Field(None, description="Filter by tenant")
    user_id: Optional[UUID] = Field(None, description="Filter by user")
    resource_type: Optional[str] = Field(None, description="Filter by resource type")
    resource_id: Optional[UUID] = Field(None, description="Filter by resource ID")
    
    # Event filters
    event_types: Optional[List[AuditEventType]] = Field(None, description="Filter by event types")
    severities: Optional[List[AuditSeverity]] = Field(None, description="Filter by severities")
    categories: Optional[List[str]] = Field(None, description="Filter by categories")
    compliance_tags: Optional[List[str]] = Field(None, description="Filter by compliance tags")
    
    # Context filters
    ip_address: Optional[str] = Field(None, description="Filter by IP address")
    session_id: Optional[UUID] = Field(None, description="Filter by session")
    request_id: Optional[str] = Field(None, description="Filter by request ID")
    
    # Text search
    search_term: Optional[str] = Field(None, description="Full-text search term")
    
    # Pagination
    page: int = Field(default=1, ge=1, description="Page number")
    page_size: int = Field(default=100, ge=1, le=1000, description="Items per page")
    
    # Sorting
    sort_by: str = Field(default="timestamp", description="Sort field")
    sort_order: str = Field(default="desc", description="Sort order (asc/desc)")
    
    # Output options
    include_sensitive: bool = Field(default=False, description="Include sensitive data")
    export_format: Optional[str] = Field(None, description="Export format (json, csv, xlsx)")
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }
    
    @validator('sort_order')
    def validate_sort_order(cls, v):
        """Validate sort order."""
        if v not in ['asc', 'desc']:
            raise ValueError('Sort order must be "asc" or "desc"')
        return v
    
    @validator('export_format')
    def validate_export_format(cls, v):
        """Validate export format."""
        if v and v not in ['json', 'csv', 'xlsx', 'pdf']:
            raise ValueError('Export format must be one of: json, csv, xlsx, pdf')
        return v


class AuditStatistics(BaseModel):
    """
    Audit statistics and metrics for reporting and monitoring.
    """
    
    # Time range
    start_time: datetime = Field(..., description="Statistics start time")
    end_time: datetime = Field(..., description="Statistics end time")
    
    # Overall counts
    total_events: int = Field(default=0, description="Total events in period")
    unique_users: int = Field(default=0, description="Unique users in period")
    unique_tenants: int = Field(default=0, description="Unique tenants in period")
    
    # Event type breakdown
    events_by_type: Dict[str, int] = Field(default_factory=dict)
    events_by_severity: Dict[str, int] = Field(default_factory=dict)
    events_by_category: Dict[str, int] = Field(default_factory=dict)
    
    # Security metrics
    security_events: int = Field(default=0, description="Security events count")
    failed_logins: int = Field(default=0, description="Failed login attempts")
    locked_accounts: int = Field(default=0, description="Accounts locked")
    
    # Compliance metrics
    compliance_events: int = Field(default=0, description="Compliance events count")
    policy_violations: int = Field(default=0, description="Policy violations")
    
    # System metrics
    api_calls: int = Field(default=0, description="API calls made")
    data_access_events: int = Field(default=0, description="Data access events")
    data_export_events: int = Field(default=0, description="Data export events")
    
    # Performance metrics
    average_response_time: Optional[float] = Field(None, description="Average response time")
    error_rate: Optional[float] = Field(None, description="Error rate percentage")
    
    # Top lists
    top_users_by_activity: List[Dict[str, Any]] = Field(default_factory=list)
    top_resources_accessed: List[Dict[str, Any]] = Field(default_factory=list)
    top_ip_addresses: List[Dict[str, Any]] = Field(default_factory=list)
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }


class AuditRetentionPolicy(BaseModel):
    """
    Audit log retention policy configuration.
    """
    
    id: UUID = Field(default_factory=uuid4)
    name: str = Field(..., description="Policy name")
    description: str = Field(..., description="Policy description")
    
    # Retention rules
    default_retention_days: int = Field(..., description="Default retention period in days")
    retention_rules: Dict[str, int] = Field(default_factory=dict, description="Event-specific retention")
    
    # Compliance requirements
    compliance_frameworks: List[str] = Field(default_factory=list)
    legal_hold_enabled: bool = Field(default=False, description="Legal hold override")
    
    # Archive settings
    archive_enabled: bool = Field(default=True, description="Enable archiving")
    archive_after_days: int = Field(..., description="Days before archiving")
    archive_location: Optional[str] = Field(None, description="Archive storage location")
    
    # Deletion settings
    permanent_delete_after_days: Optional[int] = Field(None, description="Days before permanent deletion")
    require_approval_for_deletion: bool = Field(default=True)
    
    # Status
    is_active: bool = Field(default=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    def get_retention_days(self, event_type: AuditEventType) -> int:
        """Get retention days for specific event type."""
        return self.retention_rules.get(event_type, self.default_retention_days)
    
    def should_archive(self, log_age_days: int) -> bool:
        """Check if log should be archived based on age."""
        return self.archive_enabled and log_age_days >= self.archive_after_days
    
    def should_delete(self, log_age_days: int) -> bool:
        """Check if log should be deleted based on age and policy."""
        if self.legal_hold_enabled:
            return False
        
        if self.permanent_delete_after_days:
            return log_age_days >= self.permanent_delete_after_days
        
        return False