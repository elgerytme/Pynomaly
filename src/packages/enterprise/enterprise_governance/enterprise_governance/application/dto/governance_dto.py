"""
Governance Data Transfer Objects (DTOs) for API communication.
"""

from datetime import datetime, date
from typing import Dict, List, Optional, Any, Union
from uuid import UUID

from pydantic import BaseModel, Field, validator


# Audit DTOs

class AuditLogRequest(BaseModel):
    """Request to create an audit log entry."""
    
    event_type: str = Field(..., description="Type of audit event")
    category: Optional[str] = Field(None, description="Event category")
    severity: Optional[str] = Field(None, description="Event severity")
    
    # Context
    user_id: Optional[UUID] = Field(None, description="User ID")
    session_id: Optional[UUID] = Field(None, description="Session ID")
    
    # Event details
    message: str = Field(..., description="Event description")
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional details")
    
    # Resource information
    resource_type: Optional[str] = Field(None, description="Resource type")
    resource_id: Optional[UUID] = Field(None, description="Resource ID")
    resource_name: Optional[str] = Field(None, description="Resource name")
    
    # Request context
    ip_address: Optional[str] = Field(None, description="Client IP address")
    user_agent: Optional[str] = Field(None, description="User agent")
    
    # Operational data
    operation: Optional[str] = Field(None, description="Operation performed")
    status_code: Optional[int] = Field(None, description="Status code")
    response_time_ms: Optional[float] = Field(None, description="Response time")
    
    # Compliance
    compliance_tags: List[str] = Field(default_factory=list, description="Compliance tags")


class AuditLogResponse(BaseModel):
    """Response with audit log information."""
    
    id: UUID
    event_type: str
    category: str
    severity: str
    tenant_id: UUID
    user_id: Optional[UUID]
    message: str
    timestamp: datetime
    status: str
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }


class AuditSearchRequest(BaseModel):
    """Request to search audit logs."""
    
    # Time range
    start_time: Optional[datetime] = Field(None, description="Search start time")
    end_time: Optional[datetime] = Field(None, description="Search end time")
    
    # Filters
    event_types: Optional[List[str]] = Field(None, description="Event types to include")
    severities: Optional[List[str]] = Field(None, description="Severities to include")
    categories: Optional[List[str]] = Field(None, description="Categories to include")
    user_id: Optional[UUID] = Field(None, description="Filter by user")
    resource_type: Optional[str] = Field(None, description="Filter by resource type")
    compliance_tags: Optional[List[str]] = Field(None, description="Compliance tags")
    
    # Search
    search_term: Optional[str] = Field(None, description="Text search term")
    
    # Pagination
    page: int = Field(default=1, ge=1, description="Page number")
    page_size: int = Field(default=50, ge=1, le=1000, description="Page size")
    
    # Sorting
    sort_by: str = Field(default="timestamp", description="Sort field")
    sort_order: str = Field(default="desc", description="Sort order")
    
    # Output options
    include_sensitive: bool = Field(default=False, description="Include sensitive data")
    export_format: Optional[str] = Field(None, description="Export format")


class AuditSearchResponse(BaseModel):
    """Response with audit search results."""
    
    logs: List[AuditLogResponse]
    total_count: int
    page: int
    page_size: int
    total_pages: int
    statistics: Optional[Dict[str, Any]] = None


class AuditReportRequest(BaseModel):
    """Request to generate audit report."""
    
    report_type: str = Field(default="compliance", description="Type of report")
    start_time: datetime = Field(..., description="Report start time")
    end_time: datetime = Field(..., description="Report end time")
    
    # Filters
    event_types: Optional[List[str]] = Field(None, description="Event types to include")
    compliance_frameworks: Optional[List[str]] = Field(None, description="Compliance frameworks")
    
    # Format options
    format: str = Field(default="pdf", description="Report format")
    include_details: bool = Field(default=True, description="Include detailed logs")
    include_statistics: bool = Field(default=True, description="Include statistics")


# Compliance DTOs

class ComplianceAssessmentRequest(BaseModel):
    """Request to create compliance assessment."""
    
    framework: str = Field(..., description="Compliance framework")
    assessment_name: str = Field(..., description="Assessment name")
    description: str = Field(..., description="Assessment description")
    scope: str = Field(..., description="Assessment scope")
    
    # Team
    lead_assessor: str = Field(..., description="Lead assessor")
    assessment_team: List[str] = Field(default_factory=list, description="Assessment team")
    external_auditor: Optional[str] = Field(None, description="External auditor")
    
    # Timeline
    start_date: Optional[date] = Field(None, description="Assessment start date")
    end_date: Optional[date] = Field(None, description="Assessment end date")
    report_due_date: Optional[date] = Field(None, description="Report due date")
    
    # Scope details
    systems_in_scope: List[str] = Field(default_factory=list, description="Systems in scope")
    exclusions: List[str] = Field(default_factory=list, description="Exclusions")


class ComplianceAssessmentResponse(BaseModel):
    """Response with compliance assessment information."""
    
    id: UUID
    framework: str
    assessment_name: str
    overall_status: str
    compliance_percentage: float
    
    # Counts
    total_controls: int
    controls_implemented: int
    controls_partial: int
    controls_not_implemented: int
    
    # Timeline
    start_date: date
    end_date: Optional[date]
    created_at: datetime
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            date: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }


class ControlUpdateRequest(BaseModel):
    """Request to update compliance control."""
    
    status: str = Field(..., description="Control status")
    implementation_notes: Optional[str] = Field(None, description="Implementation notes")
    evidence_refs: List[str] = Field(default_factory=list, description="Evidence references")
    responsible_party: Optional[str] = Field(None, description="Responsible party")
    
    @validator('status')
    def validate_status(cls, v):
        """Validate control status."""
        allowed_statuses = [
            "implemented", "not_implemented", "partially_implemented",
            "planned", "deferred", "not_applicable"
        ]
        if v not in allowed_statuses:
            raise ValueError(f'Status must be one of: {", ".join(allowed_statuses)}')
        return v


class ControlResponse(BaseModel):
    """Response with compliance control information."""
    
    id: UUID
    control_id: str
    control_number: str
    title: str
    description: str
    status: str
    risk_level: str
    implementation_date: Optional[date]
    last_reviewed_date: Optional[date]
    evidence_collected: List[str]
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            date: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }


class ComplianceReportRequest(BaseModel):
    """Request to generate compliance report."""
    
    assessment_id: UUID = Field(..., description="Assessment ID")
    report_type: str = Field(default="executive_summary", description="Report type")
    
    # Content options
    include_findings: bool = Field(default=True, description="Include findings")
    include_recommendations: bool = Field(default=True, description="Include recommendations")
    include_evidence: bool = Field(default=False, description="Include evidence references")
    
    # Format options
    format: str = Field(default="pdf", description="Report format")
    confidentiality_level: str = Field(default="internal", description="Confidentiality level")


# SLA DTOs

class SLARequest(BaseModel):
    """Request to create SLA."""
    
    name: str = Field(..., description="SLA name")
    sla_type: str = Field(..., description="SLA type")
    description: str = Field(..., description="SLA description")
    
    # Parties
    service_provider: str = Field(..., description="Service provider")
    service_consumer: str = Field(..., description="Service consumer")
    
    # Coverage
    services_covered: List[str] = Field(..., description="Services covered")
    service_hours: str = Field(default="24x7", description="Service hours")
    exclusions: List[str] = Field(default_factory=list, description="Exclusions")
    
    # Targets
    overall_target: float = Field(..., description="Overall SLA target")
    measurement_period: str = Field(default="monthly", description="Measurement period")
    
    # Contract
    effective_date: datetime = Field(..., description="Effective date")
    expiry_date: Optional[datetime] = Field(None, description="Expiry date")
    auto_renewal: bool = Field(default=False, description="Auto-renewal")
    
    # Reporting
    reporting_frequency: str = Field(default="monthly", description="Reporting frequency")
    notification_contacts: List[str] = Field(default_factory=list, description="Contacts")


class SLAResponse(BaseModel):
    """Response with SLA information."""
    
    id: UUID
    name: str
    sla_type: str
    status: str
    overall_target: float
    current_compliance: float
    effective_date: datetime
    expiry_date: Optional[datetime]
    is_active: bool
    
    # Performance
    metrics_count: int = Field(default=0, description="Number of metrics")
    violations_last_30_days: int = Field(default=0, description="Recent violations")
    credits_earned: float = Field(default=0.0, description="Service credits")
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }


class SLAMetricRequest(BaseModel):
    """Request to create SLA metric."""
    
    name: str = Field(..., description="Metric name")
    description: str = Field(..., description="Metric description")
    metric_type: str = Field(..., description="Metric type")
    
    # Targets
    target_value: float = Field(..., description="Target value")
    minimum_acceptable: float = Field(..., description="Minimum acceptable value")
    measurement_unit: str = Field(..., description="Unit of measurement")
    
    # Measurement
    measurement_frequency: str = Field(default="hourly", description="Measurement frequency")
    calculation_method: str = Field(default="average", description="Calculation method")
    data_source: str = Field(..., description="Data source")
    
    # Thresholds
    warning_threshold: Optional[float] = Field(None, description="Warning threshold")
    critical_threshold: Optional[float] = Field(None, description="Critical threshold")


class SLAMetricResponse(BaseModel):
    """Response with SLA metric information."""
    
    id: UUID
    name: str
    metric_type: str
    target_value: float
    current_value: Optional[float]
    measurement_unit: str
    is_meeting_target: bool
    compliance_percentage: float
    last_measured_at: Optional[datetime]
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }


class MetricMeasurementRequest(BaseModel):
    """Request to record metric measurement."""
    
    value: float = Field(..., description="Measured value")
    timestamp: Optional[datetime] = Field(None, description="Measurement timestamp")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")


class SLAViolationResponse(BaseModel):
    """Response with SLA violation information."""
    
    id: UUID
    sla_id: UUID
    metric_id: UUID
    violation_type: str
    severity: str
    description: str
    
    # Timing
    start_time: datetime
    end_time: Optional[datetime]
    duration_minutes: Optional[float]
    
    # Impact
    target_value: float
    actual_value: float
    deviation_percentage: float
    affected_services: List[str]
    
    # Status
    is_resolved: bool
    service_credits_due: float
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }


class SLAComplianceRequest(BaseModel):
    """Request to check SLA compliance."""
    
    sla_ids: Optional[List[UUID]] = Field(None, description="Specific SLAs to check")
    include_metrics: bool = Field(default=False, description="Include metric details")
    include_violations: bool = Field(default=False, description="Include recent violations")
    period_days: int = Field(default=30, description="Period to analyze")


class SLAComplianceResponse(BaseModel):
    """Response with SLA compliance information."""
    
    tenant_id: UUID
    total_slas: int
    active_slas: int
    compliant_slas: int
    violations_today: int
    overall_compliance: float
    
    # Details
    sla_details: List[Dict[str, Any]] = Field(default_factory=list)
    recent_violations: List[SLAViolationResponse] = Field(default_factory=list)
    
    # Summary metrics
    average_response_time: Optional[float] = None
    availability_percentage: Optional[float] = None
    total_credits_earned: float = Field(default=0.0)
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            UUID: lambda v: str(v),
        }


# Data Privacy DTOs

class DataPrivacyRequest(BaseModel):
    """Request to create data privacy record."""
    
    data_subject_id: str = Field(..., description="Data subject identifier")
    record_type: str = Field(..., description="Privacy record type")
    processing_purpose: str = Field(..., description="Data processing purpose")
    data_categories: List[str] = Field(..., description="Data categories")
    legal_basis: str = Field(..., description="Legal basis for processing")
    
    # Consent
    consent_given: bool = Field(default=False, description="Consent status")
    consent_method: Optional[str] = Field(None, description="How consent was obtained")
    
    # Retention
    retention_period: str = Field(..., description="Data retention period")
    retention_start_date: date = Field(..., description="Retention start date")
    
    # Sharing
    data_shared_with: List[str] = Field(default_factory=list, description="Third parties")
    transfer_countries: List[str] = Field(default_factory=list, description="Transfer countries")


class DataRightsRequest(BaseModel):
    """Request to exercise data subject rights."""
    
    data_subject_id: str = Field(..., description="Data subject identifier")
    request_type: str = Field(..., description="Type of rights request")
    description: Optional[str] = Field(None, description="Request description")
    
    @validator('request_type')
    def validate_request_type(cls, v):
        """Validate data rights request type."""
        allowed_types = ["access", "rectification", "erasure", "portability", "restrict_processing"]
        if v not in allowed_types:
            raise ValueError(f'Request type must be one of: {", ".join(allowed_types)}')
        return v


class DataPrivacyResponse(BaseModel):
    """Response with data privacy record information."""
    
    id: UUID
    data_subject_id: str
    record_type: str
    processing_purpose: str
    consent_given: bool
    retention_period: str
    scheduled_deletion_date: Optional[date]
    
    # Rights exercised
    access_requests_count: int = Field(default=0)
    rectification_requests_count: int = Field(default=0)
    erasure_requests_count: int = Field(default=0)
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            date: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }


# General Response DTOs

class GovernanceStatsResponse(BaseModel):
    """Response with governance statistics."""
    
    tenant_id: UUID
    
    # Audit stats
    audit_events_today: int = Field(default=0)
    security_events_today: int = Field(default=0)
    compliance_events_today: int = Field(default=0)
    
    # Compliance stats
    active_assessments: int = Field(default=0)
    compliance_average: float = Field(default=0.0)
    overdue_controls: int = Field(default=0)
    
    # SLA stats
    active_slas: int = Field(default=0)
    sla_compliance_average: float = Field(default=0.0)
    violations_last_7_days: int = Field(default=0)
    
    # Data privacy stats
    privacy_requests_pending: int = Field(default=0)
    data_retention_overdue: int = Field(default=0)
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            UUID: lambda v: str(v),
        }