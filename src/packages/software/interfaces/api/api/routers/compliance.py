"""
FastAPI router for compliance and audit logging.
"""

from datetime import datetime, timedelta
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, EmailStr

from interfaces.application.services.compliance_service import ComplianceService
from interfaces.domain.entities.compliance import (
    AuditAction,
    AuditSeverity,
    ComplianceFramework,
    DataClassification,
    RetentionPolicyStatus,
)
from interfaces.domain.entities.user import User
from interfaces.shared.exceptions import ValidationError
from interfaces.shared.types import TenantId, UserId

# Router setup
router = APIRouter(prefix="/api/compliance", tags=["Compliance & Audit"])


# Request/Response Models
class AuditEventResponse(BaseModel):
    id: str
    action: AuditAction
    severity: AuditSeverity
    timestamp: datetime
    user_id: str | None
    resource_type: str | None
    resource_id: str | None
    details: dict[str, Any]
    ip_address: str | None
    outcome: str
    risk_score: int
    compliance_frameworks: list[ComplianceFramework]
    is_high_risk: bool


class CreateRetentionPolicyRequest(BaseModel):
    name: str
    description: str
    data_type: str
    classification: DataClassification
    retention_period_days: int
    compliance_frameworks: list[ComplianceFramework]
    auto_delete: bool = True
    archive_before_delete: bool = True


class RetentionPolicyResponse(BaseModel):
    id: str
    name: str
    description: str
    data_type: str
    classification: DataClassification
    retention_period_days: int
    compliance_frameworks: list[ComplianceFramework]
    auto_delete: bool
    archive_before_delete: bool
    status: RetentionPolicyStatus
    created_at: datetime
    updated_at: datetime


class CreateGDPRRequestRequest(BaseModel):
    request_type: str  # "access", "rectification", "erasure", "portability", "restriction", "objection"
    data_subject_id: str
    data_subject_email: EmailStr
    request_details: str


class GDPRRequestResponse(BaseModel):
    id: str
    request_type: str
    data_subject_id: str
    data_subject_email: str
    request_details: str
    submitted_at: datetime
    status: str
    assigned_to: str | None
    completion_deadline: datetime | None
    processed_at: datetime | None
    is_overdue: bool
    notes: str


class ProcessGDPRRequestRequest(BaseModel):
    response_data: dict[str, Any] | None = None
    notes: str = ""


class ComplianceCheckResponse(BaseModel):
    id: str
    rule_id: str
    rule_name: str
    framework: ComplianceFramework
    status: str
    check_timestamp: datetime
    details: dict[str, Any]
    recommendations: list[str]
    next_check_due: datetime | None
    is_compliant: bool
    needs_attention: bool


class ComplianceReportResponse(BaseModel):
    id: str
    report_type: str
    framework: ComplianceFramework
    reporting_period_start: datetime
    reporting_period_end: datetime
    generated_at: datetime
    total_checks: int
    compliant_checks: int
    non_compliant_checks: int
    warning_checks: int
    compliance_score: float
    risk_level: str
    findings: list[ComplianceCheckResponse]
    recommendations: list[str]
    high_risk_events: int
    total_audit_events: int


class EncryptionKeyResponse(BaseModel):
    id: str
    key_name: str
    algorithm: str
    key_size: int
    purpose: str
    created_at: datetime
    expires_at: datetime | None
    status: str
    needs_rotation: bool


class CreateEncryptionKeyRequest(BaseModel):
    key_name: str
    algorithm: str
    key_size: int
    purpose: str


class BackupRecordResponse(BaseModel):
    id: str
    backup_type: str
    data_types: list[str]
    backup_location: str
    started_at: datetime
    completed_at: datetime | None
    status: str
    size_bytes: int
    compressed_size_bytes: int
    compression_ratio: float
    retention_until: datetime | None
    is_expired: bool


class CreateBackupRequest(BaseModel):
    backup_type: str
    data_types: list[str]
    backup_location: str
    encryption_key_id: str


# Dependencies
async def get_compliance_service() -> ComplianceService:
    """Get compliance service instance."""
    # TODO: Implement proper dependency injection
    pass


async def get_current_user() -> User:
    """Get current authenticated user."""
    # TODO: Implement authentication
    pass


async def require_compliance_access(
    tenant_id: UUID, current_user: User = Depends(get_current_user)
):
    """Require compliance access to specific tenant."""
    if not (
        current_user.is_super_admin()
        or current_user.has_role_in_tenant(TenantId(str(tenant_id)), ["tenant_admin"])
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied - requires compliance/admin permissions",
        )
    return current_user


# Audit Trail Endpoints
@router.get("/tenants/{tenant_id}/audit-trail", response_model=list[AuditEventResponse])
async def get_audit_trail(
    tenant_id: UUID,
    start_date: datetime | None = Query(None),
    end_date: datetime | None = Query(None),
    actions: list[AuditAction] | None = Query(None),
    user_id: UUID | None = Query(None),
    resource_type: str | None = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(require_compliance_access),
    compliance_service: ComplianceService = Depends(get_compliance_service),
):
    """Get audit trail for a tenant."""
    try:
        events = await compliance_service.get_audit_trail(
            tenant_id=TenantId(str(tenant_id)),
            start_date=start_date,
            end_date=end_date,
            actions=actions,
            user_id=UserId(str(user_id)) if user_id else None,
            resource_type=resource_type,
            limit=limit,
            offset=offset,
        )

        return [
            AuditEventResponse(
                id=event.id,
                action=event.action,
                severity=event.severity,
                timestamp=event.timestamp,
                user_id=str(event.user_id) if event.user_id else None,
                resource_type=event.resource_type,
                resource_id=event.resource_id,
                details=event.details,
                ip_address=event.ip_address,
                outcome=event.outcome,
                risk_score=event.risk_score,
                compliance_frameworks=event.compliance_frameworks,
                is_high_risk=event.is_high_risk,
            )
            for event in events
        ]
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve audit trail: {str(e)}",
        )


@router.get(
    "/tenants/{tenant_id}/audit-trail/high-risk",
    response_model=list[AuditEventResponse],
)
async def get_high_risk_events(
    tenant_id: UUID,
    days: int = Query(7, ge=1, le=365),
    current_user: User = Depends(require_compliance_access),
    compliance_service: ComplianceService = Depends(get_compliance_service),
):
    """Get high-risk audit events."""
    try:
        events = await compliance_service.get_high_risk_events(
            tenant_id=TenantId(str(tenant_id)), days=days
        )

        return [
            AuditEventResponse(
                id=event.id,
                action=event.action,
                severity=event.severity,
                timestamp=event.timestamp,
                user_id=str(event.user_id) if event.user_id else None,
                resource_type=event.resource_type,
                resource_id=event.resource_id,
                details=event.details,
                ip_address=event.ip_address,
                outcome=event.outcome,
                risk_score=event.risk_score,
                compliance_frameworks=event.compliance_frameworks,
                is_high_risk=event.is_high_risk,
            )
            for event in events
        ]
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve high-risk events: {str(e)}",
        )


# Data Retention Policy Endpoints
@router.post(
    "/tenants/{tenant_id}/retention-policies",
    response_model=RetentionPolicyResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_retention_policy(
    tenant_id: UUID,
    request: CreateRetentionPolicyRequest,
    current_user: User = Depends(require_compliance_access),
    compliance_service: ComplianceService = Depends(get_compliance_service),
):
    """Create a new data retention policy."""
    try:
        from interfaces.domain.entities.compliance import DataRetentionPolicy

        policy = DataRetentionPolicy(
            id="",  # Will be set by service
            name=request.name,
            description=request.description,
            tenant_id=TenantId(str(tenant_id)),
            data_type=request.data_type,
            classification=request.classification,
            retention_period_days=request.retention_period_days,
            compliance_frameworks=request.compliance_frameworks,
            auto_delete=request.auto_delete,
            archive_before_delete=request.archive_before_delete,
        )

        created_policy = await compliance_service.create_retention_policy(
            policy=policy, user_id=UserId(current_user.id)
        )

        return RetentionPolicyResponse(
            id=created_policy.id,
            name=created_policy.name,
            description=created_policy.description,
            data_type=created_policy.data_type,
            classification=created_policy.classification,
            retention_period_days=created_policy.retention_period_days,
            compliance_frameworks=created_policy.compliance_frameworks,
            auto_delete=created_policy.auto_delete,
            archive_before_delete=created_policy.archive_before_delete,
            status=created_policy.status,
            created_at=created_policy.created_at,
            updated_at=created_policy.updated_at,
        )
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.post("/tenants/{tenant_id}/retention-policies/apply")
async def apply_retention_policies(
    tenant_id: UUID,
    current_user: User = Depends(require_compliance_access),
    compliance_service: ComplianceService = Depends(get_compliance_service),
):
    """Apply all active retention policies for a tenant."""
    try:
        results = await compliance_service.apply_retention_policies(
            TenantId(str(tenant_id))
        )

        return {
            "message": "Retention policies applied successfully",
            "deleted_records": results["deleted_records"],
            "archived_records": results["archived_records"],
            "policies_applied": results["policies_applied"],
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to apply retention policies: {str(e)}",
        )


# GDPR Compliance Endpoints
@router.post(
    "/tenants/{tenant_id}/gdpr-requests",
    response_model=GDPRRequestResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_gdpr_request(
    tenant_id: UUID,
    request: CreateGDPRRequestRequest,
    current_user: User = Depends(require_compliance_access),
    compliance_service: ComplianceService = Depends(get_compliance_service),
):
    """Create a new GDPR data subject request."""
    try:
        gdpr_request = await compliance_service.create_gdpr_request(
            request_type=request.request_type,
            tenant_id=TenantId(str(tenant_id)),
            data_subject_id=request.data_subject_id,
            data_subject_email=request.data_subject_email,
            request_details=request.request_details,
            submitted_by=UserId(current_user.id),
        )

        return GDPRRequestResponse(
            id=gdpr_request.id,
            request_type=gdpr_request.request_type,
            data_subject_id=gdpr_request.data_subject_id,
            data_subject_email=gdpr_request.data_subject_email,
            request_details=gdpr_request.request_details,
            submitted_at=gdpr_request.submitted_at,
            status=gdpr_request.status,
            assigned_to=str(gdpr_request.assigned_to)
            if gdpr_request.assigned_to
            else None,
            completion_deadline=gdpr_request.completion_deadline,
            processed_at=gdpr_request.processed_at,
            is_overdue=gdpr_request.is_overdue,
            notes=gdpr_request.notes,
        )
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.put(
    "/tenants/{tenant_id}/gdpr-requests/{request_id}/process",
    response_model=GDPRRequestResponse,
)
async def process_gdpr_request(
    tenant_id: UUID,
    request_id: str,
    request: ProcessGDPRRequestRequest,
    current_user: User = Depends(require_compliance_access),
    compliance_service: ComplianceService = Depends(get_compliance_service),
):
    """Process a GDPR data subject request."""
    try:
        processed_request = await compliance_service.process_gdpr_request(
            request_id=request_id,
            processor_id=UserId(current_user.id),
            response_data=request.response_data,
        )

        # Update notes if provided
        if request.notes:
            processed_request.notes = request.notes

        return GDPRRequestResponse(
            id=processed_request.id,
            request_type=processed_request.request_type,
            data_subject_id=processed_request.data_subject_id,
            data_subject_email=processed_request.data_subject_email,
            request_details=processed_request.request_details,
            submitted_at=processed_request.submitted_at,
            status=processed_request.status,
            assigned_to=str(processed_request.assigned_to)
            if processed_request.assigned_to
            else None,
            completion_deadline=processed_request.completion_deadline,
            processed_at=processed_request.processed_at,
            is_overdue=processed_request.is_overdue,
            notes=processed_request.notes,
        )
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.get(
    "/tenants/{tenant_id}/gdpr-requests/overdue",
    response_model=list[GDPRRequestResponse],
)
async def get_overdue_gdpr_requests(
    tenant_id: UUID,
    current_user: User = Depends(require_compliance_access),
    compliance_service: ComplianceService = Depends(get_compliance_service),
):
    """Get GDPR requests that are overdue."""
    try:
        overdue_requests = await compliance_service.get_overdue_gdpr_requests(
            TenantId(str(tenant_id))
        )

        return [
            GDPRRequestResponse(
                id=req.id,
                request_type=req.request_type,
                data_subject_id=req.data_subject_id,
                data_subject_email=req.data_subject_email,
                request_details=req.request_details,
                submitted_at=req.submitted_at,
                status=req.status,
                assigned_to=str(req.assigned_to) if req.assigned_to else None,
                completion_deadline=req.completion_deadline,
                processed_at=req.processed_at,
                is_overdue=req.is_overdue,
                notes=req.notes,
            )
            for req in overdue_requests
        ]
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve overdue GDPR requests: {str(e)}",
        )


# Compliance Check Endpoints
@router.post(
    "/tenants/{tenant_id}/compliance-check", response_model=ComplianceReportResponse
)
async def run_compliance_check(
    tenant_id: UUID,
    framework: ComplianceFramework,
    current_user: User = Depends(require_compliance_access),
    compliance_service: ComplianceService = Depends(get_compliance_service),
):
    """Run a comprehensive compliance check."""
    try:
        report = await compliance_service.run_compliance_check(
            tenant_id=TenantId(str(tenant_id)),
            framework=framework,
            user_id=UserId(current_user.id),
        )

        findings = [
            ComplianceCheckResponse(
                id=check.id,
                rule_id=check.rule_id,
                rule_name=f"Rule {check.rule_id}",  # TODO: Get actual rule name
                framework=framework,
                status=check.status,
                check_timestamp=check.check_timestamp,
                details=check.details,
                recommendations=check.recommendations,
                next_check_due=check.next_check_due,
                is_compliant=check.is_compliant,
                needs_attention=check.needs_attention,
            )
            for check in report.findings
        ]

        return ComplianceReportResponse(
            id=report.id,
            report_type=report.report_type,
            framework=report.framework,
            reporting_period_start=report.reporting_period_start,
            reporting_period_end=report.reporting_period_end,
            generated_at=report.generated_at,
            total_checks=report.total_checks,
            compliant_checks=report.compliant_checks,
            non_compliant_checks=report.non_compliant_checks,
            warning_checks=report.warning_checks,
            compliance_score=report.compliance_score,
            risk_level=report.risk_level,
            findings=findings,
            recommendations=report.recommendations,
            high_risk_events=report.high_risk_events,
            total_audit_events=report.total_audit_events,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to run compliance check: {str(e)}",
        )


# Encryption Key Management Endpoints
@router.post(
    "/tenants/{tenant_id}/encryption-keys",
    response_model=EncryptionKeyResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_encryption_key(
    tenant_id: UUID,
    request: CreateEncryptionKeyRequest,
    current_user: User = Depends(require_compliance_access),
    compliance_service: ComplianceService = Depends(get_compliance_service),
):
    """Create a new encryption key."""
    try:
        key = await compliance_service.create_encryption_key(
            key_name=request.key_name,
            algorithm=request.algorithm,
            key_size=request.key_size,
            tenant_id=TenantId(str(tenant_id)),
            purpose=request.purpose,
            user_id=UserId(current_user.id),
        )

        return EncryptionKeyResponse(
            id=key.id,
            key_name=key.key_name,
            algorithm=key.algorithm,
            key_size=key.key_size,
            purpose=key.purpose,
            created_at=key.created_at,
            expires_at=key.expires_at,
            status=key.status,
            needs_rotation=key.needs_rotation,
        )
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.post("/tenants/{tenant_id}/encryption-keys/rotate")
async def rotate_encryption_keys(
    tenant_id: UUID,
    current_user: User = Depends(require_compliance_access),
    compliance_service: ComplianceService = Depends(get_compliance_service),
):
    """Rotate encryption keys that need rotation."""
    try:
        rotated_count = await compliance_service.rotate_encryption_keys(
            tenant_id=TenantId(str(tenant_id)), user_id=UserId(current_user.id)
        )

        return {
            "message": "Encryption key rotation completed",
            "rotated_keys": rotated_count,
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to rotate encryption keys: {str(e)}",
        )


# Backup Management Endpoints
@router.post(
    "/tenants/{tenant_id}/backups",
    response_model=BackupRecordResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_backup_record(
    tenant_id: UUID,
    request: CreateBackupRequest,
    current_user: User = Depends(require_compliance_access),
    compliance_service: ComplianceService = Depends(get_compliance_service),
):
    """Create a backup operation record."""
    try:
        backup = await compliance_service.create_backup_record(
            backup_type=request.backup_type,
            tenant_id=TenantId(str(tenant_id)),
            data_types=request.data_types,
            backup_location=request.backup_location,
            encryption_key_id=request.encryption_key_id,
            user_id=UserId(current_user.id),
        )

        return BackupRecordResponse(
            id=backup.id,
            backup_type=backup.backup_type,
            data_types=backup.data_types,
            backup_location=backup.backup_location,
            started_at=backup.started_at,
            completed_at=backup.completed_at,
            status=backup.status,
            size_bytes=backup.size_bytes,
            compressed_size_bytes=backup.compressed_size_bytes,
            compression_ratio=backup.compression_ratio,
            retention_until=backup.retention_until,
            is_expired=backup.is_expired,
        )
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


# Compliance Dashboard Endpoint
@router.get("/tenants/{tenant_id}/compliance/dashboard")
async def get_compliance_dashboard(
    tenant_id: UUID,
    current_user: User = Depends(require_compliance_access),
    compliance_service: ComplianceService = Depends(get_compliance_service),
):
    """Get compliance dashboard summary."""
    try:
        # Get recent audit events
        recent_events = await compliance_service.get_audit_trail(
            tenant_id=TenantId(str(tenant_id)),
            start_date=datetime.utcnow() - timedelta(days=7),
            limit=100,
        )

        # Get high-risk events
        high_risk_events = await compliance_service.get_high_risk_events(
            tenant_id=TenantId(str(tenant_id)), days=7
        )

        # Get overdue GDPR requests
        overdue_gdpr = await compliance_service.get_overdue_gdpr_requests(
            TenantId(str(tenant_id))
        )

        # Calculate summary statistics
        total_events = len(recent_events)
        high_risk_count = len(high_risk_events)
        failed_events = len([e for e in recent_events if e.outcome == "failure"])

        return {
            "summary": {
                "total_audit_events_7_days": total_events,
                "high_risk_events_7_days": high_risk_count,
                "failed_operations_7_days": failed_events,
                "overdue_gdpr_requests": len(overdue_gdpr),
                "compliance_status": "healthy"
                if high_risk_count < 5
                else "attention_needed",
            },
            "recent_high_risk_events": [
                {
                    "id": event.id,
                    "action": event.action.value,
                    "timestamp": event.timestamp.isoformat(),
                    "risk_score": event.risk_score,
                }
                for event in high_risk_events[:5]
            ],
            "alerts": [
                {
                    "type": "overdue_gdpr",
                    "count": len(overdue_gdpr),
                    "message": f"{len(overdue_gdpr)} GDPR requests are overdue",
                }
                if overdue_gdpr
                else None,
                {
                    "type": "high_risk_events",
                    "count": high_risk_count,
                    "message": f"{high_risk_count} high-risk events in the last 7 days",
                }
                if high_risk_count > 0
                else None,
            ],
            "recommendations": [
                "Review high-risk audit events for potential security issues",
                "Process overdue GDPR requests immediately",
                "Consider implementing additional access controls",
                "Schedule regular compliance reviews",
            ],
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate compliance dashboard: {str(e)}",
        )
