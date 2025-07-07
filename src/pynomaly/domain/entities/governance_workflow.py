"""Model governance and approval workflow domain entities."""

from __future__ import annotations

from datetime import datetime, timedelta
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class ApprovalStatus(str, Enum):
    """Approval status enumeration."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    REQUIRES_CHANGES = "requires_changes"
    ESCALATED = "escalated"
    EXPIRED = "expired"
    WITHDRAWN = "withdrawn"


class WorkflowStatus(str, Enum):
    """Workflow status enumeration."""

    DRAFT = "draft"
    ACTIVE = "active"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class ComplianceStatus(str, Enum):
    """Compliance status enumeration."""

    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    NOT_ASSESSED = "not_assessed"
    UNDER_REVIEW = "under_review"


class ApprovalPriority(str, Enum):
    """Approval priority levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"
    CRITICAL = "critical"


class ApprovalType(str, Enum):
    """Types of approvals required."""

    TECHNICAL_REVIEW = "technical_review"
    SECURITY_REVIEW = "security_review"
    COMPLIANCE_REVIEW = "compliance_review"
    BUSINESS_REVIEW = "business_review"
    PERFORMANCE_REVIEW = "performance_review"
    DATA_QUALITY_REVIEW = "data_quality_review"
    ETHICAL_REVIEW = "ethical_review"
    LEGAL_REVIEW = "legal_review"
    FINAL_APPROVAL = "final_approval"


class ComplianceRule(BaseModel):
    """Compliance rule definition."""

    id: UUID = Field(default_factory=uuid4, description="Rule identifier")
    name: str = Field(..., description="Rule name")
    description: str = Field(..., description="Rule description")
    category: str = Field(
        ..., description="Rule category (security, privacy, performance, etc.)"
    )

    # Rule definition
    rule_type: str = Field(
        ..., description="Rule type (metric_threshold, data_requirement, etc.)"
    )
    parameters: dict[str, Any] = Field(..., description="Rule parameters")
    validation_script: str | None = Field(None, description="Custom validation script")

    # Requirements
    is_mandatory: bool = Field(True, description="Whether rule is mandatory")
    severity: str = Field(
        "medium", description="Rule severity (low, medium, high, critical)"
    )

    # Metadata
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="Creation timestamp"
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow, description="Last update"
    )
    created_by: str = Field(..., description="Rule creator")
    tags: list[str] = Field(default_factory=list, description="Rule tags")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class ComplianceViolation(BaseModel):
    """Compliance rule violation."""

    rule_id: UUID = Field(..., description="Violated rule identifier")
    rule_name: str = Field(..., description="Violated rule name")
    severity: str = Field(..., description="Violation severity")
    description: str = Field(..., description="Violation description")
    current_value: Any = Field(None, description="Current value that violates the rule")
    expected_value: Any = Field(None, description="Expected value per rule")
    remediation_suggestion: str | None = Field(
        None, description="Suggested remediation"
    )
    auto_fixable: bool = Field(False, description="Whether violation can be auto-fixed")


class ComplianceReport(BaseModel):
    """Compliance assessment report."""

    id: UUID = Field(default_factory=uuid4, description="Report identifier")
    model_id: UUID = Field(..., description="Model identifier")

    # Assessment results
    overall_status: ComplianceStatus = Field(
        ..., description="Overall compliance status"
    )
    compliance_score: float = Field(
        ..., ge=0, le=1, description="Compliance score (0-1)"
    )

    # Rule assessment
    total_rules: int = Field(..., description="Total number of rules assessed")
    passed_rules: int = Field(..., description="Number of rules passed")
    failed_rules: int = Field(..., description="Number of rules failed")
    skipped_rules: int = Field(..., description="Number of rules skipped")

    # Violations
    violations: list[ComplianceViolation] = Field(
        ..., description="Compliance violations"
    )
    critical_violations: int = Field(..., description="Number of critical violations")
    high_violations: int = Field(..., description="Number of high severity violations")

    # Assessment metadata
    assessment_start_time: datetime = Field(..., description="Assessment start time")
    assessment_end_time: datetime = Field(..., description="Assessment end time")
    assessment_duration: float = Field(
        ..., description="Assessment duration in seconds"
    )

    # Metadata
    created_by: str = Field(..., description="Assessment creator")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    def is_compliant(self) -> bool:
        """Check if model is compliant."""
        return self.overall_status == ComplianceStatus.COMPLIANT

    def has_critical_violations(self) -> bool:
        """Check if there are critical violations."""
        return self.critical_violations > 0

    def get_violation_summary(self) -> dict[str, int]:
        """Get summary of violations by severity."""
        summary = {"low": 0, "medium": 0, "high": 0, "critical": 0}
        for violation in self.violations:
            if violation.severity in summary:
                summary[violation.severity] += 1
        return summary


class ApprovalRequest(BaseModel):
    """Individual approval request within a workflow."""

    id: UUID = Field(default_factory=uuid4, description="Request identifier")
    workflow_id: UUID = Field(..., description="Parent workflow identifier")

    # Approval details
    approval_type: ApprovalType = Field(..., description="Type of approval required")
    approver_role: str = Field(..., description="Required approver role")
    approver_user: str | None = Field(None, description="Specific approver user")

    # Request content
    title: str = Field(..., description="Approval request title")
    description: str = Field(..., description="Detailed description")
    justification: str | None = Field(None, description="Justification for the request")

    # Supporting documents
    supporting_documents: list[str] = Field(
        default_factory=list, description="URLs to supporting documents"
    )
    compliance_report_id: UUID | None = Field(
        None, description="Associated compliance report"
    )

    # Status and timing
    status: ApprovalStatus = Field(
        default=ApprovalStatus.PENDING, description="Approval status"
    )
    priority: ApprovalPriority = Field(
        default=ApprovalPriority.MEDIUM, description="Request priority"
    )
    due_date: datetime | None = Field(None, description="Approval due date")

    # Response
    approved_by: str | None = Field(None, description="User who approved/rejected")
    approval_date: datetime | None = Field(None, description="Approval/rejection date")
    comments: str | None = Field(None, description="Approval comments")
    conditions: list[str] = Field(
        default_factory=list, description="Approval conditions"
    )

    # Metadata
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="Creation timestamp"
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow, description="Last update"
    )
    requested_by: str = Field(..., description="User who requested approval")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    def approve(
        self,
        approver: str,
        comments: str | None = None,
        conditions: list[str] | None = None,
    ) -> None:
        """Approve the request."""
        self.status = ApprovalStatus.APPROVED
        self.approved_by = approver
        self.approval_date = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        if comments:
            self.comments = comments
        if conditions:
            self.conditions = conditions

    def reject(self, approver: str, comments: str) -> None:
        """Reject the request."""
        self.status = ApprovalStatus.REJECTED
        self.approved_by = approver
        self.approval_date = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        self.comments = comments

    def require_changes(self, approver: str, comments: str) -> None:
        """Request changes."""
        self.status = ApprovalStatus.REQUIRES_CHANGES
        self.approved_by = approver
        self.approval_date = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        self.comments = comments

    def is_expired(self) -> bool:
        """Check if the approval request is expired."""
        if self.due_date and datetime.utcnow() > self.due_date:
            return True
        return False

    def is_pending(self) -> bool:
        """Check if the approval is still pending."""
        return self.status == ApprovalStatus.PENDING


class ApprovalWorkflowConfig(BaseModel):
    """Configuration for approval workflows."""

    name: str = Field(..., description="Workflow configuration name")
    description: str = Field(..., description="Workflow description")

    # Workflow steps
    required_approvals: list[ApprovalType] = Field(
        ..., description="Required approval types"
    )
    approval_order: list[ApprovalType] | None = Field(
        None, description="Required order of approvals (None for parallel)"
    )

    # Timing
    default_due_days: int = Field(7, description="Default days for approval due date")
    escalation_days: int = Field(3, description="Days before escalation")
    expiration_days: int = Field(30, description="Days before expiration")

    # Approval rules
    require_all_approvals: bool = Field(
        True, description="Require all approvals or just majority"
    )
    allow_self_approval: bool = Field(False, description="Allow self-approval")
    require_compliance_check: bool = Field(
        True, description="Require compliance assessment"
    )

    # Auto-approval rules
    auto_approval_rules: dict[str, Any] = Field(
        default_factory=dict, description="Conditions for auto-approval"
    )

    # Notification settings
    notify_on_submission: bool = Field(
        True, description="Notify on workflow submission"
    )
    notify_on_approval: bool = Field(True, description="Notify on each approval")
    notify_on_rejection: bool = Field(True, description="Notify on rejection")
    notify_on_completion: bool = Field(
        True, description="Notify on workflow completion"
    )

    # Metadata
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="Creation timestamp"
    )
    created_by: str = Field(..., description="Configuration creator")
    tags: list[str] = Field(default_factory=list, description="Configuration tags")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class ApprovalWorkflow(BaseModel):
    """Complete approval workflow for a model."""

    id: UUID = Field(default_factory=uuid4, description="Workflow identifier")
    model_id: UUID = Field(..., description="Model identifier")
    model_version: str | None = Field(None, description="Model version being approved")

    # Workflow configuration
    config: ApprovalWorkflowConfig = Field(..., description="Workflow configuration")

    # Status and progress
    status: WorkflowStatus = Field(
        default=WorkflowStatus.DRAFT, description="Workflow status"
    )
    current_step: int = Field(0, description="Current workflow step")
    total_steps: int = Field(..., description="Total workflow steps")

    # Approval requests
    approval_requests: list[ApprovalRequest] = Field(
        ..., description="Individual approval requests"
    )

    # Compliance
    compliance_report_id: UUID | None = Field(
        None, description="Associated compliance report"
    )
    compliance_required: bool = Field(
        True, description="Whether compliance check is required"
    )

    # Timing
    submitted_at: datetime | None = Field(None, description="Workflow submission time")
    completed_at: datetime | None = Field(None, description="Workflow completion time")
    due_date: datetime | None = Field(None, description="Workflow due date")

    # Results
    final_decision: ApprovalStatus | None = Field(
        None, description="Final workflow decision"
    )
    approved_conditions: list[str] = Field(
        default_factory=list, description="Conditions attached to approval"
    )

    # Metadata
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="Creation timestamp"
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow, description="Last update"
    )
    created_by: str = Field(..., description="Workflow creator")
    submitter: str | None = Field(None, description="Workflow submitter")
    tags: list[str] = Field(default_factory=list, description="Workflow tags")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    def submit_for_approval(self, submitter: str) -> None:
        """Submit workflow for approval."""
        if self.status != WorkflowStatus.DRAFT:
            raise ValueError("Can only submit draft workflows")

        self.status = WorkflowStatus.ACTIVE
        self.submitter = submitter
        self.submitted_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()

        # Set due date
        if self.config.default_due_days > 0:
            self.due_date = datetime.utcnow() + timedelta(
                days=self.config.default_due_days
            )

    def process_approval(
        self,
        request_id: UUID,
        status: ApprovalStatus,
        approver: str,
        comments: str | None = None,
        conditions: list[str] | None = None,
    ) -> None:
        """Process an individual approval."""
        request = next((r for r in self.approval_requests if r.id == request_id), None)
        if not request:
            raise ValueError(f"Approval request {request_id} not found")

        if status == ApprovalStatus.APPROVED:
            request.approve(approver, comments, conditions)
            if conditions:
                self.approved_conditions.extend(conditions)
        elif status == ApprovalStatus.REJECTED:
            request.reject(approver, comments or "No reason provided")
        elif status == ApprovalStatus.REQUIRES_CHANGES:
            request.require_changes(approver, comments or "Changes required")

        self.updated_at = datetime.utcnow()
        self._check_workflow_completion()

    def _check_workflow_completion(self) -> None:
        """Check if workflow is complete and update status."""
        pending_requests = [r for r in self.approval_requests if r.is_pending()]
        rejected_requests = [
            r for r in self.approval_requests if r.status == ApprovalStatus.REJECTED
        ]

        # If any request is rejected, workflow is rejected
        if rejected_requests:
            self.status = WorkflowStatus.COMPLETED
            self.final_decision = ApprovalStatus.REJECTED
            self.completed_at = datetime.utcnow()
            return

        # If no pending requests, workflow is approved
        if not pending_requests:
            self.status = WorkflowStatus.COMPLETED
            self.final_decision = ApprovalStatus.APPROVED
            self.completed_at = datetime.utcnow()
            return

    def get_pending_approvals(self) -> list[ApprovalRequest]:
        """Get pending approval requests."""
        return [r for r in self.approval_requests if r.is_pending()]

    def get_expired_approvals(self) -> list[ApprovalRequest]:
        """Get expired approval requests."""
        return [r for r in self.approval_requests if r.is_expired()]

    def is_complete(self) -> bool:
        """Check if workflow is complete."""
        return self.status == WorkflowStatus.COMPLETED

    def is_approved(self) -> bool:
        """Check if workflow is approved."""
        return self.final_decision == ApprovalStatus.APPROVED

    def is_rejected(self) -> bool:
        """Check if workflow is rejected."""
        return self.final_decision == ApprovalStatus.REJECTED

    def get_progress_percentage(self) -> float:
        """Get workflow progress percentage."""
        if not self.approval_requests:
            return 0.0

        completed_requests = len(
            [
                r
                for r in self.approval_requests
                if r.status in [ApprovalStatus.APPROVED, ApprovalStatus.REJECTED]
            ]
        )

        return (completed_requests / len(self.approval_requests)) * 100

    class Config:
        """Pydantic model configuration."""

        validate_assignment = True
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            timedelta: lambda v: v.total_seconds(),
        }
