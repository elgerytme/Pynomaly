"""Model governance and approval workflow service."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID

from pynomaly.domain.entities.governance_workflow import (
    ApprovalPriority,
    ApprovalRequest,
    ApprovalStatus,
    ApprovalType,
    ApprovalWorkflow,
    ApprovalWorkflowConfig,
    ComplianceReport,
    ComplianceRule,
    ComplianceStatus,
    ComplianceViolation,
    WorkflowStatus,
)
from pynomaly.shared.protocols.repository_protocol import (
    ModelRepositoryProtocol,
)


class ModelGovernanceService:
    """Service for model governance and approval workflows."""

    def __init__(
        self,
        model_repository: ModelRepositoryProtocol,
        governance_repository: Any,  # GovernanceRepositoryProtocol when implemented
        notification_service: Any,  # NotificationService when implemented
    ):
        """Initialize the governance service.
        
        Args:
            model_repository: Model repository
            governance_repository: Governance repository
            notification_service: Notification service
        """
        self.model_repository = model_repository
        self.governance_repository = governance_repository
        self.notification_service = notification_service

    async def create_approval_workflow(
        self,
        model_id: UUID,
        config: ApprovalWorkflowConfig,
        created_by: str,
        model_version: str | None = None,
        tags: list[str] | None = None,
    ) -> ApprovalWorkflow:
        """Create an approval workflow for a model.
        
        Args:
            model_id: Model identifier
            config: Workflow configuration
            created_by: User creating the workflow
            model_version: Model version being approved
            tags: Workflow tags
            
        Returns:
            Created approval workflow
        """
        # Validate model exists
        model = await self.model_repository.get_by_id(model_id)
        if not model:
            raise ValueError(f"Model {model_id} does not exist")
        
        # Create approval requests
        approval_requests = []
        for approval_type in config.required_approvals:
            # Calculate due date
            due_date = datetime.utcnow() + timedelta(days=config.default_due_days)
            
            # Determine approver role based on approval type
            approver_role = self._get_approver_role(approval_type)
            
            approval_request = ApprovalRequest(
                workflow_id=UUID("00000000-0000-0000-0000-000000000000"),  # Will be set later
                approval_type=approval_type,
                approver_role=approver_role,
                title=f"{approval_type.value.replace('_', ' ').title()} for {model.name}",
                description=f"Please review and approve {model.name} for {approval_type.value}",
                due_date=due_date,
                requested_by=created_by,
            )
            approval_requests.append(approval_request)
        
        # Create workflow
        workflow = ApprovalWorkflow(
            model_id=model_id,
            model_version=model_version,
            config=config,
            total_steps=len(approval_requests),
            approval_requests=approval_requests,
            compliance_required=config.require_compliance_check,
            created_by=created_by,
            tags=tags or [],
        )
        
        # Update request workflow IDs
        for request in workflow.approval_requests:
            request.workflow_id = workflow.id
        
        # Store workflow
        stored_workflow = await self.governance_repository.create_workflow(workflow)
        
        return stored_workflow

    async def submit_workflow_for_approval(
        self, workflow_id: UUID, submitter: str
    ) -> ApprovalWorkflow:
        """Submit workflow for approval.
        
        Args:
            workflow_id: Workflow identifier
            submitter: User submitting the workflow
            
        Returns:
            Updated workflow
        """
        workflow = await self.governance_repository.get_workflow(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        # Run compliance check if required
        if workflow.compliance_required:
            compliance_report = await self.run_compliance_check(workflow.model_id)
            workflow.compliance_report_id = compliance_report.id
            
            # Check if compliance allows submission
            if not compliance_report.is_compliant() and compliance_report.has_critical_violations():
                raise ValueError("Cannot submit workflow with critical compliance violations")
        
        # Submit workflow
        workflow.submit_for_approval(submitter)
        
        # Update workflow
        updated_workflow = await self.governance_repository.update_workflow(workflow)
        
        # Send notifications
        if workflow.config.notify_on_submission:
            await self._send_submission_notifications(workflow)
        
        return updated_workflow

    async def process_approval(
        self,
        workflow_id: UUID,
        request_id: UUID,
        status: ApprovalStatus,
        approver: str,
        comments: str | None = None,
        conditions: list[str] | None = None,
    ) -> ApprovalWorkflow:
        """Process an approval request.
        
        Args:
            workflow_id: Workflow identifier
            request_id: Approval request identifier
            status: Approval status
            approver: User providing approval
            comments: Approval comments
            conditions: Approval conditions
            
        Returns:
            Updated workflow
        """
        workflow = await self.governance_repository.get_workflow(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        # Validate approver has permission
        await self._validate_approver_permission(workflow, request_id, approver)
        
        # Process the approval
        workflow.process_approval(request_id, status, approver, comments, conditions)
        
        # Update workflow
        updated_workflow = await self.governance_repository.update_workflow(workflow)
        
        # Send notifications
        if status == ApprovalStatus.APPROVED and workflow.config.notify_on_approval:
            await self._send_approval_notifications(workflow, request_id, approver)
        elif status == ApprovalStatus.REJECTED and workflow.config.notify_on_rejection:
            await self._send_rejection_notifications(workflow, request_id, approver, comments)
        
        # Check if workflow is complete
        if workflow.is_complete() and workflow.config.notify_on_completion:
            await self._send_completion_notifications(workflow)
        
        return updated_workflow

    async def create_compliance_rule(
        self,
        name: str,
        description: str,
        category: str,
        rule_type: str,
        parameters: dict[str, Any],
        created_by: str,
        is_mandatory: bool = True,
        severity: str = "medium",
        validation_script: str | None = None,
        tags: list[str] | None = None,
    ) -> ComplianceRule:
        """Create a compliance rule.
        
        Args:
            name: Rule name
            description: Rule description
            category: Rule category
            rule_type: Rule type
            parameters: Rule parameters
            created_by: Rule creator
            is_mandatory: Whether rule is mandatory
            severity: Rule severity
            validation_script: Custom validation script
            tags: Rule tags
            
        Returns:
            Created compliance rule
        """
        rule = ComplianceRule(
            name=name,
            description=description,
            category=category,
            rule_type=rule_type,
            parameters=parameters,
            validation_script=validation_script,
            is_mandatory=is_mandatory,
            severity=severity,
            created_by=created_by,
            tags=tags or [],
        )
        
        stored_rule = await self.governance_repository.create_rule(rule)
        
        return stored_rule

    async def run_compliance_check(
        self, model_id: UUID, rules: list[UUID] | None = None
    ) -> ComplianceReport:
        """Run compliance check for a model.
        
        Args:
            model_id: Model identifier
            rules: Specific rules to check (None for all)
            
        Returns:
            Compliance report
        """
        # Get model
        model = await self.model_repository.get_by_id(model_id)
        if not model:
            raise ValueError(f"Model {model_id} does not exist")
        
        # Get rules to check
        if rules:
            compliance_rules = []
            for rule_id in rules:
                rule = await self.governance_repository.get_rule(rule_id)
                if rule:
                    compliance_rules.append(rule)
        else:
            compliance_rules = await self.governance_repository.get_all_rules()
        
        assessment_start_time = datetime.utcnow()
        
        # Run compliance checks
        violations = []
        passed_rules = 0
        failed_rules = 0
        skipped_rules = 0
        
        for rule in compliance_rules:
            try:
                violation = await self._check_compliance_rule(model, rule)
                if violation:
                    violations.append(violation)
                    failed_rules += 1
                else:
                    passed_rules += 1
            except Exception as e:
                # Log error and skip rule
                print(f"Error checking rule {rule.id}: {e}")
                skipped_rules += 1
        
        assessment_end_time = datetime.utcnow()
        
        # Calculate overall status and score
        total_rules = len(compliance_rules)
        compliance_score = passed_rules / total_rules if total_rules > 0 else 1.0
        
        # Count violations by severity
        critical_violations = len([v for v in violations if v.severity == "critical"])
        high_violations = len([v for v in violations if v.severity == "high"])
        
        # Determine overall status
        if critical_violations > 0:
            overall_status = ComplianceStatus.NON_COMPLIANT
        elif high_violations > 0:
            overall_status = ComplianceStatus.PARTIALLY_COMPLIANT
        elif failed_rules > 0:
            overall_status = ComplianceStatus.PARTIALLY_COMPLIANT
        else:
            overall_status = ComplianceStatus.COMPLIANT
        
        # Create report
        report = ComplianceReport(
            model_id=model_id,
            overall_status=overall_status,
            compliance_score=compliance_score,
            total_rules=total_rules,
            passed_rules=passed_rules,
            failed_rules=failed_rules,
            skipped_rules=skipped_rules,
            violations=violations,
            critical_violations=critical_violations,
            high_violations=high_violations,
            assessment_start_time=assessment_start_time,
            assessment_end_time=assessment_end_time,
            assessment_duration=(assessment_end_time - assessment_start_time).total_seconds(),
            created_by="system",
        )
        
        # Store report
        stored_report = await self.governance_repository.create_compliance_report(report)
        
        return stored_report

    async def get_model_workflows(
        self, model_id: UUID, status: WorkflowStatus | None = None
    ) -> list[ApprovalWorkflow]:
        """Get workflows for a model.
        
        Args:
            model_id: Model identifier
            status: Filter by status
            
        Returns:
            List of workflows
        """
        return await self.governance_repository.get_workflows_by_model(model_id, status)

    async def get_pending_approvals(
        self, approver: str, approval_type: ApprovalType | None = None
    ) -> list[ApprovalRequest]:
        """Get pending approvals for an approver.
        
        Args:
            approver: Approver identifier
            approval_type: Filter by approval type
            
        Returns:
            List of pending approval requests
        """
        return await self.governance_repository.get_pending_approvals(approver, approval_type)

    async def get_workflow_status(self, workflow_id: UUID) -> dict[str, Any]:
        """Get workflow status and progress.
        
        Args:
            workflow_id: Workflow identifier
            
        Returns:
            Workflow status information
        """
        workflow = await self.governance_repository.get_workflow(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        return {
            "workflow": workflow,
            "progress_percentage": workflow.get_progress_percentage(),
            "pending_approvals": workflow.get_pending_approvals(),
            "expired_approvals": workflow.get_expired_approvals(),
            "is_complete": workflow.is_complete(),
            "is_approved": workflow.is_approved(),
            "is_rejected": workflow.is_rejected(),
        }

    async def escalate_approval(
        self, workflow_id: UUID, request_id: UUID, escalated_by: str, reason: str
    ) -> ApprovalWorkflow:
        """Escalate an approval request.
        
        Args:
            workflow_id: Workflow identifier
            request_id: Request identifier
            escalated_by: User escalating
            reason: Escalation reason
            
        Returns:
            Updated workflow
        """
        workflow = await self.governance_repository.get_workflow(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        # Find the request
        request = next((r for r in workflow.approval_requests if r.id == request_id), None)
        if not request:
            raise ValueError(f"Request {request_id} not found in workflow")
        
        # Update request status
        request.status = ApprovalStatus.ESCALATED
        request.comments = f"Escalated by {escalated_by}: {reason}"
        request.updated_at = datetime.utcnow()
        
        # Update workflow
        updated_workflow = await self.governance_repository.update_workflow(workflow)
        
        # Send escalation notifications
        await self._send_escalation_notifications(workflow, request, escalated_by, reason)
        
        return updated_workflow

    async def auto_approve_eligible_requests(self) -> list[ApprovalRequest]:
        """Auto-approve eligible requests based on rules.
        
        Returns:
            List of auto-approved requests
        """
        # Get workflows with auto-approval rules
        workflows = await self.governance_repository.get_workflows_with_auto_approval()
        
        auto_approved = []
        for workflow in workflows:
            for request in workflow.get_pending_approvals():
                if await self._is_eligible_for_auto_approval(workflow, request):
                    # Auto-approve the request
                    workflow.process_approval(
                        request.id,
                        ApprovalStatus.APPROVED,
                        "system",
                        "Auto-approved based on configured rules",
                    )
                    auto_approved.append(request)
        
        return auto_approved

    def _get_approver_role(self, approval_type: ApprovalType) -> str:
        """Get approver role for approval type."""
        role_mapping = {
            ApprovalType.TECHNICAL_REVIEW: "technical_reviewer",
            ApprovalType.SECURITY_REVIEW: "security_reviewer",
            ApprovalType.COMPLIANCE_REVIEW: "compliance_officer",
            ApprovalType.BUSINESS_REVIEW: "business_owner",
            ApprovalType.PERFORMANCE_REVIEW: "performance_analyst",
            ApprovalType.DATA_QUALITY_REVIEW: "data_quality_reviewer",
            ApprovalType.ETHICAL_REVIEW: "ethics_committee",
            ApprovalType.LEGAL_REVIEW: "legal_counsel",
            ApprovalType.FINAL_APPROVAL: "approval_manager",
        }
        return role_mapping.get(approval_type, "reviewer")

    async def _check_compliance_rule(
        self, model: Any, rule: ComplianceRule
    ) -> ComplianceViolation | None:
        """Check a single compliance rule against a model.
        
        Args:
            model: Model to check
            rule: Compliance rule
            
        Returns:
            Compliance violation if rule failed, None if passed
        """
        try:
            # This would implement actual rule checking logic
            # For now, return a simple implementation based on rule type
            
            if rule.rule_type == "metric_threshold":
                # Check if model metrics meet threshold
                threshold = rule.parameters.get("threshold", 0.8)
                metric_name = rule.parameters.get("metric", "accuracy")
                
                # Get model performance (placeholder)
                model_performance = 0.75  # This would be retrieved from model
                
                if model_performance < threshold:
                    return ComplianceViolation(
                        rule_id=rule.id,
                        rule_name=rule.name,
                        severity=rule.severity,
                        description=f"Model {metric_name} ({model_performance}) below threshold ({threshold})",
                        current_value=model_performance,
                        expected_value=threshold,
                        remediation_suggestion=f"Improve model {metric_name} to meet threshold",
                    )
            
            elif rule.rule_type == "data_requirement":
                # Check data requirements
                min_samples = rule.parameters.get("min_samples", 1000)
                
                # Get training data info (placeholder)
                training_samples = 800  # This would be retrieved from model
                
                if training_samples < min_samples:
                    return ComplianceViolation(
                        rule_id=rule.id,
                        rule_name=rule.name,
                        severity=rule.severity,
                        description=f"Training data ({training_samples} samples) below minimum ({min_samples})",
                        current_value=training_samples,
                        expected_value=min_samples,
                        remediation_suggestion="Collect more training data",
                    )
            
            # Rule passed
            return None
            
        except Exception as e:
            # Return violation for rule check failure
            return ComplianceViolation(
                rule_id=rule.id,
                rule_name=rule.name,
                severity="high",
                description=f"Failed to check rule: {str(e)}",
                remediation_suggestion="Fix rule validation logic",
            )

    async def _validate_approver_permission(
        self, workflow: ApprovalWorkflow, request_id: UUID, approver: str
    ) -> None:
        """Validate that approver has permission for the request."""
        request = next((r for r in workflow.approval_requests if r.id == request_id), None)
        if not request:
            raise ValueError(f"Request {request_id} not found")
        
        # Check if specific approver is required
        if request.approver_user and request.approver_user != approver:
            raise ValueError(f"Request requires approval from {request.approver_user}")
        
        # Check self-approval policy
        if not workflow.config.allow_self_approval and request.requested_by == approver:
            raise ValueError("Self-approval not allowed")
        
        # Additional role-based validation would go here

    async def _is_eligible_for_auto_approval(
        self, workflow: ApprovalWorkflow, request: ApprovalRequest
    ) -> bool:
        """Check if request is eligible for auto-approval."""
        auto_rules = workflow.config.auto_approval_rules
        
        if not auto_rules:
            return False
        
        # Check various auto-approval conditions
        # This is a placeholder implementation
        
        # Example: Auto-approve if model performance is above threshold
        if "performance_threshold" in auto_rules:
            threshold = auto_rules["performance_threshold"]
            # Get model performance (placeholder)
            model_performance = 0.9  # This would be retrieved from model
            return model_performance >= threshold
        
        return False

    async def _send_submission_notifications(self, workflow: ApprovalWorkflow) -> None:
        """Send notifications for workflow submission."""
        # This would implement notification sending logic
        pass

    async def _send_approval_notifications(
        self, workflow: ApprovalWorkflow, request_id: UUID, approver: str
    ) -> None:
        """Send notifications for approval."""
        # This would implement notification sending logic
        pass

    async def _send_rejection_notifications(
        self, workflow: ApprovalWorkflow, request_id: UUID, approver: str, comments: str | None
    ) -> None:
        """Send notifications for rejection."""
        # This would implement notification sending logic
        pass

    async def _send_completion_notifications(self, workflow: ApprovalWorkflow) -> None:
        """Send notifications for workflow completion."""
        # This would implement notification sending logic
        pass

    async def _send_escalation_notifications(
        self, workflow: ApprovalWorkflow, request: ApprovalRequest, escalated_by: str, reason: str
    ) -> None:
        """Send notifications for escalation."""
        # This would implement notification sending logic
        pass