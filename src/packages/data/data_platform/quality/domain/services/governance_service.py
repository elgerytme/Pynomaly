"""Core governance services for policy management and workflow orchestration."""

from __future__ import annotations

from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, replace as dataclass_replace

from ..entities.governance_entity import (
    PolicyRegistry, WorkflowEngine, ComplianceManager, DataSteward,
    GovernanceCommittee, GovernanceDashboard
)
from ..value_objects.governance_policy import (
    GovernancePolicy, PolicyIdentifier, PolicyStatus, PolicyType, PolicyScope,
    PolicyRule, PolicyException, PolicyApproval
)
from ..value_objects.governance_workflow import (
    GovernanceWorkflow, WorkflowStatus, WorkflowType, WorkflowParticipant,
    WorkflowTask, TaskStatus, TaskAction, WorkflowTemplate
)
from ..value_objects.compliance_framework import (
    ComplianceFrameworkDefinition, ComplianceAssessment, ComplianceViolation,
    ComplianceStatus, ViolationSeverity
)


class PolicyManagementService:
    """Service for managing governance policies and their lifecycle."""
    
    def __init__(self, policy_registry: PolicyRegistry):
        self.policy_registry = policy_registry
    
    def create_policy(
        self,
        name: str,
        description: str,
        policy_type: PolicyType,
        scope: PolicyScope,
        rules: List[PolicyRule],
        **kwargs
    ) -> Tuple[GovernancePolicy, PolicyRegistry]:
        """Create a new governance policy."""
        identifier = PolicyIdentifier.create_new(name, self.policy_registry.organization_id)
        
        policy = GovernancePolicy(
            identifier=identifier,
            name=name,
            description=description,
            policy_type=policy_type,
            scope=scope,
            rules=rules,
            **kwargs
        )
        
        updated_registry = self.policy_registry.add_policy(policy)
        
        return policy, updated_registry
    
    def submit_policy_for_approval(
        self,
        policy_id: str,
        approvers: List[str],
        approval_workflow_template: Optional[str] = None
    ) -> Tuple[GovernancePolicy, GovernanceWorkflow]:
        """Submit policy for approval process."""
        policy = self.policy_registry.get_policy(policy_id)
        if not policy:
            raise ValueError(f"Policy {policy_id} not found")
        
        if policy.status != PolicyStatus.DRAFT:
            raise ValueError(f"Policy must be in DRAFT status to submit for approval")
        
        # Update policy status
        updated_policy = dataclass_replace(
            policy,
            status=PolicyStatus.UNDER_REVIEW,
            required_approvers=approvers,
            approval_workflow=approval_workflow_template,
            last_modified=datetime.now()
        )
        
        # Create approval workflow
        workflow_participants = [
            WorkflowParticipant(
                participant_id=approver_id,
                participant_type="user",
                name=f"Approver {approver_id}",
                permissions=["approve", "reject", "comment"]
            )
            for approver_id in approvers
        ]
        
        workflow = GovernanceWorkflow.create_policy_approval_workflow(
            policy_id=policy_id,
            requester=WorkflowParticipant(
                participant_id="system",
                participant_type="system",
                name="Policy Management System"
            ),
            approvers=workflow_participants
        )
        
        return updated_policy, workflow
    
    def approve_policy(
        self,
        policy_id: str,
        approver_id: str,
        comments: str = ""
    ) -> GovernancePolicy:
        """Approve a policy."""
        policy = self.policy_registry.get_policy(policy_id)
        if not policy:
            raise ValueError(f"Policy {policy_id} not found")
        
        approval = PolicyApproval(
            approver_id=approver_id,
            approver_role="approver",
            approval_date=datetime.now(),
            approval_status="approved",
            comments=comments
        )
        
        updated_policy = policy.add_approval(approval)
        
        # Check if all required approvals are received
        if updated_policy.is_fully_approved:
            updated_policy = updated_policy.update_status(PolicyStatus.APPROVED)
        
        return updated_policy
    
    def activate_policy(self, policy_id: str) -> GovernancePolicy:
        """Activate an approved policy."""
        policy = self.policy_registry.get_policy(policy_id)
        if not policy:
            raise ValueError(f"Policy {policy_id} not found")
        
        if policy.status != PolicyStatus.APPROVED:
            raise ValueError("Policy must be approved before activation")
        
        return dataclass_replace(
            policy,
            status=PolicyStatus.ACTIVE,
            effective_date=datetime.now(),
            last_modified=datetime.now()
        )
    
    def create_policy_exception(
        self,
        policy_id: str,
        rule_id: Optional[str],
        reason: str,
        justification: str,
        requested_by: str,
        duration_days: Optional[int] = None
    ) -> PolicyException:
        """Create a policy exception request."""
        policy = self.policy_registry.get_policy(policy_id)
        if not policy:
            raise ValueError(f"Policy {policy_id} not found")
        
        if not policy.allows_exceptions:
            raise ValueError(f"Policy {policy_id} does not allow exceptions")
        
        if duration_days:
            exception = PolicyException.create_temporary(
                policy_id=policy_id,
                reason=reason,
                requested_by=requested_by,
                duration_days=duration_days,
                rule_id=rule_id,
                justification=justification
            )
        else:
            exception = PolicyException(
                exception_id=f"exception_{policy_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                policy_id=policy_id,
                rule_id=rule_id,
                reason=reason,
                justification=justification,
                requested_by=requested_by,
                is_permanent=True,
                approval_required=policy.exception_approval_required
            )
        
        return exception
    
    def evaluate_policy_compliance(
        self,
        policy_id: str,
        data_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate compliance with a policy in given context."""
        policy = self.policy_registry.get_policy(policy_id)
        if not policy:
            raise ValueError(f"Policy {policy_id} not found")
        
        if not policy.is_active:
            return {
                "compliant": True,
                "reason": "Policy is not active",
                "applicable_rules": []
            }
        
        applicable_rules = policy.get_applicable_rules(data_context)
        violations = []
        
        for rule in applicable_rules:
            # Check for active exceptions
            exception = policy.get_active_exception_for_rule(rule.rule_id)
            if exception:
                continue
            
            # Evaluate rule (simplified - would use rule engine in practice)
            if not self._evaluate_rule(rule, data_context):
                violations.append({
                    "rule_id": rule.rule_id,
                    "rule_name": rule.name,
                    "severity": rule.severity,
                    "description": rule.description
                })
        
        return {
            "compliant": len(violations) == 0,
            "violations": violations,
            "applicable_rules": [rule.rule_id for rule in applicable_rules],
            "total_rules_evaluated": len(applicable_rules)
        }
    
    def _evaluate_rule(self, rule: PolicyRule, context: Dict[str, Any]) -> bool:
        """Evaluate a policy rule against context (simplified implementation)."""
        # This would use a proper rule engine in practice
        # For now, implement basic rule evaluation
        
        if "completeness" in rule.condition:
            column = rule.parameters.get("column")
            if column and column in context:
                completeness = context[column].get("completeness", 0.0)
                threshold = rule.parameters.get("threshold", 0.95)
                return completeness >= threshold
        
        if "quality_score" in rule.condition:
            score = context.get("quality_score", 0.0)
            threshold = rule.parameters.get("threshold", 0.8)
            return score >= threshold
        
        # Default to compliant if rule type not implemented
        return True


class WorkflowOrchestrationService:
    """Service for orchestrating governance workflows."""
    
    def __init__(self, workflow_engine: WorkflowEngine):
        self.workflow_engine = workflow_engine
    
    def create_workflow_from_template(
        self,
        template_id: str,
        workflow_title: str,
        requester: WorkflowParticipant,
        context_data: Dict[str, Any]
    ) -> GovernanceWorkflow:
        """Create workflow from template."""
        template = self.workflow_engine.workflow_templates.get(template_id)
        if not template:
            raise ValueError(f"Workflow template {template_id} not found")
        
        workflow_id = f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{template_id}"
        
        # Create workflow from template (simplified)
        workflow = GovernanceWorkflow(
            workflow_id=workflow_id,
            title=workflow_title,
            workflow_type=WorkflowType.POLICY_APPROVAL,  # Default, would come from template
            template_id=template_id,
            requester=requester,
            context_data=context_data
        )
        
        return workflow
    
    def advance_workflow(
        self,
        workflow_id: str,
        task_id: str,
        action_type: str,
        actor: WorkflowParticipant,
        comments: str = ""
    ) -> Tuple[GovernanceWorkflow, WorkflowEngine]:
        """Advance workflow by completing a task."""
        workflow = self.workflow_engine.active_workflows.get(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        # Find the task
        task = next((t for t in workflow.tasks if t.task_id == task_id), None)
        if not task:
            raise ValueError(f"Task {task_id} not found in workflow")
        
        # Create task action
        action = TaskAction(
            action_id=f"action_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            action_type=action_type,
            actor=actor,
            timestamp=datetime.now(),
            comments=comments
        )
        
        # Update task with action
        updated_task = task.add_action(action)
        
        # Update workflow with new task
        updated_tasks = [
            updated_task if t.task_id == task_id else t 
            for t in workflow.tasks
        ]
        
        updated_workflow = dataclass_replace(
            workflow,
            tasks=updated_tasks,
            last_modified=datetime.now()
        )
        
        # Advance workflow if task is completed
        if updated_task.is_completed:
            updated_workflow = updated_workflow.advance_workflow()
        
        # Update workflow engine
        updated_active_workflows = dict(self.workflow_engine.active_workflows)
        
        if updated_workflow.status == WorkflowStatus.COMPLETED:
            # Move to completed workflows
            del updated_active_workflows[workflow_id]
            updated_completed = dict(self.workflow_engine.completed_workflows)
            updated_completed[workflow_id] = updated_workflow
            
            updated_engine = dataclass_replace(
                self.workflow_engine,
                active_workflows=updated_active_workflows,
                completed_workflows=updated_completed,
                total_workflows_processed=self.workflow_engine.total_workflows_processed + 1
            )
        else:
            # Update active workflow
            updated_active_workflows[workflow_id] = updated_workflow
            updated_engine = dataclass_replace(
                self.workflow_engine,
                active_workflows=updated_active_workflows
            )
        
        return updated_workflow, updated_engine
    
    def escalate_workflow(
        self,
        workflow_id: str,
        escalated_by: WorkflowParticipant,
        reason: str
    ) -> Tuple[GovernanceWorkflow, WorkflowEngine]:
        """Escalate a workflow."""
        workflow = self.workflow_engine.active_workflows.get(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        from ..value_objects.governance_workflow import EscalationTrigger
        escalated_workflow = workflow.escalate_workflow(
            trigger=EscalationTrigger.MANUAL,
            escalated_by=escalated_by,
            reason=reason
        )
        
        # Update workflow engine
        updated_active = dict(self.workflow_engine.active_workflows)
        updated_active[workflow_id] = escalated_workflow
        
        updated_engine = dataclass_replace(
            self.workflow_engine,
            active_workflows=updated_active,
            escalation_rate=self._calculate_escalation_rate()
        )
        
        return escalated_workflow, updated_engine
    
    def get_workflows_requiring_attention(self) -> List[GovernanceWorkflow]:
        """Get workflows requiring immediate attention."""
        attention_workflows = []
        
        for workflow in self.workflow_engine.active_workflows.values():
            if (workflow.is_overdue or 
                workflow.priority in ["high", "urgent"] or
                len(workflow.escalation_history) > 0):
                attention_workflows.append(workflow)
        
        return sorted(attention_workflows, key=lambda w: (
            w.priority == "urgent",
            w.priority == "high",
            w.is_overdue,
            len(w.escalation_history)
        ), reverse=True)
    
    def _calculate_escalation_rate(self) -> float:
        """Calculate workflow escalation rate."""
        total_workflows = (len(self.workflow_engine.active_workflows) + 
                          len(self.workflow_engine.completed_workflows))
        
        if total_workflows == 0:
            return 0.0
        
        escalated_count = sum(
            1 for w in list(self.workflow_engine.active_workflows.values()) + 
                           list(self.workflow_engine.completed_workflows.values())
            if len(w.escalation_history) > 0
        )
        
        return escalated_count / total_workflows


class ComplianceMonitoringService:
    """Service for monitoring and managing compliance."""
    
    def __init__(self, compliance_manager: ComplianceManager):
        self.compliance_manager = compliance_manager
    
    def conduct_compliance_assessment(
        self,
        framework_id: str,
        requirement_id: str,
        assessor_id: str,
        assessment_method: str = "automated"
    ) -> Tuple[ComplianceAssessment, ComplianceManager]:
        """Conduct compliance assessment for a requirement."""
        framework = self.compliance_manager.frameworks.get(framework_id)
        if not framework:
            raise ValueError(f"Framework {framework_id} not found")
        
        requirement = next(
            (req for req in framework.requirements if req.requirement_id == requirement_id),
            None
        )
        if not requirement:
            raise ValueError(f"Requirement {requirement_id} not found in framework")
        
        # Perform assessment (simplified)
        assessment_id = f"assessment_{requirement_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Simulate assessment results
        compliance_score = self._simulate_compliance_assessment(requirement)
        status = ComplianceStatus.COMPLIANT if compliance_score >= 80 else ComplianceStatus.NON_COMPLIANT
        
        assessment = ComplianceAssessment(
            assessment_id=assessment_id,
            requirement_id=requirement_id,
            assessor_id=assessor_id,
            assessment_date=datetime.now(),
            status=status,
            assessment_method=assessment_method,
            score=compliance_score,
            remediation_required=status != ComplianceStatus.COMPLIANT
        )
        
        # Update compliance manager
        updated_assessments = dict(self.compliance_manager.assessments)
        updated_assessments[assessment_id] = assessment
        
        updated_manager = dataclass_replace(
            self.compliance_manager,
            assessments=updated_assessments,
            overall_compliance_score=self._calculate_overall_compliance_score(updated_assessments),
            last_updated=datetime.now()
        )
        
        return assessment, updated_manager
    
    def report_compliance_violation(
        self,
        framework_id: str,
        requirement_id: str,
        violation_description: str,
        severity: ViolationSeverity,
        discovered_by: str
    ) -> Tuple[ComplianceViolation, ComplianceManager]:
        """Report a compliance violation."""
        violation_id = f"violation_{requirement_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        violation = ComplianceViolation(
            violation_id=violation_id,
            requirement_id=requirement_id,
            framework_id=framework_id,
            violation_type="compliance_breach",
            severity=severity,
            description=violation_description,
            discovered_date=datetime.now(),
            discovered_by=discovered_by,
            discovery_method="automated"
        )
        
        updated_manager = self.compliance_manager.record_violation(violation)
        
        return violation, updated_manager
    
    def generate_compliance_report(
        self,
        framework_id: str,
        period_start: datetime,
        period_end: datetime,
        generated_by: str
    ) -> ComplianceReport:
        """Generate compliance report for a framework."""
        framework = self.compliance_manager.frameworks.get(framework_id)
        if not framework:
            raise ValueError(f"Framework {framework_id} not found")
        
        # Filter assessments and violations for the period
        period_assessments = [
            assessment for assessment in self.compliance_manager.assessments.values()
            if (period_start <= assessment.assessment_date <= period_end and
                any(req.requirement_id == assessment.requirement_id 
                    for req in framework.requirements))
        ]
        
        period_violations = [
            violation for violation in self.compliance_manager.violations.values()
            if (period_start <= violation.discovered_date <= period_end and
                violation.framework_id == framework_id)
        ]
        
        # Calculate compliance metrics
        total_requirements = len(framework.requirements)
        compliant_count = len([a for a in period_assessments if a.is_compliant])
        compliance_percentage = (compliant_count / total_requirements * 100) if total_requirements > 0 else 0
        
        # Violation statistics
        violation_counts = {
            "critical": len([v for v in period_violations if v.severity == ViolationSeverity.CRITICAL]),
            "high": len([v for v in period_violations if v.severity == ViolationSeverity.HIGH]),
            "medium": len([v for v in period_violations if v.severity == ViolationSeverity.MEDIUM]),
            "low": len([v for v in period_violations if v.severity == ViolationSeverity.LOW])
        }
        
        from ..value_objects.compliance_framework import ComplianceReport
        
        report = ComplianceReport(
            report_id=f"report_{framework_id}_{datetime.now().strftime('%Y%m%d')}",
            framework_id=framework_id,
            reporting_period_start=period_start.date(),
            reporting_period_end=period_end.date(),
            generated_date=datetime.now(),
            generated_by=generated_by,
            overall_status=ComplianceStatus.COMPLIANT if compliance_percentage >= 90 else ComplianceStatus.PARTIALLY_COMPLIANT,
            compliance_percentage=compliance_percentage,
            requirements_assessed=len(period_assessments),
            requirements_compliant=compliant_count,
            requirements_non_compliant=len(period_assessments) - compliant_count,
            total_violations=len(period_violations),
            critical_violations=violation_counts["critical"],
            high_violations=violation_counts["high"],
            medium_violations=violation_counts["medium"],
            low_violations=violation_counts["low"],
            resolved_violations=len([v for v in period_violations if v.is_resolved]),
            assessments_conducted=period_assessments,
            violations_found=period_violations
        )
        
        return report
    
    def _simulate_compliance_assessment(self, requirement) -> float:
        """Simulate compliance assessment (for demonstration)."""
        # In practice, this would perform actual compliance checks
        import random
        base_score = 85.0
        variation = random.uniform(-15.0, 15.0)
        return max(0.0, min(100.0, base_score + variation))
    
    def _calculate_overall_compliance_score(self, assessments: Dict[str, ComplianceAssessment]) -> float:
        """Calculate overall compliance score."""
        if not assessments:
            return 0.0
        
        scores = [assessment.score for assessment in assessments.values() 
                 if assessment.score is not None]
        
        if not scores:
            return 0.0
        
        return sum(scores) / len(scores)


class DataStewardshipService:
    """Service for managing data stewardship and assignments."""
    
    def assign_steward_to_dataset(
        self,
        steward: DataSteward,
        dataset_id: str,
        responsibility_areas: List[str]
    ) -> DataSteward:
        """Assign steward to a dataset."""
        if steward.is_overloaded:
            raise ValueError(f"Steward {steward.name} is already overloaded")
        
        updated_steward = steward.assign_dataset(dataset_id)
        
        # Add responsibility areas
        new_areas = list(updated_steward.responsibility_areas) + responsibility_areas
        updated_steward = dataclass_replace(
            updated_steward,
            responsibility_areas=list(set(new_areas))  # Remove duplicates
        )
        
        return updated_steward
    
    def assign_steward_to_policy(
        self,
        steward: DataSteward,
        policy_id: str
    ) -> DataSteward:
        """Assign steward to oversee a policy."""
        if steward.is_overloaded:
            raise ValueError(f"Steward {steward.name} is already overloaded")
        
        return steward.assign_policy(policy_id)
    
    def recommend_steward_for_assignment(
        self,
        stewards: List[DataSteward],
        dataset_id: str,
        required_skills: List[str]
    ) -> Optional[DataSteward]:
        """Recommend best steward for an assignment."""
        # Filter available stewards
        available_stewards = [s for s in stewards if not s.is_overloaded]
        
        if not available_stewards:
            return None
        
        # Score stewards based on skills and workload
        scored_stewards = []
        for steward in available_stewards:
            skill_score = len(set(steward.responsibility_areas).intersection(set(required_skills)))
            workload_score = 1.0 - steward.workload_score  # Lower workload is better
            quality_score = steward.quality_score
            
            total_score = skill_score * 0.5 + workload_score * 0.3 + quality_score * 0.2
            scored_stewards.append((steward, total_score))
        
        # Return steward with highest score
        if scored_stewards:
            return max(scored_stewards, key=lambda x: x[1])[0]
        
        return None
    
    def escalate_to_committee(
        self,
        committee: GovernanceCommittee,
        issue_description: str,
        escalated_by: str,
        priority: str = "medium"
    ) -> GovernanceCommittee:
        """Escalate issue to governance committee."""
        if not committee.is_quorum_met:
            raise ValueError("Committee does not have quorum for decision making")
        
        escalation_item = {
            "escalation_id": f"escalation_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "description": issue_description,
            "escalated_by": escalated_by,
            "escalated_date": datetime.now(),
            "priority": priority,
            "status": "pending_review",
            "assigned_to": committee.chair_steward_id
        }
        
        new_pending = list(committee.pending_decisions) + [escalation_item]
        
        return dataclass_replace(
            committee,
            pending_decisions=new_pending
        )