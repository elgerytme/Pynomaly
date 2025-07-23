"""Repository interfaces for governance entities and value objects."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, date
from typing import Any, Optional, List, Dict
from uuid import UUID

# TODO: Replace with shared domain abstractions
# from packages.core.domain.abstractions.repository_interface import RepositoryInterface
from shared.domain.abstractions import RepositoryInterface
from ..entities.governance_entity import (
    PolicyRegistry, WorkflowEngine, ComplianceManager, DataSteward,
    GovernanceCommittee, GovernanceDashboard
)
from ..value_objects.governance_policy import (
    GovernancePolicy, PolicyIdentifier, PolicyStatus, PolicyType, PolicyScope,
    PolicyException
)
from ..value_objects.governance_workflow import (
    GovernanceWorkflow, WorkflowStatus, WorkflowType, WorkflowParticipant
)
from ..value_objects.compliance_framework import (
    ComplianceFrameworkDefinition, ComplianceAssessment, ComplianceViolation,
    ComplianceReport, AuditTrail, ComplianceStatus
)


class PolicyRegistryRepository(RepositoryInterface[PolicyRegistry], ABC):
    """Repository interface for policy registry persistence operations."""

    @abstractmethod
    async def find_by_organization_id(self, organization_id: str) -> Optional[PolicyRegistry]:
        """Find policy registry by organization ID.
        
        Args:
            organization_id: Organization ID to search for
            
        Returns:
            PolicyRegistry if found, None otherwise
        """
        pass

    @abstractmethod
    async def get_policies_by_type(
        self, registry_id: str, policy_type: PolicyType
    ) -> List[GovernancePolicy]:
        """Get policies by type from registry.
        
        Args:
            registry_id: Registry ID
            policy_type: Policy type to filter by
            
        Returns:
            List of policies of the specified type
        """
        pass

    @abstractmethod
    async def get_policies_by_scope(
        self, registry_id: str, scope: PolicyScope
    ) -> List[GovernancePolicy]:
        """Get policies by scope from registry.
        
        Args:
            registry_id: Registry ID
            scope: Policy scope to filter by
            
        Returns:
            List of policies with the specified scope
        """
        pass

    @abstractmethod
    async def get_policies_by_status(
        self, registry_id: str, status: PolicyStatus
    ) -> List[GovernancePolicy]:
        """Get policies by status from registry.
        
        Args:
            registry_id: Registry ID
            status: Policy status to filter by
            
        Returns:
            List of policies with the specified status
        """
        pass

    @abstractmethod
    async def find_conflicting_policies(
        self, registry_id: str
    ) -> List[Dict[str, Any]]:
        """Find policies that have conflicts.
        
        Args:
            registry_id: Registry ID
            
        Returns:
            List of policy conflicts with details
        """
        pass

    @abstractmethod
    async def get_policy_hierarchy(
        self, registry_id: str, parent_policy_id: str
    ) -> List[GovernancePolicy]:
        """Get policy hierarchy starting from parent policy.
        
        Args:
            registry_id: Registry ID
            parent_policy_id: Parent policy ID
            
        Returns:
            List of policies in the hierarchy
        """
        pass

    @abstractmethod
    async def get_policy_dependencies(
        self, registry_id: str, policy_id: str
    ) -> List[str]:
        """Get policy dependencies.
        
        Args:
            registry_id: Registry ID
            policy_id: Policy ID to get dependencies for
            
        Returns:
            List of dependent policy IDs
        """
        pass

    @abstractmethod
    async def search_policies(
        self, registry_id: str, search_criteria: Dict[str, Any]
    ) -> List[GovernancePolicy]:
        """Search policies by criteria.
        
        Args:
            registry_id: Registry ID
            search_criteria: Search criteria dictionary
            
        Returns:
            List of policies matching the criteria
        """
        pass

    @abstractmethod
    async def get_policy_statistics(self, registry_id: str) -> Dict[str, Any]:
        """Get policy statistics for registry.
        
        Args:
            registry_id: Registry ID
            
        Returns:
            Dictionary containing policy statistics
        """
        pass

    @abstractmethod
    async def archive_old_policies(
        self, registry_id: str, older_than_days: int = 365
    ) -> int:
        """Archive old inactive policies.
        
        Args:
            registry_id: Registry ID
            older_than_days: Archive policies older than this many days
            
        Returns:
            Number of policies archived
        """
        pass


class GovernancePolicyRepository(ABC):
    """Repository interface for individual governance policy persistence."""

    @abstractmethod
    async def save(self, policy: GovernancePolicy) -> GovernancePolicy:
        """Save governance policy.
        
        Args:
            policy: Policy to save
            
        Returns:
            Saved policy
        """
        pass

    @abstractmethod
    async def find_by_id(self, policy_id: str, version: Optional[str] = None) -> Optional[GovernancePolicy]:
        """Find policy by ID and optional version.
        
        Args:
            policy_id: Policy ID to search for
            version: Optional version, if None returns latest
            
        Returns:
            GovernancePolicy if found, None otherwise
        """
        pass

    @abstractmethod
    async def find_by_identifier(self, identifier: PolicyIdentifier) -> Optional[GovernancePolicy]:
        """Find policy by identifier.
        
        Args:
            identifier: Policy identifier
            
        Returns:
            GovernancePolicy if found, None otherwise
        """
        pass

    @abstractmethod
    async def get_all_versions(self, policy_id: str) -> List[GovernancePolicy]:
        """Get all versions of a policy.
        
        Args:
            policy_id: Policy ID
            
        Returns:
            List of all policy versions
        """
        pass

    @abstractmethod
    async def find_policies_by_owner(self, owner: str) -> List[GovernancePolicy]:
        """Find policies by owner.
        
        Args:
            owner: Policy owner
            
        Returns:
            List of policies owned by the specified owner
        """
        pass

    @abstractmethod
    async def find_policies_by_steward(self, steward_id: str) -> List[GovernancePolicy]:
        """Find policies by steward.
        
        Args:
            steward_id: Steward ID
            
        Returns:
            List of policies assigned to the steward
        """
        pass

    @abstractmethod
    async def find_policies_expiring_soon(self, days_ahead: int = 30) -> List[GovernancePolicy]:
        """Find policies expiring within specified days.
        
        Args:
            days_ahead: Number of days to look ahead
            
        Returns:
            List of policies expiring soon
        """
        pass

    @abstractmethod
    async def get_policy_exceptions(self, policy_id: str) -> List[PolicyException]:
        """Get exceptions for a policy.
        
        Args:
            policy_id: Policy ID
            
        Returns:
            List of policy exceptions
        """
        pass

    @abstractmethod
    async def add_policy_exception(
        self, policy_id: str, exception: PolicyException
    ) -> GovernancePolicy:
        """Add exception to policy.
        
        Args:
            policy_id: Policy ID
            exception: Exception to add
            
        Returns:
            Updated policy
        """
        pass

    @abstractmethod
    async def delete_policy(self, policy_id: str, version: Optional[str] = None) -> bool:
        """Delete policy.
        
        Args:
            policy_id: Policy ID
            version: Optional version, if None deletes all versions
            
        Returns:
            True if deletion was successful
        """
        pass


class WorkflowEngineRepository(RepositoryInterface[WorkflowEngine], ABC):
    """Repository interface for workflow engine persistence operations."""

    @abstractmethod
    async def find_by_organization(self, organization_id: str) -> Optional[WorkflowEngine]:
        """Find workflow engine by organization.
        
        Args:
            organization_id: Organization ID
            
        Returns:
            WorkflowEngine if found, None otherwise
        """
        pass

    @abstractmethod
    async def get_active_workflows(self, engine_id: str) -> List[GovernanceWorkflow]:
        """Get active workflows from engine.
        
        Args:
            engine_id: Engine ID
            
        Returns:
            List of active workflows
        """
        pass

    @abstractmethod
    async def get_workflows_by_type(
        self, engine_id: str, workflow_type: WorkflowType
    ) -> List[GovernanceWorkflow]:
        """Get workflows by type.
        
        Args:
            engine_id: Engine ID
            workflow_type: Workflow type to filter by
            
        Returns:
            List of workflows of the specified type
        """
        pass

    @abstractmethod
    async def get_workflows_by_status(
        self, engine_id: str, status: WorkflowStatus
    ) -> List[GovernanceWorkflow]:
        """Get workflows by status.
        
        Args:
            engine_id: Engine ID
            status: Workflow status to filter by
            
        Returns:
            List of workflows with the specified status
        """
        pass

    @abstractmethod
    async def get_overdue_workflows(self, engine_id: str) -> List[GovernanceWorkflow]:
        """Get overdue workflows.
        
        Args:
            engine_id: Engine ID
            
        Returns:
            List of overdue workflows
        """
        pass

    @abstractmethod
    async def get_workflows_by_participant(
        self, engine_id: str, participant_id: str
    ) -> List[GovernanceWorkflow]:
        """Get workflows by participant.
        
        Args:
            engine_id: Engine ID
            participant_id: Participant ID
            
        Returns:
            List of workflows involving the participant
        """
        pass

    @abstractmethod
    async def get_workflow_performance_metrics(
        self, engine_id: str, days: int = 30
    ) -> Dict[str, Any]:
        """Get workflow performance metrics.
        
        Args:
            engine_id: Engine ID
            days: Number of days to analyze
            
        Returns:
            Dictionary containing performance metrics
        """
        pass

    @abstractmethod
    async def cleanup_completed_workflows(
        self, engine_id: str, older_than_days: int = 90
    ) -> int:
        """Clean up old completed workflows.
        
        Args:
            engine_id: Engine ID
            older_than_days: Clean up workflows older than this many days
            
        Returns:
            Number of workflows cleaned up
        """
        pass


class GovernanceWorkflowRepository(ABC):
    """Repository interface for individual workflow persistence."""

    @abstractmethod
    async def save(self, workflow: GovernanceWorkflow) -> GovernanceWorkflow:
        """Save governance workflow.
        
        Args:
            workflow: Workflow to save
            
        Returns:
            Saved workflow
        """
        pass

    @abstractmethod
    async def find_by_id(self, workflow_id: str) -> Optional[GovernanceWorkflow]:
        """Find workflow by ID.
        
        Args:
            workflow_id: Workflow ID
            
        Returns:
            GovernanceWorkflow if found, None otherwise
        """
        pass

    @abstractmethod
    async def find_by_requester(self, requester_id: str) -> List[GovernanceWorkflow]:
        """Find workflows by requester.
        
        Args:
            requester_id: Requester ID
            
        Returns:
            List of workflows requested by the user
        """
        pass

    @abstractmethod
    async def find_by_assignee(self, assignee_id: str) -> List[GovernanceWorkflow]:
        """Find workflows assigned to user.
        
        Args:
            assignee_id: Assignee ID
            
        Returns:
            List of workflows assigned to the user
        """
        pass

    @abstractmethod
    async def find_workflows_requiring_attention(self) -> List[GovernanceWorkflow]:
        """Find workflows requiring immediate attention.
        
        Returns:
            List of workflows needing attention
        """
        pass

    @abstractmethod
    async def update_workflow_status(
        self, workflow_id: str, status: WorkflowStatus
    ) -> bool:
        """Update workflow status.
        
        Args:
            workflow_id: Workflow ID
            status: New status
            
        Returns:
            True if update was successful
        """
        pass

    @abstractmethod
    async def add_workflow_escalation(
        self, workflow_id: str, escalation_data: Dict[str, Any]
    ) -> bool:
        """Add escalation to workflow.
        
        Args:
            workflow_id: Workflow ID
            escalation_data: Escalation details
            
        Returns:
            True if escalation was added
        """
        pass

    @abstractmethod
    async def get_workflow_history(self, workflow_id: str) -> List[Dict[str, Any]]:
        """Get workflow history.
        
        Args:
            workflow_id: Workflow ID
            
        Returns:
            List of workflow history events
        """
        pass


class ComplianceManagerRepository(RepositoryInterface[ComplianceManager], ABC):
    """Repository interface for compliance manager persistence operations."""

    @abstractmethod
    async def find_by_organization(self, organization_id: str) -> Optional[ComplianceManager]:
        """Find compliance manager by organization.
        
        Args:
            organization_id: Organization ID
            
        Returns:
            ComplianceManager if found, None otherwise
        """
        pass

    @abstractmethod
    async def get_active_frameworks(self, manager_id: str) -> List[ComplianceFrameworkDefinition]:
        """Get active compliance frameworks.
        
        Args:
            manager_id: Manager ID
            
        Returns:
            List of active compliance frameworks
        """
        pass

    @abstractmethod
    async def get_compliance_violations(
        self, manager_id: str, status: Optional[str] = None
    ) -> List[ComplianceViolation]:
        """Get compliance violations.
        
        Args:
            manager_id: Manager ID
            status: Optional status filter
            
        Returns:
            List of compliance violations
        """
        pass

    @abstractmethod
    async def get_compliance_assessments(
        self, manager_id: str, framework_id: Optional[str] = None
    ) -> List[ComplianceAssessment]:
        """Get compliance assessments.
        
        Args:
            manager_id: Manager ID
            framework_id: Optional framework filter
            
        Returns:
            List of compliance assessments
        """
        pass

    @abstractmethod
    async def get_compliance_reports(
        self, manager_id: str, start_date: Optional[date] = None, end_date: Optional[date] = None
    ) -> List[ComplianceReport]:
        """Get compliance reports.
        
        Args:
            manager_id: Manager ID
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            List of compliance reports
        """
        pass

    @abstractmethod
    async def get_audit_trail(
        self, manager_id: str, start_date: datetime, end_date: datetime
    ) -> List[AuditTrail]:
        """Get audit trail for date range.
        
        Args:
            manager_id: Manager ID
            start_date: Start date
            end_date: End date
            
        Returns:
            List of audit trail entries
        """
        pass

    @abstractmethod
    async def calculate_compliance_score(self, manager_id: str) -> float:
        """Calculate overall compliance score.
        
        Args:
            manager_id: Manager ID
            
        Returns:
            Overall compliance score
        """
        pass

    @abstractmethod
    async def get_compliance_trends(
        self, manager_id: str, days: int = 90
    ) -> Dict[str, List[float]]:
        """Get compliance trends over time.
        
        Args:
            manager_id: Manager ID
            days: Number of days to analyze
            
        Returns:
            Dictionary with compliance trend data
        """
        pass


class DataStewardRepository(ABC):
    """Repository interface for data steward persistence operations."""

    @abstractmethod
    async def save(self, steward: DataSteward) -> DataSteward:
        """Save data steward.
        
        Args:
            steward: Steward to save
            
        Returns:
            Saved steward
        """
        pass

    @abstractmethod
    async def find_by_id(self, steward_id: str) -> Optional[DataSteward]:
        """Find steward by ID.
        
        Args:
            steward_id: Steward ID
            
        Returns:
            DataSteward if found, None otherwise
        """
        pass

    @abstractmethod
    async def find_by_department(self, department: str) -> List[DataSteward]:
        """Find stewards by department.
        
        Args:
            department: Department name
            
        Returns:
            List of stewards in the department
        """
        pass

    @abstractmethod
    async def find_by_responsibility_area(self, area: str) -> List[DataSteward]:
        """Find stewards by responsibility area.
        
        Args:
            area: Responsibility area
            
        Returns:
            List of stewards with the responsibility area
        """
        pass

    @abstractmethod
    async def find_available_stewards(self) -> List[DataSteward]:
        """Find stewards available for new assignments.
        
        Returns:
            List of available stewards (not overloaded)
        """
        pass

    @abstractmethod
    async def get_steward_workload(self, steward_id: str) -> Dict[str, Any]:
        """Get steward workload information.
        
        Args:
            steward_id: Steward ID
            
        Returns:
            Dictionary containing workload information
        """
        pass

    @abstractmethod
    async def get_steward_performance_metrics(self, steward_id: str) -> Dict[str, Any]:
        """Get steward performance metrics.
        
        Args:
            steward_id: Steward ID
            
        Returns:
            Dictionary containing performance metrics
        """
        pass

    @abstractmethod
    async def assign_dataset_to_steward(
        self, steward_id: str, dataset_id: str
    ) -> bool:
        """Assign dataset to steward.
        
        Args:
            steward_id: Steward ID
            dataset_id: Dataset ID
            
        Returns:
            True if assignment was successful
        """
        pass

    @abstractmethod
    async def assign_policy_to_steward(
        self, steward_id: str, policy_id: str
    ) -> bool:
        """Assign policy to steward.
        
        Args:
            steward_id: Steward ID
            policy_id: Policy ID
            
        Returns:
            True if assignment was successful
        """
        pass

    @abstractmethod
    async def get_steward_assignments(self, steward_id: str) -> Dict[str, List[str]]:
        """Get steward assignments.
        
        Args:
            steward_id: Steward ID
            
        Returns:
            Dictionary with datasets and policies assigned to steward
        """
        pass


class GovernanceCommitteeRepository(ABC):
    """Repository interface for governance committee persistence operations."""

    @abstractmethod
    async def save(self, committee: GovernanceCommittee) -> GovernanceCommittee:
        """Save governance committee.
        
        Args:
            committee: Committee to save
            
        Returns:
            Saved committee
        """
        pass

    @abstractmethod
    async def find_by_id(self, committee_id: str) -> Optional[GovernanceCommittee]:
        """Find committee by ID.
        
        Args:
            committee_id: Committee ID
            
        Returns:
            GovernanceCommittee if found, None otherwise
        """
        pass

    @abstractmethod
    async def find_by_member(self, steward_id: str) -> List[GovernanceCommittee]:
        """Find committees by member.
        
        Args:
            steward_id: Steward ID
            
        Returns:
            List of committees the steward is a member of
        """
        pass

    @abstractmethod
    async def get_committee_decisions(self, committee_id: str) -> List[Dict[str, Any]]:
        """Get committee decisions.
        
        Args:
            committee_id: Committee ID
            
        Returns:
            List of committee decisions
        """
        pass

    @abstractmethod
    async def get_pending_decisions(self, committee_id: str) -> List[Dict[str, Any]]:
        """Get pending committee decisions.
        
        Args:
            committee_id: Committee ID
            
        Returns:
            List of pending decisions
        """
        pass

    @abstractmethod
    async def add_committee_member(
        self, committee_id: str, steward: DataSteward
    ) -> bool:
        """Add member to committee.
        
        Args:
            committee_id: Committee ID
            steward: Steward to add
            
        Returns:
            True if member was added
        """
        pass

    @abstractmethod
    async def remove_committee_member(
        self, committee_id: str, steward_id: str
    ) -> bool:
        """Remove member from committee.
        
        Args:
            committee_id: Committee ID
            steward_id: Steward ID to remove
            
        Returns:
            True if member was removed
        """
        pass

    @abstractmethod
    async def record_committee_decision(
        self, committee_id: str, decision_data: Dict[str, Any]
    ) -> bool:
        """Record committee decision.
        
        Args:
            committee_id: Committee ID
            decision_data: Decision details
            
        Returns:
            True if decision was recorded
        """
        pass


class AuditTrailRepository(ABC):
    """Repository interface for audit trail persistence operations."""

    @abstractmethod
    async def save_audit_entry(self, audit_entry: AuditTrail) -> AuditTrail:
        """Save audit trail entry.
        
        Args:
            audit_entry: Audit entry to save
            
        Returns:
            Saved audit entry
        """
        pass

    @abstractmethod
    async def find_by_date_range(
        self, start_date: datetime, end_date: datetime
    ) -> List[AuditTrail]:
        """Find audit entries by date range.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            List of audit entries in the date range
        """
        pass

    @abstractmethod
    async def find_by_actor(self, actor_id: str) -> List[AuditTrail]:
        """Find audit entries by actor.
        
        Args:
            actor_id: Actor ID
            
        Returns:
            List of audit entries for the actor
        """
        pass

    @abstractmethod
    async def find_by_resource(self, resource_id: str) -> List[AuditTrail]:
        """Find audit entries by resource.
        
        Args:
            resource_id: Resource ID
            
        Returns:
            List of audit entries for the resource
        """
        pass

    @abstractmethod
    async def find_by_event_type(self, event_type: str) -> List[AuditTrail]:
        """Find audit entries by event type.
        
        Args:
            event_type: Event type
            
        Returns:
            List of audit entries for the event type
        """
        pass

    @abstractmethod
    async def find_high_risk_events(
        self, risk_threshold: float = 0.8
    ) -> List[AuditTrail]:
        """Find high-risk audit events.
        
        Args:
            risk_threshold: Risk score threshold
            
        Returns:
            List of high-risk audit entries
        """
        pass

    @abstractmethod
    async def get_audit_statistics(
        self, start_date: datetime, end_date: datetime
    ) -> Dict[str, Any]:
        """Get audit statistics for date range.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            Dictionary containing audit statistics
        """
        pass

    @abstractmethod
    async def archive_old_entries(self, older_than_days: int = 2557) -> int:
        """Archive old audit entries (default 7 years).
        
        Args:
            older_than_days: Archive entries older than this many days
            
        Returns:
            Number of entries archived
        """
        pass

    @abstractmethod
    async def search_audit_trail(
        self, search_criteria: Dict[str, Any]
    ) -> List[AuditTrail]:
        """Search audit trail by criteria.
        
        Args:
            search_criteria: Search criteria dictionary
            
        Returns:
            List of matching audit entries
        """
        pass

    @abstractmethod
    async def verify_audit_integrity(self) -> Dict[str, Any]:
        """Verify audit trail integrity.
        
        Returns:
            Dictionary containing integrity verification results
        """
        pass