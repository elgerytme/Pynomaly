"""Core governance entities for data quality governance framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from uuid import UUID, uuid4

from ..value_objects.governance_policy import (
    GovernancePolicy, PolicyIdentifier, PolicyStatus, PolicyType, PolicyScope
)
from ..value_objects.governance_workflow import (
    GovernanceWorkflow, WorkflowStatus, WorkflowType, WorkflowParticipant
)
from ..value_objects.compliance_framework import (
    ComplianceFrameworkDefinition, ComplianceAssessment, ComplianceViolation,
    ComplianceReport, AuditTrail, ComplianceStatus
)


@dataclass
class PolicyRegistry:
    """Central registry for all governance policies."""
    registry_id: str
    organization_id: str
    name: str = "Central Policy Registry"
    
    # Policy collections
    policies: Dict[str, GovernancePolicy] = field(default_factory=dict)
    policy_hierarchies: Dict[str, List[str]] = field(default_factory=dict)  # parent -> children
    policy_dependencies: Dict[str, List[str]] = field(default_factory=dict)  # policy -> dependencies
    
    # Registry metadata
    created_date: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    version: str = "1.0.0"
    
    # Configuration
    approval_workflow_template: Optional[str] = None
    auto_inheritance_enabled: bool = True
    conflict_resolution_strategy: str = "strict_hierarchy"  # strict_hierarchy, manual_resolution
    
    # Statistics
    total_policies: int = 0
    active_policies: int = 0
    pending_approval: int = 0
    
    def add_policy(self, policy: GovernancePolicy) -> PolicyRegistry:
        """Add policy to registry."""
        policy_key = policy.identifier.full_id
        new_policies = dict(self.policies)
        new_policies[policy_key] = policy
        
        return dataclass.replace(
            self,
            policies=new_policies,
            total_policies=len(new_policies),
            active_policies=len([p for p in new_policies.values() if p.is_active]),
            pending_approval=len([p for p in new_policies.values() 
                                 if p.status == PolicyStatus.UNDER_REVIEW]),
            last_updated=datetime.now()
        )
    
    def get_policy(self, policy_id: str, version: Optional[str] = None) -> Optional[GovernancePolicy]:
        """Get policy by ID and optional version."""
        if version:
            full_id = f"{policy_id}:{version}"
            return self.policies.get(full_id)
        
        # Get latest version if no version specified
        matching_policies = [
            (key, policy) for key, policy in self.policies.items()
            if key.startswith(f"{policy_id}:")
        ]
        
        if not matching_policies:
            return None
        
        # Return latest version (assuming semantic versioning)
        latest = max(matching_policies, key=lambda x: x[0])
        return latest[1]
    
    def get_policies_by_type(self, policy_type: PolicyType) -> List[GovernancePolicy]:
        """Get all policies of specific type."""
        return [policy for policy in self.policies.values() 
                if policy.policy_type == policy_type]
    
    def get_policies_by_scope(self, scope: PolicyScope) -> List[GovernancePolicy]:
        """Get all policies with specific scope."""
        return [policy for policy in self.policies.values() 
                if policy.scope == scope]
    
    def get_active_policies(self) -> List[GovernancePolicy]:
        """Get all active policies."""
        return [policy for policy in self.policies.values() if policy.is_active]
    
    def detect_policy_conflicts(self) -> List[Dict[str, Any]]:
        """Detect conflicts between policies."""
        conflicts = []
        active_policies = self.get_active_policies()
        
        # Simple conflict detection - can be extended with more sophisticated logic
        for i, policy1 in enumerate(active_policies):
            for policy2 in active_policies[i+1:]:
                if self._policies_conflict(policy1, policy2):
                    conflicts.append({
                        "policy1": policy1.identifier.full_id,
                        "policy2": policy2.identifier.full_id,
                        "conflict_type": "rule_overlap",
                        "description": f"Policies {policy1.name} and {policy2.name} have overlapping rules"
                    })
        
        return conflicts
    
    def _policies_conflict(self, policy1: GovernancePolicy, policy2: GovernancePolicy) -> bool:
        """Check if two policies conflict."""
        # Simple conflict detection - same scope and overlapping rules
        if policy1.scope != policy2.scope:
            return False
        
        # Check for rule conflicts
        rules1 = {rule.name for rule in policy1.rules}
        rules2 = {rule.name for rule in policy2.rules}
        
        return len(rules1.intersection(rules2)) > 0


@dataclass
class WorkflowEngine:
    """Engine for executing governance workflows."""
    engine_id: str
    name: str = "Governance Workflow Engine"
    
    # Active workflows
    active_workflows: Dict[str, GovernanceWorkflow] = field(default_factory=dict)
    completed_workflows: Dict[str, GovernanceWorkflow] = field(default_factory=dict)
    workflow_templates: Dict[str, Any] = field(default_factory=dict)
    
    # Engine configuration
    max_concurrent_workflows: int = 100
    default_sla_hours: int = 72
    auto_escalation_enabled: bool = True
    notification_enabled: bool = True
    
    # Performance metrics
    total_workflows_processed: int = 0
    average_completion_time_hours: float = 0.0
    escalation_rate: float = 0.0
    
    # Engine metadata
    created_date: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    
    def start_workflow(self, workflow: GovernanceWorkflow) -> WorkflowEngine:
        """Start a new workflow."""
        if len(self.active_workflows) >= self.max_concurrent_workflows:
            raise ValueError("Maximum concurrent workflows exceeded")
        
        started_workflow = dataclass.replace(
            workflow,
            status=WorkflowStatus.INITIATED,
            started_date=datetime.now()
        )
        
        new_active = dict(self.active_workflows)
        new_active[workflow.workflow_id] = started_workflow
        
        return dataclass.replace(
            self,
            active_workflows=new_active,
            last_updated=datetime.now()
        )
    
    def complete_workflow(self, workflow_id: str, final_decision: str) -> WorkflowEngine:
        """Complete a workflow."""
        if workflow_id not in self.active_workflows:
            raise ValueError(f"Workflow {workflow_id} not found in active workflows")
        
        workflow = self.active_workflows[workflow_id]
        completed_workflow = dataclass.replace(
            workflow,
            status=WorkflowStatus.COMPLETED,
            completed_date=datetime.now(),
            final_decision=final_decision
        )
        
        new_active = dict(self.active_workflows)
        del new_active[workflow_id]
        
        new_completed = dict(self.completed_workflows)
        new_completed[workflow_id] = completed_workflow
        
        return dataclass.replace(
            self,
            active_workflows=new_active,
            completed_workflows=new_completed,
            total_workflows_processed=self.total_workflows_processed + 1,
            last_updated=datetime.now()
        )
    
    def get_overdue_workflows(self) -> List[GovernanceWorkflow]:
        """Get workflows that are overdue."""
        return [workflow for workflow in self.active_workflows.values() 
                if workflow.is_overdue]
    
    def get_workflows_by_type(self, workflow_type: WorkflowType) -> List[GovernanceWorkflow]:
        """Get workflows by type."""
        all_workflows = list(self.active_workflows.values()) + list(self.completed_workflows.values())
        return [workflow for workflow in all_workflows 
                if workflow.workflow_type == workflow_type]


@dataclass
class ComplianceManager:
    """Manager for compliance frameworks and assessments."""
    manager_id: str
    organization_id: str
    name: str = "Compliance Management System"
    
    # Compliance frameworks
    frameworks: Dict[str, ComplianceFrameworkDefinition] = field(default_factory=dict)
    active_frameworks: Set[str] = field(default_factory=set)
    
    # Assessments and violations
    assessments: Dict[str, ComplianceAssessment] = field(default_factory=dict)
    violations: Dict[str, ComplianceViolation] = field(default_factory=dict)
    audit_trails: List[AuditTrail] = field(default_factory=list)
    
    # Reports and documentation
    compliance_reports: Dict[str, ComplianceReport] = field(default_factory=dict)
    
    # Configuration
    auto_assessment_enabled: bool = True
    continuous_monitoring_enabled: bool = True
    violation_auto_remediation: bool = False
    
    # Metrics
    overall_compliance_score: float = 0.0
    total_violations: int = 0
    critical_violations: int = 0
    resolved_violations: int = 0
    
    # Manager metadata
    created_date: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    
    def add_framework(self, framework: ComplianceFrameworkDefinition) -> ComplianceManager:
        """Add compliance framework."""
        new_frameworks = dict(self.frameworks)
        new_frameworks[framework.framework_id] = framework
        
        return dataclass.replace(
            self,
            frameworks=new_frameworks,
            last_updated=datetime.now()
        )
    
    def activate_framework(self, framework_id: str) -> ComplianceManager:
        """Activate compliance framework."""
        if framework_id not in self.frameworks:
            raise ValueError(f"Framework {framework_id} not found")
        
        new_active = set(self.active_frameworks)
        new_active.add(framework_id)
        
        return dataclass.replace(
            self,
            active_frameworks=new_active,
            last_updated=datetime.now()
        )
    
    def record_violation(self, violation: ComplianceViolation) -> ComplianceManager:
        """Record a compliance violation."""
        new_violations = dict(self.violations)
        new_violations[violation.violation_id] = violation
        
        # Update metrics
        new_total = len(new_violations)
        new_critical = len([v for v in new_violations.values() if v.is_critical])
        new_resolved = len([v for v in new_violations.values() if v.is_resolved])
        
        return dataclass.replace(
            self,
            violations=new_violations,
            total_violations=new_total,
            critical_violations=new_critical,
            resolved_violations=new_resolved,
            last_updated=datetime.now()
        )
    
    def get_active_violations(self) -> List[ComplianceViolation]:
        """Get active (unresolved) violations."""
        return [violation for violation in self.violations.values() 
                if not violation.is_resolved]
    
    def get_critical_violations(self) -> List[ComplianceViolation]:
        """Get critical violations."""
        return [violation for violation in self.violations.values() 
                if violation.is_critical]
    
    def calculate_compliance_score(self) -> float:
        """Calculate overall compliance score."""
        if not self.assessments:
            return 0.0
        
        scores = [assessment.score for assessment in self.assessments.values() 
                 if assessment.score is not None]
        
        if not scores:
            return 0.0
        
        return sum(scores) / len(scores)


@dataclass
class DataSteward:
    """Data steward responsible for data governance."""
    steward_id: str
    name: str
    email: str
    department: str
    
    # Stewardship responsibilities
    assigned_datasets: Set[str] = field(default_factory=set)
    assigned_policies: Set[str] = field(default_factory=set)
    responsibility_areas: List[str] = field(default_factory=list)
    
    # Authority levels
    approval_authority: List[str] = field(default_factory=list)  # Types of approvals they can make
    escalation_authority: bool = False
    policy_creation_authority: bool = False
    
    # Performance tracking
    active_assignments: int = 0
    completed_assignments: int = 0
    average_resolution_time_hours: float = 0.0
    quality_score: float = 0.0
    
    # Contact and availability
    notification_preferences: Dict[str, Any] = field(default_factory=dict)
    availability_schedule: Dict[str, Any] = field(default_factory=dict)
    backup_steward_id: Optional[str] = None
    
    # Metadata
    created_date: datetime = field(default_factory=datetime.now)
    last_active: datetime = field(default_factory=datetime.now)
    
    @property
    def workload_score(self) -> float:
        """Calculate steward workload score."""
        total_assignments = len(self.assigned_datasets) + len(self.assigned_policies)
        if total_assignments == 0:
            return 0.0
        
        # Simple workload calculation - can be enhanced
        return min(1.0, total_assignments / 20)  # Assuming 20 is max reasonable load
    
    @property
    def is_overloaded(self) -> bool:
        """Check if steward is overloaded."""
        return self.workload_score > 0.8
    
    def assign_dataset(self, dataset_id: str) -> DataSteward:
        """Assign dataset to steward."""
        new_datasets = set(self.assigned_datasets)
        new_datasets.add(dataset_id)
        
        return dataclass.replace(
            self,
            assigned_datasets=new_datasets,
            active_assignments=len(new_datasets) + len(self.assigned_policies),
            last_active=datetime.now()
        )
    
    def assign_policy(self, policy_id: str) -> DataSteward:
        """Assign policy to steward."""
        new_policies = set(self.assigned_policies)
        new_policies.add(policy_id)
        
        return dataclass.replace(
            self,
            assigned_policies=new_policies,
            active_assignments=len(self.assigned_datasets) + len(new_policies),
            last_active=datetime.now()
        )


@dataclass
class GovernanceCommittee:
    """Governance committee for high-level governance decisions."""
    committee_id: str
    name: str
    purpose: str
    
    # Committee composition
    members: List[DataSteward] = field(default_factory=list)
    chair_steward_id: Optional[str] = None
    secretary_steward_id: Optional[str] = None
    
    # Authority and scope
    decision_authority: List[str] = field(default_factory=list)
    policy_domains: List[str] = field(default_factory=list)
    escalation_threshold: str = "high"  # Types of issues that reach committee
    
    # Meeting management
    meeting_frequency: str = "monthly"  # weekly, monthly, quarterly
    next_meeting_date: Optional[datetime] = None
    meeting_quorum: int = 3
    
    # Decision tracking
    decisions_made: List[Dict[str, Any]] = field(default_factory=list)
    pending_decisions: List[Dict[str, Any]] = field(default_factory=list)
    
    # Performance metrics
    average_decision_time_days: float = 0.0
    decision_implementation_rate: float = 0.0
    
    # Metadata
    established_date: datetime = field(default_factory=datetime.now)
    last_meeting_date: Optional[datetime] = None
    
    @property
    def is_quorum_met(self) -> bool:
        """Check if committee has quorum."""
        return len(self.members) >= self.meeting_quorum
    
    @property
    def chair(self) -> Optional[DataSteward]:
        """Get committee chair."""
        if not self.chair_steward_id:
            return None
        
        return next((member for member in self.members 
                    if member.steward_id == self.chair_steward_id), None)
    
    def add_member(self, steward: DataSteward) -> GovernanceCommittee:
        """Add member to committee."""
        new_members = list(self.members) + [steward]
        
        return dataclass.replace(
            self,
            members=new_members
        )
    
    def make_decision(
        self, 
        decision_topic: str, 
        decision: str, 
        rationale: str,
        decided_by: str
    ) -> GovernanceCommittee:
        """Record committee decision."""
        decision_record = {
            "decision_id": f"committee_decision_{uuid4().hex[:8]}",
            "topic": decision_topic,
            "decision": decision,
            "rationale": rationale,
            "decided_by": decided_by,
            "decision_date": datetime.now(),
            "implementation_status": "pending"
        }
        
        new_decisions = list(self.decisions_made) + [decision_record]
        
        return dataclass.replace(
            self,
            decisions_made=new_decisions
        )


@dataclass
class GovernanceDashboard:
    """Comprehensive governance dashboard and analytics."""
    dashboard_id: str
    organization_id: str
    name: str = "Governance Analytics Dashboard"
    
    # Data sources
    policy_registry: Optional[PolicyRegistry] = None
    workflow_engine: Optional[WorkflowEngine] = None
    compliance_manager: Optional[ComplianceManager] = None
    
    # Dashboard configuration
    refresh_interval_minutes: int = 15
    real_time_monitoring: bool = True
    alert_thresholds: Dict[str, float] = field(default_factory=dict)
    
    # Calculated metrics
    governance_maturity_score: float = 0.0
    policy_coverage_percentage: float = 0.0
    workflow_efficiency_score: float = 0.0
    compliance_health_score: float = 0.0
    
    # Trend data
    historical_metrics: List[Dict[str, Any]] = field(default_factory=list)
    trend_analysis: Dict[str, Any] = field(default_factory=dict)
    
    # Alerts and notifications
    active_alerts: List[Dict[str, Any]] = field(default_factory=list)
    notification_subscribers: List[str] = field(default_factory=list)
    
    # Dashboard metadata
    created_date: datetime = field(default_factory=datetime.now)
    last_refreshed: datetime = field(default_factory=datetime.now)
    
    def calculate_governance_maturity(self) -> float:
        """Calculate overall governance maturity score."""
        factors = []
        
        # Policy maturity
        if self.policy_registry:
            policy_factor = min(1.0, self.policy_registry.active_policies / 50)  # Assuming 50 is mature
            factors.append(policy_factor)
        
        # Workflow maturity
        if self.workflow_engine:
            workflow_factor = min(1.0, self.workflow_engine.total_workflows_processed / 100)
            factors.append(workflow_factor)
        
        # Compliance maturity
        if self.compliance_manager:
            compliance_factor = self.compliance_manager.overall_compliance_score / 100
            factors.append(compliance_factor)
        
        if not factors:
            return 0.0
        
        return sum(factors) / len(factors)
    
    def generate_executive_summary(self) -> Dict[str, Any]:
        """Generate executive summary of governance status."""
        summary = {
            "governance_maturity": self.governance_maturity_score,
            "policy_status": {},
            "compliance_status": {},
            "workflow_status": {},
            "key_risks": [],
            "recommendations": []
        }
        
        # Policy status
        if self.policy_registry:
            summary["policy_status"] = {
                "total_policies": self.policy_registry.total_policies,
                "active_policies": self.policy_registry.active_policies,
                "pending_approval": self.policy_registry.pending_approval
            }
        
        # Compliance status
        if self.compliance_manager:
            summary["compliance_status"] = {
                "overall_score": self.compliance_manager.overall_compliance_score,
                "total_violations": self.compliance_manager.total_violations,
                "critical_violations": self.compliance_manager.critical_violations,
                "active_frameworks": len(self.compliance_manager.active_frameworks)
            }
        
        # Workflow status
        if self.workflow_engine:
            summary["workflow_status"] = {
                "active_workflows": len(self.workflow_engine.active_workflows),
                "completed_workflows": len(self.workflow_engine.completed_workflows),
                "average_completion_time": self.workflow_engine.average_completion_time_hours,
                "escalation_rate": self.workflow_engine.escalation_rate
            }
        
        return summary
    
    def refresh_metrics(self) -> GovernanceDashboard:
        """Refresh dashboard metrics."""
        new_maturity_score = self.calculate_governance_maturity()
        
        # Record historical metrics
        metric_snapshot = {
            "timestamp": datetime.now(),
            "governance_maturity": new_maturity_score,
            "policy_count": self.policy_registry.active_policies if self.policy_registry else 0,
            "compliance_score": self.compliance_manager.overall_compliance_score if self.compliance_manager else 0,
            "workflow_count": len(self.workflow_engine.active_workflows) if self.workflow_engine else 0
        }
        
        new_historical = list(self.historical_metrics) + [metric_snapshot]
        
        # Keep only last 90 days of metrics
        cutoff_date = datetime.now() - timedelta(days=90)
        new_historical = [m for m in new_historical if m["timestamp"] >= cutoff_date]
        
        return dataclass.replace(
            self,
            governance_maturity_score=new_maturity_score,
            historical_metrics=new_historical,
            last_refreshed=datetime.now()
        )