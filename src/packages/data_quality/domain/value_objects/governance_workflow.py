"""Governance workflow value objects for workflow automation and approval processes."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Union
from datetime import datetime, timedelta
from uuid import UUID, uuid4


class WorkflowType(str, Enum):
    """Types of governance workflows."""
    POLICY_APPROVAL = "policy_approval"
    EXCEPTION_REQUEST = "exception_request"
    QUALITY_ISSUE_ESCALATION = "quality_issue_escalation"
    DATA_ACCESS_REQUEST = "data_access_request"
    COMPLIANCE_REVIEW = "compliance_review"
    STEWARDSHIP_ASSIGNMENT = "stewardship_assignment"
    AUDIT_INVESTIGATION = "audit_investigation"
    INCIDENT_RESPONSE = "incident_response"


class WorkflowStatus(str, Enum):
    """Workflow execution status."""
    DRAFT = "draft"
    INITIATED = "initiated"
    IN_PROGRESS = "in_progress"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    ON_HOLD = "on_hold"
    CANCELLED = "cancelled"
    COMPLETED = "completed"
    EXPIRED = "expired"


class TaskStatus(str, Enum):
    """Individual task status within workflow."""
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"
    ESCALATED = "escalated"


class TaskType(str, Enum):
    """Types of workflow tasks."""
    APPROVAL = "approval"
    REVIEW = "review"
    INVESTIGATION = "investigation"
    DOCUMENTATION = "documentation"
    NOTIFICATION = "notification"
    VALIDATION = "validation"
    EXECUTION = "execution"
    MONITORING = "monitoring"


class EscalationTrigger(str, Enum):
    """Triggers for workflow escalation."""
    TIMEOUT = "timeout"
    REJECTION = "rejection"
    FAILURE = "failure"
    MANUAL = "manual"
    RISK_THRESHOLD = "risk_threshold"
    COMPLIANCE_VIOLATION = "compliance_violation"


@dataclass(frozen=True)
class WorkflowParticipant:
    """Participant in a governance workflow."""
    participant_id: str
    participant_type: str  # user, role, group, system
    name: str
    email: Optional[str] = None
    role: Optional[str] = None
    permissions: List[str] = field(default_factory=list)
    notification_preferences: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def can_approve(self) -> bool:
        """Check if participant can approve tasks."""
        return "approve" in self.permissions
    
    @property
    def can_escalate(self) -> bool:
        """Check if participant can escalate tasks."""
        return "escalate" in self.permissions


@dataclass(frozen=True)
class TaskAssignment:
    """Assignment of a task to a participant."""
    assigned_to: WorkflowParticipant
    assigned_date: datetime
    due_date: Optional[datetime] = None
    priority: str = "medium"  # low, medium, high, urgent
    delegation_allowed: bool = True
    escalation_rules: List[str] = field(default_factory=list)
    
    @property
    def is_overdue(self) -> bool:
        """Check if task assignment is overdue."""
        if not self.due_date:
            return False
        return datetime.now() > self.due_date
    
    @property
    def days_until_due(self) -> Optional[int]:
        """Get days until task is due."""
        if not self.due_date:
            return None
        delta = self.due_date - datetime.now()
        return max(0, delta.days)


@dataclass(frozen=True)
class TaskAction:
    """Action taken on a workflow task."""
    action_id: str
    action_type: str  # approve, reject, delegate, comment, escalate
    actor: WorkflowParticipant
    timestamp: datetime
    comments: str = ""
    decision_rationale: str = ""
    attachments: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_decision_action(self) -> bool:
        """Check if action is a decision (approve/reject)."""
        return self.action_type in ["approve", "reject"]


@dataclass(frozen=True)
class WorkflowTask:
    """Individual task within a governance workflow."""
    task_id: str
    name: str
    description: str
    task_type: TaskType
    status: TaskStatus = TaskStatus.PENDING
    
    # Assignment and timing
    assignment: Optional[TaskAssignment] = None
    created_date: datetime = field(default_factory=datetime.now)
    started_date: Optional[datetime] = None
    completed_date: Optional[datetime] = None
    
    # Task configuration
    is_mandatory: bool = True
    is_parallel: bool = False  # Can be executed in parallel with other tasks
    depends_on: List[str] = field(default_factory=list)  # Task IDs this task depends on
    auto_approve_conditions: List[str] = field(default_factory=list)
    
    # Actions and history
    actions: List[TaskAction] = field(default_factory=list)
    escalations: List[str] = field(default_factory=list)
    
    # Content and context
    task_data: Dict[str, Any] = field(default_factory=dict)
    required_information: List[str] = field(default_factory=list)
    approval_criteria: List[str] = field(default_factory=list)
    
    @property
    def is_completed(self) -> bool:
        """Check if task is completed."""
        return self.status == TaskStatus.COMPLETED
    
    @property
    def is_overdue(self) -> bool:
        """Check if task is overdue."""
        return self.assignment and self.assignment.is_overdue
    
    @property
    def latest_action(self) -> Optional[TaskAction]:
        """Get the latest action taken on this task."""
        if not self.actions:
            return None
        return max(self.actions, key=lambda a: a.timestamp)
    
    @property
    def duration_hours(self) -> Optional[float]:
        """Get task duration in hours."""
        if not self.started_date or not self.completed_date:
            return None
        delta = self.completed_date - self.started_date
        return delta.total_seconds() / 3600
    
    def can_be_started(self, completed_tasks: List[str]) -> bool:
        """Check if task can be started based on dependencies."""
        if not self.depends_on:
            return True
        return all(dep_id in completed_tasks for dep_id in self.depends_on)
    
    def add_action(self, action: TaskAction) -> WorkflowTask:
        """Add action to task."""
        new_actions = list(self.actions) + [action]
        new_status = self.status
        
        # Update status based on action
        if action.action_type == "approve" and self.task_type == TaskType.APPROVAL:
            new_status = TaskStatus.COMPLETED
        elif action.action_type == "reject":
            new_status = TaskStatus.FAILED
        elif action.action_type == "escalate":
            new_status = TaskStatus.ESCALATED
        
        return dataclass.replace(
            self,
            actions=new_actions,
            status=new_status,
            completed_date=datetime.now() if new_status == TaskStatus.COMPLETED else self.completed_date
        )


@dataclass(frozen=True)
class EscalationRule:
    """Rule for workflow escalation."""
    rule_id: str
    trigger: EscalationTrigger
    condition: str  # Condition expression
    escalate_to: WorkflowParticipant
    escalation_delay: timedelta
    max_escalations: int = 3
    escalation_message: str = ""
    is_active: bool = True
    
    def should_escalate(self, workflow_context: Dict[str, Any]) -> bool:
        """Check if escalation should be triggered."""
        if not self.is_active:
            return False
        
        # Simple condition evaluation - can be extended with rule engine
        if self.trigger == EscalationTrigger.TIMEOUT:
            return workflow_context.get("is_timeout", False)
        elif self.trigger == EscalationTrigger.REJECTION:
            return workflow_context.get("rejection_count", 0) > 0
        
        return False


@dataclass(frozen=True)
class WorkflowTemplate:
    """Template for creating governance workflows."""
    template_id: str
    name: str
    description: str
    workflow_type: WorkflowType
    version: str = "1.0.0"
    
    # Template structure
    task_templates: List[Dict[str, Any]] = field(default_factory=list)
    participant_roles: List[str] = field(default_factory=list)
    escalation_rules: List[EscalationRule] = field(default_factory=list)
    
    # Configuration
    default_sla: Optional[timedelta] = None
    auto_approval_enabled: bool = False
    parallel_execution_enabled: bool = False
    
    # Compliance and governance
    compliance_requirements: List[str] = field(default_factory=list)
    audit_requirements: List[str] = field(default_factory=list)
    
    def create_workflow(
        self,
        workflow_id: str,
        title: str,
        requester: WorkflowParticipant,
        context_data: Dict[str, Any]
    ) -> GovernanceWorkflow:
        """Create workflow instance from template."""
        # Create tasks from templates
        tasks = []
        for i, task_template in enumerate(self.task_templates):
            task = WorkflowTask(
                task_id=f"{workflow_id}_task_{i+1}",
                name=task_template["name"],
                description=task_template["description"],
                task_type=TaskType(task_template["task_type"]),
                is_mandatory=task_template.get("is_mandatory", True),
                is_parallel=task_template.get("is_parallel", False),
                depends_on=task_template.get("depends_on", []),
                task_data=task_template.get("task_data", {}),
                approval_criteria=task_template.get("approval_criteria", [])
            )
            tasks.append(task)
        
        return GovernanceWorkflow(
            workflow_id=workflow_id,
            title=title,
            workflow_type=self.workflow_type,
            template_id=self.template_id,
            requester=requester,
            tasks=tasks,
            escalation_rules=self.escalation_rules,
            context_data=context_data,
            sla_deadline=datetime.now() + self.default_sla if self.default_sla else None
        )


@dataclass(frozen=True)
class WorkflowMetrics:
    """Metrics for workflow performance analysis."""
    total_workflows: int = 0
    completed_workflows: int = 0
    average_completion_time_hours: float = 0.0
    on_time_completion_rate: float = 0.0
    escalation_rate: float = 0.0
    rejection_rate: float = 0.0
    participant_workload: Dict[str, int] = field(default_factory=dict)
    bottleneck_tasks: List[str] = field(default_factory=list)
    
    @property
    def completion_rate(self) -> float:
        """Calculate workflow completion rate."""
        if self.total_workflows == 0:
            return 0.0
        return self.completed_workflows / self.total_workflows
    
    @property
    def efficiency_score(self) -> float:
        """Calculate overall workflow efficiency score."""
        factors = [
            self.completion_rate,
            self.on_time_completion_rate,
            1.0 - self.escalation_rate,  # Lower escalation is better
            1.0 - self.rejection_rate     # Lower rejection is better
        ]
        return sum(factors) / len(factors)


@dataclass
class GovernanceWorkflow:
    """Comprehensive governance workflow instance."""
    workflow_id: str
    title: str
    workflow_type: WorkflowType
    template_id: Optional[str] = None
    
    # Workflow lifecycle
    status: WorkflowStatus = WorkflowStatus.DRAFT
    created_date: datetime = field(default_factory=datetime.now)
    started_date: Optional[datetime] = None
    completed_date: Optional[datetime] = None
    last_modified: datetime = field(default_factory=datetime.now)
    
    # Participants and assignments
    requester: Optional[WorkflowParticipant] = None
    current_assignees: List[WorkflowParticipant] = field(default_factory=list)
    all_participants: List[WorkflowParticipant] = field(default_factory=list)
    
    # Tasks and execution
    tasks: List[WorkflowTask] = field(default_factory=list)
    current_task_index: int = 0
    parallel_tasks: List[str] = field(default_factory=list)
    
    # Escalation and SLA
    escalation_rules: List[EscalationRule] = field(default_factory=list)
    escalation_history: List[Dict[str, Any]] = field(default_factory=list)
    sla_deadline: Optional[datetime] = None
    
    # Context and metadata
    context_data: Dict[str, Any] = field(default_factory=dict)
    priority: str = "medium"  # low, medium, high, urgent
    business_impact: str = "medium"  # low, medium, high, critical
    
    # Outcomes and decisions
    final_decision: Optional[str] = None
    decision_rationale: str = ""
    implementation_plan: List[str] = field(default_factory=list)
    
    @property
    def is_active(self) -> bool:
        """Check if workflow is currently active."""
        active_statuses = [
            WorkflowStatus.INITIATED,
            WorkflowStatus.IN_PROGRESS,
            WorkflowStatus.PENDING_APPROVAL
        ]
        return self.status in active_statuses
    
    @property
    def is_overdue(self) -> bool:
        """Check if workflow is overdue."""
        if not self.sla_deadline:
            return False
        return datetime.now() > self.sla_deadline and self.is_active
    
    @property
    def current_tasks(self) -> List[WorkflowTask]:
        """Get currently active tasks."""
        if not self.tasks:
            return []
        
        # Return parallel tasks if any are active
        if self.parallel_tasks:
            return [task for task in self.tasks if task.task_id in self.parallel_tasks]
        
        # Return current sequential task
        if self.current_task_index < len(self.tasks):
            return [self.tasks[self.current_task_index]]
        
        return []
    
    @property
    def completed_tasks(self) -> List[WorkflowTask]:
        """Get completed tasks."""
        return [task for task in self.tasks if task.is_completed]
    
    @property
    def progress_percentage(self) -> float:
        """Calculate workflow progress percentage."""
        if not self.tasks:
            return 0.0
        
        completed = len(self.completed_tasks)
        total = len(self.tasks)
        return (completed / total) * 100
    
    @property
    def duration_hours(self) -> Optional[float]:
        """Get workflow duration in hours."""
        if not self.started_date:
            return None
        
        end_time = self.completed_date or datetime.now()
        delta = end_time - self.started_date
        return delta.total_seconds() / 3600
    
    def can_be_approved_by(self, participant: WorkflowParticipant) -> bool:
        """Check if participant can approve current tasks."""
        current_tasks = self.current_tasks
        if not current_tasks:
            return False
        
        for task in current_tasks:
            if (task.assignment and 
                task.assignment.assigned_to.participant_id == participant.participant_id and
                task.task_type == TaskType.APPROVAL):
                return True
        
        return False
    
    def advance_workflow(self) -> GovernanceWorkflow:
        """Advance workflow to next task."""
        completed_task_ids = [task.task_id for task in self.completed_tasks]
        
        # Find next available tasks
        next_tasks = []
        for i, task in enumerate(self.tasks):
            if (not task.is_completed and 
                task.can_be_started(completed_task_ids)):
                next_tasks.append((i, task))
        
        if not next_tasks:
            # No more tasks, complete workflow
            return dataclass.replace(
                self,
                status=WorkflowStatus.COMPLETED,
                completed_date=datetime.now(),
                last_modified=datetime.now()
            )
        
        # Update current task index and parallel tasks
        if len(next_tasks) == 1:
            new_index = next_tasks[0][0]
            new_parallel = []
        else:
            # Multiple tasks can be executed in parallel
            new_index = min(i for i, _ in next_tasks)
            new_parallel = [task.task_id for _, task in next_tasks if task.is_parallel]
        
        return dataclass.replace(
            self,
            current_task_index=new_index,
            parallel_tasks=new_parallel,
            status=WorkflowStatus.IN_PROGRESS,
            last_modified=datetime.now()
        )
    
    def escalate_workflow(
        self,
        trigger: EscalationTrigger,
        escalated_by: WorkflowParticipant,
        reason: str
    ) -> GovernanceWorkflow:
        """Escalate workflow to higher authority."""
        escalation_record = {
            "timestamp": datetime.now(),
            "trigger": trigger.value,
            "escalated_by": escalated_by.participant_id,
            "reason": reason,
            "workflow_status": self.status.value
        }
        
        new_escalation_history = list(self.escalation_history) + [escalation_record]
        
        return dataclass.replace(
            self,
            escalation_history=new_escalation_history,
            priority="high" if self.priority != "urgent" else "urgent",
            last_modified=datetime.now()
        )
    
    def complete_workflow(
        self,
        final_decision: str,
        decision_rationale: str,
        completed_by: WorkflowParticipant
    ) -> GovernanceWorkflow:
        """Complete workflow with final decision."""
        completion_action = TaskAction(
            action_id=f"completion_{uuid4().hex[:8]}",
            action_type="complete",
            actor=completed_by,
            timestamp=datetime.now(),
            comments=decision_rationale,
            decision_rationale=decision_rationale
        )
        
        return dataclass.replace(
            self,
            status=WorkflowStatus.COMPLETED,
            completed_date=datetime.now(),
            final_decision=final_decision,
            decision_rationale=decision_rationale,
            last_modified=datetime.now()
        )
    
    @classmethod
    def create_policy_approval_workflow(
        cls,
        policy_id: str,
        requester: WorkflowParticipant,
        approvers: List[WorkflowParticipant],
        **kwargs
    ) -> GovernanceWorkflow:
        """Create policy approval workflow."""
        workflow_id = f"policy_approval_{policy_id}_{uuid4().hex[:8]}"
        
        # Create approval tasks
        tasks = []
        for i, approver in enumerate(approvers):
            task = WorkflowTask(
                task_id=f"{workflow_id}_approval_{i+1}",
                name=f"Policy Review and Approval",
                description=f"Review and approve policy {policy_id}",
                task_type=TaskType.APPROVAL,
                assignment=TaskAssignment(
                    assigned_to=approver,
                    assigned_date=datetime.now(),
                    due_date=datetime.now() + timedelta(days=3),
                    priority="high"
                ),
                approval_criteria=[
                    "Policy content is complete and accurate",
                    "Policy aligns with organizational standards",
                    "Policy compliance requirements are met"
                ]
            )
            tasks.append(task)
        
        return cls(
            workflow_id=workflow_id,
            title=f"Policy Approval: {policy_id}",
            workflow_type=WorkflowType.POLICY_APPROVAL,
            requester=requester,
            tasks=tasks,
            all_participants=[requester] + approvers,
            context_data={"policy_id": policy_id},
            sla_deadline=datetime.now() + timedelta(days=7),
            priority="high",
            **kwargs
        )