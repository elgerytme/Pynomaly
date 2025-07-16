"""Comprehensive governance framework service with audit trails and policy management.

This service provides enterprise-grade governance capabilities including:
- Complete audit trail tracking for all system operations
- Policy management with enforcement and compliance monitoring
- Data lineage tracking for full data lifecycle visibility
- Access control governance with permission auditing
- Model governance with version control and approval workflows
- Regulatory compliance with automated reporting and alerts
- Risk management with assessment and mitigation tracking
- Change management with approval workflows and rollback capabilities
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

from pynomaly_detection.domain.entities.governance_workflow import (
    ApprovalStatus,
    ComplianceStatus,
)

logger = logging.getLogger(__name__)


class GovernanceAction(Enum):
    """Types of governance actions that can be tracked."""

    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    EXECUTE = "execute"
    APPROVE = "approve"
    REJECT = "reject"
    DEPLOY = "deploy"
    ROLLBACK = "rollback"
    ARCHIVE = "archive"
    RESTORE = "restore"
    EXPORT = "export"
    IMPORT = "import"
    CONFIGURE = "configure"
    MONITOR = "monitor"


class PolicyType(Enum):
    """Types of governance policies."""

    DATA_ACCESS = "data_access"
    MODEL_DEPLOYMENT = "model_deployment"
    ALGORITHM_USAGE = "algorithm_usage"
    DATA_RETENTION = "data_retention"
    EXPORT_CONTROL = "export_control"
    COMPLIANCE = "compliance"
    SECURITY = "security"
    PERFORMANCE = "performance"
    QUALITY = "quality"
    CHANGE_MANAGEMENT = "change_management"


class PolicyStatus(Enum):
    """Status of governance policies."""

    DRAFT = "draft"
    ACTIVE = "active"
    INACTIVE = "inactive"
    DEPRECATED = "deprecated"
    UNDER_REVIEW = "under_review"


class RiskLevel(Enum):
    """Risk levels for governance assessments."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ComplianceStatus(Enum):
    """Compliance status for governance items."""

    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    NOT_ASSESSED = "not_assessed"
    UNDER_REVIEW = "under_review"
    REMEDIATION_REQUIRED = "remediation_required"


@dataclass
class RiskAssessment:
    """Risk assessment for governance items."""

    risk_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    risk_level: RiskLevel = RiskLevel.MEDIUM
    risk_score: float = 0.5
    risk_factors: list[str] = field(default_factory=list)
    mitigation_strategies: list[str] = field(default_factory=list)
    assessment_date: datetime = field(default_factory=datetime.utcnow)
    assessed_by: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChangeRequest:
    """Change request for governance workflows."""

    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    description: str = ""
    change_type: str = "configuration"
    impact_assessment: str = ""
    risk_assessment: RiskAssessment | None = None
    approval_status: str = "pending"
    requested_by: str = ""
    requested_at: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ComplianceMetric:
    """Compliance metric tracking."""

    metric_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metric_name: str = ""
    metric_value: float = 0.0
    target_value: float = 1.0
    metric_type: str = "percentage"
    measurement_date: datetime = field(default_factory=datetime.utcnow)
    compliance_status: ComplianceStatus = ComplianceStatus.NOT_ASSESSED
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GovernancePolicy:
    """Governance policy definition."""

    policy_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    policy_name: str = ""
    policy_type: PolicyType = PolicyType.COMPLIANCE
    policy_status: PolicyStatus = PolicyStatus.DRAFT
    policy_content: str = ""
    enforcement_level: str = "mandatory"
    created_by: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    effective_date: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AuditTrailEntry:
    """Individual audit trail entry with comprehensive tracking."""

    entry_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    action: GovernanceAction = GovernanceAction.READ
    resource_type: str = ""
    resource_id: str = ""
    resource_name: str = ""
    user_id: str = ""
    user_name: str = ""
    user_role: str = ""
    session_id: str | None = None
    source_ip: str | None = None
    user_agent: str | None = None
    description: str = ""
    before_state: dict[str, Any] | None = None
    after_state: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    risk_level: RiskLevel = RiskLevel.LOW
    compliance_frameworks: list[str] = field(default_factory=list)
    policy_violations: list[str] = field(default_factory=list)
    approval_required: bool = False
    approved_by: str | None = None
    approval_timestamp: datetime | None = None

    def get_change_summary(self) -> dict[str, Any]:
        """Get summary of changes made."""
        changes = {}

        if self.before_state and self.after_state:
            for key in set(self.before_state.keys()) | set(self.after_state.keys()):
                before_value = self.before_state.get(key)
                after_value = self.after_state.get(key)

                if before_value != after_value:
                    changes[key] = {"before": before_value, "after": after_value}

        return changes

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "entry_id": self.entry_id,
            "timestamp": self.timestamp.isoformat(),
            "action": self.action.value,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "resource_name": self.resource_name,
            "user_id": self.user_id,
            "user_name": self.user_name,
            "user_role": self.user_role,
            "session_id": self.session_id,
            "source_ip": self.source_ip,
            "description": self.description,
            "before_state": self.before_state,
            "after_state": self.after_state,
            "metadata": self.metadata,
            "risk_level": self.risk_level.value,
            "compliance_frameworks": self.compliance_frameworks,
            "policy_violations": self.policy_violations,
            "approval_required": self.approval_required,
            "approved_by": self.approved_by,
            "approval_timestamp": (
                self.approval_timestamp.isoformat() if self.approval_timestamp else None
            ),
            "changes": self.get_change_summary(),
        }


@dataclass
class GovernancePolicy:
    """Governance policy definition with enforcement rules."""

    policy_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    policy_type: PolicyType = PolicyType.DATA_ACCESS
    status: PolicyStatus = PolicyStatus.DRAFT
    version: str = "1.0.0"
    created_by: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    effective_date: datetime = field(default_factory=datetime.utcnow)
    expiration_date: datetime | None = None
    approval_required: bool = True
    approved_by: str | None = None
    approval_date: datetime | None = None
    enforcement_level: str = "strict"  # strict, warning, advisory
    scope: dict[str, Any] = field(default_factory=dict)
    rules: list[dict[str, Any]] = field(default_factory=list)
    exceptions: list[dict[str, Any]] = field(default_factory=list)
    compliance_frameworks: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    def is_active(self) -> bool:
        """Check if policy is currently active."""
        now = datetime.utcnow()

        if self.status != PolicyStatus.ACTIVE:
            return False

        if now < self.effective_date:
            return False

        if self.expiration_date and now > self.expiration_date:
            return False

        return True

    def evaluate_compliance(self, context: dict[str, Any]) -> tuple[bool, list[str]]:
        """Evaluate compliance for given context."""
        violations = []

        if not self.is_active():
            return True, violations

        # Evaluate each rule
        for rule in self.rules:
            rule_type = rule.get("type", "")
            conditions = rule.get("conditions", {})

            if not self._evaluate_rule(rule_type, conditions, context):
                violations.append(
                    rule.get("message", f"Policy rule violation: {rule_type}")
                )

        is_compliant = len(violations) == 0
        return is_compliant, violations

    def _evaluate_rule(
        self, rule_type: str, conditions: dict[str, Any], context: dict[str, Any]
    ) -> bool:
        """Evaluate individual policy rule."""
        # Simplified rule evaluation - in production would be more sophisticated

        if rule_type == "user_role_required":
            required_roles = conditions.get("roles", [])
            user_role = context.get("user_role", "")
            return user_role in required_roles

        elif rule_type == "resource_access_time":
            allowed_hours = conditions.get("hours", [])
            current_hour = datetime.utcnow().hour
            return current_hour in allowed_hours

        elif rule_type == "data_classification":
            max_classification = conditions.get("max_classification", "public")
            data_classification = context.get("data_classification", "public")

            classification_levels = {
                "public": 0,
                "internal": 1,
                "confidential": 2,
                "restricted": 3,
            }
            max_level = classification_levels.get(max_classification, 0)
            data_level = classification_levels.get(data_classification, 0)

            return data_level <= max_level

        # Default to compliant if rule type not recognized
        return True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "policy_id": self.policy_id,
            "name": self.name,
            "description": self.description,
            "policy_type": self.policy_type.value,
            "status": self.status.value,
            "version": self.version,
            "created_by": self.created_by,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "effective_date": self.effective_date.isoformat(),
            "expiration_date": (
                self.expiration_date.isoformat() if self.expiration_date else None
            ),
            "approval_required": self.approval_required,
            "approved_by": self.approved_by,
            "approval_date": (
                self.approval_date.isoformat() if self.approval_date else None
            ),
            "enforcement_level": self.enforcement_level,
            "is_active": self.is_active(),
            "scope": self.scope,
            "rules": self.rules,
            "exceptions": self.exceptions,
            "compliance_frameworks": self.compliance_frameworks,
            "tags": self.tags,
        }


@dataclass
class DataLineageEntry:
    """Data lineage tracking entry."""

    lineage_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    source_type: str = ""  # dataset, model, result, export
    source_id: str = ""
    source_name: str = ""
    operation: str = ""  # transform, train, predict, export
    destination_type: str = ""
    destination_id: str = ""
    destination_name: str = ""
    transformation_details: dict[str, Any] = field(default_factory=dict)
    quality_metrics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "lineage_id": self.lineage_id,
            "timestamp": self.timestamp.isoformat(),
            "source_type": self.source_type,
            "source_id": self.source_id,
            "source_name": self.source_name,
            "operation": self.operation,
            "destination_type": self.destination_type,
            "destination_id": self.destination_id,
            "destination_name": self.destination_name,
            "transformation_details": self.transformation_details,
            "quality_metrics": self.quality_metrics,
            "metadata": self.metadata,
        }


@dataclass
class ComplianceReport:
    """Comprehensive compliance assessment report."""

    report_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    generated_at: datetime = field(default_factory=datetime.utcnow)
    report_period_start: datetime = field(
        default_factory=lambda: datetime.utcnow() - timedelta(days=30)
    )
    report_period_end: datetime = field(default_factory=datetime.utcnow)
    compliance_frameworks: list[str] = field(default_factory=list)
    overall_compliance_score: float = 0.0
    policy_compliance: dict[str, dict[str, Any]] = field(default_factory=dict)
    violations: list[dict[str, Any]] = field(default_factory=list)
    remediation_items: list[dict[str, Any]] = field(default_factory=list)
    audit_summary: dict[str, Any] = field(default_factory=dict)
    recommendations: list[str] = field(default_factory=list)
    next_review_date: datetime = field(
        default_factory=lambda: datetime.utcnow() + timedelta(days=90)
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "report_id": self.report_id,
            "generated_at": self.generated_at.isoformat(),
            "report_period_start": self.report_period_start.isoformat(),
            "report_period_end": self.report_period_end.isoformat(),
            "compliance_frameworks": self.compliance_frameworks,
            "overall_compliance_score": self.overall_compliance_score,
            "policy_compliance": self.policy_compliance,
            "violations": self.violations,
            "remediation_items": self.remediation_items,
            "audit_summary": self.audit_summary,
            "recommendations": self.recommendations,
            "next_review_date": self.next_review_date.isoformat(),
        }


class GovernanceFrameworkService:
    """Comprehensive governance framework service for enterprise compliance and audit trails.

    This service provides:
    - Complete audit trail tracking for all operations
    - Policy management with automated enforcement
    - Data lineage tracking for full lifecycle visibility
    - Compliance monitoring and reporting
    - Risk assessment and management
    - Change management with approval workflows
    """

    def __init__(
        self,
        storage_path: Path,
        retention_years: int = 7,
        enable_real_time_monitoring: bool = True,
    ):
        """Initialize governance framework service.

        Args:
            storage_path: Path for storing governance artifacts
            retention_years: Number of years to retain audit data
            enable_real_time_monitoring: Enable real-time compliance monitoring
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.retention_years = retention_years
        self.enable_real_time_monitoring = enable_real_time_monitoring

        # Audit trail storage
        self.audit_trail: list[AuditTrailEntry] = []
        self.audit_trail_file = self.storage_path / "audit_trail.jsonl"

        # Policy storage
        self.policies: dict[str, GovernancePolicy] = {}
        self.policies_file = self.storage_path / "policies.json"

        # Risk assessments storage
        self.risk_assessments: dict[str, RiskAssessment] = {}
        self.risk_assessments_file = self.storage_path / "risk_assessments.json"

        # Change requests storage
        self.change_requests: dict[str, ChangeRequest] = {}
        self.change_requests_file = self.storage_path / "change_requests.json"

        # Data lineage storage
        self.data_lineage: list[DataLineageEntry] = []
        self.lineage_file = self.storage_path / "data_lineage.jsonl"

        # Compliance monitoring
        self.compliance_violations: list[dict[str, Any]] = []
        self.compliance_file = self.storage_path / "compliance_violations.json"

        # Compliance metrics storage
        self.compliance_metrics: dict[str, ComplianceMetric] = {}
        self.compliance_metrics_file = self.storage_path / "compliance_metrics.json"

        # Load existing data
        self._load_policies()
        self._load_audit_trail()
        self._load_data_lineage()

        logger.info("Governance framework service initialized")

    async def create_audit_entry(
        self,
        action: GovernanceAction,
        resource_type: str,
        resource_id: str,
        resource_name: str,
        user_id: str,
        user_name: str = "",
        user_role: str = "",
        description: str = "",
        before_state: dict[str, Any] | None = None,
        after_state: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        session_id: str | None = None,
        source_ip: str | None = None,
    ) -> AuditTrailEntry:
        """Create comprehensive audit trail entry.

        Args:
            action: Type of action performed
            resource_type: Type of resource affected
            resource_id: ID of the resource
            resource_name: Name of the resource
            user_id: ID of user performing action
            user_name: Name of user
            user_role: Role of user
            description: Description of action
            before_state: State before action
            after_state: State after action
            metadata: Additional metadata
            session_id: User session ID
            source_ip: Source IP address

        Returns:
            Created audit trail entry
        """
        try:
            # Assess risk level
            risk_level = await self._assess_action_risk(
                action, resource_type, user_role
            )

            # Check policy compliance
            compliance_context = {
                "action": action.value,
                "resource_type": resource_type,
                "resource_id": resource_id,
                "user_id": user_id,
                "user_role": user_role,
                "timestamp": datetime.utcnow(),
            }

            policy_violations = await self._check_policy_compliance(compliance_context)

            # Create audit entry
            entry = AuditTrailEntry(
                action=action,
                resource_type=resource_type,
                resource_id=resource_id,
                resource_name=resource_name,
                user_id=user_id,
                user_name=user_name,
                user_role=user_role,
                description=description,
                before_state=before_state,
                after_state=after_state,
                metadata=metadata or {},
                session_id=session_id,
                source_ip=source_ip,
                risk_level=risk_level,
                policy_violations=policy_violations,
            )

            # Store audit entry
            self.audit_trail.append(entry)
            await self._persist_audit_entry(entry)

            # Real-time monitoring
            if self.enable_real_time_monitoring:
                await self._process_real_time_alerts(entry)

            logger.debug(f"Audit entry created: {entry.entry_id}")
            return entry

        except Exception as e:
            logger.error(f"Failed to create audit entry: {e}")
            raise

    async def create_policy(
        self,
        name: str,
        description: str,
        policy_type: PolicyType,
        rules: list[dict[str, Any]],
        created_by: str,
        enforcement_level: str = "strict",
        scope: dict[str, Any] | None = None,
        compliance_frameworks: list[str] | None = None,
    ) -> GovernancePolicy:
        """Create new governance policy.

        Args:
            name: Policy name
            description: Policy description
            policy_type: Type of policy
            rules: Policy rules
            created_by: User creating policy
            enforcement_level: Enforcement level (strict, warning, advisory)
            scope: Policy scope
            compliance_frameworks: Applicable compliance frameworks

        Returns:
            Created governance policy
        """
        try:
            policy = GovernancePolicy(
                name=name,
                description=description,
                policy_type=policy_type,
                created_by=created_by,
                enforcement_level=enforcement_level,
                scope=scope or {},
                rules=rules,
                compliance_frameworks=compliance_frameworks or [],
            )

            self.policies[policy.policy_id] = policy
            await self._persist_policies()

            # Create audit entry for policy creation
            await self.create_audit_entry(
                action=GovernanceAction.CREATE,
                resource_type="policy",
                resource_id=policy.policy_id,
                resource_name=policy.name,
                user_id=created_by,
                description=f"Created governance policy: {policy.name}",
                after_state=policy.to_dict(),
            )

            logger.info(f"Governance policy created: {policy.name}")
            return policy

        except Exception as e:
            logger.error(f"Failed to create governance policy: {e}")
            raise

    async def activate_policy(self, policy_id: str, activated_by: str) -> bool:
        """Activate governance policy.

        Args:
            policy_id: Policy to activate
            activated_by: User activating policy

        Returns:
            True if successful
        """
        try:
            if policy_id not in self.policies:
                raise ValueError(f"Policy {policy_id} not found")

            policy = self.policies[policy_id]
            before_state = policy.to_dict()

            policy.status = PolicyStatus.ACTIVE
            policy.updated_at = datetime.utcnow()

            await self._persist_policies()

            # Create audit entry
            await self.create_audit_entry(
                action=GovernanceAction.UPDATE,
                resource_type="policy",
                resource_id=policy_id,
                resource_name=policy.name,
                user_id=activated_by,
                description=f"Activated governance policy: {policy.name}",
                before_state=before_state,
                after_state=policy.to_dict(),
            )

            logger.info(f"Policy activated: {policy.name}")
            return True

        except Exception as e:
            logger.error(f"Failed to activate policy: {e}")
            raise

    async def track_data_lineage(
        self,
        source_type: str,
        source_id: str,
        source_name: str,
        operation: str,
        destination_type: str,
        destination_id: str,
        destination_name: str,
        transformation_details: dict[str, Any] | None = None,
        quality_metrics: dict[str, Any] | None = None,
    ) -> DataLineageEntry:
        """Track data lineage for complete lifecycle visibility.

        Args:
            source_type: Type of source (dataset, model, result)
            source_id: Source identifier
            source_name: Source name
            operation: Operation performed (transform, train, predict)
            destination_type: Type of destination
            destination_id: Destination identifier
            destination_name: Destination name
            transformation_details: Details of transformation
            quality_metrics: Data quality metrics

        Returns:
            Created lineage entry
        """
        try:
            lineage_entry = DataLineageEntry(
                source_type=source_type,
                source_id=source_id,
                source_name=source_name,
                operation=operation,
                destination_type=destination_type,
                destination_id=destination_id,
                destination_name=destination_name,
                transformation_details=transformation_details or {},
                quality_metrics=quality_metrics or {},
            )

            self.data_lineage.append(lineage_entry)
            await self._persist_lineage_entry(lineage_entry)

            logger.debug(f"Data lineage tracked: {source_name} -> {destination_name}")
            return lineage_entry

        except Exception as e:
            logger.error(f"Failed to track data lineage: {e}")
            raise

    async def generate_compliance_report(
        self, frameworks: list[str] | None = None, period_days: int = 30
    ) -> ComplianceReport:
        """Generate comprehensive compliance report.

        Args:
            frameworks: Compliance frameworks to assess
            period_days: Report period in days

        Returns:
            Compliance assessment report
        """
        try:
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=period_days)

            # Filter audit entries for period
            period_entries = [
                entry
                for entry in self.audit_trail
                if start_date <= entry.timestamp <= end_date
            ]

            # Calculate compliance metrics
            total_entries = len(period_entries)
            violation_entries = [
                entry for entry in period_entries if entry.policy_violations
            ]
            compliance_score = (
                (total_entries - len(violation_entries)) / max(total_entries, 1)
            ) * 100

            # Analyze policy compliance
            policy_compliance = {}
            for policy_id, policy in self.policies.items():
                if policy.is_active():
                    policy_violations = [
                        entry
                        for entry in period_entries
                        if policy_id in entry.policy_violations
                    ]

                    policy_compliance[policy_id] = {
                        "policy_name": policy.name,
                        "total_checks": total_entries,
                        "violations": len(policy_violations),
                        "compliance_rate": (
                            (total_entries - len(policy_violations))
                            / max(total_entries, 1)
                        )
                        * 100,
                    }

            # Compile violations
            violations = []
            for entry in violation_entries:
                violations.append(
                    {
                        "entry_id": entry.entry_id,
                        "timestamp": entry.timestamp.isoformat(),
                        "action": entry.action.value,
                        "resource_type": entry.resource_type,
                        "user_id": entry.user_id,
                        "violations": entry.policy_violations,
                        "risk_level": entry.risk_level.value,
                    }
                )

            # Generate recommendations
            recommendations = await self._generate_compliance_recommendations(
                period_entries
            )

            # Create report
            report = ComplianceReport(
                report_period_start=start_date,
                report_period_end=end_date,
                compliance_frameworks=frameworks or [],
                overall_compliance_score=compliance_score,
                policy_compliance=policy_compliance,
                violations=violations,
                audit_summary={
                    "total_audit_entries": total_entries,
                    "violation_entries": len(violation_entries),
                    "unique_users": len({entry.user_id for entry in period_entries}),
                    "resource_types": len(
                        {entry.resource_type for entry in period_entries}
                    ),
                    "high_risk_actions": len(
                        [e for e in period_entries if e.risk_level == RiskLevel.HIGH]
                    ),
                },
                recommendations=recommendations,
            )

            logger.info(
                f"Compliance report generated: {compliance_score:.1f}% compliance"
            )
            return report

        except Exception as e:
            logger.error(f"Failed to generate compliance report: {e}")
            raise

    async def get_audit_trail(
        self,
        resource_type: str | None = None,
        resource_id: str | None = None,
        user_id: str | None = None,
        action: GovernanceAction | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        limit: int = 1000,
    ) -> list[AuditTrailEntry]:
        """Retrieve audit trail entries with filtering.

        Args:
            resource_type: Filter by resource type
            resource_id: Filter by resource ID
            user_id: Filter by user ID
            action: Filter by action type
            start_date: Filter by start date
            end_date: Filter by end date
            limit: Maximum number of entries

        Returns:
            Filtered audit trail entries
        """
        try:
            filtered_entries = self.audit_trail.copy()

            # Apply filters
            if resource_type:
                filtered_entries = [
                    e for e in filtered_entries if e.resource_type == resource_type
                ]

            if resource_id:
                filtered_entries = [
                    e for e in filtered_entries if e.resource_id == resource_id
                ]

            if user_id:
                filtered_entries = [e for e in filtered_entries if e.user_id == user_id]

            if action:
                filtered_entries = [e for e in filtered_entries if e.action == action]

            if start_date:
                filtered_entries = [
                    e for e in filtered_entries if e.timestamp >= start_date
                ]

            if end_date:
                filtered_entries = [
                    e for e in filtered_entries if e.timestamp <= end_date
                ]

            # Sort by timestamp (newest first) and limit
            filtered_entries.sort(key=lambda x: x.timestamp, reverse=True)

            return filtered_entries[:limit]

        except Exception as e:
            logger.error(f"Failed to retrieve audit trail: {e}")
            raise

    async def get_data_lineage(
        self,
        resource_id: str,
        direction: str = "both",  # upstream, downstream, both
    ) -> dict[str, list[DataLineageEntry]]:
        """Get data lineage for resource.

        Args:
            resource_id: Resource to trace
            direction: Direction of lineage (upstream, downstream, both)

        Returns:
            Lineage entries grouped by direction
        """
        try:
            upstream = []
            downstream = []

            if direction in ["upstream", "both"]:
                upstream = [
                    entry
                    for entry in self.data_lineage
                    if entry.destination_id == resource_id
                ]

            if direction in ["downstream", "both"]:
                downstream = [
                    entry
                    for entry in self.data_lineage
                    if entry.source_id == resource_id
                ]

            return {"upstream": upstream, "downstream": downstream}

        except Exception as e:
            logger.error(f"Failed to get data lineage: {e}")
            raise

    # Private helper methods

    async def _assess_action_risk(
        self, action: GovernanceAction, resource_type: str, user_role: str
    ) -> RiskLevel:
        """Assess risk level of action."""
        # High-risk actions
        if action in [
            GovernanceAction.DELETE,
            GovernanceAction.DEPLOY,
            GovernanceAction.EXPORT,
        ]:
            return RiskLevel.HIGH

        # Medium-risk actions
        if action in [GovernanceAction.UPDATE, GovernanceAction.CONFIGURE]:
            return RiskLevel.MEDIUM

        # Low-risk actions
        return RiskLevel.LOW

    async def _check_policy_compliance(self, context: dict[str, Any]) -> list[str]:
        """Check compliance against active policies."""
        violations = []

        for policy in self.policies.values():
            if policy.is_active():
                is_compliant, policy_violations = policy.evaluate_compliance(context)
                if not is_compliant:
                    violations.extend(policy_violations)

        return violations

    async def _process_real_time_alerts(self, entry: AuditTrailEntry) -> None:
        """Process real-time compliance alerts."""
        if entry.policy_violations:
            logger.warning(
                f"Policy violations detected in entry {entry.entry_id}: {entry.policy_violations}"
            )

        if entry.risk_level == RiskLevel.HIGH:
            logger.warning(
                f"High-risk action detected: {entry.action.value} on {entry.resource_type}"
            )

    async def _generate_compliance_recommendations(
        self, audit_entries: list[AuditTrailEntry]
    ) -> list[str]:
        """Generate compliance recommendations based on audit analysis."""
        recommendations = []

        # Analyze patterns
        violation_count = len([e for e in audit_entries if e.policy_violations])
        high_risk_count = len(
            [e for e in audit_entries if e.risk_level == RiskLevel.HIGH]
        )

        if violation_count > len(audit_entries) * 0.1:  # More than 10% violations
            recommendations.append(
                "Consider reviewing and updating governance policies to reduce violation rate"
            )

        if high_risk_count > 0:
            recommendations.append(
                "Implement additional approval workflows for high-risk operations"
            )

        # Check for missing policies
        resource_types = {e.resource_type for e in audit_entries}
        for resource_type in resource_types:
            applicable_policies = [
                p
                for p in self.policies.values()
                if resource_type in p.scope.get("resource_types", [resource_type])
            ]
            if not applicable_policies:
                recommendations.append(
                    f"Consider creating governance policies for {resource_type} resources"
                )

        return recommendations

    async def _persist_audit_entry(self, entry: AuditTrailEntry) -> None:
        """Persist audit entry to storage."""
        try:
            with open(self.audit_trail_file, "a") as f:
                f.write(json.dumps(entry.to_dict()) + "\n")
        except Exception as e:
            logger.error(f"Failed to persist audit entry: {e}")

    async def _persist_policies(self) -> None:
        """Persist policies to storage."""
        try:
            policies_data = {
                policy_id: policy.to_dict()
                for policy_id, policy in self.policies.items()
            }
            with open(self.policies_file, "w") as f:
                json.dump(policies_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to persist policies: {e}")

    async def _persist_lineage_entry(self, entry: DataLineageEntry) -> None:
        """Persist lineage entry to storage."""
        try:
            with open(self.lineage_file, "a") as f:
                f.write(json.dumps(entry.to_dict()) + "\n")
        except Exception as e:
            logger.error(f"Failed to persist lineage entry: {e}")

    def _load_policies(self) -> None:
        """Load policies from storage."""
        try:
            if self.policies_file.exists():
                with open(self.policies_file) as f:
                    policies_data = json.load(f)

                for policy_id, policy_dict in policies_data.items():
                    # Reconstruct policy object
                    policy = GovernancePolicy(
                        policy_id=policy_dict["policy_id"],
                        name=policy_dict["name"],
                        description=policy_dict["description"],
                        policy_type=PolicyType(policy_dict["policy_type"]),
                        status=PolicyStatus(policy_dict["status"]),
                        version=policy_dict["version"],
                        created_by=policy_dict["created_by"],
                        created_at=datetime.fromisoformat(policy_dict["created_at"]),
                        updated_at=datetime.fromisoformat(policy_dict["updated_at"]),
                        effective_date=datetime.fromisoformat(
                            policy_dict["effective_date"]
                        ),
                        enforcement_level=policy_dict["enforcement_level"],
                        scope=policy_dict["scope"],
                        rules=policy_dict["rules"],
                        compliance_frameworks=policy_dict["compliance_frameworks"],
                    )

                    if policy_dict.get("expiration_date"):
                        policy.expiration_date = datetime.fromisoformat(
                            policy_dict["expiration_date"]
                        )

                    self.policies[policy_id] = policy

                logger.info(f"Loaded {len(self.policies)} governance policies")

        except Exception as e:
            logger.error(f"Failed to load policies: {e}")

    def _load_audit_trail(self) -> None:
        """Load audit trail from storage."""
        try:
            if self.audit_trail_file.exists():
                with open(self.audit_trail_file) as f:
                    for line in f:
                        if line.strip():
                            entry_dict = json.loads(line)

                            # Reconstruct audit entry object
                            entry = AuditTrailEntry(
                                entry_id=entry_dict["entry_id"],
                                timestamp=datetime.fromisoformat(
                                    entry_dict["timestamp"]
                                ),
                                action=GovernanceAction(entry_dict["action"]),
                                resource_type=entry_dict["resource_type"],
                                resource_id=entry_dict["resource_id"],
                                resource_name=entry_dict["resource_name"],
                                user_id=entry_dict["user_id"],
                                user_name=entry_dict["user_name"],
                                user_role=entry_dict["user_role"],
                                description=entry_dict["description"],
                                before_state=entry_dict.get("before_state"),
                                after_state=entry_dict.get("after_state"),
                                metadata=entry_dict.get("metadata", {}),
                                risk_level=RiskLevel(entry_dict["risk_level"]),
                                policy_violations=entry_dict.get(
                                    "policy_violations", []
                                ),
                            )

                            self.audit_trail.append(entry)

                logger.info(f"Loaded {len(self.audit_trail)} audit trail entries")

        except Exception as e:
            logger.error(f"Failed to load audit trail: {e}")

    def _load_data_lineage(self) -> None:
        """Load data lineage from storage."""
        try:
            if self.lineage_file.exists():
                with open(self.lineage_file) as f:
                    for line in f:
                        if line.strip():
                            lineage_dict = json.loads(line)

                            # Reconstruct lineage entry object
                            entry = DataLineageEntry(
                                lineage_id=lineage_dict["lineage_id"],
                                timestamp=datetime.fromisoformat(
                                    lineage_dict["timestamp"]
                                ),
                                source_type=lineage_dict["source_type"],
                                source_id=lineage_dict["source_id"],
                                source_name=lineage_dict["source_name"],
                                operation=lineage_dict["operation"],
                                destination_type=lineage_dict["destination_type"],
                                destination_id=lineage_dict["destination_id"],
                                destination_name=lineage_dict["destination_name"],
                                transformation_details=lineage_dict.get(
                                    "transformation_details", {}
                                ),
                                quality_metrics=lineage_dict.get("quality_metrics", {}),
                                metadata=lineage_dict.get("metadata", {}),
                            )

                            self.data_lineage.append(entry)

                logger.info(f"Loaded {len(self.data_lineage)} data lineage entries")

        except Exception as e:
            logger.error(f"Failed to load data lineage: {e}")


@dataclass
class ChangeRequest:
    """Change request for governance approval."""

    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    description: str = ""
    requester_id: str = ""
    status: ApprovalStatus = ApprovalStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    approved_at: datetime | None = None


@dataclass
class ComplianceMetric:
    """Compliance metric tracking."""

    metric_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metric_name: str = ""
    target_value: float = 0.0
    current_value: float = 0.0
    compliance_percentage: float = 0.0
    last_updated: datetime = field(default_factory=datetime.utcnow)


@dataclass
class GovernancePolicy:
    """Governance policy definition."""

    policy_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    policy_type: PolicyType = PolicyType.DATA_ACCESS
    status: PolicyStatus = PolicyStatus.DRAFT
    rules: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
