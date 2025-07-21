"""Governance policy value objects for data quality governance framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from uuid import UUID, uuid4


class PolicyType(str, Enum):
    """Types of governance policies."""
    DATA_QUALITY = "data_quality"
    DATA_PRIVACY = "data_privacy"
    DATA_RETENTION = "data_retention"
    ACCESS_CONTROL = "access_control"
    COMPLIANCE = "compliance"
    SECURITY = "security"
    OPERATIONAL = "operational"
    BUSINESS_RULE = "business_rule"


class PolicyStatus(str, Enum):
    """Policy lifecycle status."""
    DRAFT = "draft"
    UNDER_REVIEW = "under_review"
    APPROVED = "approved"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


class PolicyScope(str, Enum):
    """Scope of policy application."""
    GLOBAL = "global"
    ORGANIZATION = "organization"
    BUSINESS_UNIT = "business_unit"
    DEPARTMENT = "department"
    PROJECT = "project"
    DATASET = "dataset"
    COLUMN = "column"


class PolicyEnforcementLevel(str, Enum):
    """Level of policy enforcement."""
    MANDATORY = "mandatory"
    RECOMMENDED = "recommended"
    ADVISORY = "advisory"
    INFORMATIONAL = "informational"


class ComplianceFramework(str, Enum):
    """Supported compliance frameworks."""
    GDPR = "gdpr"
    HIPAA = "hipaa"
    SOX = "sox"
    CCPA = "ccpa"
    PCI_DSS = "pci_dss"
    FDA_21_CFR_PART_11 = "fda_21_cfr_part_11"
    ISO_27001 = "iso_27001"
    NIST = "nist"
    CUSTOM = "custom"


@dataclass(frozen=True)
class PolicyIdentifier:
    """Unique identifier for a governance policy."""
    policy_id: str
    version: str = "1.0.0"
    organization_id: Optional[str] = None
    
    def __post_init__(self):
        if not self.policy_id:
            raise ValueError("Policy ID cannot be empty")
        if not self.version:
            raise ValueError("Policy version cannot be empty")
    
    @property
    def full_id(self) -> str:
        """Get full policy identifier."""
        if self.organization_id:
            return f"{self.organization_id}:{self.policy_id}:{self.version}"
        return f"{self.policy_id}:{self.version}"
    
    @classmethod
    def create_new(cls, policy_name: str, organization_id: Optional[str] = None) -> PolicyIdentifier:
        """Create new policy identifier."""
        policy_id = f"policy_{policy_name.lower().replace(' ', '_')}_{uuid4().hex[:8]}"
        return cls(policy_id=policy_id, organization_id=organization_id)


@dataclass(frozen=True)
class PolicyRule:
    """Individual rule within a governance policy."""
    rule_id: str
    name: str
    description: str
    condition: str  # Rule condition expression
    action: str  # Action to take when rule is triggered
    severity: str = "medium"  # low, medium, high, critical
    is_active: bool = True
    parameters: Dict[str, Any] = field(default_factory=dict)
    exemptions: List[str] = field(default_factory=list)
    
    @property
    def is_blocking(self) -> bool:
        """Check if rule is blocking (high/critical severity)."""
        return self.severity in ["high", "critical"]
    
    @classmethod
    def create_quality_rule(
        cls,
        rule_name: str,
        column_name: str,
        threshold: float,
        operator: str = ">=",
        **kwargs
    ) -> PolicyRule:
        """Create a data quality rule."""
        rule_id = f"quality_{rule_name.lower().replace(' ', '_')}_{uuid4().hex[:8]}"
        condition = f"quality_score({column_name}) {operator} {threshold}"
        action = f"alert_on_quality_breach"
        
        return cls(
            rule_id=rule_id,
            name=rule_name,
            description=f"Quality rule for {column_name} with threshold {threshold}",
            condition=condition,
            action=action,
            parameters={"column": column_name, "threshold": threshold, "operator": operator},
            **kwargs
        )
    
    @classmethod
    def create_privacy_rule(
        cls,
        rule_name: str,
        data_classification: str,
        access_restriction: str,
        **kwargs
    ) -> PolicyRule:
        """Create a data privacy rule."""
        rule_id = f"privacy_{rule_name.lower().replace(' ', '_')}_{uuid4().hex[:8]}"
        condition = f"data_classification == '{data_classification}'"
        action = f"apply_access_restriction('{access_restriction}')"
        
        return cls(
            rule_id=rule_id,
            name=rule_name,
            description=f"Privacy rule for {data_classification} data",
            condition=condition,
            action=action,
            severity="high",
            parameters={"classification": data_classification, "restriction": access_restriction},
            **kwargs
        )


@dataclass(frozen=True)
class PolicyApproval:
    """Policy approval record."""
    approver_id: str
    approver_role: str
    approval_date: datetime
    approval_status: str  # approved, rejected, conditional
    comments: str = ""
    conditions: List[str] = field(default_factory=list)
    approval_level: str = "standard"  # standard, executive, board
    
    @property
    def is_approved(self) -> bool:
        """Check if approval is positive."""
        return self.approval_status == "approved"
    
    @property
    def has_conditions(self) -> bool:
        """Check if approval has conditions."""
        return len(self.conditions) > 0


@dataclass(frozen=True)
class PolicyException:
    """Exception to a governance policy."""
    exception_id: str
    policy_id: str
    rule_id: Optional[str]
    reason: str
    justification: str
    requested_by: str
    approved_by: Optional[str] = None
    start_date: datetime = field(default_factory=datetime.now)
    end_date: Optional[datetime] = None
    is_permanent: bool = False
    approval_required: bool = True
    status: str = "pending"  # pending, approved, rejected, expired
    conditions: List[str] = field(default_factory=list)
    
    @property
    def is_active(self) -> bool:
        """Check if exception is currently active."""
        if self.status != "approved":
            return False
        
        now = datetime.now()
        if self.start_date > now:
            return False
        
        if not self.is_permanent and self.end_date and self.end_date < now:
            return False
        
        return True
    
    @property
    def days_remaining(self) -> Optional[int]:
        """Get days remaining for temporary exception."""
        if self.is_permanent or not self.end_date:
            return None
        
        remaining = (self.end_date - datetime.now()).days
        return max(0, remaining)
    
    @classmethod
    def create_temporary(
        cls,
        policy_id: str,
        reason: str,
        requested_by: str,
        duration_days: int,
        **kwargs
    ) -> PolicyException:
        """Create temporary policy exception."""
        exception_id = f"exception_{uuid4().hex[:8]}"
        end_date = datetime.now() + timedelta(days=duration_days)
        
        return cls(
            exception_id=exception_id,
            policy_id=policy_id,
            reason=reason,
            justification=f"Temporary exception for {duration_days} days",
            requested_by=requested_by,
            end_date=end_date,
            is_permanent=False,
            **kwargs
        )


@dataclass(frozen=True)
class GovernancePolicy:
    """Comprehensive governance policy definition."""
    identifier: PolicyIdentifier
    name: str
    description: str
    policy_type: PolicyType
    scope: PolicyScope
    enforcement_level: PolicyEnforcementLevel
    rules: List[PolicyRule] = field(default_factory=list)
    
    # Lifecycle management
    status: PolicyStatus = PolicyStatus.DRAFT
    created_date: datetime = field(default_factory=datetime.now)
    effective_date: Optional[datetime] = None
    expiration_date: Optional[datetime] = None
    last_modified: datetime = field(default_factory=datetime.now)
    
    # Approval and governance
    approvals: List[PolicyApproval] = field(default_factory=list)
    required_approvers: List[str] = field(default_factory=list)
    approval_workflow: Optional[str] = None
    
    # Compliance and regulatory
    compliance_frameworks: List[ComplianceFramework] = field(default_factory=list)
    regulatory_references: List[str] = field(default_factory=list)
    business_justification: str = ""
    
    # Exceptions and waivers
    exceptions: List[PolicyException] = field(default_factory=list)
    allows_exceptions: bool = True
    exception_approval_required: bool = True
    
    # Metadata and categorization
    tags: List[str] = field(default_factory=list)
    category: str = "general"
    priority: str = "medium"  # low, medium, high, critical
    owner: str = ""
    stewards: List[str] = field(default_factory=list)
    
    # Technical configuration
    configuration: Dict[str, Any] = field(default_factory=dict)
    monitoring_enabled: bool = True
    alert_settings: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_active(self) -> bool:
        """Check if policy is currently active."""
        if self.status != PolicyStatus.ACTIVE:
            return False
        
        now = datetime.now()
        
        if self.effective_date and self.effective_date > now:
            return False
        
        if self.expiration_date and self.expiration_date < now:
            return False
        
        return True
    
    @property
    def active_rules(self) -> List[PolicyRule]:
        """Get active rules in the policy."""
        return [rule for rule in self.rules if rule.is_active]
    
    @property
    def active_exceptions(self) -> List[PolicyException]:
        """Get currently active exceptions."""
        return [exception for exception in self.exceptions if exception.is_active]
    
    @property
    def is_fully_approved(self) -> bool:
        """Check if policy has all required approvals."""
        if not self.required_approvers:
            return True
        
        approved_by = {approval.approver_id for approval in self.approvals if approval.is_approved}
        required_set = set(self.required_approvers)
        
        return required_set.issubset(approved_by)
    
    @property
    def compliance_coverage(self) -> float:
        """Calculate compliance framework coverage."""
        if not self.compliance_frameworks:
            return 0.0
        
        # Simple coverage calculation based on number of frameworks
        total_frameworks = len(ComplianceFramework)
        covered_frameworks = len(self.compliance_frameworks)
        
        return min(1.0, covered_frameworks / total_frameworks)
    
    def get_applicable_rules(self, context: Dict[str, Any]) -> List[PolicyRule]:
        """Get rules applicable to given context."""
        applicable_rules = []
        
        for rule in self.active_rules:
            # Simple context matching - can be extended with more sophisticated logic
            if self._rule_applies_to_context(rule, context):
                applicable_rules.append(rule)
        
        return applicable_rules
    
    def get_active_exception_for_rule(self, rule_id: str) -> Optional[PolicyException]:
        """Get active exception for specific rule."""
        for exception in self.active_exceptions:
            if exception.rule_id == rule_id:
                return exception
        return None
    
    def add_approval(self, approval: PolicyApproval) -> GovernancePolicy:
        """Add approval to policy."""
        new_approvals = list(self.approvals) + [approval]
        return dataclass.replace(
            self,
            approvals=new_approvals,
            last_modified=datetime.now()
        )
    
    def add_exception(self, exception: PolicyException) -> GovernancePolicy:
        """Add exception to policy."""
        new_exceptions = list(self.exceptions) + [exception]
        return dataclass.replace(
            self,
            exceptions=new_exceptions,
            last_modified=datetime.now()
        )
    
    def update_status(self, new_status: PolicyStatus) -> GovernancePolicy:
        """Update policy status."""
        return dataclass.replace(
            self,
            status=new_status,
            last_modified=datetime.now()
        )
    
    def _rule_applies_to_context(self, rule: PolicyRule, context: Dict[str, Any]) -> bool:
        """Check if rule applies to given context."""
        # Simple implementation - can be extended with rule engine
        if "column_name" in rule.parameters and "column" in context:
            return rule.parameters["column_name"] == context["column"]
        
        if "data_classification" in rule.parameters and "classification" in context:
            return rule.parameters["data_classification"] == context["classification"]
        
        # Default to applicable if no specific context matching
        return True
    
    @classmethod
    def create_data_quality_policy(
        cls,
        name: str,
        description: str,
        quality_rules: List[PolicyRule],
        scope: PolicyScope = PolicyScope.ORGANIZATION,
        **kwargs
    ) -> GovernancePolicy:
        """Create data quality governance policy."""
        identifier = PolicyIdentifier.create_new(name)
        
        return cls(
            identifier=identifier,
            name=name,
            description=description,
            policy_type=PolicyType.DATA_QUALITY,
            scope=scope,
            enforcement_level=PolicyEnforcementLevel.MANDATORY,
            rules=quality_rules,
            compliance_frameworks=[ComplianceFramework.ISO_27001],
            category="data_quality",
            monitoring_enabled=True,
            **kwargs
        )
    
    @classmethod
    def create_privacy_policy(
        cls,
        name: str,
        description: str,
        privacy_rules: List[PolicyRule],
        compliance_frameworks: List[ComplianceFramework],
        **kwargs
    ) -> GovernancePolicy:
        """Create data privacy governance policy."""
        identifier = PolicyIdentifier.create_new(name)
        
        return cls(
            identifier=identifier,
            name=name,
            description=description,
            policy_type=PolicyType.DATA_PRIVACY,
            scope=PolicyScope.GLOBAL,
            enforcement_level=PolicyEnforcementLevel.MANDATORY,
            rules=privacy_rules,
            compliance_frameworks=compliance_frameworks,
            category="privacy",
            priority="high",
            allows_exceptions=False,  # Privacy rules typically don't allow exceptions
            **kwargs
        )