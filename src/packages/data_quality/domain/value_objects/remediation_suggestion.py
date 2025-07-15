"""Remediation suggestion value object for quality issue resolution."""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional
from uuid import UUID, uuid4


class RemediationAction(str, Enum):
    """Remediation action type enumeration."""
    DATA_CLEANSING = "data_cleansing"
    RULE_MODIFICATION = "rule_modification"
    PROCESS_IMPROVEMENT = "process_improvement"
    SYSTEM_UPGRADE = "system_upgrade"
    MANUAL_CORRECTION = "manual_correction"
    AUTOMATED_FIX = "automated_fix"
    POLICY_CHANGE = "policy_change"
    TRAINING = "training"


class EffortLevel(str, Enum):
    """Effort level enumeration."""
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTENSIVE = "extensive"


class Priority(str, Enum):
    """Priority level enumeration."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"
    CRITICAL = "critical"


@dataclass(frozen=True)
class EffortEstimate:
    """Effort estimation for remediation."""
    effort_level: EffortLevel
    estimated_hours: Optional[int] = None
    estimated_cost: Optional[float] = None
    resource_requirements: List[str] = None
    
    def __post_init__(self):
        """Validate effort estimate."""
        if self.estimated_hours is not None and self.estimated_hours < 0:
            raise ValueError("Estimated hours cannot be negative")
        if self.estimated_cost is not None and self.estimated_cost < 0:
            raise ValueError("Estimated cost cannot be negative")
        if self.resource_requirements is None:
            object.__setattr__(self, 'resource_requirements', [])


@dataclass(frozen=True)
class SuggestionId:
    """Remediation suggestion identifier."""
    value: UUID = None
    
    def __post_init__(self):
        if self.value is None:
            object.__setattr__(self, 'value', uuid4())


@dataclass(frozen=True)
class IssueId:
    """Quality issue identifier."""
    value: UUID = None
    
    def __post_init__(self):
        if self.value is None:
            object.__setattr__(self, 'value', uuid4())


@dataclass(frozen=True)
class RemediationSuggestion:
    """Automated remediation suggestion for quality issues."""
    
    suggestion_id: SuggestionId
    issue_id: IssueId
    action_type: RemediationAction
    description: str
    implementation_steps: List[str]
    effort_estimate: EffortEstimate
    success_probability: float
    side_effects: List[str]
    priority: Priority
    
    def __post_init__(self):
        """Validate remediation suggestion."""
        if not self.description.strip():
            raise ValueError("Description cannot be empty")
        
        if not self.implementation_steps:
            raise ValueError("At least one implementation step must be provided")
        
        if not 0.0 <= self.success_probability <= 1.0:
            raise ValueError("Success probability must be between 0.0 and 1.0")
        
        if self.side_effects is None:
            object.__setattr__(self, 'side_effects', [])
    
    def get_implementation_complexity(self) -> str:
        """Assess implementation complexity."""
        step_count = len(self.implementation_steps)
        effort_level = self.effort_estimate.effort_level
        
        if effort_level == EffortLevel.EXTENSIVE or step_count > 10:
            return "very_high"
        elif effort_level == EffortLevel.HIGH or step_count > 7:
            return "high"
        elif effort_level == EffortLevel.MEDIUM or step_count > 4:
            return "medium"
        elif effort_level == EffortLevel.LOW or step_count > 2:
            return "low"
        else:
            return "very_low"
    
    def is_automated_fix(self) -> bool:
        """Check if this is an automated fix."""
        return self.action_type == RemediationAction.AUTOMATED_FIX
    
    def is_high_risk(self) -> bool:
        """Check if remediation has high risk (many side effects or low success probability)."""
        return len(self.side_effects) > 3 or self.success_probability < 0.7
    
    def get_risk_level(self) -> str:
        """Calculate overall risk level."""
        risk_factors = 0
        
        # Side effects factor
        if len(self.side_effects) > 5:
            risk_factors += 3
        elif len(self.side_effects) > 2:
            risk_factors += 2
        elif len(self.side_effects) > 0:
            risk_factors += 1
        
        # Success probability factor
        if self.success_probability < 0.5:
            risk_factors += 3
        elif self.success_probability < 0.7:
            risk_factors += 2
        elif self.success_probability < 0.9:
            risk_factors += 1
        
        # Effort factor
        effort_risks = {
            EffortLevel.EXTENSIVE: 3,
            EffortLevel.HIGH: 2,
            EffortLevel.MEDIUM: 1,
            EffortLevel.LOW: 0,
            EffortLevel.MINIMAL: 0
        }
        risk_factors += effort_risks[self.effort_estimate.effort_level]
        
        if risk_factors >= 7:
            return "very_high"
        elif risk_factors >= 5:
            return "high"
        elif risk_factors >= 3:
            return "medium"
        elif risk_factors >= 1:
            return "low"
        else:
            return "very_low"
    
    def should_require_approval(self) -> bool:
        """Check if remediation should require approval."""
        high_risk_conditions = [
            self.get_risk_level() in ["high", "very_high"],
            self.effort_estimate.effort_level == EffortLevel.EXTENSIVE,
            self.priority == Priority.CRITICAL,
            len(self.side_effects) > 3
        ]
        return any(high_risk_conditions)
    
    def get_expected_outcome(self) -> str:
        """Generate expected outcome description."""
        if self.success_probability >= 0.9:
            confidence = "highly likely"
        elif self.success_probability >= 0.7:
            confidence = "likely"
        elif self.success_probability >= 0.5:
            confidence = "moderately likely"
        else:
            confidence = "uncertain"
        
        return f"Resolution is {confidence} to succeed with {self.action_type.value} approach"
    
    def estimate_total_cost(self) -> Optional[float]:
        """Calculate total estimated cost including effort."""
        if self.effort_estimate.estimated_cost is not None:
            return self.effort_estimate.estimated_cost
        
        # Rough cost estimation based on hours
        if self.effort_estimate.estimated_hours is not None:
            # Assume $100/hour average rate
            return self.effort_estimate.estimated_hours * 100.0
        
        return None
    
    def get_approval_justification(self) -> str:
        """Generate justification for approval request."""
        return (
            f"Remediation requires approval due to: "
            f"Risk level: {self.get_risk_level()}, "
            f"Effort: {self.effort_estimate.effort_level.value}, "
            f"Priority: {self.priority.value}, "
            f"Side effects: {len(self.side_effects)} identified"
        )