"""Business impact value object for quality issue assessment."""

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import List, Optional


class ImpactLevel(str, Enum):
    """Impact severity level enumeration."""
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ComplianceRisk(str, Enum):
    """Compliance risk level enumeration."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    REGULATORY_VIOLATION = "regulatory_violation"


class CustomerImpact(str, Enum):
    """Customer impact level enumeration."""
    NONE = "none"
    MINIMAL = "minimal"
    MODERATE = "moderate"
    SIGNIFICANT = "significant"
    SEVERE = "severe"


class OperationalImpact(str, Enum):
    """Operational impact level enumeration."""
    NONE = "none"
    MINOR_DELAY = "minor_delay"
    PROCESS_DISRUPTION = "process_disruption"
    SYSTEM_DOWNTIME = "system_downtime"
    BUSINESS_STOPPAGE = "business_stoppage"


@dataclass(frozen=True)
class MonetaryAmount:
    """Monetary amount value object."""
    amount: Decimal
    currency: str = "USD"
    
    def __post_init__(self):
        """Validate monetary amount."""
        if self.amount < 0:
            raise ValueError("Monetary amount cannot be negative")
        if not self.currency or len(self.currency) != 3:
            raise ValueError("Currency must be a 3-letter code")


@dataclass(frozen=True)
class BusinessImpact:
    """Business impact assessment for quality issues."""
    
    impact_level: ImpactLevel
    affected_processes: List[str]
    financial_impact: Optional[MonetaryAmount] = None
    compliance_risk: ComplianceRisk = ComplianceRisk.NONE
    customer_impact: CustomerImpact = CustomerImpact.NONE
    operational_impact: OperationalImpact = OperationalImpact.NONE
    
    def __post_init__(self):
        """Validate business impact data."""
        if not self.affected_processes:
            raise ValueError("At least one affected process must be specified")
        
        # Validate impact level consistency
        critical_indicators = [
            self.compliance_risk == ComplianceRisk.REGULATORY_VIOLATION,
            self.customer_impact == CustomerImpact.SEVERE,
            self.operational_impact == OperationalImpact.BUSINESS_STOPPAGE
        ]
        
        if any(critical_indicators) and self.impact_level != ImpactLevel.CRITICAL:
            raise ValueError(
                "Impact level should be CRITICAL when regulatory violations, "
                "severe customer impact, or business stoppage is present"
            )
    
    def get_priority_score(self) -> int:
        """Calculate priority score for impact triage."""
        base_scores = {
            ImpactLevel.MINIMAL: 1,
            ImpactLevel.LOW: 2,
            ImpactLevel.MEDIUM: 3,
            ImpactLevel.HIGH: 4,
            ImpactLevel.CRITICAL: 5
        }
        
        score = base_scores[self.impact_level]
        
        # Add compliance risk multiplier
        compliance_multipliers = {
            ComplianceRisk.NONE: 1.0,
            ComplianceRisk.LOW: 1.1,
            ComplianceRisk.MEDIUM: 1.3,
            ComplianceRisk.HIGH: 1.5,
            ComplianceRisk.REGULATORY_VIOLATION: 2.0
        }
        score *= compliance_multipliers[self.compliance_risk]
        
        # Add customer impact multiplier
        customer_multipliers = {
            CustomerImpact.NONE: 1.0,
            CustomerImpact.MINIMAL: 1.1,
            CustomerImpact.MODERATE: 1.2,
            CustomerImpact.SIGNIFICANT: 1.4,
            CustomerImpact.SEVERE: 1.8
        }
        score *= customer_multipliers[self.customer_impact]
        
        return int(score * 10)  # Scale to integer for easier ranking
    
    def requires_immediate_attention(self) -> bool:
        """Check if impact requires immediate attention."""
        immediate_conditions = [
            self.impact_level == ImpactLevel.CRITICAL,
            self.compliance_risk == ComplianceRisk.REGULATORY_VIOLATION,
            self.customer_impact == CustomerImpact.SEVERE,
            self.operational_impact == OperationalImpact.BUSINESS_STOPPAGE
        ]
        return any(immediate_conditions)
    
    def get_escalation_level(self) -> str:
        """Determine escalation level based on impact."""
        if self.requires_immediate_attention():
            return "executive"
        elif self.impact_level in [ImpactLevel.HIGH, ImpactLevel.CRITICAL]:
            return "management"
        elif self.impact_level == ImpactLevel.MEDIUM:
            return "supervisor"
        else:
            return "team"
    
    def has_financial_impact(self) -> bool:
        """Check if there is quantified financial impact."""
        return self.financial_impact is not None
    
    def get_financial_impact_amount(self) -> Optional[Decimal]:
        """Get financial impact amount if available."""
        return self.financial_impact.amount if self.financial_impact else None
    
    def get_impact_summary(self) -> dict:
        """Get summary of all impact dimensions."""
        return {
            "overall_level": self.impact_level.value,
            "compliance_risk": self.compliance_risk.value,
            "customer_impact": self.customer_impact.value,
            "operational_impact": self.operational_impact.value,
            "financial_impact": str(self.financial_impact.amount) if self.financial_impact else "unknown",
            "affected_processes": ", ".join(self.affected_processes),
            "escalation_level": self.get_escalation_level(),
            "priority_score": str(self.get_priority_score())
        }