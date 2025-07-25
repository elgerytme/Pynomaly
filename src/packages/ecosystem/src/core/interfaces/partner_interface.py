"""
Partner interface defining the contract for ecosystem partnerships.

This module provides interfaces and data structures for managing
strategic partnerships and integration relationships.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Set
from uuid import UUID, uuid4

import structlog

logger = structlog.get_logger(__name__)


class PartnerTier(Enum):
    """Partner tier levels with different capabilities and support."""
    COMMUNITY = "community"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"
    STRATEGIC = "strategic"


class PartnerCapability(Enum):
    """Capabilities that partners can provide."""
    DATA_STORAGE = "data_storage"
    DATA_PROCESSING = "data_processing"
    ML_TRAINING = "ml_training"
    ML_INFERENCE = "ml_inference"
    MONITORING = "monitoring"
    ALERTING = "alerting"
    VISUALIZATION = "visualization"
    WORKFLOW_ORCHESTRATION = "workflow_orchestration"
    MODEL_REGISTRY = "model_registry"
    FEATURE_STORE = "feature_store"
    DATA_CATALOG = "data_catalog"
    LINEAGE_TRACKING = "lineage_tracking"
    EXPERIMENT_TRACKING = "experiment_tracking"
    DEPLOYMENT = "deployment"
    SECURITY = "security"
    GOVERNANCE = "governance"


@dataclass
class PartnerMetrics:
    """Metrics for partner relationship."""
    
    # Usage metrics
    total_api_calls: int = 0
    successful_api_calls: int = 0
    failed_api_calls: int = 0
    average_response_time_ms: float = 0.0
    
    # Business metrics
    data_volume_gb: float = 0.0
    monthly_active_users: int = 0
    cost_per_month_usd: float = 0.0
    
    # Reliability metrics
    uptime_percentage: float = 100.0
    availability_percentage: float = 100.0
    error_rate_percentage: float = 0.0
    
    # Relationship metrics
    support_tickets_opened: int = 0
    support_tickets_resolved: int = 0
    satisfaction_score: float = 5.0  # Out of 5
    
    # Timestamps
    last_activity_at: Optional[datetime] = None
    metrics_updated_at: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def success_rate(self) -> float:
        """Calculate API success rate."""
        if self.total_api_calls == 0:
            return 100.0
        return (self.successful_api_calls / self.total_api_calls) * 100.0
    
    @property
    def support_resolution_rate(self) -> float:
        """Calculate support ticket resolution rate."""
        if self.support_tickets_opened == 0:
            return 100.0
        return (self.support_tickets_resolved / self.support_tickets_opened) * 100.0


@dataclass
class PartnerContract:
    """Contract terms and conditions for partner relationship."""
    
    # Contract identification
    contract_id: str
    partner_name: str
    contract_type: str = "standard"
    
    # Terms
    tier: PartnerTier = PartnerTier.PROFESSIONAL
    capabilities: Set[PartnerCapability] = field(default_factory=set)
    
    # Limits and quotas
    monthly_api_limit: Optional[int] = None
    monthly_data_limit_gb: Optional[float] = None
    concurrent_users_limit: Optional[int] = None
    
    # SLA terms
    uptime_sla_percentage: float = 99.0
    response_time_sla_ms: float = 1000.0
    support_response_time_hours: float = 24.0
    
    # Financial terms
    pricing_model: str = "usage_based"  # fixed, usage_based, tiered
    cost_per_api_call_usd: Optional[float] = None
    cost_per_gb_usd: Optional[float] = None
    monthly_base_cost_usd: float = 0.0
    
    # Contract lifecycle
    start_date: datetime = field(default_factory=datetime.utcnow)
    end_date: Optional[datetime] = None
    auto_renewal: bool = True
    renewal_period_months: int = 12
    
    # Compliance and security
    data_residency_requirements: List[str] = field(default_factory=list)
    compliance_standards: List[str] = field(default_factory=list)
    security_requirements: List[str] = field(default_factory=list)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def is_active(self) -> bool:
        """Check if contract is currently active."""
        now = datetime.utcnow()
        if now < self.start_date:
            return False
        if self.end_date and now > self.end_date:
            return False
        return True
    
    @property
    def days_until_expiry(self) -> Optional[int]:
        """Calculate days until contract expiry."""
        if not self.end_date:
            return None
        days = (self.end_date - datetime.utcnow()).days
        return max(0, days)


class PartnerInterface(ABC):
    """
    Abstract interface for managing ecosystem partnerships.
    
    This interface defines the contract for partner relationship
    management, including onboarding, monitoring, and governance.
    """
    
    def __init__(self, contract: PartnerContract):
        """Initialize partner with contract."""
        self.contract = contract
        self.id = uuid4()
        self.metrics = PartnerMetrics()
        self.logger = logger.bind(
            partner=contract.partner_name,
            tier=contract.tier.value,
            partner_id=str(self.id)
        )
        
        self.logger.info("Partner initialized", contract_id=contract.contract_id)
    
    # Partner lifecycle management
    
    @abstractmethod
    async def onboard(self) -> bool:
        """
        Onboard new partner.
        
        Returns:
            bool: True if onboarding successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def activate(self) -> bool:
        """
        Activate partner relationship.
        
        Returns:
            bool: True if activation successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def suspend(self, reason: str) -> bool:
        """
        Suspend partner relationship.
        
        Args:
            reason: Reason for suspension
            
        Returns:
            bool: True if suspension successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def terminate(self, reason: str) -> bool:
        """
        Terminate partner relationship.
        
        Args:
            reason: Reason for termination
            
        Returns:
            bool: True if termination successful, False otherwise
        """
        pass
    
    # Contract management
    
    @abstractmethod
    async def update_contract(self, contract: PartnerContract) -> bool:
        """
        Update partner contract.
        
        Args:
            contract: New contract terms
            
        Returns:
            bool: True if update successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def renew_contract(
        self,
        extension_months: int = 12,
        new_terms: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Renew partner contract.
        
        Args:
            extension_months: Contract extension period
            new_terms: Updated contract terms
            
        Returns:
            bool: True if renewal successful, False otherwise
        """
        pass
    
    # Capability management
    
    @abstractmethod
    async def get_available_capabilities(self) -> Set[PartnerCapability]:
        """
        Get capabilities available from partner.
        
        Returns:
            Set[PartnerCapability]: Available capabilities
        """
        pass
    
    @abstractmethod
    async def enable_capability(self, capability: PartnerCapability) -> bool:
        """
        Enable specific capability.
        
        Args:
            capability: Capability to enable
            
        Returns:
            bool: True if capability enabled, False otherwise
        """
        pass
    
    @abstractmethod
    async def disable_capability(self, capability: PartnerCapability) -> bool:
        """
        Disable specific capability.
        
        Args:
            capability: Capability to disable
            
        Returns:
            bool: True if capability disabled, False otherwise
        """
        pass
    
    # Monitoring and analytics
    
    @abstractmethod
    async def collect_metrics(self) -> PartnerMetrics:
        """
        Collect current partner metrics.
        
        Returns:
            PartnerMetrics: Current metrics
        """
        pass
    
    @abstractmethod
    async def check_sla_compliance(self) -> Dict[str, bool]:
        """
        Check SLA compliance status.
        
        Returns:
            Dict[str, bool]: SLA compliance status for each metric
        """
        pass
    
    @abstractmethod
    async def generate_usage_report(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """
        Generate usage report for specified period.
        
        Args:
            start_date: Report start date
            end_date: Report end date
            
        Returns:
            Dict[str, Any]: Usage report data
        """
        pass
    
    # Support and communication
    
    @abstractmethod
    async def create_support_ticket(
        self,
        title: str,
        description: str,
        priority: str = "medium"
    ) -> str:
        """
        Create support ticket with partner.
        
        Args:
            title: Ticket title
            description: Ticket description
            priority: Ticket priority (low, medium, high, critical)
            
        Returns:
            str: Ticket ID
        """
        pass
    
    @abstractmethod
    async def get_support_ticket_status(self, ticket_id: str) -> Dict[str, Any]:
        """
        Get support ticket status.
        
        Args:
            ticket_id: Ticket ID
            
        Returns:
            Dict[str, Any]: Ticket status information
        """
        pass
    
    # Compliance and governance
    
    @abstractmethod
    async def audit_compliance(self) -> Dict[str, Any]:
        """
        Perform compliance audit.
        
        Returns:
            Dict[str, Any]: Compliance audit results
        """
        pass
    
    @abstractmethod
    async def validate_security_requirements(self) -> Dict[str, bool]:
        """
        Validate security requirement compliance.
        
        Returns:
            Dict[str, bool]: Security validation results
        """
        pass
    
    @abstractmethod
    async def export_audit_trail(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict[str, Any]]:
        """
        Export audit trail for specified period.
        
        Args:
            start_date: Export start date
            end_date: Export end date
            
        Returns:
            List[Dict[str, Any]]: Audit trail records
        """
        pass
    
    # Public interface methods
    
    async def get_contract(self) -> PartnerContract:
        """Get current contract."""
        return self.contract
    
    async def get_metrics(self) -> PartnerMetrics:
        """Get current metrics."""
        return self.metrics
    
    async def is_contract_expiring(self, days_threshold: int = 30) -> bool:
        """
        Check if contract is expiring soon.
        
        Args:
            days_threshold: Days before expiry to consider as expiring
            
        Returns:
            bool: True if contract is expiring within threshold
        """
        days_until_expiry = self.contract.days_until_expiry
        if days_until_expiry is None:
            return False
        return days_until_expiry <= days_threshold
    
    async def calculate_monthly_cost(self) -> float:
        """
        Calculate estimated monthly cost based on current usage.
        
        Returns:
            float: Estimated monthly cost in USD
        """
        cost = self.contract.monthly_base_cost_usd
        
        # Add API call costs
        if self.contract.cost_per_api_call_usd:
            monthly_api_calls = self.metrics.total_api_calls * 30  # Estimate
            cost += monthly_api_calls * self.contract.cost_per_api_call_usd
        
        # Add data volume costs
        if self.contract.cost_per_gb_usd:
            cost += self.metrics.data_volume_gb * self.contract.cost_per_gb_usd
        
        return cost
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"Partner(name={self.contract.partner_name}, "
            f"tier={self.contract.tier.value}, "
            f"active={self.contract.is_active})"
        )