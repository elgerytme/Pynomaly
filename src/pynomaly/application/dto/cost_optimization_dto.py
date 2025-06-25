"""Data Transfer Objects for cost optimization API."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Set
from uuid import UUID

from pynomaly.domain.entities.cost_optimization import (
    ResourceType, CloudProvider, OptimizationStrategy, RecommendationType,
    RecommendationPriority, ResourceStatus
)


@dataclass
class ResourceUsageMetricsDTO:
    """DTO for resource usage metrics."""
    
    cpu_utilization_avg: float = 0.0
    cpu_utilization_max: float = 0.0
    cpu_utilization_p95: float = 0.0
    memory_utilization_avg: float = 0.0
    memory_utilization_max: float = 0.0
    memory_utilization_p95: float = 0.0
    network_bytes_in: int = 0
    network_bytes_out: int = 0
    disk_read_bytes: int = 0
    disk_write_bytes: int = 0
    gpu_utilization_avg: float = 0.0
    storage_used_gb: float = 0.0
    requests_per_second: float = 0.0
    error_rate: float = 0.0
    response_time_ms: float = 0.0
    measurement_start: Optional[datetime] = None
    measurement_end: Optional[datetime] = None
    data_points: int = 0


@dataclass
class ResourceCostDTO:
    """DTO for resource cost information."""
    
    hourly_cost: float = 0.0
    daily_cost: float = 0.0
    monthly_cost: float = 0.0
    annual_cost: float = 0.0
    compute_cost: float = 0.0
    storage_cost: float = 0.0
    network_cost: float = 0.0
    licensing_cost: float = 0.0
    billing_model: str = "on_demand"
    currency: str = "USD"
    cost_center: Optional[str] = None
    project_id: Optional[str] = None
    cost_trend_7d: float = 0.0
    cost_trend_30d: float = 0.0


@dataclass
class CloudResourceDTO:
    """DTO for cloud resource."""
    
    resource_id: Optional[str] = None
    name: str = ""
    resource_type: str = "compute"
    provider: str = "aws"
    region: str = ""
    availability_zone: str = ""
    instance_type: str = ""
    cpu_cores: int = 0
    memory_gb: float = 0.0
    storage_gb: float = 0.0
    gpu_count: int = 0
    network_performance: str = ""
    status: str = "active"
    created_at: Optional[datetime] = None
    last_accessed: Optional[datetime] = None
    scheduled_termination: Optional[datetime] = None
    usage_metrics: Optional[ResourceUsageMetricsDTO] = None
    cost_info: Optional[ResourceCostDTO] = None
    owner: Optional[str] = None
    team: Optional[str] = None
    environment: str = "production"
    tenant_id: Optional[str] = None
    tags: Dict[str, str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}


@dataclass
class OptimizationRecommendationDTO:
    """DTO for optimization recommendation."""
    
    recommendation_id: Optional[str] = None
    resource_id: Optional[str] = None
    recommendation_type: str = "rightsizing"
    priority: str = "medium"
    title: str = ""
    description: str = ""
    rationale: str = ""
    current_monthly_cost: float = 0.0
    projected_monthly_cost: float = 0.0
    monthly_savings: float = 0.0
    annual_savings: float = 0.0
    implementation_cost: float = 0.0
    payback_period_days: int = 0
    performance_impact: str = "none"
    risk_level: str = "low"
    confidence_score: float = 0.8
    action_required: str = ""
    automation_possible: bool = False
    estimated_implementation_time: str = ""
    prerequisites: List[str] = None
    created_at: Optional[datetime] = None
    created_by: str = "cost_optimizer"
    status: str = "pending"
    expires_at: Optional[datetime] = None
    affected_resources: List[str] = None
    dependencies: List[str] = None
    
    def __post_init__(self):
        if self.prerequisites is None:
            self.prerequisites = []
        if self.affected_resources is None:
            self.affected_resources = []
        if self.dependencies is None:
            self.dependencies = []


@dataclass
class CostBudgetDTO:
    """DTO for cost budget."""
    
    budget_id: Optional[str] = None
    name: str = ""
    description: str = ""
    monthly_limit: float = 0.0
    annual_limit: float = 0.0
    currency: str = "USD"
    tenant_id: Optional[str] = None
    resource_types: List[str] = None
    environments: List[str] = None
    cost_centers: List[str] = None
    current_monthly_spend: float = 0.0
    current_annual_spend: float = 0.0
    projected_monthly_spend: float = 0.0
    projected_annual_spend: float = 0.0
    alert_thresholds: List[float] = None
    alert_contacts: List[str] = None
    auto_actions_enabled: bool = False
    created_at: Optional[datetime] = None
    created_by: str = ""
    last_updated: Optional[datetime] = None
    
    def __post_init__(self):
        if self.resource_types is None:
            self.resource_types = []
        if self.environments is None:
            self.environments = []
        if self.cost_centers is None:
            self.cost_centers = []
        if self.alert_thresholds is None:
            self.alert_thresholds = [0.5, 0.8, 0.9, 1.0]
        if self.alert_contacts is None:
            self.alert_contacts = []


@dataclass
class CostOptimizationPlanDTO:
    """DTO for cost optimization plan."""
    
    plan_id: Optional[str] = None
    name: str = ""
    description: str = ""
    strategy: str = "balanced"
    tenant_id: Optional[str] = None
    resource_scope: List[str] = None
    environments: List[str] = None
    target_cost_reduction_percent: float = 0.0
    target_monthly_savings: float = 0.0
    max_performance_impact: str = "minimal"
    max_risk_level: str = "medium"
    recommendations: List[OptimizationRecommendationDTO] = None
    total_potential_savings: float = 0.0
    total_implementation_cost: float = 0.0
    estimated_implementation_days: int = 0
    approved_recommendations: int = 0
    implemented_recommendations: int = 0
    actual_savings_to_date: float = 0.0
    created_at: Optional[datetime] = None
    created_by: str = ""
    status: str = "draft"
    
    def __post_init__(self):
        if self.resource_scope is None:
            self.resource_scope = []
        if self.environments is None:
            self.environments = []
        if self.recommendations is None:
            self.recommendations = []


@dataclass
class CostAnalysisRequestDTO:
    """DTO for cost analysis request."""
    
    tenant_id: Optional[str] = None
    days: int = 30
    include_predictions: bool = True
    include_anomalies: bool = True
    include_optimization_potential: bool = True
    resource_types: List[str] = None
    environments: List[str] = None
    providers: List[str] = None
    
    def __post_init__(self):
        if self.resource_types is None:
            self.resource_types = []
        if self.environments is None:
            self.environments = []
        if self.providers is None:
            self.providers = []


@dataclass
class CostAnalysisResponseDTO:
    """DTO for cost analysis response."""
    
    analysis_id: str = ""
    generated_at: Optional[datetime] = None
    total_monthly_cost: float = 0.0
    projected_annual_cost: float = 0.0
    cost_by_resource_type: Dict[str, float] = None
    cost_by_provider: Dict[str, float] = None
    cost_by_environment: Dict[str, float] = None
    top_cost_drivers: List[Dict[str, Any]] = None
    cost_trends: Dict[str, float] = None
    inefficiency_indicators: Dict[str, Any] = None
    cost_anomalies: List[Dict[str, Any]] = None
    cost_predictions: Dict[str, Any] = None
    optimization_potential: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.cost_by_resource_type is None:
            self.cost_by_resource_type = {}
        if self.cost_by_provider is None:
            self.cost_by_provider = {}
        if self.cost_by_environment is None:
            self.cost_by_environment = {}
        if self.top_cost_drivers is None:
            self.top_cost_drivers = []
        if self.cost_trends is None:
            self.cost_trends = {}
        if self.inefficiency_indicators is None:
            self.inefficiency_indicators = {}
        if self.cost_anomalies is None:
            self.cost_anomalies = []
        if self.cost_predictions is None:
            self.cost_predictions = {}
        if self.optimization_potential is None:
            self.optimization_potential = {}


@dataclass
class OptimizationPlanRequestDTO:
    """DTO for optimization plan generation request."""
    
    strategy: str = "balanced"
    tenant_id: Optional[str] = None
    target_savings_percent: float = 0.2
    max_risk_level: str = "medium"
    max_performance_impact: str = "minimal"
    resource_scope: List[str] = None
    environments: List[str] = None
    include_scheduling: bool = True
    include_rightsizing: bool = True
    include_instance_optimization: bool = True
    include_storage_optimization: bool = True
    include_idle_cleanup: bool = True
    auto_implement_safe: bool = False
    
    def __post_init__(self):
        if self.resource_scope is None:
            self.resource_scope = []
        if self.environments is None:
            self.environments = []


@dataclass
class RecommendationImplementationDTO:
    """DTO for recommendation implementation."""
    
    recommendation_id: str = ""
    implementation_method: str = "manual"  # manual, automated, scheduled
    scheduled_at: Optional[datetime] = None
    dry_run: bool = False
    confirmation_required: bool = True
    rollback_plan: str = ""
    notification_contacts: List[str] = None
    
    def __post_init__(self):
        if self.notification_contacts is None:
            self.notification_contacts = []


@dataclass
class RecommendationImplementationResultDTO:
    """DTO for recommendation implementation result."""
    
    implementation_id: str = ""
    recommendation_id: str = ""
    status: str = "pending"  # pending, in_progress, completed, failed, rolled_back
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    success: bool = False
    error_message: Optional[str] = None
    actual_savings: float = 0.0
    actual_implementation_time: Optional[str] = None
    performance_impact_observed: str = "none"
    rollback_available: bool = False
    rollback_instructions: str = ""
    validation_results: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.validation_results is None:
            self.validation_results = {}


@dataclass
class BudgetAlertDTO:
    """DTO for budget alert."""
    
    alert_id: str = ""
    budget_id: str = ""
    budget_name: str = ""
    alert_type: str = "budget_threshold"
    threshold: float = 0.0
    current_utilization: float = 0.0
    current_spend: float = 0.0
    budget_limit: float = 0.0
    severity: str = "medium"
    triggered_at: Optional[datetime] = None
    days_until_exhausted: Optional[int] = None
    recommended_actions: List[str] = None
    
    def __post_init__(self):
        if self.recommended_actions is None:
            self.recommended_actions = []


@dataclass
class ResourceSummaryDTO:
    """DTO for resource summary."""
    
    total_resources: int = 0
    total_monthly_cost: float = 0.0
    resource_breakdown: Dict[str, Dict[str, int]] = None
    optimization_summary: Dict[str, Any] = None
    cost_efficiency_score: float = 0.0
    recommendations_available: int = 0
    potential_monthly_savings: float = 0.0
    
    def __post_init__(self):
        if self.resource_breakdown is None:
            self.resource_breakdown = {}
        if self.optimization_summary is None:
            self.optimization_summary = {}


@dataclass
class ServiceMetricsDTO:
    """DTO for service metrics."""
    
    total_resources: int = 0
    total_monthly_cost: float = 0.0
    total_savings_identified: float = 0.0
    recommendations_generated: int = 0
    recommendations_implemented: int = 0
    optimization_plans: int = 0
    budgets: int = 0
    avg_cost_per_resource: float = 0.0
    savings_rate: float = 0.0
    last_updated: Optional[datetime] = None