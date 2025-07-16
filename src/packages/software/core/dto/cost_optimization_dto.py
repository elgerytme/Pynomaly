"""Data Transfer Objects for cost optimization API."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any


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
    measurement_start: datetime | None = None
    measurement_end: datetime | None = None
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
    cost_center: str | None = None
    project_id: str | None = None
    cost_trend_7d: float = 0.0
    cost_trend_30d: float = 0.0


@dataclass
class CloudResourceDTO:
    """DTO for cloud resource."""

    resource_id: str | None = None
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
    created_at: datetime | None = None
    last_accessed: datetime | None = None
    scheduled_termination: datetime | None = None
    usage_metrics: ResourceUsageMetricsDTO | None = None
    cost_info: ResourceCostDTO | None = None
    owner: str | None = None
    team: str | None = None
    environment: str = "production"
    tenant_id: str | None = None
    tags: dict[str, str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = {}


@dataclass
class OptimizationRecommendationDTO:
    """DTO for optimization recommendation."""

    recommendation_id: str | None = None
    resource_id: str | None = None
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
    prerequisites: list[str] = None
    created_at: datetime | None = None
    created_by: str = "cost_optimizer"
    status: str = "pending"
    expires_at: datetime | None = None
    affected_resources: list[str] = None
    dependencies: list[str] = None

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

    budget_id: str | None = None
    name: str = ""
    description: str = ""
    monthly_limit: float = 0.0
    annual_limit: float = 0.0
    currency: str = "USD"
    tenant_id: str | None = None
    resource_types: list[str] = None
    environments: list[str] = None
    cost_centers: list[str] = None
    current_monthly_spend: float = 0.0
    current_annual_spend: float = 0.0
    projected_monthly_spend: float = 0.0
    projected_annual_spend: float = 0.0
    alert_thresholds: list[float] = None
    alert_contacts: list[str] = None
    auto_actions_enabled: bool = False
    created_at: datetime | None = None
    created_by: str = ""
    last_updated: datetime | None = None

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

    plan_id: str | None = None
    name: str = ""
    description: str = ""
    strategy: str = "balanced"
    tenant_id: str | None = None
    resource_scope: list[str] = None
    environments: list[str] = None
    target_cost_reduction_percent: float = 0.0
    target_monthly_savings: float = 0.0
    max_performance_impact: str = "minimal"
    max_risk_level: str = "medium"
    recommendations: list[OptimizationRecommendationDTO] = None
    total_potential_savings: float = 0.0
    total_implementation_cost: float = 0.0
    estimated_implementation_days: int = 0
    approved_recommendations: int = 0
    implemented_recommendations: int = 0
    actual_savings_to_date: float = 0.0
    created_at: datetime | None = None
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

    tenant_id: str | None = None
    days: int = 30
    include_predictions: bool = True
    include_anomalies: bool = True
    include_optimization_potential: bool = True
    resource_types: list[str] = None
    environments: list[str] = None
    providers: list[str] = None

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
    generated_at: datetime | None = None
    total_monthly_cost: float = 0.0
    projected_annual_cost: float = 0.0
    cost_by_resource_type: dict[str, float] = None
    cost_by_provider: dict[str, float] = None
    cost_by_environment: dict[str, float] = None
    top_cost_drivers: list[dict[str, Any]] = None
    cost_trends: dict[str, float] = None
    inefficiency_indicators: dict[str, Any] = None
    cost_anomalies: list[dict[str, Any]] = None
    cost_predictions: dict[str, Any] = None
    optimization_potential: dict[str, Any] = None

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
    tenant_id: str | None = None
    target_savings_percent: float = 0.2
    max_risk_level: str = "medium"
    max_performance_impact: str = "minimal"
    resource_scope: list[str] = None
    environments: list[str] = None
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
    scheduled_at: datetime | None = None
    dry_run: bool = False
    confirmation_required: bool = True
    rollback_plan: str = ""
    notification_contacts: list[str] = None

    def __post_init__(self):
        if self.notification_contacts is None:
            self.notification_contacts = []


@dataclass
class RecommendationImplementationResultDTO:
    """DTO for recommendation implementation result."""

    implementation_id: str = ""
    recommendation_id: str = ""
    status: str = "pending"  # pending, in_progress, completed, failed, rolled_back
    started_at: datetime | None = None
    completed_at: datetime | None = None
    success: bool = False
    error_message: str | None = None
    actual_savings: float = 0.0
    actual_implementation_time: str | None = None
    performance_impact_observed: str = "none"
    rollback_available: bool = False
    rollback_instructions: str = ""
    validation_results: dict[str, Any] = None

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
    triggered_at: datetime | None = None
    days_until_exhausted: int | None = None
    recommended_actions: list[str] = None

    def __post_init__(self):
        if self.recommended_actions is None:
            self.recommended_actions = []


@dataclass
class ResourceSummaryDTO:
    """DTO for resource summary."""

    total_resources: int = 0
    total_monthly_cost: float = 0.0
    resource_breakdown: dict[str, dict[str, int]] = None
    optimization_summary: dict[str, Any] = None
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
    last_updated: datetime | None = None
