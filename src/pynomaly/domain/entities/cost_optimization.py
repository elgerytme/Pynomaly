"""Cost optimization entities for cloud resource management."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from uuid import UUID, uuid4


class ResourceType(Enum):
    """Types of cloud resources."""

    COMPUTE = "compute"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK = "network"
    GPU = "gpu"
    DATABASE = "database"
    CONTAINER = "container"
    FUNCTION = "function"
    CACHE = "cache"
    QUEUE = "queue"


class CloudProvider(Enum):
    """Supported cloud providers."""

    AWS = "aws"
    AZURE = "azure"
    GCP = "gcp"
    ON_PREMISE = "on_premise"
    HYBRID = "hybrid"
    MULTI_CLOUD = "multi_cloud"


class OptimizationStrategy(Enum):
    """Cost optimization strategies."""

    AGGRESSIVE = "aggressive"
    BALANCED = "balanced"
    CONSERVATIVE = "conservative"
    PERFORMANCE_FIRST = "performance_first"
    COST_FIRST = "cost_first"
    CUSTOM = "custom"


class RecommendationType(Enum):
    """Types of cost optimization recommendations."""

    RIGHTSIZING = "rightsizing"
    SCHEDULING = "scheduling"
    RESERVED_INSTANCES = "reserved_instances"
    SPOT_INSTANCES = "spot_instances"
    AUTO_SCALING = "auto_scaling"
    STORAGE_OPTIMIZATION = "storage_optimization"
    NETWORK_OPTIMIZATION = "network_optimization"
    RESOURCE_CONSOLIDATION = "resource_consolidation"
    IDLE_RESOURCE_CLEANUP = "idle_resource_cleanup"
    WORKLOAD_MIGRATION = "workload_migration"


class RecommendationPriority(Enum):
    """Priority levels for recommendations."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFORMATIONAL = "informational"


class ResourceStatus(Enum):
    """Status of cloud resources."""

    ACTIVE = "active"
    IDLE = "idle"
    UNDERUTILIZED = "underutilized"
    OVERUTILIZED = "overutilized"
    STOPPED = "stopped"
    TERMINATED = "terminated"
    SCHEDULED = "scheduled"


@dataclass
class ResourceUsageMetrics:
    """Resource usage metrics over time."""

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

    # Time window for these metrics
    measurement_start: datetime = field(default_factory=datetime.utcnow)
    measurement_end: datetime = field(default_factory=datetime.utcnow)
    data_points: int = 0

    def get_efficiency_score(self) -> float:
        """Calculate resource efficiency score (0-1)."""
        # Weighted average of utilization metrics
        cpu_weight = 0.4
        memory_weight = 0.4
        network_weight = 0.1
        gpu_weight = 0.1

        efficiency = (
            self.cpu_utilization_avg * cpu_weight
            + self.memory_utilization_avg * memory_weight
            + min(1.0, (self.network_bytes_in + self.network_bytes_out) / (1024**3))
            * network_weight
            + self.gpu_utilization_avg * gpu_weight
        )

        return min(1.0, efficiency)

    def is_underutilized(self, thresholds: dict[str, float] = None) -> bool:
        """Check if resource is underutilized."""
        if not thresholds:
            thresholds = {"cpu": 0.2, "memory": 0.3, "gpu": 0.2}

        return (
            self.cpu_utilization_avg < thresholds.get("cpu", 0.2)
            and self.memory_utilization_avg < thresholds.get("memory", 0.3)
            and self.gpu_utilization_avg < thresholds.get("gpu", 0.2)
        )

    def is_overutilized(self, thresholds: dict[str, float] = None) -> bool:
        """Check if resource is overutilized."""
        if not thresholds:
            thresholds = {"cpu": 0.8, "memory": 0.85, "gpu": 0.9}

        return (
            self.cpu_utilization_p95 > thresholds.get("cpu", 0.8)
            or self.memory_utilization_p95 > thresholds.get("memory", 0.85)
            or self.gpu_utilization_avg > thresholds.get("gpu", 0.9)
        )


@dataclass
class ResourceCost:
    """Cost information for a resource."""

    hourly_cost: float = 0.0
    daily_cost: float = 0.0
    monthly_cost: float = 0.0
    annual_cost: float = 0.0

    # Cost breakdown
    compute_cost: float = 0.0
    storage_cost: float = 0.0
    network_cost: float = 0.0
    licensing_cost: float = 0.0

    # Billing model
    billing_model: str = "on_demand"  # on_demand, reserved, spot, savings_plan
    currency: str = "USD"
    cost_center: str | None = None
    project_id: str | None = None

    # Cost trends
    cost_trend_7d: float = 0.0  # Percentage change over 7 days
    cost_trend_30d: float = 0.0  # Percentage change over 30 days

    def calculate_annual_cost(self) -> float:
        """Calculate projected annual cost."""
        if self.hourly_cost > 0:
            return self.hourly_cost * 24 * 365
        elif self.daily_cost > 0:
            return self.daily_cost * 365
        elif self.monthly_cost > 0:
            return self.monthly_cost * 12
        else:
            return self.annual_cost

    def get_cost_per_efficiency_unit(self, efficiency_score: float) -> float:
        """Calculate cost per efficiency unit."""
        if efficiency_score <= 0:
            return float("inf")
        return self.hourly_cost / efficiency_score


@dataclass
class CloudResource:
    """Represents a cloud resource for cost optimization."""

    resource_id: UUID = field(default_factory=uuid4)
    name: str = ""
    resource_type: ResourceType = ResourceType.COMPUTE
    provider: CloudProvider = CloudProvider.AWS
    region: str = ""
    availability_zone: str = ""

    # Resource specifications
    instance_type: str = ""
    cpu_cores: int = 0
    memory_gb: float = 0.0
    storage_gb: float = 0.0
    gpu_count: int = 0
    network_performance: str = ""

    # Status and lifecycle
    status: ResourceStatus = ResourceStatus.ACTIVE
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_accessed: datetime | None = None
    scheduled_termination: datetime | None = None

    # Usage and cost data
    usage_metrics: ResourceUsageMetrics = field(default_factory=ResourceUsageMetrics)
    cost_info: ResourceCost = field(default_factory=ResourceCost)

    # Ownership and tagging
    owner: str | None = None
    team: str | None = None
    environment: str = "production"  # production, staging, development, test
    tenant_id: UUID | None = None
    tags: dict[str, str] = field(default_factory=dict)

    # Optimization metadata
    last_optimized: datetime | None = None
    optimization_locked: bool = False
    optimization_lock_reason: str = ""

    def calculate_idle_time(self) -> timedelta:
        """Calculate how long the resource has been idle."""
        if not self.last_accessed:
            return timedelta()

        return datetime.utcnow() - self.last_accessed

    def is_idle(self, idle_threshold_hours: int = 24) -> bool:
        """Check if resource is considered idle."""
        if self.status != ResourceStatus.ACTIVE:
            return True

        idle_time = self.calculate_idle_time()
        return idle_time.total_seconds() / 3600 >= idle_threshold_hours

    def get_optimization_potential(self) -> float:
        """Calculate optimization potential score (0-1)."""
        score = 0.0

        # Underutilization penalty
        if self.usage_metrics.is_underutilized():
            score += 0.4

        # Idle time penalty
        if self.is_idle():
            score += 0.3

        # Cost trend penalty
        if self.cost_info.cost_trend_7d > 0.1:  # 10% increase
            score += 0.2

        # Inefficiency penalty
        efficiency = self.usage_metrics.get_efficiency_score()
        if efficiency < 0.5:
            score += 0.1

        return min(1.0, score)

    def can_be_optimized(self) -> bool:
        """Check if resource can be optimized."""
        if self.optimization_locked:
            return False

        if self.status in [ResourceStatus.TERMINATED, ResourceStatus.STOPPED]:
            return False

        return self.get_optimization_potential() > 0.1


@dataclass
class OptimizationRecommendation:
    """Cost optimization recommendation."""

    recommendation_id: UUID = field(default_factory=uuid4)
    resource_id: UUID = field(default_factory=uuid4)
    recommendation_type: RecommendationType = RecommendationType.RIGHTSIZING
    priority: RecommendationPriority = RecommendationPriority.MEDIUM

    # Recommendation details
    title: str = ""
    description: str = ""
    rationale: str = ""

    # Financial impact
    current_monthly_cost: float = 0.0
    projected_monthly_cost: float = 0.0
    monthly_savings: float = 0.0
    annual_savings: float = 0.0
    implementation_cost: float = 0.0
    payback_period_days: int = 0

    # Performance impact
    performance_impact: str = "none"  # none, minimal, moderate, significant
    risk_level: str = "low"  # low, medium, high
    confidence_score: float = 0.8  # 0-1

    # Implementation details
    action_required: str = ""
    automation_possible: bool = False
    estimated_implementation_time: str = ""
    prerequisites: list[str] = field(default_factory=list)

    # Tracking
    created_at: datetime = field(default_factory=datetime.utcnow)
    created_by: str = "cost_optimizer"
    status: str = "pending"  # pending, approved, rejected, implemented, expired
    expires_at: datetime | None = None

    # Related resources
    affected_resources: set[UUID] = field(default_factory=set)
    dependencies: set[UUID] = field(default_factory=set)

    def calculate_roi(self) -> float:
        """Calculate return on investment percentage."""
        if self.implementation_cost <= 0:
            return float("inf") if self.annual_savings > 0 else 0.0

        return (self.annual_savings / self.implementation_cost) * 100

    def get_priority_score(self) -> float:
        """Calculate priority score for ranking recommendations."""
        # Base score from priority level
        priority_scores = {
            RecommendationPriority.CRITICAL: 1.0,
            RecommendationPriority.HIGH: 0.8,
            RecommendationPriority.MEDIUM: 0.6,
            RecommendationPriority.LOW: 0.4,
            RecommendationPriority.INFORMATIONAL: 0.2,
        }

        base_score = priority_scores.get(self.priority, 0.6)

        # Adjust for savings amount
        savings_multiplier = min(2.0, self.annual_savings / 10000)  # Normalize to $10k

        # Adjust for confidence
        confidence_multiplier = self.confidence_score

        # Adjust for risk (inverse)
        risk_multipliers = {"low": 1.0, "medium": 0.8, "high": 0.6}
        risk_multiplier = risk_multipliers.get(self.risk_level, 0.8)

        return base_score * savings_multiplier * confidence_multiplier * risk_multiplier

    def is_expired(self) -> bool:
        """Check if recommendation has expired."""
        if not self.expires_at:
            return False
        return datetime.utcnow() > self.expires_at

    def can_be_automated(self) -> bool:
        """Check if recommendation can be automatically implemented."""
        return (
            self.automation_possible
            and self.risk_level == "low"
            and self.confidence_score >= 0.8
            and not self.prerequisites
        )


@dataclass
class CostBudget:
    """Cost budget definition and tracking."""

    budget_id: UUID = field(default_factory=uuid4)
    name: str = ""
    description: str = ""

    # Budget constraints
    monthly_limit: float = 0.0
    annual_limit: float = 0.0
    currency: str = "USD"

    # Scope
    tenant_id: UUID | None = None
    resource_types: set[ResourceType] = field(default_factory=set)
    environments: set[str] = field(default_factory=set)
    cost_centers: set[str] = field(default_factory=set)

    # Current usage
    current_monthly_spend: float = 0.0
    current_annual_spend: float = 0.0
    projected_monthly_spend: float = 0.0
    projected_annual_spend: float = 0.0

    # Alerts and thresholds
    alert_thresholds: list[float] = field(default_factory=lambda: [0.5, 0.8, 0.9, 1.0])
    alert_contacts: list[str] = field(default_factory=list)
    auto_actions_enabled: bool = False

    # Tracking
    created_at: datetime = field(default_factory=datetime.utcnow)
    created_by: str = ""
    last_updated: datetime = field(default_factory=datetime.utcnow)

    def get_monthly_utilization(self) -> float:
        """Get current monthly budget utilization (0-1)."""
        if self.monthly_limit <= 0:
            return 0.0
        return self.current_monthly_spend / self.monthly_limit

    def get_annual_utilization(self) -> float:
        """Get current annual budget utilization (0-1)."""
        if self.annual_limit <= 0:
            return 0.0
        return self.current_annual_spend / self.annual_limit

    def is_over_budget(self) -> bool:
        """Check if budget is exceeded."""
        return (
            self.get_monthly_utilization() > 1.0 or self.get_annual_utilization() > 1.0
        )

    def get_triggered_alerts(self) -> list[float]:
        """Get list of alert thresholds that have been triggered."""
        monthly_util = self.get_monthly_utilization()
        return [
            threshold
            for threshold in self.alert_thresholds
            if monthly_util >= threshold
        ]

    def days_until_budget_exhausted(self) -> int | None:
        """Calculate days until budget is exhausted at current rate."""
        if self.projected_monthly_spend <= 0:
            return None

        remaining_budget = self.monthly_limit - self.current_monthly_spend
        if remaining_budget <= 0:
            return 0

        days_left_in_month = 30 - datetime.utcnow().day
        daily_burn_rate = self.projected_monthly_spend / 30

        days_until_exhausted = remaining_budget / daily_burn_rate
        return min(int(days_until_exhausted), days_left_in_month)


@dataclass
class CostOptimizationPlan:
    """Comprehensive cost optimization plan."""

    plan_id: UUID = field(default_factory=uuid4)
    name: str = ""
    description: str = ""
    strategy: OptimizationStrategy = OptimizationStrategy.BALANCED

    # Scope and constraints
    tenant_id: UUID | None = None
    resource_scope: set[UUID] = field(default_factory=set)
    environments: set[str] = field(default_factory=set)

    # Optimization targets
    target_cost_reduction_percent: float = 0.0
    target_monthly_savings: float = 0.0
    max_performance_impact: str = "minimal"
    max_risk_level: str = "medium"

    # Recommendations
    recommendations: list[OptimizationRecommendation] = field(default_factory=list)
    total_potential_savings: float = 0.0
    total_implementation_cost: float = 0.0
    estimated_implementation_days: int = 0

    # Progress tracking
    approved_recommendations: int = 0
    implemented_recommendations: int = 0
    actual_savings_to_date: float = 0.0

    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    created_by: str = ""
    status: str = "draft"  # draft, active, completed, cancelled

    def add_recommendation(self, recommendation: OptimizationRecommendation):
        """Add a recommendation to the plan."""
        self.recommendations.append(recommendation)
        self.total_potential_savings += recommendation.annual_savings
        self.total_implementation_cost += recommendation.implementation_cost

    def get_recommendations_by_priority(self) -> list[OptimizationRecommendation]:
        """Get recommendations sorted by priority score."""
        return sorted(
            self.recommendations, key=lambda r: r.get_priority_score(), reverse=True
        )

    def get_quick_wins(self) -> list[OptimizationRecommendation]:
        """Get recommendations that are quick wins (low effort, high impact)."""
        return [
            r
            for r in self.recommendations
            if (
                r.automation_possible
                and r.annual_savings > 1000
                and r.risk_level == "low"
                and r.payback_period_days <= 30
            )
        ]

    def calculate_roi(self) -> float:
        """Calculate overall plan ROI."""
        if self.total_implementation_cost <= 0:
            return float("inf") if self.total_potential_savings > 0 else 0.0

        return (self.total_potential_savings / self.total_implementation_cost) * 100

    def get_implementation_phases(self) -> list[list[OptimizationRecommendation]]:
        """Group recommendations into implementation phases."""
        recommendations = self.get_recommendations_by_priority()

        # Phase 1: Quick wins and automated recommendations
        phase1 = [r for r in recommendations if r.can_be_automated()]

        # Phase 2: Low-risk manual recommendations
        phase2 = [
            r
            for r in recommendations
            if r.risk_level == "low" and not r.can_be_automated()
        ]

        # Phase 3: Medium-risk recommendations
        phase3 = [r for r in recommendations if r.risk_level == "medium"]

        # Phase 4: High-risk recommendations
        phase4 = [r for r in recommendations if r.risk_level == "high"]

        return [phase for phase in [phase1, phase2, phase3, phase4] if phase]


@dataclass
class CostOptimization:
    """Main entity for cost optimization framework."""

    id: UUID = field(default_factory=uuid4)
    name: str = ""
    description: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    is_active: bool = True
    
    # Core optimization configuration
    strategy: OptimizationStrategy = OptimizationStrategy.BALANCED
    target_cost_reduction_percent: float = 20.0
    max_performance_impact: str = "minimal"
    max_risk_level: str = "medium"
    
    # Resources and scope
    managed_resources: list[CloudResource] = field(default_factory=list)
    optimization_plans: list[CostOptimizationPlan] = field(default_factory=list)
    active_budgets: list[CostBudget] = field(default_factory=list)
    
    # Current optimization state
    current_plan: CostOptimizationPlan | None = None
    pending_recommendations: list[OptimizationRecommendation] = field(default_factory=list)
    implemented_recommendations: list[OptimizationRecommendation] = field(default_factory=list)
    
    # Performance metrics
    total_monthly_cost: float = 0.0
    target_monthly_cost: float = 0.0
    actual_monthly_savings: float = 0.0
    projected_annual_savings: float = 0.0
    
    # Monitoring and automation
    auto_optimization_enabled: bool = False
    monitoring_enabled: bool = True
    alert_thresholds: dict[str, float] = field(default_factory=lambda: {
        "cost_increase": 0.15,  # 15% increase
        "budget_utilization": 0.9,  # 90% of budget
        "inefficiency_score": 0.7  # 70% inefficiency
    })
    
    # Configuration and metadata
    tenant_id: UUID | None = None
    environments: set[str] = field(default_factory=lambda: {"production", "staging"})
    configuration: dict[str, any] = field(default_factory=dict)
    metadata: dict[str, any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate cost optimization configuration."""
        if not self.name:
            self.name = f"CostOptimization-{self.id}"
        
        if not (0.0 <= self.target_cost_reduction_percent <= 100.0):
            raise ValueError("Target cost reduction percent must be between 0.0 and 100.0")
        
        # Calculate target monthly cost if not set
        if self.target_monthly_cost == 0.0 and self.total_monthly_cost > 0.0:
            reduction_factor = 1.0 - (self.target_cost_reduction_percent / 100.0)
            self.target_monthly_cost = self.total_monthly_cost * reduction_factor

    def add_resource(self, resource: CloudResource) -> None:
        """Add a resource to cost optimization management."""
        self.managed_resources.append(resource)
        self.total_monthly_cost += resource.cost_info.monthly_cost
        self.updated_at = datetime.utcnow()

    def remove_resource(self, resource_id: UUID) -> bool:
        """Remove a resource from management."""
        for i, resource in enumerate(self.managed_resources):
            if resource.resource_id == resource_id:
                self.total_monthly_cost -= resource.cost_info.monthly_cost
                del self.managed_resources[i]
                self.updated_at = datetime.utcnow()
                return True
        return False

    def create_optimization_plan(self, name: str = "") -> CostOptimizationPlan:
        """Create a new optimization plan."""
        plan_name = name or f"Optimization Plan {len(self.optimization_plans) + 1}"
        
        plan = CostOptimizationPlan(
            name=plan_name,
            strategy=self.strategy,
            tenant_id=self.tenant_id,
            target_cost_reduction_percent=self.target_cost_reduction_percent,
            max_performance_impact=self.max_performance_impact,
            max_risk_level=self.max_risk_level
        )
        
        self.optimization_plans.append(plan)
        self.current_plan = plan
        self.updated_at = datetime.utcnow()
        
        return plan

    def add_recommendation(self, recommendation: OptimizationRecommendation) -> None:
        """Add a new optimization recommendation."""
        self.pending_recommendations.append(recommendation)
        
        # Auto-add to current plan if it exists
        if self.current_plan:
            self.current_plan.add_recommendation(recommendation)
        
        self.updated_at = datetime.utcnow()

    def implement_recommendation(self, recommendation_id: UUID) -> bool:
        """Mark a recommendation as implemented."""
        for i, rec in enumerate(self.pending_recommendations):
            if rec.recommendation_id == recommendation_id:
                rec.status = "implemented"
                self.implemented_recommendations.append(rec)
                del self.pending_recommendations[i]
                
                # Update savings
                self.actual_monthly_savings += rec.monthly_savings
                self.projected_annual_savings += rec.annual_savings
                
                if self.current_plan:
                    self.current_plan.implemented_recommendations += 1
                    self.current_plan.actual_savings_to_date += rec.annual_savings
                
                self.updated_at = datetime.utcnow()
                return True
        return False

    def get_underutilized_resources(self) -> list[CloudResource]:
        """Get list of underutilized resources."""
        return [
            resource for resource in self.managed_resources
            if resource.usage_metrics.is_underutilized()
        ]

    def get_idle_resources(self) -> list[CloudResource]:
        """Get list of idle resources."""
        return [
            resource for resource in self.managed_resources
            if resource.is_idle()
        ]

    def get_high_cost_resources(self, threshold_percentile: float = 0.8) -> list[CloudResource]:
        """Get resources in the top cost percentile."""
        if not self.managed_resources:
            return []
        
        costs = [r.cost_info.monthly_cost for r in self.managed_resources]
        costs.sort()
        threshold_index = int(len(costs) * threshold_percentile)
        
        if threshold_index >= len(costs):
            threshold_index = len(costs) - 1
        
        threshold_cost = costs[threshold_index]
        
        return [
            resource for resource in self.managed_resources
            if resource.cost_info.monthly_cost >= threshold_cost
        ]

    def calculate_potential_savings(self) -> float:
        """Calculate total potential monthly savings."""
        return sum(rec.monthly_savings for rec in self.pending_recommendations)

    def get_cost_efficiency_score(self) -> float:
        """Calculate overall cost efficiency score (0-1)."""
        if not self.managed_resources:
            return 1.0
        
        total_efficiency = sum(
            resource.usage_metrics.get_efficiency_score() 
            for resource in self.managed_resources
        )
        
        return total_efficiency / len(self.managed_resources)

    def get_budget_utilization(self) -> float:
        """Get current budget utilization across all budgets."""
        if not self.active_budgets:
            return 0.0
        
        total_utilization = sum(
            budget.get_monthly_utilization() for budget in self.active_budgets
        )
        
        return total_utilization / len(self.active_budgets)

    def needs_attention(self) -> bool:
        """Check if cost optimization needs attention."""
        # Check budget utilization
        if self.get_budget_utilization() > self.alert_thresholds.get("budget_utilization", 0.9):
            return True
        
        # Check cost efficiency
        if self.get_cost_efficiency_score() < (1.0 - self.alert_thresholds.get("inefficiency_score", 0.7)):
            return True
        
        # Check for high number of unimplemented recommendations
        if len(self.pending_recommendations) > 10:
            return True
        
        # Check cost trend
        cost_increase = (self.total_monthly_cost - self.target_monthly_cost) / self.target_monthly_cost
        if cost_increase > self.alert_thresholds.get("cost_increase", 0.15):
            return True
        
        return False

    def get_optimization_status(self) -> dict[str, any]:
        """Get comprehensive optimization status."""
        potential_savings = self.calculate_potential_savings()
        cost_efficiency = self.get_cost_efficiency_score()
        budget_utilization = self.get_budget_utilization()
        
        return {
            "is_active": self.is_active,
            "auto_optimization_enabled": self.auto_optimization_enabled,
            "monitoring_enabled": self.monitoring_enabled,
            "total_monthly_cost": self.total_monthly_cost,
            "target_monthly_cost": self.target_monthly_cost,
            "actual_monthly_savings": self.actual_monthly_savings,
            "potential_monthly_savings": potential_savings,
            "cost_efficiency_score": cost_efficiency,
            "budget_utilization": budget_utilization,
            "managed_resources_count": len(self.managed_resources),
            "underutilized_resources_count": len(self.get_underutilized_resources()),
            "idle_resources_count": len(self.get_idle_resources()),
            "pending_recommendations_count": len(self.pending_recommendations),
            "implemented_recommendations_count": len(self.implemented_recommendations),
            "active_plans_count": len([p for p in self.optimization_plans if p.status == "active"]),
            "needs_attention": self.needs_attention(),
            "last_updated": self.updated_at.isoformat(),
        }

    def get_cost_breakdown_by_environment(self) -> dict[str, float]:
        """Get cost breakdown by environment."""
        breakdown = {}
        for resource in self.managed_resources:
            env = resource.environment
            if env not in breakdown:
                breakdown[env] = 0.0
            breakdown[env] += resource.cost_info.monthly_cost
        return breakdown

    def get_cost_breakdown_by_resource_type(self) -> dict[str, float]:
        """Get cost breakdown by resource type."""
        breakdown = {}
        for resource in self.managed_resources:
            res_type = resource.resource_type.value
            if res_type not in breakdown:
                breakdown[res_type] = 0.0
            breakdown[res_type] += resource.cost_info.monthly_cost
        return breakdown

    def generate_quick_wins_report(self) -> dict[str, any]:
        """Generate a report of quick win opportunities."""
        if not self.current_plan:
            return {"quick_wins": [], "total_potential_savings": 0.0}
        
        quick_wins = self.current_plan.get_quick_wins()
        total_savings = sum(rec.monthly_savings for rec in quick_wins)
        
        return {
            "quick_wins": [
                {
                    "recommendation_id": str(rec.recommendation_id),
                    "title": rec.title,
                    "monthly_savings": rec.monthly_savings,
                    "implementation_time": rec.estimated_implementation_time,
                    "automation_possible": rec.automation_possible,
                    "confidence_score": rec.confidence_score
                }
                for rec in quick_wins
            ],
            "total_potential_savings": total_savings,
            "implementation_count": len(quick_wins)
        }
